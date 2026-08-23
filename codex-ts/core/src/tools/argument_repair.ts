/**
 * Repairing — and, failing that, DIAGNOSING — malformed tool-call arguments.
 *
 * The model sometimes emits a tool call whose `arguments` are not valid JSON. It
 * is NOT truncation: the payload arrives complete, ending in `"}}`; what is
 * broken is escaping INSIDE a string. Measured across 44 real production threads:
 * 13 of 4683 tool calls (0.28%) — but they cluster. One session lost 3m48s to SIX
 * consecutive `doc.create` attempts, each regenerating the same ~4.7KB body.
 *
 * Observed shapes, all from Claude-on-Bedrock writing long markdown into a JSON
 * string argument:
 *   1. a literal newline inside the string (all 13; the parser reports the
 *      confusing "Unterminated string");
 *   2. a literal `"` inside the string (`the "arrival" arc`);
 *   3. a `<parameter name="…">…` block spliced after the closing brace — the
 *      model's native XML tool syntax leaking into the JSON interface.
 *
 * Two responses, in this order:
 *   · Repair-then-parse. Shapes 1 and 2 are unambiguous once you track string
 *     state; replaying the repair over all 13 recorded failures recovers 10
 *     (6 of 6 in the incident above). Shape 3 is deliberately NOT repaired —
 *     guessing how a spliced parameter was meant to merge could dispatch a tool
 *     with invented arguments, which is worse than refusing.
 *   · A DIAGNOSABLE refusal when repair fails. Stock handed the model the
 *     parser's own words ("JSON Parse error: Unterminated string") with no
 *     offset, no character and no rule — which is precisely why it retried the
 *     identical shape six times. We return where it broke, what broke it, and
 *     the rule to follow.
 *
 * Strict `JSON.parse` always runs first, so a well-formed call is never touched.
 */

// =============================================================================
// Scanning
// =============================================================================

const JSON_CONTROL_ESCAPES: Record<string, string> = {
  "\b": "\\b",
  "\f": "\\f",
  "\n": "\\n",
  "\r": "\\r",
  "\t": "\\t",
};

/** Where the arguments first went wrong, and in what way. */
export interface ArgumentFault {
  index: number;
  kind: string;
}

/**
 * True when a `"` at this point can only be the string's TERMINATOR: the next
 * non-whitespace character is one that may legally follow a string value (or the
 * input ends). Anything else means the model wrote a quote it forgot to escape.
 */
function isStringTerminator(raw: string, quoteIndex: number): boolean {
  let index = quoteIndex + 1;
  while (index < raw.length && " \t\r\n".includes(raw[index] ?? "")) {
    index += 1;
  }
  if (index >= raw.length) {
    return true;
  }
  return ",}]:".includes(raw[index] ?? "");
}

/**
 * Walk the arguments once, tracking string state, and yield BOTH the repaired
 * text and the first fault found (for the diagnostic, when the repair is not
 * enough). One walk, so the two can never disagree about where the fault is.
 */
export function scanToolArguments(raw: string): {
  fault: ArgumentFault | undefined;
  repaired: string;
} {
  let repaired = "";
  let fault: ArgumentFault | undefined;
  let inString = false;
  let escaped = false;

  for (let index = 0; index < raw.length; index += 1) {
    const char = raw[index] ?? "";
    if (escaped) {
      repaired += char;
      escaped = false;
      continue;
    }
    if (inString && char === "\\") {
      repaired += char;
      escaped = true;
      continue;
    }
    if (char === '"') {
      if (!inString) {
        inString = true;
        repaired += char;
        continue;
      }
      if (isStringTerminator(raw, index)) {
        inString = false;
        repaired += char;
        continue;
      }
      fault ??= { index, kind: 'double quote (")' };
      repaired += '\\"';
      continue;
    }
    if (inString && char.charCodeAt(0) < 0x20) {
      fault ??= {
        index,
        kind:
          char === "\n"
            ? "line break"
            : `control character (U+${char
                .charCodeAt(0)
                .toString(16)
                .padStart(4, "0")})`,
      };
      repaired +=
        JSON_CONTROL_ESCAPES[char] ??
        `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`;
      continue;
    }
    repaired += char;
  }

  return { fault, repaired };
}

// =============================================================================
// Diagnostics
// =============================================================================

/**
 * The fault in the model's own words: where it is, what it is, and what to do —
 * enough to fix THIS call instead of re-authoring the whole payload blind. The
 * three branches are the three things that actually go wrong, and each asks for
 * a different correction, so none of them may be described as just "invalid
 * JSON".
 */
export function argumentFaultMessage(
  raw: string,
  fault: ArgumentFault | undefined,
  strictError: unknown,
): string {
  const spliced = raw.includes("<parameter name=")
    ? " Your arguments also contain a `<parameter name=…>` block: this tool interface is JSON, never XML — put every parameter inside the single JSON object."
    : "";

  if (raw.trim() === "") {
    return `your arguments arrived EMPTY. Re-send the SAME call with its full JSON arguments object.${spliced}`;
  }

  if (!fault) {
    // Nothing unescaped explains it → the text itself is incomplete (a brace,
    // bracket or string never closed). Telling the model to "escape better"
    // here would send it hunting for a fault that is not there.
    const reason =
      strictError instanceof Error ? strictError.message : String(strictError);
    return (
      `your arguments are not valid JSON (${reason}) and no unescaped character ` +
      `explains it — the payload looks INCOMPLETE (an unclosed brace, bracket or ` +
      `string). Re-send the SAME call with the whole arguments object.${spliced}`
    );
  }

  const context = ` Around it: …${JSON.stringify(
    raw.slice(Math.max(0, fault.index - 70), fault.index + 70),
  )}…`;
  return (
    `your arguments are not valid JSON — a literal ${fault.kind} appears inside a ` +
    `string at character ${fault.index}, and a JSON string may not contain one raw.` +
    `${context} Write a line break as \\n and a double quote as \\", then re-send ` +
    `the SAME call.${spliced}`
  );
}

// =============================================================================
// Public entry point
// =============================================================================

/**
 * Strict parse → repair → throw a diagnosable error. Never returns a value the
 * strict parser would have rejected.
 */
export function parseToolArguments(raw: string, toolName: string): unknown {
  try {
    return JSON.parse(raw);
  } catch (strictError) {
    const { fault, repaired } = scanToolArguments(raw);
    if (repaired !== raw) {
      try {
        const parsed: unknown = JSON.parse(repaired);
        // Logged so the repair rate stays measurable instead of silently
        // masking a model that is getting worse at this.
        console.warn(
          `[codex] repaired malformed tool arguments for ${toolName}: ${
            fault ? `${fault.kind} at ${fault.index}` : "unescaped characters"
          }`,
        );
        return parsed;
      } catch {
        // The repair did not make it parseable either — fall through to the
        // diagnostic, which is built from the ORIGINAL text's fault.
      }
    }
    const error = new Error(argumentFaultMessage(raw, fault, strictError));
    error.cause = strictError;
    throw error;
  }
}

/**
 * REPLAY hygiene: canonicalize a stored `function_call.arguments` string so a
 * downstream replay converter can parse it.
 *
 * Execution repairs malformed arguments before dispatch, but history keeps the
 * model's ORIGINAL text — and a converter that plain-`JSON.parse`s it (e.g.
 * @codeproxy/core's mapInputToolCall) zeroes the call to `input: {}`. The model
 * then sees its own successful call as EMPTY and "redoes" it (observed
 * 2026-08-11: a doc create → "let me redo that call properly" → a verification
 * doc get, twice in one evening).
 *
 * Returns the canonical text, or the original when it is already valid or
 * cannot be repaired (the execution path has already surfaced its diagnostic to
 * the model in that case — there is nothing better to send).
 */
export function canonicalizeStoredArguments(
  raw: string,
  toolName: string,
): string {
  if (raw.trim() === "") {
    return raw;
  }
  try {
    JSON.parse(raw);
    return raw;
  } catch {
    try {
      return JSON.stringify(parseToolArguments(raw, toolName));
    } catch {
      return raw;
    }
  }
}
