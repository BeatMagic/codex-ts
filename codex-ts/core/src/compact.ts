/**
 * mirrors codex-rs/core/src/compact.rs
 *        codex-rs/prompts/templates/compact/prompt.md
 *        codex-rs/prompts/templates/compact/summary_prefix.md
 *
 * Inline context compaction: summarise the conversation with a fresh model
 * request and replace history with the summary + recent user messages.
 */

import type { ConversationItem } from "../../thread-store/src/types.js";
import type { TurnConfig } from "./session/turn.js";
import {
  DEFAULT_STREAM_IDLE_TIMEOUT_MS,
  messageOutputText,
  parseSseStream,
} from "./session/sse.js";
import {
  classifyHttpFailure,
  classifyStreamFailure,
  computeRetryDelay,
  DEFAULT_MAX_RETRIES,
  isContextWindowExceededError,
  isRetryableError,
  ResponsesApiError,
  sleep,
} from "./session/retry.js";
import {
  normalizeHistory,
  removeCorrespondingFor,
  removeOrphanOutputs,
} from "./normalize.js";
import { acquireRequestGate } from "./session/request_gate.js";
import {
  approxTokenCount,
  estimateItemsTokenCount,
  utf8ByteLength,
} from "./state/token_state.js";

// ─── Prompt constants (verbatim from codex-rs templates) ──────────────────────

/** mirrors prompts/templates/compact/prompt.md */
export const SUMMARIZATION_PROMPT =
  "You are performing a CONTEXT CHECKPOINT COMPACTION. " +
  "Create a handoff summary for another LLM that will resume the task.\n\n" +
  "Include:\n" +
  "- Current progress and key decisions made\n" +
  "- Important context, constraints, or user preferences\n" +
  "- What remains to be done (clear next steps)\n" +
  "- Any critical data, examples, or references needed to continue\n\n" +
  "Be concise, structured, and focused on helping the next LLM seamlessly continue the work.";

/** mirrors prompts/templates/compact/summary_prefix.md */
export const SUMMARY_PREFIX =
  "Another language model started to solve this problem and produced a summary " +
  "of its thinking process. You also have access to the state of the tools that " +
  "were used by that language model. Use this to build on the work that has already " +
  "been done and avoid duplicating work. Here is the summary produced by the other " +
  "language model, use the information in this summary to assist with your own analysis:";

/** mirrors COMPACT_USER_MESSAGE_MAX_TOKENS in compact.rs */
const COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000;

// ─── Token approximation (mirrors codex-rs/utils/output-truncation) ───────────
// approxTokenCount is ~4 UTF-8 bytes per token (state/token_state.ts).

/** Keep the longest prefix of `text` within `maxTokens` × 4 UTF-8 bytes,
 *  never splitting a surrogate pair. */
function truncateToTokens(text: string, maxTokens: number): string {
  const maxBytes = maxTokens * 4;
  if (utf8ByteLength(text) <= maxBytes) return text;
  let bytes = 0;
  let end = 0;
  while (end < text.length) {
    const code = text.charCodeAt(end);
    const pair = code >= 0xd800 && code <= 0xdbff && end + 1 < text.length;
    const step = pair ? 2 : 1;
    const size = utf8ByteLength(text.slice(end, end + step));
    if (bytes + size > maxBytes) break;
    bytes += size;
    end += step;
  }
  return text.slice(0, end);
}

// ─── History helpers ─────────────────────────────────────────────────────────

/**
 * True when a user message is itself a previous compaction summary.
 * mirrors is_summary_message (compact.rs:550).
 */
export function isSummaryMessage(message: string): boolean {
  return message.startsWith(`${SUMMARY_PREFIX}\n`);
}

/**
 * Extract plain-text content from all user messages in history, skipping
 * previous compaction summaries.
 * mirrors collect_user_messages (compact.rs:525): a message that starts with
 * SUMMARY_PREFIX is dropped, so a second compaction does not re-collect the
 * first summary as a "user message" and stack summaries on top of each other.
 * Note the summary produced by THIS compaction is appended separately by
 * buildCompactedHistory.
 *
 * `isContextual` drops host-injected context the same way (rs's
 * parse_user_message returns None for contextual fragments,
 * event_mapping.rs:114-117); it sees the flattened text collected here.
 */
export function collectUserMessages(
  history: ConversationItem[],
  isContextual?: ((text: string) => boolean) | undefined,
): string[] {
  const messages: string[] = [];
  const keep = (text: string): boolean =>
    text !== "" && !isSummaryMessage(text) && !(isContextual?.(text) ?? false);
  for (const item of history) {
    if (!("role" in item) || item.role !== "user") continue;
    if (typeof item.content === "string") {
      if (keep(item.content)) messages.push(item.content);
    } else if (Array.isArray(item.content)) {
      const text = item.content
        .filter((p): p is { type: "input_text"; text: string } => p.type === "input_text")
        .map((p) => p.text)
        .join("\n");
      if (keep(text)) messages.push(text);
    }
  }
  return messages;
}

/**
 * Build the replacement history from recent user messages + summary.
 * mirrors build_compacted_history_with_limit in compact.rs:
 *   - Walk user messages in reverse, filling up to COMPACT_USER_MESSAGE_MAX_TOKENS
 *   - Append the summary as the final user message
 */
export function buildCompactedHistory(
  userMessages: string[],
  summaryText: string,
): ConversationItem[] {
  const maxTokens = COMPACT_USER_MESSAGE_MAX_TOKENS;
  let remaining = maxTokens;
  const selected: string[] = [];

  for (let i = userMessages.length - 1; i >= 0; i--) {
    if (remaining === 0) break;
    const msg = userMessages[i]!;
    const tokens = approxTokenCount(msg);
    if (tokens <= remaining) {
      selected.unshift(msg);
      remaining -= tokens;
    } else {
      selected.unshift(truncateToTokens(msg, remaining));
      break;
    }
  }

  const history: ConversationItem[] = [];
  for (const text of selected) {
    history.push({
      type: "message",
      role: "user",
      content: [{ type: "input_text", text }],
    });
  }

  const finalSummary = summaryText || "(no summary available)";
  history.push({
    type: "message",
    role: "user",
    content: [{ type: "input_text", text: finalSummary }],
  });

  return history;
}

// ─── Compaction request ───────────────────────────────────────────────────────

/**
 * Run an inline compaction turn:
 *   1. Send SUMMARIZATION_PROMPT + current history to the model
 *   2. Collect the assistant's summary text
 *   3. Build replacement history and mutate `history` in-place
 *
 * mirrors run_inline_auto_compact_task + run_compact_task_inner_impl in compact.rs
 * Returns the raw summary text (without SUMMARY_PREFIX), or null when the
 * compaction was dropped as STALE — in that case `history` is left exactly as it
 * was found. See `isStaleCompaction` below.
 *
 * Request shape. Given `tools` (and unless `compactionRequestParity` is false)
 * the request EXTENDS the turn's sampling request: the same tools, reasoning
 * and instructions, input = [contextItems, skillInjectionItems, history,
 * prompt] — what the last sampling request sent, plus what was recorded since,
 * plus the prompt. mirrors upstream's remote compaction
 * (compact_remote_v2_attempt.rs:178-225, #44255). Anthropic renders tools →
 * system → messages and caches by prefix, so the legacy body — no tools, no
 * reasoning — could never read the main chain's cache: every compaction paid a
 * full cold write of a 600–900k history that is replaced right after. No
 * `tool_choice` is added (a tool_choice change invalidates Anthropic's
 * messages cache), so the model may answer with a tool call instead of a
 * summary; such an attempt — or one with no summary text — switches the rest
 * of this compaction to the legacy body. The prompt is the LAST input item in
 * both shapes, as `{type:"message", role:"user", content: <prompt string>}`:
 * hosts recognize compaction requests by exactly that item.
 */
export async function runInlineAutoCompactTask(
  history: ConversationItem[],
  config: TurnConfig & {
    /**
     * Asked immediately before this summary would rewrite `history`, and
     * before each summarizer request once the request gate is held. Returning
     * true means another compaction landed since this one began, so its
     * summary would describe a history that no longer exists.
     * See AutoCompactWindow.currentCompactionEpoch for why that is reachable.
     * Absent = never stale, i.e. codex-rs behavior.
     */
    isStaleCompaction?: (() => boolean) | undefined;
    /**
     * Called SYNCHRONOUSLY right after `history` has been rewritten — same
     * tick, no await in between — with the new history (now the live array)
     * and the full summary message. The caller persists the checkpoint here:
     * any await between the swap and the write would let a concurrent turn's
     * append land on disk first and be lost on resume.
     */
    onHistoryReplaced?:
      | ((replacement: ConversationItem[], summaryMessage: string) => void)
      | undefined;
    /** The turn's interrupt signal: aborts the summarizer request and any
     *  retry wait, and guarantees an interrupted compaction rewrites nothing. */
    abortSignal?: AbortSignal | undefined;
    /** The `tools` array (request JSON) the turn's sampling requests carry.
     *  Absent → the legacy request shape (see above). */
    tools?: Record<string, unknown>[] | undefined;
    /** The turn's `$skill` injection items, which its sampling requests
     *  carry between contextItems and history. Parity shape only. */
    skillInjectionItems?: ConversationItem[] | undefined;
  },
): Promise<string | null> {
  // Working copy sent to the summarizer. When the compaction request ITSELF
  // exceeds the context window, drop the OLDEST item and retry (mirrors
  // compact.rs:232-241 history.remove_first_item(): trims from the front to
  // keep the prefix cache and the recent messages intact). The replacement
  // history below is still built from the ORIGINAL `history` (mirrors rs,
  // which collects user messages from the untrimmed session history).
  // Same pairing invariants as a sampling request (mirrors for_prompt →
  // normalize_history) — Anthropic 400s on unpaired tool_use/tool_result.
  const summarizerInput: ConversationItem[] = [...history];
  normalizeHistory(summarizerInput);
  // The items this summary will describe, by identity. Concurrent turns share
  // `history` and keep appending while the summarizer runs — whatever is not
  // in this set at apply time is NEWER than the summary and is kept after
  // it, instead of being silently dropped. mirrors the rs prefix install
  // that appends the live suffix (session/mod.rs replace_compacted_history)
  // and #59404.
  const summarized = new Set<ConversationItem>(history);
  const abortSignal = config.abortSignal;
  const throwIfAborted = (): void => {
    if (abortSignal?.aborted) {
      throw new DOMException("Compaction interrupted", "AbortError");
    }
  };
  const maxRetries = config.maxRetries ?? DEFAULT_MAX_RETRIES;
  let retries = 0;
  let trimRounds = 0;
  let summaryText = "";

  // mirrors run_inline_auto_compact_task (compact.rs:118): the host may override
  // the summarization prompt (`config.compact_prompt`), else the bundled one.
  const compactPrompt = config.compactPrompt ?? SUMMARIZATION_PROMPT;
  // The sampling request's shape (see the doc comment). Cleared for the rest
  // of this compaction once an attempt calls a tool or produces no summary.
  let parity =
    config.compactionRequestParity !== false && config.tools !== undefined;
  // What the sampling request sends ahead of history. Outside
  // `summarizerInput`, so the context-window trim below only drops history.
  const requestPrefix: ConversationItem[] = [
    ...(config.contextItems ?? []),
    ...(config.skillInjectionItems ?? []),
  ];

  for (;;) {
    throwIfAborted();
    // Build the compaction request: history + the summarization prompt at the end
    const compactInput: ConversationItem[] = [
      ...(parity ? requestPrefix : []),
      ...summarizerInput,
      { type: "message", role: "user", content: compactPrompt },
    ];

    // Keys in the sampling body's order (session/turn.ts).
    const body: Record<string, unknown> = parity
      ? {
          model: config.model,
          reasoning: { effort: config.reasoningEffort ?? "medium" },
          input: compactInput,
          tools: config.tools,
          stream: true,
        }
      : {
          model: config.model,
          input: compactInput,
          stream: true,
        };
    if (config.instructions) body["instructions"] = config.instructions;

    // Same per-thread request gate as a sampling request (request_gate.ts).
    const releaseRequest = await acquireRequestGate(
      config.requestGate,
      abortSignal,
    );
    try {
      // Two turns that both decided to compact queue on the gate, and by the
      // time the second holds it the first may already have compacted. Drop
      // the stale one BEFORE sending — otherwise it pays a full summarizer
      // request over the already-compacted history only to be dropped by the
      // final check below.
      if (config.isStaleCompaction?.()) return null;
      const res = await (config.fetch ?? fetch)(`${config.baseUrl}/responses`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.apiKey}`,
        },
        body: JSON.stringify(body),
        signal: abortSignal ?? null,
      });

      if (!res.ok || !res.body) {
        const text = res.body ? await res.text() : "(no body)";
        throw classifyHttpFailure(res.status, text, res.headers);
      }

      // Collect the assistant's summary text from the stream
      summaryText = "";
      let sawCompleted = false;
      let sawToolCall = false;
      let sawSummaryDelta = false;
      for await (const raw of parseSseStream(res.body, {
        idleTimeoutMs:
          config.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS,
      })) {
        if (raw["type"] === "response.output_text.delta") {
          summaryText += String(raw["delta"] ?? "");
          sawSummaryDelta = true;
        } else if (
          raw["type"] === "response.output_item.added" ||
          raw["type"] === "response.output_item.done"
        ) {
          const item = raw["item"] as Record<string, unknown> | undefined;
          if (item?.["type"] === "function_call") sawToolCall = true;
          // A summarizer whose text arrives only in the completed message
          // item (no deltas) would otherwise yield an empty summary — rs
          // takes the summary from the recorded item (compact.rs:347-350).
          if (
            raw["type"] === "response.output_item.done" &&
            item?.["type"] === "message" &&
            !sawSummaryDelta
          ) {
            summaryText += messageOutputText(item);
          }
        } else if (raw["type"] === "response.completed") {
          // First terminal event wins: anything after it is transport noise
          // (same rule as the turn loop). Leaving the loop cancels the body.
          sawCompleted = true;
          break;
        } else if (
          raw["type"] === "response.failed" ||
          raw["type"] === "response.incomplete" ||
          raw["type"] === "error"
        ) {
          // Terminal stream event — same classification as the turn loop
          // (context-window-exceeded → trim; quota/policy failures and the
          // output cap → terminal; otherwise retryable). mirrors
          // process_responses_event.
          throw classifyStreamFailure(raw);
        }
      }
      // With tools advertised the model may call one instead of summarizing,
      // or answer with no text. Neither is transient — the same body asks the
      // same question — so retry at once with the legacy body, which has no
      // tools to call. One switch per compaction; it costs no retry budget.
      if (
        parity &&
        (sawToolCall || (sawCompleted && summaryText.trim() === ""))
      ) {
        parity = false;
        console.warn(
          `[codex] compaction request with the sampling request's tools ` +
            `returned ${sawToolCall ? "a tool call" : "no summary text"}; ` +
            `retrying this compaction without tools`,
        );
        continue;
      }
      // A summary is only a summary once the response COMPLETED: a stream that
      // just ends is cut off, and applying it would replace the history with
      // half a handoff — now durably, since the result is checkpointed.
      // mirrors compact.rs "stream closed before response.completed" (#53932
      // buffers until completion). An empty summary is rejected too, a
      // deliberate tightening: this host has seen upstreams launder errors
      // into empty 200 completions, and "(no summary available)" in place of
      // the whole history is data loss. Both stay retryable.
      if (!sawCompleted) {
        throw new ResponsesApiError(
          503,
          "summarizer stream closed before response.completed",
        );
      }
      if (summaryText.trim() === "") {
        throw new ResponsesApiError(503, "summarizer returned an empty summary");
      }
      break; // summary collected → leave the retry loop
    } catch (err) {
      // An interrupt is terminal — never retry or trim past a cancellation.
      if (abortSignal?.aborted) throw err;
      if (isContextWindowExceededError(err)) {
        if (summarizerInput.length > 1) {
          // mirrors compact.rs:233-240 + history.remove_first_item: drop the
          // oldest item AND its call/output counterpart (pairing stays intact),
          // then try again.
          const removed = summarizerInput.shift();
          if (removed) removeCorrespondingFor(summarizerInput, removed);
          // rs drops exactly ONE item per rejected probe — fine on its 200k
          // windows where the overflow is a few k tokens. A pathological
          // thread on a 1M window can overflow by 40%+ (prod 2026-07-18
          // a1d14926: 1.42M tokens), where one-item-per-request means
          // hundreds of full-size round trips — the rs semantic ("trim from
          // the front until the summarizer fits") never completes in
          // practice. Reach the same end state in bulk: after the rs-style
          // shift, keep trimming until the byte-based estimate fits a
          // GEOMETRICALLY SHRINKING target (contextWindow × 0.85^round). The
          // estimate undercounts CJK (~chars/4), so a fixed target could
          // stall above the real limit — the shrinking target guarantees
          // each rejected probe cuts strictly deeper, converging in a few
          // rounds instead of hundreds.
          trimRounds += 1;
          if (config.contextWindow !== undefined) {
            const target = Math.max(
              1,
              Math.floor(config.contextWindow * Math.pow(0.85, trimRounds)),
            );
            while (
              summarizerInput.length > 1 &&
              estimateItemsTokenCount(summarizerInput) > target
            ) {
              const next = summarizerInput.shift();
              if (next) removeCorrespondingFor(summarizerInput, next);
            }
          }
          retries = 0;
          continue;
        }
        // Nothing left to trim — mark tokens full and surface the error
        // (mirrors compact.rs:242-246).
        config.tokenState?.setFull(config.contextWindow);
        throw err;
      }
      // Transient failures back off and retry (mirrors compact.rs:248-259).
      if (retries < maxRetries && isRetryableError(err)) {
        retries += 1;
        await sleep(computeRetryDelay(err, retries), abortSignal);
        continue;
      }
      throw err;
    } finally {
      releaseRequest();
    }
  }

  // LAST CHANCE TO NOT REWRITE HISTORY. Everything above is a network round
  // trip, and `history` is the thread's LIVE array, shared with every concurrent
  // turn. The check sits here — immediately before the mutation — so that no
  // reordering of the code above can leave a window where a stale summary still
  // gets applied. Returning null also tells the caller to skip the
  // ContextCompacted event and the token reseed.
  throwIfAborted();
  if (config.isStaleCompaction?.()) {
    return null;
  }

  const fullSummary = `${SUMMARY_PREFIX}\n${summaryText}`;

  // Split the live history into what the summary covers and what arrived
  // while it was being written (see `summarized` above).
  const covered = history.filter((item) => summarized.has(item));
  const newer = history.filter((item) => !summarized.has(item));

  // Collect recent user messages from the summarized part only
  const userMessages = collectUserMessages(
    covered,
    config.isContextualUserMessage,
  );

  // Build replacement history and mutate in place: summary first, then the
  // newer items (mirrors compact_pending_input::insert_summary). An output
  // whose call was summarized away is dropped: it would be an orphan. Only
  // that half of normalization: this array BECOMES the live history, and a
  // newer call still waiting for its output (a concurrent turn's tool) must
  // not get a synthetic "aborted" here — placeholders live only in request
  // copies (historyForPrompt), or the real output lands as a second one.
  const newHistory = [
    ...buildCompactedHistory(userMessages, fullSummary),
    ...newer,
  ];
  removeOrphanOutputs(newHistory);
  history.length = 0;
  for (const item of newHistory) history.push(item);
  config.onHistoryReplaced?.(history, fullSummary);

  return summaryText;
}
