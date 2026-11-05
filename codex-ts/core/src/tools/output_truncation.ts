/**
 * mirrors codex-rs/utils/output-truncation/src/lib.rs
 *         codex-rs/utils/string/src/truncate.rs (truncate_middle_chars)
 *         codex-rs/core/src/context_manager/history.rs (truncate_function_output_payload)
 *
 * Tool outputs are middle-truncated before they enter the MODEL-VISIBLE
 * history, while the rollout keeps the original: rs record_items truncates
 * the history copy and persist_rollout_response_items writes the untouched
 * item (session/mod.rs), and a resume re-applies the policy to what it
 * replays (rollout_reconstruction.rs → record_items). Without it one large
 * output is re-sent in full on every later request until compaction — prod
 * saw 1.68M chars of tool output in a single request.
 *
 * Only the Bytes policy is ported: it is the one rs applies to every model
 * outside its catalog, which is every model this package is used with. Unlike
 * rs the policy is opt-in PER TOOL (CustomTool.maxOutputBytes): a host tool
 * that returns a document the model must read whole stays untouched.
 */

import type {
  ConversationItem,
  FunctionCallOutputContentItem,
} from "../../../thread-store/src/types.js";
import { utf8ByteLength } from "../state/token_state.js";

/**
 * The per-output budget codex-rs applies to models outside its catalog (every
 * Claude / DeepSeek / Kimi slug reached through OpenRouter):
 * TruncationPolicy::Bytes(10_000) from model_info_from_slug
 * (models-manager/src/model_info.rs) × 1.2 serialization allowance
 * (history.rs process_item). Not applied by default — pass it as
 * CustomTool.maxOutputBytes to opt a tool in.
 */
export const DEFAULT_TOOL_OUTPUT_MAX_BYTES = 12_000;

/** UTF-16 units and UTF-8 bytes of the code point starting at `index`; a lone
 *  surrogate is one unit of 3 bytes (it encodes as U+FFFD), as in
 *  utf8ByteLength. */
function codePointAt(text: string, index: number): { units: number; bytes: number } {
  const code = text.charCodeAt(index);
  if (code < 0x80) return { units: 1, bytes: 1 };
  if (code < 0x800) return { units: 1, bytes: 2 };
  if (code >= 0xd800 && code <= 0xdbff) {
    const next = text.charCodeAt(index + 1);
    if (next >= 0xdc00 && next <= 0xdfff) return { units: 2, bytes: 4 };
  }
  return { units: 1, bytes: 3 };
}

/**
 * Keep at most `maxBytes` UTF-8 bytes from the two ends of `text` (half each)
 * and replace the middle with "…N chars truncated…", N counting code points.
 * The marker sits outside the budget. Cuts only on code-point boundaries, so a
 * surrogate pair is never split. mirrors truncate_middle_chars + split_string
 * (utils/string/src/truncate.rs).
 */
export function truncateMiddleChars(text: string, maxBytes: number): string {
  const totalBytes = utf8ByteLength(text);
  if (totalBytes <= maxBytes) return text;

  const leftBudget = Math.floor(maxBytes / 2);
  const rightBudget = maxBytes - leftBudget;
  const tailStartTarget = Math.max(0, totalBytes - rightBudget);
  let prefixEnd = 0;
  let suffixStart = text.length;
  let suffixStarted = false;
  let removedChars = 0;
  let byteIndex = 0;
  for (let index = 0; index < text.length; ) {
    const { units, bytes } = codePointAt(text, index);
    if (byteIndex + bytes <= leftBudget) {
      prefixEnd = index + units;
    } else if (byteIndex >= tailStartTarget) {
      if (!suffixStarted) {
        suffixStart = index;
        suffixStarted = true;
      }
    } else {
      removedChars += 1;
    }
    byteIndex += bytes;
    index += units;
  }
  if (suffixStart < prefixEnd) suffixStart = prefixEnd;

  return `${text.slice(0, prefixEnd)}…${removedChars} chars truncated…${text.slice(suffixStart)}`;
}

/**
 * Spend the byte budget on the text items in order; images are kept and cost
 * nothing. The item that crosses the budget is middle-truncated, later text
 * items are dropped and counted in a trailing "[omitted N text items ...]".
 * mirrors truncate_function_output_items_with_policy (Bytes policy) at the
 * sync point. Returns `items` itself when nothing had to change, so an output
 * within budget is byte-identical to what the tool returned.
 */
export function truncateFunctionOutputItems(
  items: FunctionCallOutputContentItem[],
  maxBytes: number,
): FunctionCallOutputContentItem[] {
  const out: FunctionCallOutputContentItem[] = [];
  let remaining = maxBytes;
  let omittedTextItems = 0;
  let changed = false;

  for (const item of items) {
    // Anything but well-formed text (images, or an unknown part a host
    // passed through) is kept as-is, like rs keeps images.
    if (item.type !== "input_text" || typeof item.text !== "string") {
      out.push(item);
      continue;
    }
    if (remaining === 0) {
      omittedTextItems += 1;
      changed = true;
      continue;
    }
    const cost = utf8ByteLength(item.text);
    if (cost <= remaining) {
      out.push(item);
      remaining -= cost;
    } else {
      const snippet = truncateMiddleChars(item.text, remaining);
      if (snippet === "") {
        omittedTextItems += 1;
      } else {
        out.push({ type: "input_text", text: snippet });
      }
      remaining = 0;
      changed = true;
    }
  }

  if (omittedTextItems > 0) {
    out.push({
      type: "input_text",
      text: `[omitted ${omittedTextItems} text items ...]`,
    });
  }
  return changed ? out : items;
}

/**
 * Apply a byte budget to one tool output, string or content items. mirrors
 * truncate_function_output_payload (history.rs): Text → truncate_text,
 * ContentItems → truncate_function_output_items_with_policy. Anything else (a
 * host returning an off-type value) is passed through untouched.
 */
export function truncateFunctionCallOutput(
  output: string | FunctionCallOutputContentItem[],
  maxBytes: number,
): string | FunctionCallOutputContentItem[] {
  if (typeof output === "string") return truncateMiddleChars(output, maxBytes);
  if (Array.isArray(output)) return truncateFunctionOutputItems(output, maxBytes);
  return output;
}

/**
 * The model-visible form of a function_call_output under a tool's policy: the
 * item itself when there is no policy or nothing to cut, else a copy with the
 * truncated output. The caller persists the ORIGINAL item.
 */
export function applyOutputPolicy(
  item: ConversationItem,
  maxBytes: number | undefined,
): ConversationItem {
  if (item.type !== "function_call_output" || maxBytes === undefined) return item;
  const output = truncateFunctionCallOutput(item.output, maxBytes);
  return output === item.output ? item : { ...item, output };
}

/**
 * Re-apply each tool's output policy to the items a resume replayed from the
 * rollout, which stores outputs whole. Only `items[from..]` — the raw suffix:
 * a compaction checkpoint's replacement_history is already the model-visible
 * form, and truncating it again would replace its markers with new ones.
 * mirrors rollout_reconstruction.rs, which installs the replacement history
 * via history.replace() and runs record_items over the suffix only. The tool
 * of an output is found through its call_id, anywhere in `items`.
 */
export function truncateReplayedToolOutputs(
  items: ConversationItem[],
  from: number,
  maxOutputBytesFor: (toolName: string) => number | undefined,
): void {
  const toolNames = new Map<string, string>();
  for (const item of items) {
    if (item.type === "function_call") toolNames.set(item.call_id, item.name);
  }
  for (let index = Math.max(0, from); index < items.length; index += 1) {
    const item = items[index]!;
    if (item.type !== "function_call_output") continue;
    const toolName = toolNames.get(item.call_id);
    if (toolName === undefined) continue;
    items[index] = applyOutputPolicy(item, maxOutputBytesFor(toolName));
  }
}
