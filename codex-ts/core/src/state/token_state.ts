/**
 * mirrors codex-rs/core/src/context_manager/history.rs (the token_info face)
 *         codex-rs/protocol/src/protocol.rs TokenUsageInfo::fill_to_context_window
 *
 * Session-scoped token accounting: the last known size of the active context,
 * fed from server usage after each sampled response, from a byte-based estimate
 * when no usage has arrived yet (fresh/resumed threads, post-compaction), or
 * force-filled to the context window when the provider rejects a request with
 * context-window-exceeded. `autoCompactTokenStatus` in turn.ts reads this to
 * decide when to compact — mirrors sess.get_total_token_usage() in codex-rs.
 */

import type { EventMsg } from "../../../protocol/src/protocol.js";
import type { ConversationItem } from "../../../thread-store/src/types.js";
import type { AutoCompactWindow } from "./auto_compact_window.js";

/**
 * UTF-8 byte length of a string without allocating an encoded copy (history
 * strings run to megabytes). 1 byte for ASCII, 2 up to U+07FF, 3 for the rest
 * of the BMP, 4 for a surrogate pair; a lone surrogate counts as 3 (U+FFFD).
 */
export function utf8ByteLength(text: string): number {
  let bytes = 0;
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    if (code < 0x80) {
      bytes += 1;
    } else if (code < 0x800) {
      bytes += 2;
    } else if (code >= 0xd800 && code <= 0xdbff) {
      const next = text.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        bytes += 4;
        index += 1;
      } else {
        bytes += 3;
      }
    } else {
      bytes += 3;
    }
  }
  return bytes;
}

/**
 * ~4 UTF-8 bytes per token — the codex-rs approx_token_count heuristic
 * (utils/string/src/truncate.rs: bytes, not chars). Counting UTF-16 units, as
 * the original port did, under-reports CJK text about 3× — and a resumed
 * thread's first compaction decision rests on this estimate.
 */
export function approxTokenCount(text: string): number {
  return Math.ceil(utf8ByteLength(text) / 4);
}

/** Fixed byte cost of an image part, whatever its URL form (rs counts images
 *  by modality, not by the size of their data URL: history.rs, #45094). */
const IMAGE_PART_BYTES = 7_373;
/** Fixed byte cost of an audio part (no duration is known here). */
const AUDIO_PART_BYTES = 7_373;
/** Per-item envelope (type / role / call_id keys) on the wire. */
const ITEM_ENVELOPE_BYTES = 32;

function jsonStringBytes(text: string): number {
  // JSON-escaped, as on the wire: quotes and backslashes in tool output JSON
  // inflate the request, and real token density of such content is HIGHER
  // than 4 bytes/token — keeping the escaping errs on the early side.
  return utf8ByteLength(JSON.stringify(text));
}

function contentBytes(content: unknown): number {
  if (typeof content === "string") return jsonStringBytes(content);
  if (Array.isArray(content)) {
    let bytes = 0;
    for (const part of content) {
      const type = (part as { type?: unknown } | null)?.type;
      if (type === "input_text" || type === "output_text") {
        bytes += jsonStringBytes(String((part as { text?: unknown }).text ?? ""));
      } else if (type === "input_image") {
        bytes += IMAGE_PART_BYTES;
      } else if (type === "input_audio") {
        bytes += AUDIO_PART_BYTES;
      } else {
        bytes += utf8ByteLength(JSON.stringify(part) ?? "");
      }
    }
    return bytes;
  }
  return utf8ByteLength(JSON.stringify(content) ?? "");
}

/** Estimated tokens of one conversation item. */
export function estimateItemTokenCount(item: ConversationItem): number {
  let bytes = ITEM_ENVELOPE_BYTES;
  switch (item.type) {
    case "message":
      bytes += contentBytes(item.content);
      break;
    case "function_call":
      bytes += utf8ByteLength(item.name) + jsonStringBytes(item.arguments);
      break;
    case "function_call_output":
      // A string or a content-item array (view_image): images at a fixed cost.
      bytes += contentBytes(item.output as unknown);
      break;
  }
  return Math.ceil(bytes / 4);
}

/**
 * Estimate the token size of conversation items.
 * mirrors history.estimate_token_count_with_base_instructions (history.rs:149):
 * per-item estimates, summed — UTF-8 bytes / 4, images and audio at a fixed
 * cost instead of the length of their (possibly base64) URL.
 */
export function estimateItemsTokenCount(items: ConversationItem[]): number {
  let total = 0;
  for (const item of items) {
    total += estimateItemTokenCount(item);
  }
  return total;
}

/**
 * Items the MODEL produced, as opposed to items the client appended locally.
 * mirrors is_model_generated_item (history.rs:898).
 *
 * codex-ts's ConversationItem has four variants where rs's ResponseItem has a
 * dozen; the classification is the same one rs makes: assistant messages and
 * tool CALLS come back inside a sampled response (so the server's usage number
 * already covers them), while user messages and tool call OUTPUTS are written
 * by the client after that response was measured.
 */
function isModelGeneratedItem(item: ConversationItem): boolean {
  switch (item.type) {
    case "message":
      return item.role === "assistant";
    case "function_call":
      return true;
    case "function_call_output":
      return false;
  }
}

/**
 * Estimated size of the items appended AFTER the most recent model-generated
 * item — the part of the active context no server usage number has seen yet.
 * mirrors items_after_last_model_generated_item +
 * estimated_tokens_after_last_model_generated_item (history.rs:408,440).
 *
 * When there is no model-generated item at all, rs's `rposition(...)
 * .map_or(len, ..)` yields an EMPTY slice (start == len), not the whole
 * history — a brand-new thread contributes 0 here, and the caller's estimated
 * seed covers it instead. Mirrored exactly.
 */
export function estimateTokensAfterLastModelGeneratedItem(
  items: ConversationItem[],
): number {
  let start = items.length;
  for (let index = items.length - 1; index >= 0; index -= 1) {
    const item = items[index];
    if (item !== undefined && isModelGeneratedItem(item)) {
      start = index + 1;
      break;
    }
  }
  return estimateItemsTokenCount(items.slice(start));
}

/**
 * The history position just after `coveredThrough` — or, with no usable
 * boundary, the last-model-output approximation rs installs whenever it
 * rewrites history (reset_last_usage_boundary, history.rs:1097).
 */
function uncoveredStart(
  items: ConversationItem[],
  coveredThrough: ConversationItem | null | undefined,
): number | null {
  if (coveredThrough === null) return 0;
  if (coveredThrough === undefined) return null;
  const index = items.lastIndexOf(coveredThrough);
  return index < 0 ? null : index + 1;
}

export class SessionTokenState {
  /** Last known active-context size in tokens (server usage, estimate, or
   *  forced-full). null until the first sample/estimate. */
  private total: number | null = null;
  /**
   * The last HISTORY item `total` accounts for, by object identity — the
   * covered-item boundary of #58404 (rs last_server_reported_usage_item_count,
   * history.rs:536-543). An object reference rather than an index, so it
   * survives concurrent appends (and the request's pairing placeholders, which
   * live only in its normalized copy, are never a boundary). `null`: no
   * history item is covered — everything counts,
   * like rs's boundary 0 on a fresh history. `undefined`: the total was
   * recorded without a boundary (a caller that predates it) — see
   * getTotalTokenUsage for the fallback.
   */
  private coveredThrough: ConversationItem | null | undefined = null;
  /**
   * A server measurement also covers the response's own output — rs's
   * max(last_model_output_item_count, input_item_count). rs has recorded the
   * output by the time usage arrives; codex-ts records it only after the
   * stream ends, so it lands right AFTER `coveredThrough`: the model-generated
   * run there belongs to the measured response.
   */
  private coversFollowingModelOutput = false;

  /**
   * The raw last-known number, with no allowance for what has been appended
   * since. Prefer `getTotalTokenUsage(history)` for any budgeting decision —
   * this getter is the equivalent of rs's `last_token_usage.total_tokens`, not
   * of `get_total_token_usage()`.
   */
  get totalTokens(): number | null {
    return this.total;
  }

  /**
   * Active context size for a budgeting decision: the last known total PLUS an
   * estimate of every history item that total does not cover yet.
   * mirrors get_total_token_usage (history.rs:1146-1163, after #58404).
   *
   * Why the addition matters: a tool that returns 200 KB lands in history right
   * after the response that requested it, and the next request of the SAME turn
   * carries it — but no usage frame has counted it yet. Reading the bare server
   * total there says "unchanged since the last response" and sails into a
   * `prompt is too long` 400 that compaction should have prevented.
   *
   * The covered-item boundary (#58404, commit 7a030b9769) replaces the
   * provenance rule this used to apply (add a tail to a server total only).
   * That rule had a blind spot: after an ESTIMATED total — a fresh or resumed
   * thread, or right after a compaction — nothing appended later was counted
   * until a server usage frame arrived, and a response without usable usage
   * never brings one. With the boundary, anything appended after ANY kind of
   * count is counted, and the region an estimate already covers never is.
   *
   * Fallback when the boundary is unusable — the total was recorded without
   * one, or its item has since left history (a rewrite that did not recompute):
   * count the items after the last model-generated item. That is rs's own
   * approximation for rewritten history (reset_last_usage_boundary) and exactly
   * what codex-ts 0.8.0 added to a server total. The alternatives are worse in
   * opposite directions: counting the whole history adds the covered region a
   * second time, which on a large thread trips a compaction nobody needed;
   * counting nothing reopens the blind spot.
   *
   * rs also adds `get_non_last_reasoning_items_tokens()` when the server did
   * not account for reasoning. codex-ts's ConversationItem has no reasoning
   * variant — encrypted reasoning is never stored in history — so that term is
   * structurally 0 and is left out rather than ported as dead code.
   */
  getTotalTokenUsage(items: ConversationItem[]): number {
    const last = this.total ?? 0;
    let start = uncoveredStart(items, this.coveredThrough);
    if (start === null) {
      return last + estimateTokensAfterLastModelGeneratedItem(items);
    }
    if (this.coversFollowingModelOutput) {
      while (start < items.length && isModelGeneratedItem(items[start]!)) {
        start += 1;
      }
    }
    return last + estimateItemsTokenCount(items.slice(start));
  }

  /**
   * Record the terminal usage of a sampled response.
   * mirrors record_token_usage_info → history.update_token_info: the session
   * total becomes the response's total_tokens (the active context as the
   * server measured it). Falls back to input+output when total is absent.
   *
   * `coveredThrough` is the last history item the request carried (null: it
   * carried none); the response's own output, recorded right after it, is
   * covered too (history.rs:1068-1094). Omitted → the boundary is unknown and
   * getTotalTokenUsage uses its fallback — the 0.8.0 rule for a server total.
   */
  updateFromUsage(input: {
    inputTokens: number;
    outputTokens?: number | undefined;
    totalTokens?: number | undefined;
    coveredThrough?: ConversationItem | null | undefined;
  }): void {
    const total =
      input.totalTokens ?? input.inputTokens + (input.outputTokens ?? 0);
    this.total = Math.max(0, total);
    this.coveredThrough = input.coveredThrough;
    this.coversFollowingModelOutput = input.coveredThrough !== undefined;
  }

  /**
   * Restore a total persisted by a previous run of this thread (resume).
   * mirrors seeding token_info from the last rollout TokenCount
   * (session/mod.rs last_token_info_from_rollout) into a history whose
   * boundary rs resets to just after the last model-generated item
   * (reset_last_usage_boundary, history.rs:1097-1106): the persisted number
   * measured a request plus its output, and whatever the client appended after
   * that output — tool results, a user message whose request never completed —
   * is counted on top. With no model-generated item the whole history counts
   * as covered, as in rs. Pass the resumed `history`; without it the same rule
   * is applied lazily at read time.
   */
  restore(tokens: number, history?: ConversationItem[]): void {
    this.total = Math.max(0, tokens);
    this.coversFollowingModelOutput = false;
    if (history === undefined) {
      this.coveredThrough = undefined;
      return;
    }
    let boundary: ConversationItem | null = history[history.length - 1] ?? null;
    for (let index = history.length - 1; index >= 0; index -= 1) {
      const item = history[index]!;
      if (isModelGeneratedItem(item)) {
        boundary = item;
        break;
      }
    }
    this.coveredThrough = boundary;
  }

  /**
   * Seed/replace the total with a byte-based estimate — used before the first
   * server usage arrives and after compaction rewrites history.
   * mirrors recompute_token_usage (session/mod.rs:3994), which installs the
   * estimate with the boundary at the end of the history it measured
   * (set_token_info_with_boundary(.., history.raw_items().len())). Pass that
   * last item as `coveredThrough` (recomputeTokenUsage does); omitted → the
   * getTotalTokenUsage fallback.
   */
  setEstimated(
    tokens: number,
    coveredThrough?: ConversationItem | null | undefined,
  ): void {
    this.total = Math.max(0, tokens);
    this.coveredThrough = coveredThrough;
    this.coversFollowingModelOutput = false;
  }

  /**
   * Mark the context as FULL after a context-window-exceeded rejection, so the
   * next pre-sampling check is guaranteed to compact first.
   * mirrors set_total_tokens_full (session/mod.rs:4077) — a no-op when the
   * host didn't configure a context window, exactly like the rs guard. The
   * boundary is left where it was (rs's set_token_usage_full does not touch
   * it): anything uncovered only adds to a total already at the window.
   */
  setFull(contextWindow: number | undefined): void {
    if (contextWindow !== undefined) {
      this.total = contextWindow;
    }
  }
}

/** Arguments to {@link recomputeTokenUsage}. */
export interface RecomputeTokenUsageParams {
  /** Everything the next request will carry — context items AND history,
   *  history last: the final item becomes the covered-item boundary. */
  items: ConversationItem[];
  /** The instructions blob sent alongside them, if any. */
  instructions?: string | undefined;
  tokenState: SessionTokenState;
  /** Reseeded with the same estimate, unless a server value already won. */
  compactWindow?: AutoCompactWindow | undefined;
  /** Reported on the emitted TokenCount event; not used in the arithmetic. */
  contextWindow?: number | undefined;
  /** Supply to emit TokenCount, as rs's recompute does via
   *  send_token_count_event. Omit for a silent reseed. */
  emitEvent?: ((msg: EventMsg) => void) | undefined;
  /** false leaves an already-known session total alone and reseeds only the
   *  window baseline. Default true — rs's recompute always writes back. */
  overwriteTotal?: boolean | undefined;
}

/**
 * Recompute the active-context size from the WHOLE history and write it back.
 * mirrors recompute_token_usage (session/mod.rs:3994).
 *
 * Call this after anything rewrites history behind the token bookkeeping's
 * back — compaction (rs: compact.rs:385) or a host-driven rollback/rebuild
 * (rs: handlers.rs:342). Without it the session total keeps describing a
 * history that no longer exists: too high after a compaction (so it compacts
 * again immediately), too low after a rebuild that grew the context.
 *
 * Returns the estimate it wrote.
 */
export function recomputeTokenUsage(params: RecomputeTokenUsageParams): number {
  const {
    items,
    instructions,
    tokenState,
    compactWindow,
    contextWindow,
    emitEvent,
    overwriteTotal = true,
  } = params;
  const estimated =
    estimateItemsTokenCount(items) + approxTokenCount(instructions ?? "");
  if (overwriteTotal) {
    tokenState.setEstimated(estimated, items[items.length - 1] ?? null);
  }
  compactWindow?.setEstimatedPrefill(estimated);
  if (emitEvent) {
    emitEvent(totalOnlyTokenCountEvent(estimated, contextWindow));
  }
  return estimated;
}

/**
 * A TokenCount event carrying only a total — for values that are not a
 * server measurement (a recompute estimate, a forced-full mark). mirrors the
 * TokenUsage rs writes into last_token_usage: every component zeroed except
 * the total, because an estimate cannot attribute input/output/cache shares.
 */
export function totalOnlyTokenCountEvent(
  total: number,
  contextWindow: number | undefined,
): EventMsg {
  const usage = {
    input_tokens: 0,
    cached_input_tokens: 0,
    cache_write_input_tokens: 0,
    output_tokens: 0,
    reasoning_output_tokens: 0,
    total_tokens: total,
  };
  return {
    type: "TokenCount",
    event: {
      info: {
        total_token_usage: { ...usage },
        last_token_usage: { ...usage },
        model_context_window: contextWindow ?? null,
      },
      rate_limits: null,
    },
  };
}
