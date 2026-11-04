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
      // Typed string, but hosts store content-part arrays here (view_image).
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

/** Where the current total came from — decides whether locally appended items
 *  still have to be added on top of it (see getTotalTokenUsage). */
type TotalSource = "server" | "estimated" | "full";

export class SessionTokenState {
  /** Last known active-context size in tokens (server usage, estimate, or
   *  forced-full). null until the first sample/estimate. */
  private total: number | null = null;
  private source: TotalSource | null = null;

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
   * estimate of everything appended to history after the most recent
   * model-generated item.
   * mirrors get_total_token_usage (history.rs:421 / session/mod.rs:1256).
   *
   * Why the addition matters: a tool that returns 200 KB lands in history right
   * after the response that requested it, and the next request of the SAME turn
   * carries it — but no usage frame has counted it yet. Reading the bare server
   * total there says "unchanged since the last response" and sails into a
   * `prompt is too long` 400 that compaction should have prevented.
   *
   * Deviation from rs, deliberate: rs adds the appended-items estimate
   * unconditionally, including right after `recompute_token_usage` has just
   * written a FULL-history estimate into the same field — double-counting the
   * tail. codex-ts tracks provenance instead and only adds when the total came
   * from the server, which is the case the rs comment itself describes ("these
   * are local items … not reflected in last_token_usage.total_tokens"). An
   * estimated total already covers them; a forced-full total is saturated.
   *
   * rs also adds `get_non_last_reasoning_items_tokens()` when the server did
   * not account for reasoning. codex-ts's ConversationItem has no reasoning
   * variant — encrypted reasoning is never stored in history — so that term is
   * structurally 0 and is left out rather than ported as dead code.
   */
  getTotalTokenUsage(items: ConversationItem[]): number {
    const last = this.total ?? 0;
    if (this.source === "estimated" || this.source === "full") {
      return last;
    }
    return last + estimateTokensAfterLastModelGeneratedItem(items);
  }

  /**
   * Record the terminal usage of a sampled response.
   * mirrors record_token_usage_info → history.update_token_info: the session
   * total becomes the response's total_tokens (the active context as the
   * server measured it). Falls back to input+output when total is absent.
   */
  updateFromUsage(input: {
    inputTokens: number;
    outputTokens?: number | undefined;
    totalTokens?: number | undefined;
  }): void {
    const total =
      input.totalTokens ?? input.inputTokens + (input.outputTokens ?? 0);
    this.total = Math.max(0, total);
    this.source = "server";
  }

  /**
   * Restore a total persisted by a previous run of this thread (resume).
   * mirrors seeding token_info from the last rollout TokenCount
   * (session/mod.rs last_token_info_from_rollout). Treated as a server total:
   * items appended after the last model output are added on top of it.
   */
  restore(tokens: number): void {
    this.total = Math.max(0, tokens);
    this.source = "server";
  }

  /**
   * Seed/replace the total with a byte-based estimate — used before the first
   * server usage arrives and after compaction rewrites history.
   * mirrors recompute_token_usage (session/mod.rs:3994).
   */
  setEstimated(tokens: number): void {
    this.total = Math.max(0, tokens);
    this.source = "estimated";
  }

  /**
   * Mark the context as FULL after a context-window-exceeded rejection, so the
   * next pre-sampling check is guaranteed to compact first.
   * mirrors set_total_tokens_full (session/mod.rs:4077) — a no-op when the
   * host didn't configure a context window, exactly like the rs guard.
   */
  setFull(contextWindow: number | undefined): void {
    if (contextWindow !== undefined) {
      this.total = contextWindow;
      this.source = "full";
    }
  }
}

/** Arguments to {@link recomputeTokenUsage}. */
export interface RecomputeTokenUsageParams {
  /** Everything the next request will carry — context items AND history. */
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
    tokenState.setEstimated(estimated);
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
