/**
 * mirrors codex-rs/thread-store/src/types.rs  (subset)
 * mirrors codex-rs/protocol/src/protocol.rs   (RolloutItem)
 */

import type { EventMsg } from "../../protocol/src/protocol.js";

// ─── ConversationItem (what gets sent to the Responses API) ───────────────────
//
// Every variant carries an explicit `type` discriminator, mirroring codex-rs's
// `ResponseItem` enum (`#[serde(tag = "type", rename_all = "snake_case")]`):
// message items MUST serialize with `type: "message"` on the wire, exactly like
// codex-rs. Discriminating user/assistant by `role` alone (omitting `type`) is a
// silent divergence — strict Responses-API translators that key off `type` will
// drop these items.
//
// User-message content parts mirror codex-rs `ContentItem` (models.rs): text →
// `input_text`, image → `input_image { image_url, detail? }`. `detail` is omitted
// for plain user uploads (codex-rs marks it `skip_serializing_if = None`; the
// server then applies its default), and is kept optional here for the same reason.

/** mirrors codex-rs/protocol/src/models.rs ImageDetail */
export type ImageDetail = "auto" | "low" | "high" | "original";

/** mirrors codex-rs/protocol/src/models.rs ContentItem (input variants) */
export type UserContentPart =
  | { type: "input_text"; text: string }
  | { type: "input_image"; image_url: string; detail?: ImageDetail }
  /** mirrors ContentItem::InputAudio (models.rs:723) — a base64 data URI. */
  | { type: "input_audio"; audio_url: string };

export type ConversationItem =
  | {
      type: "message";
      role: "user";
      content: string | UserContentPart[];
    }
  | { type: "message"; role: "assistant"; content: string }
  | { type: "function_call"; call_id: string; name: string; arguments: string }
  | { type: "function_call_output"; call_id: string; output: string };

// ─── RolloutItem — mirrors codex-rs/protocol/src/protocol.rs RolloutItem ─────
//
// The serialisable unit persisted to the backing store. SessionMeta /
// ResponseItem / TurnContext / WorldState are omitted; codex-ts needs
// ConversationItem (history), EventMsg (TokenCount for resume seeding) and
// Compacted (the checkpoint a resume starts from).
//
// Compacted was omitted in the original port, and that was a production bug:
// compaction only rewrote the in-memory history, so a resumed thread replayed
// every raw item ever appended and came back larger than the context window
// (2026-11-04: a compacted 260KB director thread resumed as 2.5MB / 1.05M
// tokens and was rejected "prompt is too long").
//
// Back-compat: readers that predate Compacted keep only `kind ===
// "ConversationItem"` lines (LiveThread.loadConversationHistory), so they skip
// the new lines and fall back to a full raw replay — the old behavior, no crash.

/**
 * A compaction checkpoint. mirrors codex-rs CompactedItem
 * (history/src/lib.rs `Compacted(Box<CompactedItem>)`, written by
 * Session::replace_compacted_history in one append with the in-memory swap).
 */
export interface CompactedItem {
  /** Format version, for future readers. */
  v: 1;
  /** Full summary message text incl. SUMMARY_PREFIX (rs CompactedItem.message). */
  message: string;
  /**
   * The complete history right after the rewrite. A resume starts from this
   * snapshot and replays only the ConversationItems appended after this line.
   */
  replacement_history: ConversationItem[];
  /** Compaction window the rewrite opened (AutoCompactWindow.restore on resume). */
  window_number?: number | undefined;
  first_window_id?: string | undefined;
  previous_window_id?: string | null | undefined;
  window_id?: string | undefined;
  /** The recompute estimate of the compacted context — the resume seed when no
   *  later TokenCount line exists. */
  token_total_after?: number | undefined;
}

export type RolloutItem =
  | { kind: "ConversationItem"; item: ConversationItem }
  | { kind: "EventMsg"; msg: EventMsg }
  | { kind: "Compacted"; compacted: CompactedItem };

// ─── Params — mirrors codex-rs/thread-store/src/types.rs ─────────────────────

export interface CreateThreadParams {
  threadId: string;
  /** First user message, used as preview */
  preview?: string | undefined;
  model: string;
  createdAtMs: number;
}

export interface AppendThreadItemsParams {
  threadId: string;
  items: RolloutItem[];
}

export interface LoadThreadHistoryParams {
  threadId: string;
}

export interface ReadThreadParams {
  threadId: string;
}

export interface UpdateThreadMetadataParams {
  threadId: string;
  preview?: string | undefined;
  model?: string | undefined;
}

// ─── Stored types ─────────────────────────────────────────────────────────────

export interface ThreadMetadata {
  threadId: string;
  preview: string;
  model: string;
  createdAtMs: number;
  updatedAtMs: number;
}

export interface StoredThreadHistory {
  threadId: string;
  /** RolloutItems in append order */
  items: RolloutItem[];
}

export interface StoredThread {
  threadId: string;
  metadata: ThreadMetadata;
  /** Populated only when requested (mirrors include_history flag) */
  history?: StoredThreadHistory | undefined;
}
