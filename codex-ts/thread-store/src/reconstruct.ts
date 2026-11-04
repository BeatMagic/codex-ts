/**
 * mirrors codex-rs/core/src/session/rollout_reconstruction.rs (subset)
 *         codex-rs/core/src/session/mod.rs last_token_info_from_rollout
 *
 * Rebuild the model-visible context of a thread from its persisted rollout:
 * start from the NEWEST compaction checkpoint's replacement history and replay
 * only the conversation items appended after it.
 *
 * Pure: no I/O, no clocks. LiveThread.loadResumeState feeds it the stored lines.
 */

import type { CompactedItem, ConversationItem, RolloutItem } from "./types.js";

/** What a resumed thread needs to continue where it left off. */
export interface ResumeState {
  /** Model-visible history: the checkpoint snapshot + every later item. */
  history: ConversationItem[];
  /**
   * The last persisted active-context total AFTER the selected checkpoint
   * (server usage, post-compaction estimate, or forced-full), else the
   * checkpoint's own estimate, else null (legacy logs).
   */
  lastTokenTotal: number | null;
  /** The checkpoint the history was rebuilt from, when one was usable. */
  checkpoint: CompactedItem | null;
}

const CONVERSATION_ITEM_TYPES = new Set([
  "message",
  "function_call",
  "function_call_output",
]);

function isConversationItemShape(value: unknown): value is ConversationItem {
  return (
    typeof value === "object" &&
    value !== null &&
    CONVERSATION_ITEM_TYPES.has(String((value as { type?: unknown }).type))
  );
}

/**
 * A checkpoint is usable only when its snapshot is a non-empty array of
 * well-formed items. An unusable NEWEST checkpoint means a full raw replay —
 * never an older checkpoint (mirrors select_input_compaction, which never
 * falls back past the newest compaction: an older snapshot would drop the
 * items the newer, broken one summarized).
 */
function isUsableCheckpoint(value: unknown): value is CompactedItem {
  if (typeof value !== "object" || value === null) return false;
  const history = (value as { replacement_history?: unknown })
    .replacement_history;
  return (
    Array.isArray(history) &&
    history.length > 0 &&
    history.every(isConversationItemShape)
  );
}

/** total_tokens of a persisted TokenCount event, when the line carries one. */
function tokenCountTotal(item: RolloutItem): number | null {
  if (item.kind !== "EventMsg" || item.msg.type !== "TokenCount") return null;
  const total = item.msg.event.info?.total_token_usage.total_tokens;
  return typeof total === "number" && Number.isFinite(total) ? total : null;
}

function deepCopy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/**
 * Rebuild the resume state from rollout items in append order.
 *
 * Token seeding is stricter than rs: rs takes the last TokenCount anywhere,
 * which is safe there because a recompute TokenCount always follows its
 * Compacted line. A log written partly by an older runtime (which persisted
 * neither) has no such guarantee, and a pre-checkpoint total describes a
 * history that no longer exists — so only lines after the checkpoint count.
 */
export function reconstructModelContext(items: RolloutItem[]): ResumeState {
  let newestCompactedIndex = -1;
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (items[index]?.kind === "Compacted") {
      newestCompactedIndex = index;
      break;
    }
  }

  const newest =
    newestCompactedIndex >= 0
      ? (items[newestCompactedIndex] as Extract<RolloutItem, { kind: "Compacted" }>)
          .compacted
      : null;
  const checkpoint = newest !== null && isUsableCheckpoint(newest) ? newest : null;

  const history: ConversationItem[] = checkpoint
    ? deepCopy(checkpoint.replacement_history)
    : [];
  const replayFrom = checkpoint ? newestCompactedIndex + 1 : 0;
  // A full raw replay forced by a BROKEN newest checkpoint must not seed from
  // the TokenCounts written after it: those measured the compacted context,
  // while the replayed history is the raw one — seeding from them would
  // under-report exactly the thread that most needs to compact.
  const seedFromTokenCounts = checkpoint !== null || newestCompactedIndex < 0;
  let lastTokenTotal: number | null = null;

  for (let index = replayFrom; index < items.length; index += 1) {
    const item = items[index];
    if (item === undefined) continue;
    if (item.kind === "ConversationItem") {
      if (isConversationItemShape(item.item)) history.push(item.item);
      continue;
    }
    const total = seedFromTokenCounts ? tokenCountTotal(item) : null;
    if (total !== null) lastTokenTotal = total;
  }

  if (lastTokenTotal === null && checkpoint) {
    const estimate = checkpoint.token_total_after;
    if (typeof estimate === "number" && Number.isFinite(estimate)) {
      lastTokenTotal = estimate;
    }
  }

  return { history, lastTokenTotal, checkpoint };
}
