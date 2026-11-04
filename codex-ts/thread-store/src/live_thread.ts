/**
 * mirrors codex-rs/thread-store/src/live_thread.rs
 *
 * LiveThread — handle for an active thread's persistence lifecycle.
 * Keeps persistence details inside so session code stays clean.
 */

import type { ThreadStore } from "./store.js";
import type { CompactedItem, ConversationItem, RolloutItem } from "./types.js";
import type { EventMsg } from "../../protocol/src/protocol.js";
import { reconstructModelContext, type ResumeState } from "./reconstruct.js";

function deepCopy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export class LiveThread {
  private readonly threadId: string;
  private readonly store: ThreadStore;
  /**
   * Tail of the ordered append queue. Every append chains onto it, so lines
   * land in the store in CALL order even though concurrent turns share one
   * thread and a host IoBackend may issue one network write per line.
   * mirrors rs's single rollout writer task.
   *
   * Why order matters now: resume starts from the newest Compacted line and
   * replays only the items after it. An item pushed after a compaction whose
   * write lands BEFORE the Compacted line would vanish on resume; one pushed
   * before it whose write lands AFTER would come back from the dead.
   */
  private tail: Promise<void> = Promise.resolve();

  constructor(threadId: string, store: ThreadStore) {
    this.threadId = threadId;
    this.store = store;
  }

  /**
   * Queue rollout items for persistence. The items are deep-copied
   * SYNCHRONOUSLY, at call time: the file must keep what the model saw at
   * that moment (e.g. a function_call's raw arguments), not a later in-place
   * mutation of the live object. The returned promise settles once these
   * items are written — and never rejects: a failed write is logged and the
   * thread carries on (rs treats rollout persistence failures as non-fatal;
   * the worst case is a less complete resume later).
   */
  private enqueue(items: RolloutItem[]): Promise<void> {
    const copied = deepCopy(items);
    const write = this.tail.then(() =>
      this.store.appendItems({ threadId: this.threadId, items: copied }),
    );
    this.tail = write.catch((err: unknown) => {
      console.warn(
        `[codex-ts] thread ${this.threadId}: persisting ${copied.length} ` +
          `rollout item(s) failed; continuing without them:`,
        err,
      );
    });
    return this.tail;
  }

  /** Persist one or more conversation items */
  appendConversationItems(items: ConversationItem[]): Promise<void> {
    return this.enqueue(
      items.map((item): RolloutItem => ({ kind: "ConversationItem", item })),
    );
  }

  /** Persist a key event (TokenCount, …) */
  appendEvent(msg: EventMsg): Promise<void> {
    return this.enqueue([{ kind: "EventMsg", msg }]);
  }

  /** Persist a compaction checkpoint (see CompactedItem). */
  appendCompacted(compacted: CompactedItem): Promise<void> {
    return this.enqueue([{ kind: "Compacted", compacted }]);
  }

  /** Resolves once every append queued so far has been written (or logged). */
  flush(): Promise<void> {
    return this.tail;
  }

  /**
   * The RAW conversation as appended, every item in order — what a chat UI
   * replays. Unchanged by checkpoints on purpose: a transcript must not be
   * truncated just because the model's context was compacted. Use
   * loadResumeState() for what the model should see.
   */
  async loadConversationHistory(): Promise<ConversationItem[]> {
    const history = await this.store.loadHistory({
      threadId: this.threadId,
    });
    return history.items
      .filter((item): item is Extract<RolloutItem, { kind: "ConversationItem" }> =>
        item.kind === "ConversationItem",
      )
      .map((item) => item.item);
  }

  /**
   * The model-visible context to resume from: the newest compaction
   * checkpoint plus everything appended after it, and the token total to seed
   * the session with. Legacy logs (no checkpoint) replay in full.
   */
  async loadResumeState(): Promise<ResumeState> {
    const history = await this.store.loadHistory({
      threadId: this.threadId,
    });
    return reconstructModelContext(history.items);
  }
}
