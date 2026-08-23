/**
 * mirrors codex-rs/core/src/state/auto_compact_window.rs
 *
 * Tracks the token baseline for the current compaction window so that
 * BodyAfterPrefix mode can measure context growth since the last compaction
 * rather than total context size.
 */

/** mirrors AutoCompactWindowIds (auto_compact_window.rs:5).
 *  rs mints Uuid::now_v7(); codex-ts uses crypto.randomUUID() when available
 *  (browser + Node ≥ 19) and falls back to a counter-based id otherwise. */
export interface AutoCompactWindowIds {
  firstWindowId: string;
  previousWindowId: string | null;
  windowId: string;
}

let windowIdCounter = 0;

function newWindowId(): string {
  const cryptoObj: { randomUUID?: () => string } | undefined = (
    globalThis as { crypto?: { randomUUID?: () => string } }
  ).crypto;
  if (typeof cryptoObj?.randomUUID === "function") {
    return cryptoObj.randomUUID();
  }
  windowIdCounter += 1;
  return `window-${Date.now().toString(36)}-${windowIdCounter}`;
}

/** mirrors AutoCompactWindowIds::new_initial (auto_compact_window.rs:12) */
export function newInitialAutoCompactWindowIds(): AutoCompactWindowIds {
  const windowId = newWindowId();
  return { firstWindowId: windowId, previousWindowId: null, windowId };
}

/** mirrors AutoCompactWindowSnapshot (auto_compact_window.rs:22) — rs dropped the
 *  `ordinal` field when window identity moved to windowNumber + ids. */
export interface AutoCompactWindowSnapshot {
  prefillInputTokens: number | null;
}

export class AutoCompactWindow {
  /** mirrors window_number (auto_compact_window.rs:35) — starts at 0, not 1, and
   *  advances once per compaction. */
  private windowNumber = 0;
  private ids: AutoCompactWindowIds = newInitialAutoCompactWindowIds();
  private newContextWindowRequested = false;
  /**
   * Absolute input-token baseline for the current window.
   * Set from the first server-observed usage after a new window opens;
   * falls back to an estimated value when the server hasn't responded yet.
   */
  private prefillInputTokens: number | null = null;
  private prefillSource: "server" | "estimated" | null = null;
  private tokenBudgetReminderDelivered = false;
  private autoCompactFallbackDelivered = false;
  /**
   * Generation counter for landed compactions. No equivalent in codex-rs, whose
   * compaction is not concurrent with itself; see `currentCompactionEpoch`.
   */
  private compactionEpoch = 0;

  clearPrefill(): void {
    this.prefillInputTokens = null;
    this.prefillSource = null;
  }

  /** mirrors window_number() (auto_compact_window.rs:64) */
  currentWindowNumber(): number {
    return this.windowNumber;
  }

  /** mirrors ids() (auto_compact_window.rs:68) */
  currentIds(): AutoCompactWindowIds {
    return { ...this.ids };
  }

  /** mirrors restore() (auto_compact_window.rs:72) — used when a resumed thread
   *  replays its persisted compaction checkpoints. */
  restore(windowNumber: number, ids: AutoCompactWindowIds): void {
    this.windowNumber = windowNumber;
    this.ids = { ...ids };
  }

  /**
   * Open the next compaction window: bump the number, rotate the ids, and reset
   * the per-window one-shot flags. mirrors advance() (auto_compact_window.rs:77).
   *
   * Note rs's `advance()` deliberately does NOT clear the prefill baseline —
   * `Session::start_new_context_window` / `replace_history` do that separately
   * (state/session.rs:122,200). Callers that rewrite history must therefore call
   * `startNewContextWindow()` (or `clearPrefill()`) as well; see
   * runCompactAndRecompute in session/turn.ts.
   */
  advance(): { windowNumber: number; ids: AutoCompactWindowIds } {
    this.windowNumber += 1;
    this.ids = {
      firstWindowId: this.ids.firstWindowId,
      previousWindowId: this.ids.windowId,
      windowId: newWindowId(),
    };
    this.newContextWindowRequested = false;
    this.tokenBudgetReminderDelivered = false;
    this.autoCompactFallbackDelivered = false;
    return { windowNumber: this.windowNumber, ids: this.currentIds() };
  }

  /** mirrors Session::start_new_context_window (state/session.rs:199): advance
   *  the window AND drop the stale prefill baseline in one step. */
  startNewContextWindow(): { windowNumber: number; ids: AutoCompactWindowIds } {
    const advanced = this.advance();
    this.clearPrefill();
    return advanced;
  }

  /** mirrors claim_token_budget_reminder (auto_compact_window.rs:88) — true only
   *  the first time it is called within a window. */
  claimTokenBudgetReminder(): boolean {
    const claimed = !this.tokenBudgetReminderDelivered;
    this.tokenBudgetReminderDelivered = true;
    return claimed;
  }

  /** mirrors claim_auto_compact_fallback (auto_compact_window.rs:92) */
  claimAutoCompactFallback(): boolean {
    const claimed = !this.autoCompactFallbackDelivered;
    this.autoCompactFallbackDelivered = true;
    return claimed;
  }

  /** mirrors request_new_context_window (auto_compact_window.rs:96) — codex-rs
   *  sets this from the `new_context` tool so the next mid-turn checkpoint rolls
   *  the window over even when the token limit was not reached. codex-ts has no
   *  such built-in tool, so a host can drive it from a custom tool. */
  requestNewContextWindow(): void {
    this.newContextWindowRequested = true;
  }

  /** mirrors take_new_context_window_request (auto_compact_window.rs:100) */
  takeNewContextWindowRequest(): boolean {
    const requested = this.newContextWindowRequested;
    this.newContextWindowRequested = false;
    return requested;
  }

  /**
   * Records the request-input side of the first server usage sample.
   * Server-observed values take priority and are never overwritten.
   * mirrors: ensure_server_observed_prefill_from_usage
   */
  ensureServerObservedPrefill(inputTokens: number): void {
    if (this.prefillSource === "server") return;
    this.prefillInputTokens = Math.max(0, inputTokens);
    this.prefillSource = "server";
  }

  /**
   * Set an estimated prefill when actual usage isn't available yet.
   * mirrors: set_estimated_prefill
   */
  setEstimatedPrefill(tokens: number): void {
    if (this.prefillSource === "server") return;
    this.prefillInputTokens = Math.max(0, tokens);
    this.prefillSource = "estimated";
  }

  snapshot(): AutoCompactWindowSnapshot {
    return { prefillInputTokens: this.prefillInputTokens };
  }

  /**
   * How many compactions have LANDED on this thread.
   *
   * No equivalent in codex-rs: there, a compaction cannot overlap another one.
   * codex-ts runs turns concurrently (`void runTurn` in codex_thread.ts) over a
   * single shared history, and its summarizer request is not abortable, so two
   * compactions being in flight at once is reachable from several directions.
   *
   * A compaction is a decision made on a reading of history + token state. If
   * another one lands while this summarizer is in flight, that reading is void:
   * applying the late summary would rewrite a history that has already been
   * replaced, discarding everything produced in between (assistant messages and
   * tool calls are not user messages, so collectUserMessages cannot recover
   * them). Comparing the epoch at entry against the epoch at the moment of
   * mutation makes the late compaction a no-op instead of a rewrite.
   *
   * Observed without it (2026-08-17): three ContextCompacted for one legitimate
   * trigger, the last two at 34k/600k where neither trigger line is near.
   */
  currentCompactionEpoch(): number {
    return this.compactionEpoch;
  }

  /** Mark a compaction as landed, making every in-flight one stale. */
  bumpCompactionEpoch(): number {
    this.compactionEpoch += 1;
    return this.compactionEpoch;
  }

  /**
   * Tokens added since the start of this window (BodyAfterPrefix mode).
   * Falls back to total when no baseline is recorded yet.
   */
  bodyAfterPrefix(totalInputTokens: number): number {
    const baseline = this.prefillInputTokens ?? totalInputTokens;
    return Math.max(0, totalInputTokens - baseline);
  }
}
