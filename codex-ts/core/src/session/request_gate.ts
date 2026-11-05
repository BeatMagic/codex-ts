/**
 * One model request in flight per thread.
 *
 * codex-rs never has two: a thread runs one active turn, and new input is
 * steered into it (session/handlers.rs → steer_input) rather than starting a
 * second one. codex-ts runs turns unawaited (`void runTurn`), so two turns on
 * one thread can sample at the same time. Besides racing on the shared
 * history, every such overlap pays a full prompt-cache write twice: a cache
 * entry only becomes readable once the first response starts streaming, so
 * the second request cannot reuse what the first is still writing.
 *
 * The gate serializes REQUESTS, not turns. It is held for one request's fetch
 * and stream consumption (sampling or summarizer) and released before any tool
 * runs, so a turn parked in a long tool or in request_user_input never blocks
 * the request of the turn that answers it.
 */

export class RequestGate {
  private held = false;
  /** FIFO of queued acquirers; each entry hands the gate over when called. */
  private readonly waiters: (() => void)[] = [];

  /**
   * Resolves with a release function once this caller holds the gate. Waiting
   * is abortable: an aborted waiter leaves the queue and rejects with an
   * AbortError, so an interrupted turn never strands the requests behind it.
   * The release function is idempotent.
   */
  acquire(signal?: AbortSignal | undefined): Promise<() => void> {
    if (signal?.aborted) {
      return Promise.reject(interrupted());
    }
    if (!this.held) {
      this.held = true;
      return Promise.resolve(this.releaser());
    }
    return new Promise<() => void>((resolve, reject) => {
      const grant = (): void => {
        signal?.removeEventListener("abort", onAbort);
        resolve(this.releaser());
      };
      const onAbort = (): void => {
        const index = this.waiters.indexOf(grant);
        if (index !== -1) this.waiters.splice(index, 1);
        reject(interrupted());
      };
      this.waiters.push(grant);
      signal?.addEventListener("abort", onAbort, { once: true });
    });
  }

  private releaser(): () => void {
    let released = false;
    return () => {
      if (released) return;
      released = true;
      // Hand over directly (the gate stays held) so no caller arriving in
      // between can jump the queue.
      const next = this.waiters.shift();
      if (next) next();
      else this.held = false;
    };
  }
}

/**
 * Acquire the thread's gate, or run ungated when there is none (a one-shot
 * runTurn / runInlineAutoCompactTask call: its requests are sequential anyway).
 */
export function acquireRequestGate(
  gate: RequestGate | undefined,
  signal: AbortSignal | undefined,
): Promise<() => void> {
  return gate ? gate.acquire(signal) : Promise.resolve(() => undefined);
}

function interrupted(): DOMException {
  return new DOMException("Request interrupted", "AbortError");
}
