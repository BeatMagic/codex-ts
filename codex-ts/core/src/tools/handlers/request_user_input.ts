/**
 * mirrors codex-rs/core/src/tools/handlers/request_user_input.rs
 *
 * Suspends the current turn until the client submits a UserInputAnswer op
 * or the turn is interrupted. The resolver is stored in a map keyed by turn_id
 * and resolved when
 * CodexThread.submit({ type: "UserInputAnswer", id: turn_id, response })
 * is called.
 */

import type {
  RequestUserInputQuestion,
  RequestUserInputResponse,
} from "../../../../protocol/src/request_user_input.js";

export type PendingInputs = Map<
  string,
  (response: RequestUserInputResponse) => void
>;

export interface RequestUserInputContext {
  turnId: string;
  pendingInputs: PendingInputs;
  /** The turn's abort signal: an interrupt cancels the wait. */
  signal?: AbortSignal | undefined;
}

/**
 * Park until the answer arrives — or until the turn is interrupted, which
 * rejects with an AbortError and drops the resolver from `pendingInputs`
 * (mirrors request_user_input.rs: the wait ends "cancelled before receiving a
 * response" once the turn's pending-input senders are dropped). Without that a
 * parked turn could never be interrupted: no TurnAborted, and the leaked
 * resolver kept hasPendingInput(turnId) true for a turn that was gone. A late
 * UserInputAnswer then finds no resolver and is a no-op.
 */
export async function handleRequestUserInput(
  ctx: RequestUserInputContext,
  _questions: RequestUserInputQuestion[],
): Promise<RequestUserInputResponse> {
  return new Promise<RequestUserInputResponse>((resolve, reject) => {
    const signal = ctx.signal;
    const cancelled = (): DOMException =>
      new DOMException(
        "request_user_input was cancelled before receiving a response",
        "AbortError",
      );
    if (signal?.aborted) {
      reject(cancelled());
      return;
    }
    const resolver = (response: RequestUserInputResponse): void => {
      signal?.removeEventListener("abort", onAbort);
      resolve(response);
    };
    const onAbort = (): void => {
      // Only our own entry: a newer wait on the same turn id is not ours.
      if (ctx.pendingInputs.get(ctx.turnId) === resolver) {
        ctx.pendingInputs.delete(ctx.turnId);
      }
      reject(cancelled());
    };
    ctx.pendingInputs.set(ctx.turnId, resolver);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export function formatRequestUserInputOutput(
  response: RequestUserInputResponse,
): string {
  return JSON.stringify(response);
}
