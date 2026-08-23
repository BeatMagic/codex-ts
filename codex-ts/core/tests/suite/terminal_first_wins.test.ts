/**
 * The FIRST terminal event of a stream wins; anything after it is transport
 * noise, not turn state.
 *
 * Production, 2026-08-04: an upstream delivered the full stream and then held
 * the connection open. A gateway watchdog sealed the hold by appending a
 * synthetic `response.incomplete` AFTER the `response.completed` frame. Treating
 * that as terminal classified a COMPLETE turn as a retryable 503, so the client
 * re-sent the byte-identical request — a loop that burned ~27k output tokens per
 * lap and never advanced the turn.
 */

import { describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import {
  evAssistantMessage,
  evCompleted,
  evResponseCreated,
  makeSseResponse,
  sseFlat,
  waitForEvent,
} from "../common/lib.js";

describe("a terminal event arriving after response.completed", () => {
  it("does not un-complete the turn, and does not trigger a retry", async () => {
    const fetchMock = vi.fn(async () =>
      makeSseResponse(
        sseFlat([
          evResponseCreated("r1"),
          evAssistantMessage("done"),
          evCompleted("r1"),
          // The watchdog's late seal, verbatim in shape.
          {
            type: "response.incomplete",
            response: {
              id: "r1",
              status: "incomplete",
              incomplete_details: { reason: "connection closed" },
            },
          },
        ]),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({ apiKey: "test", model: "gpt-4o" });
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "hi" }],
    });

    const done = await waitForEvent(
      codex,
      (m) => m.type === "TurnComplete" || m.type === "Error",
    );
    expect(done.type).toBe("TurnComplete");
    // The money assertion: exactly one upstream request. A late terminal that
    // re-opens the turn shows up here as a second (identical) call.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    vi.unstubAllGlobals();
  });

  it("NON-INVERSION: an incomplete with NO preceding completed is still retryable", async () => {
    // First attempt: incomplete and nothing else — the turn did NOT finish, so
    // this must still be treated as a retryable stream failure. Second attempt
    // succeeds, which keeps the test off the backoff ladder.
    let attempt = 0;
    const fetchMock = vi.fn(async () => {
      attempt += 1;
      if (attempt === 1) {
        return makeSseResponse(
          sseFlat([
            evResponseCreated("r1"),
            {
              type: "response.incomplete",
              response: {
                id: "r1",
                status: "incomplete",
                incomplete_details: { reason: "connection closed" },
              },
            },
          ]),
        );
      }
      return makeSseResponse(
        sseFlat([
          evResponseCreated("r2"),
          evAssistantMessage("done"),
          evCompleted("r2"),
        ]),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({ apiKey: "test", model: "gpt-4o" });
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "hi" }],
    });
    await waitForEvent(codex, (m) => m.type === "TurnComplete", 4000);

    // Two calls = the first incomplete was NOT swallowed as a completion.
    expect(fetchMock).toHaveBeenCalledTimes(2);
    vi.unstubAllGlobals();
  });
});
