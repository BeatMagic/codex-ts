/**
 * mirrors codex-api/src/sse/responses.rs process_responses_event's terminal arms
 * (`response.failed` / `response.incomplete`) + CodexErr::is_retryable
 * (protocol/src/error.rs:358) + the turn lifecycle in tasks/mod.rs:795-813.
 *
 * Covers what an SSE-level failure does to a turn: which payload codes are
 * terminal, which are retried (and with what delay), and the fact that a failed
 * turn still completes its lifecycle with the error attached.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import {
  classifyStreamFailure,
  isRetryableError,
  parseRateLimitRetryAfterMs,
} from "../../src/session/retry.js";
import type { ErrorEvent } from "../../../protocol/src/protocol.js";
import {
  evAssistantMessage,
  evCompleted,
  evResponseCreated,
  makeSseResponse,
  sseFlat,
  waitForEvent,
  waitForEventMatch,
} from "../common/lib.js";

function okSse(): Response {
  return makeSseResponse(
    sseFlat([evResponseCreated("r1"), evAssistantMessage("ok"), evCompleted("r1")]),
  );
}

function failedSse(error: Record<string, unknown>): Response {
  return makeSseResponse(
    sseFlat([
      evResponseCreated("r1"),
      { type: "response.failed", response: { id: "r1", status: "failed", error } },
    ]),
  );
}

function incompleteSse(reason: string): Response {
  return makeSseResponse(
    sseFlat([
      evResponseCreated("r1"),
      {
        type: "response.incomplete",
        response: {
          id: "r1",
          status: "incomplete",
          incomplete_details: { reason },
        },
      },
    ]),
  );
}

function thread(fetchMock: typeof fetch, maxRetries = 3): CodexThread {
  return new CodexThread({
    apiKey: "k",
    baseUrl: "https://example.test/v1",
    fetch: fetchMock,
    model: "m",
    maxRetries,
  });
}

describe("classifyStreamFailure", () => {
  it("marks quota / policy / overload codes terminal", () => {
    // Exactly the codes rs's `response.failed` arm maps to non-retryable
    // CodexErr variants. `usage_limit_reached` is NOT here — rs classifies that
    // one on the HTTP 429 path (api_bridge.rs:96), not from an SSE payload.
    for (const code of [
      "insufficient_quota",
      "usage_not_included",
      "cyber_policy",
      "invalid_prompt",
      "bio_policy",
      "server_is_overloaded",
      "slow_down",
    ]) {
      const err = classifyStreamFailure({
        type: "response.failed",
        response: { error: { code, message: "nope" } },
      });
      expect(err.terminal, code).toBe(true);
      expect(isRetryableError(err), code).toBe(false);
    }
  });

  it("keeps unrecognised failures retryable (rs defaults to ApiError::Retryable)", () => {
    const err = classifyStreamFailure({
      type: "response.failed",
      response: { error: { code: "something_new", message: "boom" } },
    });
    expect(err.terminal).toBe(false);
    expect(isRetryableError(err)).toBe(true);
  });

  it("honors a rate-limit 'try again in Xs' hint", () => {
    const err = classifyStreamFailure({
      type: "response.failed",
      response: {
        error: {
          code: "rate_limit_exceeded",
          message: "Rate limit reached. Please try again in 1.5s. See docs.",
        },
      },
    });
    expect(err.retryAfterMs).toBe(1500);
    expect(isRetryableError(err)).toBe(true);
  });

  it("parses the hint only for rate_limit_exceeded", () => {
    expect(
      parseRateLimitRetryAfterMs("rate_limit_exceeded", "try again in 250ms"),
    ).toBe(250);
    expect(
      parseRateLimitRetryAfterMs("rate_limit_exceeded", "try again in 3 seconds"),
    ).toBe(3000);
    expect(
      parseRateLimitRetryAfterMs("server_error", "try again in 3s"),
    ).toBeUndefined();
  });

  // mirrors #53451 (bc9cad3473): an output-cap incomplete is terminal — the
  // identical re-send would hit the cap again. (Was retryable before.)
  it("turns an output-cap response.incomplete into a terminal error carrying the reason", () => {
    for (const reason of ["max_output_tokens", "max_tokens"]) {
      const err = classifyStreamFailure({
        type: "response.incomplete",
        response: { incomplete_details: { reason } },
      });
      expect(err.message).toContain(
        `Incomplete response returned, reason: ${reason}`,
      );
      expect(err.terminal, reason).toBe(true);
      expect(isRetryableError(err), reason).toBe(false);
    }
  });

  // The ace-mv cloud stream guard seals cut streams with these synthetic
  // reasons and relies on them being retried; content_filter stays retryable
  // like rs, and an unknown/missing reason stays retryable for compatibility.
  it("keeps every other response.incomplete retryable", () => {
    for (const reason of [
      "clean_eof_no_terminal_event",
      "upstream_idle_timeout",
      "upstream_stream_broke",
      "client_disconnected",
      "content_filter",
      "interrupted",
      "connection closed",
      "something_new",
    ]) {
      const err = classifyStreamFailure({
        type: "response.incomplete",
        response: { incomplete_details: { reason } },
      });
      expect(err.message).toContain(reason);
      expect(isRetryableError(err), reason).toBe(true);
    }
    // No incomplete_details at all.
    expect(
      isRetryableError(classifyStreamFailure({ type: "response.incomplete" })),
    ).toBe(true);
  });
});

describe("turn behaviour on stream failures", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("does not retry a terminal response.failed", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      failedSse({ code: "insufficient_quota", message: "out of credits" }),
    );

    const codex = thread(fetchMock as unknown as typeof fetch);
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    const error = await waitForEventMatch(codex, (msg) =>
      msg.type === "Error" ? msg.event : null,
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(error.codex_error_info).toEqual({ type: "usage_limit_exceeded" });
  });

  it("retries a retryable response.failed then succeeds", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        failedSse({ code: "rate_limit_exceeded", message: "try again in 10ms" }),
      )
      .mockResolvedValue(okSse());

    const codex = thread(fetchMock as unknown as typeof fetch);
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("retries response.incomplete then succeeds", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(incompleteSse("content_filter"))
      .mockResolvedValue(okSse());

    const codex = thread(fetchMock as unknown as typeof fetch);
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  // mirrors tasks/mod.rs:795-813 — a failed turn emits Error AND completes the
  // turn lifecycle with the terminal error attached, so hosts awaiting
  // TurnComplete are never left hanging.
  it("still completes the turn, carrying the error", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      failedSse({ code: "cyber_policy", message: "flagged" }),
    );

    const codex = thread(fetchMock as unknown as typeof fetch);
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    const complete = await waitForEventMatch(codex, (msg) =>
      msg.type === "TurnComplete" ? msg.event : null,
    );

    const error = complete.error as ErrorEvent;
    expect(error.codex_error_info).toEqual({ type: "cyber_policy" });
    expect(error.message).toContain("cyber_policy");
    expect(complete.duration_ms).toBeGreaterThanOrEqual(0);
    expect(complete.started_at).toBeGreaterThan(0);
  });

  // mirrors tasks/mod.rs:785-794: an aborted turn ends with TurnAborted, not
  // TurnComplete.
  it("ends an interrupted turn with TurnAborted", async () => {
    // A never-resolving request so the turn is still in flight when interrupted.
    const fetchMock = vi.fn().mockImplementation(
      () => new Promise<Response>(() => {}),
    );

    const codex = thread(fetchMock as unknown as typeof fetch);
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    await codex.submit({ type: "Interrupt" });

    const aborted = await waitForEventMatch(codex, (msg) =>
      msg.type === "TurnAborted" ? msg.event : null,
    );
    expect(aborted.reason).toBe("interrupted");
    expect(aborted.started_at).toBeGreaterThan(0);
    expect(aborted.duration_ms).toBeGreaterThanOrEqual(0);
  });
});
