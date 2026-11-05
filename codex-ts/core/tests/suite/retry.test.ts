import { beforeEach, describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import {
  backoff,
  computeRetryDelay,
  isRetryableError,
  isRetryableStatus,
  parseRetryAfter,
  ResponsesApiError,
} from "../../src/session/retry.js";
import {
  evAssistantMessage,
  evCompleted,
  evResponseCreated,
  makeSseResponse,
  sse,
  sseFlat,
  waitForEvent,
} from "../common/lib.js";

function okSse() {
  return makeSseResponse(
    sseFlat([
      evResponseCreated("r1"),
      evAssistantMessage("ok"),
      evCompleted("r1"),
    ]),
  );
}

// A non-2xx Response (no body) — turn.ts wraps it in a ResponsesApiError.
function errorResponse(status: number): Response {
  return {
    ok: false,
    status,
    body: null,
    headers: new Headers(),
    text: async () => `err ${status}`,
  } as unknown as Response;
}

// A 200 stream that yields one visible text delta, then drops mid-stream. Pull-
// based so the delta is delivered before the error (controller.error() would
// otherwise discard a synchronously-enqueued chunk).
function streamThenError(): Response {
  const deltaSse = sse([
    evResponseCreated("r1"),
    { type: "response.output_text.delta", delta: "partial" },
  ]);
  let stage = 0;
  return {
    ok: true,
    status: 200,
    headers: new Headers(),
    body: new ReadableStream({
      pull(controller) {
        if (stage === 0) {
          controller.enqueue(new TextEncoder().encode(deltaSse));
          stage = 1;
        } else {
          controller.error(new Error("stream dropped"));
        }
      },
    }),
    text: async () => deltaSse,
  } as unknown as Response;
}

describe("retry helpers", () => {
  it("backoff grows exponentially within the jitter band", () => {
    expect(backoff(1)).toBeGreaterThanOrEqual(180);
    expect(backoff(1)).toBeLessThanOrEqual(220);
    expect(backoff(2)).toBeGreaterThanOrEqual(360);
    expect(backoff(2)).toBeLessThanOrEqual(440);
    expect(backoff(3)).toBeGreaterThanOrEqual(720);
    expect(backoff(3)).toBeLessThanOrEqual(880);
  });

  it("classifies retryable HTTP statuses", () => {
    for (const status of [408, 409, 429, 500, 502, 503, 504]) {
      expect(isRetryableStatus(status)).toBe(true);
    }
    for (const status of [200, 400, 401, 403, 404, 422]) {
      expect(isRetryableStatus(status)).toBe(false);
    }
  });

  it("classifies retryable errors", () => {
    expect(isRetryableError(new ResponsesApiError(502, "x"))).toBe(true);
    expect(isRetryableError(new ResponsesApiError(400, "x"))).toBe(false);
    expect(isRetryableError(new TypeError("Failed to fetch"))).toBe(true);
    expect(isRetryableError(new DOMException("Aborted", "AbortError"))).toBe(
      false,
    );
  });

  it("parses Retry-After seconds and ignores when absent", () => {
    expect(parseRetryAfter(new Headers({ "retry-after": "2" }))).toBe(2000);
    expect(parseRetryAfter(new Headers())).toBeUndefined();
  });

  it("computeRetryDelay honors Retry-After, else backs off", () => {
    // What is LEFT of the advice: it counts down from when the error was built.
    const delay = computeRetryDelay(new ResponsesApiError(429, "x", 1500), 1);
    expect(delay).toBeLessThanOrEqual(1500);
    expect(delay).toBeGreaterThan(1400);
    const d = computeRetryDelay(new TypeError("net"), 1);
    expect(d).toBeGreaterThanOrEqual(180);
    expect(d).toBeLessThanOrEqual(220);
  });
});

describe("turn retries", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("retries a transient network error then succeeds", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValue(okSse());

    const codex = new CodexThread({
      apiKey: "k",
      baseUrl: "https://example.test/v1",
      fetch: fetchMock,
      model: "m",
      maxRetries: 3,
    });
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("retries a 502 then succeeds", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(errorResponse(502))
      .mockResolvedValue(okSse());

    const codex = new CodexThread({
      apiKey: "k",
      baseUrl: "https://example.test/v1",
      fetch: fetchMock,
      model: "m",
      maxRetries: 3,
    });
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not retry a non-retryable 400", async () => {
    const fetchMock = vi.fn().mockResolvedValue(errorResponse(400));

    const codex = new CodexThread({
      apiKey: "k",
      baseUrl: "https://example.test/v1",
      fetch: fetchMock,
      model: "m",
      maxRetries: 3,
    });
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    await waitForEvent(codex, (m) => m.type === "Error");

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("gives up after maxRetries and surfaces the error", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("down"));

    const codex = new CodexThread({
      apiKey: "k",
      baseUrl: "https://example.test/v1",
      fetch: fetchMock,
      model: "m",
      maxRetries: 2,
    });
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    await waitForEvent(codex, (m) => m.type === "Error");

    // initial attempt + 2 retries
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("does not retry once visible output has streamed", async () => {
    const fetchMock = vi.fn().mockResolvedValue(streamThenError());

    const codex = new CodexThread({
      apiKey: "k",
      baseUrl: "https://example.test/v1",
      fetch: fetchMock,
      model: "m",
      maxRetries: 5,
    });
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    await waitForEvent(codex, (m) => m.type === "Error");

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("disables retries when maxRetries is 0", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("down"));

    const codex = new CodexThread({
      apiKey: "k",
      baseUrl: "https://example.test/v1",
      fetch: fetchMock,
      model: "m",
      maxRetries: 0,
    });
    await codex.submit({ items: [{ text: "hi", type: "text" }], type: "UserInput" });
    await waitForEvent(codex, (m) => m.type === "Error");

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
