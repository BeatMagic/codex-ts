/**
 * Retry policy, aligned with codex-rs after the sync point:
 *   - classification: `slow_down` is a retryable rate limit, spend-limit codes
 *     are terminal quota errors, an HTTP 429 is terminal on a quota signal and
 *     otherwise retried only with server advice (#45602, #44492, #49441);
 *   - `model_not_found` is terminal (#56445);
 *   - retry advice: Retry-After in `response.error.headers` wins over the
 *     message hint, counts down from when it arrived, and is capped
 *     (#50418, #47641);
 *   - a `content_filter` retry carries recovery guidance (#49119, #49130);
 *   - opt-in retry after visible output, with reset events for the host.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { runInlineAutoCompactTask } from "../../src/compact.js";
import { CodexThread } from "../../src/codex_thread.js";
import {
  classifyHttpFailure,
  classifyStreamFailure,
  codexErrorInfoFor,
  computeRetryDelay,
  CONTENT_FILTER_GUIDANCE,
  isRetryableError,
  MAX_RETRY_AFTER_MS,
  ResponsesApiError,
} from "../../src/session/retry.js";
import { runTurn } from "../../src/session/turn.js";
import type { TurnConfig } from "../../src/session/turn.js";
import { ToolRouter } from "../../src/tools/router.js";
import type { CustomTool } from "../../src/tools/router.js";
import { GoalToolExecutor } from "../../../ext/goal/src/tool.js";
import type { EventMsg } from "../../../protocol/src/protocol.js";
import type { ConversationItem } from "../../../thread-store/src/types.js";
import {
  evAssistantMessage,
  evCompleted,
  evFunctionCall,
  evResponseCreated,
  sse,
  sseFlat,
  waitForEventMatch,
} from "../common/lib.js";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return {
    ok: true,
    status: 200,
    headers: new Headers(),
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
    text: async () => chunks.join(""),
  } as unknown as Response;
}

const okResponse = (text = "ok"): Response =>
  sseResponse([
    sseFlat([evResponseCreated("ok"), evAssistantMessage(text), evCompleted("ok")]),
  ]);

function httpError(
  status: number,
  body: object | string,
  headers: Record<string, string> = {},
): Response {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return {
    ok: false,
    status,
    headers: new Headers(headers),
    body: {} as ReadableStream<Uint8Array>,
    text: async () => text,
  } as unknown as Response;
}

const failed = (error: Record<string, unknown>): Record<string, unknown> => ({
  type: "response.failed",
  response: { id: "r", status: "failed", error },
});

const incomplete = (reason: string): Record<string, unknown> => ({
  type: "response.incomplete",
  response: { id: "r", status: "incomplete", incomplete_details: { reason } },
});

/** A message item streaming `text` as one delta (no completed). */
function evPartialMessage(id: string, text: string): object[] {
  return [
    { type: "response.output_item.added", item: { id, type: "message", content: [] } },
    { type: "response.output_text.delta", item_id: id, delta: text },
  ];
}

function spyTool(name: string): CustomTool & { calls: number } {
  const tool = {
    calls: 0,
    name,
    spec: () => ({
      type: "function" as const,
      tool: {
        name,
        description: "spy",
        parameters: { type: "object", properties: {}, additionalProperties: true },
        strict: false,
      },
    }),
    execute: async () => {
      tool.calls += 1;
      return "done";
    },
  };
  return tool as unknown as CustomTool & { calls: number };
}

function startTurn(
  fetchMock: ReturnType<typeof vi.fn>,
  extra: Partial<TurnConfig> = {},
  tools: CustomTool[] = [],
) {
  const history: ConversationItem[] = [];
  const events: EventMsg[] = [];
  const done = runTurn(
    "t1",
    [{ type: "text", text: "hi" }],
    history,
    {
      apiKey: "k",
      baseUrl: "http://x",
      model: "m",
      fetch: fetchMock as unknown as typeof fetch,
      ...extra,
    },
    new ToolRouter(new GoalToolExecutor("t1"), tools),
    new Map(),
    (msg) => events.push(msg),
  );
  return { history, events, done };
}

type WireItem = Record<string, unknown>;
const inputOf = (fetchMock: ReturnType<typeof vi.fn>, call: number): WireItem[] =>
  (
    JSON.parse((fetchMock.mock.calls[call]![1] as RequestInit).body as string) as {
      input: WireItem[];
    }
  ).input;

const GUIDANCE_TEXT = `<content_filter_guidance>\n${CONTENT_FILTER_GUIDANCE}\n</content_filter_guidance>`;
const hasGuidance = (input: WireItem[]): boolean =>
  JSON.stringify(input).includes("content_filter_guidance");

/** The display-relevant events, in order. */
const displayLog = (events: EventMsg[]): string[] =>
  events.flatMap((e) => {
    switch (e.type) {
      case "AgentMessageContentDelta":
        return [`delta ${e.event.item_id} ${e.event.delta}`];
      case "AgentMessageContentReset":
        return [`reset ${e.event.item_id}`];
      case "FunctionCallStreamStart":
        return [`call-start ${e.event.id}`];
      case "FunctionCallStreamReset":
        return [`call-reset ${e.event.id}`];
      case "AgentMessage":
        return [`message ${e.event.message}`];
      default:
        return [];
    }
  });

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

// ─── 1. Classification ────────────────────────────────────────────────────────

describe("stream failure classification", () => {
  it("treats slow_down as a retryable rate limit, honoring its message hint", () => {
    const err = classifyStreamFailure(
      failed({ code: "slow_down", message: "Slow down. Please try again in 2s." }),
    );
    expect(err.terminal).toBe(false);
    expect(isRetryableError(err)).toBe(true);
    expect(err.retryAfterMs).toBe(2000);
    expect(codexErrorInfoFor(err)).toEqual({ type: "rate_limit_exceeded" });
  });

  it("maps a streamed rate_limit_exceeded to rate_limit_exceeded", () => {
    const err = classifyStreamFailure(failed({ code: "rate_limit_exceeded", message: "x" }));
    expect(codexErrorInfoFor(err)).toEqual({ type: "rate_limit_exceeded" });
  });

  it("ends the turn on a spend-limit code, reported as a usage limit", () => {
    for (const code of [
      "credit_balance_exhausted",
      "organization_spend_limit_exceeded",
      "project_spend_limit_exceeded",
      "organization_usage_limit_exceeded",
    ]) {
      const err = classifyStreamFailure(failed({ code, message: "limit" }));
      expect(isRetryableError(err), code).toBe(false);
      expect(codexErrorInfoFor(err), code).toEqual({ type: "usage_limit_exceeded" });
    }
  });

  it("keeps server_is_overloaded terminal without advice, retryable with a Retry-After header", () => {
    const bare = classifyStreamFailure(failed({ code: "server_is_overloaded", message: "busy" }));
    expect(isRetryableError(bare)).toBe(false);
    expect(codexErrorInfoFor(bare)).toEqual({ type: "server_overloaded" });

    // A message hint is not advice for an overload — only a header is.
    const hinted = classifyStreamFailure(
      failed({ code: "server_is_overloaded", message: "try again in 3s" }),
    );
    expect(isRetryableError(hinted)).toBe(false);

    const advised = classifyStreamFailure(
      failed({ code: "server_is_overloaded", message: "busy", headers: { "Retry-After": "3" } }),
    );
    expect(isRetryableError(advised)).toBe(true);
    expect(advised.retryAfterMs).toBe(3000);
  });

  it("keeps quota and policy failures terminal even with retry advice", () => {
    for (const code of ["insufficient_quota", "cyber_policy", "model_not_found"]) {
      const err = classifyStreamFailure(
        failed({ code, message: "no", headers: { "retry-after": "1" } }),
      );
      expect(isRetryableError(err), code).toBe(false);
    }
  });
});

describe("HTTP failure classification", () => {
  it("ends the turn on a 429 carrying a quota or usage-limit signal", () => {
    for (const error of [
      { code: "insufficient_quota" },
      { code: "credit_balance_exhausted" },
      { code: "organization_spend_limit_exceeded" },
      { code: "project_spend_limit_exceeded" },
      { code: "organization_usage_limit_exceeded" },
      { type: "insufficient_quota" },
      { type: "usage_limit_reached" },
      { type: "usage_not_included" },
    ]) {
      // Even with a Retry-After: quota does not lift in seconds.
      const err = classifyHttpFailure(
        429,
        JSON.stringify({ error: { ...error, message: "limit" } }),
        new Headers({ "retry-after": "1" }),
      );
      expect(isRetryableError(err), JSON.stringify(error)).toBe(false);
      expect(codexErrorInfoFor(err), JSON.stringify(error)).toEqual({
        type: "usage_limit_exceeded",
      });
    }
  });

  it("retries any other 429 only with a Retry-After", () => {
    const body = JSON.stringify({ error: { code: "rate_limit_exceeded", message: "slow" } });
    const bare = classifyHttpFailure(429, body, new Headers());
    expect(isRetryableError(bare)).toBe(false);
    expect(codexErrorInfoFor(bare)).toEqual({
      type: "response_too_many_failed_attempts",
      http_status_code: 429,
    });
    expect(isRetryableError(classifyHttpFailure(429, "Too Many Requests", new Headers()))).toBe(
      false,
    );

    const advised = classifyHttpFailure(429, body, new Headers({ "retry-after": "2" }));
    expect(isRetryableError(advised)).toBe(true);
    expect(advised.retryAfterMs).toBe(2000);
  });

  it("retries a 503 server_is_overloaded only with a Retry-After; a 503 slow_down always", () => {
    const overloaded = JSON.stringify({ error: { code: "server_is_overloaded" } });
    expect(isRetryableError(classifyHttpFailure(503, overloaded, new Headers()))).toBe(false);
    expect(
      isRetryableError(classifyHttpFailure(503, overloaded, new Headers({ "retry-after": "1" }))),
    ).toBe(true);

    const slowDown = classifyHttpFailure(
      503,
      JSON.stringify({ error: { code: "slow_down", message: "slow" } }),
      new Headers(),
    );
    expect(isRetryableError(slowDown)).toBe(true);
    expect(codexErrorInfoFor(slowDown)).toEqual({ type: "rate_limit_exceeded" });

    // Any other 5xx is unchanged.
    expect(isRetryableError(classifyHttpFailure(503, "upstream down", new Headers()))).toBe(true);
    expect(isRetryableError(classifyHttpFailure(502, "{}", new Headers()))).toBe(true);
  });

  it("reports model_not_found from 400 / 403 / 404 bodies, never retried", () => {
    for (const status of [400, 403, 404]) {
      const err = classifyHttpFailure(
        status,
        JSON.stringify({ error: { code: "model_not_found", message: "no such model" } }),
        new Headers(),
      );
      expect(isRetryableError(err), String(status)).toBe(false);
      expect(codexErrorInfoFor(err), String(status)).toEqual({ type: "model_not_found" });
    }
  });
});

describe("classification inside a turn", () => {
  // Deliberate deviation from rs (which retries every 5xx): the ace-mv cloud
  // remaps an upstream provider's 402 to 501 so the client fails fast.
  it("does not re-send a 501", async () => {
    expect(isRetryableError(classifyHttpFailure(501, "{}", new Headers()))).toBe(false);
    const fetchMock = vi.fn().mockResolvedValue(
      httpError(501, {
        error: { code: 402, message: "This request requires more credits" },
      }),
    );
    const { done } = startTurn(fetchMock);
    const error = await done.catch((e: unknown) => e);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(codexErrorInfoFor(error)).toEqual({ type: "other" });
  });

  it("does not re-send a 429 quota error", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(httpError(429, { error: { code: "insufficient_quota", message: "x" } }));
    const { done } = startTurn(fetchMock);
    const error = await done.catch((e: unknown) => e);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(codexErrorInfoFor(error)).toEqual({ type: "usage_limit_exceeded" });
  });

  it("re-sends a 429 that says when, not one that does not", async () => {
    const advised = vi
      .fn()
      .mockResolvedValueOnce(httpError(429, "slow", { "retry-after": "0" }))
      .mockResolvedValueOnce(okResponse());
    const first = startTurn(advised);
    expect((await first.done).lastAgentMessage).toBe("ok");
    expect(advised).toHaveBeenCalledTimes(2);

    const bare = vi.fn().mockResolvedValue(httpError(429, "slow"));
    const second = startTurn(bare);
    await expect(second.done).rejects.toThrow("Responses API 429");
    expect(bare).toHaveBeenCalledTimes(1);
  });

  it("re-sends a streamed slow_down, and an overload that carries advice", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse([sse([evResponseCreated("r"), failed({ code: "slow_down", message: "try again in 5ms" })])]),
      )
      .mockResolvedValueOnce(
        sseResponse([
          sse([
            evResponseCreated("r"),
            failed({ code: "server_is_overloaded", message: "busy", headers: { "retry-after": "0" } }),
          ]),
        ]),
      )
      .mockResolvedValueOnce(okResponse());
    const { done } = startTurn(fetchMock, { maxRetries: 2 });
    expect((await done).lastAgentMessage).toBe("ok");
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("ends on model_not_found in response.failed and in error frames", async () => {
    for (const frame of [
      failed({ code: "model_not_found", message: "no such model" }),
      { type: "error", code: "model_not_found", message: "no such model" },
      { type: "error", error: { code: "model_not_found", message: "no such model" } },
    ]) {
      const fetchMock = vi
        .fn()
        .mockResolvedValue(sseResponse([sse([evResponseCreated("r"), frame])]));
      const { done } = startTurn(fetchMock);
      const error = await done.catch((e: unknown) => e);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(codexErrorInfoFor(error)).toEqual({ type: "model_not_found" });
    }
  });

  it("applies the same classification to the compaction summarizer", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(httpError(429, { error: { code: "credit_balance_exhausted" } }));
    await expect(
      runInlineAutoCompactTask(
        [{ type: "message", role: "user", content: "a" }],
        { apiKey: "k", baseUrl: "http://x", model: "m", fetch: fetchMock as unknown as typeof fetch },
      ),
    ).rejects.toThrow("credit_balance_exhausted");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

// ─── 3. Retry advice ──────────────────────────────────────────────────────────

describe("retry advice", () => {
  it("prefers a Retry-After in response.error.headers over the message hint", () => {
    const err = classifyStreamFailure(
      failed({
        code: "rate_limit_exceeded",
        message: "Please try again in 10s.",
        headers: { "Retry-After": "2" },
      }),
    );
    expect(err.retryAfterMs).toBe(2000);
  });

  it("falls back to the message hint when the header is unusable", () => {
    const err = classifyStreamFailure(
      failed({
        code: "rate_limit_exceeded",
        message: "Please try again in 10s.",
        headers: { "retry-after": "soon" },
      }),
    );
    expect(err.retryAfterMs).toBe(10_000);
  });

  it("reads header advice for unrecognised retryable failures too", () => {
    const err = classifyStreamFailure(
      failed({ code: "server_error", message: "boom", headers: { "retry-after": 4 } }),
    );
    expect(isRetryableError(err)).toBe(true);
    expect(err.retryAfterMs).toBe(4000);
  });

  it("counts the advice down from when it arrived; expired advice is zero, not backoff", () => {
    let now = 1_000;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    const err = new ResponsesApiError(503, "x", 5_000);
    now += 3_000;
    expect(computeRetryDelay(err, 1)).toBe(2_000);
    now += 10_000;
    expect(computeRetryDelay(err, 4)).toBe(0);
  });

  it(`caps the honored delay at ${MAX_RETRY_AFTER_MS}ms`, () => {
    expect(computeRetryDelay(new ResponsesApiError(429, "x", 3_600_000), 1)).toBe(
      MAX_RETRY_AFTER_MS,
    );
  });

  it("does not let an hour-long Retry-After park the turn", async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(httpError(429, "slow", { "retry-after": "3600" }))
      .mockResolvedValueOnce(okResponse());
    const { done } = startTurn(fetchMock);
    const outcome = done.then(
      (result) => result.lastAgentMessage,
      (error: unknown) => error,
    );
    await vi.advanceTimersByTimeAsync(MAX_RETRY_AFTER_MS - 1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(await outcome).toBe("ok");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

// ─── 4. Content-filter guidance ───────────────────────────────────────────────

describe("content_filter retry guidance", () => {
  it("adds the guidance to every retry of the round, never to history or the next round", async () => {
    const tool = spyTool("place_image");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(sseResponse([sse([evResponseCreated("r1"), incomplete("content_filter")])]))
      .mockRejectedValueOnce(new TypeError("network down"))
      .mockResolvedValueOnce(
        sseResponse([sseFlat([evResponseCreated("r3"), evFunctionCall("c1", "place_image", {}), evCompleted("r3")])]),
      )
      .mockResolvedValueOnce(okResponse("done"));
    const { history, done } = startTurn(fetchMock, { maxRetries: 3 }, [tool]);
    expect((await done).lastAgentMessage).toBe("done");
    expect(fetchMock).toHaveBeenCalledTimes(4);

    const blocked = inputOf(fetchMock, 0);
    expect(hasGuidance(blocked)).toBe(false);
    for (const call of [1, 2]) {
      const retried = inputOf(fetchMock, call);
      // Same request plus ONE trailing user item: the cached prefix is intact.
      expect(retried.slice(0, -1)).toEqual(blocked);
      expect(retried.at(-1)).toEqual({
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: GUIDANCE_TEXT }],
      });
    }
    // The next sampling round (after the tool ran) no longer carries it.
    expect(tool.calls).toBe(1);
    expect(hasGuidance(inputOf(fetchMock, 3))).toBe(false);
    expect(hasGuidance(history as unknown as WireItem[])).toBe(false);
  });

  it("is not added for other retryable failures", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse([sse([evResponseCreated("r1"), incomplete("clean_eof_no_terminal_event")])]),
      )
      .mockResolvedValueOnce(okResponse());
    const { done } = startTurn(fetchMock, { maxRetries: 1 });
    await done;
    expect(hasGuidance(inputOf(fetchMock, 1))).toBe(false);
  });
});

// ─── 5. Retry after visible output ────────────────────────────────────────────

describe("retry after visible output", () => {
  const cutAfter = (items: object[]): Response =>
    sseResponse([sseFlat([evResponseCreated("cut"), ...items])]);

  it("stays off by default: the turn fails, nothing is withdrawn", async () => {
    const fetchMock = vi.fn().mockResolvedValue(cutAfter(evPartialMessage("m1", "partial")));
    const { events, done } = startTurn(fetchMock);
    await expect(done).rejects.toThrow("stream closed before response.completed");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(displayLog(events)).toEqual(["delta m1 partial"]);
  });

  it("withdraws the partial text and streams the retry under the same item id", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(cutAfter(evPartialMessage("m1", "partial")))
      .mockResolvedValueOnce(
        sseResponse([
          sseFlat([
            evResponseCreated("r2"),
            evPartialMessage("m2", "full answer"),
            {
              type: "response.output_item.done",
              item: { id: "m2", type: "message", content: [{ type: "output_text", text: "full answer" }] },
            },
            evCompleted("r2"),
          ]),
        ]),
      );
    const { history, events, done } = startTurn(fetchMock, { retryAfterVisibleOutput: true });
    expect((await done).lastAgentMessage).toBe("full answer");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(displayLog(events)).toEqual([
      "delta m1 partial",
      "reset m1",
      "delta m1 full answer",
      "message full answer",
    ]);
    // The reset comes before the reconnect notice.
    const resetAt = events.findIndex((e) => e.type === "AgentMessageContentReset");
    const warningAt = events.findIndex((e) => e.type === "Warning");
    expect(resetAt).toBeLessThan(warningAt);
    expect(history.filter((i) => i.type === "message" && i.role === "assistant")).toEqual([
      { type: "message", role: "assistant", content: "full answer" },
    ]);
  });

  it("hands withdrawn ids to the retry's items in order, across an attempt that showed nothing", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(cutAfter([...evPartialMessage("m1", "a"), ...evPartialMessage("m2", "b")]))
      // Fails before any text: the withdrawn ids carry over to the next one.
      .mockResolvedValueOnce(httpError(502, "bad gateway"))
      .mockResolvedValueOnce(
        sseResponse([
          sseFlat([
            evResponseCreated("r3"),
            evPartialMessage("x1", "A"),
            evPartialMessage("x2", "B"),
            evPartialMessage("x3", "C"),
            evCompleted("r3"),
          ]),
        ]),
      );
    const { events, done } = startTurn(fetchMock, { retryAfterVisibleOutput: true, maxRetries: 2 });
    expect((await done).lastAgentMessage).toBe("ABC");
    expect(displayLog(events)).toEqual([
      "delta m1 a",
      "delta m2 b",
      "reset m1",
      "reset m2",
      "delta m1 A",
      "delta m2 B",
      "delta x3 C",
      "message ABC",
    ]);
  });

  it("also shows done-only text of the retry under the withdrawn id", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(cutAfter(evPartialMessage("m1", "partial")))
      .mockResolvedValueOnce(
        sseResponse([
          sseFlat([
            evResponseCreated("r2"),
            {
              type: "response.output_item.done",
              item: { id: "m9", type: "message", content: [{ type: "output_text", text: "final" }] },
            },
            evCompleted("r2"),
          ]),
        ]),
      );
    const { events, done } = startTurn(fetchMock, { retryAfterVisibleOutput: true });
    expect((await done).lastAgentMessage).toBe("final");
    expect(displayLog(events)).toEqual([
      "delta m1 partial",
      "reset m1",
      "delta m1 final",
      "message final",
    ]);
  });

  it("spends the same retry budget and still surfaces the last failure", async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation(async () => cutAfter(evPartialMessage("m1", "partial")));
    const { events, done } = startTurn(fetchMock, { retryAfterVisibleOutput: true, maxRetries: 1 });
    await expect(done).rejects.toThrow("stream closed before response.completed");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(displayLog(events)).toEqual(["delta m1 partial", "reset m1", "delta m1 partial"]);
  });

  it("does not retry a terminal failure after visible output", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(cutAfter([...evPartialMessage("m1", "partial"), incomplete("max_output_tokens")]));
    const { done } = startTurn(fetchMock, { retryAfterVisibleOutput: true });
    await expect(done).rejects.toThrow("max_output_tokens");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("is configurable through CodexThreadConfig", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(cutAfter(evPartialMessage("m1", "partial")))
      .mockResolvedValueOnce(okResponse("again"));
    const codex = new CodexThread({
      apiKey: "k",
      baseUrl: "http://x",
      fetch: fetchMock as unknown as typeof fetch,
      model: "m",
      retryAfterVisibleOutput: true,
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });
    const reset = await waitForEventMatch(codex, (msg) =>
      msg.type === "AgentMessageContentReset" ? msg.event : null,
    );
    expect(reset.item_id).toBe("m1");
    await waitForEventMatch(codex, (msg) => (msg.type === "TurnComplete" ? msg.event : null));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("function-call streams abandoned by a retry", () => {
  it("are reset before the retry, with or without the opt-in", async () => {
    const tool = spyTool("place_image");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse([
          sse([
            evResponseCreated("r1"),
            { type: "response.output_item.added", item: { id: "fc_1", type: "function_call", call_id: "c1", name: "place_image" } },
            { type: "response.function_call_arguments.delta", item_id: "fc_1", delta: '{"x":' },
          ]),
        ]),
      )
      .mockResolvedValueOnce(okResponse());
    const { events, done } = startTurn(fetchMock, { maxRetries: 1 }, [tool]);
    expect((await done).lastAgentMessage).toBe("ok");
    expect(displayLog(events).slice(0, 2)).toEqual(["call-start fc_1", "call-reset fc_1"]);
    expect(tool.calls).toBe(0);
  });
});
