/**
 * How a Responses stream ends, for both the sampling turn and the compaction
 * summarizer:
 *   - a stream that ends without `response.completed` is a retryable failure,
 *     never a success (mirrors turn.rs:2259-2263 CodexErr::Stream);
 *   - reading stops at `response.completed` and the body is cancelled
 *     (mirrors codex-api sse/responses.rs returning after Completed);
 *   - an idle stream times out, retryably (rs stream_idle_timeout_ms, 300s);
 *   - `response.incomplete` is terminal only for the output cap (#53451);
 *   - top-level `error` frames are terminal events (#56445);
 *   - message text delivered only in `output_item.done` is kept.
 * Plus the frames the ace-mv cloud stream guard synthesizes, verbatim in
 * shape, so its contract with this classification is pinned.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { runInlineAutoCompactTask } from "../../src/compact.js";
import { CodexThread } from "../../src/codex_thread.js";
import { codexErrorInfoFor } from "../../src/session/retry.js";
import { runTurn } from "../../src/session/turn.js";
import type { TurnConfig } from "../../src/session/turn.js";
import { SessionTokenState } from "../../src/state/token_state.js";
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

/** A 200 SSE response delivering `chunks`; `hold` keeps the connection open
 *  after them (no EOF), `onCancel` observes the body being cancelled. */
function sseResponse(
  chunks: string[],
  opts: { hold?: boolean; onCancel?: () => void } = {},
): Response {
  const encoder = new TextEncoder();
  return {
    ok: true,
    status: 200,
    headers: new Headers(),
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        if (!opts.hold) controller.close();
      },
      ...(opts.onCancel ? { cancel: opts.onCancel } : {}),
    }),
    text: async () => chunks.join(""),
  } as unknown as Response;
}

const okResponse = (text = "ok"): Response =>
  sseResponse([
    sseFlat([evResponseCreated("r2"), evAssistantMessage(text), evCompleted("r2")]),
  ]);

const incomplete = (reason: string): object => ({
  type: "response.incomplete",
  response: { id: "r1", status: "incomplete", incomplete_details: { reason } },
});

// Byte-for-byte the frames ace-mv's client-stream-guard.ts appends
// (incompleteSseFrame / failedSseFrame), leading "\n\n" included.
const cloudIncompleteFrame = (reason: string): string =>
  `\n\ndata: ${JSON.stringify({
    response: { incomplete_details: { reason } },
    type: "response.incomplete",
  })}\n\n`;
const cloudFailedFrame = (code: string, message: string): string =>
  `\n\ndata: ${JSON.stringify({
    response: { error: { code, message } },
    type: "response.failed",
  })}\n\n`;

/** A message item whose text comes ONLY in output_item.done (no deltas). */
function evDoneOnlyMessage(id: string | undefined, text: string): object[] {
  const idPart = id === undefined ? {} : { id };
  return [
    {
      type: "response.output_item.added",
      item: { ...idPart, type: "message", status: "in_progress", content: [] },
    },
    {
      type: "response.output_item.done",
      item: { ...idPart, type: "message", content: [{ type: "output_text", text }] },
    },
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

const deltasOf = (events: EventMsg[]) =>
  events.flatMap((e) => (e.type === "AgentMessageContentDelta" ? [e.event] : []));

const assistantItems = (history: ConversationItem[]) =>
  history.filter(
    (item) => item.type === "message" && item.role === "assistant",
  );

afterEach(() => {
  vi.useRealTimers();
});

// ─── Stream closed before response.completed ─────────────────────────────────

describe("a stream that ends without response.completed", () => {
  it("is retried, not treated as success — a call collected before the cut is never dispatched", async () => {
    const tool = spyTool("place_image");
    const fetchMock = vi
      .fn()
      // A complete function_call item, then a clean EOF: no completed.
      .mockResolvedValueOnce(
        sseResponse([sseFlat([evResponseCreated("r1"), evFunctionCall("c1", "place_image", {})])]),
      )
      .mockResolvedValueOnce(okResponse("after retry"));

    const { history, events, done } = startTurn(fetchMock, { maxRetries: 1 }, [tool]);
    expect((await done).lastAgentMessage).toBe("after retry");

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(tool.calls).toBe(0);
    expect(history.some((item) => item.type === "function_call")).toBe(false);
    expect(events).toContainEqual({
      type: "Warning",
      event: { message: "Reconnecting... 1/1" },
    });
  });

  it("after visible text, surfaces the error and persists nothing from the attempt", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        sse([
          evResponseCreated("r1"),
          { type: "response.output_item.added", item: { id: "m1", type: "message" } },
          { type: "response.output_text.delta", item_id: "m1", delta: "partial" },
        ]),
      ]),
    );

    const { history, events, done } = startTurn(fetchMock);
    await expect(done).rejects.toThrow(
      "stream closed before response.completed",
    );

    // Retrying would duplicate the streamed text, so no retry.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(deltasOf(events).map((d) => d.delta)).toEqual(["partial"]);
    expect(assistantItems(history)).toEqual([]);
    expect(events.some((e) => e.type === "AgentMessage")).toBe(false);
  });

  it("surfaces as a CodexThread turn error once retries are spent", async () => {
    const fetchMock = vi.fn().mockImplementation(async () =>
      sseResponse([sse([evResponseCreated("r1")])]),
    );
    const codex = new CodexThread({
      apiKey: "k",
      baseUrl: "http://x",
      fetch: fetchMock as unknown as typeof fetch,
      model: "m",
      maxRetries: 0,
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });
    const complete = await waitForEventMatch(codex, (msg) =>
      msg.type === "TurnComplete" ? msg.event : null,
    );
    expect(complete.error?.message).toContain(
      "stream closed before response.completed",
    );
  });
});

// ─── Stop reading after response.completed ────────────────────────────────────

describe("response.completed ends the stream", () => {
  it("stops reading at completed and cancels a held-open body", async () => {
    const onCancel = vi.fn();
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse(
        [
          sseFlat([
            evResponseCreated("r1"),
            evAssistantMessage("done"),
            evCompleted("r1"),
            // Anything after completed is transport noise and is never read:
            // a stray delta, an error frame and a terminal failure.
            { type: "response.output_text.delta", delta: " LATE" },
            { type: "error", code: "insufficient_quota", message: "late" },
            {
              type: "response.failed",
              response: { error: { code: "insufficient_quota", message: "late" } },
            },
          ]),
        ],
        // The upstream holds the connection open: no EOF ever arrives.
        { hold: true, onCancel },
      ),
    );

    const { events, done } = startTurn(fetchMock, { streamIdleTimeoutMs: 0 });
    expect((await done).lastAgentMessage).toBe("done");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(deltasOf(events).map((d) => d.delta)).toEqual(["done"]);
  });
});

// ─── Idle timeout ─────────────────────────────────────────────────────────────

describe("stream idle timeout", () => {
  it("abandons a silent stream after 300s by default, cancels it and retries", async () => {
    vi.useFakeTimers();
    const onCancel = vi.fn();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse([sse([evResponseCreated("r1")])], { hold: true, onCancel }),
      )
      .mockResolvedValueOnce(okResponse());

    const { events, done } = startTurn(fetchMock);
    const outcome = done.then(
      (result) => result.lastAgentMessage,
      (error: unknown) => error,
    );

    await vi.advanceTimersByTimeAsync(299_999);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1);
    expect(onCancel).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1_000); // retry backoff
    expect(await outcome).toBe("ok");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(events).toContainEqual({
      type: "Warning",
      event: { message: "Reconnecting... 1/5" },
    });
  });

  it("is disabled by a non-positive streamIdleTimeoutMs", async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockResolvedValue(sseResponse([sse([evResponseCreated("r1")])], { hold: true }));

    const { done } = startTurn(fetchMock, { streamIdleTimeoutMs: 0 });
    let settled = false;
    void done.finally(() => {
      settled = true;
    }).catch(() => undefined);

    await vi.advanceTimersByTimeAsync(60 * 60_000);
    expect(settled).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("is configurable through CodexThreadConfig", async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation(async () =>
        sseResponse([sse([evResponseCreated("r1")])], { hold: true }),
      );
    const codex = new CodexThread({
      apiKey: "k",
      baseUrl: "http://x",
      fetch: fetchMock as unknown as typeof fetch,
      model: "m",
      maxRetries: 0,
      streamIdleTimeoutMs: 50,
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });
    const error = await waitForEventMatch(codex, (msg) =>
      msg.type === "Error" ? msg.event : null,
    );
    expect(error.message).toContain("idle timeout waiting for SSE");
  });
});

// ─── response.incomplete classification ───────────────────────────────────────

describe("response.incomplete in a turn", () => {
  it("does not re-send a request cut by max_output_tokens mid tool-call", async () => {
    for (const reason of ["max_output_tokens", "max_tokens"]) {
      const fetchMock = vi.fn().mockResolvedValue(
        sseResponse([
          sse([
            evResponseCreated("r1"),
            { type: "response.output_item.added", item: { type: "function_call", call_id: "c1", name: "place_image" } },
            { type: "response.function_call_arguments.delta", item_id: "c1", delta: '{"prompt":"a very long' },
            incomplete(reason),
          ]),
        ]),
      );
      const { done } = startTurn(fetchMock);
      await expect(done, reason).rejects.toThrow(
        `Incomplete response returned, reason: ${reason}`,
      );
      expect(fetchMock, reason).toHaveBeenCalledTimes(1);
    }
  });

  it("still retries content_filter and unknown reasons", async () => {
    for (const reason of ["content_filter", "something_new"]) {
      const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(sseResponse([sse([evResponseCreated("r1"), incomplete(reason)])]))
        .mockResolvedValueOnce(okResponse());
      const { done } = startTurn(fetchMock, { maxRetries: 1 });
      expect((await done).lastAgentMessage, reason).toBe("ok");
      expect(fetchMock, reason).toHaveBeenCalledTimes(2);
    }
  });
});

// ─── The cloud stream guard's synthetic frames ────────────────────────────────

describe("frames synthesized by the ace-mv cloud stream guard", () => {
  it("retries every synthetic transport incomplete, even after a half-written frame", async () => {
    for (const reason of [
      "clean_eof_no_terminal_event",
      "upstream_idle_timeout",
      "upstream_stream_broke",
      "client_disconnected",
    ]) {
      const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(
          sseResponse([
            sse([evResponseCreated("r1")]),
            // The upstream died mid-line; the guard's leading "\n\n" seals it.
            'data: {"type":"response.output_te',
            cloudIncompleteFrame(reason),
          ]),
        )
        .mockResolvedValueOnce(okResponse());
      const { done } = startTurn(fetchMock, { maxRetries: 1 });
      expect((await done).lastAgentMessage, reason).toBe("ok");
      expect(fetchMock, reason).toHaveBeenCalledTimes(2);
    }
  });

  it("keeps the keepalive-stall failure (server_is_overloaded) terminal", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        sse([evResponseCreated("r1")]),
        ": keepalive\n\n",
        cloudFailedFrame(
          "server_is_overloaded",
          "upstream_keepalive_stall: upstream stopped producing events",
        ),
      ]),
    );
    const { done } = startTurn(fetchMock);
    const error = await done.catch((e: unknown) => e);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(codexErrorInfoFor(error)).toEqual({ type: "server_overloaded" });
  });
});

// ─── Top-level error frames ───────────────────────────────────────────────────

describe("top-level SSE error frames", () => {
  it("ends the turn on a terminal code, top-level or nested", async () => {
    const frames = [
      // OpenAI streaming `error` event shape.
      { type: "error", code: "insufficient_quota", message: "out of credits", param: null },
      // Nested shape (what rs reads; Anthropic's envelope).
      { type: "error", error: { code: "insufficient_quota", message: "out of credits" } },
    ];
    for (const frame of frames) {
      const fetchMock = vi
        .fn()
        .mockResolvedValue(sseResponse([sse([evResponseCreated("r1"), frame])]));
      const { done } = startTurn(fetchMock);
      const error = await done.catch((e: unknown) => e);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(codexErrorInfoFor(error)).toEqual({ type: "usage_limit_exceeded" });
    }
  });

  it("retries an unrecognised error frame instead of finishing the turn", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse([
          sse([
            evResponseCreated("r1"),
            { type: "error", error: { type: "overloaded_error", message: "Overloaded" } },
          ]),
        ]),
      )
      .mockResolvedValueOnce(okResponse());
    const { done } = startTurn(fetchMock, { maxRetries: 1 });
    expect((await done).lastAgentMessage).toBe("ok");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("routes context-window wording into the context-exceeded path", async () => {
    const tokenState = new SessionTokenState();
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        sse([
          evResponseCreated("r1"),
          {
            type: "error",
            error: {
              type: "invalid_request_error",
              message: "prompt is too long: 1200 tokens > 1000 maximum",
            },
          },
        ]),
      ]),
    );
    const { done } = startTurn(fetchMock, { contextWindow: 1000, tokenState });
    const error = await done.catch((e: unknown) => e);
    expect(codexErrorInfoFor(error)).toEqual({ type: "context_window_exceeded" });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    // Marked full, so the next turn compacts before sampling.
    expect(tokenState.totalTokens).toBe(1000);
  });
});

// ─── Text only in output_item.done ────────────────────────────────────────────

describe("assistant text delivered only in output_item.done", () => {
  it("is kept, persisted and emitted once for the host", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        sseFlat([evResponseCreated("r1"), evDoneOnlyMessage("m1", "hello"), evCompleted("r1")]),
      ]),
    );
    const { history, events, done } = startTurn(fetchMock);
    expect((await done).lastAgentMessage).toBe("hello");
    expect(deltasOf(events)).toEqual([
      { turn_id: "t1", item_id: "m1", delta: "hello" },
    ]);
    expect(assistantItems(history)).toEqual([
      { type: "message", role: "assistant", content: "hello" },
    ]);
  });

  it("is not double-counted when the item also streamed deltas", async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse("streamed"));
    const { events, done } = startTurn(fetchMock);
    expect((await done).lastAgentMessage).toBe("streamed");
    expect(deltasOf(events).map((d) => d.delta)).toEqual(["streamed"]);
  });

  it("is not double-counted when deltas carried no item_id and no item was announced", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        sse([
          evResponseCreated("r1"),
          { type: "response.output_text.delta", delta: "plain" },
          {
            type: "response.output_item.done",
            item: { id: "msg_real", type: "message", content: [{ type: "output_text", text: "plain" }] },
          },
          evCompleted("r1"),
        ]),
      ]),
    );
    const { done } = startTurn(fetchMock);
    expect((await done).lastAgentMessage).toBe("plain");
  });

  it("is not double-counted when deltas named their item but no item was announced", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        sse([
          evResponseCreated("r1"),
          { type: "response.output_text.delta", item_id: "msg_real", delta: "named" },
          {
            type: "response.output_item.done",
            item: { id: "msg_real", type: "message", content: [{ type: "output_text", text: "named" }] },
          },
          evCompleted("r1"),
        ]),
      ]),
    );
    const { done } = startTurn(fetchMock);
    expect((await done).lastAgentMessage).toBe("named");
  });

  it("is appended after a streamed message in the same response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        sseFlat([
          evResponseCreated("r1"),
          {
            type: "response.output_item.added",
            item: { id: "m1", type: "message", content: [] },
          },
          { type: "response.output_text.delta", item_id: "m1", delta: "one." },
          {
            type: "response.output_item.done",
            item: { id: "m1", type: "message", content: [{ type: "output_text", text: "one." }] },
          },
          evDoneOnlyMessage("m2", " two."),
          evCompleted("r1"),
        ]),
      ]),
    );
    const { events, done } = startTurn(fetchMock);
    expect((await done).lastAgentMessage).toBe("one. two.");
    expect(deltasOf(events).map((d) => [d.item_id, d.delta])).toEqual([
      ["m1", "one."],
      ["m2", " two."],
    ]);
  });

  it("counts as visible output: a later cut is surfaced, not retried", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([sseFlat([evResponseCreated("r1"), evDoneOnlyMessage("m1", "shown")])]),
    );
    const { done } = startTurn(fetchMock);
    await expect(done).rejects.toThrow("stream closed before response.completed");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

// ─── The compaction summarizer stream ─────────────────────────────────────────

describe("compaction summarizer stream", () => {
  const compactConfig = (
    fetchMock: ReturnType<typeof vi.fn>,
    extra: Partial<TurnConfig> = {},
  ): TurnConfig => ({
    apiKey: "k",
    baseUrl: "http://x",
    model: "m",
    fetch: fetchMock as unknown as typeof fetch,
    ...extra,
  });
  const history = (): ConversationItem[] => [
    { type: "message", role: "user", content: [{ type: "input_text", text: "a" }] },
    { type: "message", role: "assistant", content: "b" },
  ];
  const summaryResponse = (text = "the summary"): Response =>
    sseResponse([
      sseFlat([evResponseCreated("c"), evAssistantMessage(text), evCompleted("c")]),
    ]);

  it("takes a summary delivered only in output_item.done", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        sseFlat([evResponseCreated("c"), evDoneOnlyMessage("s1", "done-only summary"), evCompleted("c")]),
      ]),
    );
    expect(await runInlineAutoCompactTask(history(), compactConfig(fetchMock))).toBe(
      "done-only summary",
    );
  });

  it("cancels a held-open body once completed arrives", async () => {
    const onCancel = vi.fn();
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse(
        [sseFlat([evResponseCreated("c"), evAssistantMessage("the summary"), evCompleted("c")])],
        { hold: true, onCancel },
      ),
    );
    expect(await runInlineAutoCompactTask(history(), compactConfig(fetchMock))).toBe(
      "the summary",
    );
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("retries a summarizer stream that closed before completed", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse([sse([evResponseCreated("c"), { type: "response.output_text.delta", delta: "half a" }])]),
      )
      .mockResolvedValueOnce(summaryResponse());
    expect(
      await runInlineAutoCompactTask(history(), compactConfig(fetchMock, { maxRetries: 1 })),
    ).toBe("the summary");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("classifies error frames: unknown retried, terminal codes not", async () => {
    const retried = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse([sse([evResponseCreated("c"), { type: "error", code: "server_error", message: "boom" }])]),
      )
      .mockResolvedValueOnce(summaryResponse());
    expect(
      await runInlineAutoCompactTask(history(), compactConfig(retried, { maxRetries: 1 })),
    ).toBe("the summary");
    expect(retried).toHaveBeenCalledTimes(2);

    const terminal = vi.fn().mockResolvedValue(
      sseResponse([
        sse([evResponseCreated("c"), { type: "error", error: { code: "cyber_policy", message: "flagged" } }]),
      ]),
    );
    await expect(
      runInlineAutoCompactTask(history(), compactConfig(terminal, { maxRetries: 3 })),
    ).rejects.toThrow("cyber_policy");
    expect(terminal).toHaveBeenCalledTimes(1);
  });

  it("times out an idle summarizer stream and retries", async () => {
    vi.useFakeTimers();
    const onCancel = vi.fn();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse([sse([evResponseCreated("c")])], { hold: true, onCancel }),
      )
      .mockResolvedValueOnce(summaryResponse());
    const outcome = runInlineAutoCompactTask(
      history(),
      compactConfig(fetchMock, { maxRetries: 1, streamIdleTimeoutMs: 5_000 }),
    ).then(
      (summary) => summary,
      (error: unknown) => error,
    );

    await vi.advanceTimersByTimeAsync(4_999);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(onCancel).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1_000); // retry backoff
    expect(await outcome).toBe("the summary");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
