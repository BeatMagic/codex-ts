/**
 * Pre-turn compaction runs BEFORE the new user input is recorded.
 *
 * mirrors codex-rs/core/src/session/turn.rs:221-266 and commit ee93abb690
 * (#44487, "Preserve incoming prompts when pre-turn compaction fails"): the
 * summarizer sees the history up to the new ask, the ask lands after the
 * summary, and a failed compaction still records it.
 *
 * Recording first — what codex-ts did before — summarized the ask, re-collected
 * it as one more retained user message (truncated past 20k tokens), and left
 * the model's latest instruction sitting BEFORE a handoff summary.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import { SUMMARIZATION_PROMPT, SUMMARY_PREFIX } from "../../src/compact.js";
import { runTurn } from "../../src/session/turn.js";
import { ToolRouter } from "../../src/tools/router.js";
import { GoalToolExecutor } from "../../../ext/goal/src/tool.js";
import { SessionTokenState } from "../../src/state/token_state.js";
import { InMemoryIoBackend } from "../../../thread-store/src/io_backend.js";
import { LocalThreadStore } from "../../../thread-store/src/local_thread_store.js";
import type {
  ConversationItem,
  RolloutItem,
} from "../../../thread-store/src/types.js";
import {
  evAssistantMessage,
  evCompleted,
  evResponseCreated,
  makeSseResponse,
  sseFlat,
  waitForEvent,
} from "../common/lib.js";

type Body = { input: Record<string, unknown>[] };

function bodyOf(fetchMock: ReturnType<typeof vi.fn>, index: number): Body {
  const init = fetchMock.mock.calls[index]![1] as RequestInit;
  return JSON.parse(init.body as string) as Body;
}

function errorResponse(status: number, body: string): Response {
  return {
    ok: false,
    status,
    headers: new Headers(),
    body: new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(body));
        controller.close();
      },
    }),
    text: async () => body,
  } as unknown as Response;
}

const PROMPT_TOO_LONG = JSON.stringify({
  error: { message: "prompt is too long: 1039913 tokens > 1000000 maximum", code: 400 },
});

const summaryResponse = () =>
  makeSseResponse(
    sseFlat([
      evResponseCreated("compact"),
      evAssistantMessage("summary text"),
      evCompleted("compact"),
    ]),
  );

const replyResponse = () =>
  makeSseResponse(
    sseFlat([evResponseCreated("r"), evAssistantMessage("on it"), evCompleted("r")]),
  );

const user = (text: string): ConversationItem => ({
  type: "message",
  role: "user",
  content: [{ type: "input_text", text }],
});

function liveHistoryOf(thread: CodexThread): ConversationItem[] {
  return (thread as unknown as { history: ConversationItem[] }).history;
}

async function storedItems(io: InMemoryIoBackend, threadId: string): Promise<RolloutItem[]> {
  return (await new LocalThreadStore(io).loadHistory({ threadId })).items;
}

/**
 * Turn 1 is rejected as too long (marks the session FULL), so turn 2 must
 * compact before it samples. Returns the thread with turn 1 settled; `fetchMock`
 * has its first response consumed.
 */
async function threadMarkedFull(fetchMock: ReturnType<typeof vi.fn>, io: InMemoryIoBackend) {
  fetchMock.mockResolvedValueOnce(errorResponse(400, PROMPT_TOO_LONG));
  vi.stubGlobal("fetch", fetchMock);
  const thread = new CodexThread({
    apiKey: "t",
    model: "m",
    contextWindow: 1_000,
    maxRetries: 0,
    ioBackend: io,
  });
  await thread.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });
  await waitForEvent(thread, (msg) => msg.type === "TurnComplete");
  return thread;
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

describe("pre-turn compaction ordering", () => {
  it("summarizes the history WITHOUT the new ask, which then follows the summary", async () => {
    const io = new InMemoryIoBackend();
    const fetchMock = vi.fn();
    const thread = await threadMarkedFull(fetchMock, io);
    fetchMock.mockResolvedValueOnce(summaryResponse()).mockResolvedValueOnce(replyResponse());

    await thread.submit({
      type: "UserInput",
      items: [{ type: "text", text: "fresh ask" }],
      extraUserMessages: [[{ type: "text", text: "extra ask" }]],
    });
    await waitForEvent(thread, (msg) => msg.type === "TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(3);

    // The summarizer never saw this turn's input.
    const summarizer = bodyOf(fetchMock, 1);
    expect(summarizer.input[summarizer.input.length - 1]).toEqual({
      type: "message",
      role: "user",
      content: SUMMARIZATION_PROMPT,
    });
    expect(JSON.stringify(summarizer.input)).not.toContain("fresh ask");
    expect(JSON.stringify(summarizer.input)).not.toContain("extra ask");

    // The next request: …, <summary>, <new user msgs>.
    const sampling = bodyOf(fetchMock, 2).input;
    const tail = sampling.slice(-3);
    expect(JSON.stringify(tail[0])).toContain(SUMMARY_PREFIX.slice(0, 30));
    expect(tail.slice(1)).toEqual([user("fresh ask"), user("extra ask")]);
    // Kept verbatim, not as a re-collected retained message before the summary.
    expect(
      sampling.filter((item) => JSON.stringify(item).includes("fresh ask")),
    ).toHaveLength(1);
  });

  it("persists Compacted → TokenCount → the new user messages, in that order", async () => {
    const io = new InMemoryIoBackend();
    const fetchMock = vi.fn();
    const thread = await threadMarkedFull(fetchMock, io);
    fetchMock.mockResolvedValueOnce(summaryResponse()).mockResolvedValueOnce(replyResponse());

    await thread.submit({
      type: "UserInput",
      items: [{ type: "text", text: "fresh ask" }],
      extraUserMessages: [[{ type: "text", text: "extra ask" }]],
    });
    await waitForEvent(thread, (msg) => msg.type === "TurnComplete");

    const items = await storedItems(io, thread.id);
    const at = items.findIndex((item) => item.kind === "Compacted");
    expect(at).toBeGreaterThan(0);
    const next = items.slice(at, at + 4);
    expect(next.map((item) => (item.kind === "EventMsg" ? item.msg.type : item.kind))).toEqual([
      "Compacted",
      "TokenCount",
      "ConversationItem",
      "ConversationItem",
    ]);
    expect(next.slice(2).map((item) => (item as { item: ConversationItem }).item)).toEqual([
      user("fresh ask"),
      user("extra ask"),
    ]);
    // …and the checkpoint itself does not contain them (they came after it).
    expect(JSON.stringify(items[at])).not.toContain("fresh ask");
  });

  it("a failed pre-turn compaction still records the input, then surfaces the error", async () => {
    const io = new InMemoryIoBackend();
    const fetchMock = vi.fn();
    const thread = await threadMarkedFull(fetchMock, io);
    fetchMock.mockResolvedValueOnce(errorResponse(500, "upstream down"));

    await thread.submit({
      type: "UserInput",
      items: [{ type: "text", text: "fresh ask" }],
    });
    const complete = await waitForEvent(thread, (msg) => msg.type === "TurnComplete");
    expect(complete.type === "TurnComplete" && complete.event.error).toBeTruthy();
    // No sampling request after the failed compaction.
    expect(fetchMock).toHaveBeenCalledTimes(2);

    const history = liveHistoryOf(thread);
    expect(history[history.length - 1]).toEqual(user("fresh ask"));
    const items = await storedItems(io, thread.id);
    const stored = items.filter((item) => item.kind === "ConversationItem");
    expect((stored[stored.length - 1] as { item: ConversationItem }).item).toEqual(
      user("fresh ask"),
    );
    expect(items.some((item) => item.kind === "Compacted")).toBe(false);
  });

  it("an interrupted pre-turn compaction still records the input", async () => {
    const io = new InMemoryIoBackend();
    const fetchMock = vi.fn();
    const thread = await threadMarkedFull(fetchMock, io);
    fetchMock.mockImplementationOnce(async () => {
      await thread.submit({ type: "Interrupt" });
      return summaryResponse();
    });

    await thread.submit({
      type: "UserInput",
      items: [{ type: "text", text: "fresh ask" }],
    });
    await waitForEvent(thread, (msg) => msg.type === "TurnAborted");
    const history = liveHistoryOf(thread);
    expect(history[history.length - 1]).toEqual(user("fresh ask"));
    expect(JSON.stringify(history)).not.toContain(SUMMARY_PREFIX.slice(0, 30));
  });
});

// The new messages are not in history when the check runs, but the next
// request carries them, so the check counts them (upstream leaves this as a
// TODO at turn.rs:221-224).
describe("pre-turn check counts the incoming messages", () => {
  async function runWithIncoming(text: string) {
    const history: ConversationItem[] = [
      user("old ask"),
      { type: "message", role: "assistant", content: "old reply" },
    ];
    const tokenState = new SessionTokenState();
    tokenState.updateFromUsage({
      inputTokens: 890,
      totalTokens: 900,
      coveredThrough: history[0]!,
    });
    const fetch = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string) as Body;
      const last = JSON.stringify(body.input[body.input.length - 1]);
      return last.includes(SUMMARIZATION_PROMPT.slice(0, 30))
        ? summaryResponse()
        : replyResponse();
    });
    await runTurn(
      "turn",
      [{ type: "text", text }],
      history,
      {
        apiKey: "t",
        baseUrl: "http://x",
        model: "m",
        fetch: fetch as unknown as typeof globalThis.fetch,
        contextWindow: 1_000,
        tokenState,
      },
      new ToolRouter(new GoalToolExecutor("t")),
      new Map(),
      () => undefined,
    );
    return fetch.mock.calls.length;
  }

  it("compacts when only the incoming message pushes the context past the window", async () => {
    // 900 + ~209 estimated tokens ≥ 1000.
    expect(await runWithIncoming("x".repeat(800))).toBe(2);
  });

  it("does not compact when the incoming message fits", async () => {
    expect(await runWithIncoming("hi")).toBe(1);
  });

  // The turn-start seed counts them too, as when they were recorded first:
  // otherwise a fresh thread's window baseline would exclude its own first
  // message, and a long one would read as "growth" and compact an empty history.
  it("a fresh thread's long first message is baseline, not growth", async () => {
    const fetchMock = vi.fn(async () => replyResponse());
    vi.stubGlobal("fetch", fetchMock);
    const thread = new CodexThread({ apiKey: "t", model: "m", autoCompactTokenLimit: 100 });
    await thread.submit({
      type: "UserInput",
      items: [{ type: "text", text: "x".repeat(4_000) }],
    });
    await waitForEvent(thread, (msg) => msg.type === "TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const only = bodyOf(fetchMock, 0);
    expect(JSON.stringify(only.input[only.input.length - 1])).not.toContain(
      SUMMARIZATION_PROMPT.slice(0, 30),
    );
  });
});
