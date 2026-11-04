/**
 * Token state survives a restart.
 *
 * mirrors codex-rs persisting EventMsg::TokenCount (rollout/src/policy.rs) and
 * seeding token_info from the last one on resume (session/mod.rs
 * last_token_info_from_rollout), plus set_total_tokens_full →
 * send_token_count_event for the context-window self-heal.
 *
 * Without this a resumed thread starts from a fresh estimate: its first turn
 * cannot trip the growth line (baseline = total = the same estimate) and the
 * estimate under-reads dense threads — so an oversized history went straight
 * to the provider and was rejected (production 2026-11-04).
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { CodexThread } from "../../src/codex_thread.js";
import { SUMMARIZATION_PROMPT } from "../../src/compact.js";
import { runTurn } from "../../src/session/turn.js";
import { ToolRouter } from "../../src/tools/router.js";
import { GoalToolExecutor } from "../../../ext/goal/src/tool.js";
import { AutoCompactWindow } from "../../src/state/auto_compact_window.js";
import {
  approxTokenCount,
  estimateItemTokenCount,
  SessionTokenState,
  utf8ByteLength,
} from "../../src/state/token_state.js";
import { InMemoryIoBackend } from "../../../thread-store/src/io_backend.js";
import type { IoBackend } from "../../../thread-store/src/io_backend.js";
import { LiveThread } from "../../../thread-store/src/live_thread.js";
import { LocalThreadStore } from "../../../thread-store/src/local_thread_store.js";
import type {
  ConversationItem,
  RolloutItem,
} from "../../../thread-store/src/types.js";
import {
  evAssistantMessage,
  evCompleted,
  evFunctionCall,
  evResponseCreated,
  makeSseResponse,
  sseFlat,
  waitForEvent,
} from "../common/lib.js";

beforeEach(() => {
  vi.unstubAllGlobals();
});

function sseWithUsage(totalTokens: number, items: object[]): string {
  return sseFlat([
    ...items,
    {
      type: "response.completed",
      response: {
        id: "resp",
        status: "completed",
        usage: {
          input_tokens: totalTokens - 10,
          output_tokens: 10,
          total_tokens: totalTokens,
        },
      },
    },
  ]);
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
  error: {
    message: "prompt is too long: 1039913 tokens > 1000000 maximum",
    code: 400,
  },
});

async function storedItems(io: IoBackend, threadId: string): Promise<RolloutItem[]> {
  return (await new LocalThreadStore(io).loadHistory({ threadId })).items;
}

function tokenCountTotals(items: RolloutItem[]): number[] {
  return items.flatMap((item) =>
    item.kind === "EventMsg" && item.msg.type === "TokenCount"
      ? [item.msg.event.info?.total_token_usage.total_tokens ?? -1]
      : [],
  );
}

function isSummarizerBody(init: RequestInit): boolean {
  const body = JSON.parse(init.body as string) as { input: unknown[] };
  return JSON.stringify(body.input[body.input.length - 1]).includes(
    SUMMARIZATION_PROMPT.slice(0, 30),
  );
}

const summaryResponse = () =>
  makeSseResponse(
    sseFlat([
      evResponseCreated("compact"),
      evAssistantMessage("summary text"),
      evCompleted("compact"),
    ]),
  );

// =============================================================================
// What gets persisted
// =============================================================================

describe("TokenCount persistence", () => {
  it("persists the server-measured total of each sampled response", async () => {
    const io = new InMemoryIoBackend();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        makeSseResponse(
          sseWithUsage(4_321, [evResponseCreated("r"), evAssistantMessage("ok")]),
        ),
      ),
    );
    const thread = new CodexThread({ apiKey: "t", model: "m", ioBackend: io });
    await thread.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });
    await waitForEvent(thread, (msg) => msg.type === "TurnComplete");
    expect(tokenCountTotals(await storedItems(io, thread.id))).toEqual([4_321]);
  });

  it("persists the forced-full mark of a context-window rejection", async () => {
    const io = new InMemoryIoBackend();
    vi.stubGlobal("fetch", vi.fn(async () => errorResponse(400, PROMPT_TOO_LONG)));
    const thread = new CodexThread({
      apiKey: "t",
      model: "m",
      contextWindow: 1_000,
      ioBackend: io,
    });
    await thread.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });
    await waitForEvent(thread, (msg) => msg.type === "TurnComplete");
    expect(tokenCountTotals(await storedItems(io, thread.id))).toEqual([1_000]);
  });

  it("persists the post-compaction estimate right after the Compacted line", async () => {
    const io = new InMemoryIoBackend();
    const fetchMock = vi.fn();
    fetchMock
      .mockResolvedValueOnce(
        makeSseResponse(
          sseWithUsage(1_000, [evResponseCreated("r1"), evFunctionCall("c1", "get_goal", {})]),
        ),
      )
      .mockResolvedValueOnce(
        makeSseResponse(
          sseWithUsage(2_000, [evResponseCreated("r2"), evFunctionCall("c2", "get_goal", {})]),
        ),
      )
      .mockResolvedValueOnce(summaryResponse())
      .mockResolvedValueOnce(
        makeSseResponse(sseWithUsage(300, [evResponseCreated("r3"), evAssistantMessage("done")])),
      );
    vi.stubGlobal("fetch", fetchMock);
    const thread = new CodexThread({
      apiKey: "t",
      model: "m",
      autoCompactTokenLimit: 900,
      ioBackend: io,
    });
    await thread.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });
    await waitForEvent(thread, (msg) => msg.type === "TurnComplete");

    const items = await storedItems(io, thread.id);
    const compactedAt = items.findIndex((item) => item.kind === "Compacted");
    expect(compactedAt).toBeGreaterThan(0);
    const next = items[compactedAt + 1];
    expect(next?.kind === "EventMsg" && next.msg.type === "TokenCount").toBe(true);
    const compacted = (items[compactedAt] as { compacted: { token_total_after: number } })
      .compacted;
    expect(tokenCountTotals([next!])).toEqual([compacted.token_total_after]);
  });
});

// =============================================================================
// Resume seeding
// =============================================================================

describe("resume seeding", () => {
  it("a resumed thread whose persisted total is past the window compacts before its first sample", async () => {
    const io = new InMemoryIoBackend();
    const threadId = "resume-full";
    const live = new LiveThread(threadId, new LocalThreadStore(io));
    void live.appendConversationItems([
      { type: "message", role: "user", content: [{ type: "input_text", text: "old" }] },
      { type: "message", role: "assistant", content: "old reply" },
    ]);
    void live.appendEvent({
      type: "TokenCount",
      event: {
        info: {
          total_token_usage: {
            input_tokens: 0,
            cached_input_tokens: 0,
            cache_write_input_tokens: 0,
            output_tokens: 0,
            reasoning_output_tokens: 0,
            total_tokens: 1_500,
          },
          last_token_usage: {
            input_tokens: 0,
            cached_input_tokens: 0,
            cache_write_input_tokens: 0,
            output_tokens: 0,
            reasoning_output_tokens: 0,
            total_tokens: 1_500,
          },
          model_context_window: 1_000,
        },
        rate_limits: null,
      },
    });
    await live.flush();

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(summaryResponse())
      .mockResolvedValueOnce(
        makeSseResponse(sseWithUsage(200, [evResponseCreated("r"), evAssistantMessage("ok")])),
      );
    vi.stubGlobal("fetch", fetchMock);
    const resumed = await CodexThread.create({
      apiKey: "t",
      model: "m",
      contextWindow: 1_000,
      threadId,
      ioBackend: io,
    });
    await resumed.submit({ type: "UserInput", items: [{ type: "text", text: "next" }] });
    await waitForEvent(resumed, (msg) => msg.type === "TurnComplete");
    expect(isSummarizerBody(fetchMock.mock.calls[0]![1] as RequestInit)).toBe(true);
  });

  it("a rejection, then a restart: the resumed thread's first turn compacts first", async () => {
    const io = new InMemoryIoBackend();
    vi.stubGlobal("fetch", vi.fn(async () => errorResponse(400, PROMPT_TOO_LONG)));
    const first = new CodexThread({
      apiKey: "t",
      model: "m",
      contextWindow: 1_000,
      ioBackend: io,
    });
    await first.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });
    await waitForEvent(first, (msg) => msg.type === "TurnComplete");

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(summaryResponse())
      .mockResolvedValueOnce(
        makeSseResponse(sseWithUsage(200, [evResponseCreated("r"), evAssistantMessage("ok")])),
      );
    vi.stubGlobal("fetch", fetchMock);
    const resumed = await CodexThread.create({
      apiKey: "t",
      model: "m",
      contextWindow: 1_000,
      threadId: first.id,
      ioBackend: io,
    });
    await resumed.submit({ type: "UserInput", items: [{ type: "text", text: "again" }] });
    await waitForEvent(resumed, (msg) => msg.type === "TurnComplete");
    expect(isSummarizerBody(fetchMock.mock.calls[0]![1] as RequestInit)).toBe(true);
  });

  it("a small persisted total does not compact on resume", async () => {
    const io = new InMemoryIoBackend();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        makeSseResponse(sseWithUsage(120, [evResponseCreated("r"), evAssistantMessage("ok")])),
      ),
    );
    const first = new CodexThread({ apiKey: "t", model: "m", contextWindow: 1_000, ioBackend: io });
    await first.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });
    await waitForEvent(first, (msg) => msg.type === "TurnComplete");

    const fetchMock = vi.fn(async (_url: string, _init: RequestInit) =>
      makeSseResponse(sseWithUsage(150, [evResponseCreated("r"), evAssistantMessage("ok")])),
    );
    vi.stubGlobal("fetch", fetchMock);
    const resumed = await CodexThread.create({
      apiKey: "t",
      model: "m",
      contextWindow: 1_000,
      threadId: first.id,
      ioBackend: io,
    });
    await resumed.submit({ type: "UserInput", items: [{ type: "text", text: "next" }] });
    await waitForEvent(resumed, (msg) => msg.type === "TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(isSummarizerBody(fetchMock.mock.calls[0]![1] as RequestInit)).toBe(false);
  });
});

// =============================================================================
// Stale usage after a concurrent compaction
// =============================================================================

describe("usage from a request built before a compaction", () => {
  async function runWithLateUsage(compactMeanwhile: boolean) {
    const io = new InMemoryIoBackend();
    const live = new LiveThread("stale", new LocalThreadStore(io));
    const compactWindow = new AutoCompactWindow();
    const tokenState = new SessionTokenState();
    let release: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const fetch = vi.fn(async () => {
      await gate;
      return makeSseResponse(
        sseWithUsage(999_010, [evResponseCreated("r"), evAssistantMessage("late")]),
      );
    });
    const history: ConversationItem[] = [];
    const done = runTurn(
      "turn",
      [{ type: "text", text: "hi" }],
      history,
      { apiKey: "t", baseUrl: "http://x", model: "m", fetch, compactWindow, tokenState },
      new ToolRouter(new GoalToolExecutor("stale")),
      new Map(),
      () => undefined,
      live,
    );
    await vi.waitFor(() => expect(fetch).toHaveBeenCalled());
    if (compactMeanwhile) compactWindow.bumpCompactionEpoch();
    release();
    await done;
    await live.flush();
    return { tokenState, persisted: tokenCountTotals(await storedItems(io, "stale")) };
  }

  it("is applied and persisted normally when nothing compacted meanwhile", async () => {
    const { tokenState, persisted } = await runWithLateUsage(false);
    expect(tokenState.totalTokens).toBe(999_010);
    expect(persisted).toEqual([999_010]);
  });

  it("is ignored for the baseline and not persisted when a compaction landed meanwhile", async () => {
    const { tokenState, persisted } = await runWithLateUsage(true);
    expect(tokenState.totalTokens).not.toBe(999_010);
    expect(persisted).toEqual([]);
  });
});

// =============================================================================
// Estimator
// =============================================================================

describe("token estimator", () => {
  it("counts UTF-8 bytes, so CJK is not under-read 3×", () => {
    const cjk = "音".repeat(1_000); // 3 bytes each in UTF-8
    expect(utf8ByteLength(cjk)).toBe(3_000);
    expect(approxTokenCount(cjk)).toBe(750);
    expect(approxTokenCount("a".repeat(1_000))).toBe(250);
  });

  it("counts a surrogate pair as 4 bytes", () => {
    expect(utf8ByteLength("😀")).toBe(4);
  });

  it("charges an image a fixed cost, not the length of its data URL", () => {
    const image = (url: string): ConversationItem => ({
      type: "message",
      role: "user",
      content: [{ type: "input_image", image_url: url }],
    });
    const small = estimateItemTokenCount(image("https://x/y.png"));
    const huge = estimateItemTokenCount(image(`data:image/png;base64,${"A".repeat(2_000_000)}`));
    expect(huge).toBe(small);
  });

  it("estimates a content-part array stored in a function output (view_image)", () => {
    const output = [
      { type: "input_text", text: "caption" },
      { type: "input_image", image_url: `data:image/png;base64,${"A".repeat(500_000)}` },
    ];
    const tokens = estimateItemTokenCount({
      type: "function_call_output",
      call_id: "c",
      output: output as unknown as string,
    });
    expect(tokens).toBeLessThan(5_000);
  });
});
