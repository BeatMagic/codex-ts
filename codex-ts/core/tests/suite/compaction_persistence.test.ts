/**
 * Compaction checkpoints are persisted, and a resume starts from the newest one.
 *
 * mirrors codex-rs/core/src/session/compaction_replay_tests.rs
 * (reconstruct_history_matches_live_compactions) and rollout_reconstruction.rs.
 *
 * Production 2026-11-04: compaction only rewrote the in-memory history, so a
 * compacted director thread (260KB) came back after an app restart as its full
 * raw history (2.5MB, 1.05M tokens) and was rejected "prompt is too long".
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { CodexThread } from "../../src/codex_thread.js";
import {
  runInlineAutoCompactTask,
  SUMMARY_PREFIX,
} from "../../src/compact.js";
import { InMemoryIoBackend } from "../../../thread-store/src/io_backend.js";
import type { IoBackend } from "../../../thread-store/src/io_backend.js";
import { LiveThread } from "../../../thread-store/src/live_thread.js";
import { LocalThreadStore } from "../../../thread-store/src/local_thread_store.js";
import { reconstructModelContext } from "../../../thread-store/src/reconstruct.js";
import type {
  CompactedItem,
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

function sseWithUsage(inputTokens: number, items: object[]): string {
  return sseFlat([
    ...items,
    {
      type: "response.completed",
      response: {
        id: "resp",
        status: "completed",
        usage: {
          input_tokens: inputTokens,
          output_tokens: 10,
          total_tokens: inputTokens + 10,
        },
      },
    },
  ]);
}

function liveHistoryOf(thread: CodexThread): ConversationItem[] {
  return (thread as unknown as { history: ConversationItem[] }).history;
}

async function storedItems(
  io: IoBackend,
  threadId: string,
): Promise<RolloutItem[]> {
  return (await new LocalThreadStore(io).loadHistory({ threadId })).items;
}

/**
 * One turn that compacts mid-turn: get_goal (usage 1000, the baseline) →
 * update_plan (usage 2000, growth 1000 ≥ 900) → summarizer → final reply.
 */
async function runCompactingTurn(io: IoBackend): Promise<CodexThread> {
  const fetchMock = vi.fn();
  fetchMock
    .mockResolvedValueOnce(
      makeSseResponse(
        sseWithUsage(1000, [
          evResponseCreated("r1"),
          evFunctionCall("call-1", "get_goal", {}),
        ]),
      ),
    )
    .mockResolvedValueOnce(
      makeSseResponse(
        sseWithUsage(2000, [
          evResponseCreated("r2"),
          evFunctionCall("call-2", "update_plan", {
            plan: [{ step: "s", status: "completed" }],
          }),
        ]),
      ),
    )
    .mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("compact"),
          evAssistantMessage("summary text"),
          evCompleted("compact"),
        ]),
      ),
    )
    .mockResolvedValueOnce(
      makeSseResponse(
        sseWithUsage(300, [evResponseCreated("r3"), evAssistantMessage("done")]),
      ),
    );
  vi.stubGlobal("fetch", fetchMock);

  const thread = new CodexThread({
    apiKey: "test",
    model: "gpt-4o",
    autoCompactTokenLimit: 900,
    ioBackend: io,
  });
  await thread.submit({
    type: "UserInput",
    items: [{ type: "text", text: "hi" }],
  });
  await waitForEvent(thread, (msg) => msg.type === "ContextCompacted");
  await waitForEvent(thread, (msg) => msg.type === "TurnComplete");
  return thread;
}

// =============================================================================
// End to end: compact, persist, resume
// =============================================================================

describe("compaction checkpoint — end to end", () => {
  it("persists a Compacted line whose reconstruction equals the live history", async () => {
    const io = new InMemoryIoBackend();
    const thread = await runCompactingTurn(io);

    const items = await storedItems(io, thread.id);
    const checkpoints = items.filter((item) => item.kind === "Compacted");
    expect(checkpoints).toHaveLength(1);
    const compacted = (checkpoints[0] as { compacted: CompactedItem }).compacted;
    expect(compacted.v).toBe(1);
    expect(compacted.message.startsWith(`${SUMMARY_PREFIX}\n`)).toBe(true);
    expect(compacted.window_number).toBe(1);

    expect(reconstructModelContext(items).history).toEqual(liveHistoryOf(thread));
  });

  it("a resumed thread sends the compacted history, while the raw transcript stays whole", async () => {
    const io = new InMemoryIoBackend();
    const first = await runCompactingTurn(io);

    const bodies: { input: unknown[] }[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        bodies.push(JSON.parse(init.body as string) as { input: unknown[] });
        return makeSseResponse(
          sseFlat([
            evResponseCreated("r4"),
            evAssistantMessage("resumed"),
            evCompleted("r4"),
          ]),
        );
      }),
    );
    const resumed = await CodexThread.create({
      apiKey: "test",
      model: "gpt-4o",
      threadId: first.id,
      ioBackend: io,
    });
    await resumed.submit({
      type: "UserInput",
      items: [{ type: "text", text: "follow-up" }],
    });
    await waitForEvent(resumed, (msg) => msg.type === "TurnComplete");

    const sent = JSON.stringify(bodies[0]!.input);
    expect(sent).toContain(SUMMARY_PREFIX.slice(0, 30));
    // The pre-compaction tool calls were summarized away — not resurrected.
    expect(sent).not.toContain("call-1");
    expect(sent).not.toContain("call-2");
    expect(sent).toContain("follow-up");

    // The chat transcript (raw replay) still has every item ever appended.
    const raw = await new LiveThread(
      first.id,
      new LocalThreadStore(io),
    ).loadConversationHistory();
    const rawText = JSON.stringify(raw);
    expect(rawText).toContain("call-1");
    expect(rawText).toContain("call-2");
    expect(rawText).not.toContain(SUMMARY_PREFIX.slice(0, 30));
  });

  it("an older runtime's reader (ConversationItem only) still sees the raw items", async () => {
    const io = new InMemoryIoBackend();
    const thread = await runCompactingTurn(io);
    const items = await storedItems(io, thread.id);
    const oldReader = items
      .filter((item) => item.kind === "ConversationItem")
      .map((item) => (item as { item: ConversationItem }).item);
    expect(oldReader.some((item) => item.type === "function_call")).toBe(true);
    expect(JSON.stringify(oldReader)).not.toContain(SUMMARY_PREFIX.slice(0, 30));
  });

  it("writes no checkpoint when the summarizer never completes", async () => {
    const io = new InMemoryIoBackend();
    const fetchMock = vi.fn();
    fetchMock
      .mockResolvedValueOnce(
        makeSseResponse(
          sseWithUsage(1000, [
            evResponseCreated("r1"),
            evFunctionCall("call-1", "get_goal", {}),
          ]),
        ),
      )
      .mockResolvedValueOnce(
        makeSseResponse(
          sseWithUsage(2000, [
            evResponseCreated("r2"),
            evFunctionCall("call-2", "get_goal", {}),
          ]),
        ),
      )
      // Summarizer stream that just ends: no response.completed.
      .mockResolvedValue(
        makeSseResponse(
          sseFlat([evResponseCreated("compact"), evAssistantMessage("half a summ")]),
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    const thread = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      autoCompactTokenLimit: 900,
      maxRetries: 0,
      ioBackend: io,
    });
    await thread.submit({
      type: "UserInput",
      items: [{ type: "text", text: "hi" }],
    });
    const error = await waitForEvent(thread, (msg) => msg.type === "Error");
    expect(JSON.stringify(error)).toContain("before response.completed");

    const items = await storedItems(io, thread.id);
    expect(items.some((item) => item.kind === "Compacted")).toBe(false);
    expect(JSON.stringify(liveHistoryOf(thread))).not.toContain(
      SUMMARY_PREFIX.slice(0, 30),
    );
  });
});

// =============================================================================
// reconstructModelContext — pure
// =============================================================================

const user = (text: string): ConversationItem => ({
  type: "message",
  role: "user",
  content: [{ type: "input_text", text }],
});
const assistant = (text: string): ConversationItem => ({
  type: "message",
  role: "assistant",
  content: text,
});
const conv = (item: ConversationItem): RolloutItem => ({
  kind: "ConversationItem",
  item,
});
const checkpoint = (
  replacement: ConversationItem[],
  extra: Partial<CompactedItem> = {},
): RolloutItem => ({
  kind: "Compacted",
  compacted: {
    v: 1,
    message: `${SUMMARY_PREFIX}\nS`,
    replacement_history: replacement,
    ...extra,
  },
});
const tokenCount = (total: number): RolloutItem => ({
  kind: "EventMsg",
  msg: {
    type: "TokenCount",
    event: {
      info: {
        total_token_usage: {
          input_tokens: 0,
          cached_input_tokens: 0,
          cache_write_input_tokens: 0,
          output_tokens: 0,
          reasoning_output_tokens: 0,
          total_tokens: total,
        },
        last_token_usage: {
          input_tokens: 0,
          cached_input_tokens: 0,
          cache_write_input_tokens: 0,
          output_tokens: 0,
          reasoning_output_tokens: 0,
          total_tokens: total,
        },
        model_context_window: null,
      },
      rate_limits: null,
    },
  },
});

describe("reconstructModelContext", () => {
  it("a legacy log (no checkpoint) replays every conversation item", () => {
    const items = [conv(user("a")), conv(assistant("b")), conv(user("c"))];
    const state = reconstructModelContext(items);
    expect(state.history).toEqual([user("a"), assistant("b"), user("c")]);
    expect(state.checkpoint).toBeNull();
    expect(state.lastTokenTotal).toBeNull();
  });

  it("starts from the NEWEST checkpoint and replays only what follows it", () => {
    const items = [
      conv(user("old")),
      checkpoint([user("first summary")]),
      conv(user("mid")),
      checkpoint([user("second summary")]),
      conv(assistant("after")),
    ];
    expect(reconstructModelContext(items).history).toEqual([
      user("second summary"),
      assistant("after"),
    ]);
  });

  it("a malformed newest checkpoint means a full replay, never an older checkpoint", () => {
    const items = [
      conv(user("a")),
      checkpoint([user("good summary")]),
      conv(user("b")),
      tokenCount(50),
      { kind: "Compacted", compacted: { v: 1, message: "x" } } as unknown as RolloutItem,
      conv(user("c")),
    ];
    const state = reconstructModelContext(items);
    expect(state.history).toEqual([user("a"), user("b"), user("c")]);
    expect(state.checkpoint).toBeNull();
    // TokenCounts measured a compacted context; the replay is the raw one.
    expect(state.lastTokenTotal).toBeNull();
  });

  it("seeds from the last TokenCount AFTER the checkpoint, ignoring older ones", () => {
    const items = [
      tokenCount(900_000),
      checkpoint([user("summary")], { token_total_after: 1_200 }),
      conv(user("x")),
      tokenCount(3_000),
      tokenCount(4_000),
    ];
    expect(reconstructModelContext(items).lastTokenTotal).toBe(4_000);
  });

  it("falls back to the checkpoint's own estimate when nothing follows it", () => {
    const items = [
      tokenCount(900_000),
      checkpoint([user("summary")], { token_total_after: 1_200 }),
      conv(user("x")),
    ];
    expect(reconstructModelContext(items).lastTokenTotal).toBe(1_200);
  });

  it("returns a copy: mutating the resumed history leaves the stored snapshot intact", () => {
    const replacement = [user("summary")];
    const items = [checkpoint(replacement)];
    const state = reconstructModelContext(items);
    state.history.push(user("new"));
    expect(replacement).toHaveLength(1);
  });
});

// =============================================================================
// runInlineAutoCompactTask — the checkpoint hook
// =============================================================================

function summarizerFetch(events: object[]) {
  return vi.fn(async () => makeSseResponse(sseFlat(events)));
}

function compactConfig(fetch: typeof globalThis.fetch, extra: object = {}) {
  return {
    apiKey: "test",
    baseUrl: "http://localhost",
    model: "gpt-4o",
    fetch,
    maxRetries: 0,
    ...extra,
  };
}

describe("runInlineAutoCompactTask — onHistoryReplaced", () => {
  it("fires once, in the same tick as the swap, with the live history", async () => {
    const history = [user("a"), assistant("b")];
    const hook = vi.fn((replacement: ConversationItem[]) => {
      // Same array, already rewritten.
      expect(replacement).toBe(history);
      expect(JSON.stringify(replacement)).toContain(SUMMARY_PREFIX.slice(0, 30));
    });
    await runInlineAutoCompactTask(
      history,
      compactConfig(
        summarizerFetch([
          evResponseCreated("s"),
          evAssistantMessage("SUM"),
          evCompleted("s"),
        ]),
        { onHistoryReplaced: hook },
      ),
    );
    expect(hook).toHaveBeenCalledTimes(1);
  });

  it("does not fire for a stale compaction", async () => {
    const history = [user("a")];
    const hook = vi.fn();
    const result = await runInlineAutoCompactTask(
      history,
      compactConfig(
        summarizerFetch([
          evResponseCreated("s"),
          evAssistantMessage("SUM"),
          evCompleted("s"),
        ]),
        { onHistoryReplaced: hook, isStaleCompaction: () => true },
      ),
    );
    expect(result).toBeNull();
    expect(hook).not.toHaveBeenCalled();
    expect(history).toEqual([user("a")]);
  });

  it("does not fire, and rewrites nothing, when interrupted mid-summary", async () => {
    const history = [user("a")];
    const hook = vi.fn();
    const controller = new AbortController();
    const fetch = vi.fn(async () => {
      controller.abort();
      return makeSseResponse(
        sseFlat([evResponseCreated("s"), evAssistantMessage("SUM"), evCompleted("s")]),
      );
    });
    await expect(
      runInlineAutoCompactTask(
        history,
        compactConfig(fetch, {
          onHistoryReplaced: hook,
          abortSignal: controller.signal,
        }),
      ),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(hook).not.toHaveBeenCalled();
    expect(history).toEqual([user("a")]);
  });

  it("rejects an empty summary instead of replacing history with a placeholder", async () => {
    const history = [user("a")];
    const hook = vi.fn();
    await expect(
      runInlineAutoCompactTask(
        history,
        compactConfig(
          summarizerFetch([evResponseCreated("s"), evCompleted("s")]),
          { onHistoryReplaced: hook },
        ),
      ),
    ).rejects.toThrow(/empty summary/u);
    expect(hook).not.toHaveBeenCalled();
    expect(history).toEqual([user("a")]);
  });

  it("keeps items another turn appended while the summary was in flight, after the summary", async () => {
    const history = [user("a"), assistant("b")];
    const late: ConversationItem[] = [
      { type: "function_call", call_id: "late-1", name: "t", arguments: "{}" },
      { type: "function_call_output", call_id: "late-1", output: "ok" },
    ];
    const fetch = vi.fn(async () => {
      history.push(...late); // a concurrent turn, mid-flight
      return makeSseResponse(
        sseFlat([evResponseCreated("s"), evAssistantMessage("SUM"), evCompleted("s")]),
      );
    });
    await runInlineAutoCompactTask(history, compactConfig(fetch));
    const summaryIndex = history.findIndex((item) =>
      JSON.stringify(item).includes(SUMMARY_PREFIX.slice(0, 30)),
    );
    expect(summaryIndex).toBeGreaterThanOrEqual(0);
    expect(history.slice(summaryIndex + 1)).toEqual(late);
  });
});

// =============================================================================
// LiveThread — ordered, copied appends
// =============================================================================

class SlowIoBackend extends InMemoryIoBackend {
  private turn = 0;
  override async appendLine(threadId: string, line: string): Promise<void> {
    // Later calls finish FIRST unless the caller serializes them.
    this.turn += 1;
    await new Promise((resolve) => setTimeout(resolve, 12 - (this.turn % 12)));
    await super.appendLine(threadId, line);
  }
}

describe("LiveThread persistence queue", () => {
  it("writes in call order even when the backend completes writes out of order", async () => {
    const io = new SlowIoBackend();
    const live = new LiveThread("t", new LocalThreadStore(io));
    const expected: string[] = [];
    for (let index = 0; index < 12; index += 1) {
      expected.push(`m${index}`);
      void live.appendConversationItems([user(`m${index}`)]);
      if (index === 6) {
        void live.appendCompacted({
          v: 1,
          message: "S",
          replacement_history: [user("S")],
        });
      }
    }
    await live.flush();
    const items = await storedItems(io, "t");
    const texts = items.map((item) =>
      item.kind === "ConversationItem"
        ? JSON.stringify(item.item).match(/m\d+/u)?.[0]
        : "CHECKPOINT",
    );
    expect(texts).toEqual([
      ...expected.slice(0, 7),
      "CHECKPOINT",
      ...expected.slice(7),
    ]);
  });

  it("stores a copy taken at call time, not a later in-place mutation", async () => {
    const io = new InMemoryIoBackend();
    const live = new LiveThread("t", new LocalThreadStore(io));
    const call: ConversationItem = {
      type: "function_call",
      call_id: "c",
      name: "t",
      arguments: "{raw",
    };
    void live.appendConversationItems([call]);
    call.arguments = "{}"; // e.g. canonicalizeStoredArguments
    await live.flush();
    const items = await storedItems(io, "t");
    expect((items[0] as { item: { arguments: string } }).item.arguments).toBe("{raw");
  });

  it("a failed write is logged and does not break later writes", async () => {
    let fail = true;
    const io = new InMemoryIoBackend();
    const store = new LocalThreadStore(io);
    const original = store.appendItems.bind(store);
    store.appendItems = async (params) => {
      if (fail) {
        fail = false;
        throw new Error("disk full");
      }
      return original(params);
    };
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const live = new LiveThread("t", store);
    await expect(live.appendConversationItems([user("lost")])).resolves.toBeUndefined();
    await live.appendConversationItems([user("kept")]);
    const items = await storedItems(io, "t");
    expect(JSON.stringify(items)).toContain("kept");
    expect(JSON.stringify(items)).not.toContain("lost");
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});
