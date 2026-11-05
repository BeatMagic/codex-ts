/**
 * One model request in flight per thread (session/request_gate.ts).
 *
 * codex-rs never has two same-thread requests in flight (one active turn,
 * steered input). codex-ts runs turns unawaited, so without the gate two
 * turns on one thread sample at once — racing on the shared history, and each
 * paying a full prompt-cache write because an entry is readable only once the
 * first response starts streaming. The gate covers requests only: it must
 * never be held across a tool or a request_user_input wait.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import { runInlineAutoCompactTask, SUMMARIZATION_PROMPT } from "../../src/compact.js";
import { RequestGate } from "../../src/session/request_gate.js";
import { runTurn } from "../../src/session/turn.js";
import { ToolRouter } from "../../src/tools/router.js";
import type { CustomTool } from "../../src/tools/router.js";
import { GoalToolExecutor } from "../../../ext/goal/src/tool.js";
import { InMemoryIoBackend } from "../../../thread-store/src/io_backend.js";
import { LocalThreadStore } from "../../../thread-store/src/local_thread_store.js";
import type { ConversationItem } from "../../../thread-store/src/types.js";
import {
  evAssistantMessage,
  evCompleted,
  evFunctionCall,
  evResponseCreated,
  makeSseResponse,
  sseFlat,
  waitForEvent,
} from "../common/lib.js";

const tick = () => new Promise((resolve) => setTimeout(resolve, 5));

// ─── Unit ─────────────────────────────────────────────────────────────────────

describe("RequestGate", () => {
  it("grants one holder at a time, in FIFO order", async () => {
    const gate = new RequestGate();
    const order: string[] = [];
    const releaseA = await gate.acquire();
    const b = gate.acquire().then((release) => {
      order.push("b");
      return release;
    });
    const c = gate.acquire().then((release) => {
      order.push("c");
      return release;
    });
    await tick();
    expect(order).toEqual([]);

    releaseA();
    const releaseB = await b;
    await tick();
    expect(order).toEqual(["b"]);

    releaseB();
    (await c)();
    expect(order).toEqual(["b", "c"]);
  });

  it("an aborted waiter leaves the queue and the gate passes on to the next one", async () => {
    const gate = new RequestGate();
    const releaseA = await gate.acquire();
    const controller = new AbortController();
    const b = gate.acquire(controller.signal);
    let cGranted = false;
    const c = gate.acquire().then((release) => {
      cGranted = true;
      return release;
    });

    controller.abort();
    await expect(b).rejects.toMatchObject({ name: "AbortError" });

    releaseA();
    const releaseC = await c;
    expect(cGranted).toBe(true);
    releaseC();
    // Nobody is left holding it: a stale waiter would have swallowed the
    // hand-over and this acquire would hang.
    const releaseD = await gate.acquire();
    releaseD();
  });

  it("rejects at once on an already-aborted signal", async () => {
    const gate = new RequestGate();
    const controller = new AbortController();
    controller.abort();
    await expect(gate.acquire(controller.signal)).rejects.toMatchObject({
      name: "AbortError",
    });
    // …and did not take the gate.
    (await gate.acquire())();
  });

  it("a double release hands the gate over only once", async () => {
    const gate = new RequestGate();
    const releaseA = await gate.acquire();
    const granted: string[] = [];
    void gate.acquire().then(() => granted.push("b"));
    void gate.acquire().then(() => granted.push("c"));
    releaseA();
    releaseA();
    await tick();
    expect(granted).toEqual(["b"]);
  });
});

// ─── Integration ──────────────────────────────────────────────────────────────

/** A response whose body stays open for `ms`, so overlapping requests show. */
function slowResponse(sseText: string, ms: number, onClose: () => void): Response {
  const bytes = new TextEncoder().encode(sseText);
  return {
    ok: true,
    status: 200,
    body: new ReadableStream({
      start(controller) {
        setTimeout(() => {
          controller.enqueue(bytes);
          controller.close();
          onClose();
        }, ms);
      },
    }),
    text: async () => sseText,
  } as unknown as Response;
}

function blockingTool(name: string) {
  let started!: () => void;
  const startedPromise = new Promise<void>((resolve) => {
    started = resolve;
  });
  let finish!: (value: string) => void;
  const gate = new Promise<string>((resolve) => {
    finish = resolve;
  });
  const tool: CustomTool = {
    name,
    spec: () => ({
      type: "function",
      tool: {
        name,
        description: `${name}.`,
        parameters: { type: "object", properties: {}, additionalProperties: false },
        strict: false,
      },
    }),
    execute: async () => {
      started();
      return gate;
    },
  };
  return { tool, started: startedPromise, finish };
}

describe("two turns on one thread", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("never have two requests open at once", async () => {
    let open = 0;
    let maxOpen = 0;
    const fetchMock = vi.fn(async () => {
      open += 1;
      maxOpen = Math.max(maxOpen, open);
      return slowResponse(
        sseFlat([evResponseCreated("r"), evAssistantMessage("ok"), evCompleted("r")]),
        20,
        () => {
          open -= 1;
        },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({ apiKey: "test", model: "gpt-4o" });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "one" }] });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "two" }] });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(maxOpen).toBe(1);
  });

  it("do not wait for each other's tools: the gate is released before a tool runs", async () => {
    const slow = blockingTool("slow_tool");
    let request = 0;
    const fetchMock = vi.fn(async () => {
      request += 1;
      const output =
        request === 1 ? evFunctionCall("call-slow", "slow_tool", {}) : evAssistantMessage("ok");
      return makeSseResponse(
        sseFlat([evResponseCreated(`r${request}`), output, evCompleted(`r${request}`)]),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      customTools: [slow.tool],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "slow" }] });
    await slow.started;

    // The first turn is parked in its tool; the second must sample anyway.
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "meanwhile" }] });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(2);

    slow.finish("done");
    await waitForEvent(codex, (m) => m.type === "TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });
});

describe("a request waiting for the gate", () => {
  const reply = () =>
    makeSseResponse(sseFlat([evResponseCreated("r"), evAssistantMessage("ok"), evCompleted("r")]));

  it("a sampling request waits for the holder, and an interrupt while waiting sends nothing", async () => {
    const gate = new RequestGate();
    const release = await gate.acquire();
    const fetch = vi.fn(async () => reply());
    const controller = new AbortController();
    const turn = runTurn(
      "t",
      [{ type: "text", text: "hi" }],
      [],
      { apiKey: "k", baseUrl: "http://x", model: "m", fetch, requestGate: gate },
      new ToolRouter(new GoalToolExecutor("g")),
      new Map(),
      () => undefined,
      undefined,
      controller.signal,
    );
    await tick();
    expect(fetch).not.toHaveBeenCalled();

    controller.abort();
    await expect(turn).rejects.toMatchObject({ name: "AbortError" });
    expect(fetch).not.toHaveBeenCalled();
    release();
    // The aborted waiter left no claim behind.
    (await gate.acquire())();
  });

  it("the summarizer request is gated too", async () => {
    const gate = new RequestGate();
    const release = await gate.acquire();
    const history: ConversationItem[] = [
      { type: "message", role: "user", content: "a" },
    ];
    const fetch = vi.fn(async () =>
      makeSseResponse(
        sseFlat([evResponseCreated("s"), evAssistantMessage("SUM"), evCompleted("s")]),
      ),
    );
    const compaction = runInlineAutoCompactTask(history, {
      apiKey: "k",
      baseUrl: "http://x",
      model: "m",
      fetch,
      requestGate: gate,
    });
    await tick();
    expect(fetch).not.toHaveBeenCalled();

    release();
    await expect(compaction).resolves.toBe("SUM");
    expect(fetch).toHaveBeenCalledTimes(1);
    // Released after the summary, whatever happens next.
    (await gate.acquire())();
  });
});

// Two turns that both decided to compact queue on the gate. By the time the
// second holds it the first has compacted, so the second must drop out BEFORE
// sending — not pay a full summarizer request only to be dropped as stale.
describe("compactions queued on the gate", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  const summary = () =>
    makeSseResponse(
      sseFlat([evResponseCreated("s"), evAssistantMessage("SUM"), evCompleted("s")]),
    );

  it("the second, now stale, sends no summarizer request", async () => {
    const gate = new RequestGate();
    const history: ConversationItem[] = [
      { type: "message", role: "user", content: "a" },
      { type: "message", role: "assistant", content: "b" },
    ];
    const fetch = vi.fn(async () => summary());
    // The shape runCompactAndRecompute gives the hook: the generation read
    // on entry, bumped when a replacement lands.
    let epoch = 0;
    const compact = () => {
      const atEntry = epoch;
      return runInlineAutoCompactTask(history, {
        apiKey: "k",
        baseUrl: "http://x",
        model: "m",
        fetch,
        requestGate: gate,
        isStaleCompaction: () => epoch !== atEntry,
        onHistoryReplaced: () => {
          epoch += 1;
        },
      });
    };

    const results = await Promise.all([compact(), compact()]);
    expect(results).toEqual(["SUM", null]);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("two turns that both start over the window send one summarizer request", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    // A resumed thread whose history alone is past its 1 000-token window:
    // every turn's pre-sampling check trips.
    const io = new InMemoryIoBackend();
    await new LocalThreadStore(io).appendItems({
      threadId: "big",
      items: [
        { kind: "ConversationItem", item: { type: "message", role: "user", content: "x".repeat(8_000) } },
        { kind: "ConversationItem", item: { type: "message", role: "assistant", content: "ok" } },
      ],
    });
    const bodies: { input: unknown[] }[] = [];
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string) as { input: unknown[] };
      bodies.push(body);
      const summarizer = JSON.stringify(body.input[body.input.length - 1]).includes(
        SUMMARIZATION_PROMPT.slice(0, 30),
      );
      return summarizer
        ? summary()
        : makeSseResponse(
            sseFlat([evResponseCreated("r"), evAssistantMessage("ok"), evCompleted("r")]),
          );
    });
    vi.stubGlobal("fetch", fetchMock);

    const codex = await CodexThread.create({
      apiKey: "test",
      model: "m",
      contextWindow: 1_000,
      threadId: "big",
      ioBackend: io,
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "one" }] });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "two" }] });
    let completed = 0;
    let compacted = 0;
    while (completed < 2) {
      const msg = await waitForEvent(
        codex,
        (m) => m.type === "TurnComplete" || m.type === "ContextCompacted",
      );
      if (msg.type === "TurnComplete") completed += 1;
      else compacted += 1;
    }
    // History was compacted once…
    expect(compacted).toBe(1);

    // …by the one summarizer request that was sent.
    const summarizerRequests = bodies.filter((body) =>
      JSON.stringify(body.input[body.input.length - 1]).includes(SUMMARIZATION_PROMPT.slice(0, 30)),
    );
    expect(summarizerRequests).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    warn.mockRestore();
  });
});
