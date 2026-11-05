/**
 * Regression for the 2026-07-18 duplicate-tool_result incident
 * (openrouter-responses-errors/2026-07-18/a1cf9c62):
 *
 * A turn is interrupted while a tool call is mid-dispatch. The tool's promise
 * cannot be cancelled and resolves later — after the user has already started
 * the next turn. The original bug: the zombie dispatch loop then pushed the
 * REAL output into history next to the "aborted" one the next turn had
 * synthesized, so the same call_id carried TWO outputs → Anthropic 400 "each
 * tool_use must have a single result" on every subsequent request (the
 * duplicate is persisted → the thread is permanently bricked).
 *
 * Now (mirrors tools/parallel.rs + #51556): the interrupt reaches the tool
 * through ctx.signal and ends the turn at once; the in-flight call is recorded
 * — and persisted — as "aborted by user after Xs", calls of the round that
 * never started as not executed, and the late real result is discarded. Every
 * call stays paired with exactly one output, live and on disk.
 *
 * Also here: a throwing host tool no longer kills the turn (parallel.rs
 * failure_response), and Interrupt reaches every in-flight turn.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import type { CustomTool, CustomToolContext } from "../../src/tools/router.js";
import { InMemoryIoBackend } from "../../../thread-store/src/io_backend.js";
import { LiveThread } from "../../../thread-store/src/live_thread.js";
import { LocalThreadStore } from "../../../thread-store/src/local_thread_store.js";
import type { ConversationItem } from "../../../thread-store/src/types.js";
import type { EventMsg } from "../../../protocol/src/protocol.js";
import {
  evAssistantMessage,
  evCompleted,
  evFunctionCall,
  evResponseCreated,
  makeSseResponse,
  sseFlat,
  waitForEvent,
} from "../common/lib.js";

const ABORTED_AFTER = /^aborted by user after \d+\.\ds$/u;
const NOT_EXECUTED =
  "aborted by user before this call started; it was not executed";

function toolSpec(name: string): CustomTool["spec"] {
  return () => ({
    type: "function",
    tool: {
      name,
      description: `${name}.`,
      parameters: { type: "object", properties: {}, additionalProperties: false },
      strict: false,
    },
  });
}

/** A tool that blocks until the test calls finish(); records its ctx. */
function blockingTool(name: string) {
  let started!: () => void;
  const startedPromise = new Promise<void>((resolve) => {
    started = resolve;
  });
  let finish!: (value: string) => void;
  const gate = new Promise<string>((resolve) => {
    finish = resolve;
  });
  const contexts: CustomToolContext[] = [];
  const tool: CustomTool = {
    name,
    spec: toolSpec(name),
    execute: async (_args, ctx) => {
      contexts.push(ctx);
      started();
      return gate;
    },
  };
  return { tool, started: startedPromise, finish, contexts };
}

type Body = { input: ConversationItem[] };

/** First request → `firstRound`; every later one → a plain reply. */
function scriptedFetch(firstRound: object[]) {
  const bodies: Body[] = [];
  const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
    bodies.push(JSON.parse(init.body as string) as Body);
    if (bodies.length === 1) {
      return makeSseResponse(sseFlat([evResponseCreated("r1"), ...firstRound, evCompleted("r1")]));
    }
    return makeSseResponse(
      sseFlat([
        evResponseCreated(`r${bodies.length}`),
        evAssistantMessage("next turn ok"),
        evCompleted(`r${bodies.length}`),
      ]),
    );
  });
  return { fetchMock, bodies };
}

function outputsFor(items: ConversationItem[], callId: string) {
  return items.filter(
    (item): item is Extract<ConversationItem, { type: "function_call_output" }> =>
      item.type === "function_call_output" && item.call_id === callId,
  );
}

async function persistedConversation(io: InMemoryIoBackend, threadId: string) {
  return new LiveThread(threadId, new LocalThreadStore(io)).loadConversationHistory();
}

describe("interrupt while a tool call is mid-dispatch", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("ends the turn while the tool still runs; exactly one (aborted) output, live and persisted", async () => {
    // slow_tool blocks until the test resolves it — models a tool still
    // running when the user hits stop (e.g. a generation wait).
    const slow = blockingTool("slow_tool");
    const { fetchMock, bodies } = scriptedFetch([evFunctionCall("call-slow", "slow_tool", {})]);
    vi.stubGlobal("fetch", fetchMock);

    const io = new InMemoryIoBackend();
    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      customTools: [slow.tool],
      ioBackend: io,
    });

    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "run the slow tool" }],
    });
    // Wait until the dispatch loop is parked inside slow_tool.execute.
    await slow.started;

    await codex.submit({ type: "Interrupt" });

    // The turn ends WITHOUT waiting for the tool: TurnAborted arrives while
    // the tool's promise is still pending, and the tool was told via ctx.signal.
    await waitForEvent(codex, (m) => m.type === "TurnAborted");
    expect(slow.contexts[0]!.signal.aborted).toBe(true);

    // The tool finishes AFTER the interrupt — its result must be discarded.
    slow.finish("real result that must be discarded");

    // Next user turn: the request input must contain exactly ONE output for
    // call-slow — the recorded "aborted by user after Xs" one, right after
    // the call.
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "continue" }],
    });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    const input = bodies[1]!.input;
    const outputs = outputsFor(input, "call-slow");
    expect(outputs).toHaveLength(1);
    expect(outputs[0]!.output).toMatch(ABORTED_AFTER);
    expect(JSON.stringify(input)).not.toContain("real result that must be discarded");
    const callIdx = input.findIndex(
      (i) => i.type === "function_call" && i.call_id === "call-slow",
    );
    expect(input[callIdx + 1]).toEqual(outputs[0]);

    // Persisted too (it used to be left unpaired on disk): a resume finds the
    // same single output.
    const persisted = outputsFor(await persistedConversation(io, codex.id), "call-slow");
    expect(persisted).toHaveLength(1);
    expect(persisted[0]!.output).toBe(outputs[0]!.output);
  });

  it("records every call of the round that never started as not executed, and never starts it", async () => {
    const slow = blockingTool("slow_tool");
    const neverRun = vi.fn(async () => "must not run");
    const { fetchMock, bodies } = scriptedFetch([
      evFunctionCall("call-1", "slow_tool", {}),
      evFunctionCall("call-2", "other_tool", {}),
      evFunctionCall("call-3", "other_tool", {}),
    ]);
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      customTools: [slow.tool, { name: "other_tool", spec: toolSpec("other_tool"), execute: neverRun }],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await slow.started;
    await codex.submit({ type: "Interrupt" });
    await waitForEvent(codex, (m) => m.type === "TurnAborted");
    slow.finish("late");

    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "continue" }] });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    const input = bodies[1]!.input;
    expect(outputsFor(input, "call-1").map((o) => o.output)).toEqual([
      expect.stringMatching(ABORTED_AFTER),
    ]);
    expect(outputsFor(input, "call-2").map((o) => o.output)).toEqual([NOT_EXECUTED]);
    expect(outputsFor(input, "call-3").map((o) => o.output)).toEqual([NOT_EXECUTED]);
    expect(neverRun).not.toHaveBeenCalled();
  });

  it("an interrupt that lands before dispatch records the whole round as not executed", async () => {
    // Park the turn on persisting the function_call lines, i.e. after the
    // response streamed but before the first tool starts.
    let releaseWrite!: () => void;
    const writeGate = new Promise<void>((resolve) => {
      releaseWrite = resolve;
    });
    let parkedOnWrite!: () => void;
    const parked = new Promise<void>((resolve) => {
      parkedOnWrite = resolve;
    });
    class ParkingIo extends InMemoryIoBackend {
      override async appendLine(threadId: string, line: string): Promise<void> {
        if (line.includes('"type":"function_call"')) {
          parkedOnWrite();
          await writeGate;
        }
        await super.appendLine(threadId, line);
      }
    }
    const tool = vi.fn(async () => "must not run");
    const { fetchMock } = scriptedFetch([
      evFunctionCall("call-1", "a_tool", {}),
      evFunctionCall("call-2", "a_tool", {}),
    ]);
    vi.stubGlobal("fetch", fetchMock);

    const io = new ParkingIo();
    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      customTools: [{ name: "a_tool", spec: toolSpec("a_tool"), execute: tool }],
      ioBackend: io,
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await parked;
    await codex.submit({ type: "Interrupt" });
    releaseWrite();
    await waitForEvent(codex, (m) => m.type === "TurnAborted");

    expect(tool).not.toHaveBeenCalled();
    await new Promise((resolve) => setTimeout(resolve, 10));
    const persisted = await persistedConversation(io, codex.id);
    expect(outputsFor(persisted, "call-1").map((o) => o.output)).toEqual([NOT_EXECUTED]);
    expect(outputsFor(persisted, "call-2").map((o) => o.output)).toEqual([NOT_EXECUTED]);
  });

  it("TurnAborted does not wait on the host's write of the aborted outputs", async () => {
    let releaseWrite!: () => void;
    const writeGate = new Promise<void>((resolve) => {
      releaseWrite = resolve;
    });
    class SlowOutputIo extends InMemoryIoBackend {
      override async appendLine(threadId: string, line: string): Promise<void> {
        if (line.includes('"type":"function_call_output"')) await writeGate;
        await super.appendLine(threadId, line);
      }
    }
    const slow = blockingTool("slow_tool");
    const { fetchMock } = scriptedFetch([evFunctionCall("call-slow", "slow_tool", {})]);
    vi.stubGlobal("fetch", fetchMock);
    const io = new SlowOutputIo();
    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      customTools: [slow.tool],
      ioBackend: io,
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await slow.started;
    await codex.submit({ type: "Interrupt" });

    // Arrives while the write of the aborted output is still blocked.
    await waitForEvent(codex, (m) => m.type === "TurnAborted");
    expect(outputsFor(await persistedConversation(io, codex.id), "call-slow")).toEqual([]);

    releaseWrite();
    await vi.waitFor(async () =>
      expect(
        outputsFor(await persistedConversation(io, codex.id), "call-slow").map((o) => o.output),
      ).toEqual([expect.stringMatching(ABORTED_AFTER)]),
    );
    slow.finish("late");
  });

  it("reaches every in-flight turn, not just the newest", async () => {
    const first = blockingTool("first_tool");
    const second = blockingTool("second_tool");
    let request = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        request += 1;
        const call =
          request === 1
            ? evFunctionCall("call-a", "first_tool", {})
            : evFunctionCall("call-b", "second_tool", {});
        return makeSseResponse(sseFlat([evResponseCreated(`r${request}`), call, evCompleted(`r${request}`)]));
      }),
    );
    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      customTools: [first.tool, second.tool],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "a" }] });
    await first.started;
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "b" }] });
    await second.started;

    await codex.submit({ type: "Interrupt" });
    const aborted = new Set<string>();
    while (aborted.size < 2) {
      const msg = await waitForEvent(codex, (m) => m.type === "TurnAborted");
      if (msg.type === "TurnAborted") aborted.add(msg.event.turn_id ?? "");
    }
    expect(first.contexts[0]!.signal.aborted).toBe(true);
    expect(second.contexts[0]!.signal.aborted).toBe(true);
    first.finish("late");
    second.finish("late");
  });
});

describe("a throwing host tool", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("reports the error to the model as the call's output and the turn carries on", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { fetchMock, bodies } = scriptedFetch([evFunctionCall("call-x", "flaky_tool", {})]);
    vi.stubGlobal("fetch", fetchMock);
    const io = new InMemoryIoBackend();
    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      ioBackend: io,
      customTools: [
        {
          name: "flaky_tool",
          spec: toolSpec("flaky_tool"),
          execute: async () => {
            throw new Error("upstream returned 502");
          },
        },
      ],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });

    const seen: EventMsg[] = [];
    const done = await waitForEvent(codex, (m) => {
      seen.push(m);
      return m.type === "TurnComplete" || m.type === "TurnAborted";
    });
    expect(done.type === "TurnComplete" && done.event.error).toBeUndefined();
    expect(seen.some((m) => m.type === "Error")).toBe(false);

    // The model saw the failure and got to respond to it.
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(outputsFor(bodies[1]!.input, "call-x").map((o) => o.output)).toEqual([
      "upstream returned 502",
    ]);
    // …and it is persisted, so a resume does not re-run the call.
    expect(
      outputsFor(await persistedConversation(io, codex.id), "call-x").map((o) => o.output),
    ).toEqual(["upstream returned 502"]);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("an AbortError from the tool's own work is a failure, not an interrupt", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { fetchMock, bodies } = scriptedFetch([evFunctionCall("call-t", "timeout_tool", {})]);
    vi.stubGlobal("fetch", fetchMock);
    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      customTools: [
        {
          name: "timeout_tool",
          spec: toolSpec("timeout_tool"),
          execute: async () => {
            throw new DOMException("The operation timed out.", "AbortError");
          },
        },
      ],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    const done = await waitForEvent(
      codex,
      (m) => m.type === "TurnComplete" || m.type === "TurnAborted",
    );
    expect(done.type).toBe("TurnComplete");
    expect(outputsFor(bodies[1]!.input, "call-t").map((o) => o.output)).toEqual([
      "The operation timed out.",
    ]);
    vi.restoreAllMocks();
  });
});
