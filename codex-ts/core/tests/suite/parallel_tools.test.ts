/**
 * Parallel execution of parallel-safe host tools.
 *
 * mirrors codex-rs/core/src/tools/parallel.rs (the RwLock every dispatch goes
 * through: shared for a tool that supports parallel calls, exclusive
 * otherwise; #54654 made dynamic tools parallel) and turn.rs's FuturesOrdered
 * (outputs recorded in call order, whatever order they complete in).
 * codex-ts makes it opt-in per tool: CustomTool.supportsParallel.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import { ToolRouter } from "../../src/tools/router.js";
import type { CustomTool } from "../../src/tools/router.js";
import { GoalToolExecutor } from "../../../ext/goal/src/tool.js";
import { InMemoryIoBackend } from "../../../thread-store/src/io_backend.js";
import { LiveThread } from "../../../thread-store/src/live_thread.js";
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

const ABORTED_AFTER = /^aborted by user after \d+\.\ds$/u;
const NOT_EXECUTED =
  "aborted by user before this call started; it was not executed";

const tick = () => new Promise((resolve) => setTimeout(resolve, 10));

type Body = { input: ConversationItem[] };

/** A controllable tool: logs start/end, finishes when the test says so. */
function gatedTool(name: string, supportsParallel: boolean, log: string[]) {
  const finishers: ((value: string) => void)[] = [];
  const signals: AbortSignal[] = [];
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
    supportsParallel,
    execute: (args, ctx) => {
      const label = (args as { label?: string }).label ?? name;
      log.push(`${label}:start`);
      signals.push(ctx.signal);
      return new Promise<string>((resolve) => {
        finishers.push((value) => {
          log.push(`${label}:end`);
          resolve(value);
        });
      });
    },
  };
  return { tool, finishers, signals };
}

/** A tool that settles by itself. */
function instantTool(
  name: string,
  supportsParallel: boolean,
  log: string[],
  behaviour: (label: string) => Promise<string> = async (label) => `${label} ok`,
): CustomTool {
  return {
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
    supportsParallel,
    execute: async (args) => {
      const label = (args as { label?: string }).label ?? name;
      log.push(`${label}:start`);
      try {
        return await behaviour(label);
      } finally {
        log.push(`${label}:end`);
      }
    },
  };
}

/** First request → `calls`; every later one → a plain reply. */
function scriptedFetch(calls: object[]) {
  const bodies: Body[] = [];
  const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
    bodies.push(JSON.parse(init.body as string) as Body);
    const events = bodies.length === 1 ? calls : [evAssistantMessage("ok")];
    return makeSseResponse(sseFlat([evResponseCreated("r"), ...events, evCompleted("r")]));
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, bodies };
}

function outputs(items: ConversationItem[]): [string, unknown][] {
  return items
    .filter(
      (item): item is Extract<ConversationItem, { type: "function_call_output" }> =>
        item.type === "function_call_output",
    )
    .map((item) => [item.call_id, item.output]);
}

async function persisted(io: InMemoryIoBackend, threadId: string) {
  return new LiveThread(threadId, new LocalThreadStore(io)).loadConversationHistory();
}

describe("parallel-safe host tools", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("run concurrently with each other", async () => {
    const log: string[] = [];
    let active = 0;
    let maxActive = 0;
    let bothStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      bothStarted = resolve;
    });
    const overlapping = instantTool("read_a", true, log, async (label) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      if (active === 2) bothStarted();
      // Each call waits for the other to start (bounded, so a serial run
      // finishes too — and fails the assertion instead of hanging).
      await Promise.race([started, new Promise((resolve) => setTimeout(resolve, 300))]);
      active -= 1;
      return `${label} ok`;
    });
    scriptedFetch([
      evFunctionCall("c1", "read_a", { label: "a1" }),
      evFunctionCall("c2", "read_a", { label: "a2" }),
    ]);
    const codex = new CodexThread({ apiKey: "t", model: "m", customTools: [overlapping] });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    expect(maxActive).toBe(2);
    expect(log.slice(0, 2)).toEqual(["a1:start", "a2:start"]);
  });

  it("an exclusive call is a barrier both ways", async () => {
    const log: string[] = [];
    const parallel = gatedTool("read", true, log);
    const exclusive = gatedTool("write", false, log);
    scriptedFetch([
      evFunctionCall("c1", "read", { label: "p1" }),
      evFunctionCall("c2", "write", { label: "e2" }),
      evFunctionCall("c3", "read", { label: "p3" }),
    ]);
    const codex = new CodexThread({
      apiKey: "t",
      model: "m",
      customTools: [parallel.tool, exclusive.tool],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });

    await vi.waitFor(() => expect(log).toEqual(["p1:start"]));
    await tick();
    // The exclusive call waits for the parallel one started before it…
    expect(log).toEqual(["p1:start"]);
    parallel.finishers[0]!("p1 done");
    await vi.waitFor(() => expect(log).toEqual(["p1:start", "p1:end", "e2:start"]));
    await tick();
    // …and the parallel call after it waits for the exclusive one.
    expect(log).toEqual(["p1:start", "p1:end", "e2:start"]);
    exclusive.finishers[0]!("e2 done");
    await vi.waitFor(() => expect(log).toContain("p3:start"));
    parallel.finishers[1]!("p3 done");
    await waitForEvent(codex, (m) => m.type === "TurnComplete");
    expect(log).toEqual(["p1:start", "p1:end", "e2:start", "e2:end", "p3:start", "p3:end"]);
  });

  it("records outputs in call order when they complete out of order, live and on disk", async () => {
    const log: string[] = [];
    const slow = gatedTool("slow_read", true, log);
    const fast = instantTool("fast_read", true, log);
    const { bodies } = scriptedFetch([
      evFunctionCall("c1", "slow_read", {}),
      evFunctionCall("c2", "fast_read", {}),
    ]);
    const io = new InMemoryIoBackend();
    const codex = new CodexThread({
      apiKey: "t",
      model: "m",
      ioBackend: io,
      customTools: [slow.tool, fast],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await vi.waitFor(() => expect(log).toContain("fast_read:end"));
    slow.finishers[0]!("slow done");
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    expect(log).toEqual(["slow_read:start", "fast_read:start", "fast_read:end", "slow_read:end"]);
    const expected = [
      ["c1", "slow done"],
      ["c2", "fast_read ok"],
    ];
    expect(outputs(bodies[1]!.input)).toEqual(expected);
    expect(outputs(await persisted(io, codex.id))).toEqual(expected);
  });

  it("one parallel call throwing does not disturb the others", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const log: string[] = [];
    const ok = instantTool("ok_read", true, log);
    const broken = instantTool("broken_read", true, log, async () => {
      throw new Error("upstream 502");
    });
    const { bodies } = scriptedFetch([
      evFunctionCall("c1", "ok_read", { label: "x1" }),
      evFunctionCall("c2", "broken_read", {}),
      evFunctionCall("c3", "ok_read", { label: "x3" }),
    ]);
    const codex = new CodexThread({ apiKey: "t", model: "m", customTools: [ok, broken] });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    const done = await waitForEvent(
      codex,
      (m) => m.type === "TurnComplete" || m.type === "TurnAborted",
    );
    expect(done.type === "TurnComplete" && done.event.error).toBeUndefined();
    expect(outputs(bodies[1]!.input)).toEqual([
      ["c1", "x1 ok"],
      ["c2", "upstream 502"],
      ["c3", "x3 ok"],
    ]);
    // All three started before any finished: they really ran together.
    expect(log.slice(0, 3)).toEqual(["x1:start", "broken_read:start", "x3:start"]);
    vi.restoreAllMocks();
  });

  it("an interrupt mid-batch: running → aborted, finished → kept, unstarted → not executed, in call order", async () => {
    const log: string[] = [];
    const slow = gatedTool("slow_read", true, log);
    const fast = instantTool("fast_read", true, log);
    const exclusive = vi.fn(async () => "must not run");
    const { bodies } = scriptedFetch([
      evFunctionCall("c1", "slow_read", {}),
      evFunctionCall("c2", "fast_read", {}),
      evFunctionCall("c3", "write", {}),
      evFunctionCall("c4", "fast_read", { label: "late" }),
    ]);
    const io = new InMemoryIoBackend();
    const codex = new CodexThread({
      apiKey: "t",
      model: "m",
      ioBackend: io,
      customTools: [
        slow.tool,
        fast,
        {
          name: "write",
          spec: fast.spec,
          execute: exclusive,
        },
      ],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await vi.waitFor(() => expect(log).toContain("fast_read:end"));

    await codex.submit({ type: "Interrupt" });
    await waitForEvent(codex, (m) => m.type === "TurnAborted");
    expect(slow.signals[0]!.aborted).toBe(true);
    slow.finishers[0]!("late result that must be discarded");

    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "next" }] });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    const expected = [
      ["c1", expect.stringMatching(ABORTED_AFTER)],
      ["c2", "fast_read ok"],
      ["c3", NOT_EXECUTED],
      ["c4", NOT_EXECUTED],
    ];
    expect(outputs(bodies[1]!.input)).toEqual(expected);
    expect(outputs(await persisted(io, codex.id))).toEqual(expected);
    expect(exclusive).not.toHaveBeenCalled();
    expect(log).not.toContain("late:start");
  });

  it("a call whose arguments fail to parse is exclusive too", async () => {
    const log: string[] = [];
    const slow = gatedTool("slow_read", true, log);
    const fast = instantTool("fast_read", true, log);
    const { bodies } = scriptedFetch([
      evFunctionCall("c1", "slow_read", {}),
      {
        type: "response.output_item.done",
        item: { type: "function_call", call_id: "c2", name: "fast_read", arguments: "{not json" },
      },
      evFunctionCall("c3", "fast_read", { label: "after" }),
    ]);
    const codex = new CodexThread({
      apiKey: "t",
      model: "m",
      customTools: [slow.tool, fast],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await vi.waitFor(() => expect(log).toEqual(["slow_read:start"]));
    await tick();
    expect(log).toEqual(["slow_read:start"]);
    slow.finishers[0]!("slow done");
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    expect(log).toEqual(["slow_read:start", "slow_read:end", "after:start", "after:end"]);
    expect(outputs(bodies[1]!.input).map(([callId]) => callId)).toEqual(["c1", "c2", "c3"]);
    expect(String(outputs(bodies[1]!.input)[1]![1])).toContain("failed to parse function arguments");
  });
});

describe("ToolRouter.supportsParallel", () => {
  const tool = (name: string, supportsParallel?: boolean): CustomTool => ({
    name,
    spec: () => ({
      type: "function",
      tool: { name, description: "", parameters: { type: "object", properties: {} }, strict: false },
    }),
    execute: async () => "",
    ...(supportsParallel !== undefined ? { supportsParallel } : {}),
  });

  it("only a custom tool that opted in; built-ins and unknown names are exclusive", () => {
    const router = new ToolRouter(new GoalToolExecutor("g"), [
      tool("read", true),
      tool("write"),
      tool("request_user_input", true),
      tool("get_goal", true),
    ]);
    expect(router.supportsParallel("read")).toBe(true);
    expect(router.supportsParallel("write")).toBe(false);
    expect(router.supportsParallel("nope")).toBe(false);
    // Shadowed by the built-in of the same name, which is what runs.
    expect(router.supportsParallel("request_user_input")).toBe(false);
    expect(router.supportsParallel("update_plan")).toBe(false);
    expect(router.supportsParallel("get_goal")).toBe(false);
    // With goal tools off the name is an ordinary custom tool.
    const noGoals = new ToolRouter(new GoalToolExecutor("g"), [tool("get_goal", true)], {
      enableGoalTools: false,
    });
    expect(noGoals.supportsParallel("get_goal")).toBe(true);
  });
});
