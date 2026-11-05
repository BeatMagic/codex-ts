/**
 * mirrors codex-rs/core/tests/suite/tool_harness.rs (goal-related tests)
 * and ext/goal/src/ unit tests.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { CodexThread } from "../../src/codex_thread.js";
import { GoalToolExecutor } from "../../../ext/goal/src/tool.js";
import { ToolRouter } from "../../src/tools/router.js";
import type { CustomTool } from "../../src/tools/router.js";
import type { EventMsg } from "../../../protocol/src/protocol.js";
import {
  waitForEvent,
  waitForEventMatch,
  sseFlat,
  evFunctionCall,
  evAssistantMessage,
  evCompleted,
  evResponseCreated,
  makeSseResponse,
} from "../common/lib.js";

// ─── GoalToolExecutor unit tests ──────────────────────────────────────────────

describe("GoalToolExecutor", () => {
  let ex: GoalToolExecutor;
  beforeEach(() => { ex = new GoalToolExecutor("thread-1"); });

  it("get returns null goal when none exists", async () => {
    const r = JSON.parse((await ex.get()).output) as { goal: null };
    expect(r.goal).toBeNull();
  });

  it("create sets goal to Active", async () => {
    const r = JSON.parse((await ex.create("build it")).output) as {
      goal: { objective: string; status: string };
    };
    expect(r.goal.status).toBe("Active");
    expect(r.goal.objective).toBe("build it");
  });

  it("create rejects a second goal while the first is unfinished", async () => {
    await ex.create("first");
    const r = JSON.parse((await ex.create("second")).output) as { error: string };
    expect(r.error).toMatch(/unfinished goal/);
  });

  // mirrors create_thread_goal's `DO UPDATE … WHERE status = 'complete'`
  // (goals.rs:236): a finished goal is replaced and its counters reset.
  it("create replaces a completed goal", async () => {
    await ex.create("first", 500);
    await ex.recordTokens(300);
    await ex.update("complete");

    const r = JSON.parse((await ex.create("second")).output) as {
      goal: { objective: string; status: string; tokens_used: number };
    };
    expect(r.goal.objective).toBe("second");
    expect(r.goal.status).toBe("Active");
    expect(r.goal.tokens_used).toBe(0);
  });

  // mirrors validate_thread_goal_objective (protocol.rs:4055)
  it("create rejects an over-long objective", async () => {
    const r = JSON.parse((await ex.create("x".repeat(4_001))).output) as {
      error: string;
    };
    expect(r.error).toMatch(/at most 4000 characters/);
  });

  it("create stores token_budget", async () => {
    const r = JSON.parse((await ex.create("work", 500)).output) as {
      remaining_tokens: number;
    };
    expect(r.remaining_tokens).toBe(500);
  });

  it("update marks goal Complete", async () => {
    await ex.create("finish");
    const r = JSON.parse((await ex.update("complete")).output) as {
      goal: { status: string };
    };
    expect(r.goal.status).toBe("Complete");
  });

  it("update marks goal Blocked", async () => {
    await ex.create("finish");
    const r = JSON.parse((await ex.update("blocked")).output) as {
      goal: { status: string };
    };
    expect(r.goal.status).toBe("Blocked");
  });

  it("update fails with no goal", async () => {
    const r = JSON.parse((await ex.update("complete")).output) as { error: string };
    expect(r.error).toMatch(/No active goal/);
  });

  it("recordTokens reduces remaining budget", async () => {
    await ex.create("work", 1000);
    await ex.recordTokens(300);
    await ex.recordTokens(200);
    const r = JSON.parse((await ex.get()).output) as {
      goal: { tokens_used: number };
      remaining_tokens: number;
    };
    expect(r.goal.tokens_used).toBe(500);
    expect(r.remaining_tokens).toBe(500);
  });

  it("create emits a ThreadGoalUpdatedEvent", async () => {
    const { event } = await ex.create("do something");
    expect(event).not.toBeNull();
    expect(event?.goal.status).toBe("Active");
  });
});

// ─── Integration: create_goal tool call through CodexThread ───────────────────

describe("create_goal tool call", () => {
  beforeEach(() => { vi.unstubAllGlobals(); });

  it("emits ThreadGoalUpdated event and TurnComplete", async () => {
    const fetchMock = vi.fn();

    // Turn 1: model calls create_goal
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("resp-1"),
          evFunctionCall("call-1", "create_goal", { objective: "test goal" }),
          evCompleted("resp-1"),
        ]),
      ),
    );
    // Turn 2: model replies after receiving tool output
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("resp-2"),
          evAssistantMessage("Goal created."),
          evCompleted("resp-2"),
        ]),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({ apiKey: "test", model: "gpt-4o" });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "create a goal" }] });

    const goalUpdated = await waitForEventMatch(codex, (msg) =>
      msg.type === "ThreadGoalUpdated" ? msg.event : null,
    );
    expect(goalUpdated.goal.objective).toBe("test goal");
    expect(goalUpdated.goal.status).toBe("Active");

    const done = await waitForEvent(
      codex,
      (msg): msg is Extract<EventMsg, { type: "TurnComplete" }> => msg.type === "TurnComplete",
    );
    expect(done.type).toBe("TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});


// ─── Goal tools are opt-out (CodexThreadConfig.enableGoalTools) ───────────────
// rs registers them only when the goal extension is enabled
// (ext/goal/src/extension.rs goals_enabled). A host that never sets goals
// should not advertise three tool definitions on every request.

describe("enableGoalTools: false", () => {
  beforeEach(() => { vi.unstubAllGlobals(); });

  const GOAL_TOOLS = ["get_goal", "create_goal", "update_goal"];

  it("drops only the goal tools from the advertised specs", () => {
    const names = (router: ToolRouter) => router.toolSpecs().map((spec) => spec.tool.name);
    const enabled = names(new ToolRouter(new GoalToolExecutor("t")));
    const disabled = names(
      new ToolRouter(new GoalToolExecutor("t"), [], { enableGoalTools: false }),
    );
    expect(enabled).toEqual(expect.arrayContaining(GOAL_TOOLS));
    expect(disabled).toEqual(enabled.filter((name) => !GOAL_TOOLS.includes(name)));
    expect(disabled).toEqual(expect.arrayContaining(["request_user_input", "update_plan"]));
  });

  it("answers a goal-tool call with the unknown-tool roster", async () => {
    const router = new ToolRouter(new GoalToolExecutor("t"), [], {
      enableGoalTools: false,
    });
    const emitEvent = vi.fn();
    const out = JSON.parse(
      (await router.dispatch("create_goal", "call-1", { objective: "x" }, {
        turnId: "t",
        pendingInputs: new Map(),
        emitEvent,
      })) as string,
    ) as { error: string; known_tools: string[] };
    expect(out.error).toBe("Unknown tool: create_goal");
    expect(out.known_tools).toEqual(router.toolSpecs().map((spec) => spec.tool.name));
    expect(emitEvent).not.toHaveBeenCalled();
  });

  it("a host tool may take a goal tool's name once the built-in is off", async () => {
    const custom: CustomTool = {
      name: "get_goal",
      spec: () => ({
        type: "function",
        tool: {
          name: "get_goal",
          description: "host goal",
          parameters: { type: "object", properties: {}, additionalProperties: false },
          strict: false,
        },
      }),
      execute: async () => "host answer",
    };
    const router = new ToolRouter(new GoalToolExecutor("t"), [custom], {
      enableGoalTools: false,
    });
    const out = await router.dispatch("get_goal", "call-1", {}, {
      turnId: "t",
      pendingInputs: new Map(),
      emitEvent: () => undefined,
    });
    expect(out).toBe("host answer");
  });

  it("CodexThread wires it through to the request and to dispatch", async () => {
    const fetchMock = vi.fn();
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("r1"),
          evFunctionCall("call-1", "get_goal", {}),
          evCompleted("r1"),
        ]),
      ),
    );
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([evResponseCreated("r2"), evAssistantMessage("ok"), evCompleted("r2")]),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      enableGoalTools: false,
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");

    const body = (index: number) =>
      JSON.parse((fetchMock.mock.calls[index]![1] as RequestInit).body as string) as {
        tools: { name: string }[];
        input: { type: string; output?: string }[];
      };
    const advertised = body(0).tools.map((tool) => tool.name);
    for (const name of GOAL_TOOLS) expect(advertised).not.toContain(name);
    expect(advertised).toEqual(expect.arrayContaining(["request_user_input", "update_plan"]));
    const output = body(1).input.find((item) => item.type === "function_call_output");
    expect(output?.output).toContain("Unknown tool: get_goal");
  });
});
