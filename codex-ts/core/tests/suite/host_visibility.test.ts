/**
 * Three places where the host could not see what the thread was doing, each one
 * traced to a real user-visible failure:
 *
 *  - a tool call is invisible until its arguments finish streaming, so a long
 *    `doc.create` renders as a blank spinner for seconds;
 *  - an unknown tool name comes back as `{error: "Unknown tool: x"}` (the one
 *    fact the model already had), and the model concludes the CAPABILITY is
 *    missing rather than the name being wrong;
 *  - `submit({type:"UserInputAnswer"})` is a silent no-op once the resolver is
 *    gone (any reload), so the host reports "delivered" and hangs.
 */

import { describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import { ToolRouter } from "../../src/tools/router.js";
import { GoalToolExecutor } from "../../../ext/goal/src/tool.js";
import type { CustomTool } from "../../src/tools/router.js";
import type { EventMsg } from "../../../protocol/src/protocol.js";
import {
  evAssistantMessage,
  evCompleted,
  evFunctionCall,
  evResponseCreated,
  makeSseResponse,
  sse,
  sseFlat,
  waitForEvent,
} from "../common/lib.js";

function echoTool(name: string): CustomTool {
  return {
    name,
    spec: () => ({
      type: "function",
      tool: {
        name,
        description: "echo",
        parameters: { type: "object", properties: {}, additionalProperties: true },
        strict: false,
      },
    }),
    execute: async () => "ok",
  };
}

// =============================================================================
// Function-call composition events
// =============================================================================

describe("streaming function-call events", () => {
  it("announces the call at output_item.added and forwards each argument delta", async () => {
    const seen: EventMsg[] = [];
    const argsStr = '{"path":"a.md"}';
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string) as { input: unknown[] };
      // Round 2 (after the tool ran) ends the turn.
      if (body.input.some((i) => (i as { type?: string }).type === "function_call_output")) {
        return makeSseResponse(
          sseFlat([evResponseCreated("r2"), evAssistantMessage("done"), evCompleted("r2")]),
        );
      }
      return makeSseResponse(
        sse([
          evResponseCreated("r1"),
          {
            type: "response.output_item.added",
            item: { type: "function_call", id: "fc-1", call_id: "call-1", name: "doc_create" },
          },
          { type: "response.function_call_arguments.delta", item_id: "fc-1", delta: '{"path":' },
          { type: "response.function_call_arguments.delta", item_id: "fc-1", delta: '"a.md"}' },
          {
            type: "response.output_item.done",
            item: { type: "function_call", call_id: "call-1", name: "doc_create", arguments: argsStr },
          },
          evCompleted("r1"),
        ]),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      customTools: [echoTool("doc_create")],
    });
    void (async () => {
      for (;;) {
        const event = await codex.nextEvent();
        seen.push(event.msg);
        if (event.msg.type === "TurnComplete") return;
      }
    })();
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await vi.waitFor(() => expect(seen.some((m) => m.type === "TurnComplete")).toBe(true), {
      timeout: 4000,
    });

    const start = seen.find((m) => m.type === "FunctionCallStreamStart");
    expect(start).toBeDefined();
    expect(start?.type === "FunctionCallStreamStart" && start.event.name).toBe("doc_create");

    const deltas = seen.filter((m) => m.type === "FunctionCallStreamArgs");
    expect(deltas).toHaveLength(2);
    const joined = deltas
      .map((m) => (m.type === "FunctionCallStreamArgs" ? m.event.delta : ""))
      .join("");
    // Display-only, but it must still reconstruct what the model wrote.
    expect(joined).toBe(argsStr);
    vi.unstubAllGlobals();
  });
});

// =============================================================================
// Unknown tool
// =============================================================================

describe("unknown tool error", () => {
  it("carries the full roster and names the likelier cause", async () => {
    const router = new ToolRouter(new GoalToolExecutor("t1"), [echoTool("place_image")]);
    const out = await router.dispatch("minimax-h3", "call-1", {}, {
      turnId: "t1",
      emitEvent: () => undefined,
    } as never);
    const parsed = JSON.parse(out as string) as {
      error: string;
      known_tools: string[];
      hint: string;
    };

    expect(parsed.error).toContain("minimax-h3");
    // The roster is exactly what the model was advertised.
    expect(parsed.known_tools).toEqual(router.toolSpecs().map((s) => s.tool.name));
    expect(parsed.known_tools).toContain("place_image");
    // The disambiguation is the whole point: wrong NAME, not missing capability.
    expect(parsed.hint).toContain("WRONG NAME");
    expect(parsed.hint).toContain("do not revise or disown");
  });

  it("NON-INVERSION: a registered tool still dispatches", async () => {
    const router = new ToolRouter(new GoalToolExecutor("t1"), [echoTool("place_image")]);
    const out = await router.dispatch("place_image", "call-1", {}, {
      turnId: "t1",
      emitEvent: () => undefined,
    } as never);
    expect(out).toBe("ok");
  });
});

// =============================================================================
// Pending-input visibility
// =============================================================================

describe("hasPendingInput", () => {
  it("is false for a thread that has never parked on request_user_input", () => {
    const codex = new CodexThread({ apiKey: "test", model: "gpt-4o" });
    expect(codex.hasPendingInput("any-turn")).toBe(false);
  });

  it("is true only while a turn is actually waiting, and false again after the answer", async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string) as { input: unknown[] };
      if (body.input.some((i) => (i as { type?: string }).type === "function_call_output")) {
        return makeSseResponse(
          sseFlat([evResponseCreated("r2"), evAssistantMessage("done"), evCompleted("r2")]),
        );
      }
      return makeSseResponse(
        sseFlat([
          evResponseCreated("r1"),
          evFunctionCall("call-ask", "request_user_input", {
            questions: [
              {
                id: "q1",
                header: "Confirm",
                question: "Proceed?",
                options: [
                  { label: "Yes", description: "Continue." },
                  { label: "No", description: "Abort." },
                ],
              },
            ],
          }),
          evCompleted("r1"),
        ]),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({ apiKey: "test", model: "gpt-4o" });
    const submissionId = await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "ask me" }],
    });
    const asked = await waitForEvent(codex, (m) => m.type === "RequestUserInput", 4000);
    const turnId =
      asked.type === "RequestUserInput" ? asked.event.turn_id : "";

    // Parked: the host may honestly claim an answer will be delivered.
    expect(codex.hasPendingInput(turnId)).toBe(true);
    // 鈥nd only for THIS turn.
    expect(codex.hasPendingInput(`${turnId}-other`)).toBe(false);

    // Keyed by turn_id 鈥?mirrors Op::UserInputAnswer { id: request.turn_id }.
    await codex.submit({
      type: "UserInputAnswer",
      id: turnId,
      response: { answers: { q1: { answers: ["yes"] } } },
    });
    await waitForEvent(codex, (m) => m.type === "TurnComplete", 4000);
    expect(codex.hasPendingInput(turnId)).toBe(false);
    expect(submissionId).toBeTruthy();
    vi.unstubAllGlobals();
  });
});
