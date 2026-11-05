/**
 * mirrors codex-rs/core/tests/suite/request_user_input.rs
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { CodexThread } from "../../src/codex_thread.js";
import {
  MAX_AUTO_RESOLUTION_MS,
  MIN_AUTO_RESOLUTION_MS,
  normalizeRequestUserInputArgs,
} from "../../src/tools/handlers/request_user_input_spec.js";
import type { EventMsg } from "../../../protocol/src/protocol.js";
import type { RequestUserInputEvent } from "../../../protocol/src/request_user_input.js";
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

beforeEach(() => { vi.unstubAllGlobals(); });

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makeRequestUserInputArgs(callId: string) {
  return sseFlat([
    evResponseCreated("resp-1"),
    evFunctionCall(callId, "request_user_input", {
      questions: [
        {
          id: "confirm_path",
          header: "Confirm",
          question: "Proceed with deployment?",
          options: [
            { label: "Yes (Recommended)", description: "Continue." },
            { label: "No", description: "Abort." },
          ],
        },
      ],
    }),
    evCompleted("resp-1"),
  ]);
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe("request_user_input_round_trip_resolves_pending", () => {
  it("emits RequestUserInput then completes after UserInputAnswer", async () => {
    const callId = "call-rui-1";
    const fetchMock = vi.fn();

    fetchMock.mockResolvedValueOnce(
      makeSseResponse(makeRequestUserInputArgs(callId)),
    );
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("resp-2"),
          evAssistantMessage("thanks"),
          evCompleted("resp-2"),
        ]),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({ apiKey: "test", model: "gpt-4o" });
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "please confirm" }],
    });

    // Wait for the agent to request user input
    const request = await waitForEventMatch<RequestUserInputEvent>(
      codex,
      (msg) =>
        msg.type === "RequestUserInput" ? msg.event : null,
    );

    expect(request.call_id).toBe(callId);
    expect(request.questions).toHaveLength(1);
    expect(request.questions[0]?.id).toBe("confirm_path");
    // normalizeRequestUserInputArgs sets isOther = true
    expect(request.questions[0]?.isOther).toBe(true);

    // Submit the answer — keyed by turn_id (mirrors Op::UserInputAnswer { id: request.turn_id })
    await codex.submit({
      type: "UserInputAnswer",
      id: request.turn_id,
      response: {
        answers: {
          confirm_path: { answers: ["yes"] },
        },
      },
    });

    // After answering the turn should complete
    const done = await waitForEvent(
      codex,
      (msg): msg is Extract<EventMsg, { type: "TurnComplete" }> =>
        msg.type === "TurnComplete",
    );
    expect(done.type).toBe("TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("request_user_input output sent to model", () => {
  it("includes the user answers in the next Responses API request", async () => {
    const callId = "call-rui-2";
    const fetchMock = vi.fn();
    const capturedBodies: unknown[] = [];

    fetchMock.mockImplementation(async (_url: string, init: RequestInit) => {
      capturedBodies.push(JSON.parse(init.body as string));
      const idx = capturedBodies.length;
      if (idx === 1) {
        return makeSseResponse(makeRequestUserInputArgs(callId));
      }
      return makeSseResponse(
        sseFlat([
          evResponseCreated("resp-2"),
          evAssistantMessage("done"),
          evCompleted("resp-2"),
        ]),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({ apiKey: "test", model: "gpt-4o" });
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "ask me something" }],
    });

    const request = await waitForEventMatch<RequestUserInputEvent>(
      codex,
      (msg) => (msg.type === "RequestUserInput" ? msg.event : null),
    );

    await codex.submit({
      type: "UserInputAnswer",
      id: request.turn_id,
      response: { answers: { confirm_path: { answers: ["yes"] } } },
    });

    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");

    // The second request to the model should contain the function_call_output
    const secondBody = capturedBodies[1] as {
      input: { type?: string; call_id?: string; output?: string }[];
    };
    const toolOutput = secondBody.input.find(
      (i) => i.type === "function_call_output" && i.call_id === callId,
    );
    expect(toolOutput).toBeDefined();
    const parsed = JSON.parse(toolOutput!.output ?? "{}") as {
      answers: Record<string, { answers: string[] }>;
    };
    expect(parsed.answers["confirm_path"]?.answers).toEqual(["yes"]);
  });
});

describe("request_user_input missing options rejects", () => {
  it("returns error output when options are empty", async () => {
    const callId = "call-rui-bad";
    const fetchMock = vi.fn();
    const capturedBodies: unknown[] = [];

    fetchMock.mockImplementation(async (_url: string, init: RequestInit) => {
      capturedBodies.push(JSON.parse(init.body as string));
      const idx = capturedBodies.length;
      if (idx === 1) {
        return makeSseResponse(
          sseFlat([
            evResponseCreated("resp-1"),
            evFunctionCall(callId, "request_user_input", {
              questions: [
                {
                  id: "q1",
                  header: "Q",
                  question: "Choose?",
                  options: [], // invalid — empty options
                },
              ],
            }),
            evCompleted("resp-1"),
          ]),
        );
      }
      return makeSseResponse(
        sseFlat([
          evResponseCreated("resp-2"),
          evAssistantMessage("acknowledged"),
          evCompleted("resp-2"),
        ]),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({ apiKey: "test", model: "gpt-4o" });
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "question" }],
    });

    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");

    // Tool output for the bad call should contain an error, not suspend
    const secondBody = capturedBodies[1] as {
      input: { type?: string; call_id?: string; output?: string }[];
    };
    const toolOutput = secondBody.input.find(
      (i) => i.type === "function_call_output" && i.call_id === callId,
    );
    expect(toolOutput).toBeDefined();
    const parsed = JSON.parse(toolOutput!.output ?? "{}") as { error: string };
    expect(parsed.error).toMatch(/non-empty options/);
  });
});

describe("normalizeRequestUserInputArgs guards a malformed questions field", () => {
  // Under strict:false the model can call request_user_input without a
  // `questions` array; normalize must return a tool error, never throw
  // "undefined is not an object (evaluating 'args.questions')".
  it("returns an error when questions is missing", () => {
    const result = normalizeRequestUserInputArgs({} as never);
    expect("error" in result).toBe(true);
  });

  it("returns an error when questions is not an array", () => {
    const result = normalizeRequestUserInputArgs({
      questions: "nope",
    } as never);
    expect("error" in result).toBe(true);
  });

  it("returns an error when questions is empty", () => {
    const result = normalizeRequestUserInputArgs({ questions: [] });
    expect("error" in result).toBe(true);
  });

  it("normalizes a valid question and sets isOther = true", () => {
    const result = normalizeRequestUserInputArgs({
      questions: [
        {
          id: "q1",
          header: "Q",
          question: "Pick?",
          options: [{ label: "A", description: "a" }],
        },
      ],
    });
    expect("questions" in result && result.questions[0]?.isOther).toBe(true);
  });
});

// mirrors the auto_resolution_ms clamp (request_user_input_spec.rs:123)
describe("normalizeRequestUserInputArgs clamps autoResolutionMs", () => {
  const question = {
    id: "q1",
    header: "Q",
    question: "Pick?",
    options: [{ label: "A", description: "a" }],
  };

  it("keeps an in-range window", () => {
    const result = normalizeRequestUserInputArgs({
      questions: [question],
      autoResolutionMs: 90_000,
    });
    expect("questions" in result && result.autoResolutionMs).toBe(90_000);
  });

  it("clamps below MIN and above MAX", () => {
    const low = normalizeRequestUserInputArgs({
      questions: [question],
      autoResolutionMs: 1_000,
    });
    expect("questions" in low && low.autoResolutionMs).toBe(MIN_AUTO_RESOLUTION_MS);

    const high = normalizeRequestUserInputArgs({
      questions: [question],
      autoResolutionMs: 999_999,
    });
    expect("questions" in high && high.autoResolutionMs).toBe(MAX_AUTO_RESOLUTION_MS);
  });

  it("omits the field when absent or non-numeric", () => {
    const absent = normalizeRequestUserInputArgs({ questions: [question] });
    expect("questions" in absent && absent.autoResolutionMs).toBeUndefined();

    const bogus = normalizeRequestUserInputArgs({
      questions: [question],
      autoResolutionMs: "soon",
    });
    expect("questions" in bogus && bogus.autoResolutionMs).toBeUndefined();
  });
});

describe("request_user_input forwards autoResolutionMs to the event", () => {
  it("emits the clamped window on RequestUserInput", async () => {
    const callId = "call-rui-auto";
    const fetchMock = vi.fn();
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("resp-1"),
          evFunctionCall(callId, "request_user_input", {
            questions: [
              {
                id: "confirm_path",
                header: "Confirm",
                question: "Proceed?",
                options: [{ label: "Yes", description: "Continue." }],
              },
            ],
            autoResolutionMs: 10,
          }),
          evCompleted("resp-1"),
        ]),
      ),
    );
    fetchMock.mockResolvedValue(
      makeSseResponse(
        sseFlat([
          evResponseCreated("resp-2"),
          evAssistantMessage("done"),
          evCompleted("resp-2"),
        ]),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({ apiKey: "k", model: "m" });
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "deploy" }],
    });

    const event = await waitForEventMatch<RequestUserInputEvent>(
      codex,
      (msg: EventMsg) => (msg.type === "RequestUserInput" ? msg.event : null),
    );
    expect(event.autoResolutionMs).toBe(MIN_AUTO_RESOLUTION_MS);

    await codex.submit({
      type: "UserInputAnswer",
      id: event.turn_id,
      response: { answers: { confirm_path: { answers: ["Yes"] } } },
    });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");
  });
});

// mirrors handlers/request_user_input.rs: the wait ends once the turn is
// cancelled. A parked turn used to ignore Interrupt entirely — no TurnAborted,
// and the leaked resolver kept hasPendingInput(turnId) true.
describe("interrupting a parked request_user_input", () => {
  it("ends the turn, drops the pending entry, and a late answer is a no-op", async () => {
    const callId = "call-rui-interrupt";
    const capturedBodies: { input: { type?: string; call_id?: string; output?: unknown }[] }[] = [];
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      capturedBodies.push(JSON.parse(init.body as string) as (typeof capturedBodies)[number]);
      if (capturedBodies.length === 1) {
        return makeSseResponse(makeRequestUserInputArgs(callId));
      }
      return makeSseResponse(
        sseFlat([evResponseCreated("resp-2"), evAssistantMessage("ok"), evCompleted("resp-2")]),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({ apiKey: "test", model: "gpt-4o" });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "ask" }] });
    const request = await waitForEventMatch<RequestUserInputEvent>(codex, (msg) =>
      msg.type === "RequestUserInput" ? msg.event : null,
    );
    expect(codex.hasPendingInput(request.turn_id)).toBe(true);

    await codex.submit({ type: "Interrupt" });
    expect(codex.hasPendingInput(request.turn_id)).toBe(false);
    const aborted = await waitForEvent(codex, (msg) => msg.type === "TurnAborted");
    expect(aborted.type === "TurnAborted" && aborted.event.turn_id).toBe(request.turn_id);

    // A late answer finds no resolver: nothing resumes, nothing is sampled.
    await codex.submit({
      type: "UserInputAnswer",
      id: request.turn_id,
      response: { answers: { confirm_path: { answers: ["yes"] } } },
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // The call was recorded as aborted, so the next request has it paired.
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "next" }] });
    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");
    const outputs = capturedBodies[1]!.input.filter(
      (item) => item.type === "function_call_output" && item.call_id === callId,
    );
    expect(outputs.map((item) => item.output)).toEqual([
      expect.stringMatching(/^aborted by user after \d+\.\ds$/u),
    ]);
  });
});
