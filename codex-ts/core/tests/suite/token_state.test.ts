/**
 * mirrors codex-rs/core/src/context_manager/history_tests.rs (the
 * get_total_token_usage arm) + codex-rs/core/src/session/tests.rs
 * (recompute_token_usage_*)
 *
 * Two pieces of codex-rs token accounting that the first ts port left out:
 *
 *   1. `get_total_token_usage` = the last count PLUS an estimate of every item
 *      past the history that count covers (the covered-item boundary, #58404).
 *      Without it, a tool that returns 200 KB is invisible to the compaction
 *      check until the provider rejects the request it rode in on.
 *   2. `recompute_token_usage` — recompute from the WHOLE history and write
 *      back — as a callable unit, so anything that rewrites history (inline
 *      compaction here, a host-driven rebuild outside) can resync the
 *      bookkeeping instead of leaving it describing a history that is gone.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import { runTurn } from "../../src/session/turn.js";
import { ToolRouter } from "../../src/tools/router.js";
import { GoalToolExecutor } from "../../../ext/goal/src/tool.js";
import { AutoCompactWindow } from "../../src/state/auto_compact_window.js";
import {
  approxTokenCount,
  estimateItemsTokenCount,
  estimateTokensAfterLastModelGeneratedItem,
  recomputeTokenUsage,
  SessionTokenState,
} from "../../src/state/token_state.js";
import type { EventMsg } from "../../../protocol/src/protocol.js";
import type { ConversationItem } from "../../../thread-store/src/types.js";
import type { CustomTool } from "../../src/tools/router.js";
import {
  evAssistantMessage,
  evCompleted,
  evFunctionCall,
  evResponseCreated,
  makeSseResponse,
  sseFlat,
  waitForEvent,
} from "../common/lib.js";

// ─── Unit: which items count as "not yet measured by the server" ──────────────

const userMsg = (text: string): ConversationItem => ({
  type: "message",
  role: "user",
  content: text,
});
const assistantMsg = (text: string): ConversationItem => ({
  type: "message",
  role: "assistant",
  content: text,
});
const call = (callId: string): ConversationItem => ({
  type: "function_call",
  call_id: callId,
  name: "some_tool",
  arguments: "{}",
});
const output = (callId: string, text: string): ConversationItem => ({
  type: "function_call_output",
  call_id: callId,
  output: text,
});

describe("estimateTokensAfterLastModelGeneratedItem", () => {
  it("is 0 for an empty history", () => {
    expect(estimateTokensAfterLastModelGeneratedItem([])).toBe(0);
  });

  // mirrors rposition(...).map_or(len, ..) (history.rs:414): NO model-generated
  // item yields an EMPTY slice, not the whole history. A brand-new thread is
  // covered by the estimated seed instead, and counting it here would double it.
  it("is 0 when nothing was model-generated yet", () => {
    expect(
      estimateTokensAfterLastModelGeneratedItem([
        userMsg("first"),
        userMsg("second"),
      ]),
    ).toBe(0);
  });

  it("counts the user message appended after the last assistant reply", () => {
    const pending = userMsg("what about this?");
    expect(
      estimateTokensAfterLastModelGeneratedItem([
        userMsg("hi"),
        assistantMsg("hello"),
        pending,
      ]),
    ).toBe(estimateItemsTokenCount([pending]));
  });

  // The load-bearing shape: tool CALLS came back inside the sampled response
  // (already in the server's number), tool OUTPUTS did not.
  it("counts tool outputs but not the calls that produced them", () => {
    const bigOutput = output("call-1", "x".repeat(4000));
    const smallOutput = output("call-2", "ok");
    const items = [
      userMsg("go"),
      assistantMsg("working"),
      call("call-1"),
      call("call-2"),
      bigOutput,
      smallOutput,
    ];
    expect(estimateTokensAfterLastModelGeneratedItem(items)).toBe(
      estimateItemsTokenCount([bigOutput, smallOutput]),
    );
    // …and the calls really are excluded — dropping them changes nothing.
    expect(estimateTokensAfterLastModelGeneratedItem(items)).toBe(
      estimateTokensAfterLastModelGeneratedItem([
        assistantMsg("working"),
        bigOutput,
        smallOutput,
      ]),
    );
  });
});

// ─── Unit: getTotalTokenUsage ─────────────────────────────────────────────────

describe("SessionTokenState.getTotalTokenUsage", () => {
  it("adds the unmeasured tail to a SERVER total", () => {
    const state = new SessionTokenState();
    state.updateFromUsage({ inputTokens: 400, outputTokens: 100 });
    const tail = output("call-1", "y".repeat(4000));
    const items = [userMsg("go"), call("call-1"), tail];

    expect(state.totalTokens).toBe(500);
    expect(state.getTotalTokenUsage(items)).toBe(
      500 + estimateItemsTokenCount([tail]),
    );
    // The bare getter is deliberately NOT adjusted — it stays rs's
    // last_token_usage.total_tokens.
    expect(state.totalTokens).toBe(500);
  });

  // No double count: the region an estimate covers is never added again.
  it("adds nothing right after an estimate that covers the whole history", () => {
    const state = new SessionTokenState();
    const items = [userMsg("go"), assistantMsg("done"), userMsg("more")];
    state.setEstimated(estimateItemsTokenCount(items), items[items.length - 1]);

    expect(state.getTotalTokenUsage(items)).toBe(
      estimateItemsTokenCount(items),
    );
  });

  it("a forced-FULL total never reads below the window", () => {
    const state = new SessionTokenState();
    state.setFull(1000);
    expect(state.totalTokens).toBe(1000);
    expect(
      state.getTotalTokenUsage([call("c"), output("c", "z".repeat(4000))]),
    ).toBeGreaterThanOrEqual(1000);
  });

  // mirrors token_info == None → last_tokens = 0 with the boundary at 0
  // (history.rs:1138-1163 after #58404): nothing is covered yet.
  it("is the whole history when no total was ever recorded", () => {
    const state = new SessionTokenState();
    const items = [assistantMsg("hi"), output("call-1", "z".repeat(400))];
    expect(state.getTotalTokenUsage(items)).toBe(estimateItemsTokenCount(items));
  });
});

// ─── Unit: the covered-item boundary (#58404) ─────────────────────────────────
// mirrors history_tests.rs's boundary cases after commit 7a030b9769: a count —
// server, estimate or restore — covers history up to a boundary, and anything
// appended after it is estimated on top, whether or not usage ever arrives.

describe("SessionTokenState covered-item boundary", () => {
  // The blind spot the provenance rule had: after an ESTIMATE, appended items
  // were invisible until a server usage frame arrived.
  it("counts items appended AFTER an estimate", () => {
    const state = new SessionTokenState();
    const history = [userMsg("go")];
    state.setEstimated(9999, history[0]);
    const reply = assistantMsg("x".repeat(2000));
    const toolCall = call("call-1");
    const toolOutput = output("call-1", "w".repeat(2000));
    history.push(reply, toolCall, toolOutput);

    expect(state.getTotalTokenUsage(history)).toBe(
      9999 + estimateItemsTokenCount([reply, toolCall, toolOutput]),
    );
  });

  it("a server count covers the request AND the response's own output", () => {
    const state = new SessionTokenState();
    const sent = userMsg("go");
    const history = [sent];
    state.updateFromUsage({
      inputTokens: 400,
      totalTokens: 500,
      coveredThrough: sent,
    });
    // Recorded after the stream ended: the measured response's output, then
    // the tool result the client appended.
    const toolOutput = output("call-1", "y".repeat(4000));
    history.push(assistantMsg("working"), call("call-1"), toolOutput);

    expect(state.getTotalTokenUsage(history)).toBe(
      500 + estimateItemsTokenCount([toolOutput]),
    );
  });

  // The response WITHOUT usable usage: its output and everything after it stay
  // estimated (rs: "keep failed or usage-less output in the estimate until a
  // usable count covers it", history.rs:1137).
  it("a response with no usage leaves its whole output counted", () => {
    const state = new SessionTokenState();
    const history = [userMsg("go")];
    state.setEstimated(100, history[0]);
    const tail = [
      assistantMsg("x".repeat(800)),
      call("call-1"),
      output("call-1", "y".repeat(800)),
    ];
    history.push(...tail);

    expect(state.getTotalTokenUsage(history)).toBe(
      100 + estimateItemsTokenCount(tail),
    );
  });

  // Only the run right after the boundary is the measured response's output;
  // a model item further down belongs to a later, unmeasured response.
  it("covers only the output run directly after the boundary", () => {
    const state = new SessionTokenState();
    const sent = userMsg("go");
    const history: ConversationItem[] = [sent];
    state.updateFromUsage({ inputTokens: 500, coveredThrough: sent });
    const later = [
      output("call-1", "ok"),
      assistantMsg("x".repeat(800)),
      call("call-2"),
    ];
    history.push(call("call-1"), ...later);

    expect(state.getTotalTokenUsage(history)).toBe(
      500 + estimateItemsTokenCount(later),
    );
  });

  it("null boundary: the request carried no history, so all of it counts", () => {
    const state = new SessionTokenState();
    state.updateFromUsage({ inputTokens: 50, coveredThrough: null });
    const history = [assistantMsg("hi"), userMsg("x".repeat(400))];
    // The leading assistant reply is that response's output.
    expect(state.getTotalTokenUsage(history)).toBe(
      50 + estimateItemsTokenCount([history[1]!]),
    );
  });

  // Fallback: the boundary item left history without a recompute. rs's
  // reset_last_usage_boundary approximation — after the last model output —
  // not the whole history (which would add the covered region a second time).
  it("falls back to after-the-last-model-output when the boundary item is gone", () => {
    const state = new SessionTokenState();
    const gone = userMsg("summarized away");
    state.setEstimated(700, gone);
    const toolOutput = output("call-1", "v".repeat(4000));
    const history = [userMsg("a"), call("call-1"), toolOutput];

    expect(state.getTotalTokenUsage(history)).toBe(
      700 + estimateItemsTokenCount([toolOutput]),
    );
  });

  it("a total recorded without a boundary uses the same fallback (0.8.0 rule)", () => {
    const state = new SessionTokenState();
    state.setEstimated(9999);
    const tail = output("call-1", "w".repeat(2000));
    expect(state.getTotalTokenUsage([call("call-1"), tail])).toBe(
      9999 + estimateItemsTokenCount([tail]),
    );
  });

  it("a new count moves the boundary: nothing it covers is added twice", () => {
    const state = new SessionTokenState();
    const history = [userMsg("go")];
    state.setEstimated(100, history[0]);
    const toolOutput = output("call-1", "w".repeat(2000));
    history.push(call("call-1"), toolOutput);
    expect(state.getTotalTokenUsage(history)).toBeGreaterThan(100);

    state.updateFromUsage({ inputTokens: 900, coveredThrough: toolOutput });
    expect(state.getTotalTokenUsage(history)).toBe(900);
  });

  // Resume: rs installs restored history with the boundary right after the
  // last model-generated item (reset_last_usage_boundary).
  it("restore covers through the last model output; later client items count", () => {
    const state = new SessionTokenState();
    const toolOutput = output("call-1", "u".repeat(4000));
    const history = [userMsg("go"), call("call-1"), toolOutput];
    state.restore(1_500, history);
    expect(state.getTotalTokenUsage(history)).toBe(
      1_500 + estimateItemsTokenCount([toolOutput]),
    );

    const next = userMsg("next");
    history.push(next);
    expect(state.getTotalTokenUsage(history)).toBe(
      1_500 + estimateItemsTokenCount([toolOutput, next]),
    );
  });

  it("restore of a history with no model output (a compaction checkpoint) covers all of it", () => {
    const state = new SessionTokenState();
    const history = [userMsg("kept ask"), userMsg("summary")];
    state.restore(1_200, history);
    expect(state.getTotalTokenUsage(history)).toBe(1_200);
    const next = userMsg("x".repeat(400));
    history.push(next);
    expect(state.getTotalTokenUsage(history)).toBe(
      1_200 + estimateItemsTokenCount([next]),
    );
  });
});

// ─── Unit: recomputeTokenUsage ────────────────────────────────────────────────

describe("recomputeTokenUsage", () => {
  const items = [userMsg("one"), assistantMsg("two"), userMsg("three")];
  const instructions = "be brief";
  const expected =
    estimateItemsTokenCount(items) + approxTokenCount(instructions);

  it("writes the whole-history estimate into the state and the window", () => {
    const tokenState = new SessionTokenState();
    const compactWindow = new AutoCompactWindow();
    tokenState.updateFromUsage({ inputTokens: 123_456 });

    const written = recomputeTokenUsage({
      items,
      instructions,
      tokenState,
      compactWindow,
    });

    expect(written).toBe(expected);
    expect(tokenState.totalTokens).toBe(expected);
    expect(compactWindow.snapshot().prefillInputTokens).toBe(expected);
    // Written as an ESTIMATE: the stale 123,456 is gone, and the tail is not
    // added on top of the fresh number.
    expect(tokenState.getTotalTokenUsage(items)).toBe(expected);
  });

  it("leaves an existing total alone when overwriteTotal is false", () => {
    const tokenState = new SessionTokenState();
    const compactWindow = new AutoCompactWindow();
    tokenState.updateFromUsage({ inputTokens: 700, totalTokens: 700 });

    recomputeTokenUsage({
      items,
      instructions,
      tokenState,
      compactWindow,
      overwriteTotal: false,
    });

    expect(tokenState.totalTokens).toBe(700);
    // …but the window baseline — the thing that was actually missing — is set.
    expect(compactWindow.snapshot().prefillInputTokens).toBe(expected);
  });

  it("never clobbers a server-observed window baseline", () => {
    const tokenState = new SessionTokenState();
    const compactWindow = new AutoCompactWindow();
    compactWindow.ensureServerObservedPrefill(4242);

    recomputeTokenUsage({ items, instructions, tokenState, compactWindow });

    expect(compactWindow.snapshot().prefillInputTokens).toBe(4242);
  });

  // mirrors recompute_token_usage's closing send_token_count_event
  // (session/mod.rs:4031) with a TokenUsage whose components are all zero.
  it("emits TokenCount with the recomputed total when asked", () => {
    const events: EventMsg[] = [];
    recomputeTokenUsage({
      items,
      instructions,
      tokenState: new SessionTokenState(),
      contextWindow: 60_000,
      emitEvent: (msg) => events.push(msg),
    });

    expect(events).toHaveLength(1);
    const event = events[0]!;
    expect(event.type).toBe("TokenCount");
    if (event.type !== "TokenCount") throw new Error("unreachable");
    const info = event.event.info!;
    expect(info.total_token_usage.total_tokens).toBe(expected);
    expect(info.last_token_usage.total_tokens).toBe(expected);
    expect(info.last_token_usage.input_tokens).toBe(0);
    expect(info.last_token_usage.output_tokens).toBe(0);
    expect(info.last_token_usage.cached_input_tokens).toBe(0);
    expect(info.model_context_window).toBe(60_000);
  });
});

// ─── Integration: a huge tool OUTPUT must trip the compaction check ───────────
// The regression this whole change exists for. The server's usage frame is
// recorded before the tool runs, so the next request of the SAME turn carries
// an output no usage number has seen. Reading the bare session total there says
// "500 tokens, plenty of room" and sails into `prompt is too long`.

describe("mid-turn compaction sees unmeasured tool output", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  const CONTEXT_WINDOW = 8000;

  function bigOutputTool(payload: string): CustomTool {
    return {
      name: "read_file",
      spec: () => ({
        type: "function",
        tool: {
          name: "read_file",
          description: "Return a file's contents.",
          parameters: {
            type: "object",
            properties: {},
            additionalProperties: false,
          },
          strict: false,
        },
      }),
      execute: async () => payload,
    };
  }

  /** One sampled response that calls the tool, reporting a SMALL usage. */
  function toolCallRound(): string {
    return sseFlat([
      evResponseCreated("r1"),
      evFunctionCall("call-1", "read_file", {}),
      {
        type: "response.completed",
        response: {
          id: "r1",
          status: "completed",
          usage: { input_tokens: 490, output_tokens: 10, total_tokens: 500 },
        },
      },
    ]);
  }

  it("compacts when the tool output pushes the active context over the window", async () => {
    // 40 000 chars ≈ 10 000 estimated tokens — on its own already past the
    // 8 000-token window, while the server only ever reported 500.
    const payload = "x".repeat(40_000);
    expect(
      estimateItemsTokenCount([output("call-1", payload)]),
    ).toBeGreaterThan(CONTEXT_WINDOW);

    const fetchMock = vi.fn();
    fetchMock.mockResolvedValueOnce(makeSseResponse(toolCallRound()));
    // The compaction request the check must now schedule.
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("compact"),
          evAssistantMessage("summary text"),
          evCompleted("compact"),
        ]),
      ),
    );
    // …then sampling resumes on the compacted history.
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("r2"),
          evAssistantMessage("done"),
          evCompleted("r2"),
        ]),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      contextWindow: CONTEXT_WINDOW,
      customTools: [bigOutputTool(payload)],
    });
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "read it" }],
    });

    const compacted = await waitForEvent(
      codex,
      (msg) => msg.type === "ContextCompacted",
    );
    expect(compacted.type).toBe("ContextCompacted");
    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  // mirrors core/tests/suite/compact.rs's #58404 regression ("completed
  // responses without usage"): with no usage frame the total stays the turn-
  // start estimate, which covers only the user message. The provenance rule
  // added nothing to an estimate, so the 10k-token output never counted.
  it("compacts on a huge tool output even when no response carries usage", async () => {
    const payload = "x".repeat(40_000);
    const noUsageToolCall = sseFlat([
      evResponseCreated("r1"),
      evFunctionCall("call-1", "read_file", {}),
      evCompleted("r1"),
    ]);
    const fetchMock = vi.fn();
    fetchMock.mockResolvedValueOnce(makeSseResponse(noUsageToolCall));
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("compact"),
          evAssistantMessage("summary text"),
          evCompleted("compact"),
        ]),
      ),
    );
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("r2"),
          evAssistantMessage("done"),
          evCompleted("r2"),
        ]),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      contextWindow: CONTEXT_WINDOW,
      customTools: [bigOutputTool(payload)],
    });
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "read it" }],
    });

    const compacted = await waitForEvent(
      codex,
      (msg) => msg.type === "ContextCompacted",
    );
    expect(compacted.type).toBe("ContextCompacted");
    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  // #58404: "successful usage reports cover the request input even when the
  // response contains no output items". The second request carried the 10k
  // tool output and its usage counts it; adding it again would double it.
  it("a usage report covers its request even when the response has no output", async () => {
    const payload = "x".repeat(40_000);
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(makeSseResponse(toolCallRound()))
      .mockResolvedValueOnce(
        makeSseResponse(
          sseFlat([
            evResponseCreated("r2"),
            {
              type: "response.completed",
              response: {
                id: "r2",
                status: "completed",
                usage: { input_tokens: 10_990, output_tokens: 10, total_tokens: 11_000 },
              },
            },
          ]),
        ),
      );
    const history: ConversationItem[] = [];
    const tokenState = new SessionTokenState();
    await runTurn(
      "turn",
      [{ type: "text", text: "read it" }],
      history,
      { apiKey: "t", baseUrl: "http://x", model: "m", fetch, tokenState },
      new ToolRouter(new GoalToolExecutor("t"), [bigOutputTool(payload)]),
      new Map(),
      () => undefined,
    );
    expect(history.map((item) => item.type)).toEqual([
      "message",
      "function_call",
      "function_call_output",
    ]);
    expect(tokenState.getTotalTokenUsage(history)).toBe(11_000);
  });

  // Negative control on the SAME shapes: it is the output's size that trips the
  // check, not the presence of a tool call.
  it("does not compact when the same tool returns a small output", async () => {
    const fetchMock = vi.fn();
    fetchMock.mockResolvedValueOnce(makeSseResponse(toolCallRound()));
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("r2"),
          evAssistantMessage("done"),
          evCompleted("r2"),
        ]),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      contextWindow: CONTEXT_WINDOW,
      customTools: [bigOutputTool("ok")],
    });
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "read it" }],
    });

    const settled = await waitForEvent(
      codex,
      (msg) => msg.type === "ContextCompacted" || msg.type === "TurnComplete",
    );
    expect(settled.type).toBe("TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
