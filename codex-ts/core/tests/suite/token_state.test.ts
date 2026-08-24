/**
 * mirrors codex-rs/core/src/context_manager/history_tests.rs (the
 * get_total_token_usage arm) + codex-rs/core/src/session/tests.rs
 * (recompute_token_usage_*)
 *
 * Two pieces of codex-rs token accounting that the first ts port left out:
 *
 *   1. `get_total_token_usage` = the server's number PLUS an estimate of every
 *      item appended after the last model-generated one. Without it, a tool
 *      that returns 200 KB is invisible to the compaction check until the
 *      provider rejects the request it rode in on.
 *   2. `recompute_token_usage` — recompute from the WHOLE history and write
 *      back — as a callable unit, so anything that rewrites history (inline
 *      compaction here, a host-driven rebuild outside) can resync the
 *      bookkeeping instead of leaving it describing a history that is gone.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
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

  it("does NOT add the tail to an ESTIMATED total (it already covers it)", () => {
    const state = new SessionTokenState();
    const items = [userMsg("go"), assistantMsg("done"), userMsg("more")];
    state.setEstimated(estimateItemsTokenCount(items));

    expect(state.getTotalTokenUsage(items)).toBe(
      estimateItemsTokenCount(items),
    );
  });

  it("does NOT add the tail to a forced-FULL total", () => {
    const state = new SessionTokenState();
    state.setFull(1000);
    expect(
      state.getTotalTokenUsage([call("c"), output("c", "z".repeat(4000))]),
    ).toBe(1000);
  });

  // mirrors token_info == None → last_tokens = 0 (history.rs:421-426).
  it("is just the tail when no total was ever recorded", () => {
    const state = new SessionTokenState();
    const tail = output("call-1", "z".repeat(400));
    expect(state.getTotalTokenUsage([assistantMsg("hi"), tail])).toBe(
      estimateItemsTokenCount([tail]),
    );
  });

  it("switches back to adding the tail once a server sample lands", () => {
    const state = new SessionTokenState();
    const tail = output("call-1", "w".repeat(2000));
    const items = [call("call-1"), tail];
    state.setEstimated(9999);
    expect(state.getTotalTokenUsage(items)).toBe(9999);
    state.updateFromUsage({ inputTokens: 9999, totalTokens: 9999 });
    expect(state.getTotalTokenUsage(items)).toBe(
      9999 + estimateItemsTokenCount([tail]),
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
