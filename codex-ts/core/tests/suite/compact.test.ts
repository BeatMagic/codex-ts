/**
 * mirrors codex-rs/core/tests/suite/compact.rs
 *
 * Tests for inline auto-compaction:
 *   - AutoCompactWindow token tracking (BodyAfterPrefix mode)
 *   - collectUserMessages / buildCompactedHistory helpers
 *   - Mid-turn compaction triggered when inputTokens >= autoCompactTokenLimit
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { AutoCompactWindow } from "../../src/state/auto_compact_window.js";
import {
  collectUserMessages,
  buildCompactedHistory,
  isSummaryMessage,
  runInlineAutoCompactTask,
  SUMMARIZATION_PROMPT,
  SUMMARY_PREFIX,
} from "../../src/compact.js";
import { CodexThread } from "../../src/codex_thread.js";
import { SessionTokenState } from "../../src/state/token_state.js";
import { isContextWindowExceededText } from "../../src/session/retry.js";
import type { TurnConfig } from "../../src/session/turn.js";
import type { EventMsg } from "../../../protocol/src/protocol.js";
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

// ─── AutoCompactWindow unit tests ─────────────────────────────────────────────

describe("AutoCompactWindow — BodyAfterPrefix tracking", () => {
  it("starts with no baseline; bodyAfterPrefix returns 0", () => {
    const w = new AutoCompactWindow();
    // No prefill set → baseline == totalInputTokens → 0 growth
    expect(w.bodyAfterPrefix(1000)).toBe(0);
  });

  it("records server-observed prefill and measures growth", () => {
    const w = new AutoCompactWindow();
    w.ensureServerObservedPrefill(800);
    expect(w.bodyAfterPrefix(1000)).toBe(200);
  });

  it("server-observed prefill wins over estimated", () => {
    const w = new AutoCompactWindow();
    w.setEstimatedPrefill(500);
    w.ensureServerObservedPrefill(800);
    // server value (800) should win
    expect(w.bodyAfterPrefix(1000)).toBe(200);
    // Further calls to setEstimatedPrefill should be ignored
    w.setEstimatedPrefill(100);
    expect(w.bodyAfterPrefix(1000)).toBe(200);
  });

  // mirrors auto_compact_window.rs tracks_prefill_and_window_boundaries: the
  // window number starts at 0 and advance() rotates the ids; clearing the
  // baseline is start_new_context_window's job (state/session.rs:199).
  it("startNewContextWindow resets baseline and bumps the window number", () => {
    const w = new AutoCompactWindow();
    w.ensureServerObservedPrefill(800);
    expect(w.currentWindowNumber()).toBe(0);
    const initialIds = w.currentIds();
    expect(initialIds.previousWindowId).toBeNull();
    expect(initialIds.firstWindowId).toBe(initialIds.windowId);

    const advanced = w.startNewContextWindow();
    expect(advanced.windowNumber).toBe(1);
    expect(w.currentWindowNumber()).toBe(1);
    expect(advanced.ids.firstWindowId).toBe(initialIds.firstWindowId);
    expect(advanced.ids.previousWindowId).toBe(initialIds.windowId);
    expect(advanced.ids.windowId).not.toBe(initialIds.windowId);
    expect(w.snapshot().prefillInputTokens).toBeNull();
    // After reset, no growth again
    expect(w.bodyAfterPrefix(400)).toBe(0);
  });

  // mirrors claim_token_budget_reminder / claim_auto_compact_fallback +
  // request_new_context_window (auto_compact_window.rs:88-104).
  it("per-window flags are one-shot and reset on advance", () => {
    const w = new AutoCompactWindow();
    expect(w.claimTokenBudgetReminder()).toBe(true);
    expect(w.claimTokenBudgetReminder()).toBe(false);
    expect(w.claimAutoCompactFallback()).toBe(true);
    expect(w.claimAutoCompactFallback()).toBe(false);

    w.requestNewContextWindow();
    expect(w.takeNewContextWindowRequest()).toBe(true);
    expect(w.takeNewContextWindowRequest()).toBe(false);

    w.requestNewContextWindow();
    w.advance();
    // advance() clears a pending request and re-arms both one-shot flags
    expect(w.takeNewContextWindowRequest()).toBe(false);
    expect(w.claimTokenBudgetReminder()).toBe(true);
    expect(w.claimAutoCompactFallback()).toBe(true);
  });

  // mirrors restore() (auto_compact_window.rs:72)
  it("restore reinstates a persisted window number and ids", () => {
    const w = new AutoCompactWindow();
    const ids = {
      firstWindowId: "first",
      previousWindowId: "prev",
      windowId: "current",
    };
    w.restore(3, ids);
    expect(w.currentWindowNumber()).toBe(3);
    expect(w.currentIds()).toEqual(ids);
  });
});

// ─── History helper unit tests ─────────────────────────────────────────────────

describe("collectUserMessages", () => {
  it("extracts text from user messages", () => {
    const history: ConversationItem[] = [
      { type: "message", role: "user", content: "hello" },
      { type: "message", role: "assistant", content: "hi" },
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "follow-up" }],
      },
    ];
    expect(collectUserMessages(history)).toEqual(["hello", "follow-up"]);
  });

  it("skips assistant and tool messages", () => {
    const history: ConversationItem[] = [
      { type: "message", role: "assistant", content: "response" },
      { type: "function_call", call_id: "c1", name: "tool", arguments: "{}" },
    ];
    expect(collectUserMessages(history)).toEqual([]);
  });

  // mirrors collect_user_messages's is_summary_message filter (compact.rs:529):
  // a previous compaction summary is NOT a user message, so a second compaction
  // must not re-collect it and stack summaries on top of each other.
  it("skips a previous compaction summary", () => {
    const previousSummary = `${SUMMARY_PREFIX}\nearlier work recap`;
    const history: ConversationItem[] = [
      { type: "message", role: "user", content: "first ask" },
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: previousSummary }],
      },
      { type: "message", role: "user", content: "next ask" },
    ];
    expect(collectUserMessages(history)).toEqual(["first ask", "next ask"]);
    expect(isSummaryMessage(previousSummary)).toBe(true);
    expect(isSummaryMessage("first ask")).toBe(false);
  });

  // mirrors parse_user_message → None for contextual content
  // (event_mapping.rs:114-117): host-injected context is not a user message.
  it("skips messages the contextual predicate claims, given their flattened text", () => {
    const seen: string[] = [];
    const isContextual = (text: string): boolean => {
      seen.push(text);
      return text.startsWith("[SYSTEM]");
    };
    const history: ConversationItem[] = [
      { type: "message", role: "user", content: "real ask" },
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_text", text: "[SYSTEM] workspace changed" },
          { type: "input_text", text: "details" },
        ],
      },
      { type: "message", role: "user", content: "[SYSTEM] cue" },
      { type: "message", role: "user", content: "second ask" },
    ];
    expect(collectUserMessages(history, isContextual)).toEqual([
      "real ask",
      "second ask",
    ]);
    // input_text parts joined with "\n" — the same text that would be retained.
    expect(seen).toContain("[SYSTEM] workspace changed\ndetails");
  });

  it("keeps every user message without a predicate (unchanged default)", () => {
    const history: ConversationItem[] = [
      { type: "message", role: "user", content: "[SYSTEM] cue" },
      { type: "message", role: "user", content: "ask" },
    ];
    expect(collectUserMessages(history)).toEqual(["[SYSTEM] cue", "ask"]);
  });
});

describe("buildCompactedHistory", () => {
  it("produces summary as the last user message", () => {
    const result = buildCompactedHistory(["msg1", "msg2"], "SUMMARY");
    const last = result[result.length - 1];
    expect("role" in (last ?? {}) && (last as { role: string }).role).toBe("user");
    expect(JSON.stringify(last)).toContain("SUMMARY");
  });

  it("uses fallback text when summary is empty", () => {
    const result = buildCompactedHistory([], "");
    const last = result[result.length - 1];
    expect(JSON.stringify(last)).toContain("(no summary available)");
  });
});

// ─── Integration: mid-turn compaction triggered ───────────────────────────────

describe("mid-turn auto-compaction", () => {
  beforeEach(() => { vi.unstubAllGlobals(); });

  function sseWithUsage(inputTokens: number, items: object[]) {
    return sseFlat([
      ...items,
      {
        type: "response.completed",
        response: {
          id: "resp",
          status: "completed",
          usage: { input_tokens: inputTokens, output_tokens: 10, total_tokens: inputTokens + 10 },
        },
      },
    ]);
  }

  it("compacts mid-turn when BodyAfterPrefix growth >= autoCompactTokenLimit", async () => {
    // BodyAfterPrefix mode: round 1 sets the baseline; round 2 measures growth.
    // round 1 baseline = 1000 tokens, round 2 reports 2000 → growth = 1000 >= 900 → compact.
    const fetchMock = vi.fn();

    // Round 1 — model calls get_goal; reports 1000 tokens (sets baseline)
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseWithUsage(1000, [
          evResponseCreated("resp-1"),
          evFunctionCall("call-1", "get_goal", {}),
        ]),
      ),
    );

    // Round 2 — after get_goal, model calls update_plan; usage = 2000 → growth 1000 ≥ 900
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseWithUsage(2000, [
          evResponseCreated("resp-2"),
          evFunctionCall("call-2", "update_plan", { plan: [{ step: "s", status: "completed" }] }),
        ]),
      ),
    );

    // Compaction request — model returns a summary
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("resp-compact"),
          evAssistantMessage("summary text"),
          evCompleted("resp-compact"),
        ]),
      ),
    );

    // Round 3 — after compaction, model returns final reply
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("resp-3"),
          evAssistantMessage("done"),
          evCompleted("resp-3"),
        ]),
      ),
    );

    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      autoCompactTokenLimit: 900,
    });

    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });

    const compacted = await waitForEvent(codex, (msg) => msg.type === "ContextCompacted");
    expect(compacted.type).toBe("ContextCompacted");

    // No "Heads up: Long threads…" Warning after the compaction (removed
    // upstream in #55355).
    const afterCompaction: EventMsg[] = [];
    await waitForEvent(codex, (msg) => {
      afterCompaction.push(msg);
      return msg.type === "TurnComplete";
    });
    expect(afterCompaction.filter((msg) => msg.type === "Warning")).toEqual([]);

    // 4 fetches: round1, round2, compaction, round3
    expect(fetchMock).toHaveBeenCalledTimes(4);

    // Compaction request (index 2) must include SUMMARIZATION_PROMPT as last input item
    const compactBody = JSON.parse(
      (fetchMock.mock.calls[2]![1] as RequestInit).body as string,
    ) as { input: unknown[] };
    const lastItem = compactBody.input[compactBody.input.length - 1];
    expect(JSON.stringify(lastItem)).toContain(SUMMARIZATION_PROMPT.slice(0, 30));

    // Round 3 request (index 3) must contain SUMMARY_PREFIX in history
    const finalBody = JSON.parse(
      (fetchMock.mock.calls[3]![1] as RequestInit).body as string,
    ) as { input: unknown[] };
    expect(JSON.stringify(finalBody.input)).toContain(SUMMARY_PREFIX.slice(0, 30));
  });

  it("does NOT compact when usage is below the limit", async () => {
    const fetchMock = vi.fn();

    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseWithUsage(500, [
          evResponseCreated("resp-1"),
          evFunctionCall("call-1", "get_goal", {}),
        ]),
      ),
    );

    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("resp-2"),
          evAssistantMessage("done"),
          evCompleted("resp-2"),
        ]),
      ),
    );

    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      autoCompactTokenLimit: 900,
    });

    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "hi" }] });
    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");

    // Only two fetches: no compaction
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

// ─── Integration: contextual user messages are not retained ────────────────────
// CodexThreadConfig.isContextualUserMessage reaches the compaction: the host's
// hidden per-turn messages are left to the summary instead of surviving it
// verbatim and eating the 20k retained budget.

describe("compaction with isContextualUserMessage", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  async function compactOnce(
    isContextualUserMessage: ((text: string) => boolean) | undefined,
  ) {
    const fetchMock = vi.fn();
    const usage = (inputTokens: number, items: object[]) =>
      sseFlat([
        ...items,
        {
          type: "response.completed",
          response: {
            id: "resp",
            status: "completed",
            usage: { input_tokens: inputTokens, output_tokens: 10, total_tokens: inputTokens + 10 },
          },
        },
      ]);
    fetchMock
      .mockResolvedValueOnce(
        makeSseResponse(usage(1000, [evResponseCreated("r1"), evFunctionCall("c1", "get_goal", {})])),
      )
      .mockResolvedValueOnce(
        makeSseResponse(usage(2000, [evResponseCreated("r2"), evFunctionCall("c2", "get_goal", {})])),
      )
      .mockResolvedValueOnce(
        makeSseResponse(
          sseFlat([evResponseCreated("s"), evAssistantMessage("summary text"), evCompleted("s")]),
        ),
      )
      .mockResolvedValueOnce(
        makeSseResponse(sseFlat([evResponseCreated("r3"), evAssistantMessage("done"), evCompleted("r3")])),
      );
    vi.stubGlobal("fetch", fetchMock);
    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      autoCompactTokenLimit: 900,
      isContextualUserMessage,
    });
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "real ask" }],
      extraUserMessages: [[{ type: "text", text: "[SYSTEM] workspace event" }]],
    });
    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");
    return JSON.stringify(
      (JSON.parse((fetchMock.mock.calls[3]![1] as RequestInit).body as string) as {
        input: unknown[];
      }).input,
    );
  }

  it("drops the messages the predicate claims from the compacted history", async () => {
    const after = await compactOnce((text) => text.startsWith("[SYSTEM]"));
    expect(after).toContain(SUMMARY_PREFIX.slice(0, 30));
    expect(after).toContain("real ask");
    expect(after).not.toContain("[SYSTEM] workspace event");
  });

  it("retains them without a predicate", async () => {
    expect(await compactOnce(undefined)).toContain("[SYSTEM] workspace event");
  });
});

// ─── Integration: cross-turn compaction (thread-owned window) ──────────────────
// Regression for the per-turn-window bug: the compaction baseline must PERSIST
// across turns (codex-rs keeps it in session state, read via n_snapshot). A fresh
// AutoCompactWindow per runTurn reset the baseline to the current (already large)
// size every turn, so cross-turn growth never reached the limit — compaction never
// fired and long interactive threads grew past the model's context window → 400.

describe("cross-turn auto-compaction (thread-owned window)", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  function sseWithUsage(inputTokens: number, items: object[]) {
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

  it("compacts on growth accumulated ACROSS turns (baseline persists)", async () => {
    const fetchMock = vi.fn();

    // Turn 1 — plain reply, usage 1000 → sets the thread window baseline to 1000.
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseWithUsage(1000, [evResponseCreated("t1"), evAssistantMessage("ok")]),
      ),
    );

    // Turn 2 round 1 — tool call, usage 2000. With a PERSISTENT baseline (1000)
    // growth = 2000 - 1000 = 1000 ≥ 900 → compact. (A per-turn window would reset
    // the baseline to 2000 here → growth 0 → NO compaction — the bug.)
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseWithUsage(2000, [
          evResponseCreated("t2-r1"),
          evFunctionCall("call-1", "get_goal", {}),
        ]),
      ),
    );

    // Compaction request → summary.
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("t2-compact"),
          evAssistantMessage("summary text"),
          evCompleted("t2-compact"),
        ]),
      ),
    );

    // Turn 2 round 2 — after get_goal is dispatched, final reply.
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("t2-r2"),
          evAssistantMessage("done"),
          evCompleted("t2-r2"),
        ]),
      ),
    );

    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      autoCompactTokenLimit: 900,
    });

    // Turn 1 — establishes the baseline, no compaction.
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "hi" }],
    });
    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");

    // Turn 2 — cross-turn growth (1000→2000) must trigger compaction.
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "again" }],
    });
    const compacted = await waitForEvent(
      codex,
      (msg) => msg.type === "ContextCompacted",
    );
    expect(compacted.type).toBe("ContextCompacted");
    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");

    // Turn1(1) + Turn2 round1(1) + compaction(1) + Turn2 round2(1) = 4.
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });
});

// ─── Shared helpers for the context-window suites ──────────────────────────────

function makeErrorResponse(status: number, bodyText: string): Response {
  return {
    ok: false,
    status,
    headers: new Headers(),
    body: new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(bodyText));
        controller.close();
      },
    }),
    text: async () => bodyText,
  } as unknown as Response;
}

function sseWithFullUsage(
  inputTokens: number,
  totalTokens: number,
  items: object[],
) {
  return sseFlat([
    ...items,
    {
      type: "response.completed",
      response: {
        id: "resp",
        status: "completed",
        usage: {
          input_tokens: inputTokens,
          output_tokens: totalTokens - inputTokens,
          total_tokens: totalTokens,
        },
      },
    },
  ]);
}

// The EXACT body OpenRouter returned in the production incident (Anthropic raw
// error wrapped in metadata.raw — no OpenAI-canonical code anywhere).
const OPENROUTER_PROMPT_TOO_LONG_BODY = JSON.stringify({
  error: {
    message: "Provider returned error",
    code: 400,
    metadata: {
      raw: '{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 1102084 tokens > 1000000 maximum"},"request_id":"req_x"}',
      provider_name: "Azure",
    },
  },
});

// The EXACT body from the 2026-07-18 a1d14926 incident: OpenRouter's own
// ENDPOINT pre-check rejects before any provider is tried, so none of the
// provider wordings ("prompt is too long", "context_length_exceeded") appear.
// Missing this pattern bricked the self-heal: setFull/compact never armed and
// the 1.42M-token thread 400ed forever.
const OPENROUTER_ENDPOINT_TOO_LONG_BODY = JSON.stringify({
  error: {
    message:
      "This endpoint's maximum context length is 1000000 tokens. However, you requested about 1421428 tokens (1387828 of text input, 33600 of image input). Please reduce the length of either one, or use the context-compression plugin to compress your prompt automatically.",
    code: 400,
    metadata: {
      provider_name: null,
      previous_errors: [
        {
          code: 400,
          message:
            "This endpoint's maximum context length is 1000000 tokens. However, you requested about 1421428 tokens (1387828 of text input, 33600 of image input). Please reduce the length of either one, or use the context-compression plugin to compress your prompt automatically.",
        },
      ],
    },
  },
});

// ─── Unit: context-window-exceeded classification ──────────────────────────────
// mirrors is_context_window_error (codex-api/src/sse/responses.rs:513) — rs only
// matches the OpenAI-canonical code; the ts port also matches each provider's
// wording because OpenRouter/codeproxy pass the ORIGINAL provider error through.

describe("isContextWindowExceededText", () => {
  it("matches the OpenRouter/Anthropic incident payload", () => {
    expect(isContextWindowExceededText(OPENROUTER_PROMPT_TOO_LONG_BODY)).toBe(
      true,
    );
  });

  it("matches OpenRouter's own endpoint pre-check rejection (a1d14926)", () => {
    expect(isContextWindowExceededText(OPENROUTER_ENDPOINT_TOO_LONG_BODY)).toBe(
      true,
    );
  });

  it("matches the OpenAI canonical code and message", () => {
    expect(
      isContextWindowExceededText('{"code":"context_length_exceeded"}'),
    ).toBe(true);
    expect(
      isContextWindowExceededText(
        "Your input exceeds the context window of this model.",
      ),
    ).toBe(true);
  });

  it("does NOT match unrelated 4xx bodies", () => {
    expect(
      isContextWindowExceededText('{"error":{"message":"insufficient credits"}}'),
    ).toBe(false);
    expect(isContextWindowExceededText("Bad Request")).toBe(false);
  });
});

// ─── Integration: absolute context-window trigger ──────────────────────────────
// mirrors full_context_window_limit_reached (turn.rs:752-757): compaction must
// fire when TOTAL active context reaches the model window even though the
// growth-since-baseline budget (autoCompactTokenLimit) was never configured.

describe("full-context-window auto-compaction", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("compacts when total usage reaches contextWindow (no growth budget set)", async () => {
    const fetchMock = vi.fn();

    // Round 1 — tool call; usage total 810 < window 1000 → no compaction.
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseWithFullUsage(800, 810, [
          evResponseCreated("r1"),
          evFunctionCall("call-1", "get_goal", {}),
        ]),
      ),
    );

    // Round 2 — tool call; usage total 1020 ≥ window 1000 → compact.
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseWithFullUsage(1010, 1020, [
          evResponseCreated("r2"),
          evFunctionCall("call-2", "get_goal", {}),
        ]),
      ),
    );

    // Compaction request → summary.
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("compact"),
          evAssistantMessage("summary text"),
          evCompleted("compact"),
        ]),
      ),
    );

    // Round 3 — final reply.
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("r3"),
          evAssistantMessage("done"),
          evCompleted("r3"),
        ]),
      ),
    );

    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      contextWindow: 1000, // NOTE: no autoCompactTokenLimit at all
    });

    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "hi" }],
    });
    const compacted = await waitForEvent(
      codex,
      (msg) => msg.type === "ContextCompacted",
    );
    expect(compacted.type).toBe("ContextCompacted");
    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");

    expect(fetchMock).toHaveBeenCalledTimes(4);
    // The compaction request (index 2) carries the summarization prompt.
    const compactBody = JSON.parse(
      (fetchMock.mock.calls[2]![1] as RequestInit).body as string,
    ) as { input: unknown[] };
    expect(
      JSON.stringify(compactBody.input[compactBody.input.length - 1]),
    ).toContain(SUMMARIZATION_PROMPT.slice(0, 30));
  });
});

// ─── Integration: context-window-exceeded self-heal across turns ───────────────
// mirrors turn.rs:1045-1047 (set_total_tokens_full on ContextWindowExceeded) +
// run_pre_sampling_compact (turn.rs:149): a rejected request marks the session
// tokens FULL; the NEXT turn compacts BEFORE sampling instead of failing with
// the same 400 forever — this is exactly the production incident shape.

describe("context-window-exceeded self-heal", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("marks tokens full on the 400 and compacts before sampling on the next turn", async () => {
    const fetchMock = vi.fn();

    // Turn 1 — provider rejects the sampling request outright (the incident).
    fetchMock.mockResolvedValueOnce(
      makeErrorResponse(400, OPENROUTER_PROMPT_TOO_LONG_BODY),
    );

    // Turn 2 — FIRST request must be the pre-sampling compaction…
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("compact"),
          evAssistantMessage("summary text"),
          evCompleted("compact"),
        ]),
      ),
    );

    // …then the actual sampling request succeeds on the compacted history.
    fetchMock.mockResolvedValueOnce(
      makeSseResponse(
        sseFlat([
          evResponseCreated("t2"),
          evAssistantMessage("recovered"),
          evCompleted("t2"),
        ]),
      ),
    );

    vi.stubGlobal("fetch", fetchMock);

    const codex = new CodexThread({
      apiKey: "test",
      model: "gpt-4o",
      contextWindow: 1000,
    });

    // Turn 1 fails with the 400 (no retry — the same request would fail again).
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "hi" }],
    });
    const errorEvent = await waitForEvent(codex, (msg) => msg.type === "Error");
    expect(JSON.stringify(errorEvent)).toContain("prompt is too long");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // Turn 2 self-heals: compaction first, then sampling.
    await codex.submit({
      type: "UserInput",
      items: [{ type: "text", text: "continue" }],
    });
    const compacted = await waitForEvent(
      codex,
      (msg) => msg.type === "ContextCompacted",
    );
    expect(compacted.type).toBe("ContextCompacted");
    await waitForEvent(codex, (msg) => msg.type === "TurnComplete");

    expect(fetchMock).toHaveBeenCalledTimes(3);
    // Call 2 (index 1) is the compaction request (summarization prompt last)…
    const compactBody = JSON.parse(
      (fetchMock.mock.calls[1]![1] as RequestInit).body as string,
    ) as { input: unknown[] };
    expect(
      JSON.stringify(compactBody.input[compactBody.input.length - 1]),
    ).toContain(SUMMARIZATION_PROMPT.slice(0, 30));
    // …and call 3 (index 2) samples on the compacted history.
    const finalBody = JSON.parse(
      (fetchMock.mock.calls[2]![1] as RequestInit).body as string,
    ) as { input: unknown[] };
    expect(JSON.stringify(finalBody.input)).toContain(SUMMARY_PREFIX.slice(0, 30));
  });
});

// ─── Unit: compaction request trims itself out of the window ───────────────────
// mirrors compact.rs:232-241: when the compaction request ITSELF exceeds the
// window, drop the OLDEST history item and retry (prefix-preserving), instead
// of dying in the exact deadlock the production incident hit.

describe("runInlineAutoCompactTask — trim on context-window-exceeded", () => {
  function turnConfig(
    fetchMock: typeof fetch,
    tokenState: SessionTokenState,
  ): TurnConfig {
    return {
      apiKey: "test",
      baseUrl: "http://mock",
      model: "gpt-4o",
      fetch: fetchMock,
      contextWindow: 1000,
      tokenState,
      maxRetries: 0,
    };
  }

  it("drops the oldest item and retries until the summarizer fits", async () => {
    const history: ConversationItem[] = [
      { type: "message", role: "user", content: "oldest" },
      { type: "message", role: "assistant", content: "middle" },
      { type: "message", role: "user", content: "newest" },
    ];
    const tokenState = new SessionTokenState();
    const fetchMock = vi
      .fn()
      // First attempt (full history) → the compaction request is too big.
      .mockResolvedValueOnce(
        makeErrorResponse(400, OPENROUTER_PROMPT_TOO_LONG_BODY),
      )
      // Second attempt (oldest item dropped) → summary streams back.
      .mockResolvedValueOnce(
        makeSseResponse(
          sseFlat([
            evResponseCreated("compact"),
            evAssistantMessage("trimmed summary"),
            evCompleted("compact"),
          ]),
        ),
      );

    const summary = await runInlineAutoCompactTask(
      history,
      turnConfig(fetchMock as unknown as typeof fetch, tokenState),
    );

    expect(summary).toBe("trimmed summary");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    // Attempt 1 sent 3 history items + prompt; attempt 2 sent 2 + prompt.
    const firstInput = (
      JSON.parse(
        (fetchMock.mock.calls[0]![1] as RequestInit).body as string,
      ) as { input: unknown[] }
    ).input;
    const secondInput = (
      JSON.parse(
        (fetchMock.mock.calls[1]![1] as RequestInit).body as string,
      ) as { input: unknown[] }
    ).input;
    expect(firstInput).toHaveLength(4);
    expect(secondInput).toHaveLength(3);
    expect(JSON.stringify(secondInput)).not.toContain("oldest");
    // Replacement history was still built from the ORIGINAL user messages.
    expect(JSON.stringify(history)).toContain(SUMMARY_PREFIX.slice(0, 30));
    expect(JSON.stringify(history)).toContain("oldest");
  });

  it("marks tokens full and surfaces the error when nothing is left to trim", async () => {
    const history: ConversationItem[] = [
      { type: "message", role: "user", content: "only" },
    ];
    const tokenState = new SessionTokenState();
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        makeErrorResponse(400, OPENROUTER_PROMPT_TOO_LONG_BODY),
      );

    await expect(
      runInlineAutoCompactTask(
        history,
        turnConfig(fetchMock as unknown as typeof fetch, tokenState),
      ),
    ).rejects.toThrow(/prompt is too long/u);
    // mirrors compact.rs:242: set_total_tokens_full before surfacing.
    expect(tokenState.totalTokens).toBe(1000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("converges in a few probes on a massively oversized history (bulk trim, a1d14926)", async () => {
    // 400 items whose real token cost is 2× the chars/4 estimate (CJK-style
    // undercount). One-item-per-probe (plain rs behavior) would need ~370
    // probes; the geometric bulk trim must land within a handful.
    const history: ConversationItem[] = Array.from({ length: 400 }, (_, i) => ({
      type: "message" as const,
      role: "user" as const,
      content: `msg-${String(i).padStart(3, "0")}-${"x".repeat(52)}`,
    }));
    const tokenState = new SessionTokenState();
    // Simulated provider hard limit in REAL tokens, where real = chars/2
    // (twice the chars/4 estimate the trimmer budgets with).
    const HARD_LIMIT_REAL_TOKENS = 2000;
    const fetchMock = vi.fn().mockImplementation(
      async (_url: string, init: RequestInit) => {
        const realTokens = Math.ceil((init.body as string).length / 2);
        if (realTokens > HARD_LIMIT_REAL_TOKENS) {
          return makeErrorResponse(400, OPENROUTER_ENDPOINT_TOO_LONG_BODY);
        }
        return makeSseResponse(
          sseFlat([
            evResponseCreated("compact"),
            evAssistantMessage("bulk summary"),
            evCompleted("compact"),
          ]),
        );
      },
    );

    const summary = await runInlineAutoCompactTask(
      history,
      turnConfig(fetchMock as unknown as typeof fetch, tokenState),
    );

    expect(summary).toBe("bulk summary");
    // At least one rejected probe (the initial full-size request), and FAR
    // fewer round trips than the ~370 a one-item-per-probe loop would need.
    expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(fetchMock.mock.calls.length).toBeLessThanOrEqual(6);
    // The accepted request actually fit under the simulated hard limit.
    const lastBody = (
      fetchMock.mock.calls[fetchMock.mock.calls.length - 1]![1] as RequestInit
    ).body as string;
    expect(Math.ceil(lastBody.length / 2)).toBeLessThanOrEqual(
      HARD_LIMIT_REAL_TOKENS,
    );
    // Replacement history was still built from the ORIGINAL (untrimmed) one.
    expect(JSON.stringify(history)).toContain(SUMMARY_PREFIX.slice(0, 30));
  });
});
