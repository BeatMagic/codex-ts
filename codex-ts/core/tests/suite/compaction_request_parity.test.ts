/**
 * The compaction request is an EXTENSION of the sampling request.
 *
 * mirrors codex-rs/core/src/compact_remote_v2_attempt.rs:178-225 (#44255): the
 * summarizer request carries the same tools, base instructions and history as
 * the sampling request, with the compaction trigger appended last.
 *
 * Why it matters here: Anthropic renders tools → system → messages and caches
 * by prefix. The legacy body sent no tools and no reasoning, so it could never
 * read the main chain's cache — every compaction paid a full cold write of a
 * 600–900k history that is thrown away right after.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import {
  runInlineAutoCompactTask,
  SUMMARIZATION_PROMPT,
  SUMMARY_PREFIX,
} from "../../src/compact.js";
import type { SkillMetadata } from "../../src/skills.js";
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

type Body = Record<string, unknown> & { input: unknown[] };

function bodyOf(fetchMock: ReturnType<typeof vi.fn>, index: number): Body {
  const init = fetchMock.mock.calls[index]![1] as RequestInit;
  return JSON.parse(init.body as string) as Body;
}

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

const summaryResponse = (text = "summary text") =>
  makeSseResponse(
    sseFlat([
      evResponseCreated("compact"),
      evAssistantMessage(text),
      evCompleted("compact"),
    ]),
  );

const finalResponse = () =>
  makeSseResponse(
    sseFlat([evResponseCreated("r3"), evAssistantMessage("done"), evCompleted("r3")]),
  );

const SKILLS: SkillMetadata[] = [
  {
    name: "song-analyzer",
    description: "Analyze a song.",
    path: ".agents/skills/song-analyzer/SKILL.md",
  },
];

/**
 * One turn that compacts mid-turn: usage 1000 (baseline) → 2000 (growth 1000 ≥
 * 900) → summarizer (fetch index 2) → final reply (index 3). Context items
 * (AGENTS.md + skills catalog), a `$skill` injection and instructions are all
 * present, so every part of the sampling prefix is exercised.
 */
async function runCompactingTurn(
  extraConfig: Partial<ConstructorParameters<typeof CodexThread>[0]> = {},
  summarizerResponses: Response[] = [summaryResponse()],
) {
  const fetchMock = vi.fn();
  fetchMock
    .mockResolvedValueOnce(
      makeSseResponse(
        sseWithUsage(1000, [evResponseCreated("r1"), evFunctionCall("call-1", "get_goal", {})]),
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
    );
  for (const response of summarizerResponses) {
    fetchMock.mockResolvedValueOnce(response);
  }
  fetchMock.mockResolvedValueOnce(finalResponse());
  vi.stubGlobal("fetch", fetchMock);

  const codex = new CodexThread({
    apiKey: "test",
    model: "gpt-4o",
    instructions: "be useful",
    agentsMd: "PROJECT RULES",
    skills: SKILLS,
    loadSkillContent: async (skill) => `BODY:${skill.name}`,
    autoCompactTokenLimit: 900,
    maxRetries: 0,
    ...extraConfig,
  });
  await codex.submit({
    type: "UserInput",
    items: [{ type: "text", text: "use $song-analyzer please" }],
  });
  await waitForEvent(codex, (msg) => msg.type === "TurnComplete");
  return { codex, fetchMock };
}

let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  vi.unstubAllGlobals();
  warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
});

afterEach(() => {
  warn.mockRestore();
});

// =============================================================================
// The parity shape
// =============================================================================

describe("compaction request parity", () => {
  it("extends the last sampling request: same tools, reasoning, instructions and input prefix", async () => {
    const { fetchMock } = await runCompactingTurn();
    expect(fetchMock).toHaveBeenCalledTimes(4);
    const sampling = bodyOf(fetchMock, 1);
    const compaction = bodyOf(fetchMock, 2);

    // Same request properties, and nothing new (a tool_choice change alone
    // invalidates Anthropic's messages cache).
    expect(Object.keys(compaction).sort()).toEqual(Object.keys(sampling).sort());
    expect(compaction["tools"]).toEqual(sampling["tools"]);
    expect(compaction["reasoning"]).toEqual(sampling["reasoning"]);
    expect(compaction["instructions"]).toEqual(sampling["instructions"]);
    expect(compaction["model"]).toEqual(sampling["model"]);
    expect(compaction).not.toHaveProperty("tool_choice");

    // The sampling input is a strict prefix: context items, the turn's skill
    // injection, history — then what was recorded since, then the prompt.
    expect(compaction.input.slice(0, sampling.input.length)).toEqual(sampling.input);
    expect(JSON.stringify(sampling.input)).toContain("PROJECT RULES");
    expect(JSON.stringify(sampling.input)).toContain("BODY:song-analyzer");
    const appended = compaction.input.slice(sampling.input.length, -1);
    expect(appended).toEqual([
      {
        type: "function_call",
        call_id: "call-2",
        name: "update_plan",
        arguments: JSON.stringify({ plan: [{ step: "s", status: "completed" }] }),
      },
      { type: "function_call_output", call_id: "call-2", output: "Plan updated" },
    ]);
    expect(warn).not.toHaveBeenCalled();
  });

  // The ace-mv host recognizes a compaction request by an EXACT match on the
  // last input item: role "user", content the prompt string
  // (compaction-single-flight.ts isCompactionRequest). The parity shape must
  // not move or reshape it.
  it("keeps the prompt as the LAST input item, in the exact legacy shape", async () => {
    const compactPrompt = "HOST COMPACT PROMPT";
    const { fetchMock } = await runCompactingTurn({ compactPrompt });
    const compaction = bodyOf(fetchMock, 2);
    expect(compaction["tools"]).toBeDefined();
    expect(compaction.input[compaction.input.length - 1]).toEqual({
      type: "message",
      role: "user",
      content: compactPrompt,
    });
    // No other item carries the prompt.
    expect(
      compaction.input.filter((item) => JSON.stringify(item).includes(compactPrompt)),
    ).toHaveLength(1);
  });

  it("compactionRequestParity: false sends the legacy body", async () => {
    const { fetchMock } = await runCompactingTurn({ compactionRequestParity: false });
    const compaction = bodyOf(fetchMock, 2);
    expect(Object.keys(compaction).sort()).toEqual(
      ["input", "instructions", "model", "stream"].sort(),
    );
    // History + prompt only: no context items, no skill injection.
    expect(JSON.stringify(compaction.input)).not.toContain("PROJECT RULES");
    expect(JSON.stringify(compaction.input)).not.toContain("BODY:song-analyzer");
    expect(compaction.input[compaction.input.length - 1]).toEqual({
      type: "message",
      role: "user",
      content: SUMMARIZATION_PROMPT,
    });
  });

  it("a direct caller that passes no tools gets the legacy body", async () => {
    const fetchMock = vi.fn(async () => summaryResponse());
    const history: ConversationItem[] = [
      { type: "message", role: "user", content: "a" },
    ];
    await runInlineAutoCompactTask(history, {
      apiKey: "test",
      baseUrl: "http://localhost",
      model: "gpt-4o",
      fetch: fetchMock as unknown as typeof fetch,
      contextItems: [{ type: "message", role: "user", content: "CTX" }],
    });
    const body = bodyOf(fetchMock, 0);
    expect(body).not.toHaveProperty("tools");
    expect(body).not.toHaveProperty("reasoning");
    expect(body.input).toEqual([
      { type: "message", role: "user", content: "a" },
      { type: "message", role: "user", content: SUMMARIZATION_PROMPT },
    ]);
  });
});

// =============================================================================
// Fallback to the legacy shape
// =============================================================================

describe("compaction request parity — fallback", () => {
  it("a tool call instead of a summary switches to the legacy body and is never dispatched", async () => {
    // Text AND a call: only the tool-call check can catch this one.
    const toolCalling = makeSseResponse(
      sseFlat([
        evResponseCreated("compact-1"),
        evAssistantMessage("Let me look first."),
        evFunctionCall("rogue", "update_plan", {
          plan: [{ step: "x", status: "pending" }],
        }),
        evCompleted("compact-1"),
      ]),
    );
    const { fetchMock } = await runCompactingTurn({}, [
      toolCalling,
      summaryResponse("legacy summary"),
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(5);

    const parityAttempt = bodyOf(fetchMock, 2);
    const legacyAttempt = bodyOf(fetchMock, 3);
    expect(parityAttempt).toHaveProperty("tools");
    expect(Object.keys(legacyAttempt).sort()).toEqual(
      ["input", "instructions", "model", "stream"].sort(),
    );
    expect(legacyAttempt.input[legacyAttempt.input.length - 1]).toEqual({
      type: "message",
      role: "user",
      content: SUMMARIZATION_PROMPT,
    });
    // Legacy input = the history the parity attempt sent, minus the prefix.
    expect(JSON.stringify(legacyAttempt.input)).not.toContain("PROJECT RULES");
    expect(legacyAttempt.input.slice(0, -1)).toEqual(
      parityAttempt.input.slice(
        parityAttempt.input.length - legacyAttempt.input.length,
        -1,
      ),
    );

    // The legacy summary is what landed; the rogue call left no trace.
    const finalInput = JSON.stringify(bodyOf(fetchMock, 4).input);
    expect(finalInput).toContain(`${SUMMARY_PREFIX.slice(0, 30)}`);
    expect(finalInput).toContain("legacy summary");
    expect(finalInput).not.toContain("rogue");
    expect(finalInput).not.toContain("Let me look first.");
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0]![0])).toContain("tool call");
  });

  it("an empty summary from the parity shape switches to the legacy body", async () => {
    const empty = makeSseResponse(
      sseFlat([evResponseCreated("compact-1"), evCompleted("compact-1")]),
    );
    // maxRetries 0: only the shape switch — not a retry — can reach attempt 2.
    const { fetchMock } = await runCompactingTurn({}, [
      empty,
      summaryResponse("legacy summary"),
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(5);
    expect(bodyOf(fetchMock, 2)).toHaveProperty("tools");
    expect(bodyOf(fetchMock, 3)).not.toHaveProperty("tools");
    expect(JSON.stringify(bodyOf(fetchMock, 4).input)).toContain("legacy summary");
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0]![0])).toContain("no summary text");
  });

  it("switches once: a legacy attempt that also fails surfaces the error", async () => {
    const empty = () =>
      makeSseResponse(sseFlat([evResponseCreated("c"), evCompleted("c")]));
    const fetchMock = vi.fn();
    fetchMock.mockResolvedValueOnce(empty()).mockResolvedValueOnce(empty());
    const history: ConversationItem[] = [
      { type: "message", role: "user", content: "a" },
    ];
    await expect(
      runInlineAutoCompactTask(history, {
        apiKey: "test",
        baseUrl: "http://localhost",
        model: "gpt-4o",
        fetch: fetchMock as unknown as typeof fetch,
        maxRetries: 0,
        tools: [],
      }),
    ).rejects.toThrow(/empty summary/u);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(history).toEqual([{ type: "message", role: "user", content: "a" }]);
    expect(warn).toHaveBeenCalledTimes(1);
  });
});
