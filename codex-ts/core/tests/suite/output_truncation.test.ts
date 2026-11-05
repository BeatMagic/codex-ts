/**
 * mirrors codex-rs/utils/string/src/truncate/tests.rs and
 *         codex-rs/utils/output-truncation/src/truncate_tests.rs (Bytes policy)
 *
 * Per-tool output truncation: an opted-in tool's output is middle-truncated
 * before it enters the model-visible history, the rollout keeps the original,
 * and a resume re-applies the policy to the raw items it replays (but not to a
 * checkpoint's replacement_history, already the model-visible form).
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { CodexThread } from "../../src/codex_thread.js";
import {
  DEFAULT_TOOL_OUTPUT_MAX_BYTES,
  truncateFunctionCallOutput,
  truncateFunctionOutputItems,
  truncateMiddleChars,
} from "../../src/tools/output_truncation.js";
import { ToolRouter } from "../../src/tools/router.js";
import type { CustomTool } from "../../src/tools/router.js";
import { utf8ByteLength } from "../../src/state/token_state.js";
import { GoalToolExecutor } from "../../../ext/goal/src/tool.js";
import { InMemoryIoBackend } from "../../../thread-store/src/io_backend.js";
import { LiveThread } from "../../../thread-store/src/live_thread.js";
import { LocalThreadStore } from "../../../thread-store/src/local_thread_store.js";
import type {
  ConversationItem,
  FunctionCallOutputContentItem,
  RolloutItem,
} from "../../../thread-store/src/types.js";
import {
  evAssistantMessage,
  evCompleted,
  evFunctionCall,
  evResponseCreated,
  makeSseResponse,
  sseFlat,
  waitForEvent,
} from "../common/lib.js";

// ─── Unit: truncateMiddleChars (truncate_middle_chars) ────────────────────────

describe("truncateMiddleChars", () => {
  it("matches the rs byte-policy vectors", () => {
    expect(truncateMiddleChars("example output", 1)).toBe("…13 chars truncated…t");
    expect(
      truncateMiddleChars("this is an example of a long output that should be truncated", 30),
    ).toBe("this is an exam…30 chars truncated…ld be truncated");
    expect(
      truncateMiddleChars(
        "this is an example of a long output that should be truncated\nalso some other line",
        30,
      ),
    ).toBe("this is an exam…51 chars truncated…some other line");
    expect(truncateMiddleChars("😀😀😀😀😀😀😀😀😀😀\nsecond line with text\n", 20)).toBe(
      "😀😀…21 chars truncated…with text\n",
    );
  });

  it("cuts on code-point boundaries (split_string vectors)", () => {
    expect(truncateMiddleChars("😀abc😀", 10)).toBe("😀a…1 chars truncated…c😀");
    expect(truncateMiddleChars("😀😀😀😀😀", 2)).toBe("…5 chars truncated…");
    expect(truncateMiddleChars("😀😀😀😀😀", 14)).toBe("😀…3 chars truncated…😀");
    expect(truncateMiddleChars("😀😀😀😀😀", 16)).toBe("😀😀…1 chars truncated…😀😀");
    // UTF-8 bytes, not UTF-16 units: each 音 is 3 bytes.
    expect(truncateMiddleChars("音".repeat(10), 10)).toBe("音…8 chars truncated…音");
  });

  it("returns the input itself when it fits", () => {
    const text = "x".repeat(100);
    expect(truncateMiddleChars(text, 100)).toBe(text);
    expect(truncateMiddleChars("", 0)).toBe("");
  });

  it("never splits a surrogate pair and keeps at most maxBytes of content", () => {
    const text = "a😀音b😀😀c音音d😀e".repeat(3);
    const loneSurrogate = /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/u;
    for (let maxBytes = 0; maxBytes <= utf8ByteLength(text); maxBytes += 1) {
      const out = truncateMiddleChars(text, maxBytes);
      // Check each side of the marker on its own, so a pair split across
      // the cut cannot hide by joining up again.
      for (const side of out.split(/…\d+ chars truncated…/u)) {
        expect(loneSurrogate.test(side)).toBe(false);
      }
      const kept = out.replace(/…\d+ chars truncated…/u, "");
      expect(utf8ByteLength(kept)).toBeLessThanOrEqual(Math.max(maxBytes, utf8ByteLength(text)));
      if (out !== text) expect(utf8ByteLength(kept)).toBeLessThanOrEqual(maxBytes);
    }
  });
});

// ─── Unit: content items (truncate_function_output_items_with_policy) ─────────

describe("truncateFunctionOutputItems", () => {
  const text = (value: string): FunctionCallOutputContentItem => ({ type: "input_text", text: value });
  const image: FunctionCallOutputContentItem = {
    type: "input_image",
    image_url: "https://r2.example/preview.png?sig=abc",
  };

  it("spends the budget on text in order, keeps images, counts omitted text items", () => {
    const chunk =
      "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega.\n";
    const limit = utf8ByteLength(chunk) * 3;
    const items = [
      text(chunk),
      text(chunk),
      image,
      text(chunk.repeat(10)),
      text(chunk),
      text(chunk),
    ];
    const out = truncateFunctionOutputItems(items, limit);
    expect(out).toHaveLength(5);
    expect(out[0]).toEqual(text(chunk));
    expect(out[1]).toEqual(text(chunk));
    expect(out[2]).toEqual(image);
    expect(out[3]).toMatchObject({ type: "input_text", text: expect.stringContaining("chars truncated") });
    expect(out[4]).toEqual(text("[omitted 2 text items ...]"));
  });

  it("returns the very same array when everything fits (view_image stays byte-identical)", () => {
    const viewImage = [image, text('Viewing image asset "a1" (active version v2)')];
    expect(truncateFunctionOutputItems(viewImage, DEFAULT_TOOL_OUTPUT_MAX_BYTES)).toBe(viewImage);
    expect(truncateFunctionCallOutput(viewImage, DEFAULT_TOOL_OUTPUT_MAX_BYTES)).toBe(viewImage);
  });

  it("passes an off-type output through untouched", () => {
    const odd = { not: "an output" } as unknown as string;
    expect(truncateFunctionCallOutput(odd, 1)).toBe(odd);
  });
});

describe("ToolRouter.maxOutputBytesFor", () => {
  const tool = (maxOutputBytes: number | undefined, name = "t"): CustomTool => ({
    name,
    spec: () => ({
      type: "function",
      tool: { name, description: "", parameters: { type: "object", properties: {} }, strict: false },
    }),
    execute: async () => "",
    maxOutputBytes,
  });

  it("is the tool's budget when valid, else no policy; built-ins never have one", () => {
    expect(new ToolRouter(new GoalToolExecutor("g"), [tool(1000.7)]).maxOutputBytesFor("t")).toBe(1000);
    expect(new ToolRouter(new GoalToolExecutor("g"), [tool(undefined)]).maxOutputBytesFor("t")).toBeUndefined();
    expect(new ToolRouter(new GoalToolExecutor("g"), [tool(-1)]).maxOutputBytesFor("t")).toBeUndefined();
    expect(new ToolRouter(new GoalToolExecutor("g"), [tool(Number.NaN)]).maxOutputBytesFor("t")).toBeUndefined();
    expect(new ToolRouter(new GoalToolExecutor("g"), [tool(5)]).maxOutputBytesFor("update_plan")).toBeUndefined();
    // A custom tool shadowed by a built-in of the same name: the built-in runs.
    expect(
      new ToolRouter(new GoalToolExecutor("g"), [tool(5, "update_plan")]).maxOutputBytesFor("update_plan"),
    ).toBeUndefined();
    expect(
      new ToolRouter(new GoalToolExecutor("g"), [tool(5, "get_goal")]).maxOutputBytesFor("get_goal"),
    ).toBeUndefined();
    // …unless goal tools are off, when the name is an ordinary custom tool.
    expect(
      new ToolRouter(new GoalToolExecutor("g"), [tool(5, "get_goal")], {
        enableGoalTools: false,
      }).maxOutputBytesFor("get_goal"),
    ).toBe(5);
  });
});

// ─── Integration ──────────────────────────────────────────────────────────────

type Body = { input: ConversationItem[] };

function returningTool(
  name: string,
  output: string | FunctionCallOutputContentItem[],
  maxOutputBytes?: number,
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
    execute: async () => output,
    ...(maxOutputBytes !== undefined ? { maxOutputBytes } : {}),
  };
}

/** First request → the given calls; every later request → a plain reply. */
function scriptedFetch(calls: object[]) {
  const bodies: Body[] = [];
  const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
    bodies.push(JSON.parse(init.body as string) as Body);
    const events =
      bodies.length === 1 && calls.length > 0 ? calls : [evAssistantMessage("ok")];
    return makeSseResponse(sseFlat([evResponseCreated("r"), ...events, evCompleted("r")]));
  });
  return { fetchMock, bodies };
}

function outputOf(items: ConversationItem[], callId: string): unknown {
  const found = items.filter(
    (item) => item.type === "function_call_output" && item.call_id === callId,
  );
  expect(found).toHaveLength(1);
  return (found[0] as { output: unknown }).output;
}

async function persisted(io: InMemoryIoBackend, threadId: string) {
  return new LiveThread(threadId, new LocalThreadStore(io)).loadConversationHistory();
}

const BIG = `${"A".repeat(3_000)}${"B".repeat(3_000)}`;

describe("a tool that opted into truncation", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("is truncated for the model, persisted whole, and other tools are untouched", async () => {
    const { fetchMock, bodies } = scriptedFetch([
      evFunctionCall("call-big", "big_tool", {}),
      evFunctionCall("call-plain", "plain_tool", {}),
    ]);
    vi.stubGlobal("fetch", fetchMock);
    const io = new InMemoryIoBackend();
    const codex = new CodexThread({
      apiKey: "test",
      model: "m",
      ioBackend: io,
      customTools: [returningTool("big_tool", BIG, 1_000), returningTool("plain_tool", BIG)],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    const sent = bodies[1]!.input;
    expect(outputOf(sent, "call-big")).toBe(truncateMiddleChars(BIG, 1_000));
    expect(outputOf(sent, "call-big")).toBe(`${"A".repeat(500)}…5000 chars truncated…${"B".repeat(500)}`);
    expect(outputOf(sent, "call-plain")).toBe(BIG);

    // The transcript a host replays keeps the original.
    const raw = await persisted(io, codex.id);
    expect(outputOf(raw, "call-big")).toBe(BIG);
    expect(outputOf(raw, "call-plain")).toBe(BIG);
  });

  it("an image array keeps its images and only its text is cut", async () => {
    const caption = "c".repeat(5_000);
    const viewImage: FunctionCallOutputContentItem[] = [
      { type: "input_image", image_url: "https://r2.example/a.png" },
      { type: "input_text", text: caption },
    ];
    const { fetchMock, bodies } = scriptedFetch([evFunctionCall("call-img", "view_image", {})]);
    vi.stubGlobal("fetch", fetchMock);
    const codex = new CodexThread({
      apiKey: "test",
      model: "m",
      customTools: [returningTool("view_image", viewImage, 1_000)],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "look" }] });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");

    expect(outputOf(bodies[1]!.input, "call-img")).toEqual([
      { type: "input_image", image_url: "https://r2.example/a.png" },
      { type: "input_text", text: truncateMiddleChars(caption, 1_000) },
    ]);
  });

  it("the token estimator sees the truncated form: no compaction for a cut output", async () => {
    // The same shape as token_state.test.ts's "compacts when the tool output
    // pushes the active context over the window" (40 000 chars on an
    // 8 000-token window) — but the tool opted into rs's budget.
    const { fetchMock, bodies } = scriptedFetch([evFunctionCall("call-1", "read_file", {})]);
    vi.stubGlobal("fetch", fetchMock);
    const codex = new CodexThread({
      apiKey: "test",
      model: "m",
      contextWindow: 8_000,
      customTools: [returningTool("read_file", "x".repeat(40_000), DEFAULT_TOOL_OUTPUT_MAX_BYTES)],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "read it" }] });
    const settled = await waitForEvent(
      codex,
      (m) => m.type === "ContextCompacted" || m.type === "TurnComplete",
    );
    expect(settled.type).toBe("TurnComplete");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(outputOf(bodies[1]!.input, "call-1"))).toContain("chars truncated");
  });
});

describe("resume re-applies the policy", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  async function firstSession(io: InMemoryIoBackend): Promise<{ id: string; sent: unknown }> {
    const { fetchMock, bodies } = scriptedFetch([evFunctionCall("call-big", "big_tool", {})]);
    vi.stubGlobal("fetch", fetchMock);
    const codex = new CodexThread({
      apiKey: "test",
      model: "m",
      ioBackend: io,
      customTools: [returningTool("big_tool", BIG, 1_000)],
    });
    await codex.submit({ type: "UserInput", items: [{ type: "text", text: "go" }] });
    await waitForEvent(codex, (m) => m.type === "TurnComplete");
    return { id: codex.id, sent: outputOf(bodies[1]!.input, "call-big") };
  }

  async function resumedRequest(
    io: InMemoryIoBackend,
    threadId: string,
    tool: CustomTool,
  ): Promise<ConversationItem[]> {
    const { fetchMock, bodies } = scriptedFetch([]);
    vi.stubGlobal("fetch", fetchMock);
    const resumed = await CodexThread.create({
      apiKey: "test",
      model: "m",
      threadId,
      ioBackend: io,
      customTools: [tool],
    });
    await resumed.submit({ type: "UserInput", items: [{ type: "text", text: "again" }] });
    await waitForEvent(resumed, (m) => m.type === "TurnComplete");
    return bodies[0]!.input;
  }

  it("a resumed thread sends the same truncated bytes the live one did", async () => {
    const io = new InMemoryIoBackend();
    const first = await firstSession(io);
    const input = await resumedRequest(io, first.id, returningTool("big_tool", "", 1_000));
    expect(outputOf(input, "call-big")).toBe(first.sent);
    // …while the raw transcript still has the whole output.
    expect(outputOf(await persisted(io, first.id), "call-big")).toBe(BIG);
  });

  it("the policy is the resuming host's: a tool without one gets the whole output", async () => {
    const io = new InMemoryIoBackend();
    const first = await firstSession(io);
    const input = await resumedRequest(io, first.id, returningTool("big_tool", ""));
    expect(outputOf(input, "call-big")).toBe(BIG);
  });

  it("a checkpoint's replacement_history is not truncated again; the raw suffix is", async () => {
    const io = new InMemoryIoBackend();
    const store = new LocalThreadStore(io);
    const alreadyCut = truncateMiddleChars(BIG, 1_000);
    const call = (id: string): ConversationItem => ({
      type: "function_call",
      call_id: id,
      name: "big_tool",
      arguments: "{}",
    });
    const items: RolloutItem[] = [
      {
        kind: "Compacted",
        compacted: {
          v: 1,
          message: "S",
          replacement_history: [
            { type: "message", role: "user", content: "S" },
            call("in-checkpoint"),
            { type: "function_call_output", call_id: "in-checkpoint", output: alreadyCut },
          ],
        },
      },
      { kind: "ConversationItem", item: call("after") },
      {
        kind: "ConversationItem",
        item: { type: "function_call_output", call_id: "after", output: BIG },
      },
    ];
    await store.appendItems({ threadId: "ckpt", items });

    const input = await resumedRequest(io, "ckpt", returningTool("big_tool", "", 1_000));
    // Re-truncating would replace its marker with "…N chars truncated…" for
    // the marker itself, and change the bytes the cache was written with.
    expect(outputOf(input, "in-checkpoint")).toBe(alreadyCut);
    expect(outputOf(input, "after")).toBe(alreadyCut);
  });
});
