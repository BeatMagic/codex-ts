/**
 * A compaction that lands LATE must not rewrite history.
 *
 * codex-rs cannot hit this: its compaction never overlaps another. codex-ts runs
 * turns concurrently (`void runTurn`) over one shared history, and its
 * summarizer request is not abortable, so two compactions in flight at once is
 * reachable. Applying the late one discards everything produced since the first
 * landed — assistant messages and tool calls are not user messages, so
 * collectUserMessages cannot bring them back.
 *
 * Observed without the epoch (2026-08-17): three ContextCompacted for ONE
 * legitimate trigger, the last two at 34k/600k where neither trigger line is
 * anywhere near.
 */

import { describe, expect, it, vi } from "vitest";

import { runInlineAutoCompactTask } from "../../src/compact.js";
import { AutoCompactWindow } from "../../src/state/auto_compact_window.js";
import type { ConversationItem } from "../../../thread-store/src/types.js";
import { evAssistantMessage, evCompleted, evResponseCreated, makeSseResponse, sseFlat } from "../common/lib.js";

function historyWith(texts: string[]): ConversationItem[] {
  return texts.map((text) => ({
    type: "message",
    role: "user",
    content: [{ type: "input_text", text }],
  })) as ConversationItem[];
}

function summarizerConfig(overrides: Record<string, unknown> = {}) {
  return {
    apiKey: "test",
    baseUrl: "http://localhost",
    model: "gpt-4o",
    fetch: vi.fn(async () =>
      makeSseResponse(
        sseFlat([
          evResponseCreated("s1"),
          evAssistantMessage("SUMMARY TEXT"),
          evCompleted("s1"),
        ]),
      ),
    ),
    ...overrides,
  } as never;
}

// =============================================================================
// AutoCompactWindow — the counter itself
// =============================================================================

describe("AutoCompactWindow compaction epoch", () => {
  it("starts at 0 and advances once per landed compaction", () => {
    const window = new AutoCompactWindow();
    expect(window.currentCompactionEpoch()).toBe(0);
    expect(window.bumpCompactionEpoch()).toBe(1);
    expect(window.currentCompactionEpoch()).toBe(1);
    window.bumpCompactionEpoch();
    expect(window.currentCompactionEpoch()).toBe(2);
  });

  it("is independent of the window number (rolling a window is not a landing)", () => {
    const window = new AutoCompactWindow();
    window.startNewContextWindow();
    expect(window.currentCompactionEpoch()).toBe(0);
  });
});

// =============================================================================
// runInlineAutoCompactTask — the guard at the mutation point
// =============================================================================

describe("runInlineAutoCompactTask staleness guard", () => {
  it("applies the summary and returns it when nothing else landed", async () => {
    const history = historyWith(["one", "two", "three"]);
    const before = history.length;
    const applied = await runInlineAutoCompactTask(
      history,
      summarizerConfig({ isStaleCompaction: () => false }),
    );
    expect(applied).toBe("SUMMARY TEXT");
    // History was rewritten in place.
    expect(history.length).not.toBe(before);
    expect(JSON.stringify(history)).toContain("SUMMARY TEXT");
  });

  it("returns null and leaves history BYTE-IDENTICAL when a compaction landed meanwhile", async () => {
    const history = historyWith(["one", "two", "three"]);
    const snapshot = JSON.stringify(history);
    const applied = await runInlineAutoCompactTask(
      history,
      summarizerConfig({ isStaleCompaction: () => true }),
    );
    expect(applied).toBeNull();
    expect(JSON.stringify(history)).toBe(snapshot);
  });

  it("without the hook it behaves exactly like codex-rs (never stale)", async () => {
    const history = historyWith(["one"]);
    const applied = await runInlineAutoCompactTask(history, summarizerConfig());
    expect(applied).toBe("SUMMARY TEXT");
  });

  it("asks the hook again AFTER the summarizer round trip", async () => {
    // The whole point of placing the final check at the mutation: a
    // compaction that lands while the request is in flight must still be
    // caught. (The check before sending only spares a request that is stale
    // already — see the request-gate tests.)
    const asked: string[] = [];
    let landed = false;
    const config = summarizerConfig({
      fetch: vi.fn(async () => {
        asked.push("fetch");
        landed = true; // another compaction lands mid-flight
        return makeSseResponse(
          sseFlat([
            evResponseCreated("s1"),
            evAssistantMessage("SUMMARY TEXT"),
            evCompleted("s1"),
          ]),
        );
      }),
      isStaleCompaction: () => {
        asked.push("check");
        return landed;
      },
    });
    const applied = await runInlineAutoCompactTask(historyWith(["one"]), config);
    expect(asked).toEqual(["check", "fetch", "check"]);
    expect(applied).toBeNull();
  });

  it("a compaction that is stale before it sends sends nothing", async () => {
    const fetch = vi.fn();
    const history = historyWith(["one"]);
    const snapshot = JSON.stringify(history);
    const applied = await runInlineAutoCompactTask(
      history,
      summarizerConfig({ fetch, isStaleCompaction: () => true }),
    );
    expect(applied).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
    expect(JSON.stringify(history)).toBe(snapshot);
  });
});
