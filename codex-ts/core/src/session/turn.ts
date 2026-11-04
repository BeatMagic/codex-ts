/**
 * mirrors codex-rs/core/src/session/turn.rs
 *
 * runTurn() — the core agent sampling loop:
 *   1. Build Responses API request (history + tool specs)
 *   2. Stream SSE response
 *   3. Collect assistant text and function calls
 *   4. Dispatch tool calls via ToolRouter
 *   5. Emit side-effect events
 *   6. Loop until the model produces no more tool calls
 */

import type { EventMsg } from "../../../protocol/src/protocol.js";
import type { UserInput } from "../../../protocol/src/user_input.js";
import { toolSpecToRequestJson } from "../../../tools/src/tool_spec.js";
import { ToolRouter } from "../tools/router.js";
import {
  canonicalizeStoredArguments,
  parseToolArguments,
} from "../tools/argument_repair.js";
import { extractSkillMentions, renderSkillInjection } from "../skills.js";
import type { SkillMetadata } from "../skills.js";
import type { PendingInputs } from "../tools/handlers/request_user_input.js";
import type { LiveThread } from "../../../thread-store/src/live_thread.js";
import type {
  ConversationItem,
  UserContentPart,
} from "../../../thread-store/src/types.js";
import { parseSseStream } from "./sse.js";
import {
  classifyStreamFailure,
  computeRetryDelay,
  DEFAULT_MAX_RETRIES,
  isContextWindowExceededError,
  isRetryableError,
  parseRetryAfter,
  ResponsesApiError,
  sleep,
} from "./retry.js";
import { runInlineAutoCompactTask } from "../compact.js";
import { normalizeHistory } from "../normalize.js";
import { AutoCompactWindow } from "../state/auto_compact_window.js";
import {
  recomputeTokenUsage,
  SessionTokenState,
  totalOnlyTokenCountEvent,
} from "../state/token_state.js";

// ConversationItem from thread-store is the canonical type for both
// history sent to the API and items persisted to the store.
type HistoryItem = ConversationItem;

// ─── TurnConfig ──────────────────────────────────────────────────────────────

export interface TurnConfig {
  apiKey: string;
  baseUrl: string;
  /** Custom fetch for the Responses API call; defaults to the global fetch. */
  fetch?: typeof fetch | undefined;
  model: string;
  /** Reasoning effort for the Responses API. codeproxy maps `reasoning.effort`
   *  to the upstream thinking budget — REQUIRED for Gemini to stream reasoning
   *  (thought) deltas (surfaced as ReasoningContentDelta). Defaults to "medium"
   *  when omitted. */
  reasoningEffort?: "minimal" | "low" | "medium" | "high" | "xhigh" | undefined;
  instructions?: string | undefined;
  /** Discovered skills, for `$skill-name` full-body injection (Layer 2). */
  skills?: SkillMetadata[] | undefined;
  /**
   * Reads a skill's full SKILL.md on demand. Host-provided because a browser
   * has no filesystem (mirrors how IoBackend is injected).
   */
  loadSkillContent?: ((skill: SkillMetadata) => Promise<string>) | undefined;
  /**
   * Input-token threshold that triggers inline auto-compaction.
   * mirrors model_auto_compact_token_limit in codex-rs TurnContext config.
   *
   * In BodyAfterPrefix mode (the default) this is compared against tokens
   * added since the last compaction, not total context size.
   * Typical value: context_window × 0.9.
   * Omit to disable auto-compaction.
   */
  autoCompactTokenLimit?: number | undefined;
  /**
   * Session-scoped compaction window. The thread owns ONE instance and threads it
   * through every runTurn so BodyAfterPrefix measures context growth ACROSS turns
   * (mirrors codex-rs, where the window lives in session state, read via
   * n_snapshot). Omit → a fresh per-turn window (growth measured only within the
   * turn) — fine for one-shot direct calls / tests. A multi-turn interactive thread
   * MUST pass a shared instance, otherwise cross-turn accumulation never counts
   * toward the limit, compaction never fires, and the context grows past the
   * model's window → "prompt is too long".
   */
  compactWindow?: AutoCompactWindow | undefined;
  /**
   * The model's full context window in tokens (mirrors model_context_window in
   * codex-rs TurnContext). Enables the SECOND compaction trigger: compact when
   * the TOTAL active context reaches the window — independent of the growth
   * budget above (turn.rs:752-757) — and the context-window-exceeded self-heal
   * (a rejected request marks tokens full so the next turn compacts first,
   * turn.rs:1045-1047). Omit to disable both (growth-only behaviour).
   */
  contextWindow?: number | undefined;
  /**
   * Session-scoped token accounting (mirrors sess.get_total_token_usage state).
   * The thread owns ONE instance, threaded through every runTurn like
   * compactWindow, so pre-sampling checks see usage from previous turns.
   * Omit → a fresh per-turn state (one-shot/tests).
   */
  tokenState?: SessionTokenState | undefined;
  /**
   * Max retries for transient Responses request/stream failures — network
   * errors, 5xx / 408 / 409 / 429, or a stream dropped before any visible
   * output. Each retry waits an exponential backoff (200ms × 2^(n-1) ± 10%
   * jitter), honoring a server Retry-After when present. mirrors codex-rs
   * stream/request_max_retries. Defaults to DEFAULT_MAX_RETRIES (5).
   */
  maxRetries?: number | undefined;
  /**
   * Overrides the inline-compaction summarization prompt. mirrors
   * `config.compact_prompt` (compact.rs:118-123) — omit to use the bundled
   * SUMMARIZATION_PROMPT.
   */
  compactPrompt?: string | undefined;
  /**
   * Turn-scoped context messages prepended to `input` ahead of history,
   * mirroring codex-rs's contextual user fragments: user_instructions
   * (developer + AGENTS.md) and the skills catalog ride here as discrete
   * messages rather than baked into the `instructions` field. Not persisted.
   */
  contextItems?: HistoryItem[] | undefined;
}

// ─── Auto-compaction status + orchestration ──────────────────────────────────

/**
 * mirrors context_window_token_status (core/src/session/context_window.rs:23 —
 * the module upstream extracted `auto_compact_token_status` into), BodyAfterPrefix
 * scope: the limit is reached when EITHER the growth since the window baseline
 * hits `autoCompactTokenLimit`, OR the total active context reaches the model's
 * full `contextWindow`. The second condition is what keeps a session with a
 * large baseline (or a force-filled one after a context-window-exceeded
 * rejection) from sailing past the model's window.
 *
 * rs additionally widens the scope limit by
 * `token_budget.auto_compact_fallback_buffer_tokens` and reports
 * `base_window_tokens_remaining`, both only used by its feature-gated
 * token-budget reminders (session/token_budget.rs). codex-ts has no
 * `token_budget` config, so the buffer is 0 — identical behaviour — and the
 * remaining-tokens readout is left out; hosts can derive it from TokenCount's
 * `model_context_window` + `total_token_usage`.
 */
function autoCompactTokenStatus(
  config: TurnConfig,
  compactWindow: AutoCompactWindow,
  tokenState: SessionTokenState,
  history: ConversationItem[],
): {
  activeContextTokens: number;
  tokenLimitReached: boolean;
} {
  // getTotalTokenUsage, not the bare total: the items appended since the last
  // sampled response (this turn's user message, and — at the mid-turn
  // checkpoint — the tool outputs just recorded) are part of what the NEXT
  // request carries but are invisible to the server's usage number. rs feeds
  // the same adjusted figure into both limbs of this check
  // (session/context_window.rs:27,46,76).
  const activeContextTokens = tokenState.getTotalTokenUsage(history);
  const scopeTokens = compactWindow.bodyAfterPrefix(activeContextTokens);
  const scopeLimit =
    config.autoCompactTokenLimit ?? Number.POSITIVE_INFINITY;
  const fullContextWindowLimitReached =
    config.contextWindow !== undefined &&
    activeContextTokens >= config.contextWindow;
  return {
    activeContextTokens,
    tokenLimitReached:
      scopeTokens >= scopeLimit || fullContextWindowLimitReached,
  };
}

/**
 * Emit a TokenCount and persist it, so a resumed thread can seed its token
 * state from the last one (mirrors rs persisting EventMsg::TokenCount,
 * rollout/src/policy.rs). Persistence is queued, never awaited: the ordered
 * queue keeps it after every append issued before it.
 */
function recordTokenCount(
  msg: EventMsg,
  emitEvent: (msg: EventMsg) => void,
  liveThread: LiveThread | undefined,
): void {
  emitEvent(msg);
  void liveThread?.appendEvent(msg);
}

/**
 * Mark the session context FULL after a context-window-exceeded rejection and
 * record it. mirrors set_total_tokens_full → send_token_count_event
 * (session/mod.rs): the mark is persisted, so the self-heal (the next turn
 * compacts first) survives an app restart instead of dying with the process.
 */
function markContextFull(
  config: TurnConfig,
  tokenState: SessionTokenState,
  emitEvent: (msg: EventMsg) => void,
  liveThread: LiveThread | undefined,
): void {
  if (config.contextWindow === undefined) return;
  tokenState.setFull(config.contextWindow);
  recordTokenCount(
    totalOnlyTokenCountEvent(config.contextWindow, config.contextWindow),
    emitEvent,
    liveThread,
  );
}

/**
 * Run one inline compaction and reseed the token bookkeeping.
 * mirrors run_auto_compact + recompute_token_usage (compact.rs:359-386 +
 * session/mod.rs:3794-3831): after history is rewritten, the session total and
 * the NEW window's baseline both restart from a byte-based estimate of the
 * compacted context (replaced by real server usage on the next sampled
 * response).
 *
 * The rewrite is checkpointed (mirrors replace_compacted_history persisting
 * Compacted in the same lock section as the in-memory swap): a resumed thread
 * starts from the compacted history instead of replaying every raw item.
 */
async function runCompactAndRecompute(
  history: ConversationItem[],
  config: TurnConfig,
  compactWindow: AutoCompactWindow,
  tokenState: SessionTokenState,
  emitEvent: (msg: EventMsg) => void,
  liveThread: LiveThread | undefined,
  abortSignal: AbortSignal | undefined,
): Promise<void> {
  // The window's generation, read BEFORE the summarizer runs; see
  // AutoCompactWindow.currentCompactionEpoch for why a late compaction must not
  // be allowed to rewrite history.
  const epochAtEntry = compactWindow.currentCompactionEpoch();
  let applied: string | null;
  try {
    applied = await runInlineAutoCompactTask(history, {
      ...config,
      abortSignal,
      isStaleCompaction: () =>
        compactWindow.currentCompactionEpoch() !== epochAtEntry,
      // Runs in the SAME tick as the history swap. Everything that must be
      // atomic with the rewrite happens here: the epoch bump (so a concurrent
      // summarizer's stale check can no longer pass in a microtask window),
      // the new window, the reseed, and the checkpoint write — queued before
      // any append a concurrent turn can issue after the swap.
      onHistoryReplaced: (replacement, summaryMessage) => {
        // Bump AFTER a replacement actually landed, so every compaction still
        // in flight becomes stale from this instant.
        compactWindow.bumpCompactionEpoch();
        // mirrors advance_auto_compact_window (compact.rs:359) followed by
        // replace_compacted_history → replace_history's clear_prefill
        // (state/session.rs:122): one call does both here.
        const window = compactWindow.startNewContextWindow();
        const estimated = recomputeTokenUsage({
          items: [...(config.contextItems ?? []), ...replacement],
          instructions: config.instructions,
          tokenState,
          compactWindow,
          contextWindow: config.contextWindow,
        });
        void liveThread?.appendCompacted({
          v: 1,
          message: summaryMessage,
          replacement_history: replacement,
          window_number: window.windowNumber,
          first_window_id: window.ids.firstWindowId,
          previous_window_id: window.ids.previousWindowId,
          window_id: window.ids.windowId,
          token_total_after: estimated,
        });
        // Emitted only AFTER history has actually been rebuilt+replaced — not
        // optimistically before the summarizer runs. A stalled, aborted or
        // superseded compaction throws (or returns null) and emits NO
        // ContextCompacted, so a failed compaction can no longer advertise a
        // phantom success (2026-07-31: nine ContextCompacted while history
        // collapsed exactly once).
        emitEvent({ type: "ContextCompacted", event: {} });
        // rs's recompute ends in send_token_count_event (session/mod.rs:4031).
        // Without it a host's context meter keeps showing the PRE-compaction
        // number until the next sampled response — the one moment a user is
        // most likely to be looking at it. Persisted right after Compacted.
        recordTokenCount(
          totalOnlyTokenCountEvent(estimated, config.contextWindow),
          emitEvent,
          liveThread,
        );
      },
    });
  } catch (err) {
    // The summarizer itself could not fit even after trimming: compact.ts has
    // marked the tokens full — persist that mark too (see markContextFull).
    if (isContextWindowExceededError(err)) {
      markContextFull(config, tokenState, emitEvent, liveThread);
    }
    throw err;
  }
  if (applied === null) {
    // Logged because a silently dropped compaction is indistinguishable from
    // one that never triggered.
    console.warn(
      `[codex] dropped a stale compaction: the context window already rolled ` +
        `over (epoch ${epochAtEntry} → ${compactWindow.currentCompactionEpoch()}) ` +
        `while this summarizer was in flight; history was NOT rewritten`,
    );
    return;
  }
  emitEvent({
    type: "Warning",
    event: {
      message:
        "Heads up: Long threads and multiple compactions can cause the model to be less accurate. " +
        "Start a new thread when possible to keep threads small and targeted.",
    },
  });
}

// ─── runTurn ─────────────────────────────────────────────────────────────────

export interface TurnResult {
  lastAgentMessage: string;
  /** All events emitted during the turn (including streaming deltas) */
  events: EventMsg[];
}

export async function runTurn(
  turnId: string,
  userItems: UserInput[],
  history: HistoryItem[],
  config: TurnConfig,
  router: ToolRouter,
  pendingInputs: PendingInputs,
  emitEvent: (msg: EventMsg) => void,
  liveThread?: LiveThread | undefined,
  /** mirrors: codex-rs propagates a tokio CancellationToken into the turn loop;
   *  AbortSignal is the browser-native equivalent. */
  abortSignal?: AbortSignal | undefined,
  /** Additional SEPARATE user messages to record alongside `userItems` in this
   *  one turn — each entry is one message's content items and becomes its own
   *  role:user history item, never merged. Mirrors codex-rs draining several
   *  queued UserInput submissions into the turn as distinct messages (so the
   *  app can flush a whole send-queue as one turn: "send all queued, not
   *  merged into one message"). */
  extraUserMessages?: UserInput[][] | undefined,
): Promise<{ lastAgentMessage: string }> {
  // Record one user message per submitted group, in order (mutates the shared
  // history array). codex-rs records each queued UserInput submission as its OWN
  // role:user item and never merges them; we mirror that by recording `userItems`
  // (the primary message) followed by each `extraUserMessages` entry as a separate
  // message. Content mapping mirrors ResponseInputItem::from(Vec<UserInput>)
  // (protocol/src/models.rs:1735): text → input_text, image → input_image
  // { image_url } (detail default), audio → input_audio { audio_url }.
  const toUserContent = (items: UserInput[]): UserContentPart[] =>
    items
      .map((item): UserContentPart | null => {
        switch (item.type) {
          case "text":
            return { type: "input_text", text: item.text };
          case "image":
            return { type: "input_image", image_url: item.image_url };
          case "audio":
            return { type: "input_audio", audio_url: item.audio_url };
          default:
            return null;
        }
      })
      .filter((part): part is UserContentPart => part !== null);
  const userMsgs: HistoryItem[] = [userItems, ...(extraUserMessages ?? [])]
    .map(toUserContent)
    .filter((content) => content.length > 0)
    .map((content) => ({ type: "message", role: "user", content }));
  history.push(...userMsgs);
  if (userMsgs.length > 0) {
    await liveThread?.appendConversationItems(userMsgs);
  }

  // ── Layer 2: turn-scoped skill full-body injection (mirrors codex-rs) ──────
  // A `$skill-name` mention in user input pulls that skill's full SKILL.md into
  // THIS turn's request input only. Like codex-rs contextual fragments, these
  // items are NOT pushed to history / persisted — resume stays clean and the
  // body isn't duplicated every turn.
  const skillInjectionItems: HistoryItem[] = [];
  const loadSkillContent = config.loadSkillContent;
  if (config.skills && config.skills.length > 0 && loadSkillContent) {
    const userText = [userItems, ...(extraUserMessages ?? [])]
      .flat()
      .map((item) => (item.type === "text" ? item.text : ""))
      .join("\n");
    const mentioned = extractSkillMentions(userText, config.skills);
    for (const skill of mentioned) {
      const contents = await loadSkillContent(skill).catch(() => null);
      if (contents !== null) {
        skillInjectionItems.push({
          type: "message",
          role: "user",
          content: [
            { type: "input_text", text: renderSkillInjection(skill, contents) },
          ],
        });
      }
    }
  }

  const tools = router.toolSpecs().map(toolSpecToRequestJson);
  const maxRetries = config.maxRetries ?? DEFAULT_MAX_RETRIES;
  let lastAgentMessage = "";
  let itemIdCounter = 0;

  // ── Auto-compaction window tracker (mirrors AutoCompactWindow in codex-rs) ──
  // Uses BodyAfterPrefix mode: only tokens added after the last compaction count
  // toward the limit. mirrors AutoCompactTokenLimitScope::BodyAfterPrefix.
  //
  // Prefer the THREAD-OWNED window (config.compactWindow) so the baseline persists
  // ACROSS turns — codex-rs keeps this in session state. A fresh window per turn
  // would reset the baseline to the (already large) current size every turn, so
  // cross-turn growth never reaches the limit and a long interactive thread
  // silently grows past the model's context window → "prompt is too long". Fall
  // back to a private window only when no shared one is supplied (one-shot/tests).
  const compactWindow = config.compactWindow ?? new AutoCompactWindow();
  // Session token accounting — same thread-owned pattern (mirrors session state).
  const tokenState = config.tokenState ?? new SessionTokenState();

  // ── Estimated baseline seed (mirrors set_estimated_prefill, mod.rs:1290) ────
  // Until the first server usage arrives (fresh thread, resumed thread, or a
  // freshly compacted window) the window baseline and session total start from
  // a byte-based estimate of what this turn is about to send. Server usage
  // overrides both on the next sampled response.
  if (compactWindow.snapshot().prefillInputTokens === null) {
    recomputeTokenUsage({
      items: [...(config.contextItems ?? []), ...history],
      instructions: config.instructions,
      tokenState,
      compactWindow,
      // Seed only: a session total that already exists (server usage carried
      // over from an earlier window) must not be overwritten by an estimate.
      // What is missing here is the window baseline, not the total.
      overwriteTotal: tokenState.totalTokens === null,
    });
  }

  // ── Pre-sampling compaction (mirrors run_pre_sampling_compact, turn.rs:149) ─
  // Compact BEFORE the first request of the turn when the context is already
  // at/over budget — e.g. the previous turn ended with a context-window-exceeded
  // rejection (which force-filled the session total), or a resumed thread's
  // history estimate already exceeds the window. This is the self-heal path:
  // without it, every subsequent request would keep failing with 400.
  {
    const preStatus = autoCompactTokenStatus(
      config,
      compactWindow,
      tokenState,
      history,
    );
    if (preStatus.tokenLimitReached) {
      await runCompactAndRecompute(
        history,
        config,
        compactWindow,
        tokenState,
        emitEvent,
        liveThread,
        abortSignal,
      );
    }
  }

  for (;;) {
    // Bail out before sampling again if the turn was interrupted.
    if (abortSignal?.aborted) {
      throw new DOMException("Turn interrupted", "AbortError");
    }

    // ── Pairing invariants before every request (mirrors for_prompt →
    // normalize_history): synthesize "aborted" outputs for calls that never
    // got one (interrupted turn, resumed corrupted thread) and drop orphaned
    // outputs. Anthropic 400s on unpaired tool_use/tool_result otherwise.
    normalizeHistory(history);

    // ── Replay hygiene ──────────────────────────────────────────────────────
    // Execution repairs malformed tool arguments before dispatch, but history
    // keeps the model's ORIGINAL text; a downstream replay converter that
    // plain-JSON.parses it zeroes the call to `input: {}` and the model then
    // "redoes" a call that already succeeded. Canonicalize in place BEFORE the
    // item is ever sent back. See tools/argument_repair.ts.
    for (const item of history) {
      if (
        item.type === "function_call" &&
        typeof item.arguments === "string"
      ) {
        item.arguments = canonicalizeStoredArguments(
          item.arguments,
          item.name ?? "",
        );
      }
    }

    // ── Sample from the model ───────────────────────────────────────────────
    const body: Record<string, unknown> = {
      model: config.model,
      // Reasoning: codeproxy maps `reasoning.effort` → upstream thinking budget so
      // Gemini streams reasoning (thought) deltas → ReasoningContentDelta. Without
      // it the model never emits thinking (only output text). Defaults to "medium".
      reasoning: { effort: config.reasoningEffort ?? "medium" },
      // codex-rs structure: turn-scoped context (user_instructions + skills
      // catalog) and any $mention skill bodies ride at the front, ahead of the
      // persisted conversation history. None of these are persisted.
      input: [
        ...(config.contextItems ?? []),
        ...skillInjectionItems,
        ...history,
      ],
      tools,
      stream: true,
    };
    if (config.instructions) body["instructions"] = config.instructions;

    // ── Sample from the model (retry transient failures) ────────────────────
    // mirrors codex-rs's stream retry loop (responses_retry.rs): a network
    // error / 5xx / 429 / dropped stream is retried with exponential backoff.
    // Unlike codex-rs (which re-streams in place into a TUI) codex-ts emits
    // deltas live, so we only retry while NO visible output has streamed yet —
    // re-streaming after that would duplicate text downstream.
    const functionCalls: { call_id: string; name: string; arguments: string }[] =
      [];
    let assistantText = "";
    let currentItemId = `item-${++itemIdCounter}`;
    let inputTokensThisRound = 0;
    let retryAttempt = 0;

    for (;;) {
      // A retry re-streams from scratch → reset per-attempt accumulators.
      functionCalls.length = 0;
      assistantText = "";
      currentItemId = `item-${++itemIdCounter}`;
      inputTokensThisRound = 0;
      let emittedOutput = false;
      // Whether THIS attempt's stream has already delivered response.completed.
      // A terminal event is terminal — anything arriving after it is transport
      // noise, not turn state. See the response.failed arm below.
      let sawCompleted = false;
      // The compaction generation this request's history was built under.
      // Concurrent turns share the session: if another turn compacted while
      // this request was in flight, its usage measures a history that no
      // longer exists. See the usage guard in the response.completed arm.
      const epochAtBuild = compactWindow.currentCompactionEpoch();

      try {
        const res = await (config.fetch ?? fetch)(
          `${config.baseUrl}/responses`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${config.apiKey}`,
            },
            body: JSON.stringify(body),
            signal: abortSignal ?? null,
          },
        );

        if (!res.ok || !res.body) {
          const text = res.body ? await res.text() : "(no body)";
          throw new ResponsesApiError(
            res.status,
            text,
            parseRetryAfter(res.headers),
          );
        }

        // ── Parse SSE stream ──────────────────────────────────────────────
        for await (const raw of parseSseStream(res.body)) {
          switch (raw["type"]) {
            case "response.output_item.added": {
              const item = raw["item"] as Record<string, unknown> | undefined;
              // function_call 的 arguments 整体来自下面的 output_item.done；
              // codex-rs 不累积 arg-delta（codex-api/src/sse/responses.rs 丢弃
              // response.function_call_arguments.delta），故此处只跟踪 message item id。
              if (item?.["type"] === "message") {
                currentItemId = String(item["id"] ?? currentItemId);
              } else if (item?.["type"] === "function_call") {
                // Announce the call the moment the model starts emitting it —
                // name known, arguments not yet streamed — so a host can show
                // the in-progress tool during composition instead of a blank
                // spinner. DISPLAY ONLY: the dispatched arguments still come
                // solely from output_item.done (see the rule at that arm).
                emitEvent({
                  type: "FunctionCallStreamStart",
                  event: {
                    id: String(item["id"] ?? item["call_id"] ?? ""),
                    name: String(item["name"] ?? ""),
                    turn_id: turnId,
                  },
                });
              }
              break;
            }
            case "response.output_text.delta": {
              const delta = String(raw["delta"] ?? "");
              assistantText += delta;
              // Visible output has streamed — past this point a retry would
              // duplicate text, so failures below are no longer retryable.
              emittedOutput = true;
              emitEvent({
                type: "AgentMessageContentDelta",
                event: { turn_id: turnId, item_id: currentItemId, delta },
              });
              break;
            }
            case "response.reasoning_text.delta":
            case "response.reasoning_summary_text.delta": {
              // 模型正在产出 reasoning（thinking）——照搬 codex-rs
              // codex-api/src/sse/responses.rs 的 reasoning delta 分支。供 host 在模型
              // "真正 thinking"（而非仅已发请求未回）时显示 Thinking。Gemini 的 thinking
              // 经 codeproxy 翻译为 responses reasoning，正是从这两个 SSE 事件流出。
              //
              // summary_index 照搬 rs 的 ReasoningContentDelta{,Summary} 分支
              // （responses.rs:355-379）：summary 用 `summary_index`，raw
              // reasoning 用 `content_index`，供 host 区分多段 reasoning。
              // rs 的 SSE 层不带 item_id（由 core 的
              // assign_missing_streamed_response_item_id 补），故这里在事件缺失
              // item_id 时同样回落到当前 item。
              const delta = String(raw["delta"] ?? "");
              const blockIndex =
                raw["type"] === "response.reasoning_summary_text.delta"
                  ? raw["summary_index"]
                  : raw["content_index"];
              emitEvent({
                type: "ReasoningContentDelta",
                event: {
                  turn_id: turnId,
                  item_id: String(raw["item_id"] ?? currentItemId),
                  delta,
                  summary_index:
                    typeof blockIndex === "number" ? blockIndex : 0,
                },
              });
              break;
            }
            // response.function_call_arguments.delta 不累积：function_call 的
            // arguments 整体来自 output_item.done；codex-rs 亦丢弃该 delta 事件
            // （codex-api/src/sse/responses.rs 落 `_ =>`），不做累积。
            case "response.function_call_arguments.delta": {
              // Forwarded raw (keyed by item_id) so a host can accumulate and
              // show the in-progress tool during composition. DISPLAY ONLY by
              // construction: nothing here feeds `functionCalls`, so the rule
              // above — arguments come only from output_item.done — still holds.
              emitEvent({
                type: "FunctionCallStreamArgs",
                event: {
                  delta: String(raw["delta"] ?? ""),
                  id: String(raw["item_id"] ?? raw["call_id"] ?? ""),
                  turn_id: turnId,
                },
              });
              break;
            }
            case "response.output_item.done": {
              const item = raw["item"] as Record<string, unknown> | undefined;
              if (item?.["type"] === "function_call") {
                functionCalls.push({
                  call_id: String(item["call_id"]),
                  name: String(item["name"]),
                  arguments: String(item["arguments"] ?? "{}"),
                });
              }
              break;
            }
            case "response.completed": {
              // Terminal event of the OpenAI Responses stream — codex-rs dispatches
              // the same "response.completed" (codex-api/src/sse/responses.rs) to read
              // usage. (An earlier port used "response.done", which no Responses
              // backend emits, so usage never arrived and auto-compaction never armed.)
              // Mirrors record_token_usage_info (turn.rs:2072): session total ←
              // usage, window prefill ← first server-observed input_tokens.
              // Recorded BEFORE the usage guard: a completed frame without
              // parsable usage still completes the turn.
              sawCompleted = true;
              const resp = raw["response"] as
                | Record<string, unknown>
                | undefined;
              const usage = resp?.["usage"] as
                | Record<string, unknown>
                | undefined;
              if (typeof usage?.["input_tokens"] === "number") {
                inputTokensThisRound = usage["input_tokens"];
                // Usage from a request built BEFORE a compaction that landed
                // meanwhile is display-only: writing it back would overwrite
                // the post-compaction estimate with the old, larger history's
                // size (re-arming compaction at once) and seed the new
                // window's baseline from it. mirrors TokenUsageContext::
                // SpendingOnly (#53670): recorded for spend, not for the
                // context baseline.
                const staleUsage =
                  compactWindow.currentCompactionEpoch() !== epochAtBuild;
                if (!staleUsage) {
                  tokenState.updateFromUsage({
                    inputTokens: inputTokensThisRound,
                    outputTokens:
                      typeof usage["output_tokens"] === "number"
                        ? usage["output_tokens"]
                        : undefined,
                    totalTokens:
                      typeof usage["total_tokens"] === "number"
                        ? usage["total_tokens"]
                        : undefined,
                  });
                  compactWindow.ensureServerObservedPrefill(
                    inputTokensThisRound,
                  );
                }
                // mirrors send_token_count_event (session/mod.rs:3131): emit the
                // recorded usage so hosts can display context consumption. In the
                // ts port the session total is REPLACED by each response's total
                // (updateFromUsage), so total_token_usage mirrors last_token_usage
                // with the session total swapped in. rate_limits: not tracked → null.
                const numberOr = (value: unknown, fallback: number): number =>
                  typeof value === "number" ? value : fallback;
                const inputDetails = usage["input_tokens_details"] as
                  | Record<string, unknown>
                  | undefined;
                const outputDetails = usage["output_tokens_details"] as
                  | Record<string, unknown>
                  | undefined;
                const lastUsage = {
                  input_tokens: inputTokensThisRound,
                  cached_input_tokens: numberOr(
                    inputDetails?.["cached_tokens"],
                    0,
                  ),
                  // mirrors TokenUsage::cache_write_input_tokens ←
                  // usage.input_tokens_details.cache_write_tokens
                  // (codex-api/src/sse/responses.rs:137).
                  cache_write_input_tokens: numberOr(
                    inputDetails?.["cache_write_tokens"],
                    0,
                  ),
                  output_tokens: numberOr(usage["output_tokens"], 0),
                  reasoning_output_tokens: numberOr(
                    outputDetails?.["reasoning_tokens"],
                    0,
                  ),
                  total_tokens: numberOr(
                    usage["total_tokens"],
                    inputTokensThisRound + numberOr(usage["output_tokens"], 0),
                  ),
                };
                const tokenCount: EventMsg = {
                  type: "TokenCount",
                  event: {
                    info: {
                      total_token_usage: {
                        ...lastUsage,
                        total_tokens:
                          tokenState.totalTokens ?? lastUsage.total_tokens,
                      },
                      last_token_usage: lastUsage,
                      model_context_window: config.contextWindow ?? null,
                    },
                    rate_limits: null,
                  },
                };
                // A stale measurement is shown but never persisted: a resume
                // must seed from the current history's size.
                if (staleUsage) {
                  emitEvent(tokenCount);
                } else {
                  recordTokenCount(tokenCount, emitEvent, liveThread);
                }
              }
              break;
            }
            case "response.failed":
            case "response.incomplete": {
              // A failed/incomplete arriving AFTER response.completed must not
              // un-complete the turn. Observed 2026-08-04 in production: the
              // upstream delivered the full stream then held the connection
              // open, and a gateway watchdog sealed the hold with a synthetic
              // response.incomplete AFTER the completed frame. Throwing here
              // classified a COMPLETE turn as a retryable 503, so the client
              // re-sent the byte-identical request in an endless token-burning
              // loop. First terminal event wins; later ones are transport noise.
              if (sawCompleted) {
                break;
              }
              // mirrors codex-api sse/responses.rs process_responses_event's
              // terminal arms (responses.rs:387-431): `response.failed` is
              // classified from `response.error.code` — quota / cyber-policy /
              // invalid-prompt / server-overloaded are TERMINAL (rs maps them to
              // non-retryable CodexErr variants), a rate limit carries its
              // "try again in Xs" delay, anything unrecognised stays retryable —
              // and `response.incomplete` becomes a retryable stream error.
              // context-window-exceeded is detected by the catch below, which
              // also marks the session tokens full.
              throw classifyStreamFailure(raw);
            }
          }
        }
        break; // stream consumed successfully → leave the retry loop
      } catch (err) {
        // An interrupt is terminal — never retry past a cancellation.
        if (abortSignal?.aborted) throw err;
        // Context-window-exceeded is terminal AND self-healing: mark the
        // session tokens FULL so the NEXT turn's pre-sampling check compacts
        // before sampling (mirrors turn.rs:1045-1047 set_total_tokens_full).
        // Retrying the identical request would only fail again.
        if (isContextWindowExceededError(err)) {
          markContextFull(config, tokenState, emitEvent, liveThread);
          throw err;
        }
        // Stop retrying once visible output streamed (would duplicate), the
        // budget is spent, or the error is not transient.
        if (
          emittedOutput ||
          retryAttempt >= maxRetries ||
          !isRetryableError(err)
        ) {
          throw err;
        }
        retryAttempt += 1;
        emitEvent({
          type: "Warning",
          event: { message: `Reconnecting... ${retryAttempt}/${maxRetries}` },
        });
        await sleep(computeRetryDelay(err, retryAttempt), abortSignal);
      }
    }

    // ── Persist assistant message ────────────────────────────────────────────
    if (assistantText) {
      lastAgentMessage = assistantText;
      const assistantItem: HistoryItem = {
        type: "message",
        role: "assistant",
        content: assistantText,
      };
      history.push(assistantItem);
      await liveThread?.appendConversationItems([assistantItem]);
      emitEvent({ type: "AgentMessage", event: { message: assistantText } });
    }

    // ── Persist function calls ───────────────────────────────────────────────
    const callItems: HistoryItem[] = functionCalls.map((call) => ({
      type: "function_call" as const,
      call_id: call.call_id,
      name: call.name,
      arguments: call.arguments,
    }));
    for (const item of callItems) {
      history.push(item);
    }
    if (callItems.length > 0) await liveThread?.appendConversationItems(callItems);

    // ── Done when no tool calls ─────────────────────────────────────────────
    if (functionCalls.length === 0) break;

    // ── Dispatch tool calls ─────────────────────────────────────────────────
    // Abort semantics (mirrors codex-rs, where the turn task is killed
    // outright): once interrupted, never start another tool and DISCARD the
    // result of the one that was mid-flight — do NOT record it. The next
    // turn's normalizeHistory synthesizes an "aborted" output for the calls
    // left unpaired. Recording here instead would race that synthesis: the
    // new turn inserts "aborted", then this zombie loop lands the real output
    // too → duplicate outputs for one call_id → Anthropic 400 ("each tool_use
    // must have a single result") persisted into the thread
    // (openrouter-responses-errors/2026-07-18/a1cf9c62).
    for (const call of functionCalls) {
      if (abortSignal?.aborted) {
        throw new DOMException("Turn interrupted", "AbortError");
      }
      let args: unknown;
      try {
        // Strict parse runs FIRST inside parseToolArguments, so a well-formed
        // call is untouched; only already-invalid arguments reach the repair,
        // and an unrepairable one throws a message the model can act on.
        args = parseToolArguments(call.arguments, call.name);
        if (typeof args !== "object" || args === null || Array.isArray(args)) {
          // rs 的 from_str::<Struct> 对非对象（如 "5"、"[...]"）同样失败。
          throw new Error("expected a JSON object");
        }
      } catch (err) {
        // mirror codex-rs：参数无法解析 → 把错误当 function_call_output 回给模型、
        // 跳过 dispatch，绝不拿空参执行工具（tools/handlers/mod.rs:77-84 parse_arguments
        // → tools/parallel.rs:186-209 failure_response；MCP 同义 mcp_tool_call.rs:117-132）。
        // 该 call 已作为 function_call 入 history 且仍在 functionCalls，本轮不 break，
        // 外层 for(;;) 会再采样让模型看到错误 = rs 的 needs_follow_up。
        const message = err instanceof Error ? err.message : String(err);
        const errorItem: HistoryItem = {
          type: "function_call_output",
          call_id: call.call_id,
          output: `failed to parse function arguments: ${message}`,
        };
        history.push(errorItem);
        await liveThread?.appendConversationItems([errorItem]);
        continue;
      }

      const output = await router.dispatch(call.name, call.call_id, args, {
        turnId,
        pendingInputs,
        emitEvent,
      });

      // Interrupted while this tool ran → drop its output (see block comment).
      if (abortSignal?.aborted) {
        throw new DOMException("Turn interrupted", "AbortError");
      }

      const outputItem: HistoryItem = {
        type: "function_call_output",
        call_id: call.call_id,
        output,
      };
      history.push(outputItem);
      await liveThread?.appendConversationItems([outputItem]);
    }

    // ── Auto-compaction check (mirrors post-sampling compact, turn.rs:417-455) ─
    // Only triggered mid-turn (more sampling rounds follow), matching
    // `token_limit_reached && needs_follow_up` in codex-rs. Runs AFTER the tool
    // outputs are recorded — codex-rs compacts only between COMPLETE rounds; an
    // earlier port compacted between call-recording and dispatch, so the
    // rewrite dropped the pending function_calls and the outputs pushed right
    // after became orphans → Anthropic 400 "tool_use ids … without tool_result".
    // Fires on EITHER growth-since-baseline ≥ autoCompactTokenLimit OR total
    // active context ≥ contextWindow (see autoCompactTokenStatus).
    {
      const status = autoCompactTokenStatus(
        config,
        compactWindow,
        tokenState,
        history,
      );
      // mirrors should_roll_over (turn.rs:418-420): mid-turn compaction fires
      // when the token limit is reached OR a new context window was explicitly
      // requested (rs's `new_context` tool; in codex-ts a host can request one
      // via compactWindow.requestNewContextWindow()). take… clears the flag, so
      // it must be consulted every round even when the limit already tripped.
      const rollOverRequested = compactWindow.takeNewContextWindowRequest();
      if (rollOverRequested || status.tokenLimitReached) {
        await runCompactAndRecompute(
          history,
          config,
          compactWindow,
          tokenState,
          emitEvent,
          liveThread,
          abortSignal,
        );
      }
    }
  }

  return { lastAgentMessage };
}
