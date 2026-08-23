/**
 * mirrors codex-rs/core/src/codex_thread.rs
 *
 * CodexThread — manages the conversation session:
 *   - submit(op)    → enqueue an Op, return submission_id  (mirrors submit())
 *   - nextEvent()   → pull the next Event                  (mirrors next_event())
 *
 * Accepts optional ThreadStore and GoalStore for durable persistence.
 * Falls back to in-memory implementations when neither is supplied.
 */

import type {
  Event,
  EventMsg,
  Op,
  TurnAbortedEvent,
  TurnCompleteEvent,
  TurnStartedEvent,
} from "../../protocol/src/protocol.js";
import type { RequestUserInputResponse } from "../../protocol/src/request_user_input.js";
import { GoalToolExecutor } from "../../ext/goal/src/tool.js";
import { GoalStore, InMemoryGoalBackend } from "../../state/src/runtime/goals.js";
import {
  InMemoryThreadStore,
  LocalThreadStore,
  LiveThread,
} from "../../thread-store/src/index.js";
import type { ThreadStore, IoBackend } from "../../thread-store/src/index.js";
import type { ConversationItem } from "../../thread-store/src/types.js";
import { ToolRouter } from "./tools/router.js";
import type { CustomTool } from "./tools/router.js";
import { runTurn } from "./session/turn.js";
import { AutoCompactWindow } from "./state/auto_compact_window.js";
import { SessionTokenState } from "./state/token_state.js";
import { DEFAULT_BASE_INSTRUCTIONS } from "./base_instructions.js";
import { renderAvailableSkills } from "./skills.js";
import type { SkillMetadata } from "./skills.js";
import { codexErrorInfoFor, isAbortError } from "./session/retry.js";
import type { TurnConfig } from "./session/turn.js";
import type { PendingInputs } from "./tools/handlers/request_user_input.js";

// ─── Async event queue (mirrors tokio mpsc) ───────────────────────────────────

type QueueEntry<T> = { done: false; value: T } | { done: true };

class AsyncQueue<T> {
  private buf: T[] = [];
  private waiters: ((entry: QueueEntry<T>) => void)[] = [];
  private closed = false;

  push(value: T): void {
    const waiter = this.waiters.shift();
    if (waiter) waiter({ done: false, value });
    else this.buf.push(value);
  }

  close(): void {
    this.closed = true;
    for (const w of this.waiters) w({ done: true });
    this.waiters = [];
  }

  async next(): Promise<T> {
    if (this.buf.length > 0) return this.buf.shift()!;
    if (this.closed) throw new Error("event stream closed");
    return new Promise<T>((resolve, reject) => {
      this.waiters.push((entry) => {
        if (entry.done) reject(new Error("event stream closed"));
        else resolve(entry.value);
      });
    });
  }
}

// ─── ID generation ────────────────────────────────────────────────────────────

let _seq = 0;
function nextId(): string {
  return `sub-${++_seq}-${Math.random().toString(36).slice(2, 8)}`;
}

// ─── CodexThreadConfig ────────────────────────────────────────────────────────

export interface CodexThreadConfig {
  apiKey: string;
  baseUrl?: string | undefined;
  /**
   * Custom `fetch` for the Responses API calls (model sampling + inline
   * auto-compaction). Defaults to the global `fetch`. Use it to inject auth
   * headers, refresh a token on 401, route through a proxy, or mock in tests —
   * without monkey-patching the global `fetch`.
   */
  fetch?: typeof fetch | undefined;
  model: string;
  instructions?: string | undefined;
  /**
   * Thread ID to use. If omitted a new ID is generated.
   * Supply a previously used ID + a matching threadStore to resume a session.
   */
  threadId?: string | undefined;
  /**
   * Durable conversation-history store.
   * Pass an InMemoryThreadStore (default), a LocalThreadStore with your IoBackend,
   * or any custom ThreadStore implementation.
   */
  threadStore?: ThreadStore | undefined;
  /**
   * IoBackend shorthand: if you only want to inject I/O primitives without
   * wiring up a full ThreadStore, pass an IoBackend here and a LocalThreadStore
   * will be created automatically.
   */
  ioBackend?: IoBackend | undefined;
  /**
   * Durable goal store.
   * Defaults to an in-memory GoalStore (lost on reload).
   * Pass a GoalStore with a custom GoalBackend for persistence.
   */
  goalStore?: GoalStore | undefined;
  /**
   * Host-supplied custom tools. Their specs are advertised to the model
   * alongside the built-ins, and calls route to each tool's execute().
   */
  customTools?: CustomTool[] | undefined;
  /**
   * Base agent instructions prepended ahead of `instructions`. Mirrors
   * codex-rs's base_instructions layer (the agent harness that keeps the model
   * acting like a tool-calling agent). Defaults to DEFAULT_BASE_INSTRUCTIONS;
   * pass "" to disable.
   */
  baseInstructions?: string | undefined;
  /**
   * Discovered skills (name + description + path). Rendered into an always-on
   * "## Skills" catalog (Layer 1) and used to resolve `$skill-name` mentions for
   * full-body injection (Layer 2). Mirrors codex-rs's core-skills crate, except
   * discovery (scanning .agents/skills + parsing frontmatter) is the host's job
   * since a browser has no filesystem.
   */
  skills?: SkillMetadata[] | undefined;
  /** Reads a skill's full SKILL.md on demand (host-provided; browser has no fs). */
  loadSkillContent?: ((skill: SkillMetadata) => Promise<string>) | undefined;
  /**
   * Project documentation (AGENTS.md content). Mirrors codex-rs's
   * AgentsMdManager / UserInstructions fragment: injected as a discrete
   * `input` message (format: "# AGENTS.md instructions\n\n<INSTRUCTIONS>…")
   * ahead of the conversation history so the model sees it as a separate
   * context boundary — NOT merged into the `instructions` field.
   * Discovery (finding AGENTS.override.md / AGENTS.md) is the host's job
   * since a browser has no filesystem.
   */
  agentsMd?: string | undefined;
  /**
   * Input-token threshold for inline auto-compaction (BodyAfterPrefix mode).
   * mirrors model_auto_compact_token_limit in codex-rs TurnContext config.
   *
   * Recommended: context_window × 0.9  (e.g. 115 000 for gpt-4o's 128 k window).
   * Omit to disable auto-compaction entirely.
   */
  autoCompactTokenLimit?: number | undefined;
  /**
   * The model's full context window in tokens (mirrors model_context_window in
   * codex-rs). Arms the absolute compaction trigger (compact when TOTAL active
   * context reaches the window, independent of the growth budget above) and
   * the context-window-exceeded self-heal (a request rejected with "prompt is
   * too long"/"context_length_exceeded" marks the session full, so the next
   * turn compacts before sampling instead of failing forever). Strongly
   * recommended for long-lived interactive threads.
   */
  contextWindow?: number | undefined;
  /**
   * Max retries for transient Responses request/stream failures — network
   * errors, 5xx / 408 / 409 / 429, or a stream dropped before any visible
   * output. Each retry waits an exponential backoff (200ms × 2^(n-1) ± 10%
   * jitter), honoring a server Retry-After when present. mirrors codex-rs
   * stream/request_max_retries. Defaults to 5; pass 0 to disable retries.
   */
  maxRetries?: number | undefined;
  /**
   * Overrides the inline-compaction summarization prompt (mirrors codex-rs
   * `config.compact_prompt`, compact.rs:118). Omit for the bundled
   * SUMMARIZATION_PROMPT ported from prompts/templates/compact/prompt.md.
   */
  compactPrompt?: string | undefined;
}

interface ResolvedConfig {
  apiKey: string;
  baseUrl: string;
  fetch: typeof fetch;
  model: string;
  instructions?: string | undefined;
  baseInstructions: string;
  skills: SkillMetadata[];
  loadSkillContent?: ((skill: SkillMetadata) => Promise<string>) | undefined;
  agentsMd?: string | undefined;
  autoCompactTokenLimit?: number | undefined;
  contextWindow?: number | undefined;
  maxRetries?: number | undefined;
  compactPrompt?: string | undefined;
}

// ─── CodexThread ─────────────────────────────────────────────────────────────

export class CodexThread {
  private readonly config: ResolvedConfig;
  private readonly threadId: string;
  private readonly eventQueue = new AsyncQueue<Event>();

  /** In-flight conversation history (sent to the Responses API each turn) */
  private readonly history: ConversationItem[] = [];

  /** Pending request_user_input resolvers keyed by turn_id */
  private readonly pendingInputs: PendingInputs = new Map<
    string,
    (response: RequestUserInputResponse) => void
  >();

  private readonly goalExecutor: GoalToolExecutor;
  private readonly router: ToolRouter;
  private readonly liveThread: LiveThread;

  /**
   * AbortController for the in-flight turn, if any.
   * mirrors: codex-rs uses a tokio CancellationToken propagated from
   * Op::Interrupt; AbortController is the browser-native equivalent.
   */
  // Session-scoped compaction window — ONE per thread, threaded into every
  // runTurn so auto-compaction measures context growth across turns, not just
  // within a turn (mirrors codex-rs session state). See TurnConfig.compactWindow.
  private readonly compactWindow = new AutoCompactWindow();
  // Session-scoped token accounting — same pattern (mirrors sess token_info):
  // usage recorded in one turn drives the NEXT turn's pre-sampling compaction
  // check, including the context-window-exceeded self-heal. See TurnConfig.
  private readonly tokenState = new SessionTokenState();
  private currentTurnAbort: AbortController | null = null;

  constructor(config: CodexThreadConfig) {
    this.config = {
      apiKey: config.apiKey,
      baseUrl: config.baseUrl ?? "https://api.openai.com/v1",
      fetch: config.fetch ?? fetch,
      model: config.model,
      instructions: config.instructions,
      baseInstructions: config.baseInstructions ?? DEFAULT_BASE_INSTRUCTIONS,
      skills: config.skills ?? [],
      loadSkillContent: config.loadSkillContent,
      agentsMd: config.agentsMd,
      autoCompactTokenLimit: config.autoCompactTokenLimit,
      contextWindow: config.contextWindow,
      maxRetries: config.maxRetries,
      compactPrompt: config.compactPrompt,
    };

    this.threadId = config.threadId ?? nextId();

    // Resolve ThreadStore: explicit > ioBackend shorthand > in-memory
    const store: ThreadStore =
      config.threadStore ??
      (config.ioBackend
        ? new LocalThreadStore(config.ioBackend)
        : new InMemoryThreadStore());

    this.liveThread = new LiveThread(this.threadId, store);

    // Resolve GoalStore
    const goalStore = config.goalStore ?? new GoalStore(new InMemoryGoalBackend());
    this.goalExecutor = new GoalToolExecutor(this.threadId, goalStore);

    this.router = new ToolRouter(this.goalExecutor, config.customTools);
  }

  /** The thread's stable identifier (use for resume) */
  get id(): string { return this.threadId; }

  /**
   * Is a turn still parked inside `request_user_input` waiting for THIS answer?
   *
   * `submit({ type: "UserInputAnswer" })` is a silent no-op when the resolver is
   * gone (see the UserInputAnswer arm below) — and the resolver is gone after
   * every page reload, because `pendingInputs` is process memory while the
   * `function_call` it belongs to is already on disk. Without this predicate a
   * host cannot tell "answer delivered" from "answer dropped on the floor": it
   * reports success, flips to a spinner and hangs forever (2026-08-07 field
   * report). Exposing it lets the host refuse honestly and rebuild the card
   * instead.
   */
  hasPendingInput(turnId: string): boolean {
    return this.pendingInputs.has(turnId);
  }

  /**
   * Async factory — use this instead of `new` when resuming an existing thread.
   *
   * If `config.threadId` is supplied and the backing store contains history,
   * that history is loaded into the in-flight conversation before the first
   * turn runs.  Mirrors the session-initialisation path in codex-rs where
   * `load_history()` is called during `CodexThread` construction.
   *
   * For brand-new threads `new CodexThread(config)` is equivalent.
   */
  static async create(config: CodexThreadConfig): Promise<CodexThread> {
    const thread = new CodexThread(config);
    if (config.threadId) {
      const items = await thread.liveThread.loadConversationHistory();
      thread.history.push(...items);
    }
    return thread;
  }

  /**
   * Submit an Op. Returns the submission_id.
   * Mirrors: pub async fn submit(&self, op: Op) -> CodexResult<String>
   */
  async submit(op: Op): Promise<string> {
    const submissionId = nextId();

    switch (op.type) {
      case "UserInput": {
        const turnId = submissionId;
        // Per-turn overrides (op.*) take precedence over thread-level config.
        // Developer instructions are layered on top of the base agent harness.
        const devInstructions = op.instructions ?? this.config.instructions;
        // instructions field = base agent harness + developer (app-level)
        // instructions, mirroring codex-rs where base/developer go in
        // `instructions`. Project docs + catalog ride in `input` instead.
        const instructions = [this.config.baseInstructions, devInstructions]
          .filter((part): part is string => Boolean(part))
          .join("\n\n");
        // codex-rs puts the project doc + skills catalog in `input` as discrete
        // contextual messages so each boundary is explicit to the model (see
        // TurnConfig.contextItems) rather than baked into the instructions blob.
        const contextItems: ConversationItem[] = [];
        // AGENTS.md → user_instructions message (codex-rs UserInstructions
        // fragment: # AGENTS.md instructions / <INSTRUCTIONS> markers).
        if (this.config.agentsMd) {
          contextItems.push({
            type: "message",
            role: "user",
            content: [
              {
                type: "input_text",
                text: `# AGENTS.md instructions\n\n<INSTRUCTIONS>\n${this.config.agentsMd}\n</INSTRUCTIONS>`,
              },
            ],
          });
        }
        // skills catalog message (codex-rs available-skills fragment). The
        // budget warning is NOT part of the model-visible body — codex-rs sends
        // it as EventMsg::Warning (session/mod.rs:3393-3402), so we do too.
        const availableSkills = renderAvailableSkills(this.config.skills);
        if (availableSkills.body) {
          contextItems.push({
            type: "message",
            role: "user",
            content: [{ type: "input_text", text: availableSkills.body }],
          });
        }
        if (availableSkills.warningMessage) {
          this.pushEvent(submissionId, {
            type: "Warning",
            event: { message: availableSkills.warningMessage },
          });
        }
        const turnConfig: TurnConfig = {
          apiKey: this.config.apiKey,
          baseUrl: this.config.baseUrl,
          fetch: this.config.fetch,
          model: op.model ?? this.config.model,
          compactWindow: this.compactWindow,
          tokenState: this.tokenState,
          ...(instructions ? { instructions } : {}),
          ...(contextItems.length > 0 ? { contextItems } : {}),
          ...(this.config.skills.length > 0
            ? { skills: this.config.skills }
            : {}),
          ...(this.config.loadSkillContent
            ? { loadSkillContent: this.config.loadSkillContent }
            : {}),
          ...(this.config.autoCompactTokenLimit !== undefined
            ? { autoCompactTokenLimit: this.config.autoCompactTokenLimit }
            : {}),
          ...(this.config.contextWindow !== undefined
            ? { contextWindow: this.config.contextWindow }
            : {}),
          ...(this.config.maxRetries !== undefined
            ? { maxRetries: this.config.maxRetries }
            : {}),
          ...(this.config.compactPrompt !== undefined
            ? { compactPrompt: this.config.compactPrompt }
            : {}),
        };

        const abortController = new AbortController();
        this.currentTurnAbort = abortController;

        // Unix seconds, mirroring TurnStartedEvent.started_at /
        // TurnCompleteEvent.started_at (protocol.rs:2019).
        const startedAtMs = Date.now();
        const startedAt = Math.floor(startedAtMs / 1000);

        this.pushEvent(submissionId, {
          type: "TurnStarted",
          event: { turn_id: turnId } satisfies TurnStartedEvent,
        });

        void runTurn(
          turnId,
          op.items,
          this.history,
          turnConfig,
          this.router,
          this.pendingInputs,
          (msg: EventMsg) => this.pushEvent(submissionId, msg),
          this.liveThread,        // ← persistence hook
          abortController.signal, // ← interrupt hook
          op.extraUserMessages,   // ← additional separate user messages (queue flush)
        )
          .then(({ lastAgentMessage }) => {
            this.pushEvent(submissionId, {
              type: "TurnComplete",
              event: {
                turn_id: turnId,
                last_agent_message: lastAgentMessage || undefined,
                started_at: startedAt,
                completed_at: Math.floor(Date.now() / 1000),
                duration_ms: Date.now() - startedAtMs,
              } satisfies TurnCompleteEvent,
            });
          })
          .catch((err: unknown) => {
            const completedAt = Math.floor(Date.now() / 1000);
            const durationMs = Date.now() - startedAtMs;
            const error = {
              message: String(err),
              codex_error_info: codexErrorInfoFor(err),
              turn_id: turnId,
            };
            // The Error event is emitted for every failure mode, including an
            // interrupt — pre-existing codex-ts behaviour that hosts key off.
            this.pushEvent(submissionId, { type: "Error", event: error });
            // mirrors tasks/mod.rs:785-794: an ABORTED turn terminates with
            // TurnAborted, never TurnComplete.
            if (isAbortError(err)) {
              this.pushEvent(submissionId, {
                type: "TurnAborted",
                event: {
                  turn_id: turnId,
                  reason: "interrupted",
                  started_at: startedAt,
                  completed_at: completedAt,
                  duration_ms: durationMs,
                } satisfies TurnAbortedEvent,
              });
              return;
            }
            // mirrors tasks/mod.rs:795-813: a failed turn still completes the
            // turn lifecycle, with the terminal error attached to TurnComplete
            // (rs reads it from turn_context.terminal_error). Without that
            // second event a host awaiting TurnComplete would hang on failure.
            this.pushEvent(submissionId, {
              type: "TurnComplete",
              event: {
                turn_id: turnId,
                error,
                started_at: startedAt,
                completed_at: completedAt,
                duration_ms: durationMs,
              } satisfies TurnCompleteEvent,
            });
          })
          .finally(() => {
            if (this.currentTurnAbort === abortController) {
              this.currentTurnAbort = null;
            }
          });
        break;
      }

      case "UserInputAnswer": {
        const resolve = this.pendingInputs.get(op.id);
        if (resolve) {
          resolve(op.response);
          this.pendingInputs.delete(op.id);
        }
        break;
      }

      case "Interrupt":
        this.currentTurnAbort?.abort();
        break;
    }

    return submissionId;
  }

  /**
   * Pull the next event. Blocks until one is available.
   * Mirrors: pub async fn next_event(&self) -> CodexResult<Event>
   */
  async nextEvent(): Promise<Event> {
    return this.eventQueue.next();
  }

  private pushEvent(submissionId: string, msg: EventMsg): void {
    this.eventQueue.push({ id: submissionId, msg });
  }
}
