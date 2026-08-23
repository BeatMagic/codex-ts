/**
 * mirrors codex-rs/protocol/src/protocol.rs
 *
 * Subset: Op + EventMsg variants needed for goal, request_user_input,
 * and conversation (turn lifecycle + streaming text).
 */

import type { UserInput } from "./user_input.js";
import type {
  RequestUserInputEvent,
  RequestUserInputResponse,
} from "./request_user_input.js";
import type { UpdatePlanArgs } from "./plan_tool.js";

// ─── ThreadGoal ────────────────────────────────────────────────────────────────

export type ThreadGoalStatus =
  | "Active"
  | "Paused"
  | "Blocked"
  | "UsageLimited"
  | "BudgetLimited"
  | "Complete";

export interface ThreadGoal {
  /** mirrors ThreadGoal.thread_id (protocol.rs:4071) — the owning thread. */
  thread_id: string;
  objective: string;
  status: ThreadGoalStatus;
  token_budget?: number | undefined;
  tokens_used: number;
  time_used_seconds: number;
  /** Unix epoch milliseconds — mirrors created_at/updated_at (protocol.rs:4079). */
  created_at: number;
  updated_at: number;
}

/** mirrors MAX_THREAD_GOAL_OBJECTIVE_CHARS (protocol.rs:4053) */
export const MAX_THREAD_GOAL_OBJECTIVE_CHARS = 4_000;

/** mirrors validate_thread_goal_objective (protocol.rs:4055) — returns the
 *  error message, or null when the objective is acceptable. */
export function validateThreadGoalObjective(value: string): string | null {
  if (value.length === 0) {
    return "goal objective must not be empty";
  }
  if (Array.from(value).length > MAX_THREAD_GOAL_OBJECTIVE_CHARS) {
    return `goal objective must be at most ${MAX_THREAD_GOAL_OBJECTIVE_CHARS} characters`;
  }
  return null;
}

// ─── Event payloads ────────────────────────────────────────────────────────────

export interface TurnStartedEvent {
  turn_id: string;
}

/**
 * mirrors TurnCompleteEvent (protocol.rs:1986). codex-rs emits this at the END
 * of EVERY turn — successful or not — carrying the terminal error when the turn
 * failed (tasks/mod.rs:803). codex-ts mirrors that: a failing turn now emits
 * BOTH `Error` (unchanged, for hosts that key off it) and a final
 * `TurnComplete` with `error` set, so a host looping until TurnComplete no
 * longer hangs on failure.
 */
export interface TurnCompleteEvent {
  turn_id: string;
  last_agent_message?: string | undefined;
  /** Terminal error details when the turn completed unsuccessfully. */
  error?: ErrorEvent | undefined;
  /** Unix timestamp (seconds) when the turn started / completed. */
  started_at?: number | undefined;
  completed_at?: number | undefined;
  /** Turn wall-clock duration in milliseconds, when known. */
  duration_ms?: number | undefined;
}

/** mirrors TurnAbortReason (protocol.rs:4209). codex-ts only ever produces
 *  `interrupted` (Op.Interrupt); the rest exist for shape parity. */
export type TurnAbortReason =
  | "interrupted"
  | "replaced"
  | "review_ended"
  | "budget_limited";

/**
 * mirrors TurnAbortedEvent (protocol.rs:4190). codex-rs ends an aborted turn with
 * THIS event instead of TurnComplete (tasks/mod.rs:785-794), so an interrupted
 * turn has its own terminal event rather than a success-shaped one.
 */
export interface TurnAbortedEvent {
  turn_id?: string | undefined;
  reason: TurnAbortReason;
  started_at?: number | undefined;
  completed_at?: number | undefined;
  duration_ms?: number | undefined;
}

/** Full assistant text for a completed message */
export interface AgentMessageEvent {
  message: string;
}

/** Streaming text chunk — mirrors AgentMessageContentDeltaEvent */
export interface AgentMessageContentDeltaEvent {
  turn_id: string;
  item_id: string;
  delta: string;
}

/** Streaming reasoning (thinking) chunk — mirrors codex-rs
 *  ReasoningContentDeltaEvent (protocol.rs:1873). Emitted while the model streams
 *  its reasoning/thinking, before any final output_text. Lets hosts show a
 *  "thinking" state only while truly reasoning. */
export interface ReasoningContentDeltaEvent {
  turn_id: string;
  /** Reasoning item this delta belongs to — mirrors item_id (protocol.rs:1876). */
  item_id: string;
  delta: string;
  /**
   * Which reasoning block the delta belongs to — mirrors summary_index
   * (protocol.rs:1880). Sourced from the SSE `summary_index` for
   * `response.reasoning_summary_text.delta` and from `content_index` for
   * `response.reasoning_text.delta` (codex-api/src/sse/responses.rs), so hosts
   * can keep separate reasoning blocks apart. Defaults to 0, like rs's
   * `#[serde(default)]`.
   */
  summary_index: number;
}

export interface ThreadGoalUpdatedEvent {
  /** mirrors ThreadGoalUpdatedEvent.thread_id (protocol.rs:4087) */
  thread_id: string;
  turn_id?: string | undefined;
  goal: ThreadGoal;
}

/**
 * mirrors codex-rs/protocol/src/protocol.rs CodexErrorInfo (protocol.rs:1758) —
 * the client-facing error classification. codex-ts produces the subset it can
 * actually observe over HTTP/SSE (see codexErrorInfoFor in session/retry.ts);
 * the remaining variants are kept for shape parity with rs.
 */
export type CodexErrorInfo =
  | { type: "context_window_exceeded" }
  | { type: "session_budget_exceeded" }
  | { type: "usage_limit_exceeded" }
  | { type: "server_overloaded" }
  | { type: "cyber_policy" }
  | { type: "http_connection_failed"; http_status_code?: number | undefined }
  | {
      type: "response_stream_connection_failed";
      http_status_code?: number | undefined;
    }
  | { type: "internal_server_error" }
  | { type: "unauthorized" }
  | { type: "bad_request" }
  | { type: "sandbox_error" }
  | {
      type: "response_stream_disconnected";
      http_status_code?: number | undefined;
    }
  | {
      type: "response_too_many_failed_attempts";
      http_status_code?: number | undefined;
    }
  | { type: "thread_rollback_failed" }
  | { type: "other" };

export interface ErrorEvent {
  message: string;
  /**
   * mirrors ErrorEvent.codex_error_info (protocol.rs:1925) — lets hosts branch
   * on the error class instead of matching message text.
   */
  codex_error_info?: CodexErrorInfo | undefined;
  /**
   * Browser-specific extension — no equivalent in codex-rs/protocol ErrorEvent.
   * Lets hosts correlate an error with the turn that produced it (e.g. to
   * suppress stale-turn errors after an Interrupt).
   */
  turn_id?: string | undefined;
}

// ─── EventMsg ─────────────────────────────────────────────────────────────────

/** mirrors codex-rs/protocol/src/protocol.rs ContextCompactedEvent (unit struct) */
export type ContextCompactedEvent = Record<never, never>;

/** mirrors WarningEvent */
export interface WarningEvent {
  message: string;
}

/** mirrors TokenUsage (protocol.rs:2056) */
export interface TokenUsage {
  input_tokens: number;
  cached_input_tokens: number;
  /** Prompt-cache WRITE tokens — mirrors cache_write_input_tokens
   *  (protocol.rs:2063), read from `usage.input_tokens_details.cache_write_tokens`
   *  (codex-api/src/sse/responses.rs:137). 0 when the provider omits it. */
  cache_write_input_tokens: number;
  output_tokens: number;
  reasoning_output_tokens: number;
  total_tokens: number;
}

/** mirrors TokenUsageInfo (protocol.rs:2073) */
export interface TokenUsageInfo {
  total_token_usage: TokenUsage;
  last_token_usage: TokenUsage;
  model_context_window: number | null;
}

/** mirrors TokenCountEvent (protocol.rs:2000). `rate_limits` exists in rs
 *  (Option<RateLimitSnapshot>); the ts port does not track provider rate
 *  limits, so it is always null here — kept in the shape for parity. */
export interface TokenCountEvent {
  info: TokenUsageInfo | null;
  rate_limits: null;
}

// ─── Function-call composition (browser-specific) ─────────────────────────────
//
// No equivalent in codex-rs/protocol: a native TUI renders the tool call from
// `response.output_item.done`, which arrives only once the model has finished
// writing the arguments. In a browser host that gap is visible — a long
// `doc.create` spends seconds as blank "Loading" while the model composes it.
//
// These two events mirror the wire as it streams so a host can show the tool
// name immediately and the arguments as they arrive. They are DISPLAY-ONLY by
// construction: nothing derived from them feeds the dispatched call, whose
// arguments still come solely from `response.output_item.done`.

/** A function call has started streaming; its name is known, arguments are not. */
export interface FunctionCallStreamStartEvent {
  turn_id: string;
  /** The streaming item's id — matches the `id` on FunctionCallStreamArgs. */
  id: string;
  name: string;
}

/** One raw `arguments` delta for a function call that is still streaming. */
export interface FunctionCallStreamArgsEvent {
  turn_id: string;
  id: string;
  delta: string;
}

export type EventMsg =
  | { type: "TurnStarted"; event: TurnStartedEvent }
  | { type: "TurnComplete"; event: TurnCompleteEvent }
  /** mirrors EventMsg::TurnAborted — terminal event for an interrupted turn */
  | { type: "TurnAborted"; event: TurnAbortedEvent }
  | { type: "AgentMessage"; event: AgentMessageEvent }
  | { type: "AgentMessageContentDelta"; event: AgentMessageContentDeltaEvent }
  | { type: "ReasoningContentDelta"; event: ReasoningContentDeltaEvent }
  | { type: "RequestUserInput"; event: RequestUserInputEvent }
  | { type: "ThreadGoalUpdated"; event: ThreadGoalUpdatedEvent }
  | { type: "PlanUpdate"; event: UpdatePlanArgs }
  /** mirrors EventMsg::ContextCompacted — emitted when inline compaction runs */
  | { type: "ContextCompacted"; event: ContextCompactedEvent }
  /** mirrors EventMsg::TokenCount (protocol.rs:1211) — emitted after each
   *  sampled response's usage is recorded (send_token_count_event,
   *  session/mod.rs:3131); hosts use it to display context consumption. */
  | { type: "TokenCount"; event: TokenCountEvent }
  /** mirrors EventMsg::Warning — post-compaction advisory */
  | { type: "Warning"; event: WarningEvent }
  /** Browser-specific: a function call began streaming (display only). */
  | { type: "FunctionCallStreamStart"; event: FunctionCallStreamStartEvent }
  /** Browser-specific: one `arguments` delta of a streaming call (display only). */
  | { type: "FunctionCallStreamArgs"; event: FunctionCallStreamArgsEvent }
  | { type: "Error"; event: ErrorEvent };

/** Wraps EventMsg with the submission_id that originated it — mirrors codex-rs Event */
export interface Event {
  /** submission_id returned by CodexThread.submit() */
  id: string;
  msg: EventMsg;
}

// ─── Op ───────────────────────────────────────────────────────────────────────

// =============================================================================
// Browser-specific adaptation of Op::UserInput.
//
// codex-rs/protocol/src/protocol.rs carries a full `ThreadSettingsOverrides`
// struct on UserInput (cwd, approval_policy, sandbox_policy, permission_profile,
// collaboration_mode, …). These fields are native-only and meaningless in the
// browser, so we expose only the two that are relevant: per-turn model and
// per-turn instructions. Functionally equivalent to the subset of
// ThreadSettingsOverrides::model and
// ThreadSettingsOverrides::collaboration_mode.settings.developer_instructions.
// =============================================================================

export type Op =
  | {
      type: "UserInput";
      items: UserInput[];
      /**
       * Additional SEPARATE user messages to record alongside `items` in this one
       * turn — each entry is one message's content items and becomes its own
       * role:user history item, never merged into `items`. Mirrors codex-rs
       * draining several queued UserInput submissions into the turn as distinct
       * messages, so the app can flush a whole send-queue in one turn ("send all
       * queued at once, not merged into a single message"). Omitted = single
       * message, identical to before.
       */
      extraUserMessages?: UserInput[][] | undefined;
      /**
       * Per-turn instructions override. Subset of ThreadSettingsOverrides in
       * codex-rs. Falls back to the thread's instructions when omitted.
       */
      instructions?: string | undefined;
      /**
       * Per-turn model override. Subset of ThreadSettingsOverrides in codex-rs.
       * Falls back to the thread's model when omitted.
       */
      model?: string | undefined;
    }
  | {
      type: "UserInputAnswer";
      /** RequestUserInputEvent.turn_id — identifies which turn to unblock */
      id: string;
      response: RequestUserInputResponse;
    }
  | { type: "Interrupt" };

export type { RequestUserInputEvent, RequestUserInputResponse };
export type { UpdatePlanArgs } from "./plan_tool.js";
export type { StepStatus, PlanItemArg } from "./plan_tool.js";
