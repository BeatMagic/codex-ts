/**
 * mirrors codex-rs/core/src/tools/router.rs
 *
 * Routes tool calls by name to the appropriate handler.
 * Returns the tool output string + any side-effect EventMsgs to emit.
 */

import type { EventMsg } from "../../../protocol/src/protocol.js";
import type { RequestUserInputResponse } from "../../../protocol/src/request_user_input.js";
import type { ToolSpec } from "../../../tools/src/tool_spec.js";
import type { FunctionCallOutputContentItem } from "../../../thread-store/src/types.js";
import { GoalToolExecutor } from "../../../ext/goal/src/tool.js";
import {
  CREATE_GOAL_TOOL_NAME,
  GET_GOAL_TOOL_NAME,
  UPDATE_GOAL_TOOL_NAME,
  createCreateGoalTool,
  createGetGoalTool,
  createUpdateGoalTool,
} from "../../../ext/goal/src/spec.js";
import {
  REQUEST_USER_INPUT_TOOL_NAME,
  normalizeRequestUserInputArgs,
  createRequestUserInputTool,
} from "./handlers/request_user_input_spec.js";
import {
  handleRequestUserInput,
  formatRequestUserInputOutput,
} from "./handlers/request_user_input.js";
import type { PendingInputs } from "./handlers/request_user_input.js";
import {
  UPDATE_PLAN_TOOL_NAME,
  createUpdatePlanTool,
  parseUpdatePlanArgs,
} from "./handlers/plan_spec.js";

export interface ToolRouterContext {
  turnId: string;
  pendingInputs: PendingInputs;
  /** Emit an event immediately — used for events that must fire before suspension */
  emitEvent: (msg: EventMsg) => void;
  /** The turn's abort signal, handed to every tool (CustomToolContext.signal)
   *  and to a parked request_user_input. Omitted → a signal that never aborts. */
  signal?: AbortSignal | undefined;
}

// =============================================================================
// Browser-specific extension — no direct equivalent in codex-rs.
//
// codex-rs registers tools via the `ToolExecutor<ToolInvocation>` trait and
// the `CoreToolRuntime` marker; TypeScript has no traits, so we expose the
// same concept as a plain interface that callers inject into CodexThread.
// =============================================================================

/**
 * Context handed to a CustomTool's execute(). Carries the identifiers a host
 * needs to correlate the call (callId / turnId), an emitEvent escape hatch and
 * the turn's abort signal.
 */
export interface CustomToolContext {
  callId: string;
  turnId: string;
  emitEvent: (msg: EventMsg) => void;
  /**
   * Aborts when the turn is interrupted. mirrors the cancellation token rs
   * hands dynamic tools (handlers/dynamic.rs, #51556). The turn does not wait
   * for the tool: on interrupt it records "aborted by user after Xs" for the
   * call and ends at once, discarding whatever the tool returns later — so a
   * tool should stop its own work (abort its fetches, stop polling) here.
   */
  signal: AbortSignal;
}

/**
 * A host-supplied tool.
 *
 * TypeScript equivalent of implementing `ToolExecutor<ToolInvocation>` in
 * codex-rs: the host provides the spec (advertised to the model) and an async
 * execute() whose result becomes the function_call_output sent back.
 */
export interface CustomTool {
  /** Tool name as advertised to and called by the model */
  name: string;
  /** ToolSpec included in Responses API requests */
  spec(): ToolSpec;
  /**
   * Run the call; the result is fed back to the model as the output — a
   * string, or content items (e.g. `[input_image, input_text]` to show the
   * model an image; mirrors FunctionCallOutputBody::ContentItems). A throw is
   * reported to the model as the call's output; the turn carries on.
   */
  execute(
    args: unknown,
    ctx: CustomToolContext,
  ): Promise<string | FunctionCallOutputContentItem[]>;
  /**
   * Opt this tool's outputs into middle truncation: at most this many UTF-8
   * bytes of text enter the model-visible history (images are kept), with a
   * "…N chars truncated…" marker in the middle. The full output is still
   * persisted, so a transcript replays it whole; a resumed thread truncates
   * it again. DEFAULT_TOOL_OUTPUT_MAX_BYTES is codex-rs's budget. Absent →
   * the output is recorded verbatim. Leave it unset for outputs the model must
   * read whole (documents, JSON it parses, ids it copies).
   */
  maxOutputBytes?: number | undefined;
}

const GOAL_TOOL_NAMES: ReadonlySet<string> = new Set([
  GET_GOAL_TOOL_NAME,
  CREATE_GOAL_TOOL_NAME,
  UPDATE_GOAL_TOOL_NAME,
]);

/** Built-in tool switches. */
export interface ToolRouterOptions {
  /**
   * Advertise and dispatch the goal tools. Default true. rs only registers
   * them when the goal extension is enabled (ext/goal/src/extension.rs
   * `goals_enabled`); a host that never sets goals should not pay for three
   * tool definitions on every request.
   */
  enableGoalTools?: boolean | undefined;
}

/** mirrors ToolRouter in router.rs */
export class ToolRouter {
  private readonly goalExecutor: GoalToolExecutor;
  private readonly customTools: Map<string, CustomTool>;
  private readonly enableGoalTools: boolean;

  constructor(
    goalExecutor: GoalToolExecutor,
    customTools: CustomTool[] = [],
    options: ToolRouterOptions = {},
  ) {
    this.goalExecutor = goalExecutor;
    this.customTools = new Map(customTools.map((tool) => [tool.name, tool]));
    this.enableGoalTools = options.enableGoalTools ?? true;
  }

  /**
   * The output byte budget of `toolName`, when it is a custom tool that opted
   * in (CustomTool.maxOutputBytes); built-ins are never truncated, including
   * one that shadows a custom tool of the same name (see dispatch).
   */
  maxOutputBytesFor(toolName: string): number | undefined {
    const builtin =
      toolName === REQUEST_USER_INPUT_TOOL_NAME ||
      toolName === UPDATE_PLAN_TOOL_NAME ||
      (this.enableGoalTools && GOAL_TOOL_NAMES.has(toolName));
    if (builtin) return undefined;
    const maxBytes = this.customTools.get(toolName)?.maxOutputBytes;
    return typeof maxBytes === "number" && Number.isFinite(maxBytes) && maxBytes >= 0
      ? Math.floor(maxBytes)
      : undefined;
  }

  /** All tool specs to include in Responses API requests */
  toolSpecs(): ToolSpec[] {
    return [
      ...(this.enableGoalTools
        ? [createGetGoalTool(), createCreateGoalTool(), createUpdateGoalTool()]
        : []),
      createRequestUserInputTool(),
      createUpdatePlanTool(),
      ...Array.from(this.customTools.values(), (tool) => tool.spec()),
    ];
  }

  /**
   * Dispatch a tool call; may suspend (request_user_input).
   * Side-effect events are emitted immediately via ctx.emitEvent rather than
   * returned, so RequestUserInput fires before the turn suspends.
   */
  async dispatch(
    toolName: string,
    callId: string,
    rawArgs: unknown,
    ctx: ToolRouterContext,
  ): Promise<string | FunctionCallOutputContentItem[]> {
    const args = (typeof rawArgs === "object" && rawArgs !== null
      ? rawArgs
      : {}) as Record<string, unknown>;
    // Disabled goal tools are not advertised, so their names are ordinary
    // names: a call falls to the default arm like any other.
    const builtinName =
      this.enableGoalTools || !GOAL_TOOL_NAMES.has(toolName)
        ? toolName
        : undefined;

    switch (builtinName) {
      case GET_GOAL_TOOL_NAME: {
        const { output } = await this.goalExecutor.get();
        return output;
      }

      case CREATE_GOAL_TOOL_NAME: {
        const { output, event } = await this.goalExecutor.create(
          args["objective"] as string,
          args["token_budget"] as number | undefined,
        );
        if (event) ctx.emitEvent({ type: "ThreadGoalUpdated", event });
        return output;
      }

      case UPDATE_GOAL_TOOL_NAME: {
        const { output, event } = await this.goalExecutor.update(
          args["status"] as "complete" | "blocked",
        );
        if (event) ctx.emitEvent({ type: "ThreadGoalUpdated", event });
        return output;
      }

      case REQUEST_USER_INPUT_TOOL_NAME: {
        const normalized = normalizeRequestUserInputArgs(
          args as Parameters<typeof normalizeRequestUserInputArgs>[0],
        );
        if ("error" in normalized) {
          return JSON.stringify({ error: normalized.error });
        }
        // Emit BEFORE suspending so the client can see the event and answer.
        // autoResolutionMs rides along (mirrors RequestUserInputEvent
        // .auto_resolution_ms) so the host can decide whether the question is
        // blocking or may be auto-resolved after that window.
        ctx.emitEvent({
          type: "RequestUserInput",
          event: {
            call_id: callId,
            turn_id: ctx.turnId,
            questions: normalized.questions,
            ...(normalized.autoResolutionMs !== undefined
              ? { autoResolutionMs: normalized.autoResolutionMs }
              : {}),
          },
        });
        const response = await handleRequestUserInput(
          {
            turnId: ctx.turnId,
            pendingInputs: ctx.pendingInputs,
            signal: ctx.signal,
          },
          normalized.questions,
        );
        return formatRequestUserInputOutput(response);
      }

      case UPDATE_PLAN_TOOL_NAME: {
        const parsed = parseUpdatePlanArgs(args);
        if (!parsed.ok) {
          return JSON.stringify({
            error: parsed.error,
            success: false,
          });
        }
        ctx.emitEvent({ type: "PlanUpdate", event: parsed.args });
        return "Plan updated";
      }

      default: {
        const custom = this.customTools.get(toolName);
        if (custom) {
          // Hand the raw (un-normalised) args to the host; it does its own parsing.
          return custom.execute(rawArgs, {
            callId,
            turnId: ctx.turnId,
            emitEvent: ctx.emitEvent,
            signal: ctx.signal ?? new AbortController().signal,
          });
        }
        // A bare `{error: "Unknown tool: <name>"}` returns the one fact the
        // model already had, and leaves the failure ambiguous between the only
        // two causes: "I wrote the wrong name" and "this session has no such
        // capability". Models reliably pick the second, and the second is
        // catastrophic. Observed 2026-08-19 in two production sessions, both at
        // the same workflow step, both calling a tool from the model's own
        // harness that was never registered here: one then told the user that
        // the documents it had really generated "were made up with no real tool
        // behind them", while the rendered output sat in the project library.
        // Both recovered the instant the user said "try again" — the whole
        // incident was one unrecognised name plus an error that could not
        // disambiguate.
        //
        // So the error carries the roster and names the likelier cause. The
        // roster is exactly what toolSpecs() advertises, which makes an absent
        // name provable rather than a guess.
        return JSON.stringify({
          error: `Unknown tool: ${toolName}`,
          known_tools: this.toolSpecs().map((spec) => spec.tool.name),
          hint:
            `"${toolName}" is not one of this session's tools. known_tools is the ` +
            `complete list and every name on it works. This is a WRONG NAME, not a ` +
            `missing capability: take the closest name from known_tools and re-send ` +
            `the same call. Do not conclude from this error that a tool, a skill or ` +
            `a capability is unavailable, and do not revise or disown anything you ` +
            `already accomplished through the tools that ARE listed.`,
        });
      }
    }
  }
}
