# @beatmagic/codex-ts

A TypeScript port of the codex-rs agent core — runs the Codex sampling loop
(Responses API streaming + tool calls + auto-compaction) directly in Node or the
browser, with no spawned process.

```ts
import { CodexThread } from "@beatmagic/codex-ts";

const thread = new CodexThread({
  apiKey: "…",
  baseUrl: "https://api.openai.com/v1",
  model: "gpt-5.4",
});

const turnId = await thread.submit({
  type: "UserInput",
  items: [{ type: "text", text: "hello" }],
});

for (;;) {
  const { msg } = await thread.nextEvent();
  if (msg.type === "AgentMessage") console.log(msg.event.message);
  if (msg.type === "TurnComplete") break;
}
```

## Retries

Transient Responses failures are retried automatically with exponential
backoff — mirrors codex-rs's stream retry loop (`responses_retry.rs`).

A failure is **retried** when it happens **before any visible output has
streamed** and is transient:

- a network error (fetch rejects: connection reset, DNS, CORS-masked failure…);
- an HTTP `408`, `409`, `429`, or any `5xx` (e.g. a `502` during a deploy
  window);
- the SSE stream dropping before the first text delta;
- a `response.failed` event whose error code is transient (e.g.
  `rate_limit_exceeded`) or unrecognised — matching codex-rs, which defaults
  `response.failed` to retryable;
- a `response.incomplete` event (surfaced as a stream error carrying
  `incomplete_details.reason`).

A failure is **not retried** when:

- visible output has already streamed (a re-stream would duplicate text);
- the error is terminal — `4xx` other than the above (e.g. `400`, `401`), or the
  turn was interrupted (`AbortError`);
- a `response.failed` event carries a terminal error code — `insufficient_quota`,
  `usage_not_included`, `cyber_policy`, `invalid_prompt`, `bio_policy`,
  `server_is_overloaded`, `slow_down` (mirrors the non-retryable `CodexErr`
  variants in `protocol/src/error.rs`), or `context_length_exceeded`, which
  instead triggers compaction / the context self-heal;
- the retry budget (`maxRetries`) is exhausted.

Each retry waits an exponential backoff — `200ms × 2^(n-1)` with ±10% jitter —
honoring a server `Retry-After` header when present, or a provider
`…please try again in 1.5s` hint on a rate limit (mirrors `try_parse_retry_after`).
A `Warning` event (`Reconnecting... n/m`) is emitted before each attempt so the UI
can surface it.

When the retries are exhausted (or the failure was terminal), the turn emits an
`Error` event whose `codex_error_info` classifies the cause, followed by a final
`TurnComplete` carrying the same error — so a loop awaiting `TurnComplete` always
terminates. An **interrupted** turn instead ends with `TurnAborted`
(`reason: "interrupted"`), mirroring codex-rs, which never sends `TurnComplete`
for an aborted turn.

### Configuration

```ts
const thread = new CodexThread({
  apiKey: "…",
  model: "gpt-5.4",
  // Max retries for transient request/stream failures. Default: 5.
  // Pass 0 to disable retries entirely.
  maxRetries: 5,
});
```

| Option       | Default | Meaning                                                    |
| ------------ | ------- | ---------------------------------------------------------- |
| `maxRetries` | `5`     | Max retry attempts after the initial request. `0` disables. |

The same budget covers both the initial request and a stream that drops before
any output; `Retry-After` overrides the computed backoff for that attempt.

## Custom tools

Host tools are passed as `customTools` (`CustomTool`: `name`, `spec()`,
`execute(args, ctx)`). `execute` returns the call's output — a string, or an
array of `input_text` / `input_image` items (`FunctionCallOutputContentItem`,
mirrors codex-rs `FunctionCallOutputBody::ContentItems`) to hand the model an
image.

- **Interrupts.** `ctx.signal` aborts when the turn is interrupted. The turn
  does not wait for the tool: it records `aborted by user after Xs` for the
  running call (and a not-executed output for calls of the round that never
  started), persists them, and ends with `TurnAborted` at once; a result the
  tool returns later is discarded. A turn parked in `request_user_input` is
  interrupted the same way. `Interrupt` reaches every in-flight turn of the
  thread.
- **Errors.** A tool that throws does not end the turn: the error message
  becomes the call's output and the model carries on (codex-rs
  `failure_response`).
- **Parallel calls (opt-in).** `supportsParallel: true` lets a tool's calls
  run alongside other parallel-safe calls of the same round; any other call
  is exclusive — it waits for every earlier call and blocks later ones until it
  settles (codex-rs's shared/exclusive lock). Outputs reach the model in call
  order. Mark only tools that are read-only and safe to overlap.
- **Output truncation (opt-in).** `maxOutputBytes` middle-truncates a tool's
  output to that many UTF-8 bytes of text (images are kept) before it enters
  the model's history, with a `…N chars truncated…` marker — the codex-rs
  algorithm. `DEFAULT_TOOL_OUTPUT_MAX_BYTES` (12 000) is codex-rs's budget.
  Leave it unset for outputs the model must read whole.

A thread sends at most one model request at a time (sampling or compaction);
turns that overlap queue their requests, never their tools.

## Persistence and resume

With a `threadStore` or `ioBackend`, a thread's rollout is persisted as one
append-only log per thread, mirroring codex-rs's rollout:

- every conversation item (`ConversationItem`), in the order it entered history;
- a `Compacted` checkpoint each time inline auto-compaction rewrites history —
  the summary message plus the complete replacement history, written in the same
  tick as the in-memory swap;
- `TokenCount` events: the server-measured total after each sampled response,
  the estimate right after a compaction, and the forced-full mark set when the
  provider rejects a request as too long.

Appends go through an ordered queue, so lines land in call order even when
concurrent turns share a thread and the backend completes writes out of order.
A failed write is logged and skipped; it never fails the turn.

`CodexThread.create({ threadId, … })` resumes from the **newest** checkpoint and
replays only the items appended after it (codex-rs `reconstruct_rollout`), then
seeds the session token total from the last `TokenCount` after that checkpoint.
A thread that was past its context window — or was just rejected as too long —
therefore compacts before its first sample after a restart, instead of sending
an oversized request.

`LiveThread.loadConversationHistory()` still returns the **raw** transcript —
every item ever appended — for chat UIs; use `LiveThread.loadResumeState()`
for the model-visible context.

Tool outputs are persisted whole; a tool's `maxOutputBytes` applies only to the
model's copy, so a resumed thread re-applies it to the items it replays (not to
a checkpoint's replacement history, which is already the model's copy).

Logs written before checkpoints existed (no `Compacted` line) resume as a full
replay, as before. Older readers skip the new line kinds and keep their full
replay, so mixed runtimes are safe.

## Inline compaction

When the context reaches `autoCompactTokenLimit` (growth since the last
compaction) or `contextWindow` (total), the thread summarizes its history with
one extra Responses request and replaces the history with the summary plus the
most recent user messages.

- **The request extends the sampling request.** It carries the same `tools`,
  `reasoning` and `instructions`, and its `input` is what the last sampling
  request sent plus what was recorded since, so a prefix-caching provider reads
  the cache instead of writing the whole history again. The summarization
  prompt is always the **last** input item, as
  `{ type: "message", role: "user", content: <prompt> }`. If the model calls a
  tool or returns no text, that compaction retries once without tools.
- **Before a turn, compaction runs before the new input is recorded.** The
  summary covers the history up to the new message, which follows it. If the
  compaction fails, the input is still recorded.
- Context tokens are counted as the last measured or estimated total plus an
  estimate of every item that total does not cover yet.

| Option                    | Default | Meaning |
| ------------------------- | ------- | ------- |
| `compactionRequestParity` | `true`  | `false` sends the older summarizer body: history plus prompt, no tools or reasoning. |
| `isContextualUserMessage` | none    | `(text) => boolean`. User messages it returns `true` for (host-injected context, given their joined `input_text`) are not kept verbatim through a compaction. |
| `enableGoalTools`         | `true`  | `false` stops advertising `get_goal`, `create_goal` and `update_goal`; a call to one gets the unknown-tool reply. |
