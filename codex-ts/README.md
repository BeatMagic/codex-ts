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
