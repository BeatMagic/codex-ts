/**
 * mirrors codex-rs/core/src/responses_retry.rs + codex-rs/core/src/util.rs::backoff
 *
 * Shared retry decisions for Responses API requests: which failures are
 * transient (retryable), and how long to back off between attempts.
 *
 * Unlike codex-rs (which can re-stream in place into a TUI), codex-ts emits
 * AgentMessageContentDelta events live, so the turn loop only retries failures
 * that happen BEFORE any visible output was streamed — see runTurn. The decision
 * helpers here are agnostic to that policy.
 */

// mirrors codex-rs util.rs: INITIAL_DELAY_MS = 200, BACKOFF_FACTOR = 2.0,
// jitter uniform in [0.9, 1.1).
const INITIAL_DELAY_MS = 200;
const BACKOFF_FACTOR = 2;

/** Default retry budget for transient Responses failures (mirrors the spirit of
 *  codex-rs stream_max_retries; the host can override via CodexThreadConfig). */
export const DEFAULT_MAX_RETRIES = 5;

/**
 * Exponential backoff with jitter, in milliseconds.
 * delay = INITIAL_DELAY_MS * BACKOFF_FACTOR^(attempt-1) * jitter(0.9..1.1).
 * `attempt` is 1-based (first retry = 1).
 */
export function backoff(attempt: number): number {
  const exp = BACKOFF_FACTOR ** Math.max(0, attempt - 1);
  const base = INITIAL_DELAY_MS * exp;
  const jitter = 0.9 + Math.random() * 0.2;
  return Math.round(base * jitter);
}

/**
 * A non-2xx Responses API response, carrying the status and (when present) the
 * server-requested Retry-After delay so the retry loop can honor it.
 *
 * `terminal` marks a failure that must NOT be retried even though its status
 * would normally look transient — mirrors codex-rs mapping certain
 * `response.failed` payload codes to non-retryable CodexErr variants
 * (`insufficient_quota`, `cyber_policy`, `invalid_prompt`, `server_is_overloaded`…
 * see CodexErr::is_retryable, protocol/src/error.rs:358).
 */
export class ResponsesApiError extends Error {
  readonly status: number;
  readonly retryAfterMs: number | undefined;
  readonly terminal: boolean;

  constructor(
    status: number,
    body: string,
    retryAfterMs?: number | undefined,
    terminal = false,
  ) {
    super(`Responses API ${status}: ${body}`);
    this.name = "ResponsesApiError";
    this.status = status;
    this.retryAfterMs = retryAfterMs;
    this.terminal = terminal;
  }
}

/**
 * Whether an HTTP status is a transient failure worth retrying. Mirrors the
 * retryable set in codex-rs (timeouts / 5xx / unexpected status); 4xx are
 * terminal except 408 (request timeout), 409 (conflict) and 429 (rate limit).
 */
export function isRetryableStatus(status: number): boolean {
  return status === 408 || status === 409 || status === 429 || status >= 500;
}

// mirrors codex-api/src/sse/responses.rs is_context_window_error — codex-rs only
// needs the OpenAI canonical `code: "context_length_exceeded"`, because its
// upstream speaks pure OpenAI. codex-ts rides OpenRouter/codeproxy, whose error
// bodies wrap the ORIGINAL provider error verbatim (`metadata.raw`), so the
// canonical code rarely survives — match each provider's wording too:
//   - OpenAI:            "context_length_exceeded" / "exceeds the context window"
//   - Anthropic/Bedrock: "prompt is too long: N tokens > M maximum"
//   - Gemini:            "input token count ... exceeds the maximum"
//   - OpenRouter ENDPOINT pre-check (fires before any provider is tried, so
//     none of the provider wordings appear): "This endpoint's maximum context
//     length is N tokens. However, you requested about M tokens" — missing
//     this bricked the self-heal loop in prod (2026-07-18 a1d14926: a 1.42M
//     thread 400ed forever because setFull/compact never armed).
const CONTEXT_WINDOW_ERROR_RE =
  /context_length_exceeded|exceeds the context window|prompt is too long|exceeds the maximum number of tokens|input token count .{0,40}exceeds|maximum context length/iu;

/** Message-level probe — rs classifies purely by the error payload (its
 *  canonical `code`), never by HTTP status; SSE `response.failed` has no
 *  status at all. Exposed for both the turn loop and the compaction task. */
export function isContextWindowExceededText(text: string): boolean {
  return CONTEXT_WINDOW_ERROR_RE.test(text);
}

/**
 * Classify a terminal request failure as context-window-exceeded.
 * mirrors CodexErr::ContextWindowExceeded classification: turn.ts marks the
 * session tokens FULL on this error so the next turn compacts before sampling
 * (turn.rs:1045-1047), and compact.ts trims the oldest history item and
 * retries (compact.rs:232-241). Callers must check this BEFORE the retry
 * decision — the same request would fail again.
 */
export function isContextWindowExceededError(error: unknown): boolean {
  return (
    error instanceof ResponsesApiError &&
    isContextWindowExceededText(error.message)
  );
}

/**
 * Classify a thrown error as retryable. Aborts (interrupts) are terminal; an
 * explicitly terminal ResponsesApiError (see classifyStreamFailure) is never
 * retried; otherwise ResponsesApiError defers to its status, and anything else
 * (network TypeError, dropped stream, JSON/IO error) is treated as a transient
 * connection failure — mirrors codex-rs CodexErr::is_retryable
 * (ConnectionFailed / Stream / Io / UnexpectedStatus → true).
 */
export function isRetryableError(error: unknown): boolean {
  if (isAbortError(error)) {
    return false;
  }
  if (error instanceof ResponsesApiError) {
    return !error.terminal && isRetryableStatus(error.status);
  }
  return true;
}

// ─── SSE terminal-event classification (mirrors codex-api sse/responses.rs) ────

/**
 * Error codes that make a `response.failed` event terminal.
 * mirrors process_responses_event's `response.failed` arm
 * (codex-api/src/sse/responses.rs:387): each of these maps to a CodexErr that
 * `is_retryable()` rejects, so re-sending the identical request cannot help.
 *   - insufficient_quota  → CodexErr::QuotaExceeded
 *   - usage_not_included  → CodexErr::UsageNotIncluded
 *   - cyber_policy        → CodexErr::CyberPolicy
 *   - invalid_prompt / bio_policy → CodexErr::InvalidRequest
 *   - server_is_overloaded / slow_down → CodexErr::ServerOverloaded
 * (context_length_exceeded is terminal too, but is handled separately because
 * it also drives compaction / the self-heal path.)
 */
const TERMINAL_STREAM_FAILURE_CODES = new Set<string>([
  "insufficient_quota",
  "usage_not_included",
  "cyber_policy",
  "invalid_prompt",
  "bio_policy",
  "server_is_overloaded",
  "slow_down",
]);

/**
 * `response.incomplete` reasons that make the incomplete TERMINAL.
 * mirrors #53451 (bc9cad3473), where rs maps an incomplete to the
 * non-retryable ApiError::ResponseIncomplete — narrowed here to the output
 * cap, the one reason a byte-identical re-send is certain to hit again. It
 * typically trips while the model writes a huge tool-call argument, before any
 * text delta, so the turn loop would retry silently and burn up to
 * (1 + maxRetries)× the output tokens. `max_tokens` covers a proxy that passes
 * Anthropic's stop_reason through literally.
 *
 * Every other reason stays retryable, as before: content_filter (rs keeps it
 * retryable too), unknown reasons, and above all the synthetic transport
 * reasons the ace-mv cloud stream guard seals cut streams with
 * (clean_eof_no_terminal_event, upstream_idle_timeout, upstream_stream_broke,
 * client_disconnected) — it relies on those being retried, and older proxies
 * and runtimes stay in the fleet. Upstream's `interrupted` → follow-up
 * handling is not modelled.
 */
const TERMINAL_INCOMPLETE_REASONS = new Set<string>([
  "max_output_tokens",
  "max_tokens",
]);

/** mirrors rate_limit_regex (codex-api/src/sse/responses.rs:656) */
const RATE_LIMIT_RETRY_AFTER_RE =
  /try again in\s*(\d+(?:\.\d+)?)\s*(ms|s|seconds?)/i;

/**
 * Parse a provider "…please try again in 11.054s" hint into milliseconds.
 * mirrors try_parse_retry_after (codex-api/src/sse/responses.rs:599), which only
 * honors the hint for `rate_limit_exceeded`.
 */
export function parseRateLimitRetryAfterMs(
  code: string | undefined,
  message: string | undefined,
): number | undefined {
  if (code !== "rate_limit_exceeded" || !message) {
    return undefined;
  }
  const match = RATE_LIMIT_RETRY_AFTER_RE.exec(message);
  if (!match) {
    return undefined;
  }
  const value = Number(match[1]);
  if (!Number.isFinite(value)) {
    return undefined;
  }
  const unit = (match[2] ?? "").toLowerCase();
  return unit === "ms" ? Math.round(value) : Math.round(value * 1000);
}

/**
 * Turn a terminal SSE event (`response.failed` / `response.incomplete` / a
 * top-level `error` frame) into a ResponsesApiError with the same retry verdict
 * codex-rs would reach.
 *
 * mirrors process_responses_event (codex-api/src/sse/responses.rs:387-431):
 *   - `response.failed` inspects `response.error.code` and maps it to a
 *     terminal or retryable CodexErr, honoring a "try again in Xs" delay for
 *     rate limits (everything unrecognised stays Retryable, like rs);
 *   - `response.incomplete` carries the `incomplete_details.reason`; it is
 *     terminal only for the output cap (see TERMINAL_INCOMPLETE_REASONS),
 *     otherwise a retryable CodexErr::Stream;
 *   - an `error` frame is classified like `response.failed` (#56445
 *     parse_terminal_provider_error reads streamed `error` events). Its
 *     code/message sit either nested under `error` (rs, Anthropic) or at the
 *     top level (the OpenAI streaming `error` event); both are read. rs skips
 *     a frame whose code it does not recognise and reads on, so the stream
 *     ends without `response.completed` — retryable as well; failing here
 *     reaches that verdict without waiting for an EOF that may never come.
 * The 503 status is a stand-in for "stream-level failure, no HTTP status" so the
 * retryable ones land in isRetryableStatus's set.
 */
export function classifyStreamFailure(raw: Record<string, unknown>): ResponsesApiError {
  const kind = String(raw["type"] ?? "");
  const response = raw["response"] as Record<string, unknown> | undefined;

  if (kind === "response.incomplete") {
    const details = response?.["incomplete_details"] as
      | Record<string, unknown>
      | undefined;
    const reason =
      typeof details?.["reason"] === "string" ? details["reason"] : "unknown";
    return new ResponsesApiError(
      503,
      `Incomplete response returned, reason: ${reason}`,
      undefined,
      TERMINAL_INCOMPLETE_REASONS.has(reason),
    );
  }

  const nested = raw["error"];
  const error =
    kind === "error"
      ? typeof nested === "object" && nested !== null
        ? (nested as Record<string, unknown>)
        : raw
      : (response?.["error"] as Record<string, unknown> | undefined);
  const code = typeof error?.["code"] === "string" ? error["code"] : undefined;
  const message =
    typeof error?.["message"] === "string" ? error["message"] : undefined;
  const body = error !== undefined ? JSON.stringify(error) : JSON.stringify(raw);

  const terminal = code !== undefined && TERMINAL_STREAM_FAILURE_CODES.has(code);
  const retryAfterMs = parseRateLimitRetryAfterMs(code, message);
  return new ResponsesApiError(503, body, retryAfterMs, terminal);
}

export function isAbortError(error: unknown): boolean {
  return (
    error instanceof DOMException &&
    error.name === "AbortError"
  ) ||
    (error instanceof Error && error.name === "AbortError");
}

/**
 * Delay before the next attempt: honor a server Retry-After when the error
 * carries one (e.g. 429), otherwise exponential backoff.
 */
export function computeRetryDelay(error: unknown, attempt: number): number {
  if (
    error instanceof ResponsesApiError &&
    error.retryAfterMs !== undefined &&
    error.retryAfterMs >= 0
  ) {
    return error.retryAfterMs;
  }
  return backoff(attempt);
}

/**
 * Parse a Retry-After header (delta-seconds or HTTP-date) into milliseconds.
 * Returns undefined when absent or unparseable.
 */
export function parseRetryAfter(headers: Headers): number | undefined {
  const raw = headers.get("retry-after");
  if (!raw) {
    return undefined;
  }
  const seconds = Number(raw);
  if (Number.isFinite(seconds)) {
    return Math.max(0, seconds * 1000);
  }
  const dateMs = Date.parse(raw);
  if (Number.isFinite(dateMs)) {
    return Math.max(0, dateMs - Date.now());
  }
  return undefined;
}

// ─── Client-facing error classification (mirrors to_codex_protocol_error) ─────

/**
 * Classify a thrown error into the protocol's CodexErrorInfo so hosts can branch
 * on the error class instead of matching message text.
 *
 * mirrors CodexErr::to_codex_protocol_error (protocol/src/error.rs:415) fed by
 * the HTTP/SSE mapping in codex-api (api_bridge.rs:33-152, sse/responses.rs:387).
 * Only the arms codex-ts can actually reach are produced; note how much rs funnels
 * into `Other` — `InvalidRequest` (any 400), `Stream` (network failure, dropped
 * stream, unrecognised `response.failed`), `RequestTimeout` and `UnexpectedStatus`
 * (401/403/404…) all hit its `_ => Other` arm, so codex-ts must not invent finer
 * classes for them.
 */
export function codexErrorInfoFor(
  error: unknown,
): import("../../../protocol/src/protocol.js").CodexErrorInfo {
  if (isContextWindowExceededError(error)) {
    return { type: "context_window_exceeded" };
  }
  if (!(error instanceof ResponsesApiError)) {
    // A rejected fetch / dropped stream — rs TransportError::Network|Timeout →
    // CodexErr::Stream|RequestTimeout → Other.
    return { type: "other" };
  }

  // rs matches on the parsed error payload's `code` / `error_type`; codex-ts only
  // has the response body, so probe it for the same identifiers.
  const message = error.message;
  const has = (needle: string): boolean => message.includes(needle);
  if (has("cyber_policy")) {
    return { type: "cyber_policy" };
  }
  if (has("server_is_overloaded") || has("slow_down")) {
    return { type: "server_overloaded" };
  }
  // QuotaExceeded | UsageNotIncluded | UsageLimitReached all collapse into
  // UsageLimitExceeded (error.rs:419).
  if (
    has("insufficient_quota") ||
    has("usage_not_included") ||
    has("usage_limit_reached")
  ) {
    return { type: "usage_limit_exceeded" };
  }

  switch (error.status) {
    // 429 → CodexErr::RetryLimit → ResponseTooManyFailedAttempts.
    case 429:
      return {
        type: "response_too_many_failed_attempts",
        http_status_code: error.status,
      };
    // 500 → CodexErr::InternalServerError.
    case 500:
      return { type: "internal_server_error" };
    default:
      return { type: "other" };
  }
}

/**
 * Abortable sleep. Rejects with an AbortError if the signal fires first, so a
 * mid-backoff interrupt cancels the turn promptly.
 */
export function sleep(ms: number, signal?: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
