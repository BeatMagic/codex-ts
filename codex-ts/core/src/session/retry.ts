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
 * see CodexErr::retry_delay, protocol/src/error.rs).
 */
export class ResponsesApiError extends Error {
  readonly status: number;
  readonly retryAfterMs: number | undefined;
  readonly terminal: boolean;
  /** The provider error code the failure was classified from (HTTP body
   *  `error.code`, `response.error.code`, an `error` frame's code), or a
   *  `response.incomplete`'s reason. */
  readonly code: string | undefined;
  /** Monotonic instant (performance.now()) the server's retry advice runs out.
   *  Captured when the advice arrives, so the time spent before the retry
   *  sleeps counts against it — mirrors RetryAfter's deadline (#47641). */
  readonly retryAt: number | undefined;

  constructor(
    status: number,
    body: string,
    retryAfterMs?: number | undefined,
    terminal = false,
    code?: string | undefined,
  ) {
    super(`Responses API ${status}: ${body}`);
    this.name = "ResponsesApiError";
    this.status = status;
    this.retryAfterMs = retryAfterMs;
    this.terminal = terminal;
    this.code = code;
    this.retryAt =
      retryAfterMs === undefined ? undefined : performance.now() + retryAfterMs;
  }
}

/**
 * Whether an HTTP status is a transient failure worth retrying. Mirrors the
 * retryable set in codex-rs (timeouts / 5xx / unexpected status); 4xx are
 * terminal except 408 (request timeout), 409 (conflict) and 429 (rate limit).
 *
 * Deliberate deviation: 501 is terminal, although rs retries every 5xx. Not
 * Implemented is permanent by HTTP semantics, and the ace-mv cloud remaps an
 * upstream provider's 402 (a persistent billing / config error) to 501
 * precisely so the client fails fast (UPSTREAM_PROVIDER_ERROR_STATUS in
 * cloud responses-proxy/index.ts); retrying re-hits the dead upstream.
 */
export function isRetryableStatus(status: number): boolean {
  return (
    status === 408 ||
    status === 409 ||
    status === 429 ||
    (status >= 500 && status !== 501)
  );
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
 * explicitly terminal ResponsesApiError (see classifyStreamFailure /
 * classifyHttpFailure) is never retried; otherwise ResponsesApiError defers to
 * its status, and anything else (network TypeError, dropped stream, JSON/IO
 * error) is treated as a transient connection failure — mirrors codex-rs
 * CodexErr::is_retryable
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
 * Error codes that make a `response.failed` event (or an `error` frame)
 * terminal. mirrors parse_failed_response (codex-api/src/sse/
 * responses_error.rs) + parse_terminal_provider_error (codex-api/src/error.rs):
 * each maps to a CodexErr whose `retry_delay()` is None, so re-sending the
 * identical request cannot help.
 *   - insufficient_quota / credit_balance_exhausted /
 *     organization_spend_limit_exceeded / project_spend_limit_exceeded
 *     → CodexErr::QuotaExceeded (#45602); organization_usage_limit_exceeded
 *     is in rs's HTTP 429 list (#44492) and treated alike here
 *   - usage_not_included  → CodexErr::UsageNotIncluded
 *   - cyber_policy        → CodexErr::CyberPolicy
 *   - invalid_prompt / bio_policy → CodexErr::InvalidPrompt / BioPolicy
 *   - model_not_found     → CodexErr::ModelNotFound (#56445)
 *   - server_is_overloaded → CodexErr::ServerOverloaded — terminal only while
 *     the server gives no retry advice (see RETRYABLE_WITH_ADVICE_CODES)
 * `slow_down` is NOT here: #45602 made it a retryable rate limit.
 * (context_length_exceeded is terminal too, but is handled separately because
 * it also drives compaction / the self-heal path.)
 */
const TERMINAL_STREAM_FAILURE_CODES = new Set<string>([
  "insufficient_quota",
  "credit_balance_exhausted",
  "organization_spend_limit_exceeded",
  "project_spend_limit_exceeded",
  "organization_usage_limit_exceeded",
  "usage_not_included",
  "cyber_policy",
  "invalid_prompt",
  "bio_policy",
  "model_not_found",
  "server_is_overloaded",
]);

/**
 * Terminal codes that become retryable when the server says when to retry:
 * rs retries ServerOverloaded only via `server_retry_delay()` (#49441), and
 * only from a Retry-After header, never from message text. The ace-mv cloud's
 * keepalive-stall frame (`server_is_overloaded`, no headers) stays terminal.
 */
const RETRYABLE_WITH_ADVICE_CODES = new Set<string>(["server_is_overloaded"]);

/** Quota / usage-limit signals in an HTTP 429 body: `error.code` values and
 *  `error.type` values. mirrors map_api_error's TOO_MANY_REQUESTS arm
 *  (codex-api/src/api_bridge.rs, #44492) — UsageLimitReached, UsageNotIncluded
 *  and QuotaExceeded are all terminal. */
const HTTP_429_QUOTA_CODES = new Set<string>([
  "insufficient_quota",
  "credit_balance_exhausted",
  "organization_spend_limit_exceeded",
  "project_spend_limit_exceeded",
  "organization_usage_limit_exceeded",
]);
const HTTP_429_QUOTA_TYPES = new Set<string>([
  "usage_limit_reached",
  "usage_not_included",
  "insufficient_quota",
]);

/** Rate-limit codes: retryable, and the only ones whose message's
 *  "try again in Xs" counts as retry advice (#45602 added slow_down). */
const RATE_LIMIT_CODES = new Set<string>(["rate_limit_exceeded", "slow_down"]);

/**
 * Longest server-advised delay honored before a retry. rs sleeps until the
 * deadline however far away; here one large Retry-After would park the turn
 * (and the compaction that blocks it) for its whole duration. 60s matches the
 * clamp the ace-mv host's fetch retry (RETRY_AFTER_MAX_MS) and the cloud's
 * same-leg retry already apply, so no layer waits longer than another; with
 * the default budget a turn waits at most maxRetries × 60s.
 */
export const MAX_RETRY_AFTER_MS = 60_000;

/**
 * Appended to the retried request after a `content_filter` incomplete, so the
 * retry is not byte-identical to the blocked one. Text verbatim from
 * CONTENT_FILTER_GUIDANCE (codex-rs/prompts/src/model_messages.rs, #49119).
 */
export const CONTENT_FILTER_GUIDANCE =
  "Your previous response was blocked by a content filter. Do not treat this " +
  "as a transient failure or try to reproduce or work around the blocked " +
  "content through repeated attempts, altered formatting, splitting, " +
  "encoding, tools, subagents, or later wakes. Briefly explain the " +
  "limitation and offer a permitted alternative. Continue unrelated " +
  "authorized work.";

/** A `content_filter` incomplete — retryable, but the retry needs
 *  CONTENT_FILTER_GUIDANCE (rs CodexErrorDetails::ContentFilter). */
export function isContentFilterError(error: unknown): boolean {
  return error instanceof ResponsesApiError && error.code === "content_filter";
}

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
 * mirrors try_parse_retry_delay (codex-api/src/sse/responses_error.rs), which
 * only honors the hint for the rate-limit codes.
 */
export function parseRateLimitRetryAfterMs(
  code: string | undefined,
  message: string | undefined,
): number | undefined {
  if (code === undefined || !RATE_LIMIT_CODES.has(code) || !message) {
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
 *     terminal or retryable CodexErr (everything unrecognised stays
 *     Retryable, like rs). Retry advice comes from a Retry-After in
 *     `response.error.headers` (#50418), else — for rate limits only — from a
 *     "try again in Xs" in the message;
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
      reason,
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

  // A valid header wins over the message hint (#50418).
  const headerRetryAfterMs = retryAfterFromErrorHeaders(error?.["headers"]);
  const retryAfterMs =
    headerRetryAfterMs ?? parseRateLimitRetryAfterMs(code, message);
  const terminal =
    code !== undefined &&
    TERMINAL_STREAM_FAILURE_CODES.has(code) &&
    !(RETRYABLE_WITH_ADVICE_CODES.has(code) && headerRetryAfterMs !== undefined);
  return new ResponsesApiError(503, body, retryAfterMs, terminal, code);
}

/** Retry-After from a streamed error's `headers` object (header names are
 *  case-insensitive; string or number values). mirrors
 *  json_headers_to_http_headers + RetryAfter::from_headers. */
function retryAfterFromErrorHeaders(headers: unknown): number | undefined {
  if (typeof headers !== "object" || headers === null) return undefined;
  for (const [name, value] of Object.entries(headers)) {
    if (name.toLowerCase() !== "retry-after") continue;
    if (typeof value === "string" || typeof value === "number") {
      return parseRetryAfterValue(String(value));
    }
  }
  return undefined;
}

/**
 * Turn a non-2xx Responses reply into a ResponsesApiError with rs's verdict
 * (map_api_error, codex-api/src/api_bridge.rs). Status decides by default
 * (isRetryableStatus); the body's `error.code` / `error.type` refine it:
 *   - 429 carrying a quota or usage-limit signal is terminal (#44492);
 *   - any other 429 is rs's RetryLimit, and a 503 `server_is_overloaded` its
 *     ServerOverloaded: both retry only with server advice, i.e. a
 *     Retry-After header (#49441) — without one, re-sending just presses on
 *     a limiter that gave no sign of when it lifts;
 *   - a 503 `slow_down` stays retryable (#45602: a rate limit, not an
 *     overload).
 * `model_not_found` needs no rule: rs reads it from 400/403/404, all terminal
 * by status already.
 */
export function classifyHttpFailure(
  status: number,
  body: string,
  headers: Headers,
): ResponsesApiError {
  const retryAfterMs = parseRetryAfter(headers);
  let code: string | undefined;
  let type: string | undefined;
  try {
    const error = (JSON.parse(body) as Record<string, unknown> | null)?.[
      "error"
    ] as Record<string, unknown> | undefined;
    if (typeof error?.["code"] === "string") code = error["code"];
    if (typeof error?.["type"] === "string") type = error["type"];
  } catch {
    /* not a JSON error envelope — status alone decides */
  }
  let terminal = false;
  if (status === 429) {
    terminal =
      (code !== undefined && HTTP_429_QUOTA_CODES.has(code)) ||
      (type !== undefined && HTTP_429_QUOTA_TYPES.has(type)) ||
      retryAfterMs === undefined;
  } else if (status === 503 && code === "server_is_overloaded") {
    terminal = retryAfterMs === undefined;
  }
  return new ResponsesApiError(status, body, retryAfterMs, terminal, code);
}

export function isAbortError(error: unknown): boolean {
  return (
    error instanceof DOMException &&
    error.name === "AbortError"
  ) ||
    (error instanceof Error && error.name === "AbortError");
}

/**
 * Delay before the next attempt: what is left of the server's advice when the
 * error carries some — expired advice is zero, not a fallback to backoff
 * (#47641) — capped at MAX_RETRY_AFTER_MS; otherwise exponential backoff.
 */
export function computeRetryDelay(error: unknown, attempt: number): number {
  if (error instanceof ResponsesApiError && error.retryAt !== undefined) {
    const remaining = Math.ceil(error.retryAt - performance.now());
    return Math.min(MAX_RETRY_AFTER_MS, Math.max(0, remaining));
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
  return parseRetryAfterValue(raw);
}

/** A Retry-After value (delta-seconds or HTTP-date) in milliseconds. */
function parseRetryAfterValue(raw: string): number | undefined {
  if (raw.trim() === "") {
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
  // #56445: ModelNotFound from 400/403/404 bodies and streamed failures.
  if (has("model_not_found")) {
    return { type: "model_not_found" };
  }
  if (has("cyber_policy")) {
    return { type: "cyber_policy" };
  }
  if (has("server_is_overloaded")) {
    return { type: "server_overloaded" };
  }
  // QuotaExceeded | UsageNotIncluded | UsageLimitReached all collapse into
  // UsageLimitExceeded (error.rs to_codex_protocol_error); the spend-limit
  // codes are QuotaExceeded since #44492 / #45602.
  if (
    has("insufficient_quota") ||
    has("credit_balance_exhausted") ||
    has("spend_limit_exceeded") ||
    has("organization_usage_limit_exceeded") ||
    has("usage_not_included") ||
    has("usage_limit_reached")
  ) {
    return { type: "usage_limit_exceeded" };
  }
  // A streamed rate limit, or an HTTP 503 `slow_down`, is
  // CodexErr::RateLimitExceeded (#45602); an HTTP 429 stays RetryLimit below.
  if (error.status !== 429 && (has("rate_limit_exceeded") || has("slow_down"))) {
    return { type: "rate_limit_exceeded" };
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
