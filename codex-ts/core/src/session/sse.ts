/**
 * SSE stream parser for the OpenAI Responses API.
 * Extracted from turn.ts so compact.ts can share it.
 *
 * mirrors the `eventsource_stream` decoding codex-rs runs the Responses body
 * through (codex-api/src/sse/responses.rs process_sse), including its per-poll
 * idle timeout.
 */

import { ResponsesApiError } from "./retry.js";

export type RawSseEvent = Record<string, unknown>;

/** mirrors DEFAULT_STREAM_IDLE_TIMEOUT_MS (model-provider-info/src/lib.rs:26). */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000;

export interface ParseSseStreamOptions {
  /**
   * Longest wait for the next chunk of the body before the stream is given up
   * on with a retryable "idle timeout waiting for SSE" (mirrors rs wrapping
   * every `stream.next()` in `timeout(idle_timeout, …)`). Omitted or
   * non-positive: wait forever.
   */
  idleTimeoutMs?: number | undefined;
}

/**
 * Yields each event's parsed `data` JSON, following the SSE spec's framing:
 * LF / CRLF / lone-CR line endings, `data:` with or without the one optional
 * space, multi-line data joined with "\n", comments and other fields ignored,
 * and a final event dispatched at EOF even without its blank line. All framing
 * state lives across chunks — an earlier per-chunk version silently dropped any
 * event whose `data:` line and blank line arrived in different network reads,
 * `response.completed` included. `[DONE]` and malformed JSON are skipped.
 *
 * Stopping early (the consumer breaks or throws) CANCELS the body rather than
 * just releasing the lock, so the connection is closed — mirrors rs returning
 * from process_sse, which drops the byte stream.
 */
export async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
  options?: ParseSseStreamOptions,
): AsyncGenerator<RawSseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const idleTimeoutMs = options?.idleTimeoutMs ?? 0;
  // Unterminated tail of the current line, carried to the next chunk.
  let partialLine = "";
  // A chunk ended on "\r": a "\n" opening the next chunk belongs to that CRLF.
  let skipLeadingLf = false;
  let dataLines: string[] = [];
  let drained = false;

  const splitLines = (text: string): string[] => {
    const lines: string[] = [];
    let start = 0;
    if (skipLeadingLf && text.length > 0) {
      if (text[0] === "\n") start = 1;
      skipLeadingLf = false;
    }
    for (let i = start; i < text.length; i += 1) {
      const ch = text[i];
      if (ch !== "\n" && ch !== "\r") continue;
      lines.push(partialLine + text.slice(start, i));
      partialLine = "";
      if (ch === "\r") {
        if (i + 1 < text.length) {
          if (text[i + 1] === "\n") i += 1;
        } else {
          skipLeadingLf = true;
        }
      }
      start = i + 1;
    }
    partialLine += text.slice(start);
    return lines;
  };

  // Returns the event to dispatch when `line` is the blank line ending one.
  const takeLine = (line: string): RawSseEvent | undefined => {
    if (line === "") {
      if (dataLines.length === 0) return undefined;
      const data = dataLines.join("\n");
      dataLines = [];
      if (data.trim() === "[DONE]") return undefined;
      try {
        return JSON.parse(data) as RawSseEvent;
      } catch {
        /* malformed JSON — skip */
        return undefined;
      }
    }
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    // event: / id: / retry: are unused; a comment (keepalive) is a line with
    // an empty field name.
    if (field !== "data") return undefined;
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    dataLines.push(value);
    return undefined;
  };

  try {
    for (;;) {
      const { done, value } = await readChunk(reader, idleTimeoutMs);
      if (done) break;
      for (const line of splitLines(decoder.decode(value, { stream: true }))) {
        const event = takeLine(line);
        if (event !== undefined) yield event;
      }
    }
    drained = true;
    // EOF: flush the decoder, finish the last line, dispatch what is pending.
    const tail = splitLines(decoder.decode());
    if (partialLine !== "") tail.push(partialLine);
    tail.push("");
    for (const line of tail) {
      const event = takeLine(line);
      if (event !== undefined) yield event;
    }
  } finally {
    if (!drained) {
      // Early exit (consumer stopped, idle timeout, read error): close the
      // connection. cancel() reaches the source and settles any pending read
      // synchronously; its promise is not awaited so a source whose cancel
      // never settles cannot hang the turn. A cancel on an already-errored
      // stream rejects — ignore.
      reader.cancel().catch(() => undefined);
    }
    reader.releaseLock();
  }
}

/** The text of a completed `message` output item: its `output_text` parts,
 *  concatenated. */
export function messageOutputText(item: Record<string, unknown>): string {
  const content = item["content"];
  if (!Array.isArray(content)) return "";
  let text = "";
  for (const part of content as unknown[]) {
    if (typeof part !== "object" || part === null) continue;
    const record = part as Record<string, unknown>;
    if (record["type"] === "output_text" && typeof record["text"] === "string") {
      text += record["text"];
    }
  }
  return text;
}

/** reader.read(), failing with a retryable stream error after `idleTimeoutMs`
 *  without a chunk. The timer is cleared on every outcome. */
async function readChunk(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  idleTimeoutMs: number,
): Promise<ReadableStreamReadResult<Uint8Array>> {
  if (!(idleTimeoutMs > 0)) return reader.read();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const idle = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () => reject(new ResponsesApiError(503, "idle timeout waiting for SSE")),
      idleTimeoutMs,
    );
  });
  try {
    return await Promise.race([reader.read(), idle]);
  } finally {
    clearTimeout(timer);
  }
}
