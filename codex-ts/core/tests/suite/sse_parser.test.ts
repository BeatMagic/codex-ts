/**
 * SSE framing for the Responses stream (session/sse.ts), mirroring the
 * `eventsource_stream` decoding codex-rs runs the body through: state persists
 * across network chunks, every line-ending flavour, multi-line data, comments
 * and unused fields, an unterminated final event, and closing the connection
 * (cancel, not just releaseLock) when the consumer stops early.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { parseSseStream } from "../../src/session/sse.js";
import { ResponsesApiError } from "../../src/session/retry.js";

function streamOf(chunks: (string | Uint8Array)[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(
          typeof chunk === "string" ? encoder.encode(chunk) : chunk,
        );
      }
      controller.close();
    },
  });
}

/** Every byte of `text` as its own chunk. */
function byteByByte(text: string): Uint8Array[] {
  return Array.from(new TextEncoder().encode(text), (b) => Uint8Array.of(b));
}

async function collect(
  body: ReadableStream<Uint8Array>,
): Promise<Record<string, unknown>[]> {
  const out: Record<string, unknown>[] = [];
  for await (const event of parseSseStream(body)) out.push(event);
  return out;
}

const A = { type: "a", text: "first" };
const B = { type: "b", text: "second" };
const frame = (e: object) => `data: ${JSON.stringify(e)}\n\n`;

describe("parseSseStream framing", () => {
  it("parses LF-framed events", async () => {
    expect(await collect(streamOf([frame(A) + frame(B)]))).toEqual([A, B]);
  });

  // The original bug: `data:` and its blank line in different reads. This is
  // how a `response.completed` could vanish.
  it("keeps an event whose data line and blank line arrive in different chunks", async () => {
    const events = await collect(
      streamOf([`data: ${JSON.stringify(A)}\n`, "\n", frame(B)]),
    );
    expect(events).toEqual([A, B]);
  });

  it("keeps an event whose data line itself is split across chunks", async () => {
    const line = `data: ${JSON.stringify(A)}`;
    const events = await collect(
      streamOf([line.slice(0, 9), line.slice(9), "\n\n"]),
    );
    expect(events).toEqual([A]);
  });

  it("accepts CRLF line endings", async () => {
    const text = `data: ${JSON.stringify(A)}\r\n\r\ndata: ${JSON.stringify(B)}\r\n\r\n`;
    expect(await collect(streamOf([text]))).toEqual([A, B]);
  });

  it("accepts lone CR line endings", async () => {
    const text = `data: ${JSON.stringify(A)}\r\rdata: ${JSON.stringify(B)}\r\r`;
    expect(await collect(streamOf([text]))).toEqual([A, B]);
  });

  // A CR at the end of one chunk and the LF opening the next are ONE line
  // ending, not two (two would end the event early and split its data).
  it("treats a CRLF split across chunks as a single line ending", async () => {
    const events = await collect(
      streamOf(['data: {"type":"a",\r', '\ndata: "text":"first"}\r\n\r\n']),
    );
    expect(events).toEqual([A]);
  });

  it("accepts `data:` without the optional space", async () => {
    expect(await collect(streamOf([`data:${JSON.stringify(A)}\n\n`]))).toEqual([
      A,
    ]);
  });

  it("joins multiple data lines with a newline", async () => {
    const events = await collect(
      streamOf(['data: {"type":"a",\ndata: "text":"first"}\n\n']),
    );
    expect(events).toEqual([A]);
  });

  it("ignores comments and the event/id/retry fields", async () => {
    const text =
      ": keepalive\n" +
      "event: response.output_text.delta\n" +
      "id: 7\n" +
      "retry: 1000\n" +
      `data: ${JSON.stringify(A)}\n` +
      ": trailing comment\n" +
      "\n" +
      ": OPENROUTER PROCESSING\n\n";
    expect(await collect(streamOf([text]))).toEqual([A]);
  });

  it("dispatches a final event that has no trailing blank line", async () => {
    expect(
      await collect(streamOf([frame(A) + `data: ${JSON.stringify(B)}`])),
    ).toEqual([A, B]);
    expect(
      await collect(streamOf([frame(A) + `data: ${JSON.stringify(B)}\n`])),
    ).toEqual([A, B]);
  });

  it("skips [DONE] and malformed JSON, keeping later events", async () => {
    const text = "data: [DONE]\n\n" + "data: {not json\n\n" + frame(A);
    expect(await collect(streamOf([text]))).toEqual([A]);
  });

  it("parses a stream delivered one byte per chunk, multi-byte text intact", async () => {
    const C = { type: "c", text: "多字节 ✓ 😀" };
    const text =
      frame(A) + `data: ${JSON.stringify(C)}\r\n\r\n` + ": ping\r" + frame(B);
    expect(await collect(streamOf(byteByByte(text)))).toEqual([A, C, B]);
  });

  it("decodes a multi-byte UTF-8 character split across chunks", async () => {
    const C = { type: "c", text: "€😀" };
    const bytes = new TextEncoder().encode(frame(C));
    const euro = new TextEncoder().encode(`data: {"type":"c","text":"`).length;
    // Split inside the 3-byte "€", then inside the 4-byte emoji.
    const events = await collect(
      streamOf([
        bytes.slice(0, euro + 1),
        bytes.slice(euro + 1, euro + 5),
        bytes.slice(euro + 5),
      ]),
    );
    expect(events).toEqual([C]);
  });
});

describe("parseSseStream early exit", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  // Releasing the lock alone leaves the connection open; the body must be
  // cancelled so the transport is torn down (rs drops the byte stream).
  it("cancels the body when the consumer stops early", async () => {
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(frame(A) + frame(B)));
        // No close: a held-open connection.
      },
      cancel,
    });

    const seen: unknown[] = [];
    for await (const event of parseSseStream(body)) {
      seen.push(event);
      break;
    }

    expect(seen).toEqual([A]);
    expect(cancel).toHaveBeenCalledTimes(1);
  });

  it("cancels the body when the consumer throws", async () => {
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(frame(A)));
      },
      cancel,
    });
    await expect(
      (async () => {
        for await (const _event of parseSseStream(body)) {
          throw new Error("consumer failed");
        }
      })(),
    ).rejects.toThrow("consumer failed");
    expect(cancel).toHaveBeenCalledTimes(1);
  });

  it("does not cancel a body it read to EOF", async () => {
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(frame(A)));
        controller.close();
      },
      cancel,
    });
    expect(await collect(body)).toEqual([A]);
    expect(cancel).not.toHaveBeenCalled();
  });

  it("fails with a retryable idle timeout when no chunk arrives in time, and cancels", async () => {
    vi.useFakeTimers();
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(frame(A)));
        // …then silence: the connection stays open.
      },
      cancel,
    });

    const seen: unknown[] = [];
    const run = (async () => {
      for await (const event of parseSseStream(body, { idleTimeoutMs: 1000 })) {
        seen.push(event);
      }
    })();
    const outcome = run.then(
      () => "resolved",
      (error: unknown) => error,
    );

    await vi.advanceTimersByTimeAsync(999);
    expect(seen).toEqual([A]);
    expect(cancel).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1);
    const error = await outcome;
    expect(error).toBeInstanceOf(ResponsesApiError);
    expect((error as ResponsesApiError).message).toContain(
      "idle timeout waiting for SSE",
    );
    expect((error as ResponsesApiError).terminal).toBe(false);
    expect(cancel).toHaveBeenCalledTimes(1);
    // The timer is per read, and cleared on every outcome.
    expect(vi.getTimerCount()).toBe(0);
  });

  it("restarts the idle window on every chunk", async () => {
    vi.useFakeTimers();
    let controllerRef!: ReadableStreamDefaultController<Uint8Array>;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controllerRef = controller;
      },
    });
    const seen: unknown[] = [];
    const run = (async () => {
      for await (const event of parseSseStream(body, { idleTimeoutMs: 1000 })) {
        seen.push(event);
      }
    })();

    // Three chunks 800ms apart: 2400ms total, never 1000ms of silence.
    for (const event of [A, B, A]) {
      await vi.advanceTimersByTimeAsync(800);
      controllerRef.enqueue(new TextEncoder().encode(frame(event)));
    }
    controllerRef.close();
    await run;
    expect(seen).toEqual([A, B, A]);
    // One timer per read, each cleared once its read settles — a leak here
    // would leave a pending timer behind for every chunk of a long stream.
    expect(vi.getTimerCount()).toBe(0);
  });

  it("never times out when the idle timeout is disabled", async () => {
    vi.useFakeTimers();
    const body = new ReadableStream<Uint8Array>({ start() {} });
    let settled = false;
    void (async () => {
      for await (const _event of parseSseStream(body, { idleTimeoutMs: 0 })) {
        /* none */
      }
    })().finally(() => {
      settled = true;
    });
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(settled).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });
});
