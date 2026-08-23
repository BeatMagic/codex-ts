/**
 * Tool-argument repair: recover the malformed-but-unambiguous shapes, refuse the
 * ambiguous one, and always hand the model something it can act on.
 *
 * Fixtures are the shapes actually observed in production (see
 * core/src/tools/argument_repair.ts for the census): a literal newline inside a
 * JSON string, a literal `"` inside one, and an XML `<parameter name=…>` block
 * spliced after the closing brace.
 */

import { describe, expect, it, vi } from "vitest";

import {
  argumentFaultMessage,
  canonicalizeStoredArguments,
  parseToolArguments,
  scanToolArguments,
} from "../../src/tools/argument_repair.js";

describe("parseToolArguments — well-formed input is untouched", () => {
  it("returns the strict parse, and never rewrites valid arguments", () => {
    const raw = JSON.stringify({ path: "a/b.md", text: 'he said "hi"\nbye' });
    expect(parseToolArguments(raw, "doc.create")).toEqual({
      path: "a/b.md",
      text: 'he said "hi"\nbye',
    });
  });

  it("accepts a string that merely LOOKS suspicious (escaped quotes/newlines)", () => {
    const raw = '{"text":"line1\\nline2 \\"quoted\\" end"}';
    expect(parseToolArguments(raw, "t")).toEqual({
      text: 'line1\nline2 "quoted" end',
    });
  });
});

describe("parseToolArguments — repairs the two unambiguous shapes", () => {
  it("escapes a literal line break inside a string", () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    // The most common shape: the model wrote real newlines into markdown.
    const raw = '{"path":"a.md","text":"# Title\nbody line\nmore"}';
    expect(() => JSON.parse(raw)).toThrow(); // precondition: genuinely invalid
    expect(parseToolArguments(raw, "doc.create")).toEqual({
      path: "a.md",
      text: "# Title\nbody line\nmore",
    });
  });

  it("escapes a literal double quote inside a string", () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const raw = '{"text":"the "arrival" arc"}';
    expect(() => JSON.parse(raw)).toThrow();
    expect(parseToolArguments(raw, "doc.create")).toEqual({
      text: 'the "arrival" arc',
    });
  });

  it("keeps a closing quote that really is the terminator", () => {
    // The quote before `,` / `}` must stay a terminator, or the repair would
    // swallow the rest of the object into one string.
    const { repaired } = scanToolArguments('{"a":"x","b":"y"}');
    expect(repaired).toBe('{"a":"x","b":"y"}');
  });
});

describe("parseToolArguments — refuses what it cannot know", () => {
  it("does NOT invent arguments for a spliced XML parameter block", () => {
    const raw = '{"path":"a.md"}<parameter name="text">hello</parameter>';
    expect(() => parseToolArguments(raw, "doc.create")).toThrow(
      /this tool interface is JSON, never XML/,
    );
  });

  it("says INCOMPLETE (not 'escape better') when nothing unescaped explains it", () => {
    // An unclosed brace: sending the model hunting for a bad character would
    // waste a round trip on a fault that is not there.
    expect(() => parseToolArguments('{"a":1', "t")).toThrow(/INCOMPLETE/);
  });

  it("names the empty payload for what it is", () => {
    expect(() => parseToolArguments("   ", "t")).toThrow(/arrived EMPTY/);
  });
});

describe("argumentFaultMessage — the model must be able to act on it", () => {
  it("gives the offset, the character class, and the rule", () => {
    const raw = '{"text":"a\nb"}';
    const { fault } = scanToolArguments(raw);
    const message = argumentFaultMessage(raw, fault, new Error("boom"));
    expect(message).toContain("line break");
    expect(message).toContain(`at character ${fault?.index}`);
    expect(message).toContain("re-send");
    // Enough surrounding text to locate it without re-authoring the payload.
    expect(message).toContain("Around it:");
  });
});

describe("canonicalizeStoredArguments — replay hygiene", () => {
  it("rewrites a repairable stored call so a replay converter can parse it", () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const raw = '{"text":"a\nb"}';
    const canonical = canonicalizeStoredArguments(raw, "doc.create");
    expect(() => JSON.parse(canonical)).not.toThrow();
    expect(JSON.parse(canonical)).toEqual({ text: "a\nb" });
  });

  it("leaves already-valid arguments byte-identical", () => {
    const raw = '{"a":1,"b":"x"}';
    expect(canonicalizeStoredArguments(raw, "t")).toBe(raw);
  });

  it("leaves an unrepairable string alone rather than guessing", () => {
    const raw = '{"a":1';
    expect(canonicalizeStoredArguments(raw, "t")).toBe(raw);
  });

  it("leaves an empty string alone", () => {
    expect(canonicalizeStoredArguments("", "t")).toBe("");
  });
});
