//! Text: line breaking, alignment, and emission of cached glyph triangles.
//!
//! One walk - [`layoutLines`] - backs both measuring and drawing, so a
//! measured height is exactly the drawn height (the classic way UIs drift is
//! two implementations of "how tall is this string").
//!
//! Wrapping is word-based with a hard character break for words that do not
//! fit an empty line. Trailing spaces do not count toward a line's width
//! (they would push right-aligned text off its edge), leading spaces of a
//! wrapped line are dropped. No hyphenation - a wrapped line ending a
//! paragraph early is honest; a hyphen inserted by guesswork is not.

const std = @import("std");
const geom = @import("geom.zig");
const Rect = geom.Rect;
const Vec2 = geom.Vec2;
const font_mod = @import("font.zig");
const Font = font_mod.Font;
const tess = @import("tess.zig");
const DisplayList = tess.DisplayList;

pub const Align = enum { left, center, right };

pub const Options = struct {
    /// Wrap width in logical pixels; 0 (or negative) = never wrap.
    max_width: f32 = 0.0,
    alignment: Align = .left,
    /// Multiplier over the font's natural line height (1.0 = exactly it).
    line_height: f32 = 1.0,
};

pub const Line = struct {
    start: usize, // byte offset into the source text
    end: usize, // exclusive
    width: f32, // advance width, trailing spaces excluded
};

pub const Measured = struct {
    width: f32,
    height: f32,
    lines: usize,
};

/// Plenty for a log panel's worth of lines; layout never allocates.
pub const max_lines = 256;

/// Breaks `text` into lines, filling `out` with up to [`max_lines`] entries.
/// Returns the number of lines written (0 for empty text).
pub fn layoutLines(
    font: *const Font,
    text: []const u8,
    px: f32,
    opts: Options,
    out: *[max_lines]Line,
) usize {
    const max_w = if (opts.max_width > 0.0) opts.max_width else std.math.floatMax(f32);
    const space_adv = font.advanceOfCp(' ', px);

    var n: usize = 0;
    var ls: usize = 0; // line start
    var le: usize = 0; // line end (last committed word's end)
    var line_w: f32 = 0;
    var pend: f32 = 0; // pending space width (excluded until a word lands)
    var ws: usize = 0; // current word start
    var we: usize = 0; // current word end
    var ww: f32 = 0; // current word width
    var wactive = false;
    var dirty = false; // something has been seen since the last record

    const record = struct {
        fn go(o: *[max_lines]Line, out_n: *usize, a: usize, b: usize, w: f32) void {
            if (out_n.* < o.len and b > a) {
                o[out_n.*] = .{ .start = a, .end = b, .width = w };
                out_n.* += 1;
            }
        }
    }.go;

    var it: usize = 0;
    while (it < text.len) {
        const cp, const clen = nextCp(text, it);
        const b = it;
        it += clen;

        if (cp == '\n') {
            if (wactive) {
                line_w += ww;
                le = we;
                wactive = false;
            }
            if (dirty) record(out, &n, ls, le, line_w);
            dirty = false;
            ls = it;
            le = it;
            line_w = 0;
            pend = 0;
            continue;
        }
        dirty = true;
        if (cp == ' ') {
            if (wactive) {
                line_w += ww;
                le = we;
                wactive = false;
            }
            pend += space_adv;
            continue;
        }

        const adv = font.advanceOfCp(cp, px);
        if (!wactive) {
            ws = b;
            ww = 0;
            wactive = true;
        }
        ww += adv;
        we = it;

        // Is the word finished? Look ahead: a word ends at EOF, space or \n.
        const nxt: ?u32 = if (it < text.len) nextCp(text, it)[0] else null;
        const word_done = (nxt == null) or (nxt.? == ' ') or (nxt.? == '\n');
        if (!word_done) continue;

        // ---- place the completed word ---------------------------------
        if (line_w + pend + ww <= max_w) {
            line_w += pend + ww;
            le = we;
            pend = 0;
        } else if (line_w > 0.0) {
            // Wrap: current line ends, word starts the next.
            record(out, &n, ls, le, line_w);
            ls = ws;
            line_w = ww;
            le = we;
            pend = 0;
        } else {
            // A single word wider than the measure: break by characters.
            var chunk_start = ws;
            var chunk_w: f32 = 0;
            var cw: usize = ws;
            while (cw < we) {
                const ccp, const cclen = nextCp(text, cw);
                const cadv = font.advanceOfCp(ccp, px);
                if (chunk_w > 0.0 and chunk_w + cadv > max_w) {
                    record(out, &n, ls, chunk_start, chunk_w);
                    ls = chunk_start;
                    chunk_w = 0;
                }
                chunk_w += cadv;
                cw += cclen;
                chunk_start = cw;
            }
            line_w = chunk_w;
            le = we;
            pend = 0;
        }
        wactive = false;
    }
    if (wactive) {
        line_w += ww;
        le = we;
    }
    if (dirty) record(out, &n, ls, le, line_w);
    return n;
}

/// Non-allocating measurement: width of the widest line, total height.
pub fn measure(font: *const Font, text: []const u8, px: f32, opts: Options) Measured {
    var buf: [max_lines]Line = undefined;
    const n = layoutLines(font, text, px, opts, &buf);
    var w: f32 = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) w = @max(w, buf[i].width);
    const line_h = font.lineHeight(px) * opts.line_height;
    return .{ .width = w, .height = @as(f32, @floatFromInt(n)) * line_h, .lines = n };
}

/// Draws `text` with its first line's top at `rect.y`, returns the height
/// used. Overflowing lines past `rect.h` are *not* clipped by this function -
/// pass a `tess.DisplayList` clip (as `ui.zig` does) if the box must hold.
pub fn draw(
    dl: *DisplayList,
    gpa: std.mem.Allocator,
    font: *Font,
    text: []const u8,
    rect: Rect,
    px: f32,
    color: geom.Color,
    opts: Options,
) !f32 {
    var buf: [max_lines]Line = undefined;
    const n = layoutLines(font, text, px, opts, &buf);
    if (n == 0) return 0.0;
    const line_h = font.lineHeight(px) * opts.line_height;
    const baseline_off = font.ascentPx(px);
    const s = font.scale(px);

    var li: usize = 0;
    while (li < n) : (li += 1) {
        const ln = buf[li];
        const offset: f32 = switch (opts.alignment) {
            .left => 0.0,
            .center => @max(0.0, (rect.w - ln.width) * 0.5),
            .right => @max(0.0, rect.w - ln.width),
        };
        var pen = rect.x + offset;
        const baseline = rect.y + @as(f32, @floatFromInt(li)) * line_h + baseline_off;
        var ci = ln.start;
        while (ci < ln.end) {
            const cp, const clen = nextCp(text, ci);
            ci += clen;
            const gid = font.glyphIndex(cp);
            if (gid == 0 and cp != 0) continue; // unmapped: no tofu box
            const adv_px = font.advance(gid) * s;
            if (cp != ' ') {
                const g = try font.glyph(gid);
                var ti: usize = 0;
                while (ti + 2 < g.tris.len) : (ti += 3) {
                    const a = toPx(g.points[g.tris[ti]], pen, baseline, s);
                    const b = toPx(g.points[g.tris[ti + 1]], pen, baseline, s);
                    const c = toPx(g.points[g.tris[ti + 2]], pen, baseline, s);
                    try dl.tri(gpa, .{ a, b, c }, .{ color, color, color });
                }
            }
            pen += adv_px;
        }
    }
    return @as(f32, @floatFromInt(n)) * line_h;
}

/// Font units (y-up, baseline origin) -> logical pixels (y-down).
fn toPx(p: Vec2, pen: f32, baseline: f32, s: f32) Vec2 {
    return .{ .x = pen + p.x * s, .y = baseline - p.y * s };
}

/// One codepoint plus its byte length, starting at `i`.
/// Returned as a tuple so it destructures: `const cp, const len = nextCp(...)`.
/// Invalid UTF-8 yields U+FFFD with a one-byte step (never loops).
pub fn nextCp(text: []const u8, i: usize) struct { u32, usize } {
    const b0 = text[i];
    if (b0 < 0x80) return .{ b0, 1 };
    const len: usize = if (b0 & 0xE0 == 0xC0) 2 else if (b0 & 0xF0 == 0xE0) 3 else if (b0 & 0xF8 == 0xF0) 4 else 1;
    if (len == 1 or i + len > text.len) return .{ 0xFFFD, 1 };
    var cp: u32 = switch (len) {
        2 => b0 & 0x1F,
        3 => b0 & 0x0F,
        else => b0 & 0x07,
    };
    var k: usize = 1;
    while (k < len) : (k += 1) {
        const b = text[i + k];
        if (b & 0xC0 != 0x80) return .{ 0xFFFD, 1 };
        cp = (cp << 6) | (b & 0x3F);
    }
    return .{ cp, len };
}

// ---------------------------------------------------------------------------
// tests

const testing_font = struct {
    var cached: ?Font = null;

    fn get(gpa: std.mem.Allocator) !*Font {
        if (cached == null) cached = try @import("font.zig").loadBundled(gpa, "Inter-Variable.ttf");
        return &cached.?;
    }

    fn deinit() void {
        if (cached) |*f| {
            f.deinit();
            cached = null;
        }
    }
};

test "measure grows with content and empty text is zero" {
    const gpa = std.testing.allocator;
    const f = testing_font.get(gpa) catch return error.SkipZigTest;
    defer testing_font.deinit();
    const hello = measure(f, "Hello", 16.0, .{});
    const hell = measure(f, "Hell", 16.0, .{});
    const empty = measure(f, "", 16.0, .{});
    try std.testing.expect(hello.width > hell.width);
    try std.testing.expect(empty.lines == 0);
    try std.testing.expect(empty.height == 0);
    try std.testing.expect(hello.lines == 1);
    try std.testing.expect(hello.height >= 16.0);
}

test "wrapping breaks on words and never exceeds the measure" {
    const gpa = std.testing.allocator;
    const f = testing_font.get(gpa) catch return error.SkipZigTest;
    defer testing_font.deinit();

    const text = "The quick brown fox jumps over the lazy dog";
    const full = measure(f, text, 16.0, .{});
    try std.testing.expect(full.lines == 1);

    const half = measure(f, text, 16.0, .{ .max_width = full.width * 0.5 });
    try std.testing.expect(half.lines >= 2);
    try std.testing.expect(half.width <= full.width * 0.5 + 1.0);

    // Forced break on newline.
    const nl = measure(f, "one\ntwo\nthree", 16.0, .{});
    try std.testing.expectEqual(@as(usize, 3), nl.lines);

    // A word wider than the measure hard-breaks rather than overflowing.
    const long = measure(f, "Supercalifragilisticexpialidocious", 16.0, .{ .max_width = 60.0 });
    try std.testing.expect(long.lines > 1);
    try std.testing.expect(long.width <= 60.0 + 1.0);
}

test "alignment shifts without changing measured width" {
    const gpa = std.testing.allocator;
    const f = testing_font.get(gpa) catch return error.SkipZigTest;
    defer testing_font.deinit();
    const m = measure(f, "Right", 16.0, .{});
    var dl = tess.DisplayList.init();
    defer dl.deinit(gpa);
    const rect = Rect{ .x = 0, .y = 0, .w = m.width + 40, .h = m.height + 4 };
    _ = try draw(&dl, gpa, f, "Right", rect, 16.0, .{}, .{ .alignment = .right });
    var maxx: f32 = -1e9;
    for (dl.vertices.items) |v| maxx = @max(maxx, v.position[0]);
    // Right-aligned text lands at (or a hair short of) the right edge.
    try std.testing.expectApproxEqAbs(rect.right(), maxx, 2.0);
}

test "draw emits triangles inside a pushed clip" {
    const gpa = std.testing.allocator;
    const f = testing_font.get(gpa) catch return error.SkipZigTest;
    defer testing_font.deinit();
    var dl = tess.DisplayList.init();
    defer dl.deinit(gpa);
    const rect = Rect{ .x = 10, .y = 20, .w = 80, .h = 30 };
    try dl.pushClip(gpa, rect);
    const used = try draw(&dl, gpa, f, "Wrapped text goes here", rect, 14.0, .{}, .{ .max_width = rect.w });
    dl.popClip();
    try std.testing.expect(used > 0);
    try std.testing.expect(dl.vertices.items.len > 0);
    for (dl.vertices.items) |v| {
        try std.testing.expect(v.position[0] >= rect.x - 1e-3);
        try std.testing.expect(v.position[0] <= rect.right() + 1e-3);
        try std.testing.expect(v.position[1] >= rect.y - 1e-3);
        try std.testing.expect(v.position[1] <= rect.bottom() + 1e-3);
    }
}
