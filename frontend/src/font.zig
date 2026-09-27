//! A pragmatic TrueType reader: the tables text rendering actually needs
//! (`head hhea maxp hmtx loca glyf cmap`), simple *and* composite glyphs,
//! format 4 and format 12 character maps.
//!
//! What it deliberately does not do, and why:
//!
//! * **No shaping** (GSUB/GPOS): one codepoint in, one glyph out. Latin UI
//!   copy - which is what this frontend renders - is legible without
//!   ligatures; script shaping is a different project, not a hidden part of
//!   this one.
//! * **No kerning pairs**: Inter and Outfit carry theirs in GPOS, same
//!   argument. Advances come from `hmtx`, so text is metrically correct even
//!   where it is not kerned.
//! * **No `gvar`**: these are variable fonts rendered at their *default*
//!   instance - the outlines in `glyf` are real default-instance outlines,
//!   not deltas. Weight 400 for both families; hierarchy comes from size,
//!   colour and family instead of weight axes.
//!
//! Glyph outlines are flattened and triangulated **once** into the glyph
//! cache, in font units, y-up. Emitting a frame then costs one affine
//! transform per triangle corner - no bezier math on the hot path.

const std = @import("std");
const geom = @import("geom.zig");
const Vec2 = geom.Vec2;
const polygon = @import("polygon.zig");

pub const ParseError = error{
    InvalidFont,
    TruncatedFont,
    UnsupportedFont,
    OutOfMemory,
    FontFileNotFound,
};

/// One glyph, flattened and triangulated in font units (y-up).
/// Returned *by value*: the slices point at buffers owned by the cache, and
/// cache inserts never touch those buffers, so a returned value stays valid
/// across later `glyph()` calls.
pub const Glyph = struct {
    points: []Vec2,
    contours: []const polygon.Contour,
    tris: []const u32,
};

/// A flat on/off-curve point, the glyf table's native shape.
const RawPoint = struct {
    p: Vec2,
    on: bool,
};

pub const Font = struct {
    gpa: std.mem.Allocator,
    /// Owned copy: a host may hand us bytes it frees next call.
    bytes: []const u8,
    glyf_off: u32,
    loca_off: u32,
    hmtx_off: u32,
    units_per_em: f32,
    ascent: f32, // font units, positive
    descent: f32, // font units, negative
    line_gap: f32,
    num_glyphs: u32,
    num_hmetrics: u32,
    loca_long: bool,
    cmap12: ?u32 = null,
    cmap4: ?u32 = null,
    cache: std.AutoHashMap(u32, Glyph),

    pub fn init(gpa: std.mem.Allocator, data: []const u8) ParseError!Font {
        const bytes = try gpa.dupe(u8, data);
        errdefer gpa.free(bytes);

        if (bytes.len < 12) return error.TruncatedFont;
        const version = try ru32(bytes, 0);
        if (version == 0x74746366) return error.UnsupportedFont; // 'ttcf'
        if (version == 0x4F54544F) return error.UnsupportedFont; // 'OTTO' (CFF)
        const num_tables = try ru16(bytes, 4);

        // ---- table directory ------------------------------------------
        var head: ?u32 = null;
        var hhea: ?u32 = null;
        var maxp: ?u32 = null;
        var hmtx: ?u32 = null;
        var loca: ?u32 = null;
        var glyf: ?u32 = null;
        var cmap: ?u32 = null;
        var i: usize = 0;
        while (i < num_tables) : (i += 1) {
            const rec: usize = 12 + i * 16;
            if (rec + 16 > bytes.len) return error.TruncatedFont;
            const tag = bytes[rec .. rec + 4];
            const off = try ru32(bytes, rec + 8);
            if (eq4(tag, "head")) head = off else if (eq4(tag, "hhea")) hhea = off else if (eq4(tag, "maxp")) maxp = off else if (eq4(tag, "hmtx")) hmtx = off else if (eq4(tag, "loca")) loca = off else if (eq4(tag, "glyf")) glyf = off else if (eq4(tag, "cmap")) cmap = off;
        }
        if (head == null or hhea == null or maxp == null or hmtx == null or loca == null or cmap == null) return error.InvalidFont;
        if (glyf == null) return error.UnsupportedFont; // CFF/OTF: no outlines we can read

        // ---- cmap subtable choice: 12 (full Unicode) beats 4 ------------
        var best4: ?u32 = null;
        var best12: ?u32 = null;
        {
            const base = cmap.?;
            const nsub = try ru16(bytes, base + 2);
            var k: usize = 0;
            while (k < nsub) : (k += 1) {
                const rec: usize = base + 4 + k * 8;
                if (rec + 8 > bytes.len) break;
                const sub = base + (try ru32(bytes, rec + 4));
                if (sub + 2 > bytes.len) continue;
                const fmt = try ru16(bytes, sub);
                if (fmt == 12 and best12 == null) best12 = sub;
                if (fmt == 4 and best4 == null) best4 = sub;
            }
        }
        if (best12 == null and best4 == null) return error.UnsupportedFont;

        // ---- header metrics (all fallible reads happen before Font
        // exists, so the only thing an error can leak is `bytes`) ---------
        const upem = try ru16(bytes, head.? + 18);
        if (upem == 0) return error.InvalidFont;
        const loca_fmt = try ri16(bytes, head.? + 50);
        const ascent = try ri16(bytes, hhea.? + 4);
        const descent = try ri16(bytes, hhea.? + 6);
        const line_gap = try ri16(bytes, hhea.? + 8);
        const num_hmetrics = try ru16(bytes, hhea.? + 34);
        const num_glyphs = try ru16(bytes, maxp.? + 4);
        if (num_glyphs == 0 or num_hmetrics == 0) return error.InvalidFont;

        return Font{
            .gpa = gpa,
            .bytes = bytes,
            .glyf_off = glyf.?,
            .loca_off = loca.?,
            .hmtx_off = hmtx.?,
            .units_per_em = @floatFromInt(upem),
            .ascent = @floatFromInt(ascent),
            .descent = @floatFromInt(descent),
            .line_gap = @floatFromInt(line_gap),
            .num_glyphs = num_glyphs,
            .num_hmetrics = num_hmetrics,
            .loca_long = loca_fmt == 1,
            .cmap12 = best12,
            .cmap4 = best4,
            .cache = std.AutoHashMap(u32, Glyph).init(gpa),
        };
    }

    pub fn deinit(font: *Font) void {
        var it = font.cache.valueIterator();
        while (it.next()) |g| {
            font.gpa.free(g.points);
            font.gpa.free(g.contours);
            font.gpa.free(g.tris);
        }
        font.cache.deinit();
        font.gpa.free(font.bytes);
    }

    // ---- queries --------------------------------------------------------

    /// Codepoint -> glyph id via the best cmap subtable; 0 = .notdef.
    pub fn glyphIndex(font: *const Font, cp: u32) u32 {
        if (font.cmap12) |off| {
            if (glyphIndex12(font.bytes, off, cp)) |g| return g;
        }
        if (font.cmap4) |off| {
            if (glyphIndex4(font.bytes, off, cp)) |g| return g;
        }
        return 0;
    }

    /// Horizontal advance in font units (the last metric repeats, hmtx's rule).
    pub fn advance(font: *const Font, gid: u32) f32 {
        const g = @min(gid, font.num_glyphs - 1);
        const idx = @min(g, font.num_hmetrics - 1);
        const aw = ru16(font.bytes, font.hmtx_off + idx * 4) catch @as(u16, @intFromFloat(font.units_per_em * 0.5));
        return @floatFromInt(aw);
    }

    pub fn scale(font: *const Font, px: f32) f32 {
        return px / font.units_per_em;
    }

    pub fn ascentPx(font: *const Font, px: f32) f32 {
        return font.ascent * font.scale(px);
    }

    pub fn descentPx(font: *const Font, px: f32) f32 {
        return font.descent * font.scale(px);
    }

    /// One line's full height: ascent + |descent| + line gap, in pixels.
    pub fn lineHeight(font: *const Font, px: f32) f32 {
        return (font.ascent - font.descent + font.line_gap) * font.scale(px);
    }

    /// Advance width of a codepoint in pixels; 0 for unmapped codepoints
    /// (the caller skips - a tofu box needs a fallback font we don't have).
    pub fn advanceOfCp(font: *const Font, cp: u32, px: f32) f32 {
        const gid = font.glyphIndex(cp);
        if (gid == 0 and cp != 0) return 0.0;
        return font.advance(gid) * font.scale(px);
    }

    /// Width of a UTF-8 slice in pixels (no wrapping, no kerning).
    pub fn measure(font: *const Font, text: []const u8, px: f32) f32 {
        var w: f32 = 0;
        var it = utf8Iter(text);
        while (it.next()) |cp| {
            w += font.advanceOfCp(cp, px);
        }
        return w;
    }

    /// The flattened, triangulated glyph, built on first use and cached.
    pub fn glyph(font: *Font, gid: u32) ParseError!Glyph {
        const g = @min(gid, font.num_glyphs - 1);
        if (font.cache.get(g)) |cached| return cached;

        var points: std.ArrayList(Vec2) = .empty;
        var contours: std.ArrayList(polygon.Contour) = .empty;
        var tris: std.ArrayList(u32) = .empty;
        var owned = false;
        defer {
            if (!owned) {
                points.deinit(font.gpa);
                contours.deinit(font.gpa);
                tris.deinit(font.gpa);
            }
        }
        try buildGlyph(font, g, &points, &contours, 0);
        if (contours.items.len > 0) {
            try polygon.triangulate(font.gpa, &points, contours.items, &tris);
        }
        // Hand the buffers to the cache as *exact-length* allocations: the
        // testing allocator (and any strict host allocator) validates that a
        // free's length matches the allocation, and `list.items.len` rarely
        // equals `list.capacity`.
        const p_slice = try points.toOwnedSlice(font.gpa);
        errdefer font.gpa.free(p_slice);
        const c_slice = try contours.toOwnedSlice(font.gpa);
        errdefer font.gpa.free(c_slice);
        const t_slice = try tris.toOwnedSlice(font.gpa);
        errdefer font.gpa.free(t_slice);
        try font.cache.put(g, .{
            .points = p_slice,
            .contours = c_slice,
            .tris = t_slice,
        });
        owned = true;
        return font.cache.get(g).?;
    }

    // ---- glyf ------------------------------------------------------------

    fn buildGlyph(
        font: *Font,
        gid: u32,
        points: *std.ArrayList(Vec2),
        contours: *std.ArrayList(polygon.Contour),
        depth: u32,
    ) ParseError!void {
        if (depth > 8) return error.InvalidFont;
        const data = try glyphData(font, gid);
        if (data.len < 10) return; // empty glyph (space)
        const ncont = try ri16(data, 0);
        if (ncont >= 0) {
            try parseSimple(data, @intCast(ncont), font.gpa, points, contours);
        } else {
            try parseComposite(font, data, points, contours, depth);
        }
    }

    pub fn glyphData(font: *Font, gid: u32) ParseError![]const u8 {
        const g = @min(gid, font.num_glyphs - 1);
        var start: usize = undefined;
        var end: usize = undefined;
        if (font.loca_long) {
            const o = font.loca_off + g * 4;
            start = try ru32(font.bytes, o);
            end = try ru32(font.bytes, o + 4);
        } else {
            const o = font.loca_off + g * 2;
            start = @as(usize, try ru16(font.bytes, o)) * 2;
            end = @as(usize, try ru16(font.bytes, o + 2)) * 2;
        }
        if (end <= start) return &.{};
        const base = @as(usize, font.glyf_off);
        if (base + end > font.bytes.len) return error.TruncatedFont;
        return font.bytes[base + start .. base + end];
    }

    fn parseSimple(
        data: []const u8,
        ncont: u32,
        gpa: std.mem.Allocator,
        points: *std.ArrayList(Vec2),
        contours: *std.ArrayList(polygon.Contour),
    ) ParseError!void {
        if (ncont == 0) return;
        const ends_off: usize = 10;
        const instr_off = ends_off + @as(usize, ncont) * 2;
        if (instr_off + 2 > data.len) return error.TruncatedFont;
        const npts: usize = @as(usize, try ru16(data, ends_off + @as(usize, ncont - 1) * 2)) + 1;

        var off = instr_off;
        const ilen = try ru16(data, off);
        off += 2 + @as(usize, ilen);

        // Flags, run-length repeated (flag bit 3 = REPEAT).
        var flags: std.ArrayList(u8) = .empty;
        defer flags.deinit(gpa);
        while (flags.items.len < npts) {
            if (off >= data.len) return error.TruncatedFont;
            const f = data[off];
            off += 1;
            try flags.append(gpa, f);
            if (f & 0x08 != 0) {
                if (off >= data.len) return error.TruncatedFont;
                const reps = data[off];
                off += 1;
                var r: u8 = 0;
                while (r < reps) : (r += 1) try flags.append(gpa, f);
            }
        }

        // Coordinates: the spec stores ALL x-deltas first, then ALL
        // y-deltas - interleaving them per point drifts as soon as one
        // point's x-encoding differs in size from its y-encoding (which is
        // the common case, not the exception). Both blocks are cumulative.
        var raw: std.ArrayList(RawPoint) = .empty;
        defer raw.deinit(gpa);

        var x: i64 = 0;
        var pi: usize = 0;
        while (pi < npts) : (pi += 1) {
            const f = flags.items[pi];
            if (f & 0x02 != 0) { // X_SHORT
                if (off >= data.len) return error.TruncatedFont;
                const v: i64 = data[off];
                off += 1;
                x += if (f & 0x10 != 0) v else -v;
            } else if (f & 0x10 == 0) { // long x
                x += try ri16(data, off);
                off += 2;
            } // else: same x (bit4 set, no bytes)
            try raw.append(gpa, .{
                .p = .{ .x = @floatFromInt(x), .y = 0.0 },
                .on = f & 0x01 != 0,
            });
        }

        var y: i64 = 0;
        pi = 0;
        while (pi < npts) : (pi += 1) {
            const f = flags.items[pi];
            if (f & 0x04 != 0) { // Y_SHORT
                if (off >= data.len) return error.TruncatedFont;
                const v: i64 = data[off];
                off += 1;
                y += if (f & 0x20 != 0) v else -v;
            } else if (f & 0x20 == 0) { // long y
                y += try ri16(data, off);
                off += 2;
            } // else: same y (bit5 set, no bytes)
            raw.items[pi].p.y = @floatFromInt(y);
        }

        // Flatten each contour (point range between endPts).
        var first: usize = 0;
        var c: u32 = 0;
        while (c < ncont) : (c += 1) {
            const end_pt: usize = @as(usize, try ru16(data, ends_off + @as(usize, c) * 2)) + 1;
            if (end_pt < first) return error.InvalidFont; // endPts went backwards
            const end = @min(end_pt, raw.items.len);
            if (end > first) {
                try flattenContour(gpa, raw.items[first..end], points, contours);
            }
            first = end;
            if (first >= raw.items.len) break;
        }
    }

    /// Walks one contour's on/off points and appends flattened line segments
    /// (always) and quadratic subdivisions (where controls exist) to
    /// `points`, recording the new range as one contour.
    fn flattenContour(
        gpa: std.mem.Allocator,
        pts: []const RawPoint,
        points: *std.ArrayList(Vec2),
        contours: *std.ArrayList(polygon.Contour),
    ) ParseError!void {
        if (pts.len == 0) return;

        // Rotation rule: start at an on-curve point. If both ends are off,
        // synthesize the implied on-curve midpoint between them.
        var seq: std.ArrayList(RawPoint) = .empty;
        defer seq.deinit(gpa);
        if (pts[0].on) {
            try seq.appendSlice(gpa, pts);
        } else if (pts[pts.len - 1].on) {
            try seq.append(gpa, pts[pts.len - 1]);
            try seq.appendSlice(gpa, pts[0 .. pts.len - 1]);
        } else {
            const a = pts[0].p;
            const b = pts[pts.len - 1].p;
            try seq.append(gpa, .{
                .p = .{ .x = (a.x + b.x) * 0.5, .y = (a.y + b.y) * 0.5 },
                .on = true,
            });
            try seq.appendSlice(gpa, pts);
        }

        const start_idx = points.items.len;
        try points.append(gpa, seq.items[0].p);

        var i: usize = 1;
        while (i < seq.items.len) {
            const a = seq.items[i - 1];
            var j = i;
            while (j < seq.items.len and !seq.items[j].on) j += 1;
            if (j == seq.items.len) {
                // Trailing off-curve points close back onto the start.
                try emitSegment(gpa, a, seq.items[i..j], seq.items[0], points);
                i = j; // == len: loop ends
            } else {
                try emitSegment(gpa, a, seq.items[i..j], seq.items[j], points);
                i = j + 1;
            }
        }
        // If the sequence ended on an on-curve point, the closing edge back
        // to the start has not been drawn yet.
        if (seq.items[seq.items.len - 1].on and seq.items.len > 1) {
            try emitSegment(gpa, seq.items[seq.items.len - 1], &.{}, seq.items[0], points);
        }

        const len = points.items.len - start_idx;
        if (len >= 2) try contours.append(gpa, polygon.Contour.init(start_idx, len));
    }

    /// One segment between two on-curve points, possibly through `offs`
    /// (off-curve controls; consecutive controls imply on-curve midpoints).
    fn emitSegment(
        gpa: std.mem.Allocator,
        a: RawPoint,
        offs: []const RawPoint,
        b: RawPoint,
        points: *std.ArrayList(Vec2),
    ) ParseError!void {
        if (offs.len == 0) {
            try points.append(gpa, b.p);
            return;
        }
        var start = a.p;
        var ci: usize = 0;
        while (ci < offs.len) : (ci += 1) {
            const ctrl = offs[ci].p;
            const end: Vec2 = if (ci + 1 < offs.len) blk: {
                const m = offs[ci + 1].p;
                break :blk Vec2{ .x = (ctrl.x + m.x) * 0.5, .y = (ctrl.y + m.y) * 0.5 };
            } else b.p;
            try quadAppend(gpa, points, start, ctrl, end);
            start = end;
        }
    }

    /// Subdivides one quadratic into line segments fine enough that the
    /// chord error stays under half a font unit - invisible at any size the
    /// type scale uses, and only computed once per glyph.
    fn quadAppend(
        gpa: std.mem.Allocator,
        points: *std.ArrayList(Vec2),
        a: Vec2,
        c: Vec2,
        b: Vec2,
    ) ParseError!void {
        const mx = (a.x + b.x) * 0.5;
        const my = (a.y + b.y) * 0.5;
        const d = @sqrt((c.x - mx) * (c.x - mx) + (c.y - my) * (c.y - my));
        // Midpoint deviation of a single segment is d/2, error falls with
        // n^2: n = ceil(sqrt(d)) keeps it under 0.5 units.
        const n: usize = std.math.clamp(@as(usize, @intFromFloat(@ceil(@sqrt(@max(d, 1e-6))))), 1, 24);
        var s: usize = 1;
        while (s <= n) : (s += 1) {
            const t = @as(f32, @floatFromInt(s)) / @as(f32, @floatFromInt(n));
            const u = 1.0 - t;
            const x = u * u * a.x + 2.0 * u * t * c.x + t * t * b.x;
            const y = u * u * a.y + 2.0 * u * t * c.y + t * t * b.y;
            try points.append(gpa, .{ .x = x, .y = y });
        }
    }

    fn parseComposite(
        font: *Font,
        data: []const u8,
        points: *std.ArrayList(Vec2),
        contours: *std.ArrayList(polygon.Contour),
        depth: u32,
    ) ParseError!void {
        var off: usize = 10;
        var more = true;
        while (more) {
            if (off + 4 > data.len) return error.TruncatedFont;
            const flags = try ru16(data, off);
            off += 2;
            const comp_gid = try ru16(data, off);
            off += 2;

            // Args: XY offsets (common) or point matching (ignored -> 0).
            var dx: f32 = 0;
            var dy: f32 = 0;
            if (flags & 0x0001 != 0) { // ARG_1_AND_2_ARE_WORDS
                if (off + 4 > data.len) return error.TruncatedFont;
                if (flags & 0x0002 != 0) { // ARGS_ARE_XY_VALUES
                    dx = @floatFromInt(try ri16(data, off));
                    dy = @floatFromInt(try ri16(data, off + 2));
                }
                off += 4;
            } else {
                if (off + 2 > data.len) return error.TruncatedFont;
                if (flags & 0x0002 != 0) {
                    dx = @as(f32, @floatFromInt(@as(i8, @bitCast(data[off]))));
                    dy = @as(f32, @floatFromInt(@as(i8, @bitCast(data[off + 1]))));
                }
                off += 2;
            }

            var m0: f32 = 1;
            var m1: f32 = 0;
            var m2: f32 = 0;
            var m3: f32 = 1;
            if (flags & 0x0008 != 0) { // WE_HAVE_A_SCALE
                const s = try f2dot14(data, off);
                off += 2;
                m0 = s;
                m3 = s;
            } else if (flags & 0x0040 != 0) { // X_AND_Y_SCALE
                m0 = try f2dot14(data, off);
                m3 = try f2dot14(data, off + 2);
                off += 4;
            } else if (flags & 0x0080 != 0) { // TWO_BY_TWO
                m0 = try f2dot14(data, off);
                m1 = try f2dot14(data, off + 2);
                m2 = try f2dot14(data, off + 4);
                m3 = try f2dot14(data, off + 6);
                off += 8;
            }
            more = flags & 0x0020 != 0; // MORE_COMPONENTS

            var cp: std.ArrayList(Vec2) = .empty;
            defer cp.deinit(font.gpa);
            var cc: std.ArrayList(polygon.Contour) = .empty;
            defer cc.deinit(font.gpa);
            try buildGlyph(font, comp_gid, &cp, &cc, depth + 1);

            const base = points.items.len;
            for (cp.items) |p| {
                try points.append(font.gpa, .{
                    .x = m0 * p.x + m1 * p.y + dx,
                    .y = m2 * p.x + m3 * p.y + dy,
                });
            }
            for (cc.items) |ct| {
                try contours.append(font.gpa, .{
                    .start = @intCast(base + ct.start),
                    .len = ct.len,
                });
            }
        }
    }
};

// ---------------------------------------------------------------------------
// cmap

fn glyphIndex12(bytes: []const u8, off: u32, cp: u32) ?u32 {
    const fmt = ru16(bytes, off) catch return null;
    if (fmt != 12) return null;
    const ngroups = ru32(bytes, off + 12) catch return null;
    var lo: u32 = 0;
    var hi: u32 = ngroups;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const g = @as(usize, off) + 16 + @as(usize, mid) * 12;
        const start = ru32(bytes, g) catch return null;
        const end = ru32(bytes, g + 4) catch return null;
        const sg = ru32(bytes, g + 8) catch return null;
        if (cp < start) {
            hi = mid;
        } else if (cp > end) {
            lo = mid + 1;
        } else {
            return sg + (cp - start);
        }
    }
    return null;
}

fn glyphIndex4(bytes: []const u8, off: u32, cp: u32) ?u32 {
    if (cp > 0xFFFF) return null;
    const fmt = ru16(bytes, off) catch return null;
    if (fmt != 4) return null;
    const segx2 = ru16(bytes, off + 6) catch return null;
    const seg: u32 = segx2 / 2;
    const ends: u32 = off + 14;
    const starts = ends + segx2 + 2; // + reservedPad
    const deltas = starts + segx2;
    const ros = deltas + segx2;
    var i: u32 = 0;
    while (i < seg) : (i += 1) {
        const end = ru16(bytes, ends + i * 2) catch return null;
        if (end < cp) continue;
        const start = ru16(bytes, starts + i * 2) catch return null;
        if (start > cp) return null;
        const delta: i32 = ri16(bytes, deltas + i * 2) catch return null;
        const ro = ru16(bytes, ros + i * 2) catch return null;
        const d: u16 = @bitCast(@as(i16, @intCast(delta)));
        if (ro == 0) {
            // Direct: gid = (codepoint + idDelta) mod 65536.
            const c16: u16 = @truncate(cp);
            return c16 +% d;
        }
        // Through glyphIdArray: read the id, then add idDelta.
        const gaddr = ros + i * 2 + ro + (cp - start) * 2;
        const g = ru16(bytes, gaddr) catch return null;
        if (g == 0) return null;
        return g +% d;
    }
    return null;
}

// ---------------------------------------------------------------------------
// byte readers (TTF is big-endian)

pub fn ru16(b: []const u8, off: usize) ParseError!u16 {
    if (off + 2 > b.len) return error.TruncatedFont;
    return @as(u16, b[off]) << 8 | b[off + 1];
}

pub fn ri16(b: []const u8, off: usize) ParseError!i16 {
    const v = try ru16(b, off);
    return @bitCast(v);
}

fn ru32(b: []const u8, off: usize) ParseError!u32 {
    if (off + 4 > b.len) return error.TruncatedFont;
    return @as(u32, b[off]) << 24 | @as(u32, b[off + 1]) << 16 | @as(u32, b[off + 2]) << 8 | b[off + 3];
}

fn f2dot14(b: []const u8, off: usize) ParseError!f32 {
    const v = try ri16(b, off);
    return @as(f32, @floatFromInt(v)) / 16384.0;
}

fn eq4(tag: []const u8, want: []const u8) bool {
    return std.mem.eql(u8, tag, want);
}

/// Minimal UTF-8 walk: rejects nothing, maps invalid bytes to 0xFFFD-ish
/// replacement by yielding 0xFFFD - fonts have no glyph 0xFFFD requirement,
/// so invalid input shows nothing rather than desynchronising.
fn utf8Iter(text: []const u8) Utf8Iter {
    return .{ .bytes = text };
}

const Utf8Iter = struct {
    bytes: []const u8,
    i: usize = 0,

    fn next(it: *Utf8Iter) ?u32 {
        if (it.i >= it.bytes.len) return null;
        const b0 = it.bytes[it.i];
        if (b0 < 0x80) {
            it.i += 1;
            return b0;
        }
        const len: usize = if (b0 & 0xE0 == 0xC0) 2 else if (b0 & 0xF0 == 0xE0) 3 else if (b0 & 0xF8 == 0xF0) 4 else 1;
        if (len == 1 or it.i + len > it.bytes.len) {
            it.i += 1;
            return 0xFFFD;
        }
        var cp: u32 = switch (len) {
            2 => b0 & 0x1F,
            3 => b0 & 0x0F,
            else => b0 & 0x07,
        };
        var k: usize = 1;
        while (k < len) : (k += 1) {
            const b = it.bytes[it.i + k];
            if (b & 0xC0 != 0x80) {
                it.i += 1;
                return 0xFFFD;
            }
            cp = (cp << 6) | (b & 0x3F);
        }
        it.i += len;
        return cp;
    }
};

// ---------------------------------------------------------------------------
// tests

const test_fonts = [_]struct { name: []const u8, upem: f32 }{
    .{ .name = "Inter-Variable.ttf", .upem = 2048 },
    .{ .name = "Outfit-Variable.ttf", .upem = 1000 },
};

/// Loads a frontend font embedded in the library, independent of the host's
/// working directory. Caller owns the Font (and its byte copy) via `deinit`.
pub fn loadBundled(gpa: std.mem.Allocator, name: []const u8) ParseError!Font {
    const resources = @import("resources");
    const bytes: []const u8 = if (std.mem.eql(u8, name, "Inter-Variable.ttf"))
        resources.inter
    else if (std.mem.eql(u8, name, "Outfit-Variable.ttf"))
        resources.outfit
    else
        return error.FontFileNotFound;
    return Font.init(gpa, bytes);
}

fn loadTestFont(gpa: std.mem.Allocator, name: []const u8) !Font {
    return loadBundled(gpa, name);
}

test "loads embedded fonts without consulting the working directory" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.FontFileNotFound, loadBundled(gpa, "missing.ttf"));
    for (test_fonts) |tf| {
        var f = try loadBundled(gpa, tf.name);
        defer f.deinit();
        try std.testing.expectApproxEqAbs(tf.upem, f.units_per_em, 0.5);
    }
}

test "parses metrics from both bundled fonts" {
    const gpa = std.testing.allocator;
    for (test_fonts) |tf| {
        var f = loadTestFont(gpa, tf.name) catch |e| switch (e) {
            error.FontFileNotFound => return error.SkipZigTest,
            else => return e,
        };
        defer f.deinit();
        try std.testing.expectApproxEqAbs(tf.upem, f.units_per_em, 0.5);
        try std.testing.expect(f.ascent > 0);
        try std.testing.expect(f.descent < 0);
        try std.testing.expect(f.lineHeight(16.0) > 16.0);
        try std.testing.expect(f.lineHeight(16.0) < 40.0);
        try std.testing.expect(f.num_glyphs > 100);
    }
}

test "cmap maps ASCII, accents and misses nothing expected" {
    const gpa = std.testing.allocator;
    var f = loadTestFont(gpa, "Inter-Variable.ttf") catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer f.deinit();
    try std.testing.expect(f.glyphIndex('A') != 0);
    try std.testing.expect(f.glyphIndex('a') != 0);
    try std.testing.expect(f.glyphIndex('0') != 0);
    try std.testing.expect(f.glyphIndex(' ') != 0);
    // e-acute exercises the *composite glyph* path (accent over base).
    try std.testing.expect(f.glyphIndex(0x00E9) != 0);
    // A codepoint no UI font ships.
    try std.testing.expectEqual(@as(u32, 0), f.glyphIndex(0x10FFFD));
    // Round trip through UTF-8 measurement: "Hello" > "Hi" (more advances).
    try std.testing.expect(f.measure("Hello", 16.0) > f.measure("Hi", 16.0));
}

test "glyphs flatten, triangulate, and a counter leaves a hole" {
    const gpa = std.testing.allocator;
    var f = loadTestFont(gpa, "Inter-Variable.ttf") catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer f.deinit();

    const gid_o = f.glyphIndex('o');
    const g = try f.glyph(gid_o);
    try std.testing.expect(g.contours.len >= 2); // outer + counter
    try std.testing.expect(g.tris.len >= 6);

    // The hole must actually subtract: triangulated |area| is the largest
    // contour's area minus every other contour's area.
    var max_a: f32 = 0;
    var rest: f32 = 0;
    for (g.contours) |ct| {
        const a = polygon.contourArea(g.points, ct);
        if (a > max_a) {
            rest += max_a;
            max_a = a;
        } else {
            rest += a;
        }
    }
    const filled = @abs(polygon.signedArea(g.points, g.tris));
    try std.testing.expectApproxEqAbs(max_a - rest, filled, (max_a - rest) * 0.03 + 1.0);

    // A solid glyph ('I') fills its full contour area.
    const gid_I = f.glyphIndex('I');
    const gi = try f.glyph(gid_I);
    const area_I = polygon.contourArea(gi.points, gi.contours[0]);
    const filled_I = @abs(polygon.signedArea(gi.points, gi.tris));
    try std.testing.expectApproxEqAbs(area_I, filled_I, area_I * 0.03 + 1.0);
}

test "composites land inside their declared bounding box" {
    const gpa = std.testing.allocator;
    var f = loadTestFont(gpa, "Inter-Variable.ttf") catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer f.deinit();

    const gid = f.glyphIndex(0x00E9); // e-acute: composite
    const data = try f.glyphData(gid);
    const nc = try ri16(data, 0);
    try std.testing.expect(nc < 0); // really is composite
    const x_min: f32 = @floatFromInt(try ri16(data, 2));
    const y_min: f32 = @floatFromInt(try ri16(data, 4));
    const x_max: f32 = @floatFromInt(try ri16(data, 6));
    const y_max: f32 = @floatFromInt(try ri16(data, 8));
    const slack = f.units_per_em * 0.03;
    const g = try f.glyph(gid);
    try std.testing.expect(g.points.len > 0);
    for (g.points) |p| {
        try std.testing.expect(p.x >= x_min - slack);
        try std.testing.expect(p.x <= x_max + slack);
        try std.testing.expect(p.y >= y_min - slack);
        try std.testing.expect(p.y <= y_max + slack);
    }
}

test "the cache returns the same buffers on repeat" {
    const gpa = std.testing.allocator;
    var f = loadTestFont(gpa, "Inter-Variable.ttf") catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer f.deinit();
    const gid = f.glyphIndex('B');
    const g1 = try f.glyph(gid);
    const g2 = try f.glyph(gid);
    try std.testing.expectEqual(g1.points.ptr, g2.points.ptr);
    try std.testing.expectEqual(g1.tris.len, g2.tris.len);
}
