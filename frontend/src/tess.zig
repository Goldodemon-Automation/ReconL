//! The display list: widgets emit triangles here, and nothing in this file
//! knows a renderer exists. `backend.zig` uploads whatever comes out.
//!
//! Everything is in *logical pixels* with painter's-algorithm ordering -
//! callers draw backgrounds before content, so the rasteriser's
//! `RECONL_CULL_NONE` + alpha blend renders UI in draw order with no depth
//! tricks. A fixed z of 0.0 is fine: the ortho matrix in `geom.zig` shifts z
//! to 0.5, which passes reversed-Z `GREATER` against the 0.0 clear on every
//! pass, and depth writes are off in the UI pipeline so overlapping widgets
//! never fight over equal depths.
//!
//! Clipping (the missing scissor in the ABI) is done here on the CPU: the
//! clip stack is a stack of rects, every emitted triangle is Sutherland-
//! Hodgman-clipped against the intersection, and the fan is appended. UI
//! scroll areas are this and nothing else.

const std = @import("std");
const geom = @import("geom.zig");
const Rect = geom.Rect;
const Vec2 = geom.Vec2;
const Color = geom.Color;

/// Byte-for-byte `ReconLVertex` (48 bytes: pos3, normal3, uv2, color4).
/// `extern struct` so the shared library can hand this straight across the
/// C ABI without a conversion step.
pub const Vertex = extern struct {
    position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    normal: [3]f32 = .{ 0.0, 0.0, 1.0 },
    uv: [2]f32 = .{ 0.0, 0.0 },
    color: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 },

    pub fn make(p: Vec2, col: Color) Vertex {
        return .{
            .position = .{ p.x, p.y, 0.0 },
            .normal = .{ 0.0, 0.0, 1.0 },
            .uv = .{ 0.0, 0.0 },
            .color = .{ col.r, col.g, col.b, col.a },
        };
    }
};

pub const DisplayList = struct {
    vertices: std.ArrayList(Vertex),
    indices: std.ArrayList(u32),
    /// Stack of active clip rects. The intersection of the whole stack is
    /// maintained as we push, so clipping tests one rect, not N.
    clips: std.ArrayList(Rect),

    pub fn init() DisplayList {
        return .{
            .vertices = .empty,
            .indices = .empty,
            .clips = .empty,
        };
    }

    pub fn deinit(dl: *DisplayList, gpa: std.mem.Allocator) void {
        dl.vertices.deinit(gpa);
        dl.indices.deinit(gpa);
        dl.clips.deinit(gpa);
    }

    pub fn clear(dl: *DisplayList) void {
        dl.vertices.clearRetainingCapacity();
        dl.indices.clearRetainingCapacity();
        dl.clips.clearRetainingCapacity();
    }

    /// Intersects `r` with the current clip and pushes the result.
    pub fn pushClip(dl: *DisplayList, gpa: std.mem.Allocator, r: Rect) !void {
        const current = if (dl.clips.items.len == 0) r else Rect.intersect(dl.clips.items[dl.clips.items.len - 1], r);
        try dl.clips.append(gpa, current);
    }

    pub fn popClip(dl: *DisplayList) void {
        std.debug.assert(dl.clips.items.len > 0);
        dl.clips.items.len -= 1;
    }

    /// The rectangle every subsequent emit is confined to (whole logical
    /// frame when the stack is empty).
    pub fn activeClip(dl: *DisplayList, bounds: Rect) Rect {
        if (dl.clips.items.len == 0) return bounds;
        return dl.clips.items[dl.clips.items.len - 1];
    }

    // ---- raw emission ---------------------------------------------------

    /// Emits one triangle, clipped against the stack. The fan keeps the
    /// triangle's vertex order; with `CULL_NONE` winding never matters.
    pub fn tri(dl: *DisplayList, gpa: std.mem.Allocator, p: [3]Vec2, c: [3]Color) !void {
        if (dl.clips.items.len == 0) {
            try dl.push3(gpa, p, c);
            return;
        }
        // Sutherland-Hodgman against one axis-aligned rect (4 planes).
        var poly: [8]Vec2 = undefined;
        var col: [8]Color = undefined;
        @memcpy(poly[0..3], &p);
        @memcpy(col[0..3], &c);
        var n: usize = 3;
        const clip = dl.clips.items[dl.clips.items.len - 1];
        const planes = [4]struct { axis: enum { x, y }, value: f32, keep_greater: bool }{
            .{ .axis = .x, .value = clip.x, .keep_greater = true },
            .{ .axis = .x, .value = clip.right(), .keep_greater = false },
            .{ .axis = .y, .value = clip.y, .keep_greater = true },
            .{ .axis = .y, .value = clip.bottom(), .keep_greater = false },
        };
        for (planes) |pl| {
            if (n < 3) return;
            var out_poly: [8]Vec2 = undefined;
            var out_col: [8]Color = undefined;
            var m: usize = 0;
            const coord = struct {
                fn get(v: Vec2, a: @TypeOf(pl.axis)) f32 {
                    return switch (a) {
                        .x => v.x,
                        .y => v.y,
                    };
                }
            }.get;
            var i: usize = 0;
            while (i < n) : (i += 1) {
                const a = poly[i];
                const b = poly[(i + 1) % n];
                const ca = coord(a, pl.axis);
                const cb = coord(b, pl.axis);
                const a_in = if (pl.keep_greater) ca >= pl.value else ca <= pl.value;
                const b_in = if (pl.keep_greater) cb >= pl.value else cb <= pl.value;
                if (a_in) {
                    out_poly[m] = a;
                    out_col[m] = col[i];
                    m += 1;
                }
                if (a_in != b_in) {
                    const denom = cb - ca;
                    const t = if (denom == 0.0) 0.0 else (pl.value - ca) / denom;
                    const q = Vec2{
                        .x = a.x + (b.x - a.x) * t,
                        .y = a.y + (b.y - a.y) * t,
                    };
                    out_poly[m] = q;
                    out_col[m] = Color.lerp(col[i], col[(i + 1) % n], t);
                    m += 1;
                }
                if (m >= 8) break;
            }
            @memcpy(poly[0..m], out_poly[0..m]);
            @memcpy(col[0..m], out_col[0..m]);
            n = m;
        }
        if (n < 3) return;
        var i: usize = 1;
        while (i + 1 < n) : (i += 1) {
            try dl.push3(gpa, .{ poly[0], poly[i], poly[i + 1] }, .{ col[0], col[i], col[i + 1] });
        }
    }

    fn push3(dl: *DisplayList, gpa: std.mem.Allocator, p: [3]Vec2, c: [3]Color) !void {
        const base: u32 = @intCast(dl.vertices.items.len);
        try dl.vertices.ensureTotalCapacity(gpa, dl.vertices.items.len + 3);
        try dl.vertices.append(gpa, Vertex.make(p[0], c[0]));
        try dl.vertices.append(gpa, Vertex.make(p[1], c[1]));
        try dl.vertices.append(gpa, Vertex.make(p[2], c[2]));
        try dl.indices.appendSlice(gpa, &.{ base, base + 1, base + 2 });
    }

    // ---- shapes ---------------------------------------------------------

    pub fn rect(dl: *DisplayList, gpa: std.mem.Allocator, r: Rect, col: Color) !void {
        try dl.quad(gpa, .{
            .{ .x = r.x, .y = r.y },
            .{ .x = r.right(), .y = r.y },
            .{ .x = r.right(), .y = r.bottom() },
            .{ .x = r.x, .y = r.bottom() },
        }, .{ col, col, col, col });
    }

    /// Vertical gradient: top and bottom colours, linear across the height.
    /// Two triangles with equal colours on each edge *is* an exact linear
    /// vertical ramp, so no subdivision is needed.
    pub fn rectGrad(dl: *DisplayList, gpa: std.mem.Allocator, r: Rect, top: Color, bottom: Color) !void {
        try dl.quad(gpa, .{
            .{ .x = r.x, .y = r.y },
            .{ .x = r.right(), .y = r.y },
            .{ .x = r.right(), .y = r.bottom() },
            .{ .x = r.x, .y = r.bottom() },
        }, .{ top, top, bottom, bottom });
    }

    pub fn quad(dl: *DisplayList, gpa: std.mem.Allocator, p: [4]Vec2, c: [4]Color) !void {
        try dl.tri(gpa, .{ p[0], p[1], p[2] }, .{ c[0], c[1], c[2] });
        try dl.tri(gpa, .{ p[0], p[2], p[3] }, .{ c[0], c[2], c[3] });
    }

    pub fn roundedRect(dl: *DisplayList, gpa: std.mem.Allocator, r: Rect, radius: f32, col: Color) !void {
        try dl.roundedRectGrad(gpa, r, radius, col, col);
    }

    pub fn roundedRectGrad(
        dl: *DisplayList,
        gpa: std.mem.Allocator,
        r: Rect,
        radius: f32,
        top: Color,
        bottom: Color,
    ) !void {
        const rad = clampRadius(r, radius);
        if (rad < 0.5) {
            try dl.rectGrad(gpa, r, top, bottom);
            return;
        }
        var ring: [72]Vec2 = undefined;
        const n = buildRing(r, rad, ringSegs(rad), &ring);
        const center = Vec2{ .x = r.x + r.w * 0.5, .y = r.y + r.h * 0.5 };
        // Vertical ramp approximated per band: colour at a point is chosen
        // by its y, so the fan interpolates the gradient per triangle. With
        // ~11 segments per quarter the maximum colour error is a fraction of
        // a percent - invisible, and exact at the rectangle's own edges.
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const a = ring[i];
            const b = ring[(i + 1) % n];
            try dl.tri(gpa, .{ center, a, b }, .{ yColor(r, top, bottom, center), yColor(r, top, bottom, a), yColor(r, top, bottom, b) });
        }
    }

    /// Stroke: a ring between two matched-radius rectangles. Building it as
    /// a ring (rather than filling outer then inner) keeps translucent
    /// borders honest over whatever is behind them.
    pub fn stroke(dl: *DisplayList, gpa: std.mem.Allocator, r: Rect, radius: f32, thickness: f32, col: Color) !void {
        const t = thickness;
        if (t <= 0.0 or r.w <= 0.0 or r.h <= 0.0) return;
        const inner = r.inset(t);
        if (inner.is_empty()) {
            try dl.roundedRect(gpa, r, radius, col);
            return;
        }
        const rad_out = clampRadius(r, radius);
        const rad_in = clampRadius(inner, rad_out - t);
        var outer_pts: [72]Vec2 = undefined;
        var inner_pts: [72]Vec2 = undefined;
        // Both rings must carry the same point count or the quads between
        // them cannot be zipped - so the segment count comes from the outer
        // radius alone. A sharp inner corner is a zero-radius arc drawn at
        // the outer density (its points collapse onto the corner, which is
        // exactly right).
        const outer_n = if (rad_out < 0.5) blk: {
            _ = buildBoxRing(r, &outer_pts);
            break :blk buildBoxRing(inner, &inner_pts);
        } else blk: {
            const segs = ringSegs(rad_out);
            const no = buildRing(r, rad_out, segs, &outer_pts);
            const ni = buildRing(inner, rad_in, segs, &inner_pts);
            std.debug.assert(no == ni);
            break :blk no;
        };
        var i: usize = 0;
        while (i < outer_n) : (i += 1) {
            const j = (i + 1) % outer_n;
            try dl.quad(gpa, .{ outer_pts[i], outer_pts[j], inner_pts[j], inner_pts[i] }, .{ col, col, col, col });
        }
    }

    /// A thick line segment (caps are square; joins are covered by drawing
    /// consecutive segments - adequate for sparklines and underlines).
    pub fn line(dl: *DisplayList, gpa: std.mem.Allocator, a: Vec2, b: Vec2, thickness: f32, col: Color) !void {
        const d = b.sub(a);
        const len = @sqrt(d.x * d.x + d.y * d.y);
        if (len < 1e-6) return;
        const nx = -d.y / len * (thickness * 0.5);
        const ny = d.x / len * (thickness * 0.5);
        const o = Vec2{ .x = nx, .y = ny };
        try dl.quad(gpa, .{
            a.sub(o),
            b.sub(o),
            b.add(o),
            a.add(o),
        }, .{ col, col, col, col });
    }

    // ---- ring construction ---------------------------------------------

    fn clampRadius(r: Rect, radius: f32) f32 {
        return @min(@max(radius, 0.0), @min(r.w, r.h) * 0.5);
    }

    fn yColor(r: Rect, top: Color, bottom: Color, p: Vec2) Color {
        if (top.r == bottom.r and top.g == bottom.g and top.b == bottom.b and top.a == bottom.a) return top;
        const t = if (r.h <= 0.0) 0.0 else (p.y - r.y) / r.h;
        return Color.lerp(top, bottom, t);
    }

    /// 4-corner ring for a plain rectangle (radius below the AA threshold).
    fn buildBoxRing(r: Rect, out: *[72]Vec2) usize {
        out[0] = .{ .x = r.x, .y = r.y };
        out[1] = .{ .x = r.right(), .y = r.y };
        out[2] = .{ .x = r.right(), .y = r.bottom() };
        out[3] = .{ .x = r.x, .y = r.bottom() };
        return 4;
    }

    /// Segments per quarter-circle: ~2px of arc each, so a 4px radius and a
    /// 100px pill both read as smooth curves without over-tessellating.
    fn ringSegs(rad: f32) usize {
        const arc = std.math.pi * 0.5 * rad;
        return std.math.clamp(@as(usize, @intFromFloat(@ceil(arc / 2.0) + 0.5)), 3, 16);
    }

    /// Clockwise ring (y-down) starting at the top-left corner's left edge.
    /// Segment density adapts to radius: ~2px arcs, capped, so a pill button
    /// and a 14px card corner both look smooth without over-tessellating.
    fn buildRing(r: Rect, rad: f32, segs: usize, out: *[72]Vec2) usize {
        var n: usize = 0;
        const corners = [4]struct { cx: f32, cy: f32, a0: f32 }{
            .{ .cx = r.x + rad, .cy = r.y + rad, .a0 = std.math.pi }, // TL: 180 -> 270
            .{ .cx = r.right() - rad, .cy = r.y + rad, .a0 = 1.5 * std.math.pi }, // TR: 270 -> 360
            .{ .cx = r.right() - rad, .cy = r.bottom() - rad, .a0 = 0.0 }, // BR: 0 -> 90
            .{ .cx = r.x + rad, .cy = r.bottom() - rad, .a0 = 0.5 * std.math.pi }, // BL: 90 -> 180
        };
        for (corners) |c| {
            var s: usize = 0;
            while (s <= segs) : (s += 1) {
                const ang = c.a0 + (std.math.pi * 0.5) * (@as(f32, @floatFromInt(s)) / @as(f32, @floatFromInt(segs)));
                out[n] = .{ .x = c.cx + rad * @cos(ang), .y = c.cy + rad * @sin(ang) };
                n += 1;
            }
            // Drop the duplicated seam point between corners except the very
            // last (which the caller's modulo closes): each corner arc ends
            // exactly where the next begins only when rad spans the edge; in
            // between, the straight edge is the segment from this arc's end
            // to the next arc's start - keep both, they differ.
        }
        // The arcs above append segs+1 points per corner; consecutive
        // corners share no point, so the straight edges are implicit chords.
        // The first point of corner 0 is duplicated at the end of BL's arc
        // only when the ring closes: remove the duplicate.
        if (n >= 2) {
            const first = out[0];
            const last = out[n - 1];
            if (@abs(first.x - last.x) < 1e-6 and @abs(first.y - last.y) < 1e-6) {
                n -= 1;
            }
        }
        return n;
    }
};

// ---------------------------------------------------------------------------
// tests

test "a plain rect becomes two triangles with the right area" {
    const gpa = std.testing.allocator;
    var dl = DisplayList.init();
    defer dl.deinit(gpa);
    try dl.rect(gpa, .{ .x = 0, .y = 0, .w = 10, .h = 20 }, .{ .r = 1, .g = 0, .b = 0, .a = 1 });
    // Two independent triangles (the list never shares vertices between
    // primitives - clipping produces fresh ones anyway).
    try std.testing.expectEqual(@as(usize, 6), dl.vertices.items.len);
    try std.testing.expectEqual(@as(usize, 6), dl.indices.items.len);
    const area = trianglesArea(dl);
    try std.testing.expectApproxEqAbs(@as(f32, 200.0), area, 1e-3);
}

test "clip stack confines a rect to the intersection" {
    const gpa = std.testing.allocator;
    var dl = DisplayList.init();
    defer dl.deinit(gpa);
    try dl.pushClip(gpa, .{ .x = 5, .y = 5, .w = 10, .h = 10 });
    try dl.pushClip(gpa, .{ .x = 0, .y = 0, .w = 8, .h = 100 });
    try dl.rect(gpa, .{ .x = -50, .y = -50, .w = 200, .h = 200 }, .{ .r = 1, .g = 1, .b = 1, .a = 1 });
    dl.popClip();
    dl.popClip();
    // Intersection is (5,5)-(8,15): 3 x 10 = 30.
    const area = trianglesArea(dl);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), area, 1e-3);
}

test "clipping away entirely emits nothing" {
    const gpa = std.testing.allocator;
    var dl = DisplayList.init();
    defer dl.deinit(gpa);
    try dl.pushClip(gpa, .{ .x = 100, .y = 100, .w = 5, .h = 5 });
    try dl.rect(gpa, .{ .x = 0, .y = 0, .w = 10, .h = 10 }, .{});
    dl.popClip();
    try std.testing.expectEqual(@as(usize, 0), dl.vertices.items.len);
}

test "a rounded rect's ring stays inside its box and covers most of it" {
    const gpa = std.testing.allocator;
    var dl = DisplayList.init();
    defer dl.deinit(gpa);
    try dl.roundedRect(gpa, .{ .x = 0, .y = 0, .w = 100, .h = 40 }, 12.0, .{});
    // The fan is inside the box...
    for (dl.vertices.items) |v| {
        try std.testing.expect(v.position[0] >= -1e-4 and v.position[0] <= 100.0 + 1e-4);
        try std.testing.expect(v.position[1] >= -1e-4 and v.position[1] <= 40.0 + 1e-4);
    }
    // ...and covers box minus the four corner bites (about 4*(r^2 - pi r^2/4)).
    const area = trianglesArea(dl);
    const corner_loss = 4.0 * (144.0 - std.math.pi * 144.0 / 4.0);
    try std.testing.expect(area > 4000.0 - corner_loss - 20.0);
    try std.testing.expect(area < 4000.0);
}

test "stroke rings lie between outer and inner boxes" {
    const gpa = std.testing.allocator;
    var dl = DisplayList.init();
    defer dl.deinit(gpa);
    try dl.stroke(gpa, .{ .x = 0, .y = 0, .w = 100, .h = 50 }, 8.0, 2.0, .{});
    // Outer rounded box minus inner rounded box. Corner bites are
    // r^2 - pi r^2/4 each: outer r=8 loses 54.9, inner r=6 loses 30.9, so
    // (5000 - 54.9) - (4416 - 30.9) = 559.9, minus a whisker for the chords
    // the arcs are approximated with.
    const area = trianglesArea(dl);
    try std.testing.expectApproxEqAbs(@as(f32, 559.97), area, 8.0);
    for (dl.vertices.items) |v| {
        try std.testing.expect(v.position[0] >= -1e-4 and v.position[0] <= 100.0 + 1e-4);
        try std.testing.expect(v.position[1] >= -1e-4 and v.position[1] <= 50.0 + 1e-4);
    }
}

fn trianglesArea(dl: DisplayList) f32 {
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < dl.indices.items.len) : (i += 3) {
        const a = dl.vertices.items[dl.indices.items[i]];
        const b = dl.vertices.items[dl.indices.items[i + 1]];
        const c = dl.vertices.items[dl.indices.items[i + 2]];
        const abx = b.position[0] - a.position[0];
        const aby = b.position[1] - a.position[1];
        const acx = c.position[0] - a.position[0];
        const acy = c.position[1] - a.position[1];
        sum += abx * acy - aby * acx;
    }
    return @abs(sum) * 0.5;
}
