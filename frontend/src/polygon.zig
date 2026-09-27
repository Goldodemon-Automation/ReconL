//! Scanline triangulation of flattened closed contours under the *nonzero*
//! winding rule - the rule TrueType outlines are defined with, so a glyph's
//! counters (the holes in `o`, `B`, `e`) fall out of the same sweep that
//! fills its stems.
//!
//! Why scanline and not ear clipping: fonts hand us self-touching contours
//! with holes, bridges and implied on-curve points. Ear clipping needs each
//! hole surgically bridged into its outer ring first, and every bridging bug
//! shows up as a missing chunk of a letter. The scanline sweep has no
//! bridging step at all: split at every vertex y, sort the edge crossings of
//! each band by x, accumulate winding, emit a trapezoid for every interval
//! whose winding is non-zero. The output tiles the original region exactly -
//! no T-junctions (every band shares its y with its neighbours), no
//! orientation sensitivity (the caller's winding decides, as the font does).
//!
//! Emission adds its band-corner points to the caller's point list, because
//! a trapezoid's corners are interpolated positions, not input vertices.

const std = @import("std");
const geom = @import("geom.zig");
const Vec2 = geom.Vec2;

/// A contour is a half-open range into the shared point list.
pub const Contour = struct {
    start: u32,
    len: u32,

    pub fn init(start: usize, len: usize) Contour {
        return .{ .start = @intCast(start), .len = @intCast(len) };
    }
};

const Edge = struct {
    a: u32,
    b: u32,
};

const Crossing = struct {
    edge: u32,
    /// x of the edge at the band's low y and high y.
    x0: f32,
    x1: f32,
    /// Winding contribution: +1 when the edge points in +y (y-up font space).
    sign: i32,
    /// x at the band's midpoint - the sort key; edges cannot cross inside a
    /// band (bands are split at every vertex), so this order is total.
    xm: f32,
};

/// Triangulates `contours` over `points`, appending triangles to `out`
/// (indices into `points`, including points this function appends).
/// The same `points` buffer may be reused across glyphs; `out` is cleared
/// by the caller, not here.
pub fn triangulate(
    gpa: std.mem.Allocator,
    points: *std.ArrayList(Vec2),
    contours: []const Contour,
    out: *std.ArrayList(u32),
) !void {
    if (contours.len == 0) return;

    // ---- band boundaries: every distinct vertex y -----------------------
    var ys: std.ArrayList(f32) = .empty;
    defer ys.deinit(gpa);
    for (contours) |ct| {
        var i: u32 = 0;
        while (i < ct.len) : (i += 1) {
            const p = points.items[ct.start + i];
            try ys.append(gpa, p.y);
        }
    }
    if (ys.items.len < 3) return;
    std.mem.sort(f32, ys.items, {}, lessThan);
    // Dedup exact-equal vertex coordinates (shared contour endpoints).
    var ny: usize = 1;
    var k: usize = 1;
    while (k < ys.items.len) : (k += 1) {
        if (ys.items[k] != ys.items[ny - 1]) {
            ys.items[ny] = ys.items[k];
            ny += 1;
        }
    }
    ys.items.len = ny;
    if (ny < 2) return;

    // ---- edges ----------------------------------------------------------
    var edges: std.ArrayList(Edge) = .empty;
    defer edges.deinit(gpa);
    for (contours) |ct| {
        if (ct.len < 2) continue;
        var i: u32 = 0;
        while (i < ct.len) : (i += 1) {
            const a = ct.start + i;
            const b = ct.start + (i + 1) % ct.len;
            const pa = points.items[a];
            const pb = points.items[b];
            if (pa.y == pb.y and pa.x == pb.x) continue; // degenerate
            try edges.append(gpa, .{ .a = a, .b = b });
        }
    }
    if (edges.items.len == 0) return;

    // ---- sweep ----------------------------------------------------------
    var crossings: std.ArrayList(Crossing) = .empty;
    defer crossings.deinit(gpa);

    var band: usize = 0;
    while (band + 1 < ys.items.len) : (band += 1) {
        const y0 = ys.items[band];
        const y1 = ys.items[band + 1];
        const ym = y0 + (y1 - y0) * 0.5;
        crossings.clearRetainingCapacity();

        for (edges.items, 0..) |e, ei| {
            const pa = points.items[e.a];
            const pb = points.items[e.b];
            // Spans the midpoint (no vertex can lie strictly inside a band).
            const a_low = pa.y <= ym;
            const b_low = pb.y <= ym;
            if (a_low == b_low) continue;
            const dy = pb.y - pa.y;
            if (dy == 0.0) continue;
            const t0 = (y0 - pa.y) / dy;
            const t1 = (y1 - pa.y) / dy;
            const tm = (ym - pa.y) / dy;
            try crossings.append(gpa, .{
                .edge = @intCast(ei),
                .x0 = pa.x + (pb.x - pa.x) * t0,
                .x1 = pa.x + (pb.x - pa.x) * t1,
                .sign = if (pb.y > pa.y) @as(i32, 1) else -1,
                .xm = pa.x + (pb.x - pa.x) * tm,
            });
        }
        if (crossings.items.len < 2) continue;
        std.mem.sort(Crossing, crossings.items, {}, crossingLess);

        // ---- winding sweep over sorted crossings ------------------------
        var w: i32 = 0;
        var j: usize = 0;
        while (j < crossings.items.len) : (j += 1) {
            w += crossings.items[j].sign;
            if (w == 0 or j + 1 >= crossings.items.len) continue;
            const c0 = crossings.items[j];
            const c1 = crossings.items[j + 1];
            const top0 = try pushPoint(gpa, points, c0.x0, y0);
            const top1 = try pushPoint(gpa, points, c1.x0, y0);
            const bot1 = try pushPoint(gpa, points, c1.x1, y1);
            const bot0 = try pushPoint(gpa, points, c0.x1, y1);
            // (top0, top1, bot1, bot0) in y-up: y0 is the *lower* edge.
            try pushTri(gpa, out, top0, top1, bot1);
            try pushTri(gpa, out, top0, bot1, bot0);
        }
    }
}

fn pushPoint(gpa: std.mem.Allocator, points: *std.ArrayList(Vec2), x: f32, y: f32) !u32 {
    const idx: u32 = @intCast(points.items.len);
    try points.append(gpa, .{ .x = x, .y = y });
    return idx;
}

fn pushTri(gpa: std.mem.Allocator, out: *std.ArrayList(u32), a: u32, b: u32, c: u32) !void {
    try out.appendSlice(gpa, &.{ a, b, c });
}

fn lessThan(_: void, lhs: f32, rhs: f32) bool {
    return lhs < rhs;
}

fn crossingLess(_: void, lhs: Crossing, rhs: Crossing) bool {
    if (lhs.xm != rhs.xm) return lhs.xm < rhs.xm;
    return lhs.edge < rhs.edge;
}

/// Signed area of a triangle soup (y-up: positive = counter-clockwise).
pub fn signedArea(points: []const Vec2, indices: []const u32) f32 {
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i + 2 < indices.len) : (i += 3) {
        const a = points[indices[i]];
        const b = points[indices[i + 1]];
        const c = points[indices[i + 2]];
        sum += (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    }
    return sum * 0.5;
}

/// Shoelace area of one closed contour (absolute).
pub fn contourArea(points: []const Vec2, ct: Contour) f32 {
    var sum: f32 = 0.0;
    var i: u32 = 0;
    while (i < ct.len) : (i += 1) {
        const a = points[ct.start + i];
        const b = points[ct.start + (i + 1) % ct.len];
        sum += a.x * b.y - b.x * a.y;
    }
    return @abs(sum) * 0.5;
}

// ---------------------------------------------------------------------------
// tests

fn squarePts(gpa: std.mem.Allocator, x0: f32, y0: f32, x1: f32, y1: f32, ccw: bool) !struct {
    points: std.ArrayList(Vec2),
    contour: Contour,
} {
    var points: std.ArrayList(Vec2) = .empty;
    errdefer points.deinit(gpa);
    const pts = if (ccw)
        [_]Vec2{
            .{ .x = x0, .y = y0 },
            .{ .x = x1, .y = y0 },
            .{ .x = x1, .y = y1 },
            .{ .x = x0, .y = y1 },
        }
    else
        [_]Vec2{
            .{ .x = x0, .y = y0 },
            .{ .x = x0, .y = y1 },
            .{ .x = x1, .y = y1 },
            .{ .x = x1, .y = y0 },
        };
    for (pts) |p| try points.append(gpa, p);
    return .{ .points = points, .contour = Contour.init(0, 4) };
}

test "a CCW square fills its own area" {
    const gpa = std.testing.allocator;
    var s = try squarePts(gpa, 0, 0, 10, 10, true);
    defer s.points.deinit(gpa);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try triangulate(gpa, &s.points, &.{s.contour}, &out);
    try std.testing.expect(out.items.len >= 6);
    const area = signedArea(s.points.items, out.items);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), area, 1e-3);
}

test "a hole cuts out under nonzero winding" {
    const gpa = std.testing.allocator;
    var points: std.ArrayList(Vec2) = .empty;
    defer points.deinit(gpa);
    // Outer CCW 0..10, inner CW 3..7 (a real hole).
    const outer = [_]Vec2{
        .{ .x = 0, .y = 0 }, .{ .x = 10, .y = 0 }, .{ .x = 10, .y = 10 }, .{ .x = 0, .y = 10 },
    };
    const inner = [_]Vec2{
        .{ .x = 3, .y = 3 }, .{ .x = 3, .y = 7 }, .{ .x = 7, .y = 7 }, .{ .x = 7, .y = 3 },
    };
    for (outer) |p| try points.append(gpa, p);
    for (inner) |p| try points.append(gpa, p);
    const contours = [_]Contour{ Contour.init(0, 4), Contour.init(4, 4) };
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try triangulate(gpa, &points, &contours, &out);
    const area = signedArea(points.items, out.items);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0 - 16.0), area, 1e-3);
}

test "a concave W fills exactly its shoelace area" {
    const gpa = std.testing.allocator;
    var points: std.ArrayList(Vec2) = .empty;
    defer points.deinit(gpa);
    const w = [_]Vec2{
        .{ .x = 0, .y = 0 },  .{ .x = 4, .y = 10 }, .{ .x = 8, .y = 0 },
        .{ .x = 10, .y = 0 }, .{ .x = 10, .y = 12 }, .{ .x = 0, .y = 12 },
    };
    for (w) |p| try points.append(gpa, p);
    const ct = Contour.init(0, w.len);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try triangulate(gpa, &points, &.{ct}, &out);
    const expect = contourArea(points.items, ct);
    const got = signedArea(points.items, out.items);
    try std.testing.expectApproxEqAbs(expect, got, 1e-2);
}

test "two identical CCW squares fill once (winding 2 counts as filled)" {
    const gpa = std.testing.allocator;
    var points: std.ArrayList(Vec2) = .empty;
    defer points.deinit(gpa);
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        const pts = [_]Vec2{
            .{ .x = 0, .y = 0 }, .{ .x = 5, .y = 0 }, .{ .x = 5, .y = 5 }, .{ .x = 0, .y = 5 },
        };
        for (pts) |p| try points.append(gpa, p);
    }
    const contours = [_]Contour{ Contour.init(0, 4), Contour.init(4, 4) };
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try triangulate(gpa, &points, &contours, &out);
    const area = signedArea(points.items, out.items);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), area, 1e-3);
}

test "an S-curve flatten fills its shoelace area" {
    const gpa = std.testing.allocator;
    var points: std.ArrayList(Vec2) = .empty;
    defer points.deinit(gpa);
    // A blob sampled off a sine - a stand-in for a flattened glyph curve.
    var i: usize = 0;
    while (i < 40) : (i += 1) {
        const t = @as(f32, @floatFromInt(i)) / 40.0 * std.math.pi * 2.0;
        try points.append(gpa, .{
            .x = 50.0 + 40.0 * @cos(t),
            .y = 50.0 + 30.0 * @sin(t) + 8.0 * @sin(3.0 * t),
        });
    }
    const ct = Contour.init(0, 40);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try triangulate(gpa, &points, &.{ct}, &out);
    const expect = contourArea(points.items, ct);
    const got = signedArea(points.items, out.items);
    try std.testing.expectApproxEqAbs(expect, got, @max(1e-2, expect * 1e-4));
}
