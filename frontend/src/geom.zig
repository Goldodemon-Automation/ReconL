//! Geometry primitives for the UI layer: vectors, rectangles, colours, and the
//! pixel-to-clip orthographic transform the backend pushes as push constants.
//!
//! The UI works in *logical pixels*, origin top-left, y growing down - the
//! coordinate system a designer thinks in. The rasteriser's clip space has
//! y growing up (NDC +1 is the top row, which `raster/src/framegen.rs` pins).
//! [`ortho`] is the one matrix that reconciles the two, so no widget ever
//! sees anything but pixels.

pub const Vec2 = struct {
    x: f32,
    y: f32,

    pub fn add(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x + b.x, .y = a.y + b.y };
    }

    pub fn sub(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x - b.x, .y = a.y - b.y };
    }

    pub fn scale(a: Vec2, s: f32) Vec2 {
        return .{ .x = a.x * s, .y = a.y * s };
    }
};

/// An axis-aligned rectangle in logical pixels. `w`/`h` are non-negative by
/// construction: [`Rect.inset`] and [`Rect.intersect`] clamp rather than
/// producing negative extents, because a widget whose padding ate its whole
/// box must still paint nothing, not an inverted shape.
pub const Rect = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,

    pub fn right(r: Rect) f32 {
        return r.x + r.w;
    }

    pub fn bottom(r: Rect) f32 {
        return r.y + r.h;
    }

    pub fn center(r: Rect) Vec2 {
        return .{ .x = r.x + r.w * 0.5, .y = r.y + r.h * 0.5 };
    }

    /// Shrinks (or grows, with a negative `d`) on every side; clamps to empty.
    pub fn inset(r: Rect, d: f32) Rect {
        const nw = @max(0.0, r.w - 2.0 * d);
        const nh = @max(0.0, r.h - 2.0 * d);
        return .{ .x = r.x + d, .y = r.y + d, .w = nw, .h = nh };
    }

    pub fn with_size(r: Rect, w: f32, h: f32) Rect {
        return .{ .x = r.x, .y = r.y, .w = w, .h = h };
    }

    pub fn contains(r: Rect, p: Vec2) bool {
        return p.x >= r.x and p.y >= r.y and p.x < r.right() and p.y < r.bottom();
    }

    pub fn intersect(a: Rect, b: Rect) Rect {
        const x0 = @max(a.x, b.x);
        const y0 = @max(a.y, b.y);
        const x1 = @min(a.right(), b.right());
        const y1 = @min(a.bottom(), b.bottom());
        return .{
            .x = x0,
            .y = y0,
            .w = @max(0.0, x1 - x0),
            .h = @max(0.0, y1 - y0),
        };
    }

    pub fn is_empty(r: Rect) bool {
        return r.w <= 0.0 or r.h <= 0.0;
    }
};

/// Straight-alpha sRGB colour, 0..1 floats. The renderer's vertex colour is
/// exactly this layout (`ReconLVertex.color`), so a tessellated vertex copies
/// straight through with no conversion at the boundary.
pub const Color = struct {
    r: f32 = 0.0,
    g: f32 = 0.0,
    b: f32 = 0.0,
    a: f32 = 1.0,

    pub fn rgba8(r: u8, g: u8, b: u8, a: u8) Color {
        return .{
            .r = @as(f32, @floatFromInt(r)) / 255.0,
            .g = @as(f32, @floatFromInt(g)) / 255.0,
            .b = @as(f32, @floatFromInt(b)) / 255.0,
            .a = @as(f32, @floatFromInt(a)) / 255.0,
        };
    }

    pub fn hex(s: *const [7]u8) Color {
        // "#RRGGBB" -> components. Compile-time friendly for literals: a
        // nibble at a time, so comptime default values don't drag std.fmt's
        // parseInt (and its branch quota) into comptime evaluation.
        const nibble = struct {
            fn n(c: u8) u8 {
                return switch (c) {
                    '0'...'9' => c - '0',
                    'a'...'f' => c - 'a' + 10,
                    'A'...'F' => c - 'A' + 10,
                    else => 0,
                };
            }
        }.n;
        return rgba8(
            nibble(s[1]) * 16 + nibble(s[2]),
            nibble(s[3]) * 16 + nibble(s[4]),
            nibble(s[5]) * 16 + nibble(s[6]),
            255,
        );
    }

    pub fn with_alpha(c: Color, alpha: f32) Color {
        return .{ .r = c.r, .g = c.g, .b = c.b, .a = alpha };
    }

    /// Linear blend; used by every animated colour transition.
    pub fn lerp(a: Color, b: Color, t: f32) Color {
        const k = std.math.clamp(t, 0.0, 1.0);
        return .{
            .r = a.r + (b.r - a.r) * k,
            .g = a.g + (b.g - a.g) * k,
            .b = a.b + (b.b - a.b) * k,
            .a = a.a + (b.a - a.a) * k,
        };
    }
};

const std = @import("std");

/// Column-major 4x4 matrix, the layout `reconlCmdPushConstants` documents
/// (`slot 0 = view * projection`).
pub const Mat4 = [16]f32;

/// Orthographic projection from logical pixels (y down) to clip space.
///
/// Maps x in [0, w] to NDC [-1, 1] and y in [0, h] to NDC [+1, -1], so the
/// top-left pixel corner lands at NDC (-1, +1) - which is the *top* row, the
/// convention `framegen.rs` and the D3D-style raster path both pin. Depth is
/// constant 0.5: UI geometry never depth-tests against itself, and reversed-Z
/// clear is 0.0, so 0.5 passes `GREATER` on every pass it is drawn in.
pub fn ortho(w: f32, h: f32) Mat4 {
    const sx = 2.0 / @max(w, 1.0);
    const sy = -2.0 / @max(h, 1.0);
    return .{
        sx, 0.0, 0.0, 0.0,
        0.0, sy, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -1.0, 1.0, 0.5, 1.0,
    };
}

pub const IDENTITY: Mat4 = .{
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
};

test "ortho maps the four pixel corners to the four clip corners" {
    const m = ortho(200.0, 100.0);
    // Column-major: clip = M * (x, y, z, 1).
    const point = struct {
        fn mul(mat: Mat4, x: f32, y: f32) [2]f32 {
            const cx = mat[0] * x + mat[4] * y + mat[12];
            const cy = mat[1] * x + mat[5] * y + mat[13];
            return .{ cx, cy };
        }
    }.mul;
    const tl = point(m, 0.0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), tl[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tl[1], 1e-6);
    const br = point(m, 200.0, 100.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), br[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), br[1], 1e-6);
}

test "rect inset clamps to empty instead of inverting" {
    const r = Rect{ .x = 0, .y = 0, .w = 10, .h = 10 };
    const e = r.inset(6.0);
    try std.testing.expect(e.is_empty());
    try std.testing.expect(e.w >= 0.0 and e.h >= 0.0);
}

test "intersect never yields a negative extent" {
    const a = Rect{ .x = 0, .y = 0, .w = 10, .h = 10 };
    const b = Rect{ .x = 20, .y = 20, .w = 5, .h = 5 };
    const i = Rect.intersect(a, b);
    try std.testing.expect(i.is_empty());
}

test "colour hex parse matches rgba8" {
    const c = Color.hex("#5F8A80");
    const d = Color.rgba8(0x5F, 0x8A, 0x80, 255);
    try std.testing.expectApproxEqAbs(c.r, d.r, 1e-6);
    try std.testing.expectApproxEqAbs(c.g, d.g, 1e-6);
    try std.testing.expectApproxEqAbs(c.b, d.b, 1e-6);
}
