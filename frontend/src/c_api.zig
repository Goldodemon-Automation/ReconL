//! The C ABI surface of the frontend: `include/reconl_ui.h` implemented as
//! plain exports over the same types the Zig showcase drives.
//!
//! Rules this file keeps:
//!
//! * Every exported function is a thin translation - no widget logic here,
//!   so a host bug cannot behave differently from the Zig path.
//! * Nothing panics across the boundary: every error is caught, recorded in
//!   the context (or the creation slot), and reported as a zero/negative
//!   return with `reconlUiLastError` carrying the message.
//! * The structs below are `extern struct` and must match the header byte
//!   for byte; `abi_test.zig` asserts that against a real `@cImport`.
//! * Strings arrive NUL-terminated (C), leave as NUL-terminated static
//!   buffers, and any pointer returned from a context dies at the next call
//!   on that context - as the header documents.

const std = @import("std");
const front = @import("reconl-frontend");
const backend = @import("reconl-backend");

const geom = front.geom;
const theme_mod = front.theme;
const font_mod = front.font;
const Font = font_mod.Font;
const ui_mod = front.ui;

/// `ReconLUiInput` - floats and uint32s, natural 4-byte alignment, 28 bytes.
pub const CInput = extern struct {
    px: f32 = -1e6,
    py: f32 = -1e6,
    down: u32 = 0,
    pressed: u32 = 0,
    released: u32 = 0,
    wheel: f32 = 0,
    dt_ms: f32 = 1000.0 / 60.0,
};

/// `ReconLRect`.
pub const CRect = extern struct {
    x: f32 = 0,
    y: f32 = 0,
    w: f32 = 0,
    h: f32 = 0,
};

const MAGIC: u32 = 0x524C_5549; // "RLUI"

/// Creation failures have no context to record into, so they land here;
/// `reconlUiLastError(NULL)` reads it back.
threadlocal var g_create_error: [256]u8 = [_]u8{0} ** 256;
threadlocal var g_create_error_len: usize = 0;

fn setCreateErr(comptime fmt: []const u8, args: anytype) void {
    const n = std.fmt.bufPrint(g_create_error[0..255], fmt, args) catch return;
    g_create_error_len = n.len;
    g_create_error[n.len] = 0;
}

const Context = struct {
    magic: u32 = MAGIC,
    gpa: std.mem.Allocator,
    font_a: Font, // Inter   (ui / body / label / mono fallback)
    font_b: Font, // Outfit  (display: title / heading)
    fonts: ui_mod.Fonts = undefined,
    ui: ui_mod.Ui = undefined,
    renderer: backend.Renderer,
    mesh: ui_mod.Mesh = .{
        .vertices = &[_]front.tess.Vertex{},
        .indices = &[_]u32{},
    },
    err_buf: [256]u8 = [_]u8{0} ** 256,
    err_len: usize = 0,

    fn setErr(self: *Context, comptime fmt: []const u8, args: anytype) void {
        const n = std.fmt.bufPrint(self.err_buf[0..255], fmt, args) catch return;
        self.err_len = n.len;
        self.err_buf[n.len] = 0;
    }

    fn setErrZig(self: *Context, e: anyerror) void {
        self.setErr("{s}", .{@errorName(e)});
    }
};

fn ctxOf(p: ?*Context) ?*Context {
    const ctx = p orelse return null;
    if (ctx.magic != MAGIC) return null;
    return ctx;
}

fn str(z: ?[*c]const u8) []const u8 {
    const s = z orelse return "";
    return std.mem.span(s);
}

fn rectOut(r: geom.Rect) CRect {
    return .{ .x = r.x, .y = r.y, .w = r.w, .h = r.h };
}

fn finiteUnit(value: f32) f32 {
    return if (std.math.isFinite(value)) std.math.clamp(value, 0.0, 1.0) else 0.0;
}

fn boundedExtent(ctx: *const Context, value: f32) f32 {
    const extent: f64 = @as(f64, @floatFromInt(@max(ctx.renderer.width, ctx.renderer.height))) * 2.0;
    return @floatCast(if (std.math.isFinite(value)) std.math.clamp(@as(f64, value), 0.0, extent) else 0.0);
}

fn boundedPanelRect(ctx: *const Context, r: CRect) geom.Rect {
    // Panel chrome is emitted before its clip is pushed, so unbounded but
    // finite C inputs can overflow Rect.right/bottom and poison tessellation.
    // Clamp both original endpoints in f64 to a generous off-screen margin:
    // this preserves the visible portion while keeping all later f32 geometry
    // arithmetic comfortably finite.
    const extent: f64 = @as(f64, @floatFromInt(@max(ctx.renderer.width, ctx.renderer.height))) * 2.0;
    const x0: f64 = if (std.math.isFinite(r.x)) r.x else 0.0;
    const y0: f64 = if (std.math.isFinite(r.y)) r.y else 0.0;
    const w: f64 = if (std.math.isFinite(r.w)) @max(r.w, 0.0) else 0.0;
    const h: f64 = if (std.math.isFinite(r.h)) @max(r.h, 0.0) else 0.0;
    const left = std.math.clamp(x0, -extent, extent);
    const top = std.math.clamp(y0, -extent, extent);
    const right = std.math.clamp(x0 + w, -extent, extent);
    const bottom = std.math.clamp(y0 + h, -extent, extent);
    return .{
        .x = @floatCast(left),
        .y = @floatCast(top),
        .w = @floatCast(@max(0.0, right - left)),
        .h = @floatCast(@max(0.0, bottom - top)),
    };
}

fn swapchainBytes(width: u32, height: u32) ?u64 {
    const pixels = @as(u64, width) * @as(u64, height);
    return std.math.mul(u64, pixels, 8) catch null;
}

// ---------------------------------------------------------------- lifecycle

export fn reconlUiAbiVersion() u32 {
    return 1;
}

export fn reconlUiCreate(width: u32, height: u32) ?*Context {
    const gpa = std.heap.c_allocator;
    const w = @max(width, 1);
    const h = @max(height, 1);
    if (swapchainBytes(w, h) == null) {
        setCreateErr("surface dimensions overflow the swapchain byte size", .{});
        return null;
    }

    var font_a = font_mod.loadBundled(gpa, "Inter-Variable.ttf") catch |e| {
        setCreateErr("Inter-Variable.ttf: {s}", .{@errorName(e)});
        return null;
    };
    var font_b = font_mod.loadBundled(gpa, "Outfit-Variable.ttf") catch |e| {
        setCreateErr("Outfit-Variable.ttf: {s}", .{@errorName(e)});
        font_a.deinit();
        return null;
    };

    const theme = theme_mod.default;
    var renderer = backend.Renderer.init(gpa, w, h, theme.bg) catch {
        const msg = backend.createError();
        setCreateErr("renderer: {s}", .{if (msg.len > 0) msg else "unknown failure"});
        font_a.deinit();
        font_b.deinit();
        return null;
    };

    const ctx = gpa.create(Context) catch {
        renderer.deinit();
        font_a.deinit();
        font_b.deinit();
        setCreateErr("out of memory allocating the UI context", .{});
        return null;
    };
    ctx.* = .{
        .gpa = gpa,
        .font_a = font_a,
        .font_b = font_b,
        .renderer = renderer,
    };
    // Pointers into our own storage: the context is heap-stable, so these
    // never dangle.
    ctx.fonts = .{ .ui = &ctx.font_a, .display = &ctx.font_b };
    ctx.ui = ui_mod.Ui.init(gpa, ctx.fonts, theme, @floatFromInt(w), @floatFromInt(h));
    return ctx;
}

export fn reconlUiDestroy(p: ?*Context) void {
    const ctx = ctxOf(p) orelse return;
    const gpa = ctx.gpa;
    ctx.ui.deinit();
    ctx.renderer.deinit();
    ctx.font_a.deinit();
    ctx.font_b.deinit();
    ctx.magic = 0;
    gpa.destroy(ctx);
}

export fn reconlUiResize(p: ?*Context, width: u32, height: u32) i32 {
    const ctx = ctxOf(p) orelse return -1;
    if (swapchainBytes(@max(width, 1), @max(height, 1)) == null) {
        ctx.setErr("surface dimensions overflow the swapchain byte size", .{});
        return -2;
    }
    ctx.renderer.resize(width, height) catch |e| {
        const msg = ctx.renderer.lastError();
        if (msg.len > 0) ctx.setErr("{s}", .{msg}) else ctx.setErrZig(e);
        return -2;
    };
    // Surface geometry and any in-progress pointer capture refer to the old
    // coordinate space; neither is valid after a successful resize.
    ctx.ui.active_id = 0;
    // A frozen mesh contains coordinates and clipping from the old surface.
    // Discard it before allowing a render at the new size.
    ctx.ui.discardFrame();
    ctx.mesh = .{ .vertices = &[_]front.tess.Vertex{}, .indices = &[_]u32{} };
    // The swapchain moved; layout follows the same numbers.
    ctx.ui.setSize(@floatFromInt(ctx.renderer.width), @floatFromInt(ctx.renderer.height));
    return 0;
}

export fn reconlUiSetClear(p: ?*Context, r: f32, g: f32, b: f32, a: f32) void {
    const ctx = ctxOf(p) orelse return;
    ctx.renderer.setClear(.{
        .r = finiteUnit(r),
        .g = finiteUnit(g),
        .b = finiteUnit(b),
        .a = finiteUnit(a),
    });
}

export fn reconlUiLastError(p: ?*Context) [*c]const u8 {
    if (ctxOf(p)) |ctx| return @ptrCast(&ctx.err_buf);
    g_create_error[g_create_error_len] = 0;
    return @ptrCast(&g_create_error);
}

// ------------------------------------------------------------------- frame

export fn reconlUiBegin(p: ?*Context, input: ?*const CInput) void {
    const ctx = ctxOf(p) orelse return;
    ctx.mesh = .{ .vertices = &[_]front.tess.Vertex{}, .indices = &[_]u32{} };
    const inp: ui_mod.Input = if (input) |i| .{
        .px = if (std.math.isFinite(i.px)) i.px else -1e6,
        .py = if (std.math.isFinite(i.py)) i.py else -1e6,
        .down = i.down != 0,
        .pressed = i.pressed != 0,
        .released = i.released != 0,
        .wheel = if (std.math.isFinite(i.wheel)) i.wheel else 0,
        .dt_ms = if (std.math.isFinite(i.dt_ms) and i.dt_ms > 0.0) @min(i.dt_ms, 1000.0) else 1000.0 / 60.0,
    } else .{};
    ctx.ui.begin(inp) catch |e| {
        ctx.setErrZig(e);
    };
}

export fn reconlUiEnd(p: ?*Context) void {
    const ctx = ctxOf(p) orelse return;
    if (!ctx.ui.frame_open) return;
    ctx.ui.end();
}

export fn reconlUiEndFrame(p: ?*Context) u32 {
    const ctx = ctxOf(p) orelse return 0;
    if (!ctx.ui.frame_open) return 0;
    ctx.mesh = ctx.ui.endFrame();
    return if (ctx.mesh.indices.len > 0) 1 else 0;
}

export fn reconlUiRender(p: ?*Context, pixels: ?*anyopaque, size: u64) i32 {
    const ctx = ctxOf(p) orelse return -1;
    const ptr = pixels orelse {
        ctx.setErr("null pixel pointer", .{});
        return -2;
    };
    const need: u64 = @as(u64, ctx.renderer.width) * @as(u64, ctx.renderer.height) * 4;
    if (size < need) {
        ctx.setErr(
            "pixel buffer is {d} bytes; {d}x{d} RGBA8 needs {d}",
            .{ size, ctx.renderer.width, ctx.renderer.height, need },
        );
        return -4;
    }
    const slice: []u8 = @as([*]u8, @ptrCast(ptr))[0..@intCast(need)];
    ctx.renderer.render(ctx.mesh, slice) catch |e| {
        const msg = ctx.renderer.lastError();
        if (msg.len > 0) ctx.setErr("{s}", .{msg}) else ctx.setErrZig(e);
        return -5;
    };
    return 0;
}

export fn reconlUiFrameIndex(p: ?*Context) u32 {
    const ctx = ctxOf(p) orelse return 0;
    return ctx.renderer.frame_index;
}

// ----------------------------------------------------------------- widgets

export fn reconlUiLabel(p: ?*Context, text: [*c]const u8, role_i: i32, color: [*c]const f32) f32 {
    const ctx = ctxOf(p) orelse return 0;
    if (!ctx.ui.frame_open) return 0;
    const role: theme_mod.Role = switch (role_i) {
        0 => .title,
        1 => .heading,
        2 => .body,
        3 => .label,
        4 => .mono,
        else => .body,
    };
    const c: ?geom.Color = if (color == null) null else .{
        .r = finiteUnit(color[0]),
        .g = finiteUnit(color[1]),
        .b = finiteUnit(color[2]),
        .a = finiteUnit(color[3]),
    };
    const h = ctx.ui.label(str(text), role, c, .{}) catch |e| {
        ctx.setErrZig(e);
        return 0;
    };
    return h;
}

export fn reconlUiButton(p: ?*Context, id: [*c]const u8, text: [*c]const u8, flags: u32) u32 {
    const ctx = ctxOf(p) orelse return 0;
    if (!ctx.ui.frame_open) return 0;
    const hit = ctx.ui.button(str(id), str(text), .{
        .primary = flags & 1 != 0,
    }) catch |e| {
        ctx.setErrZig(e);
        return 0;
    };
    return @intFromBool(hit);
}

export fn reconlUiToggle(p: ?*Context, id: [*c]const u8, text: [*c]const u8, value: [*c]u32) u32 {
    const ctx = ctxOf(p) orelse return 0;
    if (!ctx.ui.frame_open or value == null) return 0;
    var on = value.* != 0;
    const changed = ctx.ui.toggle(str(id), str(text), &on) catch |e| {
        ctx.setErrZig(e);
        return 0;
    };
    value.* = @intFromBool(on);
    return @intFromBool(changed);
}

export fn reconlUiSlider(
    p: ?*Context,
    id: [*c]const u8,
    text: [*c]const u8,
    value: [*c]f32,
    min: f32,
    max: f32,
) u32 {
    const ctx = ctxOf(p) orelse return 0;
    if (!ctx.ui.frame_open or value == null) return 0;
    if (!std.math.isFinite(min) or !std.math.isFinite(max) or min > max) {
        ctx.setErr("slider range must be finite and min <= max", .{});
        return 0;
    }
    if (!std.math.isFinite(value.*)) value.* = min;
    const changed = ctx.ui.slider(str(id), str(text), &value.*, min, max) catch |e| {
        ctx.setErrZig(e);
        return 0;
    };
    return @intFromBool(changed);
}

export fn reconlUiProgress(p: ?*Context, id: [*c]const u8, t: f32, height: f32) void {
    const ctx = ctxOf(p) orelse return;
    if (!ctx.ui.frame_open) return;
    ctx.ui.progress(
        str(id),
        finiteUnit(t),
        boundedExtent(ctx, height),
    ) catch |e| ctx.setErrZig(e);
}

export fn reconlUiSeparator(p: ?*Context) void {
    const ctx = ctxOf(p) orelse return;
    if (!ctx.ui.frame_open) return;
    ctx.ui.separator() catch |e| ctx.setErrZig(e);
}

export fn reconlUiSpace(p: ?*Context, amount: f32) void {
    const ctx = ctxOf(p) orelse return;
    if (!ctx.ui.frame_open) return;
    ctx.ui.space(boundedExtent(ctx, amount)) catch |e| ctx.setErrZig(e);
}

export fn reconlUiBeginPanel(
    p: ?*Context,
    rect: [*c]const CRect,
    pad: f32,
    gap: f32,
    flags: u32,
) void {
    const ctx = ctxOf(p) orelse return;
    if (!ctx.ui.frame_open) return;
    const rp = rect orelse return;
    const r = rp.*;
    const shadow = flags & 1 != 0;
    const border = flags & 2 == 0;
    ctx.ui.beginPanel(
        boundedPanelRect(ctx, r),
        .{
            .pad = boundedExtent(ctx, pad),
            .gap = boundedExtent(ctx, gap),
            .shadow = shadow,
            .border = border,
        },
    ) catch |e| ctx.setErrZig(e);
}

export fn reconlUiBeginRow(p: ?*Context, height: f32, gap: f32) void {
    const ctx = ctxOf(p) orelse return;
    if (!ctx.ui.frame_open) return;
    ctx.ui.beginRow(
        boundedExtent(ctx, height),
        boundedExtent(ctx, gap),
    ) catch |e| ctx.setErrZig(e);
}

export fn reconlUiBeginScroll(p: ?*Context, id: [*c]const u8, height: f32, content_h: f32) CRect {
    const ctx = ctxOf(p) orelse return .{};
    if (!ctx.ui.frame_open) return .{};
    if (!std.math.isFinite(height) or !std.math.isFinite(content_h) or height < 0 or content_h < 0) {
        ctx.setErr("scroll dimensions must be finite and non-negative", .{});
        return .{};
    }
    const height_px = boundedExtent(ctx, height);
    const content_px = boundedExtent(ctx, content_h);
    const r = ctx.ui.beginScroll(str(id), height_px, content_px) catch |e| {
        ctx.setErrZig(e);
        return .{};
    };
    return rectOut(r);
}

export fn reconlUiSparkline(p: ?*Context, values: [*c]const f32, count: u32, height: f32) void {
    const ctx = ctxOf(p) orelse return;
    if (!ctx.ui.frame_open) return;
    const slice: []const f32 = if (values == null or count == 0)
        &.{}
    else
        values[0..count];
    ctx.ui.sparkline(slice, boundedExtent(ctx, height)) catch |e| ctx.setErrZig(e);
}

export fn reconlUiLastRect(p: ?*Context) CRect {
    const ctx = ctxOf(p) orelse return .{};
    if (!ctx.ui.frame_open) return .{};
    return rectOut(ctx.ui.last_rect);
}

// -------------------------------------------------------------------- mesh

export fn reconlUiVertexCount(p: ?*Context) u32 {
    const ctx = ctxOf(p) orelse return 0;
    return @intCast(ctx.mesh.vertices.len);
}

export fn reconlUiIndexCount(p: ?*Context) u32 {
    const ctx = ctxOf(p) orelse return 0;
    return @intCast(ctx.mesh.indices.len);
}

export fn reconlUiVertices(p: ?*Context) ?*const anyopaque {
    const ctx = ctxOf(p) orelse return null;
    if (ctx.mesh.vertices.len == 0) return null;
    return @ptrCast(ctx.mesh.vertices.ptr);
}

export fn reconlUiIndices(p: ?*Context) ?*const anyopaque {
    const ctx = ctxOf(p) orelse return null;
    if (ctx.mesh.indices.len == 0) return null;
    return @ptrCast(ctx.mesh.indices.ptr);
}
