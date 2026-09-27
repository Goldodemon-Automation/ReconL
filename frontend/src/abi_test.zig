//! Tests that drive the frontend through the *published* C header.
//!
//! The declarations come from `@cImport`ing `include/reconl_ui.h` - exactly
//! what a C, Java (Panama) or Kotlin consumer sees - and the definitions come
//! from linking `c_api.zig` into this test binary, so a signature that drifts
//! from the header fails here rather than in a foreign language's toolchain.
//!
//! These tests need the real thing: bundled fonts and libreconl. When fonts
//! are absent the suite skips (as the font tests do);
//! when reconl is absent the step itself is skipped by `build.zig`.

const std = @import("std");
const c = @cImport({
    @cInclude("reconl_ui.h");
});
// Importing also exports the implementations this test links against.
const c_api = @import("c_api.zig");

const testing = std.testing;

const W: u32 = 320;
const H: u32 = 180;

/// Creates a UI, skipping when the only missing thing is a font file and
/// failing when the reason looks like a real defect.
fn createOrSkip() !*c.ReconLUi {
    if (c.reconlUiCreate(W, H)) |ui| return ui;
    const msg = std.mem.span(c.reconlUiLastError(null));
    std.debug.print("reconlUiCreate failed: {s}\n", .{msg});
    if (std.mem.indexOf(u8, msg, "FileNotFound") != null) return error.SkipZigTest;
    if (std.mem.indexOf(u8, msg, "font") != null) return error.SkipZigTest;
    return error.TestUnexpectedResult;
}

fn input(px: f32, py: f32, down: bool, pressed: bool, released: bool) c.ReconLUiInput {
    return .{
        .px = px,
        .py = py,
        .down = @intFromBool(down),
        .pressed = @intFromBool(pressed),
        .released = @intFromBool(released),
        .wheel = 0,
        .dt_ms = 1000.0 / 60.0,
    };
}

test "the header agrees with the implementation, byte for byte" {
    try testing.expectEqual(@as(usize, 28), @sizeOf(c.ReconLUiInput));
    try testing.expectEqual(@sizeOf(c.ReconLUiInput), @sizeOf(c_api.CInput));
    try testing.expectEqual(@offsetOf(c.ReconLUiInput, "down"), @offsetOf(c_api.CInput, "down"));
    try testing.expectEqual(@offsetOf(c.ReconLUiInput, "dt_ms"), @offsetOf(c_api.CInput, "dt_ms"));
    try testing.expectEqual(@as(usize, 16), @sizeOf(c.ReconLRect));
    try testing.expectEqual(@sizeOf(c.ReconLRect), @sizeOf(c_api.CRect));

    // The version macro and the runtime answer cannot disagree.
    try testing.expectEqual(@as(u32, c.RECONLUI_ABI_VERSION), c.reconlUiAbiVersion());

    // Roles are ABI: the header's numbers are the theme enum's ordinals.
    try testing.expectEqual(@as(c_int, 0), c.RECONLUI_ROLE_TITLE);
    try testing.expectEqual(@as(c_int, 4), c.RECONLUI_ROLE_MONO);
}

test "a frame built through the header presents pixels through reconl" {
    const ui = try createOrSkip();
    defer c.reconlUiDestroy(ui);

    var in = input(-1e6, -1e6, false, false, false);
    _ = c.reconlUiBegin(ui, &in);
    const rect = c.ReconLRect{ .x = 8, .y = 8, .w = @floatFromInt(W - 16), .h = 80 };
    c.reconlUiBeginPanel(ui, &rect, 12, 8, c.RECONLUI_PANEL_SHADOW);
    _ = c.reconlUiLabel(ui, "Hello from C", c.RECONLUI_ROLE_TITLE, null);
    _ = c.reconlUiButton(ui, "ok", "OK", c.RECONLUI_BUTTON_PRIMARY);
    var on: u32 = 1;
    _ = c.reconlUiToggle(ui, "dark", "Dark mode", &on);
    var q: f32 = 0.5;
    _ = c.reconlUiSlider(ui, "q", "Quality", &q, 0, 1);
    c.reconlUiProgress(ui, "p", 0.66, 6);
    const samples = [_]f32{ 0.1, 0.4, 0.3, 0.8, 0.6, 0.9 };
    c.reconlUiSparkline(ui, &samples, samples.len, 24);
    c.reconlUiEnd(ui); // panel
    const got_mesh = c.reconlUiEndFrame(ui);

    try testing.expectEqual(@as(u32, 1), got_mesh);
    try testing.expect(c.reconlUiVertexCount(ui) > 0);
    try testing.expect(c.reconlUiIndexCount(ui) % 3 == 0);
    const vp = c.reconlUiVertices(ui);
    const ip = c.reconlUiIndices(ui);
    try testing.expect(vp != null);
    try testing.expect(ip != null);

    // The published stride claim: vertices are ReconLVertex, position first.
    // Reading x through the header's pointer must yield a pixel-ish coordinate.
    const vraw: *const anyopaque = vp.?;
    const x = @as(*const f32, @ptrCast(@alignCast(vraw))).*;
    try testing.expect(x >= -16.0 and x <= @as(f32, @floatFromInt(W)) + 16.0);

    const pixels = try testing.allocator.alloc(u8, @as(usize, W) * H * 4);
    defer testing.allocator.free(pixels);
    try testing.expectEqual(@as(i32, 0), c.reconlUiRender(ui, pixels.ptr, pixels.len));

    // The clear colour is the theme background (#0B0D0C); the panel paints
    // something else over most of the frame.
    const differs = blk: {
        var i: usize = 0;
        while (i < pixels.len) : (i += 4) {
            if (pixels[i] != 0x0B or pixels[i + 1] != 0x0D or pixels[i + 2] != 0x0C) break :blk true;
        }
        break :blk false;
    };
    try testing.expect(differs);

    // Same mesh, second render: backend output is deterministic.
    const again = try testing.allocator.alloc(u8, pixels.len);
    defer testing.allocator.free(again);
    try testing.expectEqual(@as(i32, 0), c.reconlUiRender(ui, again.ptr, again.len));
    try testing.expectEqualSlices(u8, pixels, again);
    try testing.expectEqual(@as(u32, 2), c.reconlUiFrameIndex(ui));

    // An undersized buffer is refused with a message, not a crash.
    var tiny: [64]u8 = undefined;
    try testing.expect(c.reconlUiRender(ui, &tiny, tiny.len) < 0);
    try testing.expect(std.mem.indexOf(
        u8,
        std.mem.span(c.reconlUiLastError(ui)),
        "needs",
    ) != null);

    // Failed output-buffer validation does not consume a frame number.
    try testing.expectEqual(@as(u32, 2), c.reconlUiFrameIndex(ui));
}

test "clicks land through the header: press then release over the widget" {
    const ui = try createOrSkip();
    defer c.reconlUiDestroy(ui);

    // --- button: draw once to learn where it is (lastRect is ABI) ---
    var in = input(-1e6, -1e6, false, false, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiButton(ui, "go", "Go", 0);
    const r = c.reconlUiLastRect(ui);
    _ = c.reconlUiEndFrame(ui);
    try testing.expect(r.w > 0 and r.h > 0);

    const bx = r.x + r.w * 0.5;
    const by = r.y + r.h * 0.5;

    // Press over it: no click yet.
    in = input(bx, by, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "go", "Go", 0));
    _ = c.reconlUiEndFrame(ui);

    // Release over it: that is a click.
    in = input(bx, by, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 1), c.reconlUiButton(ui, "go", "Go", 0));
    _ = c.reconlUiEndFrame(ui);

    // --- toggle: its own press/release pair, at its own rect ---
    var on: u32 = 0;
    in = input(-1e6, -1e6, false, false, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiToggle(ui, "t", "Toggle", &on);
    const tr = c.reconlUiLastRect(ui);
    _ = c.reconlUiEndFrame(ui);

    const tx = tr.x + tr.w * 0.5;
    const ty = tr.y + tr.h * 0.5;

    in = input(tx, ty, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 0), c.reconlUiToggle(ui, "t", "Toggle", &on));
    _ = c.reconlUiEndFrame(ui);
    try testing.expectEqual(@as(u32, 0), on);

    in = input(tx, ty, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 1), c.reconlUiToggle(ui, "t", "Toggle", &on));
    _ = c.reconlUiEndFrame(ui);
    try testing.expectEqual(@as(u32, 1), on);
}

test "surface dimension overflow is refused without panicking" {
    const ui = try createOrSkip();
    defer c.reconlUiDestroy(ui);
    try testing.expect(c.reconlUiResize(ui, 0xFFFF_FFFF, 0xFFFF_FFFF) < 0);
    try testing.expect(std.mem.indexOf(
        u8,
        std.mem.span(c.reconlUiLastError(ui)),
        "overflow",
    ) != null);
    var in = input(-1e6, -1e6, false, false, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiLabel(ui, "still usable", c.RECONLUI_ROLE_BODY, null);
    _ = c.reconlUiEndFrame(ui);
}

test "resize discards the old mesh and keeps the UI usable" {
    const ui = try createOrSkip();
    defer c.reconlUiDestroy(ui);

    var in = input(20, 20, false, false, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiLabel(ui, "old-size mesh must not survive resize", c.RECONLUI_ROLE_BODY, null);
    _ = c.reconlUiEndFrame(ui);
    try testing.expect(c.reconlUiVertexCount(ui) > 0);

    try testing.expectEqual(@as(i32, 0), c.reconlUiResize(ui, 640, 360));
    try testing.expectEqual(@as(u32, 0), c.reconlUiVertexCount(ui));
    try testing.expectEqual(@as(u32, 0), c.reconlUiIndexCount(ui));

    const pixels = try testing.allocator.alloc(u8, 640 * 360 * 4);
    defer testing.allocator.free(pixels);
    try testing.expectEqual(@as(i32, 0), c.reconlUiRender(ui, pixels.ptr, pixels.len));
    try testing.expectEqual(@as(u32, 1), c.reconlUiFrameIndex(ui));

    var next = input(20, 20, false, false, false);
    _ = c.reconlUiBegin(ui, &next);
    _ = c.reconlUiLabel(ui, "resized", c.RECONLUI_ROLE_BODY, null);
    _ = c.reconlUiEndFrame(ui);
    try testing.expect(c.reconlUiVertexCount(ui) > 0);
    try testing.expectEqual(@as(i32, 0), c.reconlUiRender(ui, pixels.ptr, pixels.len));
    try testing.expectEqual(@as(u32, 2), c.reconlUiFrameIndex(ui));

    // The old buffer no longer fits: refused by size, not by crash.
    const old = try testing.allocator.alloc(u8, @as(usize, W) * H * 4);
    defer testing.allocator.free(old);
    try testing.expect(c.reconlUiRender(ui, old.ptr, old.len) < 0);
}

test "press ownership survives widget ordering and surfaces but not hidden clips" {
    const ui = try createOrSkip();
    defer c.reconlUiDestroy(ui);

    var in = input(20, 60, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiBeginScroll(ui, "clip", 50, 200);
    c.reconlUiSpace(ui, 35);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "hidden", "Hidden", 0));
    c.reconlUiEnd(ui);
    _ = c.reconlUiEndFrame(ui);

    in = input(20, 60, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiBeginScroll(ui, "clip", 50, 200);
    c.reconlUiSpace(ui, 35);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "hidden", "Hidden", 0));
    c.reconlUiEnd(ui);
    _ = c.reconlUiEndFrame(ui);

    // The first widget in draw order owns a simultaneous press; a later hot
    // widget cannot steal its active id before the release frame.
    in = input(20, 10, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "first", "First", 0));
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "second", "Second", 0));
    _ = c.reconlUiEndFrame(ui);
    in = input(20, 10, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 1), c.reconlUiButton(ui, "first", "First", 0));
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "second", "Second", 0));
    _ = c.reconlUiEndFrame(ui);

    // Press the later widget; visiting the earlier widget on release must
    // not consume capture before its owner sees the edge.
    in = input(20, 50, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "ordered-first", "First", 0));
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "ordered-second", "Second", 0));
    _ = c.reconlUiEndFrame(ui);
    in = input(20, 50, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "ordered-first", "First", 0));
    try testing.expectEqual(@as(u32, 1), c.reconlUiButton(ui, "ordered-second", "Second", 0));
    _ = c.reconlUiEndFrame(ui);


    // Abandon a release frame before EndFrame, then start a fresh click.
    in = input(20, 10, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiButton(ui, "abandoned", "Abandoned", 0);
    _ = c.reconlUiEndFrame(ui);
    in = input(20, 10, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    // No EndFrame on the release: the next Begin must release the stale owner.
    in = input(20, 10, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "recovered", "Recovered", 0));
    _ = c.reconlUiEndFrame(ui);
    in = input(20, 10, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 1), c.reconlUiButton(ui, "recovered", "Recovered", 0));
    _ = c.reconlUiEndFrame(ui);

    // If the active widget disappears from the next immediate-mode tree,
    // consume its release so the next, different widget cannot inherit it.
    in = input(10, 10, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiButton(ui, "vanishing", "Vanishing", 0);
    _ = c.reconlUiEndFrame(ui);
    in = input(10, 10, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiEndFrame(ui);
    in = input(10, 10, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "replacement", "Replacement", 0));
    _ = c.reconlUiEndFrame(ui);
    in = input(10, 10, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 1), c.reconlUiButton(ui, "replacement", "Replacement", 0));
    _ = c.reconlUiEndFrame(ui);

    // Off-surface panels still draw (clipped by the renderer), but must not
    // capture a pointer that is outside the actual surface bounds.
    const offscreen = c.ReconLRect{ .x = 300, .y = 0, .w = 100, .h = 80 };
    in = input(350, 10, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    c.reconlUiBeginPanel(ui, &offscreen, 0, 0, c.RECONLUI_PANEL_NO_BORDER);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "offscreen", "Offscreen", 0));
    c.reconlUiEnd(ui);
    _ = c.reconlUiEndFrame(ui);
    in = input(350, 10, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    c.reconlUiBeginPanel(ui, &offscreen, 0, 0, c.RECONLUI_PANEL_NO_BORDER);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "offscreen", "Offscreen", 0));
    c.reconlUiEnd(ui);
    _ = c.reconlUiEndFrame(ui);
}

test "widgets before begin and after endFrame are ignored safely" {
    const ui = try createOrSkip();
    defer c.reconlUiDestroy(ui);
    var v: f32 = 0.5;
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "early", "Early", 0));
    try testing.expectEqual(@as(f32, 0), c.reconlUiLabel(ui, "early", c.RECONLUI_ROLE_BODY, null));
    c.reconlUiBeginRow(ui, 20, 4);
    c.reconlUiSpace(ui, 4);
    _ = c.reconlUiSlider(ui, "early", "Early", &v, 0, 1);
    _ = c.reconlUiEndFrame(ui);
    try testing.expectEqual(@as(u32, 0), c.reconlUiEndFrame(ui));
    var value: u32 = 0;
    try testing.expectEqual(@as(u32, 0), c.reconlUiToggle(ui, "late", "Late", &value));
    try testing.expectEqual(@as(f32, 0), c.reconlUiLabel(ui, "late", c.RECONLUI_ROLE_BODY, null));
}

test "a completed empty begin frame clears the previous mesh" {
    const ui = try createOrSkip();
    defer c.reconlUiDestroy(ui);
    const pixels = try testing.allocator.alloc(u8, @as(usize, W) * H * 4);
    defer testing.allocator.free(pixels);
    const clear = [_]u8{ 0x0B, 0x0D, 0x0C, 0xFF };
    const zero = [_]u8{0} ** 4;

    var in = input(-1e6, -1e6, false, false, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiLabel(ui, "one visible frame", c.RECONLUI_ROLE_BODY, null);
    _ = c.reconlUiEndFrame(ui);
    try testing.expect(c.reconlUiRender(ui, pixels.ptr, pixels.len) == 0);
    const previous = try testing.allocator.dupe(u8, pixels);
    defer testing.allocator.free(previous);
    try testing.expect(!std.mem.eql(u8, previous, &zero));
    try testing.expectEqual(@as(u32, 1), c.reconlUiFrameIndex(ui));

    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 0), c.reconlUiVertexCount(ui));
    try testing.expectEqual(@as(u32, 0), c.reconlUiIndexCount(ui));
    try testing.expectEqual(@as(i32, 0), c.reconlUiRender(ui, pixels.ptr, pixels.len));
    try testing.expectEqual(@as(u32, 2), c.reconlUiFrameIndex(ui));
    var i: usize = 0;
    while (i < pixels.len) : (i += 4) {
        try testing.expectEqualSlices(u8, &clear, pixels[i .. i + 4]);
    }

    try testing.expectEqual(@as(u32, 0), c.reconlUiEndFrame(ui));
    try testing.expectEqual(@as(u32, 0), c.reconlUiVertexCount(ui));
    try testing.expectEqual(@as(i32, 0), c.reconlUiRender(ui, pixels.ptr, pixels.len));
    i = 0;
    while (i < pixels.len) : (i += 4) {
        try testing.expectEqualSlices(u8, &clear, pixels[i .. i + 4]);
    }
}

test "empty, invalid, equal and extreme-range sliders stay finite and bounded" {
    const ui = try createOrSkip();
    defer c.reconlUiDestroy(ui);
    var in = input(160, 10, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiSlider(ui, "empty-range", "Empty", null, 0, 1);
    var v: f32 = 0.4;
    try testing.expectEqual(@as(u32, 0), c.reconlUiSlider(ui, "equal-range", "Equal", &v, 0.4, 0.4));
    try testing.expectEqual(@as(u32, 0), c.reconlUiSlider(ui, "invalid-range", "Invalid", &v, 1, 0));
    _ = c.reconlUiEndFrame(ui);

    in = input(160, 10, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 0), c.reconlUiSlider(ui, "empty-range", "Empty", null, 0, 1));
    try testing.expectEqual(@as(u32, 0), c.reconlUiSlider(ui, "equal-range", "Equal", &v, 0.4, 0.4));
    try testing.expectEqual(@as(u32, 0), c.reconlUiSlider(ui, "invalid-range", "Invalid", &v, 1, 0));
    try testing.expectApproxEqAbs(@as(f32, 0.4), v, 1e-6);
    _ = c.reconlUiEndFrame(ui);

    v = std.math.nan(f32);
    in = input(160, 10, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiSlider(ui, "nan-range", "NaN", &v, 0, 1);
    _ = c.reconlUiEndFrame(ui);
    in = input(160, 10, false, false, true);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiSlider(ui, "nan-range", "NaN", &v, 0, 1);
    _ = c.reconlUiEndFrame(ui);
    try testing.expect(std.math.isFinite(v));
    try testing.expect(v >= 0 and v <= 1);

    // Finite f32 endpoints do not imply their difference fits in f32.
    // Dragging the full representable range to its upper endpoint must stay
    // finite rather than overflowing in min + t * (max - min).
    const limit = std.math.floatMax(f32);
    v = -limit;
    in = input(-1e6, -1e6, false, false, false);
    _ = c.reconlUiBegin(ui, &in);
    _ = c.reconlUiSlider(ui, "wide-range", "Wide", &v, -limit, limit);
    const slider_rect = c.reconlUiLastRect(ui);
    _ = c.reconlUiEndFrame(ui);

    in = input(slider_rect.x + slider_rect.w - 1.0, slider_rect.y + slider_rect.h / 2.0, true, true, false);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 1), c.reconlUiSlider(ui, "wide-range", "Wide", &v, -limit, limit));
    _ = c.reconlUiEndFrame(ui);

    in = input(slider_rect.x + slider_rect.w - 1.0, slider_rect.y + slider_rect.h / 2.0, true, false, false);
    _ = c.reconlUiBegin(ui, &in);
    try testing.expectEqual(@as(u32, 0), c.reconlUiSlider(ui, "wide-range", "Wide", &v, -limit, limit));
    _ = c.reconlUiEndFrame(ui);
    try testing.expect(std.math.isFinite(v));
    try testing.expect(v >= -limit and v <= limit);
    try testing.expect(v > 0);

    // Resize invalidates geometry but must not leave an active press bound to
    // a widget under its old coordinates.
    var capture = input(20, 10, true, true, false);
    _ = c.reconlUiBegin(ui, &capture);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "resize-capture", "Capture", 0));
    _ = c.reconlUiEndFrame(ui);
    try testing.expectEqual(@as(i32, 0), c.reconlUiResize(ui, 640, 360));
    capture = input(20, 10, false, false, true);
    _ = c.reconlUiBegin(ui, &capture);
    try testing.expectEqual(@as(u32, 0), c.reconlUiButton(ui, "resize-capture", "Capture", 0));
    _ = c.reconlUiEndFrame(ui);
}


test "non-finite drawing inputs never poison frame geometry or pixels" {
    const ui = try createOrSkip();
    defer c.reconlUiDestroy(ui);
    const pixels = try testing.allocator.alloc(u8, @as(usize, W) * H * 4);
    defer testing.allocator.free(pixels);

    // NaN clear channels normalize to zero; ordinary finite channels clamp
    // to their RGBA8 range. Empty frames exercise the host-side clear path.
    c.reconlUiSetClear(ui, std.math.nan(f32), 0.5, 2.0, 1.0);
    try testing.expectEqual(@as(u32, 0), c.reconlUiEndFrame(ui));
    try testing.expectEqual(@as(i32, 0), c.reconlUiRender(ui, pixels.ptr, pixels.len));
    var i: usize = 0;
    while (i < pixels.len) : (i += 4) {
        try testing.expectEqualSlices(u8, &[_]u8{ 0, 128, 255, 255 }, pixels[i .. i + 4]);
    }

    c.reconlUiSetClear(ui, 0.0, 0.0, 0.0, 1.0);
    var in = input(-1e6, -1e6, false, false, false);
    _ = c.reconlUiBegin(ui, &in);
    const bad_rect = c.ReconLRect{ .x = std.math.nan(f32), .y = 8, .w = 100, .h = 80 };
    c.reconlUiBeginPanel(ui, &bad_rect, 4, 4, 0);
    _ = c.reconlUiLabel(ui, "panel", c.RECONLUI_ROLE_BODY, null);
    c.reconlUiEnd(ui);
    c.reconlUiBeginRow(ui, std.math.nan(f32), std.math.nan(f32));
    _ = c.reconlUiButton(ui, "row-button", "Button", 0);
    c.reconlUiEnd(ui);
    c.reconlUiSpace(ui, std.math.nan(f32));
    c.reconlUiProgress(ui, "progress", std.math.nan(f32), 8);
    const bad_samples = [_]f32{ 0.1, std.math.nan(f32), 0.9 };
    c.reconlUiSparkline(ui, &bad_samples, bad_samples.len, 24);
    // A finite f32 extent can still overflow when multiple slots accumulate.
    c.reconlUiProgress(ui, "large-a", 0.5, std.math.floatMax(f32));
    c.reconlUiProgress(ui, "large-b", 0.5, std.math.floatMax(f32));
    const bad_color = [_]f32{ std.math.nan(f32), 0.5, 0.5, 1.0 };
    _ = c.reconlUiLabel(ui, "color", c.RECONLUI_ROLE_BODY, &bad_color);
    _ = c.reconlUiEndFrame(ui);

    const vertices = c.reconlUiVertices(ui);
    try testing.expect(vertices != null);
    const vertex_count = c.reconlUiVertexCount(ui);
    const vertex_words: [*]const f32 = @ptrCast(@alignCast(vertices.?));
    var word: usize = 0;
    while (word < @as(usize, vertex_count) * 12) : (word += 1) {
        try testing.expect(std.math.isFinite(vertex_words[word]));
    }
    try testing.expectEqual(@as(i32, 0), c.reconlUiRender(ui, pixels.ptr, pixels.len));
}

test "a rejected scroll begin does not leave an extra clip or layout slot" {
    const ui = try createOrSkip();
    defer c.reconlUiDestroy(ui);
    var in = input(-1e6, -1e6, false, false, false);
    _ = c.reconlUiBegin(ui, &in);

    const invalid = c.reconlUiBeginScroll(ui, "bad-scroll", std.math.nan(f32), 100);
    try testing.expectEqual(@as(f32, 0), invalid.w);
    const viewport = c.reconlUiBeginScroll(ui, "good-scroll", 40, 80);
    try testing.expectEqual(@as(f32, 40), viewport.h);
    _ = c.reconlUiLabel(ui, "inside", c.RECONLUI_ROLE_BODY, null);
    c.reconlUiEnd(ui);
    _ = c.reconlUiLabel(ui, "outside", c.RECONLUI_ROLE_BODY, null);
    _ = c.reconlUiEndFrame(ui);

    const vertices = c.reconlUiVertices(ui);
    try testing.expect(vertices != null);
    const count = c.reconlUiVertexCount(ui);
    const words: [*]const f32 = @ptrCast(@alignCast(vertices.?));
    var i: usize = 0;
    while (i < @as(usize, count) * 12) : (i += 1) {
        try testing.expect(std.math.isFinite(words[i]));
    }
    const pixels = try testing.allocator.alloc(u8, @as(usize, W) * H * 4);
    defer testing.allocator.free(pixels);
    try testing.expectEqual(@as(i32, 0), c.reconlUiRender(ui, pixels.ptr, pixels.len));
}
