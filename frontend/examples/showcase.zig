//! The animated showcase: a small control-room dashboard built with the
//! widget layer, rendered *through* ReconL, written out as a PNG sequence.
//!
//! Run it with `zig build demo`. It writes `zig-out/showcase/frame_NNN.png`
//! (60 frames at 960x540) - flip through them and the hover ramps, press
//! dips, the toggle spring, the slider drag and the scroll spring are all
//! there, because every one of them is a function of `dt_ms` and the
//! scripted input, never of a wall clock. The pointer plays a fixed
//! timeline: hover the buttons, click Deploy, flip Live, drag Quality to
//! full, wheel down the activity log, drift back.
//!
//! The script targets widgets by the rects the first frame records
//! (`ui.last_rect`), so the demo stays correct if the layout shifts - it
//! never hard-codes a hit position.

const std = @import("std");
const front = @import("reconl-frontend");
const backend = @import("reconl-backend");

const geom = front.geom;
const theme_mod = front.theme;
const ui_mod = front.ui;
const font_mod = front.font;
const png = front.png;
const Rect = geom.Rect;
const Vec2 = geom.Vec2;
const Color = geom.Color;

const W: u32 = 960;
const H: u32 = 540;
const FRAMES: u32 = 60;
const DT: f32 = 1000.0 / 60.0; // 60 Hz, the dt every animation integrates
const OUT_DIR = "zig-out/showcase";

/// The rects the widgets claimed on the first (pointer-less) frame; the
/// script aims at these rather than at magic coordinates.
const Targets = struct {
    deploy: Rect = .{ .x = 0, .y = 0, .w = 0, .h = 0 },
    live: Rect = .{ .x = 0, .y = 0, .w = 0, .h = 0 },
    quality: Rect = .{ .x = 0, .y = 0, .w = 0, .h = 0 },
    log: Rect = .{ .x = 0, .y = 0, .w = 0, .h = 0 },
};

fn center(r: Rect) Vec2 {
    return .{ .x = r.x + r.w * 0.5, .y = r.y + r.h * 0.5 };
}

fn mix(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

fn mix2(a: Vec2, b: Vec2, t: f32) Vec2 {
    return .{ .x = mix(a.x, b.x, t), .y = mix(a.y, b.y, t) };
}

/// Smoothstep 0..1 across frame window [f0, f1], clamped outside it.
fn ease(f: u32, f0: u32, f1: u32) f32 {
    if (f <= f0) return 0.0;
    if (f >= f1) return 1.0;
    const x = @as(f32, @floatFromInt(f - f0)) / @as(f32, @floatFromInt(f1 - f0));
    return x * x * (3.0 - 2.0 * x);
}

/// The pointer's scripted waypoint tour.
fn pointerAt(f: u32, t: Targets) Vec2 {
    const start = Vec2{ .x = 60, .y = 60 };
    const deploy = center(t.deploy);
    const live = center(t.live);
    const log = center(t.log);
    const q0 = Vec2{ .x = t.quality.x + 12, .y = center(t.quality).y };
    const q1 = Vec2{ .x = t.quality.right() - 12, .y = center(t.quality).y };

    if (f <= 2) return start;
    if (f <= 10) return mix2(start, deploy, ease(f, 2, 10)); // arrive to click Deploy
    if (f <= 18) return mix2(deploy, live, ease(f, 12, 18)); // then to the toggle
    if (f <= 25) return mix2(live, q0, ease(f, 20, 25)); // then to the slider's left
    if (f <= 41) return mix2(q0, q1, ease(f, 26, 40)); // drag it to full, release
    if (f <= 44) return mix2(q1, log, ease(f, 42, 44)); // to the activity log
    if (f <= 58) return mix2(log, deploy, ease(f, 51, 58)); // drift back to Deploy
    return deploy;
}

fn scriptedInput(f: u32, t: Targets) ui_mod.Input {
    const p = pointerAt(f, t);
    const down = (f >= 10 and f <= 11) or (f >= 18 and f <= 19) or (f >= 25 and f <= 41);
    const pressed = f == 10 or f == 18 or f == 25;
    const released = f == 11 or f == 19 or f == 41;
    const wheel: f32 = if (f >= 44 and f <= 50) 1.0 else 0.0;
    return .{
        .px = p.x,
        .py = p.y,
        .down = down,
        .pressed = pressed,
        .released = released,
        .wheel = wheel,
        .dt_ms = DT,
    };
}

pub fn main() !void {
    var dbg: std.heap.DebugAllocator(.{}) = .init;
    defer {
        if (dbg.deinit() == .leak) std.debug.print("showcase: LEAKED MEMORY\n", .{});
    }
    const gpa = dbg.allocator();

    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.Io.Dir.cwd();
    try dir.createDirPath(io, OUT_DIR);

    var font_a = try font_mod.loadBundled(gpa, "Inter-Variable.ttf");
    defer font_a.deinit();
    var font_b = try font_mod.loadBundled(gpa, "Outfit-Variable.ttf");
    defer font_b.deinit();

    const theme = theme_mod.default;
    var ui = ui_mod.Ui.init(
        gpa,
        .{ .ui = &font_a, .display = &font_b },
        theme,
        @floatFromInt(W),
        @floatFromInt(H),
    );
    defer ui.deinit();

    var renderer = backend.Renderer.init(gpa, W, H, theme.bg) catch |e| {
        const msg = backend.createError();
        std.debug.print("showcase: renderer init failed: {s}\n", .{if (msg.len > 0) msg else @errorName(e)});
        return e;
    };
    defer renderer.deinit();

    const pixels = try gpa.alloc(u8, @as(usize, W) * H * 4);
    defer gpa.free(pixels);

    var targets = Targets{};
    var on = false;
    var quality: f32 = 0.35;
    var samples = [_]f32{0.5} ** 48;

    var f: u32 = 0;
    while (f < FRAMES) : (f += 1) {
        // Frame 0 runs pointer-less; it records where every widget landed
        // and warms the animation states from their rest values.
        const input: ui_mod.Input = if (f == 0) .{ .dt_ms = DT } else scriptedInput(f, targets);
        try ui.begin(input);

        try ui.beginPanel(
            .{ .x = 16, .y = 16, .w = @floatFromInt(W - 32), .h = @floatFromInt(H - 32) },
            .{ .pad = 20, .gap = 12, .shadow = true },
        );

        _ = try ui.label("ReconL Studio", .title, null, .{});
        _ = try ui.label(
            "every widget below is tessellated by the frontend and drawn by ReconL itself",
            .label,
            theme.text_muted,
            .{},
        );
        try ui.separator();

        try ui.beginRow(34, 8);
        _ = try ui.button("deploy", "Deploy", .{ .primary = true });
        if (f == 0) targets.deploy = ui.last_rect;
        _ = try ui.button("cancel", "Cancel", .{});
        _ = try ui.toggle("live", "Live updates", &on);
        if (f == 0) targets.live = ui.last_rect;
        ui.end(); // row

        _ = try ui.slider("quality", "Quality", &quality, 0, 1);
        if (f == 0) targets.quality = ui.last_rect;

        try ui.progress("build", @as(f32, @floatFromInt(f + 1)) / @as(f32, @floatFromInt(FRAMES)), 8);

        // The trend line: a slow sine with a pulse on every scripted click.
        std.mem.copyForwards(f32, samples[0..47], samples[1..48]);
        const tt = @as(f32, @floatFromInt(f));
        const pulse: f32 = if (f == 11 or f == 19 or f == 41) 0.35 else 0.0;
        samples[47] = std.math.clamp(0.5 + 0.4 * @sin(tt * 0.31) + pulse, 0.0, 1.0);
        try ui.sparkline(&samples, 44);

        try ui.separator();
        _ = try ui.label("Activity", .heading, null, .{});

        // Scroll viewport sized to whatever room is left - no magic numbers
        // to drift when the type scale changes.
        const room = @max(60.0, ui.remaining() - 4);
        _ = try ui.beginScroll("log", room, 420);
        if (f == 0) targets.log = ui.last_rect;
        var i: usize = 0;
        while (i < 14) : (i += 1) {
            var line_buf: [72]u8 = undefined;
            const line = try std.fmt.bufPrint(
                &line_buf,
                "09:{d:0>2}:14  frame {d} rendered {d} triangles",
                .{ i, (f * 7 +% @as(u32, @intCast(i)) * 13) % 1000, 1200 + i * 37 },
            );
            const col: Color = if (i + 4 >= 14) theme.text else theme.text_dim;
            _ = try ui.label(line, .label, col, .{});
        }
        ui.end(); // scroll
        ui.end(); // panel

        const mesh = ui.endFrame();
        try renderer.render(mesh, pixels);

        {
            var aw: std.Io.Writer.Allocating = .init(gpa);
            defer aw.deinit();
            try png.encode(gpa, W, H, pixels, &aw.writer);
            var name_buf: [256]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "{s}/frame_{d:0>3}.png", .{ OUT_DIR, f });
            try dir.writeFile(io, .{ .sub_path = name, .data = aw.writer.buffered() });
        }
        if (f % 10 == 0) std.debug.print("frame {d}/{d}\n", .{ f + 1, FRAMES });
    }

    std.debug.print(
        "showcase: {d} frames written to {s}/ ({}x{}, {d} bytes each)\n",
        .{ FRAMES, OUT_DIR, W, H, @as(usize, W) * H * 4 },
    );
}
