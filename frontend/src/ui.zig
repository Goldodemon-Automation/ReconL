//! Immediate-mode widgets: build a frame, get interaction back in the same
//! call. No retained widget tree, no hidden state beyond the small per-id
//! animation state (hover ramps, knob springs, scroll springs) that survives
//! frames so motion is continuous.
//!
//! Layout is cursor-based like the rest of the immediate-mode world: a frame
//! owns a content box, children claim slots along the main axis, `gap`
//! spaces them. Columns stretch children to full width; rows size children
//! by their content (or an explicit width). That is enough for every widget
//! below and for panels composed from them, and it never needs a second
//! layout pass - the slot a widget paints *is* the slot it hit-tests.
//!
//! Identity is imgui-style: `fnv1a(scope_seed, label)`, with `##` splitting
//! display text from id text (`"Run##once"` displays `Run`, ids
//! independently of any other `Run`). Two identical labels in one scope
//! share hover state - disambiguate with `##` rather than adding an id
//! registry.
//!
//! Everything animates *toward* its target at a rate derived from
//! `input.dt_ms`, never from a clock: run the same inputs twice and the same
//! frames come out, which is what makes the showcase diffable against the
//! renderer's own determinism rule.

const std = @import("std");
const geom = @import("geom.zig");
const Rect = geom.Rect;
const Vec2 = geom.Vec2;
const Color = geom.Color;
const theme_mod = @import("theme.zig");
const Theme = theme_mod.Theme;
const Role = theme_mod.Role;
const anim = @import("anim.zig");
const tess = @import("tess.zig");
const text = @import("text.zig");
const font_mod = @import("font.zig");
const Font = font_mod.Font;

/// The two families the type scale names; `mono` falls back to `ui` until a
/// monospace face ships in `assets/fonts/`.
pub const Fonts = struct {
    ui: *Font,
    display: *Font,
};

pub const Input = struct {
    px: f32 = -1e6,
    py: f32 = -1e6,
    down: bool = false,
    pressed: bool = false, // went down during this frame
    released: bool = false, // went up during this frame
    wheel: f32 = 0.0, // + = scrolled down, in notches
    dt_ms: f32 = 1000.0 / 60.0,
};

pub const Dir = enum { col, row };

/// A finished frame: slices into the Ui's own buffers, valid until the next
/// `begin`.
pub const Mesh = struct {
    vertices: []const tess.Vertex,
    indices: []const u32,
};

/// Per-id animation state. Springs start at 0 and chase their target each
/// frame; ramps move linearly at `theme.duration_*_ms`.
pub const WidgetState = struct {
    hover: f32 = 0.0,
    press: f32 = 0.0,
    knob: anim.Spring = .{},
    scroll: anim.Spring = .{}, // normalized offset in [-1, 0], scaled by the current content range
    value: f32 = 0.0, // e.g. progress smoothing
};

const Frame = struct {
    content: Rect, // box after padding
    dir: Dir,
    cursor: f32, // main-axis position, relative to content origin
    gap: f32,
    seed: u64, // id scope
    clip: bool, // whether this frame pushed a clip rect
};

pub const PanelOpts = struct {
    dir: Dir = .col,
    pad: f32 = 16.0,
    gap: f32 = 8.0,
    radius: f32 = -1.0, // < 0 = theme.radius_l
    bg: ?Color = null, // null = theme.surface
    border: bool = true,
    shadow: bool = false,
    /// Frame seed scope; "" derives one from nesting depth (stable while
    /// the widget tree's shape is stable).
    id: []const u8 = "",
};

pub const ButtonOpts = struct {
    primary: bool = false,
    height: f32 = 34.0,
    width: f32 = 0.0, // row mode: 0 = hug the label
    radius: f32 = -1.0,
    font_px: f32 = 0.0, // 0 = theme.font_body
};

pub const LabelOpts = struct {
    max_width: f32 = 0.0, // 0 = the frame's content width (col) or unwrapped (row)
    alignment: text.Align = .left,
    line_height: f32 = 1.25,
};

pub const Interaction = struct {
    hot: bool, // pointer is over the rect
    held: bool, // this widget owns the press
    clicked: bool, // released over it this frame
    state: *WidgetState,
};

pub const Ui = struct {
    gpa: std.mem.Allocator,
    theme: Theme,
    fonts: Fonts,
    dl: tess.DisplayList,
    input: Input,
    bounds: Rect,
    frames: std.ArrayList(Frame),
    states: std.AutoHashMap(u64, WidgetState),
    active_id: u64 = 0,
    /// Small formatting scratch for value readouts (slider labels etc).
    scratch_buf: [128]u8 = undefined,
    /// The rect the most recent slot claimed (imgui's `lastItemRect`):
    /// tooltips, carets and scripted demos read it after the widget call.
    last_rect: Rect = .{ .x = 0, .y = 0, .w = 0, .h = 0 },
    frame_open: bool = false,

    pub fn init(gpa: std.mem.Allocator, fonts: Fonts, theme: Theme, w: f32, h: f32) Ui {
        return .{
            .gpa = gpa,
            .fonts = fonts,
            .theme = theme,
            .dl = tess.DisplayList.init(),
            .input = .{},
            .bounds = .{ .x = 0, .y = 0, .w = w, .h = h },
            .frames = .empty,
            .states = std.AutoHashMap(u64, WidgetState).init(gpa),
        };
    }

    pub fn deinit(ui: *Ui) void {
        ui.dl.deinit(ui.gpa);
        ui.frames.deinit(ui.gpa);
        ui.states.deinit();
    }

    /// Discards the in-progress or frozen frame without dropping reusable capacity.
    pub fn discardFrame(ui: *Ui) void {
        ui.dl.clear();
        ui.frames.clearRetainingCapacity();
        ui.last_rect = .{ .x = 0, .y = 0, .w = 0, .h = 0 };
        ui.frame_open = false;
    }

    pub fn setSize(ui: *Ui, w: f32, h: f32) void {
        ui.bounds = .{ .x = 0, .y = 0, .w = w, .h = h };
    }

    /// Starts a frame: clears geometry, pushes the root frame covering the
    /// whole surface. Every `beginPanel`/`beginRow`/`beginScroll` nests
    /// inside it.
    pub fn begin(ui: *Ui, input: Input) !void {
        // A new Begin abandons any still-open frame. If that frame carried a
        // release, no widget or endFrame consumed it, so drop the stale capture.
        const abandoned_release = ui.frame_open and ui.input.released;
        ui.input = input;
        ui.discardFrame();
        // Otherwise, preserve capture through the release frame so the owning
        // widget can consume it; clear only when the host never reported release.
        if (abandoned_release or (ui.active_id != 0 and !input.down and !input.released)) ui.active_id = 0;
        try ui.frames.append(ui.gpa, .{
            .content = ui.bounds,
            .dir = .col,
            .cursor = 0.0,
            .gap = 0.0,
            .seed = fnv1a(ROOT_SEED, ""),
            .clip = false,
        });
        ui.frame_open = true;
    }

    /// Finishes the frame, closing anything a host left open (each container
    /// pops its own clip on the way out) and returns the mesh - valid until
    /// the next `begin`.
    pub fn endFrame(ui: *Ui) Mesh {
        while (ui.frames.items.len > 1) ui.end();
        if (ui.frames.items.len == 1 and ui.frames.items[0].clip) ui.dl.popClip();
        ui.frames.clearRetainingCapacity();
        // A release is consumed by a widget when present; if its widget was
        // omitted from this immediate-mode tree, do not carry capture on.
        if (ui.input.released) ui.active_id = 0;
        ui.frame_open = false;
        return .{ .vertices = ui.dl.vertices.items, .indices = ui.dl.indices.items };
    }

    /// Pops one container (`beginPanel`/`beginRow`/`beginScroll`), releasing
    /// its clip if it had one. The root frame cannot be popped.
    pub fn end(ui: *Ui) void {
        if (ui.frames.items.len <= 1) return;
        const f = ui.frames.items[ui.frames.items.len - 1];
        ui.frames.items.len -= 1;
        if (f.clip) ui.dl.popClip();
    }

    // ---- layout ---------------------------------------------------------

    fn top(ui: *Ui) *Frame {
        std.debug.assert(ui.frame_open and ui.frames.items.len > 0);
        return &ui.frames.items[ui.frames.items.len - 1];
    }

    fn slotCol(ui: *Ui, h: f32) Rect {
        const f = ui.top();
        const r = Rect{ .x = f.content.x, .y = saturatedAdd(f.content.y, f.cursor), .w = f.content.w, .h = h };
        f.cursor = saturatedAdd(saturatedAdd(f.cursor, h), f.gap);
        return r;
    }

    fn slotRow(ui: *Ui, w: f32) Rect {
        const f = ui.top();
        const r = Rect{ .x = saturatedAdd(f.content.x, f.cursor), .y = f.content.y, .w = w, .h = f.content.h };
        f.cursor = saturatedAdd(saturatedAdd(f.cursor, w), f.gap);
        return r;
    }

    fn slot(ui: *Ui, w: f32, h: f32) Rect {
        const r = switch (ui.top().dir) {
            .col => ui.slotCol(h),
            .row => ui.slotRow(w),
        };
        ui.last_rect = r;
        return r;
    }

    pub fn remaining(ui: *Ui) f32 {
        const f = ui.top();
        const total = if (f.dir == .col) f.content.h else f.content.w;
        return @max(0.0, total - f.cursor);
    }

    pub fn beginPanel(ui: *Ui, rect: Rect, opts: PanelOpts) !void {
        const radius = if (opts.radius < 0.0) ui.theme.radius_l else opts.radius;
        const bg = opts.bg orelse ui.theme.surface;

        if (opts.shadow) {
            var i: usize = 1;
            while (i <= 7) : (i += 1) {
                const t = @as(f32, @floatFromInt(i)) / 7.0;
                const off = 1.0 + t * 6.0;
                const c = Color.rgba8(0, 0, 0, @intFromFloat(22.0 * (1.0 - t)));
                try ui.dl.roundedRect(ui.gpa, Rect{ .x = rect.x, .y = rect.y + off, .w = rect.w, .h = rect.h }, radius, c);
            }
        }
        try ui.dl.roundedRect(ui.gpa, rect, radius, bg);
        if (opts.border) try ui.dl.stroke(ui.gpa, rect, radius, 1.0, ui.theme.border);
        try ui.dl.pushClip(ui.gpa, rect);
        errdefer ui.dl.popClip();
        const seed = if (opts.id.len > 0) fnv1a(ui.top().seed, opts.id) else fnv1a(ui.top().seed, "#panel");
        try ui.frames.append(ui.gpa, .{
            .content = rect.inset(opts.pad),
            .dir = opts.dir,
            .cursor = 0.0,
            .gap = opts.gap,
            .seed = seed,
            .clip = true,
        });
    }

    pub fn beginRow(ui: *Ui, height: f32, gap: f32) !void {
        const parent = ui.top();
        const cursor_before = parent.cursor;
        const rect = ui.slotCol(height);
        errdefer ui.top().cursor = cursor_before;
        try ui.frames.append(ui.gpa, .{
            .content = rect,
            .dir = .row,
            .cursor = 0.0,
            .gap = gap,
            .seed = fnv1a(parent.seed, "#row"),
            .clip = false,
        });
    }

    fn stateFor(ui: *Ui, id: u64) !*WidgetState {
        const gop = try ui.states.getOrPut(id);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        return gop.value_ptr;
    }

    fn hitTest(ui: *Ui, rect: Rect) bool {
        const clip = Rect.intersect(ui.dl.activeClip(ui.bounds), ui.bounds);
        const visible = Rect.intersect(rect, clip);
        return visible.contains(.{ .x = ui.input.px, .y = ui.input.py });
    }

    pub fn interact(ui: *Ui, id_str: []const u8, rect: Rect) !Interaction {
        const id = fnv1a(ui.top().seed, id_str);
        const st = try ui.stateFor(id);
        const hot = ui.hitTest(rect);

        var clicked = false;
        if (ui.input.pressed and hot and ui.active_id == 0) ui.active_id = id;
        const held = ui.active_id == id and ui.input.down;
        if (ui.input.released and ui.active_id == id) {
            clicked = hot;
            ui.active_id = 0;
        }

        const hot_f: f32 = if (hot) 1.0 else 0.0;
        const held_f: f32 = if (held) 1.0 else 0.0;
        st.hover = easeTo(st.hover, hot_f, ui.input.dt_ms, @floatFromInt(ui.theme.duration_hover_ms));
        st.press = easeTo(st.press, held_f, ui.input.dt_ms, @floatFromInt(ui.theme.duration_press_ms));
        return .{ .hot = hot, .held = held, .clicked = clicked, .state = st };
    }

    pub fn label(ui: *Ui, str: []const u8, role: Role, color: ?Color, opts: LabelOpts) !f32 {
        const f = ui.top();
        const font = ui.fontFor(role);
        const px = theme_mod.font_size(ui.theme, role);
        const wrap: text.Options = switch (f.dir) {
            .col => .{ .max_width = if (opts.max_width > 0) opts.max_width else f.content.w, .alignment = opts.alignment, .line_height = opts.line_height },
            .row => .{ .max_width = opts.max_width, .alignment = opts.alignment, .line_height = opts.line_height },
        };
        const m = text.measure(font, str, px, wrap);
        const h = if (m.lines == 0) font.lineHeight(px) * opts.line_height else m.height;
        const w: f32 = if (f.dir == .col) f.content.w else @max(m.width, 1.0);
        const r = ui.slot(w, h);
        _ = try text.draw(&ui.dl, ui.gpa, font, str, r, px, color orelse ui.theme.text, wrap);
        return h;
    }

    pub fn button(ui: *Ui, id_str: []const u8, str: []const u8, opts: ButtonOpts) !bool {
        const f = ui.top();
        const px = if (opts.font_px > 0) opts.font_px else ui.theme.font_body;
        const text_w = text.measure(ui.fonts.ui, str, px, .{}).width;
        const w: f32 = if (f.dir == .col) f.content.w else if (opts.width > 0) opts.width else text_w + 32.0;
        const r = ui.slot(w, opts.height);
        const ix = try ui.interact(id_str, r);
        const radius = if (opts.radius < 0.0) ui.theme.radius_m else opts.radius;
        var bg: Color = undefined;
        var fg: Color = undefined;
        var border: Color = undefined;
        if (opts.primary) {
            var base = Color.lerp(ui.theme.accent, ui.theme.accent_hover, ix.state.hover);
            base = Color.lerp(base, ui.theme.accent_pressed, ix.state.press);
            bg = base;
            fg = ui.theme.on_accent;
            border = Color.lerp(ui.theme.border_strong, ui.theme.accent_hover, ix.state.hover);
        } else {
            bg = Color.lerp(ui.theme.surface_raised, ui.theme.accent, ix.state.hover * 0.22);
            bg = Color.lerp(bg, ui.theme.bg, ix.state.press * 0.5);
            fg = Color.lerp(ui.theme.text, ui.theme.accent_hover, ix.state.hover);
            border = Color.lerp(ui.theme.border, ui.theme.accent, ix.state.hover);
        }
        try ui.dl.roundedRect(ui.gpa, r, radius, bg);
        try ui.dl.stroke(ui.gpa, r, radius, 1.0, border);
        try ui.centeredText(str, r, px, fg);
        return ix.clicked;
    }

    pub fn toggle(ui: *Ui, id_str: []const u8, str: []const u8, value: *bool) !bool {
        const f = ui.top();
        const px = ui.theme.font_body;
        const track_w: f32 = 44.0;
        const track_h: f32 = 24.0;
        const h: f32 = @max(track_h, theme_mod.font_size(ui.theme, .body) * 1.4);
        const w: f32 = if (f.dir == .col) f.content.w else text.measure(ui.fonts.ui, str, px, .{}).width + 12.0 + track_w;
        const r = ui.slot(w, h);
        const ix = try ui.interact(id_str, r);
        const changed = ix.clicked;
        if (changed) value.* = !value.*;
        const st = ix.state;
        st.knob.target = if (value.*) 1.0 else 0.0;
        st.knob.stiffness = ui.theme.spring_stiffness;
        st.knob.damping = ui.theme.spring_damping;
        st.knob.step(ui.input.dt_ms / 1000.0);
        const k = std.math.clamp(st.knob.value, 0.0, 1.0);
        const track_x = r.right() - track_w;
        const track = Rect{ .x = track_x, .y = r.y + (r.h - track_h) * 0.5, .w = track_w, .h = track_h };
        const label_rect = Rect{ .x = r.x, .y = r.y, .w = @max(0.0, track.x - 8.0 - r.x), .h = r.h };
        _ = try text.draw(&ui.dl, ui.gpa, ui.fonts.ui, str, label_rect, px, ui.theme.text, .{
            .max_width = label_rect.w,
            .line_height = 1.0,
        });
        const track_bg = Color.lerp(ui.theme.surface_raised, ui.theme.accent, k);
        try ui.dl.roundedRect(ui.gpa, track, ui.theme.radius_pill, track_bg);
        try ui.dl.stroke(ui.gpa, track, ui.theme.radius_pill, 1.0, Color.lerp(ui.theme.border, ui.theme.accent, k));
        const knob_r = track_h - 6.0;
        const travel = track_w - knob_r - 6.0;
        const knob = Rect{ .x = track.x + 3.0 + travel * k, .y = track.y + 3.0, .w = knob_r, .h = knob_r };
        try ui.dl.roundedRect(ui.gpa, knob, ui.theme.radius_pill, ui.theme.text);
        return changed;
    }

    pub fn slider(ui: *Ui, id_str: []const u8, str: []const u8, value: *f32, min: f32, max: f32) !bool {
        const f = ui.top();
        const px = ui.theme.font_label;
        const label_h = theme_mod.font_size(ui.theme, .label) * 1.4;
        const track_h: f32 = 6.0;
        const gap: f32 = 8.0;
        const total_h = label_h + gap + track_h;
        const w: f32 = if (f.dir == .col) f.content.w else @max(160.0, f.content.w * 0.5);
        const r = ui.slot(w, total_h);

        const min64: f64 = min;
        const max64: f64 = max;
        const span = max64 - min64;
        var t: f32 = if (span > 0.0)
            @floatCast(std.math.clamp((@as(f64, value.*) - min64) / span, 0.0, 1.0))
        else
            0.0;

        const track = Rect{ .x = r.x, .y = r.y + label_h + gap, .w = r.w, .h = track_h };
        const ix = try ui.interact(id_str, r);
        var changed = false;
        if (ix.held) {
            const nt: f32 = if (span > 0.0)
                @floatCast(std.math.clamp(
                    (@as(f64, ui.input.px) - @as(f64, track.x)) / @max(@as(f64, track.w), 1.0),
                    0.0,
                    1.0,
                ))
            else
                0.0;
            const nv: f32 = @floatCast(min64 + @as(f64, nt) * span);
            if (@abs(nv - value.*) > 1e-6) {
                value.* = nv;
                t = nt;
                changed = true;
            }
        }

        const val_str = try std.fmt.bufPrint(&ui.scratch_buf, "{d:.2}", .{value.*});
        _ = try text.draw(&ui.dl, ui.gpa, ui.fonts.ui, str, Rect{ .x = r.x, .y = r.y, .w = r.w * 0.7, .h = label_h }, px, ui.theme.text_muted, .{ .line_height = 1.0 });
        const vw = text.measure(ui.fonts.ui, val_str, px, .{}).width;
        _ = try text.draw(&ui.dl, ui.gpa, ui.fonts.ui, val_str, Rect{ .x = r.right() - vw, .y = r.y, .w = vw, .h = label_h }, px, ui.theme.text, .{ .line_height = 1.0 });
        try ui.dl.roundedRect(ui.gpa, track, ui.theme.radius_pill, ui.theme.surface_raised);
        const fill = Rect{ .x = track.x, .y = track.y, .w = track.w * t, .h = track.h };
        try ui.dl.roundedRect(ui.gpa, fill, ui.theme.radius_pill, ui.theme.accent);
        ix.state.knob.target = t;
        ix.state.knob.stiffness = ui.theme.spring_stiffness;
        ix.state.knob.damping = ui.theme.spring_damping;
        ix.state.knob.step(ui.input.dt_ms / 1000.0);
        const kt = std.math.clamp(ix.state.knob.value, 0.0, 1.0);
        const kr: f32 = 8.0;
        const knob = Rect{
            .x = track.x + track.w * kt - kr,
            .y = track.y + track.h * 0.5 - kr,
            .w = kr * 2.0,
            .h = kr * 2.0,
        };
        try ui.dl.roundedRect(ui.gpa, knob, ui.theme.radius_pill, ui.theme.text);
        try ui.dl.stroke(ui.gpa, knob, ui.theme.radius_pill, 1.0, ui.theme.border_strong);
        return changed;
    }

    pub fn progress(ui: *Ui, id_str: []const u8, t: f32, height: f32) !void {
        const f = ui.top();
        const w: f32 = if (f.dir == .col) f.content.w else @max(120.0, f.content.w * 0.4);
        const r = ui.slot(w, height);
        const target = std.math.clamp(t, 0.0, 1.0);
        const st = try ui.stateFor(fnv1a(f.seed, id_str));
        st.value = easeTo(st.value, target, ui.input.dt_ms, 300.0);
        try ui.dl.roundedRect(ui.gpa, r, ui.theme.radius_pill, ui.theme.surface_raised);
        if (st.value > 0.001) {
            const fill = Rect{ .x = r.x, .y = r.y, .w = r.w * st.value, .h = r.h };
            try ui.dl.roundedRectGrad(ui.gpa, fill, ui.theme.radius_pill, ui.theme.accent_hover, ui.theme.accent);
        }
    }

    pub fn separator(ui: *Ui) !void {
        const f = ui.top();
        if (f.dir == .col) try ui.dl.rect(ui.gpa, ui.slotCol(1.0), ui.theme.border) else try ui.dl.rect(ui.gpa, ui.slotRow(1.0), ui.theme.border);
    }

    pub fn space(ui: *Ui, amount: f32) !void {
        const f = ui.top();
        if (f.dir == .col) _ = ui.slotCol(amount) else _ = ui.slotRow(amount);
    }

    /// A clipped scrolling region: returns the viewport rect. Children are
    /// laid out in the spring-eased content space and clipped to that viewport;
    /// call `end` when done.
    pub fn beginScroll(ui: *Ui, id_str: []const u8, height: f32, content_h: f32) !Rect {
        const f = ui.top();
        const w: f32 = if (f.dir == .col) f.content.w else @max(160.0, f.content.w * 0.5);
        const cursor_before = f.cursor;
        const r = ui.slot(w, height);
        errdefer ui.top().cursor = cursor_before;
        const id = fnv1a(f.seed, id_str);
        const st = try ui.stateFor(id);
        const hot = ui.hitTest(r);
        const max_scroll = @max(0.0, content_h - r.h);
        if (hot and max_scroll > 0.0 and std.math.isFinite(ui.input.wheel) and ui.input.wheel != 0.0) {
            const delta = @as(f64, ui.input.wheel) * 48.0 / @as(f64, max_scroll);
            st.scroll.target = @floatCast(std.math.clamp(@as(f64, st.scroll.target) - delta, -1.0, 0.0));
        }
        if (max_scroll <= 0.0) st.scroll.target = 0.0;
        st.scroll.target = std.math.clamp(st.scroll.target, -1.0, 0.0);
        st.scroll.stiffness = ui.theme.spring_stiffness * 1.4;
        st.scroll.damping = ui.theme.spring_damping * 1.25;
        st.scroll.step(ui.input.dt_ms / 1000.0);
        const off = std.math.clamp(st.scroll.value, -1.0, 0.0) * max_scroll;

        try ui.dl.pushClip(ui.gpa, r);
        errdefer ui.dl.popClip();
        try ui.frames.append(ui.gpa, .{
            .content = Rect{ .x = r.x, .y = r.y + off, .w = r.w, .h = content_h },
            .dir = .col,
            .cursor = 0.0,
            .gap = f.gap,
            .seed = fnv1a(id, "#scroll"),
            .clip = true,
        });
        return r;
    }

    pub fn sparkline(ui: *Ui, values: []const f32, height: f32) !void {
        const f = ui.top();
        const w: f32 = if (f.dir == .col) f.content.w else @max(120.0, f.content.w * 0.3);
        const r = ui.slot(w, height);
        if (values.len < 2) return;
        var lo: f64 = values[0];
        var hi: f64 = values[0];
        for (values[1..]) |v| {
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const range = @max(hi - lo, 1e-6);
        const xstep = r.w / @as(f32, @floatFromInt(values.len - 1));
        const area = Color.lerp(ui.theme.accent, ui.theme.bg, 0.72);
        var i: usize = 0;
        while (i + 1 < values.len) : (i += 1) {
            const a = sparkPt(r, values, i, xstep, lo, range);
            const b = sparkPt(r, values, i + 1, xstep, lo, range);
            try ui.dl.quad(ui.gpa, .{ a, b, .{ .x = b.x, .y = r.bottom() }, .{ .x = a.x, .y = r.bottom() } }, .{ area, area, Color.lerp(area, ui.theme.bg, 0.6), Color.lerp(area, ui.theme.bg, 0.6) });
        }
        i = 0;
        while (i + 1 < values.len) : (i += 1) {
            const a = sparkPt(r, values, i, xstep, lo, range);
            const b = sparkPt(r, values, i + 1, xstep, lo, range);
            try ui.dl.line(ui.gpa, a, b, 2.0, ui.theme.accent_hover);
        }
    }

    pub fn fontFor(ui: *Ui, role: Role) *Font {
        return switch (theme_mod.font_family(role)) {
            .display => ui.fonts.display,
            .ui, .mono => ui.fonts.ui,
        };
    }

    pub fn centeredText(ui: *Ui, str: []const u8, r: Rect, px: f32, color: Color) !void {
        const m = text.measure(ui.fonts.ui, str, px, .{}).width;
        const line_h = ui.fonts.ui.lineHeight(px);
        const pos = Rect{ .x = r.x + (r.w - m) * 0.5, .y = r.y + (r.h - line_h) * 0.5, .w = m + 2.0, .h = line_h + 2.0 };
        _ = try text.draw(&ui.dl, ui.gpa, ui.fonts.ui, str, pos, px, color, .{});
    }
};

fn sparkPt(r: Rect, vals: []const f32, i: usize, xstep: f32, lo: f64, range: f64) Vec2 {
    const v: f64 = vals[i];
    const t: f32 = @floatCast(std.math.clamp((v - lo) / range, 0.0, 1.0));
    return .{ .x = r.x + xstep * @as(f32, @floatFromInt(i)), .y = r.bottom() - 2.0 - (r.h - 4.0) * t };
}

fn saturatedAdd(a: f32, b: f32) f32 {
    const limit: f64 = std.math.floatMax(f32);
    return @floatCast(std.math.clamp(@as(f64, a) + @as(f64, b), -limit, limit));
}

pub const ROOT_SEED: u64 = 0x5245434f4e4c55; // "RECONLU"

pub fn fnv1a(seed: u64, bytes: []const u8) u64 {
    var h: u64 = if (seed == 0) 0xcbf29ce484222325 else seed;
    for (bytes) |b| {
        h ^= b;
        h *%= 0x100000001b3;
    }
    return h;
}

fn easeTo(cur: f32, target: f32, dt_ms: f32, duration_ms: f32) f32 {
    if (duration_ms <= 0.0) return target;
    const step = dt_ms / duration_ms;
    if (cur < target) return @min(cur + step, target);
    return @max(cur - step, target);
}

var test_font_a: ?Font = null;
var test_font_b: ?Font = null;

fn testFonts(gpa: std.mem.Allocator) !Fonts {
    if (test_font_a == null) {
        test_font_a = try font_mod.loadBundled(gpa, "Inter-Variable.ttf");
        test_font_b = try font_mod.loadBundled(gpa, "Outfit-Variable.ttf");
    }
    return .{ .ui = &test_font_a.?, .display = &test_font_b.? };
}

fn freeTestFonts(gpa: std.mem.Allocator) void {
    _ = gpa;
    if (test_font_a) |*f| {
        f.deinit();
        test_font_a = null;
    }
    if (test_font_b) |*f| {
        f.deinit();
        test_font_b = null;
    }
}

test "column cursor advances by child height plus gap" {
    const gpa = std.testing.allocator;
    const fonts = testFonts(gpa) catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer freeTestFonts(gpa);
    var ui = Ui.init(gpa, fonts, theme_mod.default, 400, 300);
    defer ui.deinit();

    try ui.begin(.{});
    try ui.beginPanel(.{ .x = 0, .y = 0, .w = 200, .h = 200 }, .{ .pad = 10, .gap = 5 });
    try ui.space(10);
    try ui.space(10);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), ui.top().cursor, 1e-4);
    ui.end();
    try std.testing.expectEqual(@as(usize, 0), ui.dl.clips.items.len);
    _ = ui.endFrame();
    try std.testing.expectEqual(@as(usize, 0), ui.frames.items.len);
}

test "a click needs press then release over the button" {
    const gpa = std.testing.allocator;
    const fonts = testFonts(gpa) catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer freeTestFonts(gpa);
    var ui = Ui.init(gpa, fonts, theme_mod.default, 400, 300);
    defer ui.deinit();
    try ui.begin(.{ .px = 50, .py = 10, .down = true, .pressed = true });
    const hit1 = try ui.button("go", "Go", .{});
    _ = ui.endFrame();
    try std.testing.expect(!hit1);
    try ui.begin(.{ .px = 50, .py = 10, .released = true });
    const hit2 = try ui.button("go", "Go", .{});
    _ = ui.endFrame();
    try std.testing.expect(hit2);
}

test "hover ramps up while the pointer is over a widget" {
    const gpa = std.testing.allocator;
    const fonts = testFonts(gpa) catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer freeTestFonts(gpa);
    var ui = Ui.init(gpa, fonts, theme_mod.default, 400, 300);
    defer ui.deinit();
    try ui.begin(.{ .px = 50, .py = 10 });
    _ = try ui.interact("hot", .{ .x = 0, .y = 0, .w = 400, .h = 34 });
    const after1 = ui.states.get(fnv1a(fnv1a(ROOT_SEED, ""), "hot")).?.hover;
    try std.testing.expect(after1 > 0.0 and after1 < 1.0);
    _ = ui.endFrame();
    var i: usize = 0;
    while (i < 20) : (i += 1) {
        try ui.begin(.{ .px = 50, .py = 10 });
        _ = try ui.interact("hot", .{ .x = 0, .y = 0, .w = 400, .h = 34 });
        _ = ui.endFrame();
    }
    const settled = ui.states.get(fnv1a(fnv1a(ROOT_SEED, ""), "hot")).?.hover;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), settled, 1e-6);
    try ui.begin(.{ .px = -5, .py = -5 });
    _ = try ui.interact("hot", .{ .x = 0, .y = 0, .w = 400, .h = 34 });
    _ = ui.endFrame();
    const away = ui.states.get(fnv1a(fnv1a(ROOT_SEED, ""), "hot")).?.hover;
    try std.testing.expect(away < settled);
}

test "toggle flips its value and its spring target follows" {
    const gpa = std.testing.allocator;
    const fonts = testFonts(gpa) catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer freeTestFonts(gpa);
    var ui = Ui.init(gpa, fonts, theme_mod.default, 400, 300);
    defer ui.deinit();
    var on = false;
    try ui.begin(.{ .px = 30, .py = 10, .down = true, .pressed = true });
    _ = try ui.toggle("t", "Enable", &on);
    _ = ui.endFrame();
    try std.testing.expect(!on);
    try ui.begin(.{ .px = 30, .py = 10, .released = true });
    const changed = try ui.toggle("t", "Enable", &on);
    _ = ui.endFrame();
    try std.testing.expect(changed and on);
    const st = ui.states.get(fnv1a(fnv1a(ROOT_SEED, ""), "t")).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), st.knob.target, 1e-6);
}

test "labels emit triangles and endFrame closes leaked panels" {
    const gpa = std.testing.allocator;
    const fonts = testFonts(gpa) catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer freeTestFonts(gpa);
    var ui = Ui.init(gpa, fonts, theme_mod.default, 400, 300);
    defer ui.deinit();
    try ui.begin(.{});
    try ui.beginPanel(.{ .x = 4, .y = 4, .w = 392, .h = 292 }, .{});
    _ = try ui.label("Hello, ReconL", .body, null, .{});
    const mesh = ui.endFrame();
    try std.testing.expect(mesh.vertices.len > 0);
    try std.testing.expect(mesh.indices.len >= 3);
    try std.testing.expectEqual(@as(usize, 0), ui.frames.items.len);
    try std.testing.expectEqual(@as(usize, 0), ui.dl.clips.items.len);
    for (mesh.indices) |idx| try std.testing.expect(idx < mesh.vertices.len);
}

test "slider drag moves the value toward the pointer" {
    const gpa = std.testing.allocator;
    const fonts = testFonts(gpa) catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer freeTestFonts(gpa);
    var ui = Ui.init(gpa, fonts, theme_mod.default, 400, 300);
    defer ui.deinit();
    var v: f32 = 0.0;
    try ui.begin(.{ .px = 10, .py = 5, .down = true, .pressed = true });
    _ = try ui.slider("s", "Gain", &v, 0.0, 1.0);
    _ = ui.endFrame();
    try ui.begin(.{ .px = 10 + 380 * 0.75, .py = 5, .down = true });
    const changed = try ui.slider("s", "Gain", &v, 0.0, 1.0);
    _ = ui.endFrame();
    try std.testing.expect(changed);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), v, 0.05);
}

test "the wheel scrolls only over the viewport, and the spring clamps at the end" {
    const gpa = std.testing.allocator;
    const fonts = testFonts(gpa) catch |e| switch (e) {
        error.FontFileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer freeTestFonts(gpa);
    var ui = Ui.init(gpa, fonts, theme_mod.default, 400, 300);
    defer ui.deinit();
    try ui.begin(.{ .px = -500, .py = -500, .wheel = 1 });
    const rest = try ui.beginScroll("log", 100, 300);
    const r = ui.last_rect;
    const rest_content_y = ui.top().content.y;
    try std.testing.expectApproxEqAbs(r.y, rest.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), rest.h, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), ui.top().content.h, 1e-4);
    _ = ui.endFrame();
    try ui.begin(.{ .px = 50, .py = 50, .wheel = 1 });
    const moved = try ui.beginScroll("log", 100, 300);
    const moved_content_y = ui.top().content.y;
    try std.testing.expectApproxEqAbs(r.y, moved.y, 1e-4);
    try std.testing.expect(moved_content_y < rest_content_y);
    _ = ui.endFrame();
    var i: usize = 0;
    while (i < 20) : (i += 1) {
        try ui.begin(.{ .px = 50, .py = 50, .wheel = 1 });
        _ = try ui.beginScroll("log", 100, 300);
        _ = ui.endFrame();
    }
    i = 0;
    while (i < 120) : (i += 1) {
        try ui.begin(.{ .px = 50, .py = 50 });
        _ = try ui.beginScroll("log", 100, 300);
        _ = ui.endFrame();
    }
    try ui.begin(.{ .px = 50, .py = 50 });
    const bottom = try ui.beginScroll("log", 100, 300);
    const bottom_content_y = ui.top().content.y;
    try std.testing.expectApproxEqAbs(rest.y, bottom.y, 1e-4);
    try std.testing.expectApproxEqAbs(rest_content_y - 200.0, bottom_content_y, 0.5);
    _ = ui.endFrame();
    try std.testing.expectApproxEqAbs(r.y, ui.last_rect.y, 1e-4);
}

