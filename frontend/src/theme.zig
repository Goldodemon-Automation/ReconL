//! The ReconL design tokens: one palette per surface, dark-first, brand teal
//! as the only chromatic accent.
//!
//! Values come from the brand block in `PROMPT.md` section 1: Near Black
//! `#0B0D0C` background, Deep Pine `#5F8A80` wordmark, muted Pine `#7FA79B`
//! for secondary text and rules, Off-White `#F5F5F3` body copy. Everything
//! else is a monochrome step between them - no purple, no orange-on-blue,
//! no neon pairs. Type: Outfit for headings, Inter for body/UI, monospace
//! for code - both font files ship in `assets/fonts/`.
//!
//! The theme is a plain struct, not a global: a host can instantiate two
//! (dark and light) and switch per frame, which is how the showcase shows
//! the same widgets under both without rebuilding anything.

const geom = @import("geom.zig");
const Color = geom.Color;

/// Semantic colours. Named by role, never by hue, so a theme swap is a
/// struct assignment and not a search-and-replace.
pub const Theme = struct {
    // Surfaces, darkest to lightest.
    bg: Color = Color.hex("#0B0D0C"),
    surface: Color = Color.hex("#141816"),
    surface_raised: Color = Color.hex("#1C221F"),
    // Text.
    text: Color = Color.hex("#F5F5F3"),
    text_muted: Color = Color.hex("#7FA79B"),
    text_dim: Color = Color.rgba8(0x7F, 0xA7, 0x9B, 0x66),
    // The brand accent (the logo's own teal - the palette rule's exception).
    accent: Color = Color.hex("#5F8A80"),
    accent_hover: Color = Color.hex("#6FA095"),
    accent_pressed: Color = Color.hex("#4E776D"),
    on_accent: Color = Color.hex("#0B0D0C"),
    // Structure.
    border: Color = Color.rgba8(0x5F, 0x8A, 0x80, 0x33),
    border_strong: Color = Color.rgba8(0x5F, 0x8A, 0x80, 0x66),
    // Danger stays inside the monochrome-plus-teal rule: a desaturated
    // brick, never a fire-engine red.
    danger: Color = Color.hex("#A4544C"),
    // Focus ring: the accent at full alpha, drawn outside the widget.
    focus: Color = Color.hex("#7FA79B"),

    // Spacing scale (4px base - the spacing rhythm every layout uses).
    gap_xs: f32 = 4,
    gap_s: f32 = 8,
    gap_m: f32 = 12,
    gap_l: f32 = 16,
    gap_xl: f32 = 24,

    // Corner radii.
    radius_s: f32 = 4,
    radius_m: f32 = 8,
    radius_l: f32 = 14,
    radius_pill: f32 = 999, // clamped to half-height by the tessellator

    // Type scale (px). Headings Outfit, body Inter - see `text.zig`, which
    // picks the family from the role rather than from a call-site string.
    font_title: f32 = 28,
    font_heading: f32 = 20,
    font_body: f32 = 15,
    font_label: f32 = 13,
    font_mono: f32 = 14,

    // Motion defaults, consumed by `anim.zig`.
    duration_hover_ms: u32 = 120,
    duration_press_ms: u32 = 80,
    duration_open_ms: u32 = 240,
    spring_stiffness: f32 = 320.0,
    spring_damping: f32 = 26.0,
};

/// Text roles pair a size, a family and a weight, so a widget asks for
/// `role_heading` and never for "Outfit 600".
pub const Role = enum {
    title,
    heading,
    body,
    label,
    mono,
};

pub fn font_size(theme: Theme, role: Role) f32 {
    return switch (role) {
        .title => theme.font_title,
        .heading => theme.font_heading,
        .body => theme.font_body,
        .label => theme.font_label,
        .mono => theme.font_mono,
    };
}

/// `"Outfit"` / `"Inter"` / `"mono"`: the family key `font.zig` indexes by.
/// Outfit carries headings, Inter everything else, mono for numbers/code.
pub fn font_family(role: Role) Family {
    return switch (role) {
        .title, .heading => .display,
        .body, .label => .ui,
        .mono => .mono,
    };
}

pub const Family = enum { display, ui, mono };

pub const default = Theme{};

test "the brand tokens are the brand block's values" {
    const t = default;
    // Near Black.
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0x0B) / 255.0, t.bg.r, 1e-6);
    // Deep Pine wordmark.
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0x5F) / 255.0, t.accent.r, 1e-6);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0x8A) / 255.0, t.accent.g, 1e-6);
    // Off-White body copy.
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0xF5) / 255.0, t.text.r, 1e-6);
    // The spacing scale is a 4px rhythm.
    try @import("std").testing.expectEqual(@as(f32, 8), t.gap_s);
    try @import("std").testing.expectEqual(@as(f32, 16), t.gap_l);
}

test "roles map to the brand families" {
    try @import("std").testing.expectEqual(Family.display, font_family(.title));
    try @import("std").testing.expectEqual(Family.ui, font_family(.body));
    try @import("std").testing.expectEqual(Family.mono, font_family(.mono));
}
