//! The ReconL frontend core: everything needed to *decide* a frame of UI -
//! layout, text, animation, tessellation - and none of the rendering.
//!
//! The rendering half lives in `backend.zig`, which is deliberately a
//! separate module: core tests never link reconl, and the demo/ABI layers
//! import both. Nothing here reads a wall clock or an OS event; the host
//! feeds inputs, the widgets answer. That is what keeps the frontend on the
//! renderer's determinism rule (`docs/determinism.md`).
//!
//! The C ABI (`c_api.zig`, `include/reconl_ui.h`) is a thin skin over these
//! same types - Java, Kotlin and C drive exactly what the Zig showcase drives.

pub const geom = @import("geom.zig");
pub const theme = @import("theme.zig");
pub const anim = @import("anim.zig");
pub const tess = @import("tess.zig");
pub const polygon = @import("polygon.zig");
pub const font = @import("font.zig");
pub const text = @import("text.zig");
pub const ui = @import("ui.zig");
pub const png = @import("png.zig");

// Pull every file's tests into `zig build test`. Explicit rather than
// `refAllDecls`, so a future lazy-analysis surprise can't silently drop a
// test file from the suite.
test {
    _ = @import("geom.zig");
    _ = @import("theme.zig");
    _ = @import("anim.zig");
    _ = @import("tess.zig");
    _ = @import("polygon.zig");
    _ = @import("font.zig");
    _ = @import("text.zig");
    _ = @import("ui.zig");
    _ = @import("png.zig");
}
