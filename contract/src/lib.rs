//! What a host says to a backend, and what a backend answers, before any pixels
//! exist.
//!
//! [`FrameInput`] is one frame's input and [`ShadowRequest`] is the part of it a
//! host may vary per frame. They live in their own crate, depending on nothing
//! but the data types they are made of, because *both* backends take them: the
//! hardware backend used to import them from the reference backend's crate,
//! which made a GPU backend depend on a CPU one to describe its own input. The
//! contract is now neutral ground - a new backend depends on this crate and on
//! nothing else.
//!
//! Nothing here decides anything. A field is present because a backend cannot
//! derive it: the shadow request is what the host asked for, and the tier ladder
//! decides what it gets. The ladder itself is not here: it lives with the device
//! that owns the frame cost and the tier (`ffi/src/offload.rs`), and a backend
//! renders the tier it is told to render.

use reconl_core::tier::ShadowFilter;
use reconl_raster::math::{Mat4, Vec3};
use reconl_raster::shade::LightSet;
use reconl_raster::DrawItem;
use reconl_shadow as shadow;

/// A frame's worth of requests that the host is allowed to vary per frame.
#[derive(Clone, Copy, Debug)]
pub struct ShadowRequest {
    pub enabled: bool,
    pub cascades: u32,
    /// Texel budget for the shadow maps, in bytes.
    pub texel_budget_bytes: u64,
    pub filter: ShadowFilter,
    pub max_distance: f32,
    pub blend_band: f32,
    /// Frames between static-cascade refreshes. `1` = every frame.
    pub refresh_interval_frames: u32,
    /// Allow the static cascade cache to live in the disk arena (T4).
    pub allow_disk_cache: bool,
    /// Bias to use instead of the tier's preset, when the host pins one.
    ///
    /// `None` is the documented policy: [`shadow::bias_preset`] derives the
    /// values from the tier, the map size and the filter, so a weaker tier gets
    /// more slack. `Some` is the ABI's `normal_bias`/`depth_bias`/`slope_bias`,
    /// used as given - which is what lets a scene render the same image on two
    /// tiers, since the presets differ by design and that difference lands on a
    /// shadow's edge as a pixel of coverage.
    pub bias: Option<shadow::BiasPreset>,
}

impl Default for ShadowRequest {
    fn default() -> Self {
        Self {
            enabled: true,
            cascades: 3,
            texel_budget_bytes: 24 << 20,
            filter: ShadowFilter::Pcf3x3,
            max_distance: 120.0,
            blend_band: 4.0,
            refresh_interval_frames: 1,
            allow_disk_cache: true,
            bias: None,
        }
    }
}

/// One frame's input. Borrowed, so recording a frame allocates nothing.
///
/// The draw list belongs to the frame: the layer that owns the vertex buffers
/// builds it once per frame and every backend reads it. A backend that has to
/// stamp its own lighting state into a draw writes into a list it owns - the
/// reference tier's colour list - so no backend allocates one per frame.
pub struct FrameInput<'a> {
    pub frame_index: u64,
    pub width: u32,
    pub height: u32,
    /// The viewport the host asked for this frame's colour pass, in *frame*
    /// pixels, or `(0, 0)` for the whole frame - the documented default.
    ///
    /// A backend resolves it against whatever target it actually renders into
    /// with [`reconl_raster::rendered_viewport`], because a tier may render at a
    /// fraction of the frame. Shadow passes ignore it and draw their whole map.
    pub viewport: (u32, u32),
    pub camera_view: Mat4,
    pub fov_y_deg: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    /// The shadow-casting directional light's direction, for cascade fitting.
    pub light_dir: Vec3,
    /// Hash of every shadow-relevant light parameter; part of the cache key.
    pub light_hash: u64,
    pub lights: LightSet,
    pub shadow: ShadowRequest,
    pub clear_color: [f32; 4],
    pub clear_depth: f32,
    pub clear_color_enabled: bool,
    pub clear_depth_enabled: bool,
    /// Camera-space world revision and static-geometry revision, both already
    /// folded into the cascade cache key by `reconl-scene`.
    pub world_revision: u64,
    pub static_geometry_revision: u64,
    /// Whether the caller will read this frame's colour checksum.
    ///
    /// Only the FFI audit reads one. Every other caller sets this false, which
    /// is worth stating because the checksum is a full pass over the frame's
    /// colour target on every submit. When false the checksum is zero rather
    /// than a stale one from an earlier frame.
    pub checksum: bool,
    pub draws: &'a [DrawItem<'a>],
}
