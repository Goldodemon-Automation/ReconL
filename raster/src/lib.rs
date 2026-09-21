//! The reference rasteriser: tiled, fixed-point, reversed-Z, and the tier every
//! other tier is diffed against.
//!
//! Design rules that exist to make the output comparable (docs/determinism.md):
//!
//! * **8-bit subpixel precision and the top-left fill rule** ([`fixed`]).
//! * **Reversed-Z, `[0,1]` depth, GREATER test, clear to 0.0** ([`math`]).
//! * **Tiles are independent.** A tile's pixels are written by exactly one
//!   thread, in a fixed triangle order, so the frame is bit-identical for 1, 2, 4
//!   or 8 workers. That is a test, not a hope
//!   (`tests/render.rs::frame_is_bit_identical_for_1_2_4_and_8_workers`).
//! * **No allocation inside a frame.** All storage is reserved in
//!   [`tile::Rasterizer::prepare`] before the first draw; `prepare` must be sized
//!   by the host, and anything that still needs to grow is counted and reported.
//! * **SIMD only where it is exact.** [`simd`] uses vectorised fill/copy/clear,
//!   which cannot change a pixel, and reports which instruction sets it found.
//!   The shaded path is scalar in every tier, on purpose.

pub mod clip;
pub mod fixed;
pub mod framegen;
pub mod math;
pub mod shade;
pub mod simd;
pub mod texture;
pub mod tile;

pub use clip::ClipVertex;
pub use framegen::{generate as generate_frame, Camera as FrameCamera, History as FrameHistory};
pub use tile::{Rasterizer, RasterStats};

use reconl_core::alloc::{HostAlloc, HostVec};
use reconl_core::error::{Code, Error, Result};

/// Attribute layout of a vertex, fixed by the ABI.
pub const ATTR_COUNT: usize = 12;
pub const ATTR_POSITION: usize = 0;
pub const ATTR_NORMAL: usize = 3;
pub const ATTR_UV: usize = 6;
pub const ATTR_COLOR: usize = 8;

pub const MAX_TEXTURE_SLOTS: usize = 8;

/// Resolves a pass's viewport to the pixel area of a render target.
///
/// This is the single rule every tier maps clip space through, so a host that
/// asks for a sub-rect gets the same one on hardware and on the reference tier.
///
/// The viewport is in *frame* pixels - the units the host sized its own window
/// and swapchain in - and is scaled to `target` so that a tier which renders at
/// a fraction of the frame (`resolution_scale`) confines the same fraction.
///
/// * `(0, 0)` means the whole target. That is the documented default, so a host
///   that never sets a viewport keeps the full-frame path byte for byte.
/// * A viewport larger than the frame is clamped to the target rather than
///   refused: mapping to more than the target can hold is a no-op region, and
///   the binner's tiles are the target's, not the request's.
/// * The result is never zero in either axis, so an empty viewport cannot make a
///   pass silently draw nothing.
pub fn rendered_viewport(viewport: (u32, u32), frame: (u32, u32), target: (u32, u32)) -> (u32, u32) {
    let axis = |vp: u32, frame: u32, target: u32| -> u32 {
        if vp == 0 {
            return target;
        }
        let frame = frame.max(1) as u64;
        let scaled = (vp as u64 * target as u64 + frame / 2) / frame;
        scaled.clamp(1, target as u64) as u32
    };
    (axis(viewport.0, frame.0, target.0), axis(viewport.1, frame.1, target.1))
}

/// Mirrors `ReconLVertex`, and is what a host uploads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex {
    pub const fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], color: [f32; 4]) -> Self {
        Self { position, normal, uv, color }
    }

    pub fn plain(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, normal: [0.0, 1.0, 0.0], uv: [0.0, 0.0], color }
    }

    #[inline]
    pub fn attrs(&self) -> [f32; ATTR_COUNT] {
        [
            self.position[0], self.position[1], self.position[2],
            self.normal[0], self.normal[1], self.normal[2],
            self.uv[0], self.uv[1],
            self.color[0], self.color[1], self.color[2], self.color[3],
        ]
    }
}

/// Mirrors `ReconLCullMode`, `ReconLBlendMode` and `ReconLCompareFunc`.
pub const CULL_NONE: u32 = 0;
pub const CULL_BACK: u32 = 1;
pub const CULL_FRONT: u32 = 2;

pub const COMPARE_LESS: u32 = 0;
pub const COMPARE_GREATER: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipelineState {
    pub blend: u32,
    pub cull: u32,
    pub depth_compare: u32,
    /// `false` means the depth test is not performed at all (no-op pass).
    pub depth_test: bool,
    pub depth_write: bool,
    /// Flip the normal on back faces (two-sided lighting).
    pub two_sided: bool,
}

impl Default for PipelineState {
    fn default() -> Self {
        Self {
            blend: shade::BLEND_OPAQUE,
            cull: CULL_BACK,
            depth_compare: COMPARE_GREATER,
            depth_test: true,
            depth_write: true,
            two_sided: false,
        }
    }
}

/// What a draw does per fragment.
#[derive(Clone, Copy)]
pub enum ShaderRef<'a> {
    /// Depth only: the shadow pass, or a depth prepass. No colour, no shading.
    DepthOnly,
    Surface(shade::SurfaceShader<'a>),
}

impl<'a> ShaderRef<'a> {
    pub fn is_depth_only(&self) -> bool {
        matches!(self, ShaderRef::DepthOnly)
    }
}

/// One draw call: vertices, an optional index buffer, and state.
///
/// `transform` is the model-view-projection the colour pass uses. `model` is
/// kept separately because the shadow pass needs to compose the *light's* matrix
/// with the same model transform, and reconstructing the model from the
/// composed matrix is not possible in floating point.
#[derive(Clone, Copy)]
pub struct DrawItem<'a> {
    pub vertices: &'a [Vertex],
    pub indices: Option<&'a [u32]>,
    pub transform: math::Mat4,
    pub model: math::Mat4,
    pub pipeline: PipelineState,
    pub shader: ShaderRef<'a>,
    /// 1 = participates in the cached static cascade (see `reconl-scene`).
    pub dynamic: bool,
    pub casts_shadow: bool,
}

/// A colour target and/or a depth target, both `f32`.
///
/// Colour is RGBA in `[0,1]`; depth is reversed-Z in `[0,1]` with 0.0 as the
/// clear value. Keeping `f32` through the whole pipeline is what lets the
/// golden diff see the same numbers the GPU tier does.
pub struct Target {
    pub width: u32,
    pub height: u32,
    pub color: Option<HostVec<f32>>,
    pub depth: Option<HostVec<f32>>,
    alloc: HostAlloc,
}

impl Target {
    pub fn new_color(alloc: HostAlloc, width: u32, height: u32) -> Result<Self> {
        let mut t = Self { width, height, color: None, depth: None, alloc };
        let mut color = HostVec::new(alloc);
        color.resize_with((width as usize) * (height as usize) * 4, || 0.0)?;
        t.color = Some(color);
        Ok(t)
    }

    pub fn new_depth(alloc: HostAlloc, width: u32, height: u32) -> Result<Self> {
        let mut t = Self { width, height, color: None, depth: None, alloc };
        let mut depth = HostVec::new(alloc);
        depth.resize_with((width as usize) * (height as usize), || 0.0)?;
        t.depth = Some(depth);
        Ok(t)
    }

    /// A colour target with a depth buffer attached.
    pub fn with_depth(mut self) -> Result<Self> {
        let mut depth = HostVec::new(self.alloc);
        depth.resize_with((self.width as usize) * (self.height as usize), || 0.0)?;
        self.depth = Some(depth);
        Ok(self)
    }

    pub fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    pub fn color_slice(&self) -> Option<&[f32]> {
        self.color.as_ref().map(|c| c.as_slice())
    }

    pub fn color_slice_mut(&mut self) -> Option<&mut [f32]> {
        self.color.as_mut().map(|c| c.as_mut_slice())
    }

    pub fn depth_slice(&self) -> Option<&[f32]> {
        self.depth.as_ref().map(|d| d.as_slice())
    }

    pub fn depth_slice_mut(&mut self) -> Option<&mut [f32]> {
        self.depth.as_mut().map(|d| d.as_mut_slice())
    }

    pub fn clear_color(&mut self, rgba: [f32; 4]) {
        if let Some(color) = self.color_slice_mut() {
            for px in color.chunks_exact_mut(4) {
                px[0] = rgba[0];
                px[1] = rgba[1];
                px[2] = rgba[2];
                px[3] = rgba[3];
            }
        }
    }

    /// Clears depth to `value`. Reversed-Z: the default clear is 0.0 (far).
    pub fn clear_depth(&mut self, value: f32) {
        if let Some(depth) = self.depth_slice_mut() {
            simd::fill_f32(depth, value);
        }
    }

    /// Converts colour to tightly packed RGBA8, the format present-to-memory
    /// hands back.
    pub fn to_rgba8(&self, out: &mut [u8]) -> Result<()> {
        self.to_rgba8_rows(out, self.width as usize * 4, false)
    }

    /// Converts colour into a destination with a row pitch, optionally flipped,
    /// row by row.
    ///
    /// This is the whole of how a frame becomes the host's bytes: the conversion
    /// and the presentation layout happen in one pass, so a present-to-memory
    /// host never pays for a second, tightly packed copy of the frame between
    /// this target and its own buffer. `pitch` is the destination's row length in
    /// bytes and must be at least one row wide.
    pub fn to_rgba8_rows(&self, out: &mut [u8], pitch: usize, flip: bool) -> Result<()> {
        let color = match self.color_slice() {
            Some(c) => c,
            None => return Err(Error::new(Code::NotReady, "target has no colour buffer")),
        };
        let rows = self.height as usize;
        let row_bytes = self.width as usize * 4;
        let needed = rows
            .saturating_sub(1)
            .saturating_mul(pitch)
            .saturating_add(row_bytes);
        if pitch < row_bytes || out.len() < needed {
            return Err(Error::new(Code::InvalidArgument, "readback buffer is too small"));
        }
        for row in 0..rows {
            let source = if flip { rows - 1 - row } else { row };
            let src = &color[source * row_bytes..(source + 1) * row_bytes];
            let at = row * pitch;
            simd::rgba_f32_to_unorm8(src, &mut out[at..at + row_bytes]);
        }
        Ok(())
    }
}

/// Exact float-to-unorm8 rounding: `round(clamp(x,0,1) * 255)`.
#[inline]
pub fn to_u8(v: f32) -> u8 {
    let c = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    (c * 255.0 + 0.5) as u8
}

/// FNV-1a over a byte slice. The frame checksum the null backend and the
/// determinism tests compare.
pub fn checksum_bytes(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for b in data {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// FNV-1a over the raw bits of an `f32` slice.
pub fn checksum_f32(data: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for v in data {
        for b in v.to_bits().to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_attr_layout_matches_the_abi_offsets() {
        let v = Vertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.25, 0.75],
            color: [0.1, 0.2, 0.3, 0.4],
        };
        let a = v.attrs();
        assert_eq!(&a[ATTR_POSITION..ATTR_POSITION + 3], &[1.0, 2.0, 3.0]);
        assert_eq!(&a[ATTR_NORMAL..ATTR_NORMAL + 3], &[0.0, 1.0, 0.0]);
        assert_eq!(&a[ATTR_UV..ATTR_UV + 2], &[0.25, 0.75]);
        assert_eq!(&a[ATTR_COLOR..ATTR_COLOR + 4], &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(a.len(), ATTR_COUNT);
    }

    #[test]
    fn unorm8_conversion_rounds_and_clamps() {
        assert_eq!(to_u8(0.0), 0);
        assert_eq!(to_u8(1.0), 255);
        assert_eq!(to_u8(0.5), 128);
        assert_eq!(to_u8(-3.0), 0);
        assert_eq!(to_u8(9.0), 255);
        assert_eq!(to_u8(f32::NAN), 0);
        assert_eq!(to_u8(1.0 / 255.0), 1);
    }

    #[test]
    fn target_clear_and_readback() {
        let alloc = HostAlloc::system();
        let mut t = Target::new_color(alloc, 4, 4).unwrap().with_depth().unwrap();
        t.clear_color([0.0, 0.5, 1.0, 1.0]);
        t.clear_depth(0.0);
        let mut out = vec![0u8; 4 * 4 * 4];
        t.to_rgba8(&mut out).unwrap();
        assert_eq!(&out[0..4], &[0, 128, 255, 255]);
        assert!(t.depth_slice().unwrap().iter().all(|d| *d == 0.0));
    }

    /// The readback's row layout is what a host with a pitched or bottom-up
    /// buffer depends on, and what the backends now apply as they write. A wrong
    /// stride and a wrong flip are both invisible in a tight, top-down buffer,
    /// so each boundary gets asserted here rather than inferred: a pitch wider
    /// than a row, a flip, a single row, and the two refusals that must be errors
    /// rather than a panic or overlapping rows.
    #[test]
    fn readback_rows_honour_the_pitch_and_the_flip() {
        let alloc = HostAlloc::system();
        let mut t = Target::new_color(alloc, 2, 2).unwrap();
        // Two rows, distinguishable per row: row 0 red, row 1 blue.
        let color = t.color_slice_mut().unwrap();
        for (i, px) in color.chunks_exact_mut(4).enumerate() {
            let red = i < 2;
            px[0] = if red { 1.0 } else { 0.0 };
            px[2] = if red { 0.0 } else { 1.0 };
            px[3] = 1.0;
        }

        let row_bytes = 2 * 4;
        let pitch = row_bytes + 8;
        let mut out = vec![0xABu8; pitch * 2];
        t.to_rgba8_rows(&mut out, pitch, true).unwrap();
        // Flipped: the buffer's first row is the image's last - blue, then red.
        assert_eq!(&out[0..4], &[0, 0, 255, 255], "row 0 is the image's last row");
        assert_eq!(&out[pitch..pitch + 4], &[255, 0, 0, 255]);
        assert!(
            out[row_bytes..pitch].iter().all(|b| *b == 0xAB)
                && out[pitch + row_bytes..].iter().all(|b| *b == 0xAB),
            "the pitch is the host's, not ours to write"
        );

        // The same frame unflipped, tight: the two must be row-reversals of each
        // other, which is the whole of what `flip` means.
        let mut tight = vec![0u8; row_bytes * 2];
        t.to_rgba8_rows(&mut tight, row_bytes, false).unwrap();
        assert_eq!(&tight[0..4], &[255, 0, 0, 255]);
        assert_eq!(&tight[row_bytes..row_bytes + 4], &[0, 0, 255, 255]);
        assert_eq!(&out[0..row_bytes], &tight[row_bytes..row_bytes * 2]);

        // A single row: a flip is a no-op, not an underflow.
        let mut one = Target::new_color(alloc, 2, 1).unwrap();
        one.clear_color([0.25, 0.25, 0.25, 1.0]);
        let mut buf = vec![0u8; row_bytes];
        one.to_rgba8_rows(&mut buf, row_bytes, true).unwrap();
        assert_eq!(&buf[0..4], &[64, 64, 64, 255]);

        // Refusals, not panics: a destination too small for the last row, and a
        // pitch narrower than a row (which would make rows overlap).
        let too_small = vec![0u8; row_bytes * 2 - 1];
        assert_eq!(
            t.to_rgba8_rows(&mut too_small.clone(), row_bytes, false).unwrap_err().code,
            Code::InvalidArgument
        );
        let mut narrow = vec![0u8; row_bytes * 2];
        assert_eq!(
            t.to_rgba8_rows(&mut narrow, row_bytes - 1, false).unwrap_err().code,
            Code::InvalidArgument
        );
    }

    /// The viewport rule, which both tiers and both grid sizes go through, so a
    /// mistake here is a cross-tier difference rather than a rounding detail.
    #[test]
    fn the_viewport_resolves_to_a_sub_rect_of_the_target() {
        // Unset means the whole target, in either axis.
        assert_eq!(rendered_viewport((0, 0), (64, 64), (64, 64)), (64, 64));
        assert_eq!(rendered_viewport((0, 48), (64, 64), (64, 64)), (64, 48));
        // The whole frame, asked for explicitly, is the whole target to the byte.
        assert_eq!(rendered_viewport((64, 64), (64, 64), (64, 64)), (64, 64));
        assert_eq!(rendered_viewport((1920, 1080), (1920, 1080), (1920, 1080)), (1920, 1080));
        // A tier that renders at half the frame confines the same fraction.
        assert_eq!(rendered_viewport((32, 32), (64, 64), (32, 32)), (16, 16));
        assert_eq!(rendered_viewport((0, 0), (64, 64), (32, 32)), (32, 32));
        // Rounding: half of an odd frame goes to the nearer target pixel, never
        // to zero and never past the target.
        assert_eq!(rendered_viewport((33, 33), (64, 64), (32, 32)), (17, 17));
        assert_eq!(rendered_viewport((1, 1), (64, 64), (32, 32)), (1, 1));
        assert_eq!(rendered_viewport((1, 1), (1920, 1080), (960, 540)), (1, 1));
        // An oversized viewport is clamped to the target, not refused and not
        // allowed to bin tiles that do not exist.
        assert_eq!(rendered_viewport((128, 128), (64, 64), (64, 64)), (64, 64));
        assert_eq!(rendered_viewport((u32::MAX, u32::MAX), (1920, 1080), (960, 540)), (960, 540));
    }

    #[test]
    fn checksum_is_stable_and_sensitive() {
        let a = [0.1f32, 0.2, 0.3];
        let mut b = a;
        assert_eq!(checksum_f32(&a), checksum_f32(&b));
        b[1] += 1.0e-6;
        assert_ne!(checksum_f32(&a), checksum_f32(&b));
        assert_eq!(checksum_bytes(b"reconl"), checksum_bytes(b"reconl"));
    }
}
