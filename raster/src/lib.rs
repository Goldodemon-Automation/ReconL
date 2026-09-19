//! The reference rasteriser: tiled, fixed-point, reversed-Z, and the tier every
//! other tier is diffed against.
//!
//! Design rules that exist to make the output comparable (docs/determinism.md):
//!
//! * **8-bit subpixel precision and the top-left fill rule** ([`fixed`]).
//! * **Reversed-Z, `[0,1]` depth, GREATER test, clear to 0.0** ([`math`]).
//! * **Tiles are independent.** A tile's pixels are written by exactly one
//!   thread, in a fixed triangle order, so the frame is bit-identical for 1, 2, 4
//!   or 8 workers. That is a test, not a hope (`tests/thread_determinism.rs`).
//! * **No allocation inside a frame.** All storage is reserved in
//!   [`tile::Rasterizer::prepare`] before the first draw; `prepare` must be sized
//!   by the host, and anything that still needs to grow is counted and reported.
//! * **SIMD only where it is exact.** [`simd`] uses vectorised fill/copy/clear,
//!   which cannot change a pixel, and reports which instruction sets it found.
//!   The shaded path is scalar in every tier, on purpose.

pub mod clip;
pub mod fixed;
pub mod math;
pub mod shade;
pub mod simd;
pub mod texture;
pub mod tile;

pub use clip::ClipVertex;
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
        let color = match self.color_slice() {
            Some(c) => c,
            None => return Err(Error::new(Code::NotReady, "target has no colour buffer")),
        };
        let needed = self.pixels() * 4;
        if out.len() < needed {
            return Err(Error::new(Code::InvalidArgument, "readback buffer is too small"));
        }
        for (i, px) in color.chunks_exact(4).enumerate() {
            out[i * 4] = to_u8(px[0]);
            out[i * 4 + 1] = to_u8(px[1]);
            out[i * 4 + 2] = to_u8(px[2]);
            out[i * 4 + 3] = to_u8(px[3]);
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
