//! Mip chain generation.
//!
//! A box filter with a fixed sample order and integer weights, so the result is
//! bit-identical everywhere. Anything cheaper (a bilinear tap chain) or cheaper
//! looking (an approximation) would make `soft-cpu` and hardware produce
//! different pixels at distance, which is exactly what the goldens are for.

use reconl_core::alloc::{HostAlloc, HostVec};
use reconl_core::error::{Code, Error, Result};

/// One level: RGBA8, tightly packed rows.
pub struct MipLevel {
    pub width: u32,
    pub height: u32,
    pub pixels: HostVec<u8>,
}

/// The full chain, level 0 first.
pub struct MipChain {
    pub levels: HostVec<MipLevel>,
}

impl MipChain {
    pub fn level_count(width: u32, height: u32) -> u32 {
        let mut levels = 1u32;
        let (mut w, mut h) = (width.max(1), height.max(1));
        while w > 1 || h > 1 {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            levels += 1;
        }
        levels
    }

    pub fn level(&self, index: usize) -> Option<&MipLevel> {
        self.levels.get(index)
    }

    /// Total bytes across every level.
    pub fn bytes(&self) -> u64 {
        self.levels.iter().map(|l| (l.pixels.len() as u64)).sum()
    }
}

fn to_u8(v: f32) -> u8 {
    let c = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    (c * 255.0 + 0.5) as u8
}

/// Generates `mip_levels` levels (0 means "the full chain") from RGBA8 pixels.
pub fn generate_mip_chain(
    alloc: HostAlloc,
    width: u32,
    height: u32,
    rgba8: &[u8],
    mip_levels: u32,
) -> Result<MipChain> {
    if width == 0 || height == 0 {
        return Err(Error::new(Code::InvalidArgument, "zero-sized texture"));
    }
    let expected = (width as usize) * (height as usize) * 4;
    if rgba8.len() < expected {
        return Err(Error::new(Code::InvalidArgument, "texture upload is smaller than its dimensions"));
    }
    let full = MipChain::level_count(width, height);
    let levels = if mip_levels == 0 { full } else { mip_levels.min(full).max(1) };

    let mut chain = MipChain { levels: HostVec::new(alloc) };
    let mut base = MipLevel { width, height, pixels: HostVec::new(alloc) };
    base.pixels.resize_with(expected, || 0)?;
    base.pixels.as_mut_slice().copy_from_slice(&rgba8[..expected]);
    chain.levels.push(base)?;

    for level in 1..levels {
        let prev = chain.levels.get((level - 1) as usize).expect("previous level exists");
        let (pw, ph) = (prev.width, prev.height);
        let w = (pw / 2).max(1);
        let h = (ph / 2).max(1);
        let mut out = MipLevel { width: w, height: h, pixels: HostVec::new(alloc) };
        out.pixels.resize_with((w as usize) * (h as usize) * 4, || 0)?;
        {
            let src = prev.pixels.as_slice();
            let dst = out.pixels.as_mut_slice();
            for y in 0..h {
                for x in 0..w {
                    // Box filter over the up-to-2x2 footprint, integer accumulators
                    // so the rounding is exact and identical on every tier.
                    let x0 = (x * 2).min(pw - 1);
                    let x1 = (x * 2 + 1).min(pw - 1);
                    let y0 = (y * 2).min(ph - 1);
                    let y1 = (y * 2 + 1).min(ph - 1);
                    for c in 0..4 {
                        let idx = |px: u32, py: u32| ((py as usize) * (pw as usize) + px as usize) * 4 + c;
                        let sum = src[idx(x0, y0)] as u32
                            + src[idx(x1, y0)] as u32
                            + src[idx(x0, y1)] as u32
                            + src[idx(x1, y1)] as u32;
                        dst[((y as usize) * (w as usize) + x as usize) * 4 + c] = ((sum + 2) / 4) as u8;
                    }
                }
            }
        }
        chain.levels.push(out)?;
    }
    Ok(chain)
}

/// Builds a checkerboard RGBA8 image, the fixture the mip tests and the golden
/// scenes use.
pub fn checkerboard(width: u32, height: u32, cell: u32, a: [u8; 4], b: [u8; 4]) -> Vec<u8> {
    let cell = cell.max(1);
    let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y as usize) * (width as usize) + x as usize) * 4;
            let color = if ((x / cell) + (y / cell)) % 2 == 0 { a } else { b };
            out[idx..idx + 4].copy_from_slice(&color);
        }
    }
    out
}

/// Kept for callers that need the float conversion the rasteriser uses.
pub fn unorm8_to_f32(v: u8) -> f32 {
    v as f32 / 255.0
}

pub fn f32_to_unorm8(v: f32) -> u8 {
    to_u8(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_count_matches_the_chain() {
        assert_eq!(MipChain::level_count(1, 1), 1);
        assert_eq!(MipChain::level_count(2, 2), 2);
        assert_eq!(MipChain::level_count(4, 4), 3);
        assert_eq!(MipChain::level_count(8, 4), 4);
        assert_eq!(MipChain::level_count(1024, 1024), 11);
    }

    #[test]
    fn a_full_chain_halves_until_one_pixel() {
        let alloc = HostAlloc::system();
        // Cell size 1, so the first level's 2x2 footprint straddles both colours.
        let pixels = checkerboard(16, 16, 1, [255, 255, 255, 255], [0, 0, 0, 255]);
        let chain = generate_mip_chain(alloc, 16, 16, &pixels, 0).unwrap();
        assert_eq!(chain.levels.len(), 5);
        assert_eq!((chain.levels[0].width, chain.levels[0].height), (16, 16));
        assert_eq!((chain.levels[4].width, chain.levels[4].height), (1, 1));
        // A 2x2 checker averages to mid grey at the first level.
        assert_eq!(chain.levels[1].pixels.as_slice()[0], 128);
        assert_eq!(chain.levels[1].pixels.as_slice()[1], 128);
        assert_eq!(chain.levels[1].pixels.as_slice()[3], 255);
    }

    #[test]
    fn generation_is_deterministic_and_capped_by_the_request() {
        let alloc = HostAlloc::system();
        let pixels = checkerboard(32, 8, 1, [200, 10, 30, 255], [10, 200, 30, 128]);
        let a = generate_mip_chain(alloc, 32, 8, &pixels, 0).unwrap();
        let b = generate_mip_chain(alloc, 32, 8, &pixels, 0).unwrap();
        assert_eq!(a.levels.len(), b.levels.len());
        for (x, y) in a.levels.iter().zip(b.levels.iter()) {
            assert_eq!(x.pixels.as_slice(), y.pixels.as_slice());
        }
        let capped = generate_mip_chain(alloc, 32, 8, &pixels, 3).unwrap();
        assert_eq!(capped.levels.len(), 3);
    }

    #[test]
    fn short_uploads_are_refused() {
        let alloc = HostAlloc::system();
        let err = match generate_mip_chain(alloc, 4, 4, &[0u8; 8], 0) {
            Err(e) => e,
            Ok(_) => panic!("a short upload was accepted"),
        };
        assert_eq!(err.code, Code::InvalidArgument);
    }

    #[test]
    fn unorm_round_trip_is_lossless_for_every_byte() {
        for v in 0..=255u8 {
            assert_eq!(f32_to_unorm8(unorm8_to_f32(v)), v, "byte {v} did not round-trip");
        }
    }
}
