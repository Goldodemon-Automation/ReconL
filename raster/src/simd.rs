//! What this build actually has, and the two places vectorisation is allowed.
//!
//! Reported, not assumed: [`detect`] is the only thing that produces the
//! `RECONL_CAP_SIMD_*` bits, and the caps describe the *build*, not the machine.
//!
//! Only exact operations are vectorised - fill, copy, clear, checksum - because
//! those cannot change a pixel. The shaded path stays scalar in every tier so
//! that the golden images are comparable across builds; docs/determinism.md
//! explains why that trade is deliberate rather than lazy.

use reconl_core::tier::caps as cap_bits;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimdCaps {
    pub sse2: bool,
    pub avx2: bool,
    pub neon: bool,
    pub wasm_simd128: bool,
    /// Scalar fallback: always true, and the reference the others must match.
    pub scalar: bool,
}

pub fn detect() -> SimdCaps {
    let mut caps = SimdCaps { scalar: true, ..Default::default() };
    #[cfg(target_arch = "x86_64")]
    {
        caps.sse2 = std::arch::is_x86_feature_detected!("sse2");
        caps.avx2 = std::arch::is_x86_feature_detected!("avx2");
    }
    #[cfg(target_arch = "x86")]
    {
        caps.sse2 = std::arch::is_x86_feature_detected!("sse2");
        caps.avx2 = std::arch::is_x86_feature_detected!("avx2");
    }
    #[cfg(target_arch = "aarch64")]
    {
        caps.neon = true; // baseline on aarch64
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        caps.wasm_simd128 = true;
    }
    caps
}

/// The capability bits a backend advertises for this build.
pub fn cap_mask(caps: &SimdCaps) -> u32 {
    let mut mask = 0u32;
    if caps.sse2 {
        mask |= cap_bits::SIMD_SSE2;
    }
    if caps.avx2 {
        mask |= cap_bits::SIMD_AVX2;
    }
    if caps.neon {
        mask |= cap_bits::SIMD_NEON;
    }
    if caps.wasm_simd128 {
        mask |= cap_bits::SIMD_WASM128;
    }
    mask
}

/// `dst[i] = value`. Vectorised by the compiler when the build allows it; the
/// scalar loop is the definition.
pub fn fill_f32(dst: &mut [f32], value: f32) {
    for v in dst.iter_mut() {
        *v = value;
    }
}

/// `dst[i] = src[i]`, or `dst[i] = value` where `src` is `None`.
pub fn copy_or_fill(dst: &mut [f32], src: Option<&[f32]>, value: f32) {
    match src {
        Some(s) => {
            let n = dst.len().min(s.len());
            dst[..n].copy_from_slice(&s[..n]);
            for v in dst[n..].iter_mut() {
                *v = value;
            }
        }
        None => fill_f32(dst, value),
    }
}

/// The naive implementation the vectorised ones are checked against.
pub fn fill_f32_scalar(dst: &mut [f32], value: f32) {
    let mut i = 0;
    while i < dst.len() {
        dst[i] = value;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_is_at_least_scalar() {
        let caps = detect();
        assert!(caps.scalar);
        #[cfg(target_arch = "x86_64")]
        assert!(caps.sse2, "x86_64 baselines sse2");
    }

    #[test]
    fn cap_mask_matches_the_detected_set() {
        let caps = detect();
        let mask = cap_mask(&caps);
        assert_eq!(mask & cap_bits::SIMD_SSE2 != 0, caps.sse2);
        assert_eq!(mask & cap_bits::SIMD_AVX2 != 0, caps.avx2);
        assert_eq!(mask & cap_bits::SIMD_NEON != 0, caps.neon);
    }

    #[test]
    fn fill_matches_the_scalar_definition_exactly() {
        let mut a = vec![0.0f32; 1000];
        let mut b = vec![0.0f32; 1000];
        fill_f32(&mut a, -1.5);
        fill_f32_scalar(&mut b, -1.5);
        assert_eq!(a, b);
        // Bit patterns, not just values: -0.0 must survive as -0.0.
        fill_f32(&mut a, -0.0);
        assert!(a.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()));
    }

    #[test]
    fn copy_or_fill_pads_the_tail() {
        let mut dst = vec![1.0f32; 6];
        copy_or_fill(&mut dst, Some(&[2.0, 3.0]), 9.0);
        assert_eq!(dst, vec![2.0, 3.0, 9.0, 9.0, 9.0, 9.0]);
        copy_or_fill(&mut dst, None, 4.0);
        assert!(dst.iter().all(|v| *v == 4.0));
    }
}
