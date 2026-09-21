//! What this build actually has, and the two places vectorisation is allowed.
//!
//! Reported, not assumed: [`detect`] is the only thing that produces the
//! `RECONL_CAP_SIMD_*` bits, and the caps describe the *build*, not the machine.
//!
//! Only exact operations are vectorised - fill, copy, clear, checksum, and the
//! unorm8 conversion a readback performs - because those cannot change a pixel:
//! every input maps to one defined output byte. The shaded path stays scalar in
//! every tier so that the golden images are comparable across builds;
//! docs/determinism.md explains why that trade is deliberate rather than lazy.

use crate::to_u8;
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

/// Converts tightly packed RGBA `f32` to tightly packed RGBA8.
///
/// A readback pays this once per presented frame, and at 1080p the scalar form
/// was 8.6 million conversions per frame - the single largest part of the
/// reference tier's present. It is safe to vectorise because it is a
/// *conversion*, not shading: each input maps to the byte [`to_u8`] defines, so
/// no choice of instruction can move a pixel. The tests below prove the vector
/// path agrees with the scalar definition byte for byte, including for the
/// values that break naive conversions: NaN, signed zero, subnormals,
/// out-of-range values and the rounding boundary.
///
/// Writes `min(src.len(), dst.len()) / 4` whole pixels; a partial pixel at the
/// end of either slice is ignored, exactly as the scalar definition does.
pub fn rgba_f32_to_unorm8(src: &[f32], dst: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is part of the x86_64 baseline ABI, the loop bounds are
    // derived from both slice lengths, and both pointers are read/written only
    // inside those bounds.
    unsafe {
        rgba_f32_to_unorm8_sse2(src, dst)
    }
    #[cfg(not(target_arch = "x86_64"))]
    rgba_f32_to_unorm8_scalar(src, dst);
}

/// The definition every other implementation must reproduce.
pub fn rgba_f32_to_unorm8_scalar(src: &[f32], dst: &mut [u8]) {
    let pixels = (src.len() / 4).min(dst.len() / 4);
    for i in 0..pixels {
        for k in 0..4 {
            dst[i * 4 + k] = to_u8(src[i * 4 + k]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn rgba_f32_to_unorm8_sse2(src: &[f32], dst: &mut [u8]) {
    use core::arch::x86_64::*;

    let pixels = (src.len() / 4).min(dst.len() / 4);
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);
    let half = _mm_set1_ps(0.5);

    // One pixel's four channels: clamp into [0,1], scale, round half up by
    // truncating after adding 0.5. `maxps(v, 0)` returns the second operand for
    // a NaN input, which is the zero the scalar `is_nan` arm produces, so the
    // two agree on NaN as well as on every finite value.
    let quantise = |p: __m128| -> __m128i {
        let c = unsafe { _mm_min_ps(_mm_max_ps(p, zero), one) };
        unsafe { _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(c, scale), half)) }
    };

    let mut i = 0usize;
    while i + 4 <= pixels {
        let s = unsafe { src.as_ptr().add(i * 4) };
        let q0 = quantise(unsafe { _mm_loadu_ps(s) });
        let q1 = quantise(unsafe { _mm_loadu_ps(s.add(4)) });
        let q2 = quantise(unsafe { _mm_loadu_ps(s.add(8)) });
        let q3 = quantise(unsafe { _mm_loadu_ps(s.add(12)) });
        // Two 32->16 packs and one 16->8 pack interleave back into pixel order:
        // four pixels of RGBA leave in one 16-byte store.
        let words = unsafe { _mm_packus_epi16(_mm_packs_epi32(q0, q1), _mm_packs_epi32(q2, q3)) };
        unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(i * 4) as *mut __m128i, words) };
        i += 4;
    }
    rgba_f32_to_unorm8_scalar(&src[i * 4..], &mut dst[i * 4..]);
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

    /// Every value that can distinguish one conversion rule from another, plus
    /// the exact bytes the scalar definition produces for them.
    fn adversarial_components() -> Vec<f32> {
        let mut v = vec![
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            1.0e-45, // smallest positive subnormal
            -1.0e-45,
            0.5,
            0.5 - 1.0e-7,
            0.5 + 1.0e-7,
            1.0 / 255.0,
            1.0 / 510.0, // exactly the half-way rounding boundary
            254.0 / 255.0,
            1.0 - f32::EPSILON,
            f32::MIN_POSITIVE,
        ];
        // A spread of values whose 0.5 boundary lands inside and outside the
        // representable grid.
        let mut x = 0.123_456_7f32;
        for _ in 0..400 {
            x = (x * 1.618_034).fract().max(0.0);
            v.push(x);
            v.push(-x);
        }
        v
    }

    #[test]
    fn the_vector_path_agrees_with_the_scalar_definition() {
        let values = adversarial_components();
        for count in [0usize, 1, 3, 4, 5, 7, 8, 15, 16, 17, 64] {
            let src: Vec<f32> = (0..count * 4)
                .map(|i| values[i % values.len()])
                .collect();
            let mut fast = vec![0u8; count * 4];
            let mut defined = vec![0u8; count * 4];
            rgba_f32_to_unorm8(&src, &mut fast);
            rgba_f32_to_unorm8_scalar(&src, &mut defined);
            assert_eq!(fast, defined, "{count} pixels");
        }
    }

    #[test]
    fn the_conversion_matches_the_documented_rule() {
        let src = [0.0f32, 0.5, 1.0, -0.25, f32::NAN, 2.0, f32::NEG_INFINITY, 0.0];
        let mut out = vec![0u8; 8];
        rgba_f32_to_unorm8(&src, &mut out);
        assert_eq!(out, vec![0, 128, 255, 0, 0, 255, 0, 0]);
    }

    #[test]
    fn a_partial_pixel_is_ignored_rather_than_written_past_the_end() {
        let src = [0.25f32; 6];
        let mut out = vec![7u8; 4];
        rgba_f32_to_unorm8(&src, &mut out);
        assert_eq!(out, vec![64, 64, 64, 64]);
        let src = [0.25f32; 3];
        let mut out = vec![7u8; 4];
        rgba_f32_to_unorm8(&src, &mut out);
        assert_eq!(out, vec![7u8; 4], "no whole pixel, nothing written");
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
