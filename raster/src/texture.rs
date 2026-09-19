//! Texture sampling.
//!
//! The feature floor is frozen (docs/determinism.md): RGBA8 sources, nearest and
//! bilinear filtering, point or linear mip selection, repeat/clamp/mirror wrap.
//! Anything else a backend might do is a declared cap, not a surprise.
//!
//! Wrapping and texel selection use integer arithmetic on the wrapped coordinate
//! so that a tap at u = 1.0 lands on the same texel in every run and every tier.

/// One mip level of one array slice, RGBA8, tightly packed rows.
#[derive(Clone, Copy)]
pub struct TextureLevel<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
}

impl<'a> TextureLevel<'a> {
    #[inline]
    pub fn texel(&self, x: u32, y: u32) -> [f32; 4] {
        let idx = ((y as usize) * (self.width as usize) + (x as usize)) * 4;
        if idx + 3 >= self.data.len() {
            return [0.0, 0.0, 0.0, 1.0];
        }
        let d = &self.data[idx..idx + 4];
        [
            d[0] as f32 / 255.0,
            d[1] as f32 / 255.0,
            d[2] as f32 / 255.0,
            d[3] as f32 / 255.0,
        ]
    }
}

/// ReconLSamplerFilter / ReconLSamplerWrap values, as plain numbers.
pub const FILTER_NEAREST: u32 = 0;
pub const FILTER_LINEAR: u32 = 1;
pub const FILTER_NEAREST_MIP_LINEAR: u32 = 2;
pub const FILTER_LINEAR_MIP_LINEAR: u32 = 3;

pub const WRAP_REPEAT: u32 = 0;
pub const WRAP_CLAMP: u32 = 1;
pub const WRAP_MIRROR: u32 = 2;

#[derive(Clone, Copy)]
pub struct Texture2D<'a> {
    pub levels: &'a [TextureLevel<'a>],
    pub filter: u32,
    pub wrap_u: u32,
    pub wrap_v: u32,
    pub mip_lod_bias: f32,
    /// Highest level this view may sample.
    ///
    /// Under a RAM cap ReconL stops *promoting* upper mips out of the spill
    /// arena and clamps here instead. A coarser mip is a quality cap that is
    /// reported in `ReconLStats`; it is never a wrong pixel, and it is the
    /// difference between the out-of-core tier degrading and failing.
    pub max_lod: f32,
}

impl<'a> Texture2D<'a> {
    pub const fn none() -> Self {
        Self {
            levels: &[],
            filter: FILTER_LINEAR,
            wrap_u: WRAP_REPEAT,
            wrap_v: WRAP_REPEAT,
            mip_lod_bias: 0.0,
            max_lod: f32::MAX,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    pub fn mip_count(&self) -> u32 {
        self.levels.len() as u32
    }

    /// Normalises a coordinate into `[0, size)`.
    #[inline]
    fn wrap(&self, coord: f32, mode: u32) -> f32 {
        if !coord.is_finite() {
            return 0.0;
        }
        match mode {
            WRAP_CLAMP => coord.clamp(0.0, 1.0),
            WRAP_MIRROR => {
                let t = coord.rem_euclid(2.0);
                if t > 1.0 {
                    2.0 - t
                } else {
                    t
                }
            }
            _ => coord - coord.floor(),
        }
    }

    /// Maps an out-of-range texel index back into `[0, size)` for a wrap mode.
    /// Bilinear taps need this per-neighbour, not once per coordinate.
    #[inline]
    fn wrap_texel(i: i64, size: i64, mode: u32) -> i64 {
        if size <= 0 {
            return 0;
        }
        match mode {
            WRAP_CLAMP => i.clamp(0, size - 1),
            WRAP_MIRROR => {
                let period = 2 * size;
                let m = i.rem_euclid(period);
                if m >= size {
                    period - 1 - m
                } else {
                    m
                }
            }
            _ => i.rem_euclid(size),
        }
    }

    #[inline]
    fn texel_of(&self, uv: [f32; 2], level: usize) -> [f32; 4] {
        let lvl = &self.levels[level];
        let u = self.wrap(uv[0], self.wrap_u);
        let v = self.wrap(uv[1], self.wrap_v);
        let x = ((u * lvl.width as f32) as i64).clamp(0, lvl.width as i64 - 1) as u32;
        let y = ((v * lvl.height as f32) as i64).clamp(0, lvl.height as i64 - 1) as u32;
        lvl.texel(x, y)
    }

    #[inline]
    fn bilinear(&self, uv: [f32; 2], level: usize) -> [f32; 4] {
        let lvl = &self.levels[level];
        let u = self.wrap(uv[0], self.wrap_u);
        let v = self.wrap(uv[1], self.wrap_v);
        let fx = u * lvl.width as f32 - 0.5;
        let fy = v * lvl.height as f32 - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let sample = |x: i64, y: i64| -> [f32; 4] {
            let x = Self::wrap_texel(x, lvl.width as i64, self.wrap_u) as u32;
            let y = Self::wrap_texel(y, lvl.height as i64, self.wrap_v) as u32;
            lvl.texel(x, y)
        };
        let c00 = sample(x0, y0);
        let c10 = sample(x0 + 1, y0);
        let c01 = sample(x0, y0 + 1);
        let c11 = sample(x0 + 1, y0 + 1);
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            let top = c00[i] + (c10[i] - c00[i]) * tx;
            let bottom = c01[i] + (c11[i] - c01[i]) * tx;
            out[i] = top + (bottom - top) * ty;
        }
        out
    }

    /// Samples `uv` at an explicit level of detail. Returns black-transparent
    /// when the texture has no levels, which is a cap, not an error.
    pub fn sample(&self, uv: [f32; 2], lod: f32, out: &mut [f32; 4]) {
        if self.levels.is_empty() {
            *out = [0.0, 0.0, 0.0, 0.0];
            return;
        }
        let point_filter = self.filter == FILTER_NEAREST || self.filter == FILTER_LINEAR;
        let point_mip = self.filter == FILTER_NEAREST || self.filter == FILTER_NEAREST_MIP_LINEAR;
        let resident = (self.levels.len() - 1) as f32;
        let cap = self.max_lod.min(resident);
        let max_level = (cap.floor().clamp(0.0, resident)) as usize;
        let lod = (lod + self.mip_lod_bias).max(0.0).min(cap);

        if point_filter {
            // No mip chain: level 0 only, like the frozen feature floor says.
            let level = 0usize;
            *out = if point_mip { self.texel_of(uv, level) } else { self.bilinear(uv, level) };
            return;
        }

        let l0 = (lod.floor() as i64).clamp(0, max_level as i64) as usize;
        if point_mip {
            *out = self.texel_of(uv, l0);
            return;
        }
        let l1 = (l0 + 1).min(max_level);
        let a = self.bilinear(uv, l0);
        if l1 == l0 {
            *out = a;
            return;
        }
        let t = (lod - l0 as f32).clamp(0.0, 1.0);
        let b = self.bilinear(uv, l1);
        for i in 0..4 {
            out[i] = a[i] + (b[i] - a[i]) * t;
        }
    }
}

/// Level of detail from screen-space uv derivatives, the `log2(max(...))` rule
/// from the docs. `du_dx`, `dv_dx`, `du_dy`, `dv_dy` are in texels-per-pixel.
#[inline]
pub fn lod_from_derivatives(du_dx: f32, dv_dx: f32, du_dy: f32, dv_dy: f32, texels_u: f32, texels_v: f32) -> f32 {
    let rho_x = (du_dx * texels_u).abs().max((dv_dx * texels_v).abs());
    let rho_y = (du_dy * texels_u).abs().max((dv_dy * texels_v).abs());
    let rho = rho_x.max(rho_y);
    if rho <= 1.0e-6 {
        0.0
    } else {
        rho.log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker() -> Vec<u8> {
        // 2x2: red, green / blue, white
        vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ]
    }

    // Leaked deliberately: these are process-lifetime fixtures, so the borrow
    // checker sees `'static` levels the way a real texture upload would.
    fn tex(data: Vec<u8>, filter: u32, wrap: u32) -> Texture2D<'static> {
        let level: &'static TextureLevel<'static> = Box::leak(Box::new(TextureLevel {
            width: 2,
            height: 2,
            data: Box::leak(data.into_boxed_slice()),
        }));
        let levels: &'static [TextureLevel<'static>] = Box::leak(vec![*level].into_boxed_slice());
        Texture2D { levels, filter, wrap_u: wrap, wrap_v: wrap, mip_lod_bias: 0.0, max_lod: f32::MAX }
    }

    #[test]
    fn nearest_picks_the_expected_texel() {
        let t = tex(checker(), FILTER_NEAREST, WRAP_CLAMP);
        let mut out = [0.0; 4];
        t.sample([0.25, 0.25], 0.0, &mut out);
        assert_eq!(out, [1.0, 0.0, 0.0, 1.0], "top-left is red");
        t.sample([0.75, 0.25], 0.0, &mut out);
        assert_eq!(out, [0.0, 1.0, 0.0, 1.0], "top-right is green");
        t.sample([0.75, 0.75], 0.0, &mut out);
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0], "bottom-right is white");
    }

    #[test]
    fn bilinear_blends_two_texels_evenly_at_the_seam() {
        let t = tex(checker(), FILTER_LINEAR, WRAP_CLAMP);
        let mut out = [0.0; 4];
        t.sample([0.5, 0.01], 0.0, &mut out);
        assert!((out[0] - 0.5).abs() < 1.0e-3, "red {}", out[0]);
        assert!((out[1] - 0.5).abs() < 1.0e-3, "green {}", out[1]);
    }

    #[test]
    fn repeat_wrap_is_periodic_and_clamp_is_not() {
        let r = tex(checker(), FILTER_NEAREST, WRAP_REPEAT);
        let c = tex(checker(), FILTER_NEAREST, WRAP_CLAMP);
        let mut a = [0.0; 4];
        let mut b = [0.0; 4];
        r.sample([1.25, 0.25], 0.0, &mut a);
        r.sample([0.25, 0.25], 0.0, &mut b);
        assert_eq!(a, b, "repeat must wrap");
        c.sample([1.25, 0.25], 0.0, &mut a);
        c.sample([0.99, 0.25], 0.0, &mut b);
        assert_eq!(a, b, "clamp must saturate");
    }

    #[test]
    fn mirror_wrap_folds_back() {
        let t = Texture2D {
            levels: tex(checker(), FILTER_NEAREST, WRAP_MIRROR).levels,
            filter: FILTER_NEAREST,
            wrap_u: WRAP_MIRROR,
            wrap_v: WRAP_MIRROR,
            mip_lod_bias: 0.0,
            max_lod: f32::MAX,
        };
        let mut a = [0.0; 4];
        let mut b = [0.0; 4];
        t.sample([0.25, 0.25], 0.0, &mut a);
        t.sample([1.75, 0.25], 0.0, &mut b);
        assert_eq!(a, b, "mirror of 1.75 is 0.25");
        let mut out = [0.0; 4];
        t.sample([-0.25, 0.25], 0.0, &mut out);
        assert_eq!(out, a, "mirror of -0.25 is 0.25");
    }

    #[test]
    fn lod_rule_matches_the_documented_formula() {
        // one texel of uv per pixel on a 256-texel texture -> lod 8
        let lod = lod_from_derivatives(1.0 / 256.0, 0.0, 0.0, 1.0 / 256.0, 256.0, 256.0);
        assert!((lod - 0.0).abs() < 1.0e-4, "lod {lod}");
        let lod = lod_from_derivatives(1.0, 0.0, 0.0, 1.0, 256.0, 256.0);
        assert!((lod - 8.0).abs() < 1.0e-3, "lod {lod}");
        assert_eq!(lod_from_derivatives(0.0, 0.0, 0.0, 0.0, 512.0, 512.0), 0.0);
    }

    #[test]
    fn the_lod_cap_clamps_trilinear_selection() {
        // Two levels: without a cap, a large lod would sample level 1.
        let data = checker();
        let base: &'static TextureLevel<'static> = Box::leak(Box::new(TextureLevel {
            width: 2,
            height: 2,
            data: Box::leak(data.into_boxed_slice()),
        }));
        let half: &'static TextureLevel<'static> = Box::leak(Box::new(TextureLevel {
            width: 1,
            height: 1,
            data: Box::leak(vec![255u8, 255, 255, 255].into_boxed_slice()),
        }));
        let levels: &'static [TextureLevel<'static>] = Box::leak(vec![*base, *half].into_boxed_slice());
        let mut t = Texture2D { levels, filter: FILTER_NEAREST_MIP_LINEAR, wrap_u: WRAP_CLAMP, wrap_v: WRAP_CLAMP, mip_lod_bias: 0.0, max_lod: f32::MAX };
        let mut out = [0.0; 4];
        t.sample([0.25, 0.25], 4.0, &mut out);
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0], "uncapped: the top mip is white");
        t.max_lod = 0.0;
        t.sample([0.25, 0.25], 4.0, &mut out);
        assert_eq!(out, [1.0, 0.0, 0.0, 1.0], "capped to level 0: the red texel");
    }

    #[test]
    fn empty_texture_samples_transparent_black() {
        let t = Texture2D::none();
        let mut out = [1.0; 4];
        t.sample([0.5, 0.5], 0.0, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }
}
