//! Shading: the lighting model, the shadow lookup and the blend modes.
//!
//! Every formula here is the *reference* formula. The HLSL port is expected to
//! produce the same expression in the same order (docs/determinism.md,
//! "Shading"), which is why nothing here uses `mul_add`, reordered sums, or a
//! fast inverse square root.

use crate::math::{scale, sub, Vec3};
use crate::texture::Texture2D;

pub const MAX_LIGHTS: usize = 16;
pub const MAX_CASCADES: usize = 4;

pub const LIGHT_DIRECTIONAL: u32 = 0;
pub const LIGHT_SPOT: u32 = 1;
pub const LIGHT_POINT: u32 = 2;

pub const FILTER_HARD: u32 = 0;
pub const FILTER_PCF3X3: u32 = 1;
pub const FILTER_PCF5X5: u32 = 2;
pub const FILTER_PCSS_LITE: u32 = 3;

pub const BLEND_OPAQUE: u32 = 0;
pub const BLEND_ALPHA: u32 = 1;
pub const BLEND_ADDITIVE: u32 = 2;
pub const BLEND_MULTIPLY: u32 = 3;

/// `pcss-lite` penumbra constant: texels of blur per unit of relative blocker
/// distance. Fixed, documented, and not a tuning knob, because a filter that
/// changes with the weather cannot be diffed.
pub const PCSS_LITE_K: f32 = 16.0;
pub const PCSS_LITE_MAX_RADIUS: f32 = 4.0;

#[derive(Clone, Copy, Debug)]
pub struct Light {
    pub kind: u32,
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub cos_inner: f32,
    pub cos_outer: f32,
    pub cast_shadow: bool,
    /// Index into the shadow-cascade list, or `u32::MAX` when unshadowed.
    pub cascade_base: u32,
}

impl Light {
    pub fn directional(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            kind: LIGHT_DIRECTIONAL,
            position: [0.0; 3],
            direction,
            color,
            intensity,
            range: 0.0,
            cos_inner: 1.0,
            cos_outer: 1.0,
            cast_shadow: true,
            cascade_base: u32::MAX,
        }
    }
}

/// The lights a frame draws with, and the ambient term.
#[derive(Clone, Copy)]
pub struct LightSet {
    pub lights: [Option<Light>; MAX_LIGHTS],
    pub count: u32,
    pub ambient: Vec3,
}

impl Default for LightSet {
    fn default() -> Self {
        Self::new()
    }
}

impl LightSet {
    pub const fn new() -> Self {
        Self { lights: [None; MAX_LIGHTS], count: 0, ambient: [0.06, 0.07, 0.075] }
    }

    /// Returns false when the set is full: the caller counts the drop rather
    /// than silently rendering without the light.
    pub fn push(&mut self, light: Light) -> bool {
        if (self.count as usize) >= MAX_LIGHTS {
            return false;
        }
        self.lights[self.count as usize] = Some(light);
        self.count += 1;
        true
    }
}

/// A shadow map as the shader sees it: reversed-Z depth in `[0, 1]`.
#[derive(Clone, Copy)]
pub struct ShadowMapRef<'a> {
    pub width: u32,
    pub height: u32,
    pub depth: &'a [f32],
}

#[derive(Clone, Copy)]
pub struct Cascade {
    pub view_proj: [f32; 16],
    /// Camera-space distance at which this cascade stops being used.
    pub split_distance: f32,
    pub map_index: u32,
    /// World size of one shadow texel, for the normal-offset bias.
    pub texel_world: f32,
    /// World-space depth range this cascade's ortho volume spans.
    ///
    /// A slope-scaled depth bias is a *world-space* error (one texel's footprint
    /// foreshortened by `tan(theta)`); turning it into a reversed-Z depth bias
    /// needs the depth range the depth buffer covers, or the bias silently
    /// scales with the scene instead of with the map.
    pub depth_span: f32,
}

/// Everything the shadow lookup needs: the cascades, their maps, and the
/// documented bias policy.
#[derive(Clone, Copy)]
pub struct ShadowLookup<'a> {
    pub cascades: &'a [Cascade],
    pub maps: &'a [ShadowMapRef<'a>],
    /// Third row of the camera view matrix, for camera-space distance.
    pub view_z_row: [f32; 4],
    pub filter: u32,
    pub normal_bias: f32,
    pub depth_bias: f32,
    pub slope_bias: f32,
    pub max_distance: f32,
    pub blend_band: f32,
}

impl<'a> ShadowLookup<'a> {
    #[inline]
    pub fn view_depth(&self, world: Vec3) -> f32 {
        let z = self.view_z_row[0] * world[0] + self.view_z_row[1] * world[1] + self.view_z_row[2] * world[2] + self.view_z_row[3];
        -z
    }

    /// Which cascade covers this camera-space distance, if any.
    pub fn select_cascade(&self, distance: f32) -> Option<usize> {
        if self.cascades.is_empty() {
            return None;
        }
        for (i, c) in self.cascades.iter().enumerate() {
            if distance <= c.split_distance {
                return Some(i);
            }
        }
        Some(self.cascades.len() - 1)
    }

    fn map_of(&self, cascade: &Cascade) -> Option<&ShadowMapRef<'a>> {
        self.maps.get(cascade.map_index as usize)
    }

    /// Fraction of incoming light that reaches this fragment, in `[0, 1]`.
    ///
    /// `to_light` is the unit vector from the fragment toward the light; it is
    /// what makes the slope-scaled bias a function of the real incidence angle
    /// instead of a guess.
    ///
    /// 1.0 means fully lit and is also what every fail-safe returns: outside the
    /// cascade, beyond the max distance, or with a missing map.
    pub fn factor(&self, world: Vec3, normal: Vec3, to_light: Vec3) -> f32 {
        let distance = self.view_depth(world);
        if distance > self.max_distance {
            return 1.0;
        }
        let cascade_index = match self.select_cascade(distance) {
            Some(i) => i,
            None => return 1.0,
        };
        let cascade = &self.cascades[cascade_index];
        let map = match self.map_of(cascade) {
            Some(m) => m,
            None => return 1.0,
        };
        if map.depth.is_empty() || map.width == 0 || map.height == 0 {
            return 1.0;
        }

        let lit = self.factor_for(cascade, map, world, normal, to_light);

        // Cascade crossfade: inside the blend band before this cascade's split,
        // mix toward the next cascade so the seam does not pop.
        let band = self.blend_band.max(0.0);
        let outer = if band > 0.0 && distance > cascade.split_distance - band {
            let t = ((distance - (cascade.split_distance - band)) / band).clamp(0.0, 1.0);
            if let Some(next) = self.cascades.get(cascade_index + 1) {
                if let Some(next_map) = self.map_of(next) {
                    let f = self.factor_for(next, next_map, world, normal, to_light);
                    return lit + (f - lit) * t;
                }
            }
            // Last cascade: fade toward fully lit across the band.
            return lit + (1.0 - lit) * t;
        } else {
            lit
        };

        // Distance fade to unshadowed at max_distance.
        if band > 0.0 && distance > self.max_distance - band {
            let t = ((distance - (self.max_distance - band)) / band).clamp(0.0, 1.0);
            return outer + (1.0 - outer) * t;
        }
        outer
    }

    fn factor_for(&self, cascade: &Cascade, map: &ShadowMapRef<'a>, world: Vec3, normal: Vec3, to_light: Vec3) -> f32 {
        // Normal-offset bias first (moves the sample point off the surface),
        // then a slope-scaled depth bias, exactly as documented.
        let offset = scale(normal, self.normal_bias * cascade.texel_world.max(1.0e-6));
        let world = [world[0] + offset[0], world[1] + offset[1], world[2] + offset[2]];
        let m = &cascade.view_proj;
        let x = m[0] * world[0] + m[4] * world[1] + m[8] * world[2] + m[12];
        let y = m[1] * world[0] + m[5] * world[1] + m[9] * world[2] + m[13];
        let z = m[2] * world[0] + m[6] * world[1] + m[10] * world[2] + m[14];
        let w = m[3] * world[0] + m[7] * world[1] + m[11] * world[2] + m[15];
        if w.abs() < 1.0e-9 {
            return 1.0;
        }
        let ndc = [x / w, y / w, z / w];
        let uv = [ndc[0] * 0.5 + 0.5, 1.0 - (ndc[1] * 0.5 + 0.5)];
        if uv[0] < 0.0 || uv[0] > 1.0 || uv[1] < 0.0 || uv[1] > 1.0 {
            // Outside the cascade: unshadowed, which is the fail-safe.
            return 1.0;
        }

        let slope = slope_bias_term(normal, to_light);
        // Documented bias policy (docs/bias.md):
        //
        //     bias = depth_bias + slope_bias * texel_world * tan(theta) / depth_span
        //
        // The second term is the world-space depth error of a sloped texel -
        // `texel_world * tan(theta)` world units - converted into the depth units
        // of this cascade. Leaving it in world units would make the bias a
        // function of the scene's scale, and leaving the conversion out entirely
        // would make it a function of the map size instead: a 512-texel cascade
        // and a 4096-texel one would need different constants for the same
        // scene, which is exactly what the "one table, tier-aware" rule forbids.
        let texel_world = cascade.texel_world.max(1.0e-6);
        let depth_span = cascade.depth_span.abs().max(1.0e-6);
        let bias = self.depth_bias + self.slope_bias * texel_world * slope / depth_span;
        // Reversed-Z: larger depth is nearer the light, and a fragment is lit
        // when it is at least as near as the stored occluder (`tap`). So the
        // bias has to move the reference depth *toward* the light, which is the
        // larger direction - subtracting it would make every surface shadow
        // itself by exactly the bias, and the acne would grow with the bias
        // instead of shrinking.
        let reference = ndc[2] + bias;

        match self.filter {
            FILTER_HARD => self.tap(map, uv, 0.0, 0.0, reference),
            FILTER_PCF5X5 => self.pcf(map, uv, reference, 2),
            FILTER_PCSS_LITE => self.pcss_lite(map, uv, reference),
            _ => self.pcf(map, uv, reference, 1),
        }
    }

    /// One comparison: lit when the stored depth is at least the fragment depth.
    /// Reversed-Z: larger depth is nearer, so a nearer occluder means shadowed.
    #[inline]
    fn tap(&self, map: &ShadowMapRef<'a>, uv: [f32; 2], du: f32, dv: f32, reference: f32) -> f32 {
        let texel_u = du / map.width as f32;
        let texel_v = dv / map.height as f32;
        let u = uv[0] + texel_u;
        let v = uv[1] + texel_v;
        let x = ((u * map.width as f32) as i64).clamp(0, map.width as i64 - 1) as usize;
        let y = ((v * map.height as f32) as i64).clamp(0, map.height as i64 - 1) as usize;
        let idx = y * map.width as usize + x;
        let stored = match map.depth.get(idx) {
            Some(d) => *d,
            None => return 1.0,
        };
        // Reversed-Z: the fragment is lit when it is at least as near as the
        // stored surface, i.e. when its depth is the larger of the two.
        if reference + 1.0e-7 >= stored {
            1.0
        } else {
            0.0
        }
    }

    fn pcf(&self, map: &ShadowMapRef<'a>, uv: [f32; 2], reference: f32, radius: i32) -> f32 {
        let mut lit = 0.0f32;
        let mut taps = 0.0f32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                lit += self.tap(map, uv, dx as f32, dy as f32, reference);
                taps += 1.0;
            }
        }
        if taps == 0.0 {
            1.0
        } else {
            lit / taps
        }
    }

    /// `pcss-lite`: a blocker search, then a variable-radius PCF.
    ///
    /// It is an approximation of contact hardening and is never described as
    /// soft shadows. It is also T0/T1 only - the tier ladder upgrades the filter
    /// back down on the reference tiers.
    fn pcss_lite(&self, map: &ShadowMapRef<'a>, uv: [f32; 2], reference: f32) -> f32 {
        let mut blocker_sum = 0.0f32;
        let mut blockers = 0.0f32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let texel_u = dx as f32 / map.width as f32;
                let texel_v = dy as f32 / map.height as f32;
                let u = uv[0] + texel_u;
                let v = uv[1] + texel_v;
                let x = ((u * map.width as f32) as i64).clamp(0, map.width as i64 - 1) as usize;
                let y = ((v * map.height as f32) as i64).clamp(0, map.height as i64 - 1) as usize;
                if let Some(d) = map.depth.get(y * map.width as usize + x) {
                    // A blocker is nearer than the fragment, which in reversed-Z
                    // means a *larger* stored depth.
                    if *d > reference {
                        blocker_sum += *d;
                        blockers += 1.0;
                    }
                }
            }
        }
        if blockers == 0.0 {
            return 1.0;
        }
        let avg_blocker = blocker_sum / blockers;
        let width = (PCSS_LITE_K * (avg_blocker - reference).max(0.0) / avg_blocker.max(1.0e-4))
            .clamp(0.0, PCSS_LITE_MAX_RADIUS);
        // 5x5 taps scaled by the penumbra width, keeping the tap count fixed so
        // the cost does not depend on the scene.
        let mut lit = 0.0f32;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let fx = dx as f32 * width * 0.5;
                let fy = dy as f32 * width * 0.5;
                lit += self.tap(map, uv, fx, fy, reference);
            }
        }
        lit / 25.0
    }
}

/// Maximum slope-bias multiplier at grazing incidence. Fixed, documented, and
/// identical in the HLSL port: without it, one grazing triangle gets a bias
/// large enough to detach its whole shadow.
pub const MAX_SLOPE_BIAS: f32 = 8.0;

/// `tan(theta) = sqrt(1 - (N.L)^2) / (N.L)`, the classic slope-scaled bias term.
#[inline]
pub fn slope_bias_term(normal: Vec3, to_light: Vec3) -> f32 {
    let ndl = crate::math::dot(normal, to_light);
    if ndl <= 1.0e-4 {
        return MAX_SLOPE_BIAS;
    }
    let tan_theta = (1.0 - ndl * ndl).max(0.0).sqrt() / ndl;
    tan_theta.min(MAX_SLOPE_BIAS)
}

/// The reference surface shader: unlit / textured / Lambert / textured Lambert.
#[derive(Clone, Copy)]
pub struct SurfaceShader<'a> {
    pub textured: bool,
    pub lit: bool,
    pub receives_shadow: bool,
    pub texture: Option<Texture2D<'a>>,
    pub lights: Option<&'a LightSet>,
    pub shadows: Option<&'a ShadowLookup<'a>>,
    pub flip_normal: bool,
}

impl<'a> SurfaceShader<'a> {
    pub const fn unlit() -> Self {
        Self {
            textured: false,
            lit: false,
            receives_shadow: false,
            texture: None,
            lights: None,
            shadows: None,
            flip_normal: false,
        }
    }

    /// # Safety-relevant note: the attribute layout is position, normal, uv,
    /// colour (`crate::ATTR_*`).
    #[inline]
    pub fn shade(&self, attr: &[f32; crate::ATTR_COUNT], lod: f32, out: &mut [f32; 4]) {
        let world = [attr[crate::ATTR_POSITION], attr[crate::ATTR_POSITION + 1], attr[crate::ATTR_POSITION + 2]];
        let mut normal = [attr[crate::ATTR_NORMAL], attr[crate::ATTR_NORMAL + 1], attr[crate::ATTR_NORMAL + 2]];
        let uv = [attr[crate::ATTR_UV], attr[crate::ATTR_UV + 1]];
        let base = [
            attr[crate::ATTR_COLOR],
            attr[crate::ATTR_COLOR + 1],
            attr[crate::ATTR_COLOR + 2],
            attr[crate::ATTR_COLOR + 3],
        ];

        let mut texel = [1.0f32, 1.0, 1.0, 1.0];
        if self.textured {
            if let Some(t) = self.texture.as_ref() {
                t.sample(uv, lod, &mut texel);
            }
        }
        if self.flip_normal {
            normal = [-normal[0], -normal[1], -normal[2]];
        }

        let albedo = [base[0] * texel[0], base[1] * texel[1], base[2] * texel[2]];

        if !self.lit {
            out[0] = clamp01(albedo[0]);
            out[1] = clamp01(albedo[1]);
            out[2] = clamp01(albedo[2]);
            out[3] = clamp01(base[3] * texel[3]);
            return;
        }

        let lights = self.lights;
        let shadows = self.shadows;
        let mut lit = [0.0f32; 3];
        if let Some(ls) = lights {
            lit[0] = ls.ambient[0];
            lit[1] = ls.ambient[1];
            lit[2] = ls.ambient[2];
            let n = crate::math::normalize(normal);
            for slot in ls.lights.iter().take(ls.count as usize).flatten() {
                let (to_light, attenuation) = match slot.kind {
                    LIGHT_DIRECTIONAL => {
                        let d = crate::math::normalize([-slot.direction[0], -slot.direction[1], -slot.direction[2]]);
                        (d, 1.0)
                    }
                    LIGHT_POINT => {
                        let delta = sub(slot.position, world);
                        let d = crate::math::length(delta);
                        let dir = if d > 1.0e-6 { scale(delta, 1.0 / d) } else { [0.0, 1.0, 0.0] };
                        let atten = if slot.range <= 0.0 { 1.0 } else { clamp01(1.0 - d / slot.range) };
                        (dir, atten * atten)
                    }
                    _ => {
                        let delta = sub(slot.position, world);
                        let d = crate::math::length(delta);
                        let dir = if d > 1.0e-6 { scale(delta, 1.0 / d) } else { [0.0, 1.0, 0.0] };
                        let cos_theta = -crate::math::dot(crate::math::normalize(slot.direction), dir);
                        let inner = slot.cos_inner;
                        let outer = slot.cos_outer;
                        let spot = if inner - outer <= 1.0e-6 {
                            if cos_theta >= outer {
                                1.0
                            } else {
                                0.0
                            }
                        } else {
                            clamp01((cos_theta - outer) / (inner - outer))
                        };
                        let atten = if slot.range <= 0.0 { 1.0 } else { clamp01(1.0 - d / slot.range) };
                        (dir, atten * atten * spot)
                    }
                };
                let ndl = crate::math::dot(n, to_light).max(0.0);
                if ndl <= 0.0 {
                    continue;
                }
                let mut shadow = 1.0f32;
                if self.receives_shadow && slot.cast_shadow {
                    if let Some(s) = shadows {
                        shadow = s.factor(world, n, to_light);
                    }
                }
                let k = slot.intensity * ndl * attenuation * shadow;
                lit[0] += slot.color[0] * k;
                lit[1] += slot.color[1] * k;
                lit[2] += slot.color[2] * k;
            }
        }

        out[0] = clamp01(albedo[0] * lit[0]);
        out[1] = clamp01(albedo[1] * lit[1]);
        out[2] = clamp01(albedo[2] * lit[2]);
        out[3] = clamp01(base[3] * texel[3]);
    }
}

#[inline]
pub fn clamp01(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(0.0, 1.0)
    }
}

/// Blend `src` onto `dst` in place. Documented per mode; alpha is straight
/// (non-premultiplied) for [`BLEND_ALPHA`] and [`BLEND_MULTIPLY`].
///
/// Takes `&mut [f32]` so the rasteriser can hand it a four-element slice of the
/// target's pixel array without a copy.
#[inline]
pub fn blend(mode: u32, src: [f32; 4], dst: &mut [f32]) {
    match mode {
        BLEND_ALPHA => {
            let a = clamp01(src[3]);
            for i in 0..3 {
                dst[i] = clamp01(src[i] * a + dst[i] * (1.0 - a));
            }
            dst[3] = clamp01(a + dst[3] * (1.0 - a));
        }
        BLEND_ADDITIVE => {
            let a = clamp01(src[3]);
            for i in 0..3 {
                dst[i] = clamp01(dst[i] + src[i] * a);
            }
            dst[3] = clamp01(dst[3] + a);
        }
        BLEND_MULTIPLY => {
            let a = clamp01(src[3]);
            for i in 0..3 {
                let product = dst[i] * clamp01(src[i]);
                dst[i] = clamp01(dst[i] + (product - dst[i]) * a);
            }
        }
        _ => {
            dst[0] = clamp01(src[0]);
            dst[1] = clamp01(src[1]);
            dst[2] = clamp01(src[2]);
            dst[3] = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup<'a>(cascades: &'a [Cascade], maps: &'a [ShadowMapRef<'a>]) -> ShadowLookup<'a> {
        ShadowLookup {
            cascades,
            maps,
            // The row that maps world space to view-space z for a camera at the
            // origin looking down -z: view_z = z, so `view_depth` = -z.
            view_z_row: [0.0, 0.0, 1.0, 0.0],
            filter: FILTER_PCF3X3,
            normal_bias: 0.0,
            depth_bias: 0.0,
            slope_bias: 0.0,
            max_distance: 100.0,
            blend_band: 0.0,
        }
    }

    #[test]
    fn light_set_refuses_to_overflow_silently() {
        let mut ls = LightSet::new();
        for _ in 0..MAX_LIGHTS {
            assert!(ls.push(Light::directional([0.0, -1.0, 0.0], [1.0; 3], 1.0)));
        }
        assert!(!ls.push(Light::directional([0.0, -1.0, 0.0], [1.0; 3], 1.0)));
        assert_eq!(ls.count, MAX_LIGHTS as u32);
    }

    #[test]
    fn unlit_shading_is_albedo_times_texture() {
        let shader = SurfaceShader::unlit();
        let mut attr = [0.0f32; crate::ATTR_COUNT];
        attr[crate::ATTR_COLOR] = 0.5;
        attr[crate::ATTR_COLOR + 1] = 0.25;
        attr[crate::ATTR_COLOR + 2] = 1.5; // clamps
        attr[crate::ATTR_COLOR + 3] = 0.5;
        let mut out = [0.0; 4];
        shader.shade(&attr, 0.0, &mut out);
        assert_eq!(out, [0.5, 0.25, 1.0, 0.5]);
    }

    #[test]
    fn lambert_facing_light_is_brighter_than_away_from_it() {
        let mut lights = LightSet::new();
        lights.ambient = [0.0, 0.0, 0.0];
        lights.push(Light::directional([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 1.0));
        let shader = SurfaceShader {
            textured: false,
            lit: true,
            receives_shadow: false,
            texture: None,
            lights: Some(&lights),
            shadows: None,
            flip_normal: false,
        };
        let mut up = [0.0f32; crate::ATTR_COUNT];
        up[crate::ATTR_NORMAL + 1] = 1.0;
        up[crate::ATTR_COLOR] = 1.0;
        up[crate::ATTR_COLOR + 1] = 1.0;
        up[crate::ATTR_COLOR + 2] = 1.0;
        up[crate::ATTR_COLOR + 3] = 1.0;
        let mut out_up = [0.0; 4];
        shader.shade(&up, 0.0, &mut out_up);

        let mut down = up;
        down[crate::ATTR_NORMAL + 1] = -1.0;
        let mut out_down = [0.0; 4];
        shader.shade(&down, 0.0, &mut out_down);

        assert!(out_up[0] > 0.9, "lit surface: {:?}", out_up);
        assert!(out_down[0] < 0.1, "unlit surface: {:?}", out_down);
    }

    #[test]
    fn shadow_factor_is_one_when_no_cascade_covers_the_point() {
        let empty: [Cascade; 0] = [];
        let maps: [ShadowMapRef; 0] = [];
        let l = lookup(&empty, &maps);
        assert_eq!(l.factor([0.0, 0.0, -10.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]), 1.0);
        assert_eq!(l.view_depth([0.0, 0.0, -10.0]), 10.0);
    }

    #[test]
    fn packed_map_reports_lit_and_shadowed_correctly() {
        // A 2x2 map, reversed-Z: 1.0 is nearest.
        let map = ShadowMapRef { width: 2, height: 2, depth: &[1.0, 1.0, 1.0, 1.0] };
        let cascades = [Cascade {
            view_proj: crate::math::IDENTITY,
            split_distance: 50.0,
            map_index: 0,
            texel_world: 0.1,
            depth_span: 1.0,
        }];
        let maps = [map];
        let l = lookup(&cascades, &maps);
        // identity view_proj, point in the middle of the map, depth 0.5 < 1.0
        // -> occluded -> 0.0
        let factor = l.factor([0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        assert_eq!(factor, 0.0, "stored 1.0 occludes 0.5 in reversed-Z");

        // A fragment nearer than the map's contents is lit.
        assert_eq!(l.factor([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]), 1.0);
    }

    #[test]
    fn a_correct_depth_bias_never_makes_a_surface_shadow_itself() {
        // A fragment that *is* the stored depth must be lit whatever the bias
        // is. In reversed-Z the bias has to move the reference depth toward the
        // light (larger); subtracting it makes acne grow with the bias, which is
        // the failure mode this test exists to catch.
        let map = ShadowMapRef { width: 2, height: 2, depth: &[0.5, 0.5, 0.5, 0.5] };
        let cascades = [Cascade {
            view_proj: crate::math::IDENTITY,
            split_distance: 50.0,
            map_index: 0,
            texel_world: 0.1,
            depth_span: 1.0,
        }];
        let maps = [map];
        for depth_bias in [0.0f32, 1.0e-4, 1.0e-3, 1.0e-2] {
            let mut l = lookup(&cascades, &maps);
            l.depth_bias = depth_bias;
            assert_eq!(
                l.factor([0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
                1.0,
                "a surface at the stored depth was made to shadow itself (bias {})",
                depth_bias
            );
        }
        // The bias is a tolerance, not an eraser: a fragment genuinely behind
        // the stored surface is still shadowed.
        let mut l = lookup(&cascades, &maps);
        l.depth_bias = 0.02;
        assert_eq!(l.factor([0.0, 0.0, 0.4], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]), 0.0);
        // And it is bounded: a fragment nearer than the stored surface is lit.
        assert_eq!(l.factor([0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]), 1.0);
    }

    #[test]
    fn blend_modes_follow_the_documented_formulas() {
        let mut dst = [1.0, 1.0, 1.0, 1.0];
        blend(BLEND_ALPHA, [0.0, 0.0, 0.0, 0.5], &mut dst);
        assert!((dst[0] - 0.5).abs() < 1.0e-6);

        let mut dst = [0.2, 0.2, 0.2, 1.0];
        blend(BLEND_ADDITIVE, [0.4, 0.0, 0.0, 1.0], &mut dst);
        assert!((dst[0] - 0.6).abs() < 1.0e-6 && dst[1] == 0.2);

        let mut dst = [1.0, 1.0, 1.0, 1.0];
        blend(BLEND_MULTIPLY, [0.5, 0.5, 0.5, 1.0], &mut dst);
        assert!((dst[0] - 0.5).abs() < 1.0e-6);

        let mut dst = [0.0, 0.0, 0.0, 0.0];
        blend(BLEND_OPAQUE, [0.3, 0.2, 0.1, 0.25], &mut dst);
        assert_eq!(dst, [0.3, 0.2, 0.1, 1.0], "opaque forces alpha 1");
    }

    #[test]
    fn slope_bias_grows_with_incidence_and_is_bounded() {
        // Head-on: no slope term at all.
        assert!(slope_bias_term([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]) < 1.0e-6);
        // Grazing: bounded by MAX_SLOPE_BIAS rather than exploding.
        let grazing = slope_bias_term([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        assert_eq!(grazing, MAX_SLOPE_BIAS);
        // 45 degrees: tan(45) = 1
        let mid = slope_bias_term([0.0, 1.0, 0.0], crate::math::normalize([0.0, 1.0, -1.0]));
        assert!((mid - 1.0).abs() < 1.0e-3, "mid slope {mid}");
        // Back-facing: the N.L clamp keeps it finite.
        assert_eq!(slope_bias_term([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]), MAX_SLOPE_BIAS);
    }
}
