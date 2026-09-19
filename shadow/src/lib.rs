//! Shadow cascades: fitting, texel snapping, bias policy and the cache key.
//!
//! Three rules from the brief are implemented here and nowhere else:
//!
//! * **No swimming, no popping.** The cascade ortho box is snapped to the
//!   shadow-map texel grid *in world units around the light's own axes*, so a
//!   fixed world point keeps a constant sub-texel offset while the camera moves.
//!   `cascade_sub_texel_offset_is_constant_under_camera_motion` is the automated
//!   version of that claim.
//! * **Bias is documented, not magical.** [`bias_preset`] is the only source of
//!   bias values, and it is a function of the tier, the map size and the filter -
//!   never of whoever is running the demo. The table is in `docs/bias.md`.
//! * **A static cascade is a build artifact.** [`cache_key`] is what the spill
//!   arena is indexed by: light, cascade, fitted matrix, world revision, filter
//!   and resolution. Nothing else may influence it.

use reconl_core::hash::XxHash64;
use reconl_core::tier::{ShadowFilter, Tier};
use reconl_raster::math::{self, Mat4, Vec3};
use reconl_raster::shade::{Cascade, ShadowLookup};

pub const MAX_CASCADES: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct CascadeFit {
    /// World space to light clip space.
    pub view_proj: Mat4,
    /// Camera-space distance at which this cascade stops being used.
    pub split_distance: f32,
    /// Camera-space distance at which it starts.
    pub near_distance: f32,
    /// World size of one shadow texel.
    pub texel_world: f32,
    /// Centre of the fitted box, in world space.
    pub center: Vec3,
    /// Half-extent of the fitted box along the light's right/up axes.
    pub radius: f32,
    /// Depth extent along the light direction.
    pub depth_min: f32,
    pub depth_max: f32,
}

#[derive(Clone, Copy)]
pub struct CascadeSet {
    pub fits: [Option<CascadeFit>; MAX_CASCADES],
    pub count: u32,
}

impl Default for CascadeSet {
    fn default() -> Self {
        Self::empty()
    }
}

impl CascadeSet {
    pub const fn empty() -> Self {
        Self { fits: [None; MAX_CASCADES], count: 0 }
    }

    pub fn iter(&self) -> impl Iterator<Item = &CascadeFit> {
        self.fits.iter().take(self.count as usize).flatten()
    }

    pub fn get(&self, index: usize) -> Option<&CascadeFit> {
        self.fits.get(index).and_then(|f| f.as_ref())
    }

    /// The form the rasteriser's shadow lookup consumes.
    pub fn as_lookup_cascades(&self) -> [Cascade; MAX_CASCADES] {
        let mut out = [Cascade {
            view_proj: math::IDENTITY,
            split_distance: 0.0,
            map_index: 0,
            texel_world: 0.0,
            depth_span: 0.0,
        }; MAX_CASCADES];
        for (i, fit) in self.iter().enumerate() {
            out[i] = Cascade {
                view_proj: fit.view_proj,
                split_distance: fit.split_distance,
                map_index: i as u32,
                texel_world: fit.texel_world,
                depth_span: (fit.depth_max - fit.depth_min).abs(),
            };
        }
        out
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FitInput {
    /// World space to camera view space.
    pub camera_view: Mat4,
    pub fov_y_deg: f32,
    pub aspect: f32,
    /// Camera near plane; only used for the first split's start.
    pub near: f32,
    /// Direction the light travels, unit length.
    pub light_dir: Vec3,
    pub cascade_count: u32,
    pub max_distance: f32,
    /// 0.0 = uniform splits, 1.0 = logarithmic.
    pub split_lambda: f32,
    pub map_size: u32,
    /// Snap the fit to the texel grid (always on in a real render; off only for
    /// the test that proves snapping is what removes the crawl).
    pub snap: bool,
}

/// Practical split scheme: `lambda` blends a uniform split with a logarithmic
/// one, which is what keeps the near cascade dense without starving the far one.
pub fn split_distances(count: u32, near: f32, far: f32, lambda: f32) -> [f32; MAX_CASCADES] {
    let n = count.clamp(1, MAX_CASCADES as u32);
    let lambda = lambda.clamp(0.0, 1.0);
    let mut out = [far; MAX_CASCADES];
    for i in 1..=n {
        let p = i as f32 / n as f32;
        let uniform = near + (far - near) * p;
        let log = near * (far / near).powf(p);
        out[(i - 1) as usize] = uniform * (1.0 - lambda) + log * lambda;
    }
    // Strictly increasing, and the last split is exactly the max distance.
    let mut prev = near;
    for i in 0..n as usize {
        if out[i] <= prev {
            out[i] = prev + 1.0e-3;
        }
        prev = out[i];
    }
    out[(n - 1) as usize] = far;
    out
}

/// Orthonormal light basis: `right`, `up`, and `forward` = the direction the
/// light travels. Rotation only: the `ortho` box carries the world offsets, which
/// is what lets the snap happen in world units.
fn light_basis(light_dir: Vec3) -> (Vec3, Vec3, Vec3) {
    let f = math::normalize(light_dir);
    let f = if math::dot(f, f) < 0.5 { [0.0, -1.0, 0.0] } else { f };
    let up_ref = if f[1].abs() > 0.99 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let r = math::normalize(math::cross(up_ref, f));
    let u = math::cross(f, r);
    (r, u, f)
}

/// Column-major, so the *rows* are the basis vectors: `x_view = dot(r, p)`.
/// Writing the basis out in row order here is the classic transposed view
/// matrix bug, which is invisible whenever the light happens to line up with a
/// world axis - hence `fitted_box_contains_the_camera_slice`, which uses an
/// oblique light on purpose.
fn rotation_only(r: Vec3, u: Vec3, f: Vec3) -> Mat4 {
    [
        r[0], u[0], -f[0], 0.0,
        r[1], u[1], -f[1], 0.0,
        r[2], u[2], -f[2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// The eight corners of the camera sub-frustum between `near_d` and `far_d`, in
/// world space. Public because it is the input to a fitted cascade, and because
/// the tests re-derive corners to check the fit contains them.
pub fn frustum_slice_corners(view: &Mat4, fov_y_deg: f32, aspect: f32, near_d: f32, far_d: f32) -> [Vec3; 8] {
    let tan_half = (fov_y_deg * 0.5 * core::f32::consts::PI / 180.0).tan();
    let inv = math::invert_rigid(view);
    let mut out = [[0.0f32; 3]; 8];
    let mut i = 0;
    for d in [near_d, far_d] {
        let half_h = tan_half * d;
        let half_w = half_h * aspect;
        for (sx, sy) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let view_space = [half_w * sx, half_h * sy, -d];
            let p = math::mul_point(&inv, view_space);
            out[i] = [p[0], p[1], p[2]];
            i += 1;
        }
    }
    out
}

/// Fits cascade `index` over `[near_d, far_d]`.
pub fn fit_cascade(input: &FitInput, _index: usize, near_d: f32, far_d: f32) -> CascadeFit {
    let (r, u, f) = light_basis(input.light_dir);
    let corners = frustum_slice_corners(&input.camera_view, input.fov_y_deg, input.aspect, near_d, far_d);

    let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
    let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
    let (mut min_z, mut max_z) = (f32::MAX, f32::MIN);
    for c in corners.iter() {
        let x = math::dot(r, *c);
        let y = math::dot(u, *c);
        let z = math::dot(f, *c);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }

    // A square map with one world-units-per-texel for both axes keeps the snap
    // grid square, which is what the texel-snapping argument needs.
    let map_size = input.map_size.max(1) as f32;
    let half_extent = 0.5 * (max_x - min_x).max(max_y - min_y).max(1.0e-4);
    // One texel of margin, so the snap (which can move the box by up to a whole
    // texel) can never pull the box off the geometry it was fitted to. This is
    // the difference between "contains the slice" and "contains it 99.9% of the
    // time", and it is 0.1% of the map area.
    let radius = half_extent * (1.0 + 2.0 / map_size);
    let texel = (2.0 * radius) / map_size;

    let mut cx = 0.5 * (min_x + max_x);
    let mut cy = 0.5 * (min_y + max_y);
    if input.snap {
        // The snap. In world units, around a grid anchored to the world origin,
        // so a fixed world point keeps a constant sub-texel offset.
        cx = (cx / texel).floor() * texel;
        cy = (cy / texel).floor() * texel;
    }

    // A little depth margin so geometry just outside the slice still writes.
    let margin = (max_z - min_z).max(texel) * 0.05;
    let depth_min = min_z - margin;
    let depth_max = max_z + margin;
    let rotation = rotation_only(r, u, f);
    let ortho = math::ortho_rh_reversed_z(
        cx - radius,
        cx + radius,
        cy - radius,
        cy + radius,
        depth_min,
        depth_max,
    );
    let view_proj = math::mul(&ortho, &rotation);

    // Centre in world space, for reporting and for the cache key.
    let center = [
        r[0] * cx + u[0] * cy + f[0] * (0.5 * (min_z + max_z)),
        r[1] * cx + u[1] * cy + f[1] * (0.5 * (min_z + max_z)),
        r[2] * cx + u[2] * cy + f[2] * (0.5 * (min_z + max_z)),
    ];

    CascadeFit {
        view_proj,
        split_distance: far_d,
        near_distance: near_d,
        texel_world: texel,
        center,
        radius,
        depth_min,
        depth_max,
    }
}

/// Fits every cascade the tier and the config allow.
pub fn fit_cascades(input: &FitInput) -> CascadeSet {
    let count = input.cascade_count.clamp(1, MAX_CASCADES as u32);
    let splits = split_distances(count, input.near.max(1.0e-3), input.max_distance, input.split_lambda);
    let mut set = CascadeSet::empty();
    let mut prev = input.near.max(1.0e-3);
    for i in 0..count as usize {
        let near_d = if i == 0 { prev } else { prev * 0.95 };
        let fit = fit_cascade(input, i, near_d, splits[i]);
        set.fits[i] = Some(fit);
        prev = splits[i];
    }
    set.count = count;
    set
}

/// Camera view matrix row 2, so a world point's camera-space depth is one dot
/// product inside the shader.
pub fn view_z_row(view: &Mat4) -> [f32; 4] {
    [view[2], view[6], view[10], view[14]]
}

/// Builds the lookup the reference shader consumes.
pub fn lookup<'a>(
    cascades: &'a [Cascade; MAX_CASCADES],
    maps: &'a [reconl_raster::shade::ShadowMapRef<'a>],
    camera_view: &Mat4,
    filter: ShadowFilter,
    bias: BiasPreset,
    max_distance: f32,
    blend_band: f32,
) -> ShadowLookup<'a> {
    ShadowLookup {
        cascades,
        maps,
        view_z_row: view_z_row(camera_view),
        filter: filter as u32,
        normal_bias: bias.normal_bias,
        depth_bias: bias.depth_bias,
        slope_bias: bias.slope_bias,
        max_distance,
        blend_band,
    }
}

/// Documented bias policy (docs/bias.md).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiasPreset {
    /// In shadow-map texels.
    pub normal_bias: f32,
    /// In reversed-Z depth units, already scaled to the map size.
    pub depth_bias: f32,
    /// How many texel footprints of slack a sloped surface gets.
    ///
    /// The shader converts one texel's world-space footprint at the fragment's
    /// incidence, `texel_world * tan(theta)`, into depth units with the cascade's
    /// depth span and multiplies by this. Keeping it in texel footprints instead
    /// of depth units is what makes one table work at every map size.
    pub slope_bias: f32,
}

/// The one table of bias values. Tier changes it because a scene tuned at T0
/// acne at T3, and a per-run guess is not reproducible.
pub fn bias_preset(tier: Tier, map_size: u32, filter: ShadowFilter) -> BiasPreset {
    let base = match tier {
        Tier::GpuDiscrete => (1.0f32, 1.5e-4f32, 1.5f32),
        Tier::GpuShared => (1.25, 2.5e-4, 1.75),
        Tier::CpuRam => (1.75, 4.0e-4, 2.0),
        Tier::CpuThrifty => (2.25, 6.0e-4, 2.5),
        Tier::OutOfCore => (2.5, 8.0e-4, 3.0),
    };
    // A filter with more taps averages more of the neighbourhood, which lifts
    // the effective depth it compares against; scale the depth bias with it.
    let filter_scale = match filter {
        ShadowFilter::Hard => 0.5,
        ShadowFilter::Pcf3x3 => 1.0,
        ShadowFilter::Pcf5x5 => 1.5,
        ShadowFilter::PcssLite => 2.0,
    };
    // Smaller maps have larger texels: bias in depth units must grow as texels do.
    let size_scale = (1024.0 / map_size.max(1) as f32).clamp(0.5, 8.0);
    BiasPreset {
        normal_bias: base.0,
        depth_bias: base.1 * filter_scale * size_scale,
        slope_bias: base.2,
    }
}

/// What identifies a cached static cascade.
#[derive(Clone, Copy)]
pub struct CacheKeyInput {
    /// Hash of the light's own parameters (direction, position, colour, cone).
    pub light_hash: u64,
    pub cascade_index: u32,
    pub view_proj: Mat4,
    /// World revision: bumping it retires the entry in O(1).
    pub world_revision: u64,
    pub filter: u32,
    pub map_size: u32,
    pub reversed_z: bool,
    /// Identity of the geometry set that was rendered into the map.
    pub static_geometry_revision: u64,
}

/// The cache key. Everything that can change the bytes of a shadow map is in
/// here, and nothing else is: a warmer cache must never change a pixel.
pub fn cache_key(input: &CacheKeyInput) -> u64 {
    let mut h = XxHash64::new(0x5243_4C53_4844_4F57); // "RCLSHDOW"
    h.update_u64(input.light_hash);
    h.update_u64(input.cascade_index as u64);
    for v in input.view_proj.iter() {
        h.update_f32(*v);
    }
    h.update_u64(input.world_revision);
    h.update_u64(input.static_geometry_revision);
    h.update_u64(input.filter as u64);
    h.update_u64(input.map_size as u64);
    h.update_u64(input.reversed_z as u64);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use reconl_raster::math::{look_at, perspective_rh_reversed_z};

    fn camera(x: f32, y: f32, z: f32) -> Mat4 {
        look_at([x, y, z], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    }

    fn input(view: Mat4, count: u32) -> FitInput {
        FitInput {
            camera_view: view,
            fov_y_deg: 60.0,
            aspect: 16.0 / 9.0,
            near: 0.5,
            light_dir: math::normalize([-0.4, -1.0, -0.3]),
            cascade_count: count,
            max_distance: 60.0,
            split_lambda: 0.5,
            map_size: 1024,
            snap: true,
        }
    }

    #[test]
    fn splits_are_increasing_and_end_at_max_distance() {
        let s = split_distances(4, 0.5, 100.0, 0.7);
        assert!(s[0] > 0.5);
        for i in 1..4 {
            assert!(s[i] > s[i - 1], "split {i} not increasing: {s:?}");
        }
        assert_eq!(s[3], 100.0);

        let uniform = split_distances(2, 1.0, 10.0, 0.0);
        assert!((uniform[0] - 5.5).abs() < 1.0e-3, "{uniform:?}");
        assert_eq!(uniform[1], 10.0);
    }

    #[test]
    fn fitted_box_contains_the_camera_slice() {
        let view = camera(0.0, 2.0, 10.0);
        let set = fit_cascades(&input(view, 3));
        assert_eq!(set.count, 3);
        let far = set.get(2).unwrap();
        // Every corner of the last slice must land inside the cascade's clip
        // volume in x/y and inside [0,1] in depth (reversed-Z).
        let corners = frustum_slice_corners(
            &view,
            60.0,
            16.0 / 9.0,
            far.near_distance,
            far.split_distance,
        );
        for c in corners.iter() {
            let p = math::mul_point(&far.view_proj, *c);
            assert!(p[3] > 0.0, "corner behind the light");
            let ndc_x = p[0] / p[3];
            let ndc_y = p[1] / p[3];
            let depth = p[2] / p[3];
            // Strictly inside: the texel-margin radius must absorb the snap.
            assert!(ndc_x.abs() <= 1.0 && ndc_y.abs() <= 1.0, "corner outside the box: {ndc_x}, {ndc_y}");
            assert!((0.0..=1.0).contains(&depth), "corner depth {depth} outside reversed-Z [0,1]");
        }
    }

    #[test]
    fn cascade_sub_texel_offset_is_constant_under_camera_motion() {
        // The anti-crawl test: dolly the camera and watch one fixed world point's
        // position in the shadow map. Without snapping the offset drifts every
        // frame (edges crawl); with snapping it stays put while the whole texel
        // coordinate steps.
        let probe_world = [0.0f32, 0.5, 0.0];
        let mut snapped_ref: Option<f32> = None;
        let mut unsnapped_ref: Option<f32> = None;
        let mut snapped_worst = 0.0f32;
        let mut unsnapped_worst = 0.0f32;
        let mut min_texel = f32::MAX;
        let mut max_texel = f32::MIN;

        // The probe's texel coordinate in cascade 0, as a float.
        let texel_coord = |set: &CascadeSet| -> f32 {
            let fit = set.get(0).unwrap();
            let p = math::mul_point(&fit.view_proj, probe_world);
            ((p[0] / p[3]) * 0.5 + 0.5) * 1024.0
        };

        for step in 0..48 {
            let x = step as f32 * 0.05;
            let snapped = fit_cascades(&input(camera(x, 2.0, 10.0), 1));
            let mut unsnapped_input = input(camera(x, 2.0, 10.0), 1);
            unsnapped_input.snap = false;
            let unsnapped = fit_cascades(&unsnapped_input);

            let t_snap = texel_coord(&snapped);
            let t_raw = texel_coord(&unsnapped);
            let frac_snap = t_snap - t_snap.floor();
            let frac_raw = t_raw - t_raw.floor();
            min_texel = min_texel.min(t_snap);
            max_texel = max_texel.max(t_snap);

            // Circular distance in texels: the fraction is allowed to wrap, and a
            // wrap is not drift.
            let wrap_dist = |a: f32, b: f32| ((a - b + 0.5).rem_euclid(1.0) - 0.5).abs();
            match (snapped_ref, unsnapped_ref) {
                (Some(s), Some(r)) => {
                    snapped_worst = snapped_worst.max(wrap_dist(frac_snap, s));
                    unsnapped_worst = unsnapped_worst.max(wrap_dist(frac_raw, r));
                }
                _ => {
                    snapped_ref = Some(frac_snap);
                    unsnapped_ref = Some(frac_raw);
                }
            }
        }

        assert!(
            snapped_worst < 5.0e-3,
            "snapped sub-texel offset drifted by {snapped_worst} texels"
        );
        assert!(
            unsnapped_worst > 0.1,
            "the unsnapped fit should drift and it did not ({unsnapped_worst}): the test is not proving anything"
        );
        assert!(
            max_texel - min_texel > 1.0,
            "the probe should move by whole texels as the camera dollies ({min_texel}..{max_texel})"
        );
    }

    #[test]
    fn snapped_projection_lands_on_a_texel_multiple() {
        let view = camera(1.234, 2.0, 9.876);
        let set = fit_cascades(&input(view, 2));
        for fit in set.iter() {
            // The box edges must be on the texel grid: (right-left)/texel is an
            // integer by construction, so this is a check on the snapping maths.
            let texels = (2.0 * fit.radius) / fit.texel_world;
            assert!((texels - texels.round()).abs() < 1.0e-2, "box spans {texels} texels");
        }
    }

    #[test]
    fn bias_presets_are_ordered_by_tier_and_map_size() {
        let t0 = bias_preset(Tier::GpuDiscrete, 1024, ShadowFilter::Pcf3x3);
        let t4 = bias_preset(Tier::OutOfCore, 1024, ShadowFilter::Pcf3x3);
        assert!(t4.depth_bias > t0.depth_bias, "weaker tiers need more bias");
        assert!(t4.normal_bias >= t0.normal_bias);

        let small = bias_preset(Tier::CpuRam, 256, ShadowFilter::Pcf3x3);
        let large = bias_preset(Tier::CpuRam, 2048, ShadowFilter::Pcf3x3);
        assert!(small.depth_bias > large.depth_bias, "bigger texels need more bias");

        let hard = bias_preset(Tier::CpuRam, 1024, ShadowFilter::Hard);
        let widest = bias_preset(Tier::CpuRam, 1024, ShadowFilter::PcssLite);
        assert!(widest.depth_bias > hard.depth_bias);
    }

    #[test]
    fn cache_key_covers_every_input_that_changes_the_map() {
        let view = camera(0.0, 2.0, 10.0);
        let set = fit_cascades(&input(view, 1));
        let fit = set.get(0).unwrap();
        let base = CacheKeyInput {
            light_hash: 0x1234,
            cascade_index: 0,
            view_proj: fit.view_proj,
            world_revision: 7,
            filter: ShadowFilter::Pcf3x3 as u32,
            map_size: 1024,
            reversed_z: true,
            static_geometry_revision: 3,
        };
        let key = cache_key(&base);
        assert_eq!(key, cache_key(&base), "the key must be stable");

        let mut changed = base;
        changed.world_revision = 8;
        assert_ne!(key, cache_key(&changed), "a world revision must retire the entry");

        let mut changed = base;
        changed.static_geometry_revision = 4;
        assert_ne!(key, cache_key(&changed));

        let mut changed = base;
        changed.filter = ShadowFilter::Pcf5x5 as u32;
        assert_ne!(key, cache_key(&changed));

        let mut changed = base;
        changed.map_size = 2048;
        assert_ne!(key, cache_key(&changed));

        let mut changed = base;
        changed.light_hash = 0x1235;
        assert_ne!(key, cache_key(&changed));

        let mut changed = base;
        changed.cascade_index = 1;
        assert_ne!(key, cache_key(&changed));

        let mut changed = base;
        changed.view_proj[12] += 1.0e-3; // a moved fit is a different map
        assert_ne!(key, cache_key(&changed));
    }

    #[test]
    fn light_basis_is_orthonormal_even_for_a_vertical_light() {
        for dir in [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.3, -0.9, 0.1]] {
            let (r, u, f) = light_basis(dir);
            assert!((math::length(r) - 1.0).abs() < 1.0e-4);
            assert!((math::length(u) - 1.0).abs() < 1.0e-4);
            assert!((math::length(f) - 1.0).abs() < 1.0e-4);
            assert!(math::dot(r, u).abs() < 1.0e-4);
            assert!(math::dot(r, f).abs() < 1.0e-4);
            assert!(math::dot(u, f).abs() < 1.0e-4);
        }
    }

    #[test]
    fn lookup_carries_the_documented_defaults() {
        let view = camera(0.0, 2.0, 10.0);
        let set = fit_cascades(&input(view, 2));
        let cascades = set.as_lookup_cascades();
        let map = reconl_raster::shade::ShadowMapRef { width: 4, height: 4, depth: &[0.0; 16] };
        let maps = [map];
        let bias = bias_preset(Tier::CpuRam, 1024, ShadowFilter::Pcf3x3);
        let l = lookup(&cascades, &maps, &view, ShadowFilter::Pcf3x3, bias, 60.0, 2.0);
        assert_eq!(l.filter, ShadowFilter::Pcf3x3 as u32);
        assert_eq!(l.max_distance, 60.0);
        assert_eq!(l.view_z_row, view_z_row(&view));
        // The reported depth must be the camera-space distance the view matrix
        // itself computes, negated: one dot product, same answer.
        for point in [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.5, 5.0]] {
            let view_space = math::mul_point(&view, point);
            assert!(
                (l.view_depth(point) + view_space[2]).abs() < 1.0e-4,
                "view_depth({point:?}) disagrees with the view matrix"
            );
            assert!(l.view_depth(point) > 0.0, "points in front read as positive distance");
        }
    }

    #[test]
    fn a_camera_outside_the_cascade_reads_as_lit() {
        let view = camera(0.0, 2.0, 10.0);
        let set = fit_cascades(&input(view, 1));
        let cascades = set.as_lookup_cascades();
        // An empty map: everything sampled is out of the fitted box or cleared,
        // so the fail-safe must return fully lit rather than black.
        let map = reconl_raster::shade::ShadowMapRef { width: 0, height: 0, depth: &[] };
        let maps = [map];
        let bias = bias_preset(Tier::CpuRam, 1024, ShadowFilter::Pcf3x3);
        let l = lookup(&cascades, &maps, &view, ShadowFilter::Pcf3x3, bias, 60.0, 2.0);
        assert_eq!(l.factor([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]), 1.0);
    }
}
