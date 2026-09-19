//! The reference scene: the one arrangement of geometry, light and camera that
//! the golden image, the cross-tier comparison and the tools all share.
//!
//! This module is the scene's only owner. `reconl-diff` renders it, `reconl-bench`
//! times it, and a comparison between two tools or two tiers is a comparison of
//! the same scene rather than of two scenes that agree today. Every constant
//! below is load-bearing for the golden, and the reasoning is kept next to it,
//! because the next person to change a number needs to know what it was doing.
//!
//! The one rule when editing: a change here changes `tests/golden` and the
//! cross-tier numbers, so it needs `reconl-diff compare` re-run and the
//! regenerated golden committed with the reason.

use crate::math;
use reconl::abi;

/// A vertex in the layout the reference pipeline declares: position, normal, UV,
/// colour, which is `ReconLVertex`'s layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub uv: [f32; 2],
    pub col: [f32; 4],
}

/// One draw: a vertex buffer and the indices into it.
#[derive(Clone, Debug)]
pub struct Chunk {
    pub verts: Vec<V>,
    pub indices: Vec<u32>,
}

/// The camera the frame is rendered from, and declared to the shadow system.
///
/// The ABI carries the *view* separately from the vertex transform because
/// cascade fitting needs an un-multiplied view, so these two must not drift: the
/// projection below is folded into `view_proj` for slot 0 and the view is passed
/// again in `ReconLCamera`.
#[derive(Clone, Copy, Debug)]
pub struct CameraSpec {
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
    pub eye_height: f32,
    pub eye_distance: f32,
}

impl CameraSpec {
    /// The pitched view, written out in closed form.
    ///
    /// With the eye at `(0, height, distance)` looking at the origin and up +Y,
    /// the basis is exact: right = +X, up = `(0, distance, -height)/L`, back =
    /// `(0, height, distance)/L`, translated `-L` along back, where `L` is the
    /// eye's distance from the origin. Writing it out keeps the tools on the
    /// published ABI alone, with no dependency on the renderer's maths.
    pub fn view(&self) -> [f32; 16] {
        let (height, distance) = (self.eye_height, self.eye_distance);
        let len = (height * height + distance * distance).sqrt();
        [
            1.0,
            0.0,
            0.0,
            0.0, //
            0.0,
            distance / len,
            height / len,
            0.0, //
            0.0,
            -height / len,
            distance / len,
            0.0, //
            0.0,
            0.0,
            -len,
            1.0,
        ]
    }

    /// The right-handed, reversed-Z projection - the engine's convention, and the
    /// same frustum the camera declares above.
    pub fn proj(&self, aspect: f32) -> [f32; 16] {
        math::perspective_rh_reversed_z(self.fov_y_deg, aspect, self.near, self.far)
    }

    /// Slot 0 of the constant buffer: projection * view, column-major.
    pub fn view_proj(&self, aspect: f32) -> [f32; 16] {
        math::mul(&self.proj(aspect), &self.view())
    }

    /// The camera as the frame descriptor carries it.
    pub fn abi(&self) -> abi::ReconLCamera {
        // SAFETY: `ReconLCamera` is plain data and every field is set below.
        let mut c: abi::ReconLCamera = unsafe { core::mem::zeroed() };
        c.base = crate::hdr::<abi::ReconLCamera>();
        c.view = self.view();
        c.fov_y_deg = self.fov_y_deg;
        c.near = self.near;
        c.far = self.far;
        c
    }
}

/// The one directional light, and whether it casts.
#[derive(Clone, Copy, Debug)]
pub struct LightSpec {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub cast_shadow: bool,
}

impl LightSpec {
    /// The light as the light list carries it.
    pub fn abi(&self) -> abi::ReconLLight {
        // SAFETY: plain data, every field set below.
        let mut l: abi::ReconLLight = unsafe { core::mem::zeroed() };
        l.base = crate::hdr::<abi::ReconLLight>();
        l.r#type = 0;
        l.cast_shadow = u32::from(self.cast_shadow);
        l.direction = self.direction;
        l.color = self.color;
        l.intensity = self.intensity;
        l.range = 0.0;
        l
    }
}

/// `--shadows`: the three modes a run can be measured in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowMode {
    /// No shadow pass and no shadow terms.
    Off,
    /// The default: the static cascade is re-rendered whenever the world moves.
    On,
    /// The static cascade is held across frames and, where the tier has one,
    /// backed by the disk arena. Whether the tier honours this shows up in the
    /// cache counters, not in an assumption.
    Cached,
}

impl ShadowMode {
    /// Parses `off`, `on` or `cached`.
    pub fn from_name(name: &str) -> Option<ShadowMode> {
        Some(match name {
            "off" | "0" => ShadowMode::Off,
            "on" | "1" => ShadowMode::On,
            "cached" => ShadowMode::Cached,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            ShadowMode::Off => "off",
            ShadowMode::On => "on",
            ShadowMode::Cached => "cached",
        }
    }
}

/// The shadow configuration a scene asks for.
#[derive(Clone, Copy, Debug)]
pub struct ShadowSpec {
    pub enabled: bool,
    pub cascade_count: u32,
    pub texel_budget_bytes: u64,
    pub filter: u32,
    pub max_distance: f32,
    pub blend_band: f32,
    pub normal_bias: f32,
    pub depth_bias: f32,
    pub slope_bias: f32,
    pub resolution_scale: f32,
    pub allow_disk_cache: bool,
    pub freeze_static_cascade: bool,
    pub refresh_interval_frames: u32,
}

impl ShadowSpec {
    /// The reference configuration, and the reason for each value:
    ///
    /// * **Two cascades over an 8 MiB texel budget**, which both tiers round to
    ///   512x512 maps (`CpuRam` scales budget by 0.25, `GpuShared` by 0.5), so the
    ///   tiers differ in rasteriser and not in shadow resolution. The shadow
    ///   straddles the split, so both cascades hold shadow and a wrong split or
    ///   bias is visible in a cascade's own territory.
    /// * **PCF 3x3**, both tiers' cap at that budget.
    /// * **`split_lambda` 0.75** (both tiers' default) puts the first split at
    ///   14.3 units, inside the shadow, which sits at 13.7..16.3 units from the
    ///   camera, with the 3-unit crossfade band at 12.8..15.8 across the middle. A
    ///   band over *lit* ground would blend two identical factors and prove
    ///   nothing.
    /// * **The bias triple is pinned** rather than left to the tier, and that is
    ///   what makes this a *reference* scene: the tier's own preset is larger on
    ///   software tiers by design, and that difference lands on a shadow's edge as
    ///   a pixel of coverage - the same scene through `d3d11` differs from the
    ///   reference on 220 pixels of 4096 with the presets in force and on 14 with
    ///   these pinned. Pinning them is what the ABI's bias fields are for.
    pub fn reference() -> ShadowSpec {
        ShadowSpec {
            enabled: true,
            cascade_count: 2,
            texel_budget_bytes: 8 << 20,
            filter: 1,
            max_distance: 77.0,
            blend_band: 3.0,
            normal_bias: 1.25,
            depth_bias: 5.0e-4,
            slope_bias: 1.75,
            resolution_scale: 1.0,
            allow_disk_cache: false,
            freeze_static_cascade: false,
            refresh_interval_frames: 1,
        }
    }

    /// The reference configuration in one of the three measured modes.
    pub fn for_mode(mode: ShadowMode) -> ShadowSpec {
        let mut spec = ShadowSpec::reference();
        match mode {
            ShadowMode::Off => spec.enabled = false,
            ShadowMode::On => {}
            ShadowMode::Cached => {
                spec.freeze_static_cascade = true;
                spec.allow_disk_cache = true;
                // The static cascade still refreshes periodically; the counters
                // say whether the tier reused it in between.
                spec.refresh_interval_frames = 60;
            }
        }
        spec
    }

    /// The configuration as the frame descriptor carries it.
    pub fn abi(&self) -> abi::ReconLShadowConfig {
        // SAFETY: plain data, every field set below.
        let mut s: abi::ReconLShadowConfig = unsafe { core::mem::zeroed() };
        s.base = crate::hdr::<abi::ReconLShadowConfig>();
        s.enabled = u32::from(self.enabled);
        s.cascade_count = self.cascade_count;
        s.texel_budget_bytes = self.texel_budget_bytes;
        s.filter = self.filter;
        s.max_distance = self.max_distance;
        s.blend_band = self.blend_band;
        s.normal_bias = self.normal_bias;
        s.depth_bias = self.depth_bias;
        s.slope_bias = self.slope_bias;
        s.resolution_scale = self.resolution_scale;
        s.allow_disk_cache = u32::from(self.allow_disk_cache);
        s.freeze_static_cascade = u32::from(self.freeze_static_cascade);
        s.refresh_interval_frames = self.refresh_interval_frames;
        s
    }
}

/// Everything a frame needs to be drawn.
#[derive(Clone, Debug)]
pub struct Scene {
    pub chunks: Vec<Chunk>,
    pub camera: CameraSpec,
    pub light: LightSpec,
    pub shadows: ShadowSpec,
}

/// The ground plane's half-extent: the quad reaches past the frame on every side,
/// so every ground pixel is a candidate receiver and the shadow's share of the
/// frame is its share of the visible ground.
const GROUND_HALF: f32 = 16.0;
/// The caster's height: its screen position and its shadow's differ by this much
/// parallax, so it does not hide what it casts.
const CASTER_HEIGHT: f32 = 4.3;
const CAMERA_HEIGHT: f32 = 13.0;
const CAMERA_DISTANCE: f32 = 10.0;

impl Scene {
    /// The reference scene: a shadowed, textured-catch triangle in front of a
    /// blue ground quad, under one low directional light that throws the
    /// shadow toward the camera.
    ///
    /// The arrangement is the point. A golden is only worth committing if the
    /// feature it is named for dominates it, so the shadow is made large rather
    /// than incidentally present - at 64x64 it covers ~695 pixels, or 17% of the
    /// frame against 18% of the visible ground, where the same scene under a
    /// vertical light covers a 35-pixel strip.
    pub fn reference() -> Scene {
        let up = [0.0, 1.0, 0.0];
        let blue = [0.25, 0.35, 1.0, 1.0];
        let white = [0.95, 0.95, 0.95, 1.0];
        let v = |p: [f32; 3], n: [f32; 3], c: [f32; 4]| V { pos: p, nrm: n, uv: [0.0, 0.0], col: c };

        let g = GROUND_HALF;
        let ground = Chunk {
            verts: vec![
                v([-g, 0.0, -g], up, blue),
                v([g, 0.0, -g], up, blue),
                v([g, 0.0, g], up, blue),
                v([-g, 0.0, g], up, blue),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        };

        // Wound like the ground, and that is not cosmetic: the shadow pass culls
        // the far side on the light's view, so a caster would be culled with it.
        //
        // Wide and shallow (11.8 units across, 3.6 deep) on purpose. Its screen
        // height is its depth times the sine of the camera's pitch, while the gap
        // between it and its own shadow is its height times the cosine - so a deep
        // caster is one that hides its own shadow. Width is bounded by the frame:
        // the shadow is cast at its own distance from the camera, where the window
        // is only so wide, so a wider caster throws a shadow the frame crops.
        let h = CASTER_HEIGHT;
        let caster = Chunk {
            verts: vec![
                v([-6.25, h, -3.9], up, white),
                v([5.55, h, -4.5], up, white),
                v([-0.35, h, -0.9], up, white),
            ],
            indices: vec![0, 1, 2],
        };

        Scene {
            chunks: vec![ground, caster],
            camera: CameraSpec {
                fov_y_deg: 45.0,
                near: 0.5,
                far: 100.0,
                eye_height: CAMERA_HEIGHT,
                eye_distance: CAMERA_DISTANCE,
            },
            // The low sun is where the shadow's size comes from: a shadow's area
            // on the ground is the caster's divided by cos of the light's angle to
            // the normal, so at 57 degrees the same caster throws 1.9x the area -
            // and the tilt is aimed toward the camera, which lands it where the
            // ground's pixels-per-unit is largest and separates it from the caster
            // on screen. A caster that must fit the frame caps its own shadow's
            // area, so tilting the light is the only axis left that grows it.
            light: LightSpec {
                direction: [0.10, -1.0, 1.55],
                color: [1.0, 1.0, 1.0],
                intensity: 1.0,
                cast_shadow: true,
            },
            shadows: ShadowSpec::reference(),
        }
    }

    /// The reference scene in a given shadow mode.
    pub fn with_shadow_mode(mode: ShadowMode) -> Scene {
        let mut scene = Scene::reference();
        scene.shadows = ShadowSpec::for_mode(mode);
        scene.light.cast_shadow = mode != ShadowMode::Off;
        scene
    }

    /// Total indexed vertices across every chunk: what a draw-call count and a
    /// triangle count are made of.
    pub fn triangle_count(&self) -> u32 {
        (self.chunks.iter().map(|c| c.indices.len()).sum::<usize>() / 3) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene's own definition of itself: one ground quad and one triangle,
    /// three triangles in total, under a light that casts.
    #[test]
    fn the_reference_scene_is_the_shape_it_documents() {
        let scene = Scene::reference();
        assert_eq!(scene.chunks.len(), 2);
        assert_eq!(scene.chunks[0].verts.len(), 4);
        assert_eq!(scene.chunks[1].verts.len(), 3);
        assert_eq!(scene.triangle_count(), 3, "two ground triangles plus the caster");
        assert!(scene.light.cast_shadow);
        assert!(scene.shadows.enabled);
        assert_eq!(scene.shadows.cascade_count, 2);
        // The caster is wound like the ground, and that is a correctness property
        // rather than a convention: the shadow pass culls the far side on the
        // light's view, so a caster wound the other way is culled with it and
        // silently stops casting. Both first triangles therefore turn the same
        // way about their own winding.
        // Both are planes seen from above, so the turn to compare is the one
        // about +Y: the cross product's Y component, which does not vanish for
        // either because both sets of vertices spread across X. (The X and Z
        // components do vanish - every vertex of each chunk shares one of them.)
        let turn = |chunk: &Chunk| -> f32 {
            let a = chunk.verts[chunk.indices[0] as usize].pos;
            let b = chunk.verts[chunk.indices[1] as usize].pos;
            let d = chunk.verts[chunk.indices[2] as usize].pos;
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            ab[2] * ad[0] - ab[0] * ad[2]
        };
        let (ground_turn, caster_turn) = (turn(&scene.chunks[0]), turn(&scene.chunks[1]));
        assert!(
            ground_turn * caster_turn > 0.0,
            "caster winding {caster_turn} must match the ground's {ground_turn}",
        );
    }

    /// The transform the golden depends on: a square frame has one focal length,
    /// `w` is the view distance, and reversed-Z really does put the near plane on
    /// 1.0 - checked by pushing points along the camera's own forward axis
    /// through the matrix the renderer will hand to the GPU.
    #[test]
    fn the_reference_transform_is_reversed_z_and_square_by_default() {
        let scene = Scene::reference();
        let proj = scene.camera.proj(1.0);
        let focal = 1.0 / (45.0f32 * 0.5 * core::f32::consts::PI / 180.0).tan();
        assert_eq!(proj[0], focal);
        assert_eq!(proj[5], focal, "a square frame has one focal length");
        assert_eq!(proj[11], -1.0, "w is the view distance");
        assert!(proj[10] > 0.0 && proj[14] > 0.0, "reversed-Z: near is 1, far is 0");

        let view = scene.camera.view();
        assert_eq!(view[0], 1.0, "right is +X");
        assert_eq!(view[12], 0.0);
        assert_eq!(view[15], 1.0);
        let len = (13.0f32 * 13.0 + 10.0 * 10.0).sqrt();
        assert!((view[14] + len).abs() < 1e-4, "translated back by the eye distance");

        // Push world points along the camera's forward axis through the actual
        // transform: depth 1 at the near plane, 0 at the far plane, monotonically
        // decreasing between them.
        let vp = scene.camera.view_proj(1.0);
        let forward = [0.0, -13.0 / len, -10.0 / len];
        let depth_at = |t: f32| {
            let p = [forward[0] * t, 13.0 + forward[1] * t, 10.0 + forward[2] * t];
            let row = |r: usize| vp[r] * p[0] + vp[4 + r] * p[1] + vp[8 + r] * p[2] + vp[12 + r];
            row(2) / row(3)
        };
        assert!((depth_at(0.5) - 1.0).abs() < 1e-4, "near -> {}", depth_at(0.5));
        assert!(depth_at(100.0).abs() < 1e-4, "far -> {}", depth_at(100.0));
        assert!(depth_at(1.0) > depth_at(10.0), "nearer is larger under COMPARE_GREATER");
    }

    #[test]
    fn the_three_shadow_modes_differ_only_in_shadows() {
        let off = Scene::with_shadow_mode(ShadowMode::Off);
        let on = Scene::with_shadow_mode(ShadowMode::On);
        let cached = Scene::with_shadow_mode(ShadowMode::Cached);
        assert!(!off.shadows.enabled && !off.light.cast_shadow);
        assert_eq!(on.shadows.abi().enabled, 1);
        assert_eq!(cached.shadows.abi().freeze_static_cascade, 1);
        assert_eq!(cached.shadows.abi().allow_disk_cache, 1);
        // Geometry and camera are identical across modes: a shadow mode changes
        // the shadow pass, never the scene.
        assert_eq!(off.chunks.len(), cached.chunks.len());
        assert_eq!(off.camera.view(), on.camera.view());
    }

    #[test]
    fn shadow_mode_names_round_trip() {
        for mode in [ShadowMode::Off, ShadowMode::On, ShadowMode::Cached] {
            assert_eq!(ShadowMode::from_name(mode.name()), Some(mode));
        }
        assert_eq!(ShadowMode::from_name("sometimes"), None);
    }
}
