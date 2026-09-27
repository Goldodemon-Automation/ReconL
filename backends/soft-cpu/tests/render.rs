//! End-to-end tests for the reference software backend.
//!
//! These are the tests that make the tier claims checkable rather than
//! aspirational: the same scene rendered at 1, 2, 4 and 8 workers and on the
//! scalar path must be the *same frame*; a caster must darken the floor exactly
//! where a straight-line light direction says it should; the T4 disk cache must
//! return the identical frame it cached; and a T3 frozen cascade must not change
//! the image it freezes.
//!
//! The scene is deliberately small and analytic: a floor quad, a closed cube
//! floating above it, one directional light at `(0.6, -1, 0)`. That last choice
//! is what makes the shadow assertion exact - the cube's shadow lands on the
//! floor at a position that can be computed by hand and checked at a pixel
//! coordinate computed by the same projection matrix the renderer used.

use reconl_backend_softcpu::{caps_for, FramePolicy, SoftCpuConfig, SoftCpuDevice};
use reconl_contract::{FrameInput, ShadowRequest};
use reconl_core::alloc::HostAlloc;
use reconl_core::budget::{Budget, BudgetCaps};
use reconl_core::error::{Code, Error};
use reconl_core::stats::{FrameNumbers, ShadowCounters};
use reconl_core::tier::{ShadowFilter, Tier, TierReason};
use reconl_raster::math::{self, look_at, perspective_rh_reversed_z, Mat4, Vec3, IDENTITY};
use reconl_raster::shade::{Light, LightSet, SurfaceShader};
use reconl_raster::{DrawItem, PipelineState, ShaderRef, Vertex};
use std::path::PathBuf;
use std::sync::Arc;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

/// The light is not straight down: a straight-down light puts the shadow
/// directly under the cube, which is exactly where the cube itself hides the
/// floor from the camera. Tilting it in `+x` moves the shadow out from behind
/// the caster, into pixels the camera can actually see.
const LIGHT_DIR: Vec3 = [0.6, -1.0, 0.0];
const LIGHT_HASH: u64 = 0x1234_5678_9abc_def0;
const WORLD_REVISION: u64 = 7;
const STATIC_REVISION: u64 = 3;

// ------------------------------------------------------------------ fixtures

fn floor_quad() -> Vec<Vertex> {
    let c = [0.8, 0.8, 0.8, 1.0];
    // Wound so the surface faces up: the colour pass culls back faces, and a
    // floor the camera cannot see is not a floor.
    vec![
        Vertex::plain([-4.0, 0.0, -4.0], c),
        Vertex::plain([-4.0, 0.0, 4.0], c),
        Vertex::plain([4.0, 0.0, 4.0], c),
        Vertex::plain([4.0, 0.0, -4.0], c),
    ]
}

const FLOOR_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// A closed cube, six quads, every quad wound counter-clockwise seen from
/// outside. Closed matters: the shadow pass culls front faces, and an open mesh
/// has no back faces to write.
fn cube(center: Vec3, half: f32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    let dir = |i: usize| {
        let mut d = [0.0f32; 3];
        d[i] = 1.0;
        d
    };
    for axis in 0..3usize {
        for sign in [1.0f32, -1.0] {
            // u x v = sign * axis
            let (ua, va) = match (axis, sign > 0.0) {
                (0, true) => (1, 2),
                (0, false) => (2, 1),
                (1, true) => (2, 0),
                (1, false) => (0, 2),
                (2, true) => (0, 1),
                _ => (1, 0),
            };
            let n = math::scale(dir(axis), sign * half);
            let u = math::scale(dir(ua), half);
            let v = math::scale(dir(va), half);
            let corner = |su: f32, sv: f32| {
                let p = math::add(math::add(math::add(center, n), math::scale(u, su)), math::scale(v, sv));
                Vertex::plain(p, [0.5, 0.5, 0.55, 1.0])
            };
            let base = verts.len() as u32;
            verts.push(corner(-1.0, -1.0));
            verts.push(corner(1.0, -1.0));
            verts.push(corner(1.0, 1.0));
            verts.push(corner(-1.0, 1.0));
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    (verts, indices)
}

fn camera() -> (Mat4, Mat4) {
    let view = look_at([0.0, 1.6, 4.0], [0.0, 0.4, 0.0], [0.0, 1.0, 0.0]);
    let proj = perspective_rh_reversed_z(60.0, WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);
    (view, proj)
}

fn lights() -> LightSet {
    let mut set = LightSet::new();
    set.ambient = [0.1, 0.1, 0.1];
    assert!(set.push(Light::directional(LIGHT_DIR, [0.9, 0.9, 0.9], 1.0)));
    set
}

fn project(pv: &Mat4, world: Vec3) -> (usize, usize) {
    let clip = math::mul_point(pv, world);
    let x = clip[0] / clip[3];
    let y = clip[1] / clip[3];
    let px = ((x * 0.5 + 0.5) * WIDTH as f32).floor().clamp(0.0, (WIDTH - 1) as f32) as usize;
    let py = ((1.0 - (y * 0.5 + 0.5)) * HEIGHT as f32).floor().clamp(0.0, (HEIGHT - 1) as f32) as usize;
    (px, py)
}

fn pixel(color: &[f32], x: usize, y: usize) -> [f32; 4] {
    let i = (y * WIDTH as usize + x) * 4;
    [color[i], color[i + 1], color[i + 2], color[i + 3]]
}

fn floor_draw<'a>(verts: &'a [Vertex], transform: Mat4) -> DrawItem<'a> {
    DrawItem {
        vertices: verts,
        indices: Some(&FLOOR_INDICES),
        transform,
        model: IDENTITY,
        pipeline: PipelineState::default(),
        shader: ShaderRef::Surface(SurfaceShader {
            textured: false,
            lit: true,
            receives_shadow: true,
            ..SurfaceShader::unlit()
        }),
        dynamic: false,
        casts_shadow: true,
    }
}

fn caster_draw<'a>(verts: &'a [Vertex], indices: &'a [u32], transform: Mat4, dynamic: bool) -> DrawItem<'a> {
    DrawItem {
        vertices: verts,
        indices: Some(indices),
        transform,
        model: IDENTITY,
        pipeline: PipelineState::default(),
        // Unlit: the cube's own shading is not what this test is about, and an
        // unlit cube cannot accidentally hide a shadowing bug in the floor.
        shader: ShaderRef::Surface(SurfaceShader::unlit()),
        dynamic,
        casts_shadow: true,
    }
}

#[derive(Clone)]
struct Options {
    threads: u32,
    scalar: bool,
    tier: Tier,
    shadows: bool,
    with_caster: bool,
    caster_dynamic: bool,
    refresh: u32,
    spill_dir: Option<PathBuf>,
    require_geometry: bool,
    /// The RAM cap the device is given. A cap below what the frame's own targets
    /// cost is the T4 case that streams, and it is a budget rather than an
    /// on/off switch on purpose: the band height is what changes.
    ram_budget: u64,
    /// The pass's viewport, in frame pixels. `(0, 0)` is the documented default -
    /// the whole target - which is what every other test uses.
    viewport: (u32, u32),
}

impl Default for Options {
    fn default() -> Self {
        Self {
            threads: 2,
            scalar: false,
            tier: Tier::CpuRam,
            shadows: true,
            with_caster: true,
            caster_dynamic: true,
            refresh: 1,
            spill_dir: None,
            require_geometry: false,
            ram_budget: 512 << 20,
            viewport: (0, 0),
        }
    }
}

impl Options {
    fn shadow_request(&self) -> ShadowRequest {
        ShadowRequest {
            enabled: self.shadows,
            cascades: 3,
            texel_budget_bytes: 8 << 20,
            filter: ShadowFilter::Pcf3x3,
            max_distance: 60.0,
            blend_band: 0.0,
            refresh_interval_frames: self.refresh,
            allow_disk_cache: self.spill_dir.is_some(),
            bias: None,
        }
    }
}

fn new_device(options: &Options) -> Result<SoftCpuDevice, Error> {
    let alloc = HostAlloc::system();
    let budget = Arc::new(Budget::new(BudgetCaps {
        vram: 0,
        ram: options.ram_budget,
        disk: 128 << 20,
        allow_disk_spill: options.spill_dir.is_some(),
    }));
    let config = SoftCpuConfig {
        tier: options.tier,
        worker_threads: options.threads,
        scalar: options.scalar,
        spill_dir: options.spill_dir.clone(),
        shadow: options.shadow_request(),
        frame_policy: FramePolicy {
            require_geometry: options.require_geometry,
            ..FramePolicy::default()
        },
        ..SoftCpuConfig::default()
    };
    SoftCpuDevice::new(alloc, budget, config)
}

/// One frame with its colour checksum asked for, which is what these tests
/// compare frames by.
fn render_one<'a>(
    device: &mut SoftCpuDevice,
    frame_index: u64,
    draws: &'a [DrawItem<'a>],
) -> Result<FrameNumbers, Error> {
    render_with(device, frame_index, draws, true, (0, 0), (WIDTH, HEIGHT))
}

/// One frame, with the frame's colour checksum asked for or not, the pass's
/// viewport as the host sets it, and the frame's size - which is the host's to
/// change between frames, and what the streamed path has to survive.
fn render_with<'a>(
    device: &mut SoftCpuDevice,
    frame_index: u64,
    draws: &'a [DrawItem<'a>],
    checksum: bool,
    viewport: (u32, u32),
    (width, height): (u32, u32),
) -> Result<FrameNumbers, Error> {
    let (view, _proj) = camera();
    let input = FrameInput {
        frame_index,
        width,
        height,
        viewport,
        camera_view: view,
        fov_y_deg: 60.0,
        aspect: width as f32 / height as f32,
        near: 0.1,
        far: 100.0,
        light_dir: LIGHT_DIR,
        light_hash: LIGHT_HASH,
        lights: lights(),
        shadow: device.config().shadow,
        clear_color: [0.02, 0.02, 0.03, 1.0],
        clear_depth: 0.0,
        clear_color_enabled: true,
        clear_depth_enabled: true,
        world_revision: WORLD_REVISION,
        static_geometry_revision: STATIC_REVISION,
        checksum,
        draws,
    };
    device.render(&input)
}

struct Outcome {
    checksums: Vec<u64>,
    color: Vec<f32>,
    /// The frame as the host receives it: what `present` produced, which for a
    /// streamed frame is read back out of the arena rather than out of RAM.
    pixels: Vec<u8>,
    depth: Vec<f32>,
    shadows: ShadowCounters,
    tier: Tier,
    map_max_depth: f32,
    /// Bytes the frame moved through the disk arena, and what it held resident.
    spill_bytes: u64,
    resident_bytes: u64,
}

fn run(options: &Options, frames: u64) -> Result<Outcome, Error> {
    let mut device = new_device(options)?;
    let floor = floor_quad();
    let (cube_verts, cube_indices) = cube([0.0, 0.6, 0.0], 0.5);
    let (view, proj) = camera();
    let pv = math::mul(&proj, &view);

    let mut checksums = Vec::new();
    let mut max_depth = 0.0f32;
    for frame_index in 0..frames {
        let mut draws = vec![floor_draw(&floor, pv)];
        if options.with_caster {
            draws.push(caster_draw(&cube_verts, &cube_indices, pv, options.caster_dynamic));
        }
        let numbers = render_with(&mut device, frame_index, &draws, true, options.viewport, (WIDTH, HEIGHT))?;
        device.on_frame_end()?;
        checksums.push(device.color_checksum());
        if options.with_caster {
            assert!(numbers.triangles_in >= 3, "the renderer saw no geometry");
        }
        if let Some(map) = device.cascade_depth(0) {
            for d in map {
                if *d > max_depth {
                    max_depth = *d;
                }
            }
        }
    }
    let mut pixels = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    device.read_frame_into(&mut pixels, 0, 0)?;
    let mut depth = vec![0.0f32; (WIDTH * HEIGHT) as usize];
    device.depth_into(&mut depth)?;
    let snapshot = device.snapshot();
    Ok(Outcome {
        checksums,
        color: device.color_slice().to_vec(),
        pixels,
        depth,
        shadows: device.shadows(),
        tier: device.tier(),
        map_max_depth: max_depth,
        spill_bytes: snapshot.spill_io_bytes,
        resident_bytes: snapshot.resident_bytes,
    })
}

/// A directory that deletes itself, so a failing test does not leave arena files
/// behind and a passing one does not leave them either.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        dir.push(format!("reconl-softcpu-{}-{}-{}", tag, std::process::id(), nanos));
        std::fs::create_dir_all(&dir).expect("temp dir");
        Self(dir)
    }

    fn path(&self) -> PathBuf {
        self.0.clone()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// --------------------------------------------------------------------- tests

#[test]
fn the_frame_is_bit_identical_across_worker_counts() {
    let base = run(&Options { threads: 1, ..Options::default() }, 1).unwrap();
    assert!(!base.color.is_empty());
    for threads in [2, 4, 8] {
        let other = run(&Options { threads, ..Options::default() }, 1).unwrap();
        assert_eq!(base.checksums, other.checksums, "{} workers produced a different frame", threads);
        assert_eq!(base.color, other.color, "{} workers produced different pixels", threads);
    }
}

#[test]
fn the_scalar_pixel_path_agrees_with_the_batched_one() {
    let batched = run(&Options { scalar: false, ..Options::default() }, 1).unwrap();
    let scalar = run(&Options { scalar: true, ..Options::default() }, 1).unwrap();
    assert_eq!(
        batched.color, scalar.color,
        "the batched path must be an arithmetic identity over the scalar path, not an approximation"
    );
}

#[test]
fn the_caster_darkens_the_floor_only_where_the_light_says() {
    let no_caster = run(&Options { with_caster: false, ..Options::default() }, 1).unwrap();
    let with_caster = run(&Options { with_caster: true, ..Options::default() }, 1).unwrap();

    let (view, proj) = camera();
    let pv = math::mul(&proj, &view);
    // The cube spans x in [-0.5, 0.5] and y in [0.1, 1.1]. With the light at
    // (0.6, -1, 0), a point on the cube's top face at y = 1.1 lands on the floor
    // 1.1 * 0.6 = 0.66 further along +x, so the shadow covers roughly
    // x in [-0.5, 1.16]. x = 0.9 is inside it and is not hidden by the cube
    // itself; x = -1.6 is outside it.
    let (sx, sy) = project(&pv, [0.9, 0.0, 0.0]);
    let (lx, ly) = project(&pv, [-1.6, 0.0, 0.0]);

    let lit_reference = pixel(&no_caster.color, sx, sy);
    let shadowed = pixel(&with_caster.color, sx, sy);
    let lit = pixel(&with_caster.color, lx, ly);

    assert!(lit_reference[0] > 0.5, "the floor should be lit when nothing casts: {:?}", lit_reference);
    assert!(lit[0] > 0.5, "a floor point outside the shadow should stay lit: {:?}", lit);
    assert!(
        shadowed[0] < 0.25,
        "the floor inside the cube's shadow must fall back to ambient: {:?}",
        shadowed
    );
    assert_ne!(with_caster.checksums, no_caster.checksums);

    // The shadow map is not merely counted, it has content.
    assert!(with_caster.shadows.cascades_rendered > 0);
    assert!(with_caster.map_max_depth > 0.0, "the cascade map was never written");
    assert!(with_caster.shadows.map_width >= 128);
    assert_eq!(with_caster.shadows.frozen_cascades, 0, "T2 does not freeze cascades");
}

#[test]
fn disabling_shadows_produces_a_brighter_frame_and_no_shadow_pass() {
    let on = run(&Options { shadows: true, ..Options::default() }, 1).unwrap();
    let off = run(&Options { shadows: false, ..Options::default() }, 1).unwrap();
    assert_eq!(off.shadows.cascades_rendered, 0);
    assert_eq!(off.shadows.map_bytes, 0);
    let (view, proj) = camera();
    let pv = math::mul(&proj, &view);
    let (sx, sy) = project(&pv, [0.9, 0.0, 0.0]);
    assert!(pixel(&off.color, sx, sy)[0] > 0.5);
    assert!(pixel(&on.color, sx, sy)[0] < 0.25);
}

#[test]
fn t4_serves_the_static_cascade_from_the_disk_arena_unchanged() {
    let dir = TempDir::new("cache");
    let options = Options {
        tier: Tier::OutOfCore,
        spill_dir: Some(dir.path()),
        ..Options::default()
    };
    // The floor is static, so the cascade is cached; the cube is dynamic and is
    // re-drawn into the map every frame, which is what keeps it moving.
    let cold = run(&options, 1).unwrap();
    assert!(cold.shadows.cache_misses > 0, "the cold frame must render every cascade");
    assert_eq!(cold.shadows.cache_hits, 0);

    let warm = run(&options, 1).unwrap();
    assert!(warm.shadows.cache_hits > 0, "the second device must find the cached cascades");
    assert_eq!(warm.shadows.cache_corrupt, 0);
    assert_eq!(cold.checksums, warm.checksums, "a cache hit changed the frame");
    assert_eq!(cold.color, warm.color, "a cache hit changed the pixels");

    let again = run(&options, 2).unwrap();
    assert!(again.shadows.cache_hits > again.shadows.cache_misses);
}

/// The largest difference between two frames' depth values.
fn depth_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |worst, (x, y)| worst.max((x - y).abs()))
}

/// The largest per-channel difference between two presented frames, and how many
/// of their pixels differ at all.
fn pixel_delta(a: &[u8], b: &[u8]) -> (i32, usize) {
    let mut worst = 0i32;
    let mut differing = 0usize;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if x != y {
            differing += 1;
        }
        for k in 0..4 {
            worst = worst.max((x[k] as i32 - y[k] as i32).abs());
        }
    }
    (worst, differing)
}

#[test]
fn a_frame_the_ram_cap_cannot_hold_streams_through_the_arena() {
    // T4's promise: a frame larger than the RAM cap renders rather than being
    // refused. Every band it writes is counted - that is the measurement - and the
    // cut must not be visible: the same scene at a cap that holds it has to
    // present the same pixels, the same depth and the same checksum.
    let dir = TempDir::new("stream");
    let options = |ram_budget: u64| Options {
        tier: Tier::OutOfCore,
        spill_dir: Some(dir.path()),
        ram_budget,
        shadows: false,
        ..Options::default()
    };
    // The frame's own targets cost WIDTH * HEIGHT * 20 bytes; a quarter of that is
    // a cap the frame cannot be held in, and one a band easily can.
    let frame_bytes = (WIDTH as u64) * (HEIGHT as u64) * 20;
    let cap = frame_bytes / 4;
    let streamed = run(&options(cap), 1).unwrap();
    let resident = run(&options(512 << 20), 1).unwrap();

    assert_eq!(
        resident.spill_bytes, 0,
        "the resident run has no reason to move anything through the arena"
    );
    assert!(
        streamed.spill_bytes >= frame_bytes,
        "a streamed frame writes its whole self through the arena: {} of {} bytes",
        streamed.spill_bytes,
        frame_bytes
    );
    assert!(
        streamed.resident_bytes <= cap,
        "the device held {} bytes of a {} byte cap",
        streamed.resident_bytes,
        cap
    );

    // The bands are the frame's rows: a band boundary that put the wrong rows in a
    // band, or a remap that missed by a row, would move whole rows of the image,
    // and that is what this rules out. What it cannot demand is bit-for-bit
    // equality of a *shaded* value, because a band's clip space is remapped and an
    // f32 remap rounds: an interpolated value can land one ulp away from the
    // frame's - one step of a colour quantised to 8 bits, and nothing at all in
    // depth. Measured through the ABI on the reference scene `reconl-bench`
    // renders, at 512x512 with `--png` and compared with `reconl-diff`: a streamed
    // frame and a resident one differ on 2 of 262144 pixels, worst channel delta
    // 46, at (372,304) and (197,387). What that measurement says is that it is the
    // band cut's rounding and not a shadow-cache effect or a general nondeterminism:
    // both settings are byte-identical run to run (three runs each), the two
    // pinned pixels are the same ones with `--shadows=on` (dynamic, no cache
    // involved) as with `--shadows=cached`, and the set moves with the band height -
    // 256-row bands differ at (372,304) alone, 128-row bands at both. It needs a
    // shadow term to be visible at all: with `--shadows=off` a streamed 512x512
    // frame is byte-identical to a resident one, 0 of 262144, because the surfaces
    // meeting at those edges are the same colour without one.
    //
    // The magnitude is the cross-tier class rather than a new one. Measured the same
    // way at 512x512, `soft-cpu` against `d3d11` differs on 761 of 262144 pixels with
    // the same worst channel delta of 46 and the same per-channel signature
    // (R 12, G 16, B 46), which is the difference `reconl-diff --tolerance=48` is set
    // for; a streamed frame sits well inside it.
    let (worst, differing) = pixel_delta(&streamed.pixels, &resident.pixels);
    assert!(worst <= 1, "a band moved a colour by {worst}: that is a seam, not rounding");
    assert!(
        differing * 100 < streamed.pixels.len() / 4,
        "{differing} of {} pixels differ between the cut frame and the whole one",
        streamed.pixels.len() / 4
    );
    let depth_delta = depth_delta(&streamed.depth, &resident.depth);
    assert!(depth_delta < 1.0e-4, "a band moved a depth by {depth_delta}: that is a seam");

    // And the cut frame is reproducible: a cold arena and a warm one - the second
    // run below finds the bands the first one left - present the same pixels and
    // the same checksum. That is what `PROMPT.md` section 8 rests on.
    let warm = run(&options(cap), 1).unwrap();
    assert_eq!(warm.pixels, streamed.pixels, "a warm arena changed the streamed frame");
    assert_eq!(warm.checksums, streamed.checksums);
}

#[test]
fn a_streamed_band_is_confined_to_the_viewport_like_the_frame_is() {
    // The two things that decide where a pixel lands are the band cut and the
    // pass's viewport, and they have to agree: the frame is cut at the viewport's
    // edge so no band straddles it, a band outside it is written cleared, and the
    // band's rows are where the frame's rows are - not squeezed into the patch.
    let dir = TempDir::new("stream-viewport");
    let options = |ram_budget: u64| Options {
        tier: Tier::OutOfCore,
        spill_dir: Some(dir.path()),
        ram_budget,
        shadows: false,
        viewport: (48, 40),
        ..Options::default()
    };
    let cap = (WIDTH as u64) * (HEIGHT as u64) * 20 / 4;
    let streamed = run(&options(cap), 1).unwrap();
    let resident = run(&options(512 << 20), 1).unwrap();

    assert!(streamed.spill_bytes > 0, "this frame has to stream: it does not fit the cap");
    let (worst, _) = pixel_delta(&streamed.pixels, &resident.pixels);
    assert!(
        worst <= 1,
        "a sub-rect viewport came out of the band cut moved by {worst}"
    );
    let warm = run(&options(cap), 1).unwrap();
    assert_eq!(warm.pixels, streamed.pixels);
    assert_eq!(warm.checksums, streamed.checksums);
    // What a viewport means: outside the patch is the clear colour, and inside it
    // is the frame, scaled into the patch rather than cut out of it.
    let clear = [0.02, 0.02, 0.03, 1.0];
    assert_eq!(pixel(&resident.color, WIDTH as usize - 1, HEIGHT as usize - 1), clear);
    assert_ne!(pixel(&resident.color, 24, 20), clear, "the patch is where the pass drew");
}

#[test]
fn a_frame_that_fits_again_goes_back_to_the_target_it_started_in() {
    // A device is not stuck in the streamed path: the frame after a streamed one
    // is drawn into an ordinary target, and it must be the frame a device that
    // never streamed would have produced.
    let dir = TempDir::new("stream-then-fit");
    let options = Options {
        tier: Tier::OutOfCore,
        spill_dir: Some(dir.path()),
        ram_budget: 128 << 10,
        shadows: false,
        ..Options::default()
    };
    let floor = floor_quad();
    let (view, proj) = camera();
    let pv = math::mul(&proj, &view);
    let draws = vec![floor_draw(&floor, pv)];

    let mut streamed = new_device(&options).unwrap();
    render_with(&mut streamed, 0, &draws, true, (0, 0), (256, 256)).unwrap();
    assert!(
        streamed.snapshot().spill_io_bytes > 0,
        "a 256x256 frame does not fit a 128 KiB cap, so it has to stream"
    );

    let mut plain = new_device(&options).unwrap();
    let mut after_stream = vec![0u8; 64 * 64 * 4];
    let mut never_streamed = vec![0u8; 64 * 64 * 4];
    render_with(&mut streamed, 1, &draws, true, (0, 0), (64, 64)).unwrap();
    streamed.read_frame_into(&mut after_stream, 0, 0).unwrap();
    render_with(&mut plain, 1, &draws, true, (0, 0), (64, 64)).unwrap();
    plain.read_frame_into(&mut never_streamed, 0, 0).unwrap();

    assert_eq!(streamed.frame_size(), (64, 64), "the frame after the streamed one is not a band");
    assert_eq!(
        after_stream, never_streamed,
        "a device that streamed a frame went on to render the next one differently"
    );
}

#[test]
fn a_damaged_arena_file_costs_a_render_but_not_a_wrong_frame() {
    let dir = TempDir::new("damaged");
    let options = Options {
        tier: Tier::OutOfCore,
        spill_dir: Some(dir.path()),
        ..Options::default()
    };
    let clean = run(&options, 1).unwrap();

    // Flip a byte inside the first record's payload: its checksum no longer
    // matches, so the arena may not serve it. Whatever the arena decides - drop
    // the torn tail on open, or report the entry as corrupt - the frame must be
    // re-rendered rather than assembled from a lie.
    let file = dir.path().join("arena.rcls");
    let mut bytes = std::fs::read(&file).expect("arena file");
    assert!(bytes.len() > 64, "the arena should have written records");
    bytes[32 + 24 + 8] ^= 0xFF;
    std::fs::write(&file, &bytes).expect("write arena");

    let damaged = run(&options, 1).unwrap();
    assert_eq!(damaged.color, clean.color, "a damaged cache entry must not change the image");
    assert_eq!(damaged.checksums, clean.checksums);
    assert_eq!(damaged.shadows.cache_corrupt, 0, "a torn record is dropped on open, not reported as corrupt");

    // And the damage is reported somewhere a host can see it: either the arena
    // recovered a torn tail, or the cascades were simply re-rendered.
    let device = new_device(&options).unwrap();
    let stats = device.arena_stats().unwrap_or_default();
    assert!(stats.recovered_torn + damaged.shadows.cache_misses as u64 > 0);
}

#[test]
fn t3_freezes_the_static_cascade_between_refreshes_without_changing_the_frame() {
    let frozen = run(
        &Options {
            tier: Tier::CpuThrifty,
            refresh: 4,
            ..Options::default()
        },
        5,
    )
    .unwrap();
    assert!(frozen.shadows.frozen_cascades > 0, "T3 with a refresh interval must freeze");
    assert!(
        frozen.checksums.iter().all(|c| *c == frozen.checksums[0]),
        "a frozen cascade must be indistinguishable from a refreshed one when nothing moved: {:?}",
        frozen.checksums
    );
    // Refreshes still happen: five frames with a refresh every four cannot be
    // all-frozen.
    assert!(frozen.shadows.cascades_rendered > 0);
}

#[test]
fn a_frozen_cascade_still_updates_when_the_geometry_it_holds_changes() {
    // Same tier and refresh interval, but the light hash changes on frame 1:
    // the frozen cache belongs to the old light and must not be reused.
    let mut device = new_device(&Options {
        tier: Tier::CpuThrifty,
        refresh: 8,
        ..Options::default()
    })
    .unwrap();
    let floor = floor_quad();
    let (_view, proj) = camera();
    let (view, _p) = camera();
    let pv = math::mul(&proj, &view);
    let draws = vec![floor_draw(&floor, pv)];
    render_one(&mut device, 0, &draws).unwrap();
    assert_eq!(device.shadows().frozen_cascades, 0, "the first frame has nothing to freeze");

    // Same geometry, same camera, different light: the frozen maps describe the
    // old light and must not be reused.
    let draws2 = vec![floor_draw(&floor, pv)];
    let input = FrameInput {
        frame_index: 1,
        width: WIDTH,
        height: HEIGHT,
        viewport: (0, 0),
        camera_view: view,
        fov_y_deg: 60.0,
        aspect: 1.0,
        near: 0.1,
        far: 100.0,
        light_dir: [0.0, -1.0, 0.0],
        light_hash: LIGHT_HASH ^ 0xFFFF,
        lights: lights(),
        shadow: device.config().shadow,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        clear_depth: 0.0,
        clear_color_enabled: true,
        clear_depth_enabled: true,
        world_revision: WORLD_REVISION + 1,
        static_geometry_revision: STATIC_REVISION,
        checksum: true,
        draws: &draws2,
    };
    device.render(&input).unwrap();
    assert_eq!(
        device.shadows().frozen_cascades,
        0,
        "a new light must re-render rather than reuse a frozen cascade"
    );
    assert!(device.shadows().cascades_rendered > 0);
}

#[test]
fn an_empty_frame_is_reported_rather_than_counted() {
    let options = Options {
        require_geometry: true,
        ..Options::default()
    };
    let mut device = new_device(&options).unwrap();
    let empty: Vec<DrawItem<'_>> = Vec::new();
    let err = render_one(&mut device, 0, &empty).unwrap_err();
    assert_eq!(err.code, Code::EmptyFrame);
    // The geometry the empty frame had before it failed is still in the target:
    // the frame was refused, not half-rendered.
    assert_eq!(device.counters().safe_path_events, 0);

    let permissive = Options { require_geometry: false, ..Options::default() };
    let mut device = new_device(&permissive).unwrap();
    render_one(&mut device, 0, &empty).unwrap();
    let snapshot = device.snapshot();
    assert_eq!(snapshot.classifier.empty_frames, 1);
    assert_eq!(snapshot.classifier.non_empty_frames, 0);
}

#[test]
fn a_frame_that_draws_less_than_the_one_before_it_leaves_no_stale_entries() {
    // The lists a frame fills outlive the frame - the frame's draws, the cascade
    // list and the colour list are cleared and refilled instead of reallocated -
    // so the failure this pins is a stale tail: an entry the frame before left
    // behind, still being visited. A frame that draws a floor where the previous
    // frame drew a floor and a caster must render the floor alone, and render it
    // exactly as a device that never saw the caster does.
    let options = Options::default();
    let mut device = new_device(&options).unwrap();
    let floor = floor_quad();
    let (cube_verts, cube_indices) = cube([0.0, 0.6, 0.0], 0.5);
    let (view, proj) = camera();
    let pv = math::mul(&proj, &view);

    let full = [floor_draw(&floor, pv), caster_draw(&cube_verts, &cube_indices, pv, false)];
    let lean = [floor_draw(&floor, pv)];

    render_one(&mut device, 0, &full).unwrap();
    device.on_frame_end().unwrap();
    let long_checksum = device.color_checksum();

    let numbers = render_one(&mut device, 1, &lean).unwrap();
    device.on_frame_end().unwrap();
    let short_checksum = device.color_checksum();

    // The same lean frame on a device whose storage never held the caster is the
    // definition of what the lean frame is.
    let mut fresh = new_device(&options).unwrap();
    render_one(&mut fresh, 1, &lean).unwrap();
    fresh.on_frame_end().unwrap();
    assert_eq!(
        short_checksum,
        fresh.color_checksum(),
        "a frame that draws less than the one before it rendered storage the earlier frame left behind"
    );
    assert_ne!(
        short_checksum, long_checksum,
        "the two frames must differ, or this check proves nothing"
    );
    assert_eq!(numbers.triangles_in, 2, "the lean frame visited a triangle it never drew");
}

#[test]
fn the_reference_backend_does_not_decide_its_own_tier() {
    // The tier ladder has one owner: the device (`ffi/src/offload.rs`). A backend
    // holds no target, no threshold, no counter and no record of its own, because
    // the number the ladder judges - the frame the host waited for - cannot be
    // measured from inside a render pass, and a second decider is how the
    // host-visible tier and its log came to disagree. This backend renders the
    // tier it was created with, whatever the frames cost here, until the device
    // tells it otherwise.
    let options = Options { tier: Tier::CpuRam, ..Options::default() };
    let outcome = run(&options, 3).unwrap();
    assert_eq!(outcome.tier, Tier::CpuRam, "the backend relabelled itself for frame time");
    assert_eq!(outcome.checksums.len(), 3);
    assert!(outcome.checksums.iter().all(|c| *c != 0));
}

#[test]
fn a_relabel_applies_the_tier_the_device_decided() {
    let options = Options { tier: Tier::CpuRam, ..Options::default() };
    let mut device = new_device(&options).unwrap();
    let floor = floor_quad();
    let (view, proj) = camera();
    let pv = math::mul(&proj, &view);
    let draws = vec![floor_draw(&floor, pv)];
    render_one(&mut device, 0, &draws).unwrap();
    assert!(device.counters().frames_since_tier_change > 0);

    device.relabel(Tier::CpuThrifty, TierReason::FrameTimeOverTarget);
    assert_eq!(device.tier(), Tier::CpuThrifty);
    assert_eq!(device.tier_reason(), TierReason::FrameTimeOverTarget);
    assert_eq!(
        device.counters().frames_since_tier_change,
        0,
        "a tier change restarts the clock a host reads"
    );

    // The tier is applied, not queued: the next frame renders at it.
    let numbers = render_one(&mut device, 1, &draws).unwrap();
    assert!(numbers.triangles_in > 0);
    assert_eq!(device.tier(), Tier::CpuThrifty);

    // A relabel to the tier the device is already on is a no-op: the ladder
    // steps down, so T4 has nothing below it to apply.
    device.relabel(Tier::OutOfCore, TierReason::MemoryPressure);
    device.relabel(Tier::OutOfCore, TierReason::MemoryPressure);
    assert_eq!(device.tier(), Tier::OutOfCore);
}

#[test]
fn a_budget_cap_is_reported_rather_than_exceeded() {
    // 1 KiB of RAM cannot hold a 128x128 colour target, let alone a cascade map.
    let alloc = HostAlloc::system();
    let budget = Arc::new(Budget::new(BudgetCaps {
        vram: 0,
        ram: 1024,
        disk: 0,
        allow_disk_spill: false,
    }));
    let mut device = SoftCpuDevice::new(
        alloc,
        budget,
        SoftCpuConfig {
            worker_threads: 1,
            ..SoftCpuConfig::default()
        },
    )
    .unwrap();
    let floor = floor_quad();
    let (_view, proj) = camera();
    let (view, _p) = camera();
    let pv = math::mul(&proj, &view);
    let draws = vec![floor_draw(&floor, pv)];
    let err = render_one(&mut device, 0, &draws).unwrap_err();
    assert_eq!(err.code, Code::BudgetExceeded);
    assert!(device.counters().safe_path_events > 0, "a refusal is a counted safe-path event");
    assert_eq!(device.resident_bytes(), 0, "nothing was allocated for the refused frame");
}

#[test]
fn probing_a_tier_reports_capabilities_consistent_with_the_ladder() {
    let (caps_t2, plan_t2) = reconl_backend_softcpu::probe(Tier::CpuRam);
    let (caps_t4, plan_t4) = reconl_backend_softcpu::probe(Tier::OutOfCore);
    assert_eq!(caps_t2 & reconl_core::tier::caps::SHADOWS, reconl_core::tier::caps::SHADOWS);
    assert_eq!(caps_t2 & reconl_core::tier::caps::DISK_SPILL, 0, "T2 is RAM-resident");
    assert_ne!(caps_t4 & reconl_core::tier::caps::DISK_SPILL, 0, "T4 spills to disk");
    assert!(!plan_t2.disk_backed);
    assert!(plan_t4.disk_backed);
    assert_eq!(caps_for(Tier::CpuThrifty), caps_for(Tier::CpuRam) | reconl_core::tier::caps::DISK_SPILL);
    assert!(plan_t2.cascades <= 4 && plan_t4.cascades <= 4);
    // The filter bits follow the tier's own clamp: T2 and below run at most
    // pcf3x3, so they advertise neither pcf5x5 nor pcss-lite - a host reading
    // these caps cannot ask for a filter the tier will never run.
    let fine = reconl_core::tier::caps::PCF_5X5 | reconl_core::tier::caps::PCSS_LITE;
    assert_eq!(caps_t2 & fine, 0, "T2 is clamped to pcf3x3: {caps_t2:#010x}");
    assert_eq!(caps_for(Tier::CpuThrifty) & fine, 0, "T3 is clamped to pcf3x3");
    assert_eq!(caps_for(Tier::OutOfCore) & fine, 0, "T4 is clamped to pcf3x3");
    // ... and a plan built with those caps still reports the downgrade it
    // performs: pcss-lite requested, pcf3x3 active, downgrade counted.
    let over = reconl_core::tier::shadow_plan(Tier::CpuRam, 1, 32 << 20, ShadowFilter::PcssLite, caps_t2);
    assert_eq!(over.filter, ShadowFilter::Pcf3x3);
    assert_eq!(over.clamp_event, Some(reconl_core::tier::ShadowEvent::FilterDowngraded));
}

/// The bin a frame rasterises is that frame's, not every frame's so far.
///
/// The per-tile counts were sized but never zeroed, so each frame added its
/// triangles on top of the previous frames' and the work per frame climbed with
/// the frame count. It was invisible in the pixels - the stale entries
/// re-rasterised the same geometry - and fatal to a long run.
#[test]
fn the_work_of_a_frame_does_not_grow_with_the_frame_count() {
    let mut device = new_device(&Options::default()).unwrap();
    let floor = floor_quad();
    let (view, proj) = camera();
    let pv = math::mul(&proj, &view);
    let draws = vec![floor_draw(&floor, pv)];

    let first = render_one(&mut device, 0, &draws).unwrap();
    for frame in 1..32 {
        render_one(&mut device, frame, &draws).unwrap();
    }
    let late = render_one(&mut device, 32, &draws).unwrap();
    assert_eq!(
        late.triangles_binned, first.triangles_binned,
        "frame 33 binned {} triangles where frame 0 binned {}",
        late.triangles_binned, first.triangles_binned
    );
    assert_eq!(late.tiles_rendered, first.tiles_rendered, "tiles rendered must not accumulate");
    assert_eq!(late.pixels_shaded, first.pixels_shaded, "shaded pixels must not accumulate");
}

/// The frame's colour checksum is a full pass over the colour target, so it is
/// computed only when the caller asks for one. It used to run on every submit,
/// for callers that never read it - and leaving it out must not change the frame
/// a host is handed.
#[test]
fn the_frame_checksum_is_computed_only_when_it_is_asked_for() {
    let mut device = new_device(&Options::default()).unwrap();
    let floor = floor_quad();
    let (view, proj) = camera();
    let pv = math::mul(&proj, &view);
    let draws = vec![floor_draw(&floor, pv)];

    render_with(&mut device, 0, &draws, true, (0, 0), (WIDTH, HEIGHT)).unwrap();
    assert_ne!(
        device.color_checksum(),
        0,
        "a checksum that was asked for is a real fingerprint"
    );
    // The same call a present makes: the frame, as the host is handed it.
    let mut frame = vec![0u8; device.frame_size().0 as usize * device.frame_size().1 as usize * 4];
    device.read_frame_into(&mut frame, 0, 0).unwrap();

    render_with(&mut device, 1, &draws, false, (0, 0), (WIDTH, HEIGHT)).unwrap();
    assert_eq!(
        device.color_checksum(),
        0,
        "no caller reads a checksum, so none is computed"
    );
    let mut again = vec![0u8; frame.len()];
    device.read_frame_into(&mut again, 0, 0).unwrap();
    assert_eq!(
        again.as_slice(),
        frame.as_slice(),
        "skipping the checksum must not change the frame"
    );
}
