//! Behavioural cross-tier tests: the claims that need a real scene.
//!
//! A saturated triangle proves a backend draws. It does not prove that the
//! shadow pass occludes anything, or that the fixed-function blend states
//! reproduce `shade::blend` - so these tests build scenes whose *numbers* only
//! come out right if the hardware path really runs the same shading the
//! reference does. Each one renders the same scene on both tiers through the
//! public ABI and compares what came back.
//!
//! The scene conventions are the reference's own: a reversed-Z camera pushed
//! into the view-projection slot, world-space vertices, and a directional light
//! whose shadow map covers the camera frustum.

#![allow(non_snake_case)]

use reconl::abi;
use reconl::abi::{ReconLLight, ReconLLightList, ReconLShadowConfig, ReconLVertex};
use reconl_core::{ABIStruct, StructHeader};
use reconl_raster::math::{look_at, mul, perspective_rh_reversed_z, IDENTITY};
use reconl_raster::{CULL_BACK, CULL_FRONT};
use std::ffi::CString;

const W: u32 = 64;
const H: u32 = 64;
const SHADOW_CONFIG: u32 = 16; // ReconLStructType::SHADOW_CONFIG

// ---------------------------------------------------------------- test host

extern "C" fn t_alloc(_user: *mut core::ffi::c_void, size: usize, alignment: usize) -> *mut core::ffi::c_void {
    let align = alignment.clamp(16, 4096);
    let total = match size.max(1).checked_add(align).and_then(|v| v.checked_add(16)) {
        Some(t) => t,
        None => return core::ptr::null_mut(),
    };
    let layout = match std::alloc::Layout::from_size_align(total, 16) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };
    let raw = unsafe { std::alloc::alloc(layout) };
    if raw.is_null() {
        return core::ptr::null_mut();
    }
    let base = raw as usize;
    let aligned = (base + 16 + align - 1) & !(align - 1);
    unsafe { ((aligned - 8) as *mut usize).write_unaligned(base) };
    aligned as *mut core::ffi::c_void
}

extern "C" fn t_free(_user: *mut core::ffi::c_void, ptr: *mut core::ffi::c_void, _size: usize) {
    if ptr.is_null() {
        return;
    }
    let base = unsafe { ((ptr as usize - 8) as *const usize).read_unaligned() } as *mut u8;
    unsafe {
        std::alloc::dealloc(base, std::alloc::Layout::from_size_align_unchecked(1, 16));
    }
}

extern "C" fn t_realloc(
    user: *mut core::ffi::c_void,
    ptr: *mut core::ffi::c_void,
    old_size: usize,
    new_size: usize,
    alignment: usize,
) -> *mut core::ffi::c_void {
    let fresh = t_alloc(user, new_size, alignment);
    if fresh.is_null() {
        return core::ptr::null_mut();
    }
    if !ptr.is_null() {
        unsafe {
            core::ptr::copy_nonoverlapping(ptr as *const u8, fresh as *mut u8, old_size.min(new_size));
        }
        t_free(user, ptr, old_size);
    }
    fresh
}

fn test_allocator() -> abi::ReconLAllocator {
    abi::ReconLAllocator {
        alloc: Some(t_alloc),
        realloc: Some(t_realloc),
        free: Some(t_free),
        user: core::ptr::null_mut(),
    }
}

fn hdr<H: ABIStruct>() -> StructHeader {
    StructHeader::new(core::mem::size_of::<H>() as u32, H::STRUCT_TYPE)
}

// ------------------------------------------------------------------- scenes

/// One draw: geometry plus the pipeline it is drawn with.
struct TestDraw {
    verts: Vec<ReconLVertex>,
    indices: Vec<u32>,
    shading: u32,
    blend: u32,
    cull: u32,
    receives_shadow: u32,
    casts_shadow: u32,
}

struct LightSpec {
    direction: [f32; 3],
    intensity: f32,
    cast_shadow: u32,
}

struct ShadowSpec {
    enabled: u32,
    cascades: u32,
    filter: u32,
    /// Shadow-map memory the host asks for. Both tiers scale it (the reference
    /// by 0.25, the hardware tier by 0.5) and then round to a power of two, so
    /// matched pairs below still land on the same map size.
    budget_bytes: u64,
    max_distance: f32,
    blend_band: f32,
    /// The bias the scene pins, or zeros for the tier's own preset. Pinned by
    /// the shadowed scene for the reason its own comment gives: the tiers'
    /// presets differ by design, and that difference is a pixel of coverage on
    /// every shadow edge.
    normal_bias: f32,
    depth_bias: f32,
    slope_bias: f32,
}

struct Scene {
    draws: Vec<TestDraw>,
    view_proj: [f32; 16],
    /// The camera the frame declares, or `None` for a scene written directly in
    /// clip space - which is what the identity in slot 0 implies, and what a
    /// host that predates `ReconLFrameDesc::camera` sends.
    camera: Option<abi::ReconLCamera>,
    light: LightSpec,
    shadow: ShadowSpec,
}

/// What a device run asks for: the ladder settings the offload policy reads,
/// how many frames to render, and whether the host allows the device to change
/// tier on its own. One frame with the ladder off is what every comparison test
/// needs; the offload tests need all three to vary.
#[derive(Clone, Copy)]
struct Run {
    backend: u32,
    tier: u32,
    allow_downgrade: u32,
    target_frame_ms: u32,
    downgrade_after_frames: u32,
    frames: u32,
    width: u32,
    height: u32,
    expect: Expect,
    /// Whether the scene's geometry is declared static: `RECONL_BUFFER_STATIC`
    /// on every vertex and index buffer this run creates. The reference tier's
    /// cascade cache is keyed on that declaration and nothing else, so a leg
    /// that leaves it off is the control - the same scene, the same frames, the
    /// same budget, and no cache.
    static_geometry: bool,
    /// Whether the host asks for the cached cascade, which at T4 is the disk
    /// tier's own reason to exist: freeze the cascade between refreshes, and let
    /// the arena hold it instead of RAM.
    cached: bool,
    /// The name of a directory under the temp directory the device may put its
    /// spill arena in. `Some` is also the host's opt-in to disk at all
    /// (`allow_disk_spill`): without it there is no arena to cache in, however
    /// static the geometry and however cached the plan.
    spill: Option<&'static str>,
    /// The RAM cap the device is given. `0` is the header's "unlimited" and the
    /// default, because most runs are not about the frame's own memory. A run
    /// that wants the frame to *stream* names a cap the frame cannot be held in.
    ram_cap: u64,
    /// The disk budget the device is given: the arena's hard cap, and the number
    /// every claim about bytes on disk is measured against.
    disk_cap: u64,
}

/// What a run expects of its frames. The two refusals are here because they are
/// the cases that must not move the device: both are the host's own mistake, and
/// both report the same code a genuine device removal does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Expect {
    /// A frame that renders and presents.
    Frame,
    /// Targets the driver refuses, so the frame fails at submit.
    DriverRejects,
    /// A valid frame presented into a buffer too small for it, so the frame is
    /// dropped at present.
    PresentTooSmall,
    /// A frame whose own targets cannot be held inside the budgets this run
    /// declares, so `reconlBeginFrame` refuses it before anything is written. The
    /// outcome then carries the ABI's error text in place of pixels: the refusal
    /// *is* the finding, and its text is what a host has to act on.
    RefusesAtBegin,
}

impl Run {
    /// One frame at the test resolution, no tier changes.
    fn one(backend: u32, tier: u32) -> Self {
        Self {
            backend,
            tier,
            allow_downgrade: abi::allow_downgrade::NONE,
            target_frame_ms: 1000,
            downgrade_after_frames: 16,
            frames: 1,
            width: W,
            height: H,
            expect: Expect::Frame,
            static_geometry: false,
            cached: false,
            spill: None,
            ram_cap: 0,
            disk_cap: 64 << 20,
        }
    }
}

/// One frame on one backend, ladder off: the shape every comparison test uses.
fn render(backend: u32, tier: u32, scene: &Scene) -> (Vec<u8>, abi::ReconLStats) {
    run(Run::one(backend, tier), scene).pop().expect("one frame")
}

/// Renders `scene` through the public ABI `run.frames` times and returns each
/// frame's presented RGBA8 pixels and stats, in order. Per-frame state is the
/// only way to see the offload policy, which `reconlPresent` steps and nothing
/// else does. Panics with the ABI's own error text on failure.
///
/// A run that names a spill directory gets a memory budget that allows the disk
/// spill and points at it, which is the host's opt-in to the arena the T4 static
/// cascade lives in.
fn run(run: Run, scene: &Scene) -> Vec<(Vec<u8>, abi::ReconLStats)> {
    let mut frames: Vec<(Vec<u8>, abi::ReconLStats)> = Vec::new();
    // The arena's directory, held for the run's life: a device drops its arena
    // when the last handle to it goes, and a host that removed the directory
    // under it would be testing the error path rather than the cache.
    let spill_dir = run.spill.map(|name| {
        let path = spill_path(name);
        std::fs::create_dir_all(&path).expect("the spill directory");
        path
    });
    let spill_c = spill_dir
        .as_ref()
        .map(|path| CString::new(path.to_string_lossy().as_bytes()).expect("the spill path"));
    // SAFETY: an all-zero budget is the documented "no budget", and every field
    // this run sets is set before the descriptor points at it.
    let mut budget: abi::ReconLMemoryBudget = unsafe { core::mem::zeroed() };
    if let Some(dir) = &spill_c {
        budget.base = hdr::<abi::ReconLMemoryBudget>();
        budget.ram_cap_bytes = run.ram_cap;
        budget.disk_cap_bytes = run.disk_cap;
        budget.allow_disk_spill = 1;
        budget.spill_dir = dir.as_ptr();
    }
    unsafe {
        let mut dd: abi::ReconLDeviceDesc = core::mem::zeroed();
        dd.base = hdr::<abi::ReconLDeviceDesc>();
        dd.backend_hint = run.backend;
        dd.budget = if spill_c.is_some() { &budget as *const _ } else { core::ptr::null() };
        dd.tier_hint = run.tier;
        dd.allow_downgrade = run.allow_downgrade;
        dd.worker_threads = 1;
        dd.target_frame_ms = run.target_frame_ms;
        dd.downgrade_after_frames = run.downgrade_after_frames;
        dd.seed = 7;
        dd.allocator = test_allocator();

        let mut device: *mut reconl::DeviceHandle = core::ptr::null_mut();
        if reconl::reconlCreateDevice(&dd, &mut device) != abi::result::OK {
            panic!("create device: {}", global_error());
        }

        let mut sd: abi::ReconLSwapchainDesc = core::mem::zeroed();
        sd.base = hdr::<abi::ReconLSwapchainDesc>();
        sd.width = run.width;
        sd.height = run.height;
        sd.format = 1; // RGBA8
        sd.image_count = 2;
        sd.present_to_memory = 1;
        sd.depth_format = 1;
        let mut swapchain: *mut reconl::SwapchainHandle = core::ptr::null_mut();
        assert_eq!(
            reconl::reconlCreateSwapchain(device, &sd, &mut swapchain),
            abi::result::OK,
            "swapchain: {}",
            global_error()
        );

        let mut cd: abi::ReconLCommandListDesc = core::mem::zeroed();
        cd.base = hdr::<abi::ReconLCommandListDesc>();
        cd.capacity_bytes = 4096;
        let mut commands: *mut reconl::CommandListHandle = core::ptr::null_mut();
        assert_eq!(
            reconl::reconlCreateCommandList(device, &cd, &mut commands),
            abi::result::OK,
            "command list: {}",
            global_error()
        );

        // One pipeline and two buffers per draw; all kept alive until Present.
        let mut pipelines = Vec::new();
        for draw in &scene.draws {
            let mut pd: abi::ReconLPipelineDesc = core::mem::zeroed();
            pd.base = hdr::<abi::ReconLPipelineDesc>();
            pd.shading = draw.shading;
            pd.blend = draw.blend;
            pd.cull = draw.cull;
            pd.depth_compare = 1; // GREATER, reversed-Z
            pd.depth_write = 1;
            pd.receives_shadow = draw.receives_shadow;
            pd.casts_shadow = draw.casts_shadow;
            let mut pipeline: *mut reconl::PipelineHandle = core::ptr::null_mut();
            assert_eq!(
                reconl::reconlCreatePipeline(device, &pd, &mut pipeline),
                abi::result::OK,
                "pipeline: {}",
                global_error()
            );
            pipelines.push(pipeline);
        }
        let mut geometry = Vec::new();
        for draw in &scene.draws {
            let mut vbd: abi::ReconLBufferDesc = core::mem::zeroed();
            vbd.base = hdr::<abi::ReconLBufferDesc>();
            vbd.size_bytes = (draw.verts.len() * core::mem::size_of::<ReconLVertex>()) as u64;
            vbd.usage = buffer_usage(1, run.static_geometry); // VERTEX
            vbd.data = draw.verts.as_ptr() as *const core::ffi::c_void;
            vbd.data_size = vbd.size_bytes;
            let mut vb: *mut reconl::BufferHandle = core::ptr::null_mut();
            assert_eq!(
                reconl::reconlCreateBuffer(device, &vbd, &mut vb),
                abi::result::OK,
                "vertex buffer: {}",
                global_error()
            );
            let mut ibd: abi::ReconLBufferDesc = core::mem::zeroed();
            ibd.base = hdr::<abi::ReconLBufferDesc>();
            ibd.size_bytes = (draw.indices.len() * 4) as u64;
            ibd.usage = buffer_usage(2, run.static_geometry); // INDEX
            ibd.data = draw.indices.as_ptr() as *const core::ffi::c_void;
            ibd.data_size = ibd.size_bytes;
            let mut ib: *mut reconl::BufferHandle = core::ptr::null_mut();
            assert_eq!(
                reconl::reconlCreateBuffer(device, &ibd, &mut ib),
                abi::result::OK,
                "index buffer: {}",
                global_error()
            );
            geometry.push((vb, ib));
        }

        let mut light: ReconLLight = core::mem::zeroed();
        light.base = hdr::<abi::ReconLLight>();
        light.r#type = 0; // directional
        light.direction = scene.light.direction;
        light.color = [1.0, 1.0, 1.0];
        light.intensity = scene.light.intensity;
        light.cast_shadow = scene.light.cast_shadow;
        let ll = ReconLLightList {
            base: hdr::<abi::ReconLLightList>(),
            count: 1,
            reserved: 0,
            lights: &light,
        };

        let mut scfg: ReconLShadowConfig = core::mem::zeroed();
        scfg.base = StructHeader::new(
            core::mem::size_of::<ReconLShadowConfig>() as u32,
            SHADOW_CONFIG,
        );
        scfg.enabled = scene.shadow.enabled;
        scfg.cascade_count = scene.shadow.cascades;
        scfg.texel_budget_bytes = scene.shadow.budget_bytes;
        scfg.filter = scene.shadow.filter;
        scfg.max_distance = scene.shadow.max_distance;
        scfg.blend_band = scene.shadow.blend_band;
        scfg.normal_bias = scene.shadow.normal_bias;
        scfg.depth_bias = scene.shadow.depth_bias;
        scfg.slope_bias = scene.shadow.slope_bias;
        // The cached cascade, as the ABI asks for it: the cascade is frozen
        // between refreshes, and the arena is allowed to hold it.
        scfg.refresh_interval_frames = if run.cached { 60 } else { 1 };
        scfg.freeze_static_cascade = u32::from(run.cached);
        scfg.allow_disk_cache = u32::from(run.cached);

        let fd: abi::ReconLFrameDesc = abi::ReconLFrameDesc {
            base: hdr::<abi::ReconLFrameDesc>(),
            width: run.width,
            height: run.height,
            seed: 1,
            reserved: 0,
            lights: &ll,
            shadows: &scfg,
            camera: match &scene.camera {
                Some(camera) => camera as *const abi::ReconLCamera,
                None => core::ptr::null(),
            },
            framegen: core::ptr::null(),
        };
        for _ in 0..run.frames {
            let mut pixels = vec![0u8; (run.width * run.height * 4) as usize];
            let mut stats: abi::ReconLStats = core::mem::zeroed();
            stats.base = hdr::<abi::ReconLStats>();
            let began = reconl::reconlBeginFrame(device, &fd as *const _ as *mut _);
            if run.expect == Expect::RefusesAtBegin {
                // The refusal is recorded against the device, which is the handle the
                // caller still holds: read it before anything can clear it.
                let refusal = device_error(device);
                assert_eq!(
                    began,
                    abi::result::BUDGET_EXCEEDED,
                    "a frame the declared budgets cannot hold has to be refused with the budget \
                     code: {refusal}"
                );
                assert_eq!(
                    reconl::reconlGetStats(device, &mut stats),
                    abi::result::OK,
                    "stats: {}",
                    device_error(device)
                );
                frames.push((refusal.into_bytes(), stats));
                break;
            }
            assert_eq!(began, abi::result::OK, "begin frame: {}", global_error());

            reconl::reconlCmdReset(commands);
            let mut rp: abi::ReconLRenderPassDesc = core::mem::zeroed();
            rp.base = hdr::<abi::ReconLRenderPassDesc>();
            rp.color_count = 0;
            rp.viewport_width = run.width;
            rp.viewport_height = run.height;
            rp.load_color = 1;
            rp.load_depth = 1;
            rp.clear_color = [0.0, 0.0, 0.0, 1.0]; // black: distinct from every scene colour
            rp.clear_depth = 0.0;
            reconl::reconlCmdBeginRenderPass(commands, &rp);

            for (index, draw) in scene.draws.iter().enumerate() {
                reconl::reconlCmdSetPipeline(commands, pipelines[index]);
                reconl::reconlCmdPushConstants(
                    commands,
                    0,
                    scene.view_proj.as_ptr() as *const core::ffi::c_void,
                    64,
                );
                reconl::reconlCmdPushConstants(
                    commands,
                    1,
                    IDENTITY.as_ptr() as *const core::ffi::c_void,
                    64,
                );
                let (vb, ib) = geometry[index];
                reconl::reconlCmdSetVertexBuffer(commands, 0, vb, 0);
                reconl::reconlCmdSetIndexBuffer(commands, ib, 0, 1); // UINT32
                let r = reconl::reconlCmdDrawIndexed(commands, draw.indices.len() as u32, 0, 0);
                assert_eq!(r, abi::result::OK, "draw {index}: {}", global_error());
            }
            reconl::reconlCmdEndRenderPass(commands);

            let submit = reconl::reconlSubmit(device, commands, core::ptr::null_mut());
            if run.expect == Expect::DriverRejects {
                assert_ne!(
                    submit,
                    abi::result::OK,
                    "the driver was expected to reject this frame"
                );
                assert_eq!(
                    reconl::reconlGetStats(device, &mut stats),
                    abi::result::OK,
                    "stats: {}",
                    device_error(device)
                );
                frames.push((Vec::new(), stats));
                break;
            }
            assert_eq!(submit, abi::result::OK, "submit: {}", device_error(device));

            let mut prd: abi::ReconLPresentDesc = core::mem::zeroed();
            prd.base = hdr::<abi::ReconLPresentDesc>();
            prd.out_pixels = pixels.as_mut_ptr() as *mut core::ffi::c_void;
            prd.out_pixels_size = if run.expect == Expect::PresentTooSmall {
                16
            } else {
                pixels.len() as u64
            };
            prd.out_row_pitch = run.width * 4;
            prd.out_format = 1;
            let present = reconl::reconlPresent(device, swapchain, &mut prd);
            if run.expect == Expect::PresentTooSmall {
                assert_eq!(
                    present,
                    abi::result::INVALID_ARGUMENT,
                    "presenting into a buffer too small for the frame: {}",
                    device_error(device)
                );
            } else {
                assert_eq!(present, abi::result::OK, "present: {}", device_error(device));
            }

            assert_eq!(
                reconl::reconlGetStats(device, &mut stats),
                abi::result::OK,
                "stats: {}",
                device_error(device)
            );
            frames.push((pixels, stats));
            if run.expect != Expect::Frame {
                break;
            }
        }

        for (vb, ib) in geometry {
            reconl::reconlRelease(vb as *mut core::ffi::c_void);
            reconl::reconlRelease(ib as *mut core::ffi::c_void);
        }
        for pipeline in pipelines {
            reconl::reconlRelease(pipeline as *mut core::ffi::c_void);
        }
        reconl::reconlRelease(commands as *mut core::ffi::c_void);
        reconl::reconlRelease(swapchain as *mut core::ffi::c_void);
        reconl::reconlRelease(device as *mut core::ffi::c_void);
    }
    frames
}

fn error_text(device: *mut reconl::DeviceHandle) -> String {
    let mut ei: abi::ReconLErrorInfo = unsafe { core::mem::zeroed() };
    ei.base = hdr::<abi::ReconLErrorInfo>();
    if unsafe { reconl::reconlGetLastError(device, &mut ei) } != abi::result::OK {
        return "<no error recorded>".to_string();
    }
    let len = ei.message.iter().position(|&b| b == 0).unwrap_or(ei.message.len());
    String::from_utf8_lossy(&ei.message[..len]).into_owned()
}

fn global_error() -> String {
    error_text(core::ptr::null_mut())
}

fn device_error(device: *mut reconl::DeviceHandle) -> String {
    error_text(device)
}

fn d3d11_usable() -> bool {
    let mut info: abi::ReconLProbeInfo = unsafe { core::mem::zeroed() };
    info.base = hdr::<abi::ReconLProbeInfo>();
    if unsafe { reconl::reconlProbe(core::ptr::null(), &mut info) } != abi::result::OK {
        return false;
    }
    (0..info.entry_count as usize)
        .any(|i| info.entries[i].backend == abi::backend::D3D11 && info.entries[i].usable != 0)
}

fn v(position: [f32; 3], normal: [f32; 3], color: [f32; 4]) -> ReconLVertex {
    ReconLVertex { position, normal, uv: [0.0, 0.0], color }
}

/// The default camera for these scenes: above and behind the origin, looking
/// down at it, with the reversed-Z projection the reference expects.
const CAMERA_FOV_Y_DEG: f32 = 45.0;
const CAMERA_NEAR: f32 = 0.5;
const CAMERA_FAR: f32 = 100.0;

/// World space to camera space: the view matrix on its own. Above and behind
/// the origin at 13/10, which is the eye `tools/reconl-diff` writes into its own
/// view matrix by hand.
fn camera_view() -> [f32; 16] {
    look_at([0.0, 13.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
}

/// The transform pushed into constant slot 0: view * projection.
fn camera() -> [f32; 16] {
    let proj = perspective_rh_reversed_z(
        CAMERA_FOV_Y_DEG,
        W as f32 / H as f32,
        CAMERA_NEAR,
        CAMERA_FAR,
    );
    mul(&proj, &camera_view())
}

/// The same camera, declared to the frame so the shadow system can see it: the
/// view matrix on its own, plus the frustum it is paired with. Constant slot 0
/// carries `view * projection`, and a projection cannot be taken back out of a
/// view, so a host that wants its cascades fitted to what it is actually
/// looking at has to say where the camera is separately.
fn frame_camera() -> abi::ReconLCamera {
    let mut camera: abi::ReconLCamera = unsafe { core::mem::zeroed() };
    camera.base = hdr::<abi::ReconLCamera>();
    camera.view = camera_view();
    camera.fov_y_deg = CAMERA_FOV_Y_DEG;
    camera.near = CAMERA_NEAR;
    camera.far = CAMERA_FAR;
    camera
}

/// A flat quad in the XZ plane at height `y`, facing up, spanning `half` units.
fn xz_quad(y: f32, half: f32) -> TestDraw {
    let verts = vec![
        v([-half, y, -half], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
        v([half, y, -half], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
        v([half, y, half], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
        v([-half, y, half], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
    ];
    TestDraw {
        verts,
        indices: vec![0, 1, 2, 0, 2, 3],
        shading: abi::shading::LAMBERT,
        blend: 0,
        cull: 0,
        receives_shadow: 0,
        casts_shadow: 0,
    }
}

/// One configuration of the shadow system, as the ABI takes it. The knobs are
/// the ones `ReconLShadowConfig` documents, and the matrix below walks them.
#[derive(Clone, Copy)]
struct ShadowCase {
    enabled: u32,
    cascades: u32,
    filter: u32,
    budget_bytes: u64,
    band: f32,
}

impl ShadowCase {
    /// The scene's own configuration: two cascades, PCF 3x3, an 8 MiB budget
    /// (which both tiers round to 512x512) and a 3-unit crossfade band.
    const BASE: ShadowCase = ShadowCase {
        enabled: 1,
        cascades: 2,
        filter: 1,
        budget_bytes: 8 << 20,
        band: 3.0,
    };

    fn cascades(cascades: u32) -> Self {
        ShadowCase { cascades, ..Self::BASE }
    }
    fn filter(filter: u32) -> Self {
        ShadowCase { filter, ..Self::BASE }
    }
    fn budget(budget_bytes: u64) -> Self {
        ShadowCase { budget_bytes, ..Self::BASE }
    }
    fn band(band: f32) -> Self {
        ShadowCase { band, ..Self::BASE }
    }
}

/// Ground plus an occluder held above it, lit by a shadow-casting directional
/// light. The only way the ground's pixels come out dimmer with shadows on is
/// if the shadow pass really occludes.
fn occluder_scene(shadows_enabled: u32) -> Scene {
    occluder_scene_case(ShadowCase {
        enabled: shadows_enabled,
        ..ShadowCase::BASE
    })
}

/// The same scene at any configuration the matrix asks for.
fn occluder_scene_case(case: ShadowCase) -> Scene {
    let mut ground = xz_quad(0.0, 16.0);
    ground.color_all([0.25, 0.35, 1.0, 1.0]);
    ground.receives_shadow = 1;
    ground.casts_shadow = 1;

    // A wide, shallow triangle rather than a square: the shadow is displaced by
    // the caster's height times the light's tilt, so a caster deeper than that
    // displacement hides its own shadow, and one wider than the frame window at
    // the shadow's distance gets cropped.
    const CASTER_HEIGHT: f32 = 4.3;
    let caster = TestDraw {
        verts: vec![
            v([-6.25, CASTER_HEIGHT, -3.9], [0.0, 1.0, 0.0], [0.95, 0.95, 0.95, 1.0]),
            v([5.55, CASTER_HEIGHT, -4.5], [0.0, 1.0, 0.0], [0.95, 0.95, 0.95, 1.0]),
            v([-0.35, CASTER_HEIGHT, -0.9], [0.0, 1.0, 0.0], [0.95, 0.95, 0.95, 1.0]),
        ],
        indices: vec![0, 1, 2],
        shading: abi::shading::LAMBERT,
        blend: 0,
        cull: 0,
        receives_shadow: 1,
        casts_shadow: 1,
    };

    Scene {
        draws: vec![ground, caster],
        view_proj: camera(),
        camera: Some(frame_camera()),
        light: LightSpec {
            // A low sun, aimed so the shadow falls *toward* the camera: the tilt
            // stretches the shadow's area (by 1/cos of the angle to the normal,
            // here 1.9x) and the direction lands that stretch where the ground's
            // pixels-per-unit is largest, clear of the caster rather than under
            // it.
            direction: [0.10, -1.0, 1.55],
            intensity: 1.0,
            cast_shadow: 1,
        },
        shadow: ShadowSpec {
            enabled: case.enabled,
            budget_bytes: case.budget_bytes,
            cascades: case.cascades,
            filter: case.filter,
            // Puts the first split at 14.3 units, inside the shadow (13.7..16.3
            // from the camera), so cascade 0 renders its near half, the crossfade
            // band at 12.8..15.8 crosses it, and cascade 1 takes the far corner.
            max_distance: 77.0,
            blend_band: case.band,
            // Pinned, so the scene renders the same image on both tiers. The
            // tier presets differ by design (software tiers get more slack),
            // and left to them the same scene differs between tiers on 220 of
            // 4096 pixels - the shadow's edge, moved by a pixel of coverage.
            // These are the hardware preset's values scaled for the 512x512
            // maps both tiers round this budget to. Measured with them pinned:
            // 14 pixels differ, all of them within 4px of the shadow.
            normal_bias: 1.25,
            depth_bias: 5.0e-4,
            slope_bias: 1.75,
        },
    }
}

impl TestDraw {
    fn color_all(&mut self, color: [f32; 4]) {
        for vert in &mut self.verts {
            vert.color = color;
        }
    }
}

// -------------------------------------------------------------------- checks

/// The two numbers `reconl-diff`'s documented tolerance is stated in: the
/// largest per-channel delta anywhere in the image, and how many pixels differ
/// at all.
fn delta_report(a: &[u8], b: &[u8]) -> (i32, usize, usize) {
    let mut worst = 0i32;
    let mut differing = 0usize;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let mut differs = false;
        for channel in 0..4 {
            let delta = (pa[channel] as i32 - pb[channel] as i32).abs();
            differs |= delta != 0;
            worst = worst.max(delta);
        }
        if differs {
            differing += 1;
        }
    }
    (worst, differing, a.len() / 4)
}

/// Mean absolute channel difference between two images.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    let mut total = 0u64;
    for (x, y) in a.iter().zip(b.iter()) {
        total += (*x as i32 - *y as i32).unsigned_abs() as u64;
    }
    total as f64 / a.len().max(1) as f64
}

/// Pixels where `b` is brighter than `a` by at least `margin`, on the green
/// channel (every colour in these scenes is neutral, so green stands in for the
/// luminance the shading actually produced).
fn brighter_count(a: &[u8], b: &[u8], margin: i32) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(pa, pb)| pb[1] as i32 - pa[1] as i32 >= margin)
        .count()
}

fn coverage(pixels: &[u8]) -> (usize, u8) {
    let lit = pixels.chunks_exact(4).filter(|px| px[1] > 8).count();
    let max = pixels.chunks_exact(4).map(|px| px[1]).max().unwrap_or(0);
    (lit, max)
}

/// The pixels shadows darkened: dimmer by more than `margin` with shadows on
/// than in the *same* scene with them off. That difference is the shadow's
/// footprint however the scene is lit, which is what makes it a measurement of
/// the shadow rather than of the palette.
fn shadow_mask(lit: &[u8], unlit: &[u8], margin: i32) -> Vec<bool> {
    lit.chunks_exact(4)
        .zip(unlit.chunks_exact(4))
        .map(|(a, b)| b[1] as i32 - a[1] as i32 >= margin)
        .collect()
}

/// Grow or shrink a mask by `radius` pixels, 4-connected, clipped at the frame.
/// Shrinking to the interior is how the boundary is separated from the body of
/// the shadow; growing it is how the boundary's neighbourhood is swept for
/// divergence that is *not* on the shadow at all.
fn grow(mask: &[bool], radius: usize, outward: bool) -> Vec<bool> {
    let (w, h) = (W as usize, H as usize);
    let mut out = mask.to_vec();
    for _ in 0..radius {
        let prev = out.clone();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let mut neighbours = Vec::with_capacity(4);
                if x > 0 {
                    neighbours.push(prev[i - 1]);
                }
                if x + 1 < w {
                    neighbours.push(prev[i + 1]);
                }
                if y > 0 {
                    neighbours.push(prev[i - w]);
                }
                if y + 1 < h {
                    neighbours.push(prev[i + w]);
                }
                out[i] = if outward {
                    prev[i] || neighbours.iter().any(|n| *n)
                } else {
                    prev[i] && neighbours.iter().all(|n| *n)
                };
            }
        }
    }
    out
}

/// Pixels differing between two images, restricted to a mask (or to its
/// complement).
fn differing_within(a: &[u8], b: &[u8], mask: &[bool], inside: bool) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .zip(mask)
        .filter(|(_, m)| **m == inside)
        .filter(|((pa, pb), _)| pa[..3] != pb[..3])
        .count()
}

fn mask_count(mask: &[bool]) -> usize {
    mask.iter().filter(|m| **m).count()
}

/// How far the farthest pixel the two tiers disagree on sits from the nearest
/// pixel the reference's shadow changes.
///
/// This is what turns "outside the shadow's neighbourhood" into a measurement
/// instead of a chosen radius. A coarser shadow map quantises the caster's
/// silhouette into larger steps, so the band of screen pixels where the tiers'
/// texel boundaries land differently is legitimately wider at 256x256 than at
/// 1024x1024 - and if the divergence is a real one, it will not be near the
/// shadow at all and this number is large.
fn max_divergence_distance(a: &[u8], b: &[u8], mask: &[bool]) -> usize {
    let (w, h) = (W as i32, H as i32);
    let mut worst = 0usize;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if a[i * 4..i * 4 + 3] == b[i * 4..i * 4 + 3] {
                continue;
            }
            let mut nearest = usize::MAX;
            for my in 0..h {
                for mx in 0..w {
                    if !mask[(my * w + mx) as usize] {
                        continue;
                    }
                    let d = ((mx - x).abs() + (my - y).abs()) as usize;
                    if d < nearest {
                        nearest = d;
                    }
                }
            }
            worst = worst.max(nearest.min(999));
        }
    }
    worst
}

/// The scene has to be *capable* of shadowing before the hardware result means
/// anything: the reference tier, which is the definition of correct, must
/// itself darken the ground when shadows are switched on.
///
/// It did not. Three separate defects each prevented it, and each was fatal on
/// its own, so they are worth naming where the numbers are:
///
/// * **The camera reached the fit as `view * projection`.** That is what
///   constant slot 0 carries, and the FFI layer copied it into
///   `FrameInput::camera_view`. Cascade fitting inverts the view as a rigid
///   basis (`invert_rigid`), which a projection folded into a view is not, so
///   the light's box was fitted around a frustum that did not exist and every
///   lookup landed outside it.
/// * **The tile grid was stale.** The rasteriser computed it once, from the
///   first target it saw, and never rebinned, so a 512x512 shadow map binned
///   tile indices that only exist in a 64x64 colour target.
/// * **The shadow pass culled front faces**, which for a single-sided caster
///   facing the light is the caster itself. The map stayed empty, and no light
///   in this project - every scene here casts from a quad or a triangle facing
///   the light - could cast a shadow. The pass's own comment says the *far* side
///   is what gets culled, which is `CULL_BACK`; the constant was the bug.
#[test]
fn the_reference_tier_shadows_the_occluded_ground() {
    let (lit, stats) = render(abi::backend::SOFT_CPU, 2, &occluder_scene(1));
    let (unlit, _) = render(abi::backend::SOFT_CPU, 2, &occluder_scene(0));
    let (covered, peak) = coverage(&unlit);
    assert!(
        covered > 400 && peak > 100,
        "the test scene drew too little to judge: {covered} lit pixels, peak {peak}"
    );
    let shadowed = brighter_count(&lit, &unlit, 20);
    eprintln!(
        "reference: {covered} lit pixels, peak {peak}, {shadowed} pixels darker; \
         cascades {} (map {}x{}, filter {}), shadowed lights {}, triangles_in {} \
         binned {} culled {}, fail_safe_unshadowed {}",
        stats.shadows.cascades_active,
        stats.shadows.map_width,
        stats.shadows.map_height,
        stats.shadows.filter_active,
        stats.shadows.shadowed_lights,
        stats.frame.triangles_in,
        stats.frame.triangles_binned,
        stats.frame.triangles_culled,
        stats.shadows.fail_safe_unshadowed,
    );
    // Not merely present: a scene whose shadow is a strip can pass a "more than
    // nothing" check while a filter, bias or split regression moves nothing
    // measurable. This one is 13.6% of the frame (measured 559 of 4096), so the
    // shadow really is the subject of the image.
    let total = (W * H) as usize;
    assert!(
        shadowed * 100 >= total * 12,
        "the reference tier darkens only {shadowed} of {total} pixels with shadows on ({:.1}%), \
         too little for the hardware comparison below to be a test of anything",
        100.0 * shadowed as f64 / total as f64
    );
}

/// The hardware tier's shadow pass: with shadows on, the ground under the
/// occluder is measurably darker than the same scene with shadows off. A shadow
/// map that is rendered but never samples, or samples a cleared map, produces
/// no darkening at all and fails here.
#[test]
fn the_hardware_tier_shadows_the_occluded_ground() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device; skipping the hardware shadow test");
        return;
    }
    let (lit, stats) = render(abi::backend::D3D11, 1, &occluder_scene(1));
    let (unlit, _) = render(abi::backend::D3D11, 1, &occluder_scene(0));
    let (covered, peak) = coverage(&unlit);
    assert!(
        covered > 400 && peak > 100,
        "the hardware tier drew too little to judge: {covered} lit pixels, peak {peak}"
    );
    let shadowed = brighter_count(&lit, &unlit, 20);
    eprintln!(
        "d3d11: {covered} lit pixels, peak {peak}, {shadowed} pixels darker; \
         cascades {} (map {}x{}), shadowed lights {}, triangles_in {} binned {}",
        stats.shadows.cascades_active,
        stats.shadows.map_width,
        stats.shadows.map_height,
        stats.shadows.shadowed_lights,
        stats.frame.triangles_in,
        stats.frame.triangles_binned,
    );
    let total = (W * H) as usize;
    assert!(
        shadowed * 100 >= total * 12,
        "the hardware tier darkens only {shadowed} of {total} pixels with shadows on ({:.1}%): \
         the shadow pass is not reaching the pixel shader, or is reaching it with a shadow far \
         smaller than the reference's",
        100.0 * shadowed as f64 / total as f64
    );
}

/// Two identical frames through the ABI are the same image on the hardware
/// tier. D3D11 does not promise this by itself, so it is a property this
/// backend has to hold: a shadow map or constant buffer that leaks between
/// frames would show up here as a one-off difference.
#[test]
fn the_hardware_tier_renders_a_shadowed_scene_deterministically() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device; skipping the shadowed determinism check");
        return;
    }
    let scene = occluder_scene(1);
    let (first, _) = render(abi::backend::D3D11, 1, &scene);
    let (second, _) = render(abi::backend::D3D11, 1, &scene);
    assert_eq!(
        first, second,
        "two identical shadowed frames on the hardware tier produced different images"
    );
}

/// And the two tiers have to agree on that shadowed image, which is the whole
/// point of having a reference: same cascades, same filter, same pixels.
///
/// They cannot agree *exactly*: two rasterisers fill the same 512x512 shadow
/// map, and where the caster's silhouette crosses a texel they disagree by a
/// fraction of a texel. The light is low, so the receiver's ray grazes the
/// caster at a shallow angle and that sub-texel disagreement is magnified into
/// a pixel of shadow-edge movement on screen. This is the measurement §12 risk 7
/// asks for, and it fixes the tolerance rather than taste:
///
/// | metric | measured | bound here |
/// |--------|----------|------------|
/// | shadow coverage, each tier | 559 / 4096 = 13.6% | >= 12% |
/// | shadow extent gap | 0 px of 559 | <= 2% |
/// | pixels differing, total | 14 / 4096 = 0.34% | <= 1% |
/// | of those, inside the shadow's 4px neighbourhood | 14 | all of them |
/// | of those, outside it or in the shadow's interior | 0 | == 0 |
/// | worst channel delta | 46 | <= 48 |
/// | mean absolute channel difference | 0.0154 | < 0.1 |
///
/// The counts are what make this a statement about the shadow rather than a
/// formality: every pixel the two tiers disagree on is within four pixels of a
/// pixel the shadow changes, and the shadow's interior and the whole lit frame
/// are bit-identical. A tier whose PCF taps, cascade split, crossfade or bias
/// drifted would move the boundary by more than four pixels and land pixels
/// outside that neighbourhood, or change the shadow's extent, and fail here.
///
/// With the tiers' *own* bias presets - which differ by design, weaker tiers
/// getting more slack - the same measurement reads 220 pixels differing (5.37%)
/// with the extent 554 against 559. Both are honest numbers for the same scene;
/// the scene above pins the bias because a golden whose image changes with the
/// tier that rendered it is not a reference.
#[test]
fn both_tiers_agree_on_a_shadowed_scene() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device; skipping the shadowed cross-tier comparison");
        return;
    }
    let scene = occluder_scene(1);
    let (reference, _) = render(abi::backend::SOFT_CPU, 2, &scene);
    let (hardware, _) = render(abi::backend::D3D11, 1, &scene);
    let (reference_unlit, _) = render(abi::backend::SOFT_CPU, 2, &occluder_scene(0));
    let (hardware_unlit, _) = render(abi::backend::D3D11, 1, &occluder_scene(0));

    let (ref_lit, ref_peak) = coverage(&reference);
    let (gpu_lit, gpu_peak) = coverage(&hardware);
    let mean = mean_abs_diff(&reference, &hardware);
    let (worst, differing, pixels) = delta_report(&reference, &hardware);

    // Where the divergence is, not just how much of it there is. The shadow is
    // measured on each tier independently; its interior is erosion to the pixels
    // no boundary can reach, and everything outside the dilation is ground the
    // shadow does not touch at all - where the two tiers have to agree exactly,
    // since both drew the same two flat triangles.
    let ref_shadow = shadow_mask(&reference, &reference_unlit, 20);
    let gpu_shadow = shadow_mask(&hardware, &hardware_unlit, 20);
    let (ref_shadow_px, gpu_shadow_px) = (mask_count(&ref_shadow), mask_count(&gpu_shadow));
    let core = grow(&ref_shadow, 4, false);
    let band = grow(&ref_shadow, 4, true);
    let outside = differing_within(&reference, &hardware, &band, false);
    let in_core = differing_within(&reference, &hardware, &core, true);
    let in_band = differing_within(&reference, &hardware, &band, true);
    eprintln!(
        "shadowed scene: soft-cpu {ref_lit} lit (peak {ref_peak}) vs d3d11 {gpu_lit} lit \
         (peak {gpu_peak}); shadow {ref_shadow_px} px vs {gpu_shadow_px} px of {pixels} \
         ({:.1}% of the frame); mean absolute channel difference {mean:.4}; worst channel \
         delta {worst}; {differing}/{pixels} pixels differ ({:.2}%) - {in_core} in the \
         eroded core, {in_band} in the shadow's 4px neighbourhood, {outside} outside it",
        100.0 * ref_shadow_px as f64 / pixels as f64,
        100.0 * differing as f64 / pixels as f64,
    );
    assert!(
        (ref_lit as i64 - gpu_lit as i64).abs() <= (ref_lit / 50).max(6) as i64,
        "the tiers disagree on how much of the frame is lit: {ref_lit} vs {gpu_lit}"
    );
    // The shadow has to be substantial on both tiers, and the same size on both,
    // before their agreement on it can mean anything.
    assert!(
        ref_shadow_px * 100 >= pixels * 12 && gpu_shadow_px * 100 >= pixels * 12,
        "the shadow covers {ref_shadow_px} px on the reference and {gpu_shadow_px} px on the \
         hardware, too little of {pixels} for the comparison below to be a test of anything"
    );
    let extent_gap = (ref_shadow_px as i64 - gpu_shadow_px as i64).abs();
    assert!(
        extent_gap * 50 <= ref_shadow_px as i64,
        "the tiers disagree on the shadow's extent by {extent_gap} px: {ref_shadow_px} vs \
         {gpu_shadow_px}"
    );
    assert_eq!(
        outside, 0,
        "{outside} pixels outside the shadow's neighbourhood differ between the tiers; the two \
         rasterisers have to agree exactly on everything the shadow does not touch"
    );
    assert_eq!(
        in_core, 0,
        "{in_core} pixels in the shadow's interior differ between the tiers, which is not edge \
         coverage: something in the shading differs away from the boundary"
    );
    assert!(
        in_band * 100 <= pixels,
        "{in_band} of {pixels} pixels differ inside the shadow's neighbourhood, beyond \
         reconl-diff's 1% budget for a tolerated comparison"
    );
    assert!(
        worst <= 48,
        "a channel differs by {worst} between the tiers, beyond the measured 46 that \
         reconl-diff documents as --tolerance=48"
    );
    assert!(
        mean < 0.1,
        "the tiers diverged on the shadowed scene: mean absolute channel difference {mean:.4}"
    );
}

/// The configuration walk: every knob `ReconLShadowConfig` documents, driven
/// through the ABI on both tiers, with what each tier actually ran printed next
/// to what it was asked for. The scene's own configuration is one row of this
/// table; the rest of the knobs - the other cascade counts, the other filters,
/// the map-size budgets and the band widths - were implemented and never
/// exercised, so each is measured here rather than assumed.
#[test]
fn every_shadow_configuration_renders_and_is_reported() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device; skipping the shadow configuration walk");
        return;
    }
    // The unshadowed image does not depend on the shadow configuration, so it
    // is measured once per tier and reused as the baseline every shadow mask is
    // taken against.
    let (reference_unlit, _) = render(abi::backend::SOFT_CPU, 2, &occluder_scene(0));
    let (hardware_unlit, _) = render(abi::backend::D3D11, 1, &occluder_scene(0));

    let cases: [(&str, ShadowCase); 12] = [
        ("c1 pcf3x3 8MiB", ShadowCase::cascades(1)),
        ("c2 pcf3x3 8MiB", ShadowCase::BASE),
        ("c3 pcf3x3 8MiB", ShadowCase::cascades(3)),
        ("c4 pcf3x3 8MiB", ShadowCase::cascades(4)),
        ("c2 hard   8MiB", ShadowCase::filter(0)),
        ("c2 pcf5x5 8MiB", ShadowCase::filter(2)),
        ("c2 pcss   8MiB", ShadowCase::filter(3)),
        ("c2 pcf3x3 2MiB", ShadowCase::budget(2 << 20)),
        ("c2 pcf3x3 32MiB", ShadowCase::budget(32 << 20)),
        ("c1 pcf3x3 4MiB", ShadowCase { cascades: 1, budget_bytes: 4 << 20, ..ShadowCase::BASE }),
        ("c2 pcf3x3 band0", ShadowCase::band(0.0)),
        ("c2 pcf3x3 band8", ShadowCase::band(8.0)),
    ];

    let pixels = (W * H) as usize;
    let mut rows: Vec<(String, abi::ReconLStats, abi::ReconLStats, usize, usize, usize, i32, f64)> = Vec::new();
    for (label, case) in cases {
        let (reference, ref_stats) = render(abi::backend::SOFT_CPU, 2, &occluder_scene_case(case));
        let (hardware, gpu_stats) = render(abi::backend::D3D11, 1, &occluder_scene_case(case));
        let ref_mask = shadow_mask(&reference, &reference_unlit, 20);
        let gpu_mask = shadow_mask(&hardware, &hardware_unlit, 20);
        let ref_shadow = mask_count(&ref_mask);
        let gpu_shadow = mask_count(&gpu_mask);
        let (worst, differing, _) = delta_report(&reference, &hardware);
        let mean = mean_abs_diff(&reference, &hardware);
        rows.push((label.to_string(), ref_stats, gpu_stats, ref_shadow, gpu_shadow, differing, worst, mean));
        // Where the tiers disagree, not just how much: everything outside the
        // reference shadow's 4px neighbourhood, and everything inside its eroded
        // interior, has to be identical - both tiers drew the same flat ground
        // and the same caster. Only the boundary is allowed to differ.
        // The edge zone is the neighbourhood of the shadow *either* tier sees,
        // and the interior is the eroded area *both* agree is shadow. Using the
        // reference's mask alone would assume the two tiers' shadow extents agree
        // to within the radius, which holds at 512x512 and does not at 256x256,
        // where a texel covers four times the screen area and the boundary itself
        // is several pixels wide.
        let either: Vec<bool> = ref_mask
            .iter()
            .zip(&gpu_mask)
            .map(|(a, b)| *a || *b)
            .collect();
        let both: Vec<bool> = ref_mask
            .iter()
            .zip(&gpu_mask)
            .map(|(a, b)| *a && *b)
            .collect();
        // The radius is the widest divergence measured for any row at this map
        // size and filter: 4px at 512 and 1024 texels, 6px at 256, and 5px where
        // the filters themselves differ (a 5x5 penumbra is simply wider).
        let same_filter = ref_stats.shadows.filter_active == gpu_stats.shadows.filter_active;
        let radius = if !same_filter || ref_stats.shadows.map_width < 512 {
            8
        } else {
            4
        };
        let band = grow(&either, radius, true);
        let core = grow(&both, radius, false);
        let outside = differing_within(&reference, &hardware, &band, false);
        let in_core = differing_within(&reference, &hardware, &core, true);
        let in_band = differing_within(&reference, &hardware, &band, true);
        let how_far = max_divergence_distance(&reference, &hardware, &either);
        eprintln!(
            "      {label}: {outside} of {differing} differ outside the {radius}px band, {in_core} in \
             the shared interior, farthest divergence {how_far}px from any shadow"
        );
        // A plan the two tiers did not both run is not a cross-tier comparison:
        // the reference caps cascades at 2 and the filter at pcf3x3, so c3/c4,
        // pcf5x5 and pcss are configurations only the hardware tier can run. The
        // reference has no oracle for those, and saying so is the honest result.
        let comparable = ref_stats.shadows.cascades_active == gpu_stats.shadows.cascades_active
            && ref_stats.shadows.filter_active == gpu_stats.shadows.filter_active
            && ref_stats.shadows.map_width == gpu_stats.shadows.map_width;
        // Every row, comparable or not: whatever changed, it changed the shadow's
        // edge and nothing else. The lit frame away from the shadow is the same
        // two flat triangles on both tiers, so any divergence there is a bug in
        // the shading or the fit rather than in the filter or the split.
        assert_eq!(
            outside, 0,
            "{label}: {outside} of the {differing} differing pixels are outside the {radius}px \
             neighbourhood of the shadow: the tiers have to agree exactly on everything the shadow \
             does not touch (farthest divergence {how_far}px)"
        );
        if same_filter {
            assert_eq!(
                in_core, 0,
                "{label}: {in_core} pixels in the interior both tiers call shadow differ, which is \
                 not edge coverage"
            );
        }
        if comparable {
            assert!(
                in_band * 100 <= pixels,
                "{label}: {in_band} of {pixels} pixels differ on the shadow's edge, beyond \
                 reconl-diff's 1% budget"
            );
            // A one-tap filter has no partial coverage to disagree about, so a
            // single texel on the caster's silhouette flips the full contrast
            // (measured: 1 pixel, delta 90, against 14 pixels at delta 46 for
            // pcf3x3). The bound is therefore on the count when the count is
            // tiny, and on the delta otherwise.
            assert!(
                worst <= 48 || differing <= 2,
                "{label}: a channel differs by {worst} on {differing} pixels - too much for \
                 partial coverage at the shadow's edge"
            );
            assert!(mean < 0.1, "{label}: mean absolute channel difference {mean:.4}");
        } else {
            eprintln!(
                "    (plans differ, so only the localisation above is asserted: the tiers ran \
                 {}/{}/{} and {}/{}/{})",
                ref_stats.shadows.cascades_active,
                ref_stats.shadows.map_width,
                ref_stats.shadows.filter_active,
                gpu_stats.shadows.cascades_active,
                gpu_stats.shadows.map_width,
                gpu_stats.shadows.filter_active,
            );
        }
        eprintln!(
            "{label}: requested(c{}/f{}/b{}MB/band{}) | soft-cpu: cascades {}, map {}x{}, \
             filter {}/{}, {ref_shadow} px shadow | d3d11: cascades {}, map {}x{}, filter {}/{}, \
             {gpu_shadow} px shadow | cross-tier {differing}/{pixels} differ ({:.2}%), worst {worst}, \
             mean {mean:.4}",
            case.cascades,
            case.filter,
            case.budget_bytes >> 20,
            case.band,
            ref_stats.shadows.cascades_active,
            ref_stats.shadows.map_width,
            ref_stats.shadows.map_height,
            ref_stats.shadows.filter_active,
            ref_stats.shadows.filter_requested,
            gpu_stats.shadows.cascades_active,
            gpu_stats.shadows.map_width,
            gpu_stats.shadows.map_height,
            gpu_stats.shadows.filter_active,
            gpu_stats.shadows.filter_requested,
            100.0 * differing as f64 / pixels as f64,
        );
        // Whatever the tier did with the request, shadows are on and the map is
        // real: a request that produced no map at all would make every number
        // above meaningless.
        assert!(
            ref_stats.shadows.cascades_active >= 1 && gpu_stats.shadows.cascades_active >= 1,
            "{label}: a cascade is active on both tiers; got {}/{} - {}",
            ref_stats.shadows.cascades_active,
            gpu_stats.shadows.cascades_active,
            ref_stats.shadows.map_width,
        );
        assert!(
            ref_stats.shadows.map_width == ref_stats.shadows.map_height
                && gpu_stats.shadows.map_width == gpu_stats.shadows.map_height
                && ref_stats.shadows.map_width > 0
                && gpu_stats.shadows.map_width > 0,
            "{label}: both tiers must report a square map; got {}x{} and {}x{}",
            ref_stats.shadows.map_width,
            ref_stats.shadows.map_height,
            gpu_stats.shadows.map_width,
            gpu_stats.shadows.map_height,
        );
        assert!(
            ref_stats.shadows.filter_requested == case.filter
                && gpu_stats.shadows.filter_requested == case.filter,
            "{label}: the tier must echo the requested filter; got {}/{}",
            ref_stats.shadows.filter_requested,
            gpu_stats.shadows.filter_requested,
        );
        assert!(
            ref_shadow * 100 >= pixels * 12 && gpu_shadow * 100 >= pixels * 12,
            "{label}: the shadow covers {ref_shadow} px on the reference and {gpu_shadow} px on the \
             hardware, too little of {pixels} for the comparison to mean anything"
        );
    }

    // Cross-case invariants: the point of walking the matrix is that each knob
    // moves a number in a direction a regression cannot fake. Each of these is
    // measured on *both* tiers independently, so a backend whose kernel, split
    // or band drifted fails on its own rather than only in the comparison.
    let row = |name: &str| -> &(String, abi::ReconLStats, abi::ReconLStats, usize, usize, usize, i32, f64) {
        rows.iter()
            .find(|r| r.0 == name)
            .unwrap_or_else(|| panic!("no row for {name}"))
    };
    let extent = |name: &str, tier: usize| -> usize {
        let r = row(name);
        if tier == 0 {
            r.3
        } else {
            r.4
        }
    };
    for (tier, whose) in [(0usize, "soft-cpu"), (1usize, "d3d11")] {
        // Filter width: a wider kernel spreads the darkening past the hard edge,
        // so the shadow's measured footprint grows. A mis-set tap count or a
        // kernel that silently fell back to the wrong radius collapses this.
        let hard = extent("c2 hard   8MiB", tier);
        let pcf3 = extent("c2 pcf3x3 8MiB", tier);
        assert!(
            hard < pcf3,
            "{whose}: hard PCF should darken fewer pixels than pcf3x3 (a wider kernel spreads the \
             penumbra outward); measured {hard} vs {pcf3}"
        );
        // Crossfade band: no band leaves the cascade boundary hard, a wider band
        // softens it, so the footprint shrinks monotonically in band width.
        let band0 = extent("c2 pcf3x3 band0", tier);
        let band3 = extent("c2 pcf3x3 8MiB", tier);
        let band8 = extent("c2 pcf3x3 band8", tier);
        assert!(
            band0 > band3 && band3 > band8,
            "{whose}: a wider crossfade band must soften the boundary monotonically; measured \
             band0 {band0}, band3 {band3}, band8 {band8}"
        );
    }
    // pcf5x5 and 3 cascades exist only on the hardware tier (the reference caps
    // at pcf3x3 and 2 cascades), so they are pinned by their own invariants
    // rather than against a reference that cannot run them.
    let gpu_pcf3 = extent("c2 pcf3x3 8MiB", 1);
    let gpu_pcf5 = extent("c2 pcf5x5 8MiB", 1);
    assert!(
        gpu_pcf5 > gpu_pcf3,
        "d3d11: pcf5x5 must spread the penumbra wider than pcf3x3; measured {gpu_pcf5} vs {gpu_pcf3}"
    );
    let gpu_c2 = extent("c2 pcf3x3 8MiB", 1);
    let gpu_c3 = extent("c3 pcf3x3 8MiB", 1);
    assert!(
        gpu_c3 != gpu_c2,
        "d3d11: three cascades must render a different split from two; both measured {gpu_c2}"
    );
    // Map size: the same filter and split at 256, 512 and 1024 texels. A coarser
    // map puts more of the caster's silhouette on a texel, so the two tiers'
    // texel-boundary disagreements grow as the map shrinks - monotone in three
    // steps, on both tiers, which is what pins the fit's map-size handling.
    let d256 = row("c2 pcf3x3 2MiB").5;
    let d512 = row("c2 pcf3x3 8MiB").5;
    let d1024 = row("c2 pcf3x3 32MiB").5;
    assert!(
        d256 > d512 && d512 > d1024,
        "the tiers' texel-boundary disagreement must shrink as the map grows; measured {d256} px \
         at 256, {d512} at 512, {d1024} at 1024"
    );
    // A request only one tier can honour is *reported*, not silently
    // substituted: the host can see which of its requests were clamped.
    let r = row("c4 pcf3x3 8MiB");
    assert!(
        r.1.shadows.cascades_active == 2 && r.2.shadows.cascades_active == 3,
        "a request for 4 cascades must be clamped per tier and reported: reference {}, hardware {}",
        r.1.shadows.cascades_active,
        r.2.shadows.cascades_active
    );
    let r = row("c2 pcss   8MiB");
    assert!(
        r.1.shadows.filter_requested == 3 && r.1.shadows.filter_active == 1,
        "the reference tier must report pcss-lite as requested and pcf3x3 as active, not silently \
         substitute: requested {}, active {}",
        r.1.shadows.filter_requested,
        r.1.shadows.filter_active
    );
    assert!(
        r.2.shadows.filter_requested == 3 && r.2.shadows.filter_active == 2,
        "the hardware tier must report pcss-lite as requested and pcf5x5 as active: requested {}, \
         active {}",
        r.2.shadows.filter_requested,
        r.2.shadows.filter_active
    );
}

/// Two overlapping triangles at different depths with a half-transparent one in
/// front: a red backdrop, a blue front at alpha 0.5, both unlit so only the
/// blend arithmetic is under test. Reversed-Z with a strict `>` comparison means
/// the front triangle at the higher depth wins the test and blends over the back.
fn blend_scene(mode: u32) -> Scene {
    let back = TestDraw {
        verts: vec![
            v([-0.9, -0.9, 0.4], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
            v([0.9, -0.9, 0.4], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
            v([0.0, 0.9, 0.4], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
        ],
        indices: vec![0, 1, 2],
        shading: abi::shading::UNLIT,
        blend: 0, // the backdrop is opaque whatever mode is under test
        cull: 0,
        receives_shadow: 0,
        casts_shadow: 0,
    };
    let front = TestDraw {
        verts: vec![
            v([-0.9, -0.9, 0.6], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.5]),
            v([0.9, -0.9, 0.6], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.5]),
            v([0.0, 0.9, 0.6], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.5]),
        ],
        indices: vec![0, 1, 2],
        shading: abi::shading::UNLIT,
        blend: mode,
        cull: 0,
        receives_shadow: 0,
        casts_shadow: 0,
    };
    Scene {
        draws: vec![back, front],
        // Identity view-projection: this scene is written directly in clip
        // space, so the geometry is exactly where it is specified.
        view_proj: IDENTITY,
        // No camera: this scene is written in clip space, so the identity view
        // and the default frustum are exactly what it means. Shadows are off
        // here, and it keeps the no-camera path covered.
        camera: None,
        light: LightSpec { direction: [0.0, 0.0, -1.0], intensity: 1.0, cast_shadow: 0 },
        shadow: ShadowSpec {
            enabled: 0,
            cascades: 1,
            filter: 0,
            budget_bytes: 8 << 20,
            max_distance: 60.0,
            blend_band: 0.0,
            normal_bias: 0.0,
            depth_bias: 0.0,
            slope_bias: 0.0,
        },
    }
}

/// One triangle, written in clip space, wound the way the reference calls
/// front-facing: counter-clockwise in the y-up NDC it is specified in, which is
/// clockwise on the y-down screen both tiers rasterise into.
fn single_triangle_scene(cull: u32) -> Scene {
    let front = TestDraw {
        verts: vec![
            v([-0.9, -0.9, 0.4], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
            v([0.9, -0.9, 0.4], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
            v([0.0, 0.9, 0.4], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
        ],
        indices: vec![0, 1, 2],
        shading: abi::shading::UNLIT,
        blend: 0,
        cull,
        receives_shadow: 0,
        casts_shadow: 0,
    };
    Scene {
        draws: vec![front],
        view_proj: IDENTITY,
        camera: None,
        light: LightSpec { direction: [0.0, 0.0, -1.0], intensity: 1.0, cast_shadow: 0 },
        shadow: ShadowSpec {
            enabled: 0,
            cascades: 1,
            filter: 0,
            budget_bytes: 8 << 20,
            max_distance: 60.0,
            blend_band: 0.0,
            normal_bias: 0.0,
            depth_bias: 0.0,
            slope_bias: 0.0,
        },
    }
}

/// Which face a `CULL_*` constant names is part of the ABI, not of a backend:
/// `CULL_BACK` has to keep the triangle the reference calls front-facing, and
/// `CULL_FRONT` has to drop it, on every tier.
///
/// This is the pin for a defect the shadow work uncovered: the hardware tier set
/// `FrontCounterClockwise` the wrong way round, which inverted every `CULL_*`
/// constant there. Nothing in this suite used a culling pipeline, so it went
/// unnoticed until the shadow pass - which culls the far side by design - culled
/// the caster instead and rendered an empty map.
#[test]
fn both_tiers_classify_faces_the_same_way() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device; skipping the face-classification comparison");
        return;
    }
    let front_kept = |backend: u32, tier: u32| coverage(&render(backend, tier, &single_triangle_scene(CULL_BACK)).0).0;
    let front_dropped = |backend: u32, tier: u32| coverage(&render(backend, tier, &single_triangle_scene(CULL_FRONT)).0).0;

    let (reference_kept, hardware_kept) = (front_kept(abi::backend::SOFT_CPU, 2), front_kept(abi::backend::D3D11, 1));
    let (reference_dropped, hardware_dropped) =
        (front_dropped(abi::backend::SOFT_CPU, 2), front_dropped(abi::backend::D3D11, 1));
    eprintln!(
        "cull back: soft-cpu {reference_kept} px, d3d11 {hardware_kept} px; cull front: soft-cpu \
         {reference_dropped} px, d3d11 {hardware_dropped} px"
    );

    assert!(
        reference_kept > 0 && reference_dropped == 0,
        "the reference did not classify this triangle the way the scene assumes: {reference_kept} \
         pixels kept culling back, {reference_dropped} kept culling front"
    );
    assert_eq!(
        (hardware_kept, hardware_dropped),
        (reference_kept, reference_dropped),
        "the hardware tier culls the opposite face from the reference for the same CULL_* constant"
    );
}

/// Every blend mode as fixed-function state, checked against the reference's
/// hand-written `shade::blend`. The hardware mapping is the part of this backend
/// most likely to be subtly wrong, and a wrong mapping shows up as a colour that
/// differs where the two triangles overlap.
#[test]
fn every_blend_mode_matches_the_reference() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device; skipping the blend-mode comparison");
        return;
    }
    let (reference_opaque, _) = render(abi::backend::SOFT_CPU, 2, &blend_scene(0));
    for mode in 0..4u32 {
        let scene = blend_scene(mode);
        let (reference, _) = render(abi::backend::SOFT_CPU, 2, &scene);
        let (hardware, _) = render(abi::backend::D3D11, 1, &scene);
        let tier_gap = mean_abs_diff(&reference, &hardware);
        let mode_gap = mean_abs_diff(&reference_opaque, &reference);
        eprintln!(
            "blend mode {mode}: tiers differ by {tier_gap:.4}, reference differs from opaque by \
             {mode_gap:.4}"
        );
        assert!(
            tier_gap < 3.0,
            "blend mode {mode} differs between the tiers by {tier_gap:.3} per channel: the \
             fixed-function state does not reproduce shade::blend"
        );
        // A mode that changed nothing on either tier would make the comparison
        // above vacuous, so the reference must actually look different from
        // opaque for every non-opaque mode.
        if mode != 0 {
            assert!(
                mode_gap > 1.0,
                "blend mode {mode} is indistinguishable from opaque on the reference tier, so \
                 this comparison proves nothing"
            );
        }
    }
}

// ------------------------------------------------------------------- offload

/// The resolution the offload tests run at. At 64x64 the two tiers render in a
/// few milliseconds of each other and machine load can flip the sign of the
/// calibration's comparison between one frame and the next; at 256x256 the
/// reference tier is about seven times slower with a single worker (30.7 ms
/// against 4.2 ms, measured by `reconl-bench --threads=1`), which no plausible
/// noise moves, and a frame still costs tens of milliseconds rather than
/// hundreds.
const OFFLOAD_W: u32 = 256;
const OFFLOAD_H: u32 = 256;

/// A hardware device run at any size, with the ladder settings the offload policy
/// reads. `reconlPresent` is what steps that policy and nothing else does, so a
/// run's per-frame stats are the only way to watch it decide.
fn ladder_at(size: u32, frames: u32, target_ms: u32, after: u32, allow: u32) -> Run {
    Run {
        allow_downgrade: allow,
        target_frame_ms: target_ms,
        downgrade_after_frames: after,
        frames,
        width: size,
        height: size,
        ..Run::one(abi::backend::D3D11, 1)
    }
}

/// The same at the offload resolution, where the legs that are not ladder legs
/// run: no offload allowed, or a target no device here can miss.
fn ladder(frames: u32, target_ms: u32, after: u32, allow: u32) -> Run {
    ladder_at(OFFLOAD_W, frames, target_ms, after, allow)
}

/// The reference tier's own render of the same frames, for the two things the
/// offload tests compare against: its frame cost, and its pixels.
fn reference_frames(scene: &Scene) -> Vec<(Vec<u8>, abi::ReconLStats)> {
    run(
        Run {
            frames: 2,
            width: OFFLOAD_W,
            height: OFFLOAD_H,
            ..Run::one(abi::backend::SOFT_CPU, 2)
        },
        scene,
    )
}

/// The reason codes the tier log carries, straight from the library's own
/// enum, so a test cannot pass by agreeing with a stale copy of them.
fn reason(reason: reconl_core::tier::TierReason) -> u32 {
    reason as u32
}

fn detail_of(entry: &abi::ReconLDowngrade) -> String {
    let len = entry.detail.iter().position(|&b| b == 0).unwrap_or(entry.detail.len());
    String::from_utf8_lossy(&entry.detail[..len]).into_owned()
}

/// A ring's entries as a host reads them: the change, the frame it happened on,
/// and the reason it carries.
fn ring(stats: &abi::ReconLStats) -> Vec<(u32, u32, u32, u64, String)> {
    let count = (stats.downgrade_count as usize).min(stats.downgrade_capacity as usize);
    (0..count)
        .map(|i| {
            let e = &stats.downgrades[i];
            (e.from, e.to, e.reason, e.frame_index, detail_of(e))
        })
        .collect()
}

/// The number that follows `marker` in a downgrade's detail.
///
/// The policy writes the measurement that decided it into the entry it caused -
/// the offload entry carries the hardware cost the calibration is compared
/// against, the return entry the comparison it acted on - so a test can read the
/// numbers the policy itself used rather than measuring two more devices and
/// hoping the machine schedules them the same way.
fn number_after(detail: &str, marker: &str) -> Option<u64> {
    let rest = &detail[detail.find(marker)? + marker.len()..];
    let digits: String = rest
        .trim_start_matches(|c: char| !c.is_ascii_digit())
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

// ---------------------------------------------------- the ladder legs, once

/// How many frames a ladder leg drives, and how many quiet frames at its end
/// count as "the ladder has quit". The policy's own state bounds the changes a
/// plan can take - one return, then one offload that reuses the measurement - so
/// a quiet tail is a verdict and not a guess, and a leg is driven to it rather
/// than sampled at a fixed frame.
const LADDER_FRAMES: u32 = 24;
const LADDER_QUIET_FRAMES: usize = 3;

/// How many times a leg re-derives its window before it gives up.
///
/// The window is a measurement of this host, taken while the rest of the suite
/// may be loading it, so it can stop being reachable between the pilot run and
/// the leg run. Re-measuring is what makes the leg deterministic without weakening
/// it: every attempt drives the same sequence and is judged by the same clause.
const LADDER_ATTEMPTS: u32 = 3;

/// The window a ladder leg points the device at, measured out of both tiers on
/// this machine.
///
/// A hard-coded millisecond makes the verdict a property of the host, and the
/// number answers two questions of which the *slower* one decides which binds:
/// the hardware has to *miss* the target - that miss is the trigger - and where
/// the reference tier is the faster tier, the one that wins the calibration and
/// then has to arm the settle window on its own frames, that tier needs a frame
/// *inside* it. A fresh device's first frame bounds the target from above,
/// because it is the frame the offload fires on and the first frame a rebuilt
/// device presents.
#[derive(Clone, Copy)]
struct LadderWindow {
    target_ms: u32,
    cpu_cheapest_ns: u64,
    cpu_typical_ns: u64,
    gpu_first_ns: u64,
}

/// Measures both tiers with their ladders off and derives the window, or fails
/// with both measurements.
///
/// The target is the ABI's smallest whole millisecond, raised to the smallest
/// millisecond that covers the reference tier's **cheapest** warm frame only when
/// a fresh hardware device's first frame is over **twice** the raised value.
///
/// Cheapest, not typical: the settle window needs *one* frame inside the target
/// (`within_target` counts frames inside it and this threshold is one), so the
/// cheap end is what the window actually needs - and raising the target on the
/// cheap end makes the raise a property of a *slow host*, where every frame of
/// both tiers is inflated together, instead of a property of *load*, where the
/// reference tier's median frame drifts over a millisecond while the hardware's
/// first frame does not. That distinction is the difference between a pin and a
/// coin flip: the target doubles, and a rebuilt hardware device's first frame is
/// no longer reliably over it.
///
/// Twice, not merely over it, for the ceiling: the first frame is a device setup
/// frame, and the spread between one cold device and the next on a loaded machine
/// is large enough that a target only just under it stops being a target for the
/// device a leg then builds. Requiring real room means the raised target is only
/// ever used in the regime that needs it - the remembered leg, where the
/// reference tier wins the calibration and its own frames have to re-arm the
/// settle window - and every other machine, including a thrashing one, gets the
/// smallest millisecond, which every hardware frame measured here was over.
fn ladder_window(leg: &str, scene: &Scene, size: u32) -> Result<LadderWindow, String> {
    // A tier's first frame, its cheapest warm frame and its typical warmth, with
    // the ladder off.
    let pilot = |backend: u32, tier: u32| -> (u64, u64, u64) {
        let frames = run(
            Run { frames: 8, width: size, height: size, ..Run::one(backend, tier) },
            scene,
        );
        let mut warm: Vec<u64> = frames
            .iter()
            .skip(1)
            .map(|(_, stats)| stats.frame.total_ns)
            .filter(|ns| *ns > 0)
            .collect();
        warm.sort_unstable();
        let first = frames.first().map(|(_, stats)| stats.frame.total_ns).unwrap_or(0);
        (first, warm.first().copied().unwrap_or(0), warm.get(warm.len() / 2).copied().unwrap_or(0))
    };
    let (cpu_first, cpu_cheapest, cpu_typical) = pilot(abi::backend::SOFT_CPU, 2);
    let (gpu_first, _gpu_cheapest, _gpu_typical) = pilot(abi::backend::D3D11, 1);
    let cpu_ms = ((cpu_cheapest + 999_999) / 1_000_000).max(1) as u32;
    let cap_ms = gpu_first.saturating_sub(1) / 1_000_000;
    let target_ms = if u64::from(cpu_ms) * 2 <= cap_ms { cpu_ms } else { 1 };
    if cap_ms < 1 || u64::from(target_ms) > cap_ms {
        return Err(format!(
            "{leg}: no whole-millisecond target this host can use. The reference tier's first frame \
             is {cpu_first} ns, its cheapest warm frame {cpu_cheapest} ns (typical {cpu_typical} ns), \
             which {cpu_ms} ms would cover, while a fresh hardware device's first frame is \
             {gpu_first} ns - {cap_ms} ms of room over the hardware, which is not twice what \
             covering the reference tier's frames would need. The sequence this leg pins cannot be \
             exercised on this host in this measurement."
        ));
    }
    Ok(LadderWindow {
        target_ms,
        cpu_cheapest_ns: cpu_cheapest,
        cpu_typical_ns: cpu_typical,
        gpu_first_ns: gpu_first,
    })
}

/// One device's ladder run, with a host's reading of it in one place: the offload
/// entries, the returns, and the bound the anti-thrash rule puts on every leg.
struct LadderRun {
    leg: &'static str,
    frames: Vec<(Vec<u8>, abi::ReconLStats)>,
}

/// Drives one hardware device through the ladder at `window` - the host's bit
/// set, the policy's own thresholds - and reports the trigger the window was
/// derived against: the arm's device is a fresh one, so its first frame is a
/// device setup frame, and it has to miss. `Err` is this host crossing its own
/// measured window, which is what the caller re-derives for, and not a verdict
/// about the ladder.
fn ladder_run(leg: &'static str, scene: &Scene, size: u32, window: LadderWindow) -> Result<LadderRun, String> {
    let frames = run(
        ladder_at(size, LADDER_FRAMES, window.target_ms, 1, abi::allow_downgrade::TIER),
        scene,
    );
    let first = &frames[0].1;
    if first.safe_path_events != 1 || first.backend != abi::backend::SOFT_CPU {
        return Err(format!(
            "{leg}: a hardware device's first frame cost {} ns and did not miss the derived {} ms \
             target (the reference tier's cheapest frame here is {} ns, its typical {} ns, and a \
             fresh hardware device's first frame {} ns; backend {}, safe-path events {}): the \
             offload this sequence is built on did not happen",
            first.frame.total_ns,
            window.target_ms,
            window.cpu_cheapest_ns,
            window.cpu_typical_ns,
            window.gpu_first_ns,
            first.backend,
            first.safe_path_events
        ));
    }
    Ok(LadderRun { leg, frames })
}

/// A ladder leg on a window derived from this machine, re-derived when the leg
/// does not say what it exists to say.
///
/// `reachable` is the caller's verdict on one run - the sequence that leg is
/// about - and it is the same clause the caller asserts afterwards, so nothing is
/// weakened by asking it here: the retry buys the leg a fresh measurement of the
/// host rather than a weaker standard, and a leg that never satisfies it is
/// returned as it stands so the caller's own assertion fails with everything the
/// leg measured. Only a host where *no* attempt could even be set up panics here,
/// with the derivation's own reason.
fn ladder_leg(
    leg: &'static str,
    scene: &Scene,
    size: u32,
    reachable: impl Fn(&LadderRun) -> bool,
) -> (LadderWindow, LadderRun) {
    let mut why = String::new();
    let mut last: Option<(LadderWindow, LadderRun)> = None;
    for attempt in 1..=LADDER_ATTEMPTS {
        match ladder_window(leg, scene, size).and_then(|window| {
            ladder_run(leg, scene, size, window).map(|run| (window, run))
        }) {
            Ok((window, run)) => {
                if reachable(&run) {
                    return (window, run);
                }
                why = format!("{leg}: attempt {attempt} drove the sequence but it did not reach its verdict");
                last = Some((window, run));
            }
            Err(reason) => why = reason,
        }
    }
    last.unwrap_or_else(|| panic!("{leg}: no window this host can use after {LADDER_ATTEMPTS} measurements: {why}"))
}

impl LadderRun {
    fn frames(&self) -> &[(Vec<u8>, abi::ReconLStats)] {
        &self.frames
    }

    fn last(&self) -> &abi::ReconLStats {
        &self.frames.last().expect("a leg rendered frames").1
    }

    /// The offload entries the policy wrote, in order. The reference tier's own
    /// relabels share `FRAME_TIME_OVER_TARGET` and say so in their detail
    /// (`this device's own tier`); they are a different decision - the device's
    /// *tier*, not its backend - so they are not offloads.
    fn offloads(&self) -> Vec<String> {
        let last = self.last();
        let mut entries = Vec::new();
        for i in 0..last.downgrade_capacity as usize {
            let entry = &last.downgrades[i];
            if entry.reason != reason(reconl_core::tier::TierReason::FrameTimeOverTarget) {
                continue;
            }
            let detail = detail_of(entry);
            if !detail.contains("this device's own tier") {
                entries.push(detail);
            }
        }
        entries
    }

    /// How many times the device returned to the hardware. Counted by reason, not
    /// by the tier it came from: the reference tier's own ladder may have
    /// relabelled itself before the return, so a return can be logged from T3 or
    /// T4 rather than from the T2 the offload adopted.
    fn returns(&self) -> usize {
        let last = self.last();
        (0..last.downgrade_capacity as usize)
            .filter(|i| {
                last.downgrades[*i].reason == reason(reconl_core::tier::TierReason::Recovery)
            })
            .count()
    }

    /// The rule every leg obeys, and the thing that makes "the ladder quit"
    /// checkable rather than assumed: at most three backend changes per plan - the
    /// offload, the return the settle window bought, and the offload that reuses
    /// the measurement - at most one of them a return, and `LADDER_QUIET_FRAMES`
    /// frames at the end that changed nothing.
    fn assert_settled_within_bounds(&self) {
        let leg = self.leg;
        let last = self.last();
        assert_eq!(
            last.frames_presented, LADDER_FRAMES,
            "{leg}: {} of {LADDER_FRAMES} frames were presented",
            last.frames_presented
        );
        assert_eq!(last.frames_dropped, 0, "{leg}: a frame was dropped");
        assert_eq!(last.failures, 0, "{leg}: the host saw a failed call");
        assert!(
            last.safe_path_events <= 3,
            "{leg}: the device changed backend {} times: one plan is allowed one round trip",
            last.safe_path_events
        );
        let returns = self.returns();
        assert!(returns <= 1, "{leg}: the device returned to the hardware {returns} times");
        let tail = &self.frames[self.frames.len() - LADDER_QUIET_FRAMES..];
        assert!(
            tail.iter().all(|(_, stats)| stats.safe_path_events == last.safe_path_events),
            "{leg}: the ladder was still changing backend in its last {LADDER_QUIET_FRAMES} frames \
             ({} -> {})",
            tail[0].1.safe_path_events,
            last.safe_path_events
        );
        assert!(
            last.safe_path_events < 2 || returns == 1,
            "{leg}: the device took {} backend changes but the log holds {returns} returns to the \
             hardware",
            last.safe_path_events
        );
    }
}

/// The core of the offload, end to end on the real ABI: a hardware tier that
/// misses its frame-time target hands the next frame to the reference tier, that
/// frame is the calibration, and the measurement the policy took for itself
/// decides whether the device stays there or comes back. The frame it hands over
/// is the reference tier's frame, byte for byte.
///
/// This is the ladder leg that runs the *shadowed* plan key - `occluder_scene(1)`
/// is two cascades, PCF 3x3 and a bounded shadow-map budget, which is what the
/// policy files its measurement under - so the ladder is exercised with a plan
/// whose shadow work and whose plan key are both real.
#[test]
fn an_overloaded_hardware_tier_offloads_and_the_measurement_decides() {
    if !d3d11_usable() {
        return;
    }
    let scene = occluder_scene(1);
    let leg = "the one-round-trip leg (256x256, shadowed plan)";

    // The reference tier's own render of the same frame, for the byte-for-byte
    // check below. It renders two frames because the calibration frame is frame
    // index 1, not 0, and a frame's content depends on its index.
    let reference = reference_frames(&scene);

    let (_, run) = ladder_leg(leg, &scene, OFFLOAD_W, |_| true);
    let frames = run.frames();

    // Frame 0 was the hardware's, and it missed the derived target: the device is
    // on the reference tier before frame 1 is recorded. (`ladder_run` has already
    // failed if the miss did not happen.)
    let first = &frames[0].1;
    assert_eq!(first.safe_path_events, 1, "{leg}: one backend change: the offload itself");
    assert_eq!(
        first.downgrades[0].reason,
        reason(reconl_core::tier::TierReason::FrameTimeOverTarget),
        "{leg}: the offload is not logged with the reason that caused it"
    );
    assert_eq!(first.tier, 2, "{leg}: the reference tier the device continued on");
    assert_eq!(first.frames_presented, 1);
    assert_eq!(first.failures, 0, "{leg}: no call failed: the host saw a frame, not an error");

    // Frame 1 is the calibration, rendered by the reference tier, and it is the
    // same frame the reference tier renders given the same input.
    assert_eq!(
        frames[1].0, reference[1].0,
        "{leg}: the offloaded frame differs from the reference tier's own render of it"
    );
    assert_eq!(frames[1].1.frames_presented, 2);

    // The decision follows the measurement, and only the measurement: the
    // device is back on the hardware exactly when the reference tier measured
    // slower there. Both costs are the policy's own, read back from the entries
    // it wrote, rather than measured again here: measuring them again measures
    // two other devices on a busy machine, and the hardware tier's first frame
    // in a cold process costs several times its warm one, which makes such an
    // assertion about the scheduler rather than about the policy.
    let second = &frames[1].1;
    let offload_detail = detail_of(&frames[0].1.downgrades[0]);
    let gpu_ns = number_after(&offload_detail, "the hardware measured ")
        .unwrap_or_else(|| panic!("{leg}: the offload records no hardware cost: {offload_detail}"));
    let cpu_ns = if second.backend == abi::backend::D3D11 {
        let entry = second
            .downgrades
            .iter()
            .find(|d| d.reason == reason(reconl_core::tier::TierReason::Recovery))
            .unwrap_or_else(|| panic!("{leg}: the return to the hardware is not in the tier log"));
        let detail = detail_of(entry);
        number_after(&detail, "the reference tier measured ")
            .unwrap_or_else(|| panic!("{leg}: the return records no comparison: {detail}"))
    } else {
        // Still on the reference tier, so its own last frame is the calibration.
        second.frame.total_ns
    };
    assert!(
        cpu_ns > 1_000_000,
        "{leg}: the reference tier's measured cost is not a frame's cost: {cpu_ns} ns"
    );
    // And it is a real cost, not a number the policy invented: the same frame
    // rendered by a reference-tier device of its own is the same order of
    // magnitude. Not the same number - the calibration is a fresh device's first
    // frame - so the band is wide.
    let control_ns = reference[1].1.frame.total_ns;
    assert!(
        control_ns > 0 && cpu_ns >= control_ns / 8 && cpu_ns <= control_ns * 8,
        "{leg}: the recorded reference-tier cost ({cpu_ns} ns) is not the same order as the tier's \
         own measurement of that frame ({control_ns} ns)"
    );
    let cpu_slower = cpu_ns >= gpu_ns;
    assert_eq!(
        second.backend == abi::backend::D3D11,
        cpu_slower,
        "{leg}: the calibration decided against its own measurement (reference {cpu_ns} ns, \
         hardware {gpu_ns} ns)"
    );
    if cpu_slower {
        assert_eq!(second.safe_path_events, 2, "{leg}: the return is a second backend change");
        // The reason is the offload's own: the hardware tier's frame-time ladder
        // (telemetry that relabels a GPU tier, and pre-dates the offload) says
        // FRAME_TIME_OVER_TARGET, so RECOVERY can only come from this policy.
        assert_eq!(
            second.tier_reason,
            reason(reconl_core::tier::TierReason::Recovery),
            "{leg}: the device came back to the hardware without recording why"
        );
        // Found by reason, not by position: the frame's own tier step - if the run
        // had already armed one when the calibration frame closed - is recorded
        // *before* the return, which is the order they happened in
        // (`docs/offload.md`, "Two layers, one number").
        let entries = ring(second);
        let recovery = entries
            .iter()
            .rev()
            .find(|(_, _, why, _, _)| *why == reason(reconl_core::tier::TierReason::Recovery))
            .unwrap_or_else(|| {
                panic!("{leg}: the device came back to the hardware without recording why: {entries:?}")
            });
        assert!(
            recovery.4.contains("measured"),
            "{leg}: the return does not record the measurement that caused it: {}",
            recovery.4
        );
        assert_eq!(
            entries.last().map(|entry| entry.2),
            Some(reason(reconl_core::tier::TierReason::Recovery)),
            "{leg}: the backend change is not the last thing the frame recorded: {entries:?}"
        );
    }

    // The bound on every leg of the ladder, and the quiet tail that shows the
    // ladder quit instead of sampling one frame and assuming it had: the returned
    // trip is attempted at most once per plan, which is what keeps a workload
    // oscillating around the target from thrashing between tiers, and the third
    // backend change it allows is the offload that reuses the measurement instead
    // of paying for a second calibration (see the remembered leg below, which
    // reaches that branch; at this size the reference tier always loses the
    // calibration, so the device goes back to the hardware and stays).
    run.assert_settled_within_bounds();
}

/// The shadowed plan key the remembered leg runs at: the occluder scene at 32x32,
/// with the shadow map's texel budget cut so the reference tier's own shadow pass
/// leaves whole milliseconds between it and the hardware - the window the
/// remembered sequence needs. The key still carries a real shadow configuration
/// (enabled, two cascades, PCF), because that is the plan the measurement is
/// filed under; it differs from `occluder_scene(1)`'s only in the budget.
fn shadowed_ladder_scene() -> Scene {
    occluder_scene_case(ShadowCase::budget(2 << 20))
}

/// The other half of the remembered rule, run on two plan keys.
///
/// `docs/offload.md`: "the result is remembered per plan so the calibration is
/// never paid twice for the same regression". The 256x256 leg above cannot reach
/// that half: the reference tier is about seven times slower there, so it always
/// *loses* the calibration, the device goes straight back to the hardware, and a
/// hardware frame that misses afterwards is never offloaded again at that plan.
/// This leg needs the opposite regime - the reference tier wins the calibration,
/// the settle window buys the device a return trip, and the frame that misses
/// after that return reuses the measurement - and derives its window from both
/// tiers' own measured costs to get there.
///
/// It runs that sequence on a plan key whose shadow work is real and on one
/// without, because the plan a comparison is filed under is the resolution and
/// the shadow configuration together: the rule holding on one key and not the
/// other would be a defect in exactly the dimension the memory is keyed on.
#[test]
fn the_tier_log_holds_every_change_however_the_backend_moved() {
    if !d3d11_usable() {
        return;
    }
    let size = 32u32;
    let leg = "the log leg (32x32, shadowed plan)";
    let scene = shadowed_ladder_scene();
    // This leg says something about the record only if the device actually moved
    // between backends, so that is the verdict the window is re-derived for.
    let (_, run) = ladder_leg(leg, &scene, size, |run| {
        run.last().safe_path_events >= 2
    });
    let frames = run.frames();
    let last = &frames[frames.len() - 1].1;

    // This leg moves the frames between backends, which is what the record has to
    // survive: it offloads, returns to the hardware and offloads again.
    assert!(
        last.safe_path_events >= 2,
        "{leg}: the leg did not move the device between backends ({} change(s)), so it says \
         nothing about whether the record survives one",
        last.safe_path_events
    );

    // One owner, one log: the record is the device's, and a change that happened
    // stays readable after the backend that made it is gone. Read against the
    // *live* backend's ring, an entry recorded during a calibration frame vanished
    // the moment the frames moved back to the hardware - and the entry that
    // replaced it could name a tier no entry in the ring had ever produced.
    let final_ring = ring(last);
    for (index, (_, stats)) in frames.iter().enumerate() {
        for entry in ring(stats) {
            assert!(
                final_ring.contains(&entry),
                "{leg}: frame {index} read an entry that is gone by the last frame: {entry:?}\n\
                 the log at the end holds {final_ring:#?}"
            );
        }
    }

    // "In the order they happened": the frame each change carries never goes
    // backwards, whichever layer made it.
    for pair in final_ring.windows(2) {
        assert!(
            pair[0].3 <= pair[1].3,
            "{leg}: the log is out of order: {} at frame {} is listed before {} at frame {}",
            pair[0].4,
            pair[0].3,
            pair[1].4,
            pair[1].3
        );
    }

    // The count is every change the device has made, not the entries that happen
    // to be in a ring of 16: it never goes backwards, and it covers the changes a
    // host saw happen.
    let mut previous = 0u32;
    for (index, (_, stats)) in frames.iter().enumerate() {
        assert!(
            stats.downgrade_count >= previous,
            "{leg}: frame {index} reports {} changes after {} earlier ones",
            stats.downgrade_count,
            previous
        );
        previous = stats.downgrade_count;
    }
    assert!(
        last.downgrade_count >= last.safe_path_events,
        "{leg}: {} backend change(s) but only {} entry(ies) logged",
        last.safe_path_events,
        last.downgrade_count
    );
}

#[test]
fn a_plan_is_calibrated_once_however_often_it_offloads() {
    if !d3d11_usable() {
        return;
    }
    let size = 32u32;
    for (leg, scene) in [
        ("the remembered leg (32x32, shadowed plan)", shadowed_ladder_scene()),
        ("the remembered leg (32x32, no shadows)", single_triangle_scene(0)),
    ] {
        // The verdict this leg exists for - the second offload, the one that
        // reuses the measurement - is what a window is re-derived for when this
        // host's frames move under it.
        let (window, run) = ladder_leg(leg, &scene, size, |run| run.offloads().len() >= 2);
        run.assert_settled_within_bounds();

        // The sequence, read for what the policy wrote rather than for a count.
        let offloads = run.offloads();
        let returns = run.returns();
        assert!(
            !offloads.is_empty() && offloads[0].contains("calibrating the reference tier"),
            "{leg}: the first offload of a plan is its calibration: {offloads:?}"
        );
        // The branch this leg exists for. Reaching it is the policy's to do with a
        // window the derivation above has established - a target the reference
        // tier's frames are inside and a fresh hardware device's first frame is
        // over - so a leg that does not reach it fails with everything it
        // measured, rather than passing on an unexercised path.
        assert!(
            offloads.len() >= 2,
            "{leg}: the remembered offload was never reached. With a {} ms target (the reference \
             tier's typical warm frame {} ns, a fresh hardware device's first frame {} ns) the \
             ladder changed backend {} time(s), returned to the hardware {returns} time(s) and \
             wrote {} offload(s): {offloads:?}. The window the derivation established says this \
             sequence was reachable, so either the ladder stopped short of reusing the \
             measurement or this host's frames moved under it.",
            window.target_ms,
            window.cpu_typical_ns,
            window.gpu_first_ns,
            run.last().safe_path_events,
            offloads.len()
        );
        for (index, detail) in offloads.iter().enumerate().skip(1) {
            assert!(
                !detail.contains("calibrating the reference tier"),
                "{leg}: offload {} of this plan paid for a second calibration: {detail}",
                index + 1
            );
            assert!(
                detail.contains("already measured"),
                "{leg}: offload {} of this plan does not record that it reused the measurement: \
                 {detail}",
                index + 1
            );
        }
    }
}

/// The other half of the rule: no offload without a miss. A host whose frames
/// are inside its target keeps its hardware, whatever the CPU could do.
#[test]
fn a_generous_target_never_offloads() {
    if !d3d11_usable() {
        return;
    }
    let frames = run(ladder(4, 5000, 1, abi::allow_downgrade::TIER), &occluder_scene(1));
    for (index, (_, stats)) in frames.iter().enumerate() {
        assert_eq!(stats.backend, abi::backend::D3D11, "frame {index} left the hardware");
        assert_eq!(stats.safe_path_events, 0, "frame {index} took a safe path");
        assert_eq!(stats.downgrade_count, 0, "frame {index} logged a tier change");
    }
    assert_eq!(frames[3].1.frames_presented, 4);
}

/// `RECONL_ALLOW_DOWNGRADE_TIER` is the existing opt-out, and it has to mean
/// what it says: a device whose host left it out never moves itself, however
/// badly it misses the target.
#[test]
fn a_host_that_forbids_tier_changes_keeps_its_hardware() {
    if !d3d11_usable() {
        return;
    }
    let scene = occluder_scene(1);
    for allow in [abi::allow_downgrade::NONE, abi::allow_downgrade::SHADOWS] {
        let frames = run(ladder(3, 1, 1, allow), &scene);
        for (index, (_, stats)) in frames.iter().enumerate() {
            assert_eq!(
                stats.backend,
                abi::backend::D3D11,
                "allow_downgrade {allow:#x} still moved the device off the hardware at frame {index}"
            );
            assert_eq!(stats.safe_path_events, 0, "the offload ran with the bit clear");
            assert_ne!(
                stats.tier_reason,
                reason(reconl_core::tier::TierReason::Recovery),
                "a return trip this policy made, with the host's bit clear"
            );
        }

        // The opt-out is an opt-out from *backend changes*, not from the ladder:
        // a host whose frames miss the target still gets the device's own relabel,
        // which keeps the hardware and lowerers its quality tier. That response is
        // the device's to make and the device's to record - the backend decides
        // nothing and keeps no log - so it has to be in the ring a host reads.
        let last = &frames[frames.len() - 1].1;
        assert!(
            last.downgrade_count >= 1,
            "allow_downgrade {allow:#x}: the device's frames were over the 1 ms target and it did \
             not relabel itself (tier {}, entries {})",
            last.tier,
            last.downgrade_count
        );
        assert!(
            last.downgrade_count <= last.downgrade_capacity,
            "the ring reports more entries than it can hold"
        );
        let entries = ring(last);
        let relabel = entries
            .iter()
            .rev()
            .find(|(_, _, why, _, detail)| {
                *why == reason(reconl_core::tier::TierReason::FrameTimeOverTarget)
                    && detail.contains("this device's own tier")
            })
            .unwrap_or_else(|| panic!("allow_downgrade {allow:#x}: the relabel was not logged: {entries:?}"));
        assert_eq!(
            relabel.1,
            last.tier,
            "the host reads tier {} while the last tier the device's own relabel logged is {}: \
             {entries:?}",
            last.tier,
            relabel.1
        );
    }
}

/// The ladder's schedule, which a host can see and therefore has to be pinned:
/// the device's own relabel acts on the frame *after* the cost that armed it, one
/// frame behind the offload, which acts on the frame it just closed. Both layers
/// answer from the device's one run, so a relabel is never charged to the frame
/// that armed it - the frame `ffi/src/offload.rs` answers this from, and the
/// order the backends' own ladders kept before both layers shared the device's
/// run.
///
/// The offload is opted out of, so the relabel is the only response this device
/// can make, and the arm is asserted rather than hoped for: every frame at this
/// resolution costs tens of milliseconds against the 1 ms target the ladder legs
/// use, and the test fails on the measurement if that is not true here.
#[test]
fn the_ladder_charges_a_relabel_to_the_frame_after_the_cost() {
    if !d3d11_usable() {
        return;
    }
    let frames = run(ladder(2, 1, 1, abi::allow_downgrade::NONE), &occluder_scene(1));
    let target_ns = 1_000_000;
    assert!(
        frames[0].1.frame.total_ns > target_ns,
        "this pin needs a first frame that misses the target: it cost {} ns against {target_ns} ns",
        frames[0].1.frame.total_ns
    );
    // The frame that armed the run was charged nothing: the host reads the tier
    // it created the device at, and an empty ring.
    assert_eq!(
        frames[0].1.tier, 1,
        "the first frame stepped a tier the run had not armed yet (entries {})",
        frames[0].1.downgrade_count
    );
    assert_eq!(
        frames[0].1.downgrade_count, 0,
        "a tier change was logged on the frame that armed the run rather than the one after it: {}",
        ring(&frames[0].1).len()
    );
    // The frame after it is the frame the relabel acts on, and it says so.
    let second = &frames[1].1;
    assert_eq!(
        second.downgrade_count, 1,
        "the frame after the miss did not charge the relabel it armed: {:?}",
        ring(second)
    );
    let entry = &second.downgrades[0];
    assert_eq!(
        (entry.from, entry.to, entry.frame_index),
        (1, 2, 1),
        "the relabel is not charged to the frame after the cost that armed it"
    );
    assert!(
        detail_of(entry).contains("this device's own tier"),
        "the entry does not say which layer made the change: {}",
        detail_of(entry)
    );
    assert_eq!(second.tier, 2, "the host reads the tier the relabel logged");
}

/// The trigger is the driver's verdict, not the error code. Every failure this
/// backend raises reports `RECONL_ERR_DEVICE_LOST` today - a refused descriptor
/// and a refused present included - so a device that offloaded on that code
/// would abandon a healthy GPU over a frame the host asked for wrongly. Both
/// refusals are driven here, through the ABI.
#[test]
fn a_refused_call_does_not_offload_the_device() {
    if !d3d11_usable() {
        return;
    }
    let scene = occluder_scene(1);
    for expect in [Expect::DriverRejects, Expect::PresentTooSmall] {
        // A 32768-wide target is past D3D11's 16384 limit and far below the
        // 512 MiB allocation ceiling, so only the driver refuses it.
        let settings = Run {
            width: if expect == Expect::DriverRejects { 32768 } else { W },
            height: if expect == Expect::DriverRejects { 1 } else { H },
            expect,
            ..ladder(1, 1, 1, abi::allow_downgrade::TIER)
        };
        let frames = run(settings, &scene);
        let stats = &frames[0].1;
        assert_eq!(
            stats.backend,
            abi::backend::D3D11,
            "a refused call ({expect:?}) moved the device off the hardware"
        );
        assert_eq!(stats.safe_path_events, 0, "a refused call ({expect:?}) is not a fault");
        assert_eq!(
            stats.frames_dropped, 1,
            "a refused call ({expect:?}) must still drop the frame, as before the offload existed"
        );
    }
}

/// A buffer usage word for a run's geometry: the usage bit the draw needs, plus
/// `RECONL_BUFFER_STATIC` when the run declares its geometry stationary.
fn buffer_usage(base: u32, static_geometry: bool) -> u32 {
    base | if static_geometry { abi::buffer_usage::STATIC } else { 0 }
}

/// Where a run's spill arena lives: the temp directory, named for the test that
/// asked for it and the process running it, so two test binaries cannot share
/// one arena by accident.
fn spill_path(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("reconl-tiers-{}-{name}", std::process::id()));
    path
}

/// A spill directory a test owns for as long as it runs. The arena file *is* the
/// cache, so the directory is emptied before the test and removed after it: a
/// run that inherited an arena another one left would be measuring that arena
/// rather than its own, and a cache hit in a test that never filled it is
/// exactly the false pass this exists to rule out.
struct SpillDir(&'static str);

impl SpillDir {
    fn new(name: &'static str) -> Self {
        let _ = std::fs::remove_dir_all(spill_path(name));
        Self(name)
    }

    fn name(&self) -> &'static str {
        self.0
    }
}

impl Drop for SpillDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(spill_path(self.0));
    }
}

/// The ABI's one statement about whether geometry moves is `RECONL_BUFFER_STATIC`
/// - "eligible for the disk cache arena" - and the reference tier's cascade
/// cache is keyed on nothing else. Before that bit was read, every draw reached
/// the backend as dynamic, so the cache this header advertises
/// (`RECONL_CAP_CACHED_CASCADE` alongside `RECONL_CAP_OUT_OF_CORE`) was
/// unreachable from any host, the project's own bench included: `--shadows=cached`
/// there measured `hits 0 misses 0` and `spill 0 B`.
///
/// Three legs, one scene, one budget: a device that declares its geometry static
/// and finds nothing yet, the same device again against the arena the first one
/// filled, and a control whose buffers say the geometry moves. The first leg is
/// what fills the disk cache, the second is a hit off it, and the control has no
/// cache at all - which is the declaration, not the scene, being measured. What
/// the cache serves has to be the same image the fill chose: all three legs
/// present the same pixels, in the same order.
/// What a frame's own targets cost per pixel: RGBA32F colour (16 bytes) plus f32
/// depth (4). The same number the reference backend's `target_bytes` reserves
/// with, so a test can say what a frame needs without asking the backend.
const FRAME_BYTES_PER_PIXEL: u64 = 20;

/// The frame these disk-cap tests render. Measured: at 256x256 with a 1 MiB RAM
/// cap the frame streams as 3 bands of 128 rows and needs 1,310,792 bytes on
/// disk, which is small enough to render in milliseconds and large enough that
/// nothing about the cap is incidental.
const CAP_W: u32 = 256;
const CAP_H: u32 = 256;

/// A RAM cap that holds the cascade maps and one band of the frame above, and not
/// the frame itself: what makes it stream.
const CAP_RAM: u64 = 1 << 20;

/// The bytes a `width x height` frame's own targets cost.
fn frame_target_bytes(width: u32, height: u32) -> u64 {
    u64::from(width) * u64::from(height) * FRAME_BYTES_PER_PIXEL
}

/// What the arena file for spill directory `name` measures on disk, or 0 where
/// the arena never created one.
fn arena_file_bytes(name: &str) -> u64 {
    std::fs::metadata(spill_path(name).join("arena.rcls"))
        .map(|m| m.len())
        .unwrap_or(0)
}

/// The reference backend's probe row for a host that named `spill_dir`: its
/// capability word and the best tier it claims, as `reconlProbe` reports them.
fn probe_soft_cpu(spill_dir: Option<&std::path::Path>) -> (u32, u32) {
    let dir = spill_dir.map(|p| CString::new(p.to_string_lossy().as_bytes()).expect("spill path"));
    let mut desc: abi::ReconLProbeDesc = unsafe { core::mem::zeroed() };
    desc.base = hdr::<abi::ReconLProbeDesc>();
    desc.spill_dir = dir.as_ref().map_or(core::ptr::null(), |c| c.as_ptr());
    let mut info: abi::ReconLProbeInfo = unsafe { core::mem::zeroed() };
    info.base = hdr::<abi::ReconLProbeInfo>();
    assert_eq!(
        unsafe { reconl::reconlProbe(&desc as *const _, &mut info) },
        abi::result::OK,
        "probe: {}",
        global_error()
    );
    let index = (0..info.entry_count as usize)
        .find(|&i| info.entries[i].backend == abi::backend::SOFT_CPU)
        .expect("the probe does not report the reference backend");
    (info.entries[index].caps, info.entries[index].best_tier)
}

/// The disk budget is the arena's hard cap, and this is where that has to hold: a
/// `disk_cap_bytes` declared through the ABI, a frame the RAM cap cannot hold, and
/// the bytes on disk inside the number the host declared.
///
/// Before the budget reached the arena at all - `config_from_desc` built the
/// reference backend's config without `arena_bytes`, so the arena opened with no
/// cap - the same shape of run wrote 26 MiB against a 2 MiB declaration. What this
/// asserts is exactly the thing that was missing: the cap is the cap, and the
/// device says which cap it was given.
#[test]
fn a_streamed_frame_never_writes_past_the_disk_cap_the_host_declared() {
    let scene = occluder_scene(1);
    let dir = SpillDir::new("disk-cap");
    // Room for the frame's bands and one record header each, and not much else.
    let cap = frame_target_bytes(CAP_W, CAP_H) + (16 << 10);
    let frames = run(
        Run {
            width: CAP_W,
            height: CAP_H,
            frames: 2,
            ram_cap: CAP_RAM,
            disk_cap: cap,
            spill: Some(dir.name()),
            ..Run::one(abi::backend::SOFT_CPU, 4)
        },
        &scene,
    );
    let last = &frames[frames.len() - 1].1;
    eprintln!(
        "disk cap {cap}: presented {} failed {} live {} of {} declared, arena file {} bytes",
        last.frames_presented,
        last.failures,
        last.memory.spill_disk_bytes,
        last.memory.spill_disk_cap_bytes,
        arena_file_bytes(dir.name())
    );

    assert_eq!(
        last.frames_presented as usize,
        frames.len(),
        "a frame the RAM cap cannot hold must still present, by streaming"
    );
    assert_eq!(last.failures, 0, "a streamed frame is not a failure");
    assert_eq!(
        last.memory.spill_disk_cap_bytes, cap,
        "the host's disk cap did not reach the device's budget"
    );
    assert!(
        last.memory.spill_disk_bytes > 0,
        "nothing was written to the arena, so this frame did not stream: {} live bytes, {} on \
         disk",
        last.memory.spill_disk_bytes,
        arena_file_bytes(dir.name())
    );
    assert!(
        last.memory.spill_disk_bytes <= cap && arena_file_bytes(dir.name()) <= cap,
        "the arena is over its cap: {} live bytes and {} on disk, declared {cap}",
        last.memory.spill_disk_bytes,
        arena_file_bytes(dir.name())
    );
}

/// The other half of the same guarantee: a cap the frame does not fit is a refusal
/// that names what it needed, and nothing is written past the cap on the way to it.
///
/// Measured on the shipped bench at this size, a streamed frame needs 1,310,792
/// bytes on disk; the cap here is half the frame's own targets, so no band of it fits
/// either.
#[test]
fn a_frame_the_disk_cap_cannot_hold_is_refused_naming_what_it_needed() {
    let scene = occluder_scene(1);
    let dir = SpillDir::new("disk-cap-refused");
    let cap = frame_target_bytes(CAP_W, CAP_H) / 2;
    let frames = run(
        Run {
            width: CAP_W,
            height: CAP_H,
            frames: 1,
            expect: Expect::RefusesAtBegin,
            ram_cap: CAP_RAM,
            disk_cap: cap,
            spill: Some(dir.name()),
            ..Run::one(abi::backend::SOFT_CPU, 4)
        },
        &scene,
    );
    let text = String::from_utf8_lossy(&frames[0].0).into_owned();
    eprintln!("refused under a {cap} byte cap: {text}");

    assert!(
        text.contains(&cap.to_string()),
        "the refusal does not name the cap it was given ({cap}): {text}"
    );
    // What it needed, as a number larger than what it may use: a refusal a host can
    // act on says how far over it is.
    let needed: u64 = text
        .split("needs ")
        .nth(1)
        .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|digits| digits.parse().ok())
        .unwrap_or_else(|| panic!("the refusal names no requirement: {text}"));
    assert!(
        needed > cap,
        "the refusal says it needed {needed}, which is not more than the {cap} byte cap"
    );
    assert!(
        arena_file_bytes(dir.name()) <= cap,
        "a refused frame wrote {} bytes against a {cap} byte cap - that is the defect, not the \
         refusal",
        arena_file_bytes(dir.name())
    );
}

/// The probe is how a host finds out about the disk tier before it creates
/// anything, and its answer has to match the machine: the reference backend's
/// `disk-spill`/`out-of-core` caps and its best tier follow whether a spill
/// directory can actually be used.
///
/// Answered from a hard-coded zero, every probe reported those capabilities absent
/// - even with a writable directory named - while a device created with a disk
/// budget reported both, so the same binary disagreed with itself about a
/// capability it has.
#[test]
fn the_probe_reports_the_disk_tier_only_where_an_arena_can_live() {
    let dir = SpillDir::new("probe-spill");
    let writable = spill_path(dir.name());
    std::fs::create_dir_all(&writable).expect("spill dir");
    let file = spill_path("probe-file");
    std::fs::write(&file, b"not a directory").expect("probe file");

    let (dir_caps, dir_tier) = probe_soft_cpu(Some(&writable));
    let (file_caps, file_tier) = probe_soft_cpu(Some(&file));
    eprintln!(
        "probe: directory caps {dir_caps:#010x} best tier {dir_tier}, file caps {file_caps:#010x} \
         best tier {file_tier}"
    );

    let disk = abi::caps::DISK_SPILL | abi::caps::OUT_OF_CORE;
    assert_eq!(
        dir_caps & disk,
        disk,
        "a writable spill directory has to report the disk tier's capabilities: {dir_caps:#010x}"
    );
    assert_eq!(
        dir_tier, 4,
        "and T4 as the best tier the reference backend can reach"
    );
    assert_eq!(
        file_caps & disk,
        0,
        "a path that is a file cannot hold an arena, so those caps are not real: {file_caps:#010x}"
    );
    assert_eq!(
        file_tier, 2,
        "and without a usable directory the reference backend is a T2 machine"
    );

    // The probe answered a question rather than taking the directory: nothing it
    // needed is left in there.
    let left: Vec<String> = std::fs::read_dir(&writable)
        .expect("spill dir")
        .map(|e| e.expect("dir entry").file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        left.is_empty(),
        "the probe left {left:?} in the spill directory"
    );
    let _ = std::fs::remove_file(&file);
}

#[test]
fn the_static_declaration_is_what_the_cascade_cache_is_keyed_on() {
    let scene = occluder_scene(1);
    let dir = SpillDir::new("static-cascade");
    let leg = |static_geometry: bool| Run {
        frames: 2,
        static_geometry,
        cached: true,
        spill: Some(dir.name()),
        ..Run::one(abi::backend::SOFT_CPU, 4)
    };
    let cold = run(leg(true), &scene);
    let warm = run(leg(true), &scene);
    let moving = run(leg(false), &scene);

    // The shadow counters are the device's own, cumulative over the frames it
    // has presented, so a leg is read at its last frame: what a leg did is what
    // its last snapshot has.
    let last = |frames: &[(Vec<u8>, abi::ReconLStats)]| frames[frames.len() - 1].1.clone();

    eprintln!(
        "static cache: cold hits {} misses {} frozen {}, warm hits {} misses {} frozen {} (disk {} \
         bytes, map {}x{}, tier {}), moving hits {} misses {}",
        last(&cold).shadows.cache_hits,
        last(&cold).shadows.cache_misses,
        last(&cold).shadows.frozen_cascades,
        last(&warm).shadows.cache_hits,
        last(&warm).shadows.cache_misses,
        last(&warm).shadows.frozen_cascades,
        last(&warm).memory.spill_cache_bytes,
        last(&warm).shadows.map_width,
        last(&warm).shadows.map_height,
        last(&warm).tier,
        last(&moving).shadows.cache_hits,
        last(&moving).shadows.cache_misses,
    );

    // The T4 static cascade is a disk cache, and a device with an empty arena has
    // to fill it before anything can hit it. The frames after the first are what
    // the plan freezes instead - the other half of the same plan, counted rather
    // than assumed.
    assert_eq!(
        last(&cold).shadows.cache_hits,
        0,
        "a cold device reported a cascade cache hit: there is nothing in the arena yet"
    );
    assert!(
        last(&cold).shadows.cache_misses >= 1 && last(&cold).memory.spill_cache_bytes > 0,
        "a cold device rendered the static cascade and did not put it in the arena: \
         misses {}, frozen {}, disk cache {} bytes (caps {:#x})",
        last(&cold).shadows.cache_misses,
        last(&cold).shadows.frozen_cascades,
        last(&cold).memory.spill_cache_bytes,
        last(&cold).caps
    );
    assert!(
        last(&cold).shadows.frozen_cascades >= 1,
        "the T4 plan froze no cascade between refreshes: {} frozen over {} frames",
        last(&cold).shadows.frozen_cascades,
        cold.len()
    );

    // The second device, against the same arena and the same plan, finds it.
    assert!(
        last(&warm).shadows.cache_hits >= 1,
        "a warm arena served no cascade: this device's static geometry was not cacheable, so \
         the declaration did not reach the backend (hits {}, misses {}, disk cache {} bytes)",
        last(&warm).shadows.cache_hits,
        last(&warm).shadows.cache_misses,
        last(&warm).memory.spill_cache_bytes
    );
    assert_eq!(
        last(&warm).shadows.cache_misses,
        0,
        "the warm device re-rendered a cascade the arena was holding"
    );
    assert!(
        last(&warm).shadows.cache_bytes_hit > 0,
        "the cascade cache reported a hit and no bytes: {} bytes read, {} bytes hit",
        last(&warm).shadows.cache_bytes_read,
        last(&warm).shadows.cache_bytes_hit
    );

    // The control: the same scene, the same frames, the same plan, the same
    // budget - and no cache, because its buffers say the geometry moves.
    assert_eq!(
        (
            last(&moving).shadows.cache_hits,
            last(&moving).shadows.cache_misses,
            last(&moving).memory.spill_cache_bytes
        ),
        (0, 0, 0),
        "a device whose geometry is declared dynamic used the static cascade cache anyway: \
         the cache is not keyed on the declaration"
    );

    // What the cache serves, and what the declaration decides, is the same image:
    // a cascade off the disk and a cascade rendered from scratch have to shade the
    // same pixels, or the cache is serving a stale or partial map.
    for (index, ((cold_frame, _), (warm_frame, _))) in cold.iter().zip(warm.iter()).enumerate() {
        let (worst, differing, pixels) = delta_report(cold_frame, warm_frame);
        assert!(
            cold_frame == warm_frame,
            "frame {index} presented from the cache differs from the frame it was cached from: \
             {differing} of {pixels} pixels, worst channel delta {worst}"
        );
    }
    for (index, ((cold_frame, _), (moving_frame, _))) in cold.iter().zip(moving.iter()).enumerate() {
        let (worst, differing, pixels) = delta_report(cold_frame, moving_frame);
        assert!(
            cold_frame == moving_frame,
            "frame {index} differs between static and moving geometry: {differing} of {pixels} \
             pixels, worst channel delta {worst} - the declaration changed the image, not just \
             what is cached"
        );
    }
}

/// The classification contract: a driver failure that is not a removal no
/// longer carries the blanket `RECONL_ERR_DEVICE_LOST` a host would destroy a
/// healthy device over. The cheapest driver-reachable refusal is a target past
/// D3D11's 16384-texel limit - `CreateTexture2D` answers `E_INVALIDARG` - so
/// the host gets the argument code the header's taxonomy names, with the
/// device still usable. This test fails under the old blanket mapping (it saw
/// `-6`); `a_refused_call_does_not_offload_the_device` above pins the
/// behavioural half of the same contract.
#[test]
fn a_driver_rejected_frame_reports_an_argument_error_not_a_loss() {
    if !d3d11_usable() {
        return;
    }
    let frames = run(
        Run {
            width: 32768,
            height: 1,
            expect: Expect::DriverRejects,
            ..ladder(1, 1, 1, abi::allow_downgrade::TIER)
        },
        &occluder_scene(1),
    );
    let stats = &frames[0].1;
    assert_eq!(
        stats.failures, 1,
        "the refused submit was not counted as a failure"
    );
    assert_eq!(
        stats.last_result,
        abi::result::INVALID_ARGUMENT,
        "a driver-rejected frame must surface the argument code the header \
         documents for it, not {}",
        stats.last_result
    );
}
