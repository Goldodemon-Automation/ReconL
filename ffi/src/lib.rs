//! `reconl`: the C ABI surface. Everything a host can call lives here, and
//! nothing that a host can call lives anywhere else.
//!
//! Structure of the crate, in the order a call flows through it:
//!
//! 1. [`abi`] - the `#[repr(C)]` mirror of `include/reconl/reconl.h`.
//! 2. `sizing` - what a descriptor costs, from the descriptor alone. Pure
//!    arithmetic, so the price of a request is decided and tested before any
//!    budget or allocator is involved.
//! 3. handles - every object is a ref-counted, host-allocated block that begins
//!    with [`HandleHeader`]. `reconlRetain`/`reconlRelease` work on `void*` and
//!    read that header, so any handle can be passed to them and a handle of the
//!    wrong kind is caught rather than misread.
//! 4. the device - owns the budget, the tier, the stats, and exactly one of the
//!    two milestone-1 backends.
//! 5. entry points - thin: validate, translate, delegate, record. No policy of
//!    their own, because policy that lives in the binding surface cannot be
//!    tested from Rust: they ask `sizing` what a request costs and `Budget`
//!    whether it is affordable, then hand it to the backend.
//!
//! ## Panics
//!
//! The release profile is `panic = "abort"`, so there is nothing to unwind
//! across the boundary in a shipped build. In a `panic = "unwind"` build (tests,
//! and `cargo run` examples) every entry point is wrapped in `catch_unwind` and
//! converts a panic into `RECONL_ERR_PANIC` with the safe path taken and
//! counted - so a bug in the renderer is a return code, not a crashed host.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

pub mod abi;
mod sizing;

use abi::*;
use reconl_backend_d3d11::{D3d11Config, D3d11Device};
use reconl_backend_null::{NullConfig, NullDevice};
use reconl_backend_softcpu::{FrameInput, SoftCpuConfig, SoftCpuDevice};
use reconl_core::alloc::{host_stats, HostAlloc, HostAllocatorA, HostVec};
use reconl_core::budget::{Budget, BudgetCaps, Reservation};
use reconl_core::error::{Code, Error, Result};
use reconl_core::log::{self, Level};
use reconl_core::{err, log_info, log_warn};
use reconl_core::stats::{Counters, Downgrade, DowngradeLog, FrameNumbers, ShadowCounters, Stats};
use reconl_core::tier::{shadow_plan, ShadowFilter, Tier, TierReason, TIER_COUNT};
use reconl_core::{check_header, Text};

// Re-exported for hosts and tools that build ABI structs: `ABIStruct` carries
// the `struct_size`/type constants `check_header` validates against.
pub use reconl_core::{ABIStruct, StructHeader};
use reconl_raster::math::{self, Mat4, Vec3, IDENTITY};
use reconl_raster::shade::{Light, LightSet, SurfaceShader};
use reconl_raster::{DrawItem, PipelineState, ShaderRef, Vertex, COMPARE_GREATER, CULL_BACK, MAX_TEXTURE_SLOTS};
use reconl_resource::mips::generate_mip_chain;
use reconl_scene::WorldRevision;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

// ------------------------------------------------------------------ handles

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Device = 1,
    Buffer = 2,
    Texture = 3,
    Pipeline = 4,
    Swapchain = 5,
    CommandList = 6,
    Fence = 7,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Device => "device",
            Kind::Buffer => "buffer",
            Kind::Texture => "texture",
            Kind::Pipeline => "pipeline",
            Kind::Swapchain => "swapchain",
            Kind::CommandList => "command list",
            Kind::Fence => "fence",
        }
    }
}

/// First member of every handle. `void*`-typed ABI calls (`reconlRetain`,
/// `reconlRelease`) read this to find out what they were handed.
#[repr(C)]
pub struct HandleHeader {
    pub kind: u32,
    /// 1 for a live handle. The device holds one implicit reference per live
    /// child, so a child outliving the host's device reference keeps the device
    /// alive rather than dangling.
    pub refcount: AtomicU32,
    pub device: *mut DeviceHandle,
}

/// Anything with a [`HandleHeader`] first.
pub unsafe trait Handle {
    fn header(&self) -> &HandleHeader;
    fn kind() -> Kind;
}

macro_rules! impl_handle {
    ($t:ty, $kind:expr) => {
        unsafe impl Handle for $t {
            fn header(&self) -> &HandleHeader {
                &self.header
            }
            fn kind() -> Kind {
                $kind
            }
        }
    };
}

/// Allocates a handle with the host allocator. ReconL never uses the Rust global
/// allocator for objects the host owns.
unsafe fn handle_new<T: Handle>(alloc: HostAlloc, value: T) -> *mut T {
    match unsafe { alloc.alloc_bytes(core::mem::size_of::<T>(), core::mem::align_of::<T>().max(16)) } {
        Ok(p) => {
            let ptr = p.as_ptr() as *mut T;
            unsafe { ptr.write(value) };
            ptr
        }
        Err(_) => core::ptr::null_mut(),
    }
}

unsafe fn handle_free<T: Handle>(alloc: HostAlloc, ptr: *mut T) {
    if ptr.is_null() {
        return;
    }
    let bytes = core::mem::size_of::<T>();
    unsafe { core::ptr::drop_in_place(ptr) };
    let nonnull = unsafe { core::ptr::NonNull::new_unchecked(ptr as *mut u8) };
    unsafe { alloc.free_bytes(nonnull, bytes, core::mem::align_of::<T>().max(16)) };
}

/// Validates a handle pointer of a known kind and returns its header.
unsafe fn check_handle(ptr: *mut c_void, expected: Kind, what: &str) -> Result<*mut HandleHeader> {
    if ptr.is_null() {
        return err!(Code::InvalidArgument, "null {} handle", what);
    }
    let header = ptr as *mut HandleHeader;
    // SAFETY: every handle this library hands out begins with a HandleHeader.
    let kind = unsafe { (*header).kind };
    if kind != expected as u32 {
        return err!(
            Code::InvalidHandle,
            "{} handle was passed where a {} was expected",
            Kind::from_u32(kind).name(),
            what
        );
    }
    Ok(header)
}

impl Kind {
    fn from_u32(v: u32) -> Kind {
        match v {
            1 => Kind::Device,
            2 => Kind::Buffer,
            3 => Kind::Texture,
            4 => Kind::Pipeline,
            5 => Kind::Swapchain,
            6 => Kind::CommandList,
            7 => Kind::Fence,
            _ => Kind::Device,
        }
    }
}

// -------------------------------------------------------------- child handles

#[repr(C)]
pub struct BufferHandle {
    header: HandleHeader,
    bytes: HostVec<u8>,
    /// Held for the buffer's lifetime, so its bytes stay counted against the
    /// device budget until the host releases it.
    ram: Reservation,
    usage: u32,
    name: Text<64>,
}
impl_handle!(BufferHandle, Kind::Buffer);

#[repr(C)]
pub struct TextureHandle {
    header: HandleHeader,
    width: u32,
    height: u32,
    format: u32,
    usage: u32,
    /// RGBA8 mip chain, tightly packed.
    levels: HostVec<HostVec<u8>>,
    level_sizes: HostVec<(u32, u32)>,
    /// The whole chain's bytes, counted against the device budget for as long
    /// as the texture lives.
    ram: Reservation,
    name: Text<64>,
}
impl_handle!(TextureHandle, Kind::Texture);

impl TextureHandle {
    fn mip_count(&self) -> u32 {
        self.levels.len() as u32
    }

    fn level_bytes(&self) -> u64 {
        let mut total = 0u64;
        for level in self.levels.iter() {
            total += level.len() as u64;
        }
        total
    }
}

#[repr(C)]
pub struct PipelineHandle {
    header: HandleHeader,
    shading: u32,
    blend: u32,
    cull: u32,
    depth_compare: u32,
    depth_write: bool,
    texture_slots: u32,
    receives_shadow: bool,
    casts_shadow: bool,
    two_sided_shadow: bool,
    name: Text<64>,
}
impl_handle!(PipelineHandle, Kind::Pipeline);

#[repr(C)]
pub struct SwapchainHandle {
    header: HandleHeader,
    width: u32,
    height: u32,
    format: u32,
    present_to_memory: bool,
    depth: bool,
}
impl_handle!(SwapchainHandle, Kind::Swapchain);

#[repr(C)]
pub struct FenceHandle {
    header: HandleHeader,
    signaled: bool,
    frame_index: u64,
}
impl_handle!(FenceHandle, Kind::Fence);

// ------------------------------------------------------------------ commands

#[derive(Clone, Copy)]
enum Command {
    BeginPass {
        target: *mut TextureHandle,
        load_color: bool,
        load_depth: bool,
        clear_color: [f32; 4],
        clear_depth: f32,
        width: u32,
        height: u32,
    },
    EndPass,
    SetPipeline {
        pipeline: *mut PipelineHandle,
    },
    SetVertexBuffer {
        stream: u32,
        buffer: *mut BufferHandle,
        offset: u64,
    },
    SetIndexBuffer {
        buffer: *mut BufferHandle,
        offset: u64,
        format: u32,
    },
    SetTexture {
        slot: u32,
        texture: *mut TextureHandle,
        filter: u32,
        wrap_u: u32,
        wrap_v: u32,
    },
    PushConstants {
        slot: u32,
        data: [u8; 64],
        size: u32,
    },
    Draw {
        vertex_count: u32,
        first_vertex: u32,
    },
    DrawIndexed {
        index_count: u32,
        first_index: u32,
        vertex_offset: i32,
    },
}

#[repr(C)]
pub struct CommandListHandle {
    header: HandleHeader,
    commands: HostVec<Command>,
    capacity_bytes: u32,
    /// Bytes recorded since the last reset, so the host can see the budget it
    /// asked for being respected (`reconlCmdCount` is not enough on its own).
    bytes: u32,
    recording: bool,
    /// The slot array, counted against the device budget.
    ram: Reservation,
    name: Text<64>,
}
impl_handle!(CommandListHandle, Kind::CommandList);

/// The documented push-constant slots (docs/abi.md).
pub const PUSH_CONSTANT_VIEW_PROJ: u32 = 0;
pub const PUSH_CONSTANT_MODEL: u32 = 1;

// -------------------------------------------------------------------- devices

enum BackendKind {
    SoftCpu(Box<SoftCpuDevice>),
    D3d11(Box<D3d11Device>),
    Null(Box<NullDevice>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FrameState {
    Idle,
    Open,
    Submitted,
}

/// A draw the frame can be re-created from.
///
/// The raw pointers are into buffers the device owns; a buffer cannot be
/// destroyed while its device lives, and the audit re-render happens inside the
/// device, so they stay valid for as long as this record does.
#[derive(Clone, Copy)]
struct DrawRecord {
    vertices: *const Vertex,
    vertex_count: u32,
    indices: *const u32,
    index_count: u32,
    transform: Mat4,
    model: Mat4,
    pipeline: PipelineState,
    shading: u32,
    textured: bool,
    lit: bool,
    receives_shadow: bool,
    dynamic: bool,
    casts_shadow: bool,
}

struct FrameRecord {
    index: u64,
    width: u32,
    height: u32,
    camera_view: Mat4,
    fov_y_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
    light_dir: Vec3,
    light_hash: u64,
    lights: LightSet,
    shadow: reconl_backend_softcpu::ShadowRequest,
    clear_color: [f32; 4],
    clear_depth: f32,
    clear_color_on: bool,
    clear_depth_on: bool,
    draws: HostVec<DrawRecord>,
    checksum: u64,
    depth_checksum: u64,
    triangles: u64,
    /// World revisions captured at `BeginFrame`, so the cascade cache key the
    /// backend sees is the one the scene layer published for this frame.
    world_revision: u64,
    static_revision: u64,
}

impl FrameRecord {
    /// A frame record with no content, backed by the host allocator.
    ///
    /// The allocator is a parameter rather than a `Default` field because a
    /// default that invents one is how a draw list ends up quietly using the
    /// zeroed allocator, which refuses every allocation and leaves the frame
    /// empty without an error.
    fn new(alloc: HostAlloc) -> Self {
        Self {
            index: 0,
            width: 0,
            height: 0,
            camera_view: IDENTITY,
            fov_y_deg: 60.0,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
            light_dir: [0.0, -1.0, 0.0],
            light_hash: 0,
            lights: LightSet::new(),
            shadow: reconl_backend_softcpu::ShadowRequest::default(),
            clear_color: [0.0, 0.0, 0.0, 1.0],
            clear_depth: 0.0,
            clear_color_on: true,
            clear_depth_on: true,
            draws: HostVec::new(alloc),
            checksum: 0,
            depth_checksum: 0,
            triangles: 0,
            world_revision: 1,
            static_revision: 1,
        }
    }
}

#[repr(C)]
pub struct DeviceHandle {
    header: HandleHeader,
    alloc: HostAlloc,
    budget: Arc<Budget>,
    backend: BackendKind,
    tier: Tier,
    tier_reason: TierReason,
    tier_locked: bool,
    caps: u32,
    device_name: Text<64>,
    driver: Text<64>,
    stats: Stats,
    last_error: Option<Error>,
    frame_state: FrameState,
    frame: FrameRecord,
    shadow: reconl_backend_softcpu::ShadowRequest,
    null_downgrades: DowngradeLog,
    audit_every_frames: u32,
    world: WorldRevision,
    spill_dir: Option<PathBuf>,
    arena_dir_note: Text<192>,
    /// What a backend rebuild needs, and the state of the offload policy
    /// (docs/offload.md).
    origin: Origin,
    offload: Offload,
    /// The device's own record of backend changes. A backend's downgrade log
    /// cannot hold these: the backend that recorded one may no longer exist.
    offload_log: DowngradeLog,
    /// `desc.allow_downgrade`, which is what makes `RECONL_DOWNGRADE_TIER` the
    /// host's opt-out from the offload.
    allow_downgrade: u32,
}
impl_handle!(DeviceHandle, Kind::Device);

/// What building this device's backends needs, kept for the device's life.
///
/// An offload is a change of backend, and changing a backend needs the config
/// the original was built from - by then the host's descriptor is long gone. A
/// device created on the reference tier carries no hardware config and never
/// offloads, because it is already where an offload would put it.
struct Origin {
    /// The hardware config, or `None` for a device the host created on the
    /// reference tier. It also carries the tier a return trip goes back to,
    /// which is the tier the device was *built* at: the ladder's own descents
    /// land in the backend, and a rebuild goes back to the start.
    gpu: Option<D3d11Config>,
    cpu: SoftCpuConfig,
}

/// What a measured tier comparison is filed under.
///
/// A comparison is only valid for the frame it was measured on: resolution and
/// the shadow plan (which is per frame, from `ReconLShadowConfig`) both change
/// what a tier costs. Filing the result under both is what makes it expire when
/// either changes, with no invalidation logic to get wrong.
type PlanKey = (u32, u32, u32, u32, u64);

fn plan_key(width: u32, height: u32, shadow: &reconl_backend_softcpu::ShadowRequest) -> PlanKey {
    (width, height, shadow.cascades, shadow.filter as u32, shadow.texel_budget_bytes)
}

/// The offload state (docs/offload.md): measured, then remembered.
///
/// Nothing here is a guess. The comparison that decides whether an overload is
/// worth offloading is the reference tier's own measured frame cost against the
/// hardware's, at one plan, taken from the first frame the reference tier
/// rendered - which is why the calibration costs no extra frame.
struct Offload {
    /// Where the reference tier's cost was measured, and whether it lost there.
    /// `measured_key` is `None` until a calibration has run, so an unmeasured
    /// device never trades its hardware for the CPU. The cost itself is in the
    /// downgrade entry's detail, which is where a host reads it.
    measured_key: Option<PlanKey>,
    /// The reference tier measured slower than the hardware at `measured_key`.
    /// The question is settled there and the offload is not attempted again.
    cpu_lost: bool,
    /// A return trip to the hardware was made at this key. One per key: a
    /// workload that oscillates must not thrash between tiers, and "the hardware
    /// was tried again and missed" is what "this frame does not fit the GPU"
    /// means. It also means the CPU's cost here is known, so a second offload at
    /// this key needs no calibration.
    returned_key: Option<PlanKey>,
    /// Consecutive frames over the hardware's target, and inside it while
    /// offloaded. One threshold drives the ladder in both directions.
    over_target: u32,
    within_target: u32,
    /// The first offloaded frame is also the calibration, so its cost is
    /// compared before the decision to stay stands.
    calibrating: bool,
    /// The hardware cost the calibration is compared against.
    gpu_ns: u64,
    /// Hardware faults this device has seen. A rebuilt backend that faults too
    /// is not returned to, so a broken driver cannot start a rebuild loop.
    faults: u32,
}

impl Offload {
    const fn new() -> Self {
        Self {
            measured_key: None,
            cpu_lost: false,
            returned_key: None,
            over_target: 0,
            within_target: 0,
            calibrating: false,
            gpu_ns: 0,
            faults: 0,
        }
    }
}

/// The process-wide last error, used when `reconlGetLastError(NULL, ..)` is
/// called. Global mutable state is otherwise forbidden; this is the one the
/// ABI needs, and it is mutex-guarded rather than lock-free on purpose.
fn global_error() -> &'static Mutex<Option<Error>> {
    static CELL: OnceLock<Mutex<Option<Error>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

fn record_error(device: *mut DeviceHandle, err: Error) -> i32 {
    let code = err.code.as_i32();
    if !device.is_null() {
        // SAFETY: the caller validated the device handle for this call.
        unsafe {
            let state = &mut (*device);
            state.stats.counters.failures += 1;
            state.stats.counters.last_result = Some(err.code);
            state.last_error = Some(err);
        }
    } else if let Ok(mut slot) = global_error().lock() {
        *slot = Some(err);
    }
    code
}

impl DeviceHandle {
    fn backend_id(&self) -> u32 {
        match &self.backend {
            BackendKind::SoftCpu(_) => backend::SOFT_CPU,
            BackendKind::D3d11(_) => backend::D3D11,
            BackendKind::Null(_) => backend::NULL,
        }
    }

    fn shadow_counters(&self) -> ShadowCounters {
        match &self.backend {
            BackendKind::SoftCpu(d) => d.shadows(),
            BackendKind::D3d11(d) => d.shadows(),
            BackendKind::Null(d) => d.stats_snapshot().1,
        }
    }

    fn frame_numbers(&self) -> FrameNumbers {
        match &self.backend {
            BackendKind::SoftCpu(d) => d.snapshot().frame,
            BackendKind::D3d11(d) => d.snapshot().frame,
            BackendKind::Null(d) => d.stats_snapshot().2,
        }
    }

    fn softcpu(&self) -> Option<&SoftCpuDevice> {
        match &self.backend {
            BackendKind::SoftCpu(d) => Some(d),
            BackendKind::D3d11(_) | BackendKind::Null(_) => None,
        }
    }

    fn softcpu_mut(&mut self) -> Option<&mut SoftCpuDevice> {
        match &mut self.backend {
            BackendKind::SoftCpu(d) => Some(d),
            BackendKind::D3d11(_) | BackendKind::Null(_) => None,
        }
    }

    /// Every downgrade the device knows about: the ones it made itself (a
    /// backend change) followed by the current backend's own ladder.
    fn downgrade_entries(&self) -> Vec<Downgrade> {
        let mut entries: Vec<Downgrade> = self.offload_log.iter().copied().collect();
        entries.extend(match &self.backend {
            BackendKind::SoftCpu(d) => d.downgrades().iter().copied().collect::<Vec<_>>(),
            BackendKind::D3d11(d) => d.downgrades().iter().copied().collect::<Vec<_>>(),
            BackendKind::Null(_) => self.null_downgrades.iter().copied().collect::<Vec<_>>(),
        });
        entries
    }

    /// Whether this device may continue its frames on the reference tier.
    ///
    /// Three things have to hold: the host left `RECONL_DOWNGRADE_TIER` set, the
    /// device was built with a hardware backend, and the hardware config to come
    /// back to exists.
    fn may_offload(&self) -> bool {
        (self.allow_downgrade & allow_downgrade::TIER) != 0
            && self.origin.gpu.is_some()
            && self.backend_id() == backend::D3D11
    }

    /// Whether this device is currently rendering on the reference tier because
    /// it offloaded there, as opposed to having been created there.
    fn offloaded(&self) -> bool {
        self.origin.gpu.is_some() && self.backend_id() == backend::SOFT_CPU
    }

    /// Builds the reference-tier backend an offload renders on.
    fn build_cpu_backend(&self) -> Result<SoftCpuDevice> {
        let mut config = self.origin.cpu.clone();
        // The ladder's own step: a hardware tier lands on T2, because a GPU that
        // cannot serve a frame cannot serve a higher CPU tier either.
        config.tier = if self.tier >= Tier::CpuRam { self.tier } else { Tier::CpuRam };
        SoftCpuDevice::new(self.alloc, Arc::clone(&self.budget), config)
    }

    /// Rebuilds the hardware backend a return trip goes back to.
    fn build_gpu_backend(&self) -> Result<D3d11Device> {
        let config = self.origin.gpu.clone().ok_or_else(|| {
            Error::new(Code::BackendUnavailable, "this device has no hardware backend to return to")
        })?;
        D3d11Device::new(self.alloc, Arc::clone(&self.budget), config)
    }

    /// Adopts `backend` as the renderer, with the bookkeeping a host reads: the
    /// tier, the caps, the names, the counter that says the safe path was taken,
    /// and one downgrade entry carrying the reason.
    fn adopt_backend(&mut self, backend: BackendKind, reason: TierReason, frame_index: u64, detail: &str) {
        let from = self.tier;
        let (to, caps, name, driver, id) = match &backend {
            BackendKind::SoftCpu(d) => (d.tier(), d.caps(), d.device_name(), d.driver(), backend::SOFT_CPU),
            BackendKind::D3d11(d) => (d.tier(), d.caps(), d.device_name(), d.driver(), backend::D3D11),
            BackendKind::Null(d) => (d.tier(), d.caps(), d.device_name(), d.driver(), backend::NULL),
        };
        self.backend = backend;
        self.tier = to;
        self.tier_reason = reason;
        self.caps = caps;
        self.device_name.set(name);
        self.driver.set(driver);
        self.stats.backend = id;
        self.stats.caps = caps;
        self.stats.tier = to;
        self.stats.tier_reason = reason;
        self.stats.tier_reason_text.set(reason.text());
        self.stats.device_name.set(name);
        self.stats.counters.safe_path_events += 1;
        self.stats.counters.frames_since_tier_change = 0;
        self.offload_log.record(Downgrade::new(from, to, reason, frame_index, 0, detail));
        log_warn!(
            "now rendering on {} at tier {} ({}): {}",
            reconl_core::tier::Backend::from_u32(id).name(),
            to.name(),
            reason.text(),
            detail
        );
    }

    /// Hands the frames to the reference tier after a hardware fault.
    ///
    /// No comparison here: a device that has been removed is not a slow device,
    /// so the reference tier is faster by definition and the calibration would
    /// be measuring nothing.
    fn offload_on_fault(&mut self, frame_index: u64, detail: &str) -> Result<()> {
        let cpu = self.build_cpu_backend()?;
        self.offload.faults += 1;
        self.offload.calibrating = false;
        self.offload.over_target = 0;
        self.offload.within_target = 0;
        self.adopt_backend(
            BackendKind::SoftCpu(Box::new(cpu)),
            TierReason::DeviceRemoved,
            frame_index,
            detail,
        );
        Ok(())
    }

    /// Offloads because the hardware missed its target, and - unless the
    /// comparison is already known - marks the next frame as the calibration that
    /// decides whether the offload stands.
    fn offload_on_overload(&mut self, frame_index: u64, detail: &str, calibrate: bool) -> Result<()> {
        let cpu = self.build_cpu_backend()?;
        self.offload.calibrating = calibrate;
        self.offload.over_target = 0;
        self.offload.within_target = 0;
        self.adopt_backend(
            BackendKind::SoftCpu(Box::new(cpu)),
            TierReason::FrameTimeOverTarget,
            frame_index,
            detail,
        );
        Ok(())
    }

    /// Rebuilds the hardware backend after a settle window inside the target.
    fn return_to_gpu(&mut self, frame_index: u64, detail: &str) -> Result<()> {
        let gpu = self.build_gpu_backend()?;
        self.offload.over_target = 0;
        self.offload.within_target = 0;
        self.adopt_backend(
            BackendKind::D3d11(Box::new(gpu)),
            // Up the ladder, which is what the reason says: a host reading the
            // log can tell a recovery from how the device started.
            TierReason::Recovery,
            frame_index,
            detail,
        );
        Ok(())
    }

    /// The measured policy, run after every frame that rendered: decides what
    /// the *next* frame renders on.
    ///
    /// Cheap in the steady state - a comparison and two counters - and the only
    /// place the device changes its own backend, so a host reading the stats sees
    /// one decision rather than several half-applied ones. Every rule here is a
    /// rule from docs/offload.md; the arithmetic is the threshold the ladder
    /// already uses, which is why the two cannot disagree.
    fn apply_offload_policy(&mut self, frame_index: u64, width: u32, height: u32) -> Result<()> {
        let target_ms = self.origin.cpu.target_frame_ms;
        let target_ns = u64::from(target_ms) * 1_000_000;
        let threshold = self.origin.cpu.over_target_frames_to_downgrade;
        let total_ns = self.frame_numbers().total_ns;
        let key = plan_key(width, height, &self.frame.shadow);

        if self.backend_id() == backend::D3D11 {
            // Measured overload. The target being zero means the ladder is off,
            // which is a host's way of saying "do not decide for me".
            self.offload.over_target = if target_ns > 0 && total_ns > target_ns {
                self.offload.over_target.saturating_add(1)
            } else {
                0
            };
            if threshold == 0
                || self.offload.over_target < threshold
                || self.offload.faults > 0
                || !self.may_offload()
            {
                return Ok(());
            }
            if self.offload.measured_key == Some(key) {
                if self.offload.cpu_lost {
                    // Measured here: the CPU renders this frame slower, so
                    // handing it frames would make every frame slower.
                    return Ok(());
                }
                // Measured here and the CPU won. The hardware was tried again
                // after a return trip and missed, so this frame genuinely does
                // not fit it - and the calibration's answer is already known.
                let detail = format!(
                    "{} frames over the {} ms target at {}x{}, where the reference tier already measured faster",
                    self.offload.over_target, target_ms, width, height
                );
                self.offload_on_overload(frame_index, &detail, false)?;
                return Ok(());
            }
            // Not measured here: one frame on the CPU is the price of not making
            // every subsequent frame slow.
            self.offload.gpu_ns = total_ns;
            // The measurement the calibration is compared against, written into
            // the entry the miss caused: a host (and the test that pins this)
            // can then read why the device came back, rather than guessing.
            let detail = format!(
                "{} frames over the {} ms target at {}x{}; the hardware measured {} ns; calibrating the reference tier",
                self.offload.over_target, target_ms, width, height, total_ns
            );
            self.offload_on_overload(frame_index, &detail, true)?;
            return Ok(());
        }

        if !self.offloaded() {
            // A device created on the reference tier has nowhere to go.
            return Ok(());
        }

        if self.offload.calibrating {
            // The first offloaded frame is the calibration: the reference tier's
            // own cost, measured, against the hardware's - no extra frame spent.
            self.offload.calibrating = false;
            self.offload.measured_key = Some(key);
            if target_ns > 0 && self.offload.gpu_ns > 0 && total_ns >= self.offload.gpu_ns {
                self.offload.cpu_lost = true;
                let detail = format!(
                    "the reference tier measured {} against the hardware's {} at {}x{}, so the offload was not a win",
                    total_ns, self.offload.gpu_ns, width, height
                );
                self.return_to_gpu(frame_index, &detail)?;
            }
            return Ok(());
        }

        // The return trip. It needs a target to be inside and a settle window of
        // frames inside it, and it is attempted at most once per plan: a workload
        // that oscillates around the target must not thrash between tiers. A
        // device that offloaded because of a fault rather than a miss therefore
        // stays offloaded on a host that set no frame-time target, which is the
        // only host that cannot say whether the hardware has recovered.
        if target_ns == 0 || threshold == 0 || self.offload.faults > 1 || self.offload.returned_key == Some(key) {
            return Ok(());
        }
        self.offload.within_target = if total_ns <= target_ns {
            self.offload.within_target.saturating_add(1)
        } else {
            0
        };
        if self.offload.within_target >= threshold {
            self.offload.returned_key = Some(key);
            let detail = format!(
                "{} frames inside the {} ms target: rebuilding the hardware backend",
                self.offload.within_target, target_ms
            );
            self.return_to_gpu(frame_index, &detail)?;
        }
        Ok(())
    }

    /// Renders `frame` on the reference backend the device is now running.
    ///
    /// The draw records in `frame` point into the host's own buffer blocks -
    /// that is why `FrameRecord` keeps them past Submit - so the reference tier
    /// can reproduce a frame the hardware did not finish, exactly. That is what
    /// makes an offloaded frame byte-identical to the same frame rendered on the
    /// reference tier directly.
    fn render_frame_on_soft(&mut self, frame: &mut FrameRecord) -> Result<()> {
        let draw_items = build_draws(self.alloc, frame.draws.as_slice())?;
        let input = frame_input(frame, draw_items.as_slice());
        let (color, depth, shadows) = {
            let soft = self.softcpu_mut().ok_or_else(|| {
                Error::new(Code::BackendUnavailable, "the reference backend is not available")
            })?;
            soft.prepare_frame(frame.width, frame.height)?;
            soft.render(&input)?;
            (soft.color_checksum(), soft.depth_checksum(), soft.snapshot().shadows)
        };
        frame.checksum = color;
        frame.depth_checksum = depth;
        self.stats.shadows = shadows;
        Ok(())
    }

    /// A hardware fault during a frame: continue on the reference tier and
    /// render the frame that faulted there, out of the frame the device still
    /// holds.
    ///
    /// Used by Present, where the frame was already rendered on the GPU and only
    /// its readback failed; Submit renders its own frame, which it is still
    /// holding, so it calls the two steps directly.
    fn recover_frame(&mut self, fault: &Error) -> Result<()> {
        let mut frame = std::mem::replace(&mut self.frame, FrameRecord::new(self.alloc));
        let detail = format!("frame {}: {}", frame.index, fault.message.as_str());
        let result = match self.offload_on_fault(frame.index, &detail) {
            Ok(()) => self.render_frame_on_soft(&mut frame),
            Err(e) => Err(e),
        };
        self.frame = frame;
        result
    }

    /// The present path of a device that just lost its hardware: the frame is
    /// re-rendered on the reference tier and the host gets the pixels it came
    /// for, in the layout it asked for.
    fn present_after_fault(
        &mut self,
        fault: &Error,
        pixels: Option<&mut [u8]>,
        out_size: u64,
        out_pitch: u32,
        flip: u32,
    ) -> Result<()> {
        self.recover_frame(fault)?;
        let (width, height) = self
            .softcpu()
            .map(|soft| soft.frame_size())
            .ok_or_else(|| Error::new(Code::BackendUnavailable, "the reference backend is not available"))?;
        if let Some(pixels) = pixels {
            let pitch = if out_pitch == 0 { width * 4 } else { out_pitch };
            let needed = (height.saturating_sub(1) as u64) * pitch as u64 + width as u64 * 4;
            if out_size < needed {
                return err!(Code::InvalidArgument, "the present buffer is too small for {}x{}", width, height);
            }
            let mut tight = vec![0u8; (width * height * 4) as usize];
            if let Some(soft) = self.softcpu_mut() {
                soft.readback(&mut tight)?;
            }
            blit(pixels, &tight, width, height, pitch, flip);
        }
        if let Some(soft) = self.softcpu_mut() {
            soft.on_frame_end()?;
        }
        Ok(())
    }

    fn record_downgrade(&mut self, from: Tier, to: Tier, reason: TierReason, frame_index: u64, detail: &str) {
        match &mut self.backend {
            BackendKind::SoftCpu(soft) => {
                soft.step_down(reason, frame_index, detail);
            }
            BackendKind::D3d11(gpu) => {
                gpu.step_down(reason, frame_index, detail);
            }
            BackendKind::Null(_) => {
                self.null_downgrades.record(Downgrade::new(from, to, reason, frame_index, 0, detail));
            }
        }
    }

}

// ---------------------------------------------------------------- entry glue

/// Drops the open frame if the call that owns it fails.
///
/// A device left in `Open` or `Submitted` rejects every later `BeginFrame` with
/// `FrameInProgress`, and there is no ABI call that abandons a frame, so without
/// this a host that records one bad draw - or whose present cannot be delivered -
/// can never render again. Both `Submit` and `Present` take one of these once the
/// call has committed to the frame and mark it `keep` only on success, so every
/// failure in between ends the frame instead of stranding it. The counted drop is
/// what `ReconLStats.frames_dropped` is for.
///
/// Holds the device as a raw pointer on purpose: the caller keeps using its own
/// `&mut`, and the two borrows are disjoint by construction (this one is only
/// read in `drop`, after the caller's last use).
struct FrameOwner {
    device: *mut DeviceHandle,
    /// Set once the frame reached a state worth keeping.
    keep: bool,
}

impl FrameOwner {
    fn new(device: *mut DeviceHandle) -> Self {
        Self { device, keep: false }
    }
}

impl Drop for FrameOwner {
    fn drop(&mut self) {
        if self.keep {
            return;
        }
        // SAFETY: the device outlives the entry point that created this.
        let device = unsafe { &mut *self.device };
        device.frame_state = FrameState::Idle;
        device.stats.counters.frames_dropped += 1;
    }
}

/// Runs an entry point body, converting a `Result` into an ABI code and
/// recording the error where the host will look for it.
pub fn run_entry(device: *mut DeviceHandle, body: impl FnOnce() -> Result<()>) -> i32 {
    match body() {
        Ok(()) => result::OK,
        Err(err) => record_error(device, err),
    }
}

/// An entry point with a panic boundary in `panic = "unwind"` builds. In the
/// shipped profile (`panic = "abort"`) a panic aborts the process, which is the
/// documented contract for `panic` across FFI.
#[cfg(panic = "unwind")]
pub fn guarded_entry(device: *mut DeviceHandle, body: impl FnOnce() -> Result<()>) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(r) => run_entry(device, || r),
        Err(_) => {
            record_error(device, Error::new(Code::Panic, "a ReconL invariant broke; the safe path was taken"))
        }
    }
}

#[cfg(not(panic = "unwind"))]
pub fn guarded_entry(device: *mut DeviceHandle, body: impl FnOnce() -> Result<()>) -> i32 {
    run_entry(device, body)
}

macro_rules! entry {
    ($device:expr, $body:block) => {{
        let device_ptr: *mut DeviceHandle = $device;
        // The body is wrapped in its own block so the `block` fragment can be
        // used in closure-body position at all.
        $crate::guarded_entry(device_ptr, move || -> ::reconl_core::Result<()> { $body })
    }};
}

/// Validates the device handle and yields `&mut DeviceHandle`.
macro_rules! device_mut {
    ($ptr:expr) => {{
        let header = unsafe { check_handle($ptr as *mut c_void, Kind::Device, "device")? };
        let device = header as *mut DeviceHandle;
        unsafe { &mut *device }
    }};
}

/// Validates a child handle and proves it belongs to `device`.
macro_rules! child {
    ($device:expr, $ptr:expr, $kind:expr, $what:expr) => {{
        let header = unsafe { check_handle($ptr as *mut c_void, $kind, $what)? };
        let device_header = &$device.header;
        if unsafe { (*header).device } != (device_header as *const HandleHeader as *mut DeviceHandle) {
            return err!(
                Code::InvalidHandle,
                "{} belongs to a different device",
                $what
            );
        }
        header
    }};
}

// ------------------------------------------------------------------- version

#[no_mangle]
pub extern "C" fn reconlVersion(major: *mut u32, minor: *mut u32, patch: *mut u32) {
    unsafe {
        if !major.is_null() {
            *major = VERSION_MAJOR;
        }
        if !minor.is_null() {
            *minor = VERSION_MINOR;
        }
        if !patch.is_null() {
            *patch = VERSION_PATCH;
        }
    }
}

const VERSION_STRING: &[u8] = b"0.1.0\0";

#[no_mangle]
pub extern "C" fn reconlVersionString() -> *const i8 {
    VERSION_STRING.as_ptr() as *const i8
}

const UNKNOWN_NAME: &[u8] = b"unknown\0";
const RESULT_OK: &[u8] = b"RECONL_OK\0";
const RESULT_INVALID_ARGUMENT: &[u8] = b"RECONL_ERR_INVALID_ARGUMENT\0";
const RESULT_OUT_OF_MEMORY: &[u8] = b"RECONL_ERR_OUT_OF_MEMORY\0";
const RESULT_NOT_SUPPORTED: &[u8] = b"RECONL_ERR_NOT_SUPPORTED\0";
const RESULT_BACKEND_UNAVAILABLE: &[u8] = b"RECONL_ERR_BACKEND_UNAVAILABLE\0";
const RESULT_BUDGET_EXCEEDED: &[u8] = b"RECONL_ERR_BUDGET_EXCEEDED\0";
const RESULT_DEVICE_LOST: &[u8] = b"RECONL_ERR_DEVICE_LOST\0";
const RESULT_INVALID_HANDLE: &[u8] = b"RECONL_ERR_INVALID_HANDLE\0";
const RESULT_STRUCT_SIZE: &[u8] = b"RECONL_ERR_STRUCT_SIZE\0";
const RESULT_WRONG_STRUCT_TYPE: &[u8] = b"RECONL_ERR_WRONG_STRUCT_TYPE\0";
const RESULT_ABI_VERSION: &[u8] = b"RECONL_ERR_ABI_VERSION\0";
const RESULT_FRAME_IN_PROGRESS: &[u8] = b"RECONL_ERR_FRAME_IN_PROGRESS\0";
const RESULT_NO_FRAME: &[u8] = b"RECONL_ERR_NO_FRAME\0";
const RESULT_NOT_READY: &[u8] = b"RECONL_ERR_NOT_READY\0";
const RESULT_IO: &[u8] = b"RECONL_ERR_IO\0";
const RESULT_CORRUPT_CACHE: &[u8] = b"RECONL_ERR_CORRUPT_CACHE\0";
const RESULT_DEGRADED: &[u8] = b"RECONL_ERR_DEGRADED\0";
const RESULT_PANIC: &[u8] = b"RECONL_ERR_PANIC\0";
const RESULT_EMPTY_FRAME: &[u8] = b"RECONL_ERR_EMPTY_FRAME\0";

#[no_mangle]
pub extern "C" fn reconlResultName(r: i32) -> *const i8 {
    let name: &[u8] = match r {
        result::OK => RESULT_OK,
        result::INVALID_ARGUMENT => RESULT_INVALID_ARGUMENT,
        result::OUT_OF_MEMORY => RESULT_OUT_OF_MEMORY,
        result::NOT_SUPPORTED => RESULT_NOT_SUPPORTED,
        result::BACKEND_UNAVAILABLE => RESULT_BACKEND_UNAVAILABLE,
        result::BUDGET_EXCEEDED => RESULT_BUDGET_EXCEEDED,
        result::DEVICE_LOST => RESULT_DEVICE_LOST,
        result::INVALID_HANDLE => RESULT_INVALID_HANDLE,
        result::STRUCT_SIZE => RESULT_STRUCT_SIZE,
        result::WRONG_STRUCT_TYPE => RESULT_WRONG_STRUCT_TYPE,
        result::ABI_VERSION => RESULT_ABI_VERSION,
        result::FRAME_IN_PROGRESS => RESULT_FRAME_IN_PROGRESS,
        result::NO_FRAME => RESULT_NO_FRAME,
        result::NOT_READY => RESULT_NOT_READY,
        result::IO => RESULT_IO,
        result::CORRUPT_CACHE => RESULT_CORRUPT_CACHE,
        result::DEGRADED => RESULT_DEGRADED,
        result::PANIC => RESULT_PANIC,
        result::EMPTY_FRAME => RESULT_EMPTY_FRAME,
        _ => UNKNOWN_NAME,
    };
    name.as_ptr() as *const i8
}

#[no_mangle]
pub extern "C" fn reconlTierName(tier: u32) -> *const i8 {
    static NAMES: [&[u8]; 5] = [
        b"T0/gpu-discrete\0",
        b"T1/gpu-shared\0",
        b"T2/cpu-ram\0",
        b"T3/cpu-thrifty\0",
        b"T4/out-of-core\0",
    ];
    let index = (tier as usize).min(TIER_COUNT - 1);
    NAMES[index].as_ptr() as *const i8
}

#[no_mangle]
pub extern "C" fn reconlBackendName(backend: u32) -> *const i8 {
    let name: &[u8] = match backend {
        1 => b"soft-cpu\0",
        2 => b"null\0",
        3 => b"d3d11\0",
        4 => b"d3d12\0",
        5 => b"vulkan\0",
        6 => b"gl\0",
        7 => b"metal\0",
        8 => b"webgpu\0",
        9 => b"wasm-webgl2\0",
        _ => b"none\0",
    };
    name.as_ptr() as *const i8
}

// ------------------------------------------------------------------- logging

#[no_mangle]
pub extern "C" fn reconlSetLogLevel(level: u32) {
    log::set_level(match level {
        0 => Level::Off,
        1 => Level::Error,
        2 => Level::Warn,
        3 => Level::Info,
        4 => Level::Debug,
        _ => Level::Trace,
    });
}

#[no_mangle]
pub extern "C" fn reconlGetLogLevel() -> u32 {
    match log::level() {
        Level::Off => 0,
        Level::Error => 1,
        Level::Warn => 2,
        Level::Info => 3,
        Level::Debug => 4,
        Level::Trace => 5,
    }
}

/// Installs a host log sink. Called from the emitting thread; the sink must not
/// call back into ReconL.
#[no_mangle]
pub extern "C" fn reconlSetLogSink(
    sink: Option<unsafe extern "C" fn(user: *mut c_void, level: u32, message: *const i8, file: *const i8, line: u32)>,
    user: *mut c_void,
) {
    log::set_sink(sink, user);
}

// --------------------------------------------------------------------- probe

fn probe_entry(backend_id: u32, name: &str, device_name: &str, usable: bool, caps: u32, best_tier: Tier, vram: u64, ram: u64, cascades: u32, budget: u64, note: &str) -> ReconLBackendProbe {
    let mut entry = ReconLBackendProbe {
        backend: backend_id,
        name: [0; abi::RECONL_MAX_NAME],
        usable: if usable { 1 } else { 0 },
        caps,
        best_tier: best_tier as u32,
        vram_bytes: vram,
        ram_bytes: ram,
        max_cascades: cascades,
        shadow_texel_budget: budget,
        device_name: [0; abi::RECONL_MAX_NAME],
        note: [0; abi::RECONL_MAX_MESSAGE],
    };
    set_str(&mut entry.name, name);
    set_str(&mut entry.device_name, device_name);
    set_str(&mut entry.note, note);
    entry
}

#[no_mangle]
pub unsafe extern "C" fn reconlProbe(desc: *const ReconLProbeDesc, out: *mut ReconLProbeInfo) -> i32 {
    guarded_entry(core::ptr::null_mut(), move || {
        if out.is_null() {
            return err!(Code::InvalidArgument, "null probe output");
        }
        let mut spill_dir = None;
        if !desc.is_null() {
            // SAFETY: the caller passed a readable descriptor for the duration
            // of this call; `check_header` validates its size before we read it.
            let desc = unsafe {
                check_header::<ReconLProbeDesc>(
                    desc as *const reconl_core::StructHeader,
                    struct_type::PROBE_DESC,
                    core::mem::size_of::<ReconLProbeDesc>() as u32,
                    "ReconLProbeDesc",
                )?
            };
            if !desc.spill_dir.is_null() {
                // SAFETY: documented as a NUL-terminated string owned by the caller.
                spill_dir = Some(unsafe { cstr(desc.spill_dir) });
            }
        }

        let ram = detect_ram_bytes();
        let disk = spill_dir
            .as_deref()
            .map(free_disk_bytes)
            .unwrap_or_else(|| free_disk_bytes_default());

        let (caps_soft, plan) = reconl_backend_softcpu::probe(Tier::CpuRam);
        let (_caps_t4, plan_t4) = reconl_backend_softcpu::probe(Tier::OutOfCore);
        let disk_ok = disk > (64u64 << 20);
        let soft_note = if disk_ok {
            "T2 by default; T3 freezes cascades, T4 spills the static cascade to the arena"
        } else {
            "T2 only: no writable spill directory with free space was found"
        };
        // The hardware path is probed for real: a device has to be creatable at
        // feature level 11.0, not merely declared in the ABI.
        let adapters = reconl_backend_d3d11::probe_adapters().unwrap_or_default();
        let gpu = adapters.first();
        let gpu_usable = gpu.map(|a| a.feature_level_11).unwrap_or(false);
        let gpu_caps = reconl_backend_d3d11::caps_for(Tier::GpuShared);
        let gpu_plan = shadow_plan(
            Tier::GpuShared,
            3,
            24 << 20,
            ShadowFilter::Pcf3x3,
            gpu_caps,
        );
        // The ladder's recommendation, and the one `tier_plan` will make when a
        // device is created without a hint.
        let preferred = if gpu_usable {
            (Tier::GpuShared, backend::D3D11)
        } else {
            (if disk_ok { Tier::OutOfCore } else { Tier::CpuRam }, backend::SOFT_CPU)
        };
        let mut info = ReconLProbeInfo {
            base: reconl_core::StructHeader::new(core::mem::size_of::<ReconLProbeInfo>() as u32, struct_type::PROBE_INFO),
            entry_count: 3,
            entries: [ReconLBackendProbe {
                backend: 0,
                name: [0; abi::RECONL_MAX_NAME],
                usable: 0,
                caps: 0,
                best_tier: 0,
                vram_bytes: 0,
                ram_bytes: 0,
                max_cascades: 0,
                shadow_texel_budget: 0,
                device_name: [0; abi::RECONL_MAX_NAME],
                note: [0; abi::RECONL_MAX_MESSAGE],
            }; abi::RECONL_MAX_BACKENDS],            recommended_tier: preferred.0 as u32,
            recommended_backend: preferred.1,
        };

        info.entries[0] = probe_entry(
            backend::D3D11,
            "d3d11",
            gpu.map(|a| a.description.as_str()).unwrap_or("no D3D11 adapter"),
            gpu_usable,
            gpu_caps,
            Tier::GpuShared,
            gpu.map(|a| a.dedicated_video_memory).unwrap_or(0),
            ram,
            gpu_plan.cascades,
            gpu_plan.map_bytes,
            if gpu_usable {
                "T0/T1: the same passes on hardware, diffed against the soft-cpu golden"
            } else {
                "no D3D11 adapter granting feature level 11.0 was found"
            },
        );
        info.entries[1] = probe_entry(
            backend::SOFT_CPU,
            "soft-cpu",
            "reference tiled rasteriser",
            true,
            caps_soft | if disk_ok { caps::DISK_SPILL | caps::OUT_OF_CORE } else { 0 },
            if disk_ok { Tier::OutOfCore } else { Tier::CpuRam },
            0,
            ram,
            4,
            plan.map_bytes,
            soft_note,
        );
        info.entries[2] = probe_entry(
            backend::NULL,
            "null",
            "deterministic no-op",
            true,
            caps_soft,
            Tier::CpuRam,
            0,
            ram,
            plan_t4.cascades,
            0,
            "counts commands and present calls without producing pixels; the CI backend",
        );

        if !info.entries[0].name.is_empty() {
            // SAFETY: `out` was checked non-null above.
            unsafe { out.write(info) };
        }
        Ok(())
    })
}

/// # Safety
/// `ptr` must point at a NUL-terminated string.
unsafe fn cstr(ptr: *const i8) -> String {
    let mut len = 0usize;
    // SAFETY: the caller guarantees a NUL-terminated string.
    unsafe {
        while *ptr.add(len) != 0 && len < 4096 {
            len += 1;
        }
        let slice = core::slice::from_raw_parts(ptr as *const u8, len);
        String::from_utf8_lossy(slice).into_owned()
    }
}

fn detect_ram_bytes() -> u64 {
    // No OS-specific probing in v0.1: a wrong number is worse than an honest
    // "unknown". The host can pass the real figure in `ReconLMemoryBudget`.
    host_stats().peak_bytes.max(1 << 30)
}

fn free_disk_bytes_default() -> u64 {
    let dir = reconl_resource::spill::default_spill_dir();
    free_disk_bytes(dir.to_string_lossy().as_ref())
}

fn free_disk_bytes(_path: &str) -> u64 {
    // v0.1 writes the arena and lets the OS refuse; reporting a guess here would
    // be a lie the tier resolver would believe. The arena reports a full disk as
    // a counted downgrade instead.
    0
}

// ------------------------------------------------------------------- device

fn d3d11_config_from_desc(desc: &ReconLDeviceDesc, tier: Tier) -> D3d11Config {
    D3d11Config {
        tier,
        adapter_index: 0,
        resolution_scale: 1.0,
        target_frame_ms: desc.target_frame_ms,
        over_target_frames_to_downgrade: desc.downgrade_after_frames,
        // The shadow request is per frame (`ReconLShadowConfig` on the frame
        // descriptor), exactly as it is for the reference backend.
        shadow: reconl_backend_softcpu::ShadowRequest::default(),
        ..D3d11Config::default()
    }
}

fn config_from_desc(desc: &ReconLDeviceDesc, spill_dir: Option<PathBuf>, tier: Tier) -> SoftCpuConfig {
    SoftCpuConfig {
        tier,
        worker_threads: desc.worker_threads,
        seed: desc.seed,
        target_frame_ms: desc.target_frame_ms,
        over_target_frames_to_downgrade: desc.downgrade_after_frames,
        spill_dir,
        frame_policy: reconl_backend_softcpu::FramePolicy::default(),
        ..SoftCpuConfig::default()
    }
}

unsafe fn create_device(desc: *const ReconLDeviceDesc, out: *mut *mut DeviceHandle) -> Result<()> {
    if out.is_null() {
        return err!(Code::InvalidArgument, "null device output");
    }
    // SAFETY: `out` is checked; the caller owns the pointer until we write it.
    unsafe { *out = core::ptr::null_mut() };
    if desc.is_null() {
        return err!(Code::InvalidArgument, "null device descriptor");
    }
    // SAFETY: the caller passed a descriptor for this call.
    let desc = unsafe {
        check_header::<ReconLDeviceDesc>(
            desc as *const reconl_core::StructHeader,
            struct_type::DEVICE_DESC,
            core::mem::size_of::<ReconLDeviceDesc>() as u32,
            "ReconLDeviceDesc",
        )?
    };

    if desc.allocator.is_zeroed() {
        return err!(
            Code::InvalidArgument,
            "no allocator supplied; ReconL will not use the C runtime allocator"
        );
    }
    let allocator_a = HostAllocatorA {
        alloc: desc.allocator.alloc,
        realloc: desc.allocator.realloc,
        free: desc.allocator.free,
        user: desc.allocator.user,
    };
    let alloc = HostAlloc::from_abi(allocator_a);
    alloc.self_check()?;

    // Budget: the host's caps, or an honest default.
    let (caps, allow_disk_spill, spill_dir) = if desc.budget.is_null() {
        (BudgetCaps { vram: 0, ram: 0, disk: 0, allow_disk_spill: false }, false, None)
    } else {
        // SAFETY: the caller passed a budget struct for this call.
        let b = unsafe {
            check_header::<ReconLMemoryBudget>(
                desc.budget as *const reconl_core::StructHeader,
                struct_type::MEMORY_BUDGET,
                core::mem::size_of::<ReconLMemoryBudget>() as u32,
                "ReconLMemoryBudget",
            )?
        };
        let spill = if b.spill_dir.is_null() {
            None
        } else {
            // SAFETY: documented as a NUL-terminated string owned by the caller.
            Some(PathBuf::from(unsafe { cstr(b.spill_dir) }))
        };
        (
            BudgetCaps {
                vram: b.vram_cap_bytes,
                ram: b.ram_cap_bytes,
                disk: b.disk_cap_bytes,
                allow_disk_spill: b.allow_disk_spill != 0,
            },
            b.allow_disk_spill != 0,
            spill,
        )
    };
    let budget = Arc::new(Budget::new(caps));

    // The ladder starts where the *requested* backend can actually run. With no
    // hint the backend follows the machine: a creatable D3D11 device means the
    // hardware tiers, and a machine with no usable GPU API starts at the T2
    // reference tier - which is the CI case the design requires to work.
    let gpu_available = reconl_backend_d3d11::hardware_available();
    let (mut tier, mut tier_reason) = if desc.tier_hint != 0 {
        (Tier::from_u32(desc.tier_hint), TierReason::HostRequest)
    } else {
        match desc.backend_hint {
            backend::D3D11 => (Tier::GpuShared, TierReason::HostRequest),
            backend::SOFT_CPU | backend::NULL => (Tier::CpuRam, TierReason::HostRequest),
            _ if gpu_available => (Tier::GpuShared, TierReason::StartupProbe),
            _ => (Tier::CpuRam, TierReason::NoGpuApi),
        }
    };

    // With no backend named the tier picks one, so the two cannot disagree.
    let requested_backend = if desc.backend_hint == backend::NONE {
        if tier <= Tier::GpuShared {
            backend::D3D11
        } else {
            backend::SOFT_CPU
        }
    } else {
        desc.backend_hint
    };

    // Reconcile the tier with the backend *before* the device is built: a GPU
    // tier label on a software device, or the reverse, would be a lie in the
    // stats a host reads back.
    match requested_backend {
        backend::D3D11 if tier > Tier::GpuShared => {
            tier = Tier::GpuShared;
            tier_reason = TierReason::HostRequest;
        }
        backend::SOFT_CPU | backend::NULL if tier <= Tier::GpuShared => {
            tier = Tier::CpuRam;
            tier_reason = TierReason::NoGpuApi;
        }
        _ => {}
    }

    let spill_dir = spill_dir.or_else(|| if allow_disk_spill { Some(reconl_resource::spill::default_spill_dir()) } else { None });

    // Both backends' configs, built here from the host's descriptor. An offload
    // (docs/offload.md) changes backend at any later frame, and by then this
    // descriptor is the host's memory, not ours.
    let cpu_config = config_from_desc(desc, spill_dir.clone(), tier);
    let gpu_config = d3d11_config_from_desc(desc, tier);

    let backend = match requested_backend {
        backend::NULL => {
            let config = NullConfig { fake_tier: tier, command_capacity: 4096, ..NullConfig::default() };
            BackendKind::Null(Box::new(NullDevice::new(config)))
        }
        backend::D3D11 => {
            BackendKind::D3d11(Box::new(D3d11Device::new(alloc, Arc::clone(&budget), gpu_config.clone())?))
        }
        other => {
            if other != backend::SOFT_CPU {
                // The remaining GPU backends are declared in the ABI and not
                // built yet: refusing is the documented behaviour, not a
                // silent fallback to software.
                return err!(
                    Code::BackendUnavailable,
                    "backend {} is declared in the ABI but not built in this release; use RECONL_BACKEND_D3D11 or RECONL_BACKEND_SOFT_CPU",
                    reconl_core::tier::Backend::from_u32(other).name()
                );
            }
            let mut device = SoftCpuDevice::new(alloc, Arc::clone(&budget), cpu_config.clone())?;
            if tier != Tier::CpuRam {
                // Honour the host's tier hint within the software ladder.
                while device.tier() < tier {
                    let from = device.tier();
                    let to = device.tier().step_down();
                    if to == from {
                        break;
                    }
                    device.step_down(TierReason::HostRequest, 0, "host tier hint");
                }
            }
            BackendKind::SoftCpu(Box::new(device))
        }
    };

    let backend_id = match &backend {
        BackendKind::SoftCpu(_) => backend::SOFT_CPU,
        BackendKind::D3d11(_) => backend::D3D11,
        BackendKind::Null(_) => backend::NULL,
    };
    let device_caps = match &backend {
        BackendKind::SoftCpu(d) => d.caps(),
        BackendKind::D3d11(d) => d.caps(),
        BackendKind::Null(d) => d.caps(),
    };
    let device_name = match &backend {
        BackendKind::SoftCpu(d) => d.device_name(),
        BackendKind::D3d11(d) => d.device_name(),
        BackendKind::Null(d) => d.device_name(),
    };
    let driver = match &backend {
        BackendKind::SoftCpu(d) => d.driver(),
        BackendKind::D3d11(d) => d.driver(),
        BackendKind::Null(d) => d.driver(),
    };

    let mut stats = Stats::new(backend_id, tier, tier_reason);
    stats.caps = device_caps;
    stats.device_name.set(device_name);
    stats.tier_reason_text.set(tier_reason.text());

    let mut handle = DeviceHandle {
        header: HandleHeader { kind: Kind::Device as u32, refcount: AtomicU32::new(1), device: core::ptr::null_mut() },
        alloc,
        budget,
        backend,
        tier,
        tier_reason,
        tier_locked: false,
        caps: device_caps,
        device_name: Text::new(),
        driver: Text::new(),
        stats,
        last_error: None,
        frame_state: FrameState::Idle,
        frame: FrameRecord::new(alloc),
        shadow: reconl_backend_softcpu::ShadowRequest::default(),
        null_downgrades: DowngradeLog::new(),
        audit_every_frames: 0,
        world: WorldRevision::new(),
        spill_dir,
        arena_dir_note: Text::new(),
        origin: Origin {
            // Only a device that was built with the hardware backend can
            // offload: one created on the reference tier is already where an
            // offload would put it, and must never claim a GPU it does not have.
            gpu: (backend_id == backend::D3D11).then_some(gpu_config),
            cpu: cpu_config,
        },
        offload: Offload::new(),
        offload_log: DowngradeLog::new(),
        allow_downgrade: desc.allow_downgrade,
    };
    handle.device_name.set(device_name);
    handle.driver.set(driver);
    handle.shadow = shadow_request_from(None);

    let ptr = unsafe { handle_new(alloc, handle) };
    if ptr.is_null() {
        return err!(Code::OutOfMemory, "the host allocator refused the device handle");
    }
    // `handle_new` copies the handle into the host allocator, so the copy holds
    // the address that must be recorded inside it. Pointing `header.device` at
    // the stack copy leaves every entry point that reads it - the error slot,
    // the spill arena, the per-frame owner lookups - dereferencing a dead frame.
    // SAFETY: `handle_new` returned a live device in the host allocator; nothing
    // frees it before `reconlRelease`, and this runs before `*out` publishes it.
    unsafe { (*ptr).header.device = ptr };
    // SAFETY: `out` was checked non-null.
    unsafe { *out = ptr };
    Ok(())
}

fn shadow_request_from(config: Option<&ReconLShadowConfig>) -> reconl_backend_softcpu::ShadowRequest {
    match config {
        None => reconl_backend_softcpu::ShadowRequest::default(),
        Some(c) => reconl_backend_softcpu::ShadowRequest {
            enabled: c.enabled != 0,
            cascades: c.cascade_count,
            texel_budget_bytes: c.texel_budget_bytes,
            filter: ShadowFilter::from_u32(c.filter),
            max_distance: c.max_distance,
            blend_band: c.blend_band,
            refresh_interval_frames: c.refresh_interval_frames.max(1),
            allow_disk_cache: c.allow_disk_cache != 0,
            // The three bias fields are the host pinning the policy. All zero
            // means "use yours", which is the default and what every host that
            // predates the fields sends; any non-zero value means the host owns
            // all three, and they are used as given rather than rescaled.
            bias: (c.normal_bias != 0.0 || c.depth_bias != 0.0 || c.slope_bias != 0.0).then(|| {
                reconl_shadow::BiasPreset {
                    normal_bias: c.normal_bias,
                    depth_bias: c.depth_bias,
                    slope_bias: c.slope_bias,
                }
            }),
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn reconlCreateDevice(desc: *const ReconLDeviceDesc, out: *mut *mut DeviceHandle) -> i32 {
    guarded_entry(core::ptr::null_mut(), || unsafe { create_device(desc, out) })
}

#[no_mangle]
pub unsafe extern "C" fn reconlGetDeviceLimits(device: *mut DeviceHandle, out: *mut ReconLDeviceLimits) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if out.is_null() {
            return err!(Code::InvalidArgument, "null limits output");
        }
        let budget = device.budget.snapshot();
        let worker_threads = match &device.backend {
            BackendKind::SoftCpu(d) => d.worker_threads(),
            BackendKind::D3d11(_) | BackendKind::Null(_) => 0,
        };
        let tile = match &device.backend {
            BackendKind::SoftCpu(d) => d.tile_size(),
            BackendKind::D3d11(_) | BackendKind::Null(_) => 0,
        };
        // A hardware tier reports the adapter's own video memory here, which is
        // the number a host sizing its resources actually wants.
        let vram = match &device.backend {
            BackendKind::D3d11(d) => d.dedicated_video_memory(),
            BackendKind::SoftCpu(_) | BackendKind::Null(_) => 0,
        };
        let mut limits = ReconLDeviceLimits {
            base: reconl_core::StructHeader::new(core::mem::size_of::<ReconLDeviceLimits>() as u32, struct_type::DEVICE_LIMITS),
            vram_bytes: vram,
            ram_bytes: if budget.ram_cap == 0 { detect_ram_bytes() } else { budget.ram_cap },
            disk_bytes: budget.spill_disk_cap,
            max_allocation_bytes: budget.max_allocation_bytes,
            shadow_texel_budget_bytes: device.shadow.texel_budget_bytes,
            worker_threads_max: worker_threads,
            tile_size_min: tile,
            max_cascades: reconl_core::tier::rules(device.tier).max_cascades,
            max_lights: reconl_raster::shade::MAX_LIGHTS as u32,
            caps: device.caps,
            backend: device.backend_id(),
            device_name: [0; abi::RECONL_MAX_NAME],
            driver: [0; abi::RECONL_MAX_NAME],
        };
        set_str(&mut limits.device_name, device.device_name.as_str());
        set_str(&mut limits.driver, device.driver.as_str());
        // SAFETY: checked non-null.
        unsafe { out.write(limits) };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlGetMemoryStats(device: *mut DeviceHandle, out: *mut ReconLMemoryStats) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if out.is_null() {
            return err!(Code::InvalidArgument, "null memory stats output");
        }
        let mut stats = memory_stats(device);
        stats.base = reconl_core::StructHeader::new(core::mem::size_of::<ReconLMemoryStats>() as u32, struct_type::MEMORY_BUDGET);
        unsafe { out.write(stats) };
        Ok(())
    })
}

fn memory_stats(device: &DeviceHandle) -> ReconLMemoryStats {
    let budget = device.budget.snapshot();
    let host = host_stats();
    let arena = device.softcpu().and_then(|d| d.arena_stats()).unwrap_or_default();
    ReconLMemoryStats {
        base: reconl_core::StructHeader::new(core::mem::size_of::<ReconLMemoryStats>() as u32, struct_type::MEMORY_BUDGET),
        ram_resident_bytes: device.softcpu().map(|d| d.resident_bytes()).unwrap_or(0),
        ram_budget_bytes: budget.ram_cap,
        ram_peak_bytes: budget.ram_peak,
        spill_resident_bytes: budget.spill_resident,
        spill_disk_bytes: arena.bytes,
        spill_disk_cap_bytes: budget.spill_disk_cap,
        spill_evicted_bytes: arena.evicted_bytes,
        spill_cache_bytes: arena.bytes,
        spill_entries: arena.entries as u32,
        spill_evictions: arena.evictions as u32,
        spill_compactions: arena.compactions as u32,
        spill_recovered_entries: arena.recovered_torn as u32,
        spill_errors: arena.errors as u32,
        host_alloc_calls: host.alloc_calls as u32,
        host_alloc_bytes: host.alloc_bytes,
        host_free_calls: host.free_calls as u32,
        reserved: 0,
    }
}

// ----------------------------------------------------------------- resources

#[no_mangle]
pub unsafe extern "C" fn reconlCreateBuffer(device: *mut DeviceHandle, desc: *const ReconLBufferDesc, out: *mut *mut BufferHandle) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if out.is_null() || desc.is_null() {
            return err!(Code::InvalidArgument, "null buffer descriptor or output");
        }
        unsafe { *out = core::ptr::null_mut() };
        let desc = unsafe {
            check_header::<ReconLBufferDesc>(
                desc as *const reconl_core::StructHeader,
                struct_type::BUFFER_DESC,
                core::mem::size_of::<ReconLBufferDesc>() as u32,
                "ReconLBufferDesc",
            )?
        };
        if desc.usage == 0 {
            return err!(Code::InvalidArgument, "a buffer needs at least one usage bit");
        }
        if desc.size_bytes == 0 {
            return err!(Code::InvalidArgument, "a buffer needs a non-zero size");
        }
        // Size before memory: the single-allocation ceiling and the RAM cap are
        // both checked here, so an absurd descriptor never reaches the allocator.
        let ram = device.budget.admit_ram(desc.size_bytes, "a buffer")?;
        let mut bytes = HostVec::with_capacity(device.alloc, desc.size_bytes as usize)?;
        bytes.resize_with(desc.size_bytes as usize, || 0u8)?;
        if !desc.data.is_null() {
            if desc.data_size > desc.size_bytes {
                return err!(Code::InvalidArgument, "initial data is larger than the buffer");
            }
            // SAFETY: the caller says `data_size` bytes are readable.
            let src = unsafe { core::slice::from_raw_parts(desc.data as *const u8, desc.data_size as usize) };
            bytes.as_mut_slice()[..src.len()].copy_from_slice(src);
        }
        let handle = BufferHandle {
            header: HandleHeader { kind: Kind::Buffer as u32, refcount: AtomicU32::new(1), device: device as *mut DeviceHandle },
            bytes,
            ram,
            usage: desc.usage,
            name: Text::new(),
        };
        let mut handle = handle;
        if !desc.debug_name.is_null() {
            // SAFETY: documented as a NUL-terminated string.
            handle.name.set(&unsafe { cstr(desc.debug_name) });
        }
        let ptr = unsafe { handle_new(device.alloc, handle) };
        if ptr.is_null() {
            return err!(Code::OutOfMemory, "the host allocator refused the buffer handle");
        }
        retain_device(device);
        unsafe { *out = ptr };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlWriteBuffer(device: *mut DeviceHandle, buffer: *mut BufferHandle, offset: u64, data: *const c_void, size: u64) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        child!(device, buffer, Kind::Buffer, "buffer");
        let buffer = unsafe { &mut *buffer };
        if data.is_null() {
            return err!(Code::InvalidArgument, "null data");
        }
        let end = offset.saturating_add(size);
        if end > buffer.bytes.len() as u64 {
            return err!(Code::InvalidArgument, "write of {} bytes at {} exceeds the buffer", size, offset);
        }
        // SAFETY: the caller says `size` bytes are readable.
        let src = unsafe { core::slice::from_raw_parts(data as *const u8, size as usize) };
        buffer.bytes.as_mut_slice()[offset as usize..end as usize].copy_from_slice(src);
        Ok(())
    })
}

fn retain_device(device: &mut DeviceHandle) {
    device.header.refcount.fetch_add(1, Ordering::Relaxed);
}

#[no_mangle]
pub unsafe extern "C" fn reconlCreateTexture(device: *mut DeviceHandle, desc: *const ReconLTextureDesc, out: *mut *mut TextureHandle) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if out.is_null() || desc.is_null() {
            return err!(Code::InvalidArgument, "null texture descriptor or output");
        }
        unsafe { *out = core::ptr::null_mut() };
        let desc = unsafe {
            check_header::<ReconLTextureDesc>(
                desc as *const reconl_core::StructHeader,
                struct_type::TEXTURE_DESC,
                core::mem::size_of::<ReconLTextureDesc>() as u32,
                "ReconLTextureDesc",
            )?
        };
        if desc.width == 0 || desc.height == 0 {
            return err!(Code::InvalidArgument, "a texture needs non-zero dimensions");
        }
        if !matches!(desc.format, 1 | 3) {
            return err!(
                Code::NotSupported,
                "format {} is not a sampling format this release creates; use RECONL_FORMAT_R8G8B8A8_UNORM",
                desc.format
            );
        }
        let full_chain = sizing::mip_levels(desc.width, desc.height);
        if desc.mip_levels > full_chain {
            return err!(Code::InvalidArgument, "mip_levels exceeds the full chain");
        }
        // One walk of the chain decides the level count, the price the budget is
        // asked about, and what each level costs to allocate.
        let chain = sizing::TextureChain::of(
            desc.width,
            desc.height,
            if desc.mip_levels == 0 { full_chain } else { desc.mip_levels },
        );
        let ram = device.budget.admit_ram(chain.bytes, "a texture")?;
        let mut levels = HostVec::with_capacity(device.alloc, chain.levels() as usize)?;
        let mut sizes = HostVec::with_capacity(device.alloc, chain.levels() as usize)?;
        for &(w, h) in &chain.sizes {
            let level_bytes = (w as usize) * (h as usize) * 4;
            let mut level = HostVec::with_capacity(device.alloc, level_bytes)?;
            level.resize_with(level_bytes, || 0u8)?;
            levels.push(level)?;
            sizes.push((w, h))?;
        }
        let handle = TextureHandle {
            header: HandleHeader { kind: Kind::Texture as u32, refcount: AtomicU32::new(1), device: device as *mut DeviceHandle },
            width: desc.width,
            height: desc.height,
            format: desc.format,
            usage: desc.usage,
            levels,
            level_sizes: sizes,
            ram,
            name: Text::new(),
        };
        let ptr = unsafe { handle_new(device.alloc, handle) };
        if ptr.is_null() {
            return err!(Code::OutOfMemory, "the host allocator refused the texture handle");
        }
        retain_device(device);
        unsafe { *out = ptr };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlWriteTexture(device: *mut DeviceHandle, texture: *mut TextureHandle, level: *const ReconLTextureLevel) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        child!(device, texture, Kind::Texture, "texture");
        if level.is_null() {
            return err!(Code::InvalidArgument, "null texture level");
        }
        let level_desc = unsafe {
            check_header::<ReconLTextureLevel>(
                level as *const reconl_core::StructHeader,
                struct_type::TEXTURE_DESC,
                core::mem::size_of::<ReconLTextureLevel>() as u32,
                "ReconLTextureLevel",
            )?
        };
        let texture = unsafe { &mut *texture };
        if level_desc.layer != 0 {
            return err!(Code::NotSupported, "array layers beyond 0 are not written in this release");
        }
        let (w, h) = match texture.level_sizes.get(level_desc.mip as usize) {
            Some(size) => *size,
            None => return err!(Code::InvalidArgument, "mip {} does not exist", level_desc.mip),
        };
        let rows = if level_desc.row_count == 0 { h } else { level_desc.row_count };
        if rows > h {
            return err!(Code::InvalidArgument, "row_count exceeds the level height");
        }
        let pitch = if level_desc.row_pitch == 0 { w * 4 } else { level_desc.row_pitch };
        if pitch < w * 4 {
            return err!(Code::InvalidArgument, "row_pitch is smaller than one row of RGBA8");
        }
        let needed = (rows as u64 - 1) * (pitch as u64) + (w as u64) * 4;
        if level_desc.data_size < needed {
            return err!(Code::InvalidArgument, "the level payload is too small for {}x{} rows", w, h);
        }
        // SAFETY: `needed` bytes were confirmed readable by the caller's size.
        let src = unsafe { core::slice::from_raw_parts(level_desc.data as *const u8, level_desc.data_size as usize) };
        let dest = &mut texture.levels.as_mut_slice()[level_desc.mip as usize];
        for row in 0..rows as usize {
            let dst_at = row * (w as usize) * 4;
            let src_at = row * (pitch as usize);
            dest.as_mut_slice()[dst_at..dst_at + (w as usize) * 4].copy_from_slice(&src[src_at..src_at + (w as usize) * 4]);
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlReadTexture(device: *mut DeviceHandle, texture: *mut TextureHandle, mip: u32, layer: u32, out: *mut c_void, out_size: u64, out_row_pitch: u32) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        child!(device, texture, Kind::Texture, "texture");
        if out.is_null() {
            return err!(Code::InvalidArgument, "null readback pointer");
        }
        if layer != 0 {
            return err!(Code::NotSupported, "array layers beyond 0 are not readable in this release");
        }
        let texture = unsafe { &*texture };
        let (w, h) = match texture.level_sizes.get(mip as usize) {
            Some(size) => *size,
            None => return err!(Code::InvalidArgument, "mip {} does not exist", mip),
        };
        let pitch = if out_row_pitch == 0 { w * 4 } else { out_row_pitch };
        let needed = (h as u64 - 1) * pitch as u64 + w as u64 * 4;
        if out_size < needed {
            return err!(Code::InvalidArgument, "the readback buffer is too small for {}x{}", w, h);
        }
        let src = texture.levels.get(mip as usize).map(|l| l.as_slice()).unwrap_or(&[]);
        // SAFETY: the caller provides `out_size` writable bytes.
        let dest = unsafe { core::slice::from_raw_parts_mut(out as *mut u8, out_size as usize) };
        for row in 0..h as usize {
            let src_at = row * (w as usize) * 4;
            let dst_at = row * pitch as usize;
            dest[dst_at..dst_at + (w as usize) * 4].copy_from_slice(&src[src_at..src_at + (w as usize) * 4]);
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlGenerateMips(device: *mut DeviceHandle, texture: *mut TextureHandle) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        child!(device, texture, Kind::Texture, "texture");
        let texture = unsafe { &mut *texture };
        if texture.levels.len() < 2 {
            return err!(Code::InvalidArgument, "the texture has no mip chain");
        }
        // The generator is the same one the resource layer tests: a box filter
        // with a fixed sample order, so the GPU tiers can be diffed against it.
        let base = texture.levels.get(0).map(|l| l.as_slice()).unwrap_or(&[]);
        let chain = generate_mip_chain(device.alloc, texture.width, texture.height, base, texture.levels.len() as u32)?;
        for (index, level) in chain.levels.iter().enumerate() {
            if index == 0 {
                continue;
            }
            if let Some(dest) = texture.levels.get_mut(index) {
                let pixels = level.pixels.as_slice();
                let n = pixels.len().min(dest.len());
                dest.as_mut_slice()[..n].copy_from_slice(&pixels[..n]);
            }
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCreatePipeline(device: *mut DeviceHandle, desc: *const ReconLPipelineDesc, out: *mut *mut PipelineHandle) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if out.is_null() || desc.is_null() {
            return err!(Code::InvalidArgument, "null pipeline descriptor or output");
        }
        unsafe { *out = core::ptr::null_mut() };
        let desc = unsafe {
            check_header::<ReconLPipelineDesc>(
                desc as *const reconl_core::StructHeader,
                struct_type::PIPELINE_DESC,
                core::mem::size_of::<ReconLPipelineDesc>() as u32,
                "ReconLPipelineDesc",
            )?
        };
        if desc.texture_slots > MAX_TEXTURE_SLOTS as u32 {
            return err!(Code::InvalidArgument, "texture_slots exceeds {}", MAX_TEXTURE_SLOTS);
        }
        if desc.shading > shading::TEXTURED_LAMBERT {
            return err!(Code::InvalidArgument, "unknown shading mode {}", desc.shading);
        }
        if desc.shading >= shading::TEXTURED && desc.texture_slots == 0 {
            return err!(Code::InvalidArgument, "a textured pipeline needs at least one texture slot");
        }
        let mut handle = PipelineHandle {
            header: HandleHeader { kind: Kind::Pipeline as u32, refcount: AtomicU32::new(1), device: device as *mut DeviceHandle },
            shading: desc.shading,
            blend: desc.blend,
            cull: desc.cull,
            depth_compare: desc.depth_compare,
            depth_write: desc.depth_write != 0,
            texture_slots: desc.texture_slots,
            receives_shadow: desc.receives_shadow != 0,
            casts_shadow: desc.casts_shadow != 0,
            two_sided_shadow: desc.flags & 1 != 0,
            name: Text::new(),
        };
        if !desc.debug_name.is_null() {
            // SAFETY: documented as a NUL-terminated string.
            handle.name.set(&unsafe { cstr(desc.debug_name) });
        }
        let ptr = unsafe { handle_new(device.alloc, handle) };
        if ptr.is_null() {
            return err!(Code::OutOfMemory, "the host allocator refused the pipeline handle");
        }
        retain_device(device);
        unsafe { *out = ptr };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCreateSwapchain(device: *mut DeviceHandle, desc: *const ReconLSwapchainDesc, out: *mut *mut SwapchainHandle) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if out.is_null() || desc.is_null() {
            return err!(Code::InvalidArgument, "null swapchain descriptor or output");
        }
        unsafe { *out = core::ptr::null_mut() };
        let desc = unsafe {
            check_header::<ReconLSwapchainDesc>(
                desc as *const reconl_core::StructHeader,
                struct_type::SWAPCHAIN_DESC,
                core::mem::size_of::<ReconLSwapchainDesc>() as u32,
                "ReconLSwapchainDesc",
            )?
        };
        if desc.width == 0 || desc.height == 0 {
            return err!(Code::InvalidArgument, "a swapchain needs non-zero dimensions");
        }
        if desc.image_count < 1 || desc.image_count > 3 {
            return err!(Code::InvalidArgument, "image_count must be 1..3");
        }
        if !matches!(desc.format, 1 | 2 | 3) {
            return err!(Code::NotSupported, "only 8-bit RGBA/BGRA swapchain formats are supported in this release");
        }
        // A swapchain allocates nothing itself, but it names the image sizes the
        // host will allocate and present, so it passes the same ceiling.
        device.budget.check_allocation(sizing::swapchain_bytes(&desc), "a swapchain")?;
        let handle = SwapchainHandle {
            header: HandleHeader { kind: Kind::Swapchain as u32, refcount: AtomicU32::new(1), device: device as *mut DeviceHandle },
            width: desc.width,
            height: desc.height,
            format: desc.format,
            present_to_memory: desc.present_to_memory != 0,
            depth: desc.depth_format != 0,
        };
        let ptr = unsafe { handle_new(device.alloc, handle) };
        if ptr.is_null() {
            return err!(Code::OutOfMemory, "the host allocator refused the swapchain handle");
        }
        retain_device(device);
        unsafe { *out = ptr };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCreateCommandList(device: *mut DeviceHandle, desc: *const ReconLCommandListDesc, out: *mut *mut CommandListHandle) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if out.is_null() {
            return err!(Code::InvalidArgument, "null command list output");
        }
        unsafe { *out = core::ptr::null_mut() };
        let capacity_bytes = if desc.is_null() {
            sizing::COMMAND_BYTES * 1024
        } else {
            let desc = unsafe {
                check_header::<ReconLCommandListDesc>(
                    desc as *const reconl_core::StructHeader,
                    struct_type::COMMAND_LIST_DESC,
                    core::mem::size_of::<ReconLCommandListDesc>() as u32,
                    "ReconLCommandListDesc",
                )?
            };
            if desc.capacity_bytes == 0 {
                sizing::COMMAND_BYTES * 1024
            } else {
                desc.capacity_bytes
            }
        };
        let (slots, slot_bytes) = sizing::command_list_slots(capacity_bytes);
        let ram = device.budget.admit_ram(slot_bytes, "a command list")?;
        let commands = HostVec::with_capacity(device.alloc, slots as usize)?;
        let mut handle = CommandListHandle {
            header: HandleHeader { kind: Kind::CommandList as u32, refcount: AtomicU32::new(1), device: device as *mut DeviceHandle },
            commands,
            ram,
            capacity_bytes: slot_bytes as u32,
            bytes: 0,
            recording: false,
            name: Text::new(),
        };
        if !desc.is_null() && !unsafe { (*desc).debug_name }.is_null() {
            // SAFETY: documented as a NUL-terminated string.
            let name = unsafe { cstr((*desc).debug_name) };
            handle.name.set(&name);
        }
        let ptr = unsafe { handle_new(device.alloc, handle) };
        if ptr.is_null() {
            return err!(Code::OutOfMemory, "the host allocator refused the command list handle");
        }
        retain_device(device);
        unsafe { *out = ptr };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCreateFence(device: *mut DeviceHandle, signaled: u32, out: *mut *mut FenceHandle) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if out.is_null() {
            return err!(Code::InvalidArgument, "null fence output");
        }
        unsafe { *out = core::ptr::null_mut() };
        let handle = FenceHandle {
            header: HandleHeader { kind: Kind::Fence as u32, refcount: AtomicU32::new(1), device: device as *mut DeviceHandle },
            signaled: signaled != 0,
            frame_index: 0,
        };
        let ptr = unsafe { handle_new(device.alloc, handle) };
        if ptr.is_null() {
            return err!(Code::OutOfMemory, "the host allocator refused the fence handle");
        }
        retain_device(device);
        unsafe { *out = ptr };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlWaitFence(device: *mut DeviceHandle, fence: *mut FenceHandle, timeout_ns: u64) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        child!(device, fence, Kind::Fence, "fence");
        let fence = unsafe { &*fence };
        if fence.signaled {
            return Ok(());
        }
        // Every milestone-1 backend is synchronous, so a fence is signaled by
        // the time Submit returns. A non-signaled fence here means the host
        // waited on a fence that was never submitted.
        if timeout_ns == 0 {
            return err!(Code::NotReady, "the fence is not signaled");
        }
        err!(Code::NotReady, "the fence was never submitted; this release signals fences inside Submit")
    })
}

// ------------------------------------------------------------------ commands

#[no_mangle]
pub unsafe extern "C" fn reconlCmdReset(list: *mut CommandListHandle) -> i32 {
    entry!(core::ptr::null_mut(), {
        if list.is_null() {
            return err!(Code::InvalidArgument, "null command list");
        }
        let list = unsafe { &mut *list };
        list.commands.clear();
        list.bytes = 0;
        list.recording = true;
        Ok(())
    })
}

fn push_command(list: &mut CommandListHandle, command: Command) -> Result<()> {
    if !list.recording {
        return err!(Code::InvalidArgument, "the command list is not recording; call reconlCmdReset first");
    }
    if list.bytes + sizing::COMMAND_BYTES > list.capacity_bytes {
        return err!(
            Code::BudgetExceeded,
            "the command list is full ({} bytes); grow the capacity at creation",
            list.capacity_bytes
        );
    }
    list.commands.push(command)?;
    list.bytes += sizing::COMMAND_BYTES;
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn reconlCmdBeginRenderPass(list: *mut CommandListHandle, desc: *const ReconLRenderPassDesc) -> i32 {
    entry!(core::ptr::null_mut(), {
        if list.is_null() || desc.is_null() {
            return err!(Code::InvalidArgument, "null command list or render pass");
        }
        let list = unsafe { &mut *list };
        let desc = unsafe {
            check_header::<ReconLRenderPassDesc>(
                desc as *const reconl_core::StructHeader,
                struct_type::RENDER_PASS_DESC,
                core::mem::size_of::<ReconLRenderPassDesc>() as u32,
                "ReconLRenderPassDesc",
            )?
        };
        if desc.color_count > 1 {
            return err!(
                Code::NotSupported,
                "this release renders one colour attachment per pass; {} were given",
                desc.color_count
            );
        }
        for command in list.commands.iter() {
            if matches!(command, Command::BeginPass { .. }) {
                return err!(
                    Code::NotSupported,
                    "this release supports one render pass per command list; a second pass is refused rather than ignored"
                );
            }
        }
        let target = desc.color[0].texture as *mut TextureHandle;
        push_command(
            list,
            Command::BeginPass {
                target,
                load_color: desc.load_color != 0,
                load_depth: desc.load_depth != 0,
                clear_color: desc.clear_color,
                clear_depth: desc.clear_depth,
                width: desc.viewport_width,
                height: desc.viewport_height,
            },
        )?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCmdEndRenderPass(list: *mut CommandListHandle) -> i32 {
    entry!(core::ptr::null_mut(), {
        if list.is_null() {
            return err!(Code::InvalidArgument, "null command list");
        }
        let list = unsafe { &mut *list };
        push_command(list, Command::EndPass)?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCmdSetPipeline(list: *mut CommandListHandle, pipeline: *mut PipelineHandle) -> i32 {
    entry!(core::ptr::null_mut(), {
        if list.is_null() {
            return err!(Code::InvalidArgument, "null command list");
        }
        let list = unsafe { &mut *list };
        push_command(list, Command::SetPipeline { pipeline })?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCmdSetVertexBuffer(list: *mut CommandListHandle, stream: u32, buffer: *mut BufferHandle, offset: u64) -> i32 {
    entry!(core::ptr::null_mut(), {
        if list.is_null() {
            return err!(Code::InvalidArgument, "null command list");
        }
        if stream != 0 {
            return err!(Code::NotSupported, "this release has one vertex stream");
        }
        let list = unsafe { &mut *list };
        push_command(list, Command::SetVertexBuffer { stream, buffer, offset })?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCmdSetIndexBuffer(list: *mut CommandListHandle, buffer: *mut BufferHandle, offset: u64, format: u32) -> i32 {
    entry!(core::ptr::null_mut(), {
        if list.is_null() {
            return err!(Code::InvalidArgument, "null command list");
        }
        if format > 1 {
            return err!(Code::InvalidArgument, "unknown index format {}", format);
        }
        let list = unsafe { &mut *list };
        push_command(list, Command::SetIndexBuffer { buffer, offset, format })?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCmdSetTexture(list: *mut CommandListHandle, slot: u32, texture: *mut TextureHandle, sampler: *const ReconLSamplerDesc) -> i32 {
    entry!(core::ptr::null_mut(), {
        if list.is_null() {
            return err!(Code::InvalidArgument, "null command list");
        }
        if slot >= MAX_TEXTURE_SLOTS as u32 {
            return err!(Code::InvalidArgument, "texture slot {} is out of range", slot);
        }
        let (filter, wrap_u, wrap_v) = if sampler.is_null() {
            (0u32, 0u32, 0u32)
        } else {
            let s = unsafe {
                check_header::<ReconLSamplerDesc>(
                    sampler as *const reconl_core::StructHeader,
                    struct_type::SAMPLER_DESC,
                    core::mem::size_of::<ReconLSamplerDesc>() as u32,
                    "ReconLSamplerDesc",
                )?
            };
            (s.filter, s.wrap_u, s.wrap_v)
        };
        let list = unsafe { &mut *list };
        push_command(list, Command::SetTexture { slot, texture, filter, wrap_u, wrap_v })?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCmdPushConstants(list: *mut CommandListHandle, slot: u32, data: *const c_void, size_bytes: u32) -> i32 {
    entry!(core::ptr::null_mut(), {
        if list.is_null() || data.is_null() {
            return err!(Code::InvalidArgument, "null command list or data");
        }
        if size_bytes == 0 || size_bytes > 64 {
            return err!(Code::InvalidArgument, "push constants are 1..64 bytes");
        }
        let mut buffer = [0u8; 64];
        // SAFETY: the caller says `size_bytes` bytes are readable.
        let src = unsafe { core::slice::from_raw_parts(data as *const u8, size_bytes as usize) };
        buffer[..src.len()].copy_from_slice(src);
        let list = unsafe { &mut *list };
        push_command(list, Command::PushConstants { slot, data: buffer, size: size_bytes })?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCmdDraw(list: *mut CommandListHandle, vertex_count: u32, first_vertex: u32) -> i32 {
    entry!(core::ptr::null_mut(), {
        if list.is_null() {
            return err!(Code::InvalidArgument, "null command list");
        }
        if vertex_count == 0 {
            return err!(Code::InvalidArgument, "a draw needs vertices");
        }
        let list = unsafe { &mut *list };
        push_command(list, Command::Draw { vertex_count, first_vertex })?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCmdDrawIndexed(list: *mut CommandListHandle, index_count: u32, first_index: u32, vertex_offset: i32) -> i32 {
    entry!(core::ptr::null_mut(), {
        if list.is_null() {
            return err!(Code::InvalidArgument, "null command list");
        }
        if index_count == 0 {
            return err!(Code::InvalidArgument, "a draw needs indices");
        }
        let list = unsafe { &mut *list };
        push_command(list, Command::DrawIndexed { index_count, first_index, vertex_offset })?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlCmdCount(list: *const CommandListHandle) -> u32 {
    if list.is_null() {
        return 0;
    }
    unsafe { (*list).commands.len() as u32 }
}

// --------------------------------------------------------------------- frame

#[no_mangle]
pub unsafe extern "C" fn reconlBeginFrame(device: *mut DeviceHandle, desc: *mut ReconLFrameDesc) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if desc.is_null() {
            return err!(Code::InvalidArgument, "null frame descriptor");
        }
        if device.frame_state != FrameState::Idle {
            return err!(Code::FrameInProgress, "a frame is already open");
        }
        let desc_ref = unsafe {
            check_header::<ReconLFrameDesc>(
                desc as *const reconl_core::StructHeader,
                struct_type::FRAME_DESC,
                // The prefix, not the whole struct: `camera` is the extension
                // region and a caller that predates it is still read.
                ReconLFrameDesc::PREFIX_SIZE,
                "ReconLFrameDesc",
            )?
        };
        if desc_ref.width == 0 || desc_ref.height == 0 {
            return err!(Code::InvalidArgument, "a frame needs non-zero dimensions");
        }

        let mut lights = LightSet::new();
        let mut light_dir = [0.0f32, -1.0, 0.0];
        let mut light_hash = 0xcbf2_9ce4_8422_2325u64;
        let mut shadowed_lights = 0u32;
        if !desc_ref.lights.is_null() {
            // SAFETY: the caller passed a light list for this call.
            let list = unsafe {
                check_header::<ReconLLightList>(
                    desc_ref.lights as *const reconl_core::StructHeader,
                    struct_type::LIGHT_LIST,
                    core::mem::size_of::<ReconLLightList>() as u32,
                    "ReconLLightList",
                )?
            };
            if list.count > 0 {
                if list.lights.is_null() {
                    return err!(Code::InvalidArgument, "the light list is empty but the pointer is null");
                }
                // SAFETY: the caller says `count` lights are readable.
                let raw = unsafe { core::slice::from_raw_parts(list.lights, list.count as usize) };
                if raw.len() > reconl_raster::shade::MAX_LIGHTS {
                    return err!(Code::InvalidArgument, "at most {} lights", reconl_raster::shade::MAX_LIGHTS);
                }
                for l in raw {
                    // SAFETY: the caller passed these structs for this call.
                    let l = unsafe {
                        check_header::<ReconLLight>(
                            l as *const ReconLLight as *const reconl_core::StructHeader,
                            struct_type::LIGHT,
                            core::mem::size_of::<ReconLLight>() as u32,
                            "ReconLLight",
                        )?
                    };
                    let mut light = match l.r#type {
                        0 => {
                            light_dir = l.direction;
                            Light::directional(l.direction, l.color, l.intensity)
                        }
                        1 => Light {
                            kind: reconl_raster::shade::LIGHT_SPOT,
                            position: l.position,
                            direction: l.direction,
                            color: l.color,
                            intensity: l.intensity,
                            range: l.range,
                            cos_inner: (l.cone_inner_deg * core::f32::consts::PI / 180.0).cos(),
                            cos_outer: (l.cone_outer_deg * core::f32::consts::PI / 180.0).cos(),
                            cast_shadow: l.cast_shadow != 0,
                            cascade_base: u32::MAX,
                        },
                        2 => Light {
                            kind: reconl_raster::shade::LIGHT_POINT,
                            position: l.position,
                            direction: [0.0, -1.0, 0.0],
                            color: l.color,
                            intensity: l.intensity,
                            range: l.range,
                            cos_inner: 1.0,
                            cos_outer: 1.0,
                            cast_shadow: l.cast_shadow != 0,
                            cascade_base: u32::MAX,
                        },
                        other => return err!(Code::InvalidArgument, "unknown light type {}", other),
                    };
                    // Only one light gets the cascade budget per frame. This is
                    // decided by whether one has already been claimed, not by how
                    // many lights have been pushed so far: that count is still 0
                    // for the first light, so a one-light host - the common case -
                    // could never cast a shadow at all.
                    light.cast_shadow = l.cast_shadow != 0 && shadowed_lights == 0;
                    if light.cast_shadow {
                        shadowed_lights += 1;
                    }
                    if !lights.push(light) {
                        return err!(Code::InvalidArgument, "too many lights");
                    }
                    light_hash = hash_light(light_hash, l);
                }
            }
        }
        if shadowed_lights == 0 {
            device.stats.shadows.shadowed_lights = 0;
        }

        let shadow = if desc_ref.shadows.is_null() {
            device.shadow
        } else {
            // SAFETY: the caller passed a config for this call.
            let config = unsafe {
                check_header::<ReconLShadowConfig>(
                    desc_ref.shadows as *const reconl_core::StructHeader,
                    struct_type::SHADOW_CONFIG,
                    core::mem::size_of::<ReconLShadowConfig>() as u32,
                    "ReconLShadowConfig",
                )?
            };
            shadow_request_from(Some(config))
        };

        device.frame = FrameRecord::new(device.alloc);
        device.frame.index = device.stats.counters.frames_presented as u64 + (device.stats.counters.frames_dropped as u64);
        device.frame.width = desc_ref.width;
        device.frame.height = desc_ref.height;
        device.frame.lights = lights;
        device.frame.light_dir = light_dir;
        device.frame.light_hash = light_hash;
        device.frame.shadow = shadow;
        device.frame.aspect = desc_ref.width as f32 / desc_ref.height as f32;
        // The shadow system's camera, which is not the vertex transform in
        // constant slot 0 and never was: cascade fitting inverts the view as a
        // rigid basis and measures camera-space depth against it, and neither
        // survives being given a view * projection. A caller that does not
        // reach the `camera` field gets the identity view and the default
        // frustum, which is the right camera for geometry drawn in clip space.
        if desc_ref.has_camera() && !desc_ref.camera.is_null() {
            // SAFETY: `has_camera` proved the field is inside the caller's
            // struct, and the caller guarantees `camera` points at a readable
            // ReconLCamera for this call.
            let cam = unsafe {
                check_header::<ReconLCamera>(
                    desc_ref.camera as *const reconl_core::StructHeader,
                    struct_type::CAMERA,
                    core::mem::size_of::<ReconLCamera>() as u32,
                    "ReconLCamera",
                )?
            };
            if !(cam.near > 0.0) || !(cam.far > cam.near) {
                return err!(
                    Code::InvalidArgument,
                    "camera near is {} and far is {}; the fit needs 0 < near < far",
                    cam.near,
                    cam.far
                );
            }
            if !(cam.fov_y_deg > 0.0 && cam.fov_y_deg < 180.0) {
                return err!(Code::InvalidArgument, "camera fov_y_deg is {}", cam.fov_y_deg);
            }
            device.frame.camera_view = cam.view;
            device.frame.fov_y_deg = cam.fov_y_deg;
            device.frame.near = cam.near;
            device.frame.far = cam.far;
        }
        device.frame.world_revision = device.world.revision();
        device.frame.static_revision = device.world.static_geometry_revision();

        // Reserve the frame's targets now, so recording and submitting allocate
        // nothing new. A refused reservation is reported here rather than taken
        // out mid-frame - and it happens before the frame is marked open, so a
        // commit that cannot be carried out leaves the device where the host can
        // begin again instead of in `Open` with a frame it can never record.
        let (width, height) = (device.frame.width, device.frame.height);
        if let Some(soft) = device.softcpu_mut() {
            soft.prepare_frame(width, height)?;
        }
        device.frame_state = FrameState::Open;
        Ok(())
    })
}

fn hash_light(mut hash: u64, l: &ReconLLight) -> u64 {
    let mut feed = |bytes: &[u8]| {
        for b in bytes {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(&l.r#type.to_le_bytes());
    feed(&l.cast_shadow.to_le_bytes());
    for v in l.position.iter().chain(l.direction.iter()).chain(l.color.iter()) {
        feed(&v.to_bits().to_le_bytes());
    }
    feed(&l.intensity.to_bits().to_le_bytes());
    feed(&l.range.to_bits().to_le_bytes());
    feed(&l.cone_inner_deg.to_bits().to_le_bytes());
    feed(&l.cone_outer_deg.to_bits().to_le_bytes());
    hash
}

// -------------------------------------------------------------------- submit

struct PassState {
    in_pass: bool,
    pipeline: Option<PipelineState>,
    pipeline_rec: Option<*mut PipelineHandle>,
    vertex: Option<(*mut BufferHandle, u64)>,
    index: Option<(*mut BufferHandle, u64, u32)>,
    texture: Option<(*mut TextureHandle, u32, u32, u32)>,
    view_proj: Mat4,
    model: Mat4,
    clear_color: [f32; 4],
    clear_depth: f32,
    load_color: bool,
    load_depth: bool,
}

impl Default for PassState {
    fn default() -> Self {
        Self {
            in_pass: false,
            pipeline: None,
            pipeline_rec: None,
            vertex: None,
            index: None,
            texture: None,
            view_proj: IDENTITY,
            model: IDENTITY,
            clear_color: [0.0; 4],
            clear_depth: 0.0,
            load_color: true,
            load_depth: true,
        }
    }
}

fn pipeline_state(pipeline: &PipelineHandle) -> PipelineState {
    PipelineState {
        blend: pipeline.blend,
        cull: match pipeline.cull {
            0 => reconl_raster::CULL_NONE,
            2 => reconl_raster::CULL_FRONT,
            _ => CULL_BACK,
        },
        depth_compare: if pipeline.depth_compare == 0 { 0 } else { COMPARE_GREATER },
        depth_test: true,
        depth_write: pipeline.depth_write,
        // Two-sided lighting is not a milestone-1 pipeline flag: the ABI's flag 1
        // is "two-sided shadow render", which the shadow pass already does by
        // culling front faces.
        two_sided: false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn reconlSubmit(device: *mut DeviceHandle, list: *const CommandListHandle, fence: *mut FenceHandle) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if list.is_null() {
            return err!(Code::InvalidArgument, "null command list");
        }
        if device.frame_state != FrameState::Open {
            return err!(Code::NoFrame, "no frame is open; call reconlBeginFrame first");
        }
        let list = unsafe { &*list };
        child!(device, list as *const CommandListHandle as *mut CommandListHandle, Kind::CommandList, "command list");
        if !fence.is_null() {
            // Both handles are checked before the frame is committed: a handle
            // from another device is a caller bug that must not cost the host
            // the frame it is about to render, so the frame stays open and the
            // call can be retried with a valid one.
            child!(device, fence, Kind::Fence, "fence");
        }
        // Past this point the frame is committed: any failure ends it.
        let mut owner = FrameOwner::new(device as *mut DeviceHandle);

        let mut state = PassState::default();
        let mut draws: HostVec<DrawRecord> = HostVec::with_capacity(device.alloc, list.commands.len())?;
        let mut saw_pass = false;

        for command in list.commands.iter() {
            match *command {
                Command::BeginPass { target, load_color, load_depth, clear_color, clear_depth, width: _, height: _ } => {
                    if state.in_pass {
                        return err!(Code::InvalidArgument, "BeginRenderPass inside a render pass");
                    }
                    if !target.is_null() {
                        child!(device, target, Kind::Texture, "texture");
                    }
                    state.in_pass = true;
                    state.load_color = load_color;
                    state.load_depth = load_depth;
                    state.clear_color = clear_color;
                    state.clear_depth = clear_depth;
                    saw_pass = true;
                    device.frame.clear_color = clear_color;
                    device.frame.clear_depth = clear_depth;
                    device.frame.clear_color_on = load_color;
                    device.frame.clear_depth_on = load_depth;
                }
                Command::EndPass => {
                    if !state.in_pass {
                        return err!(Code::InvalidArgument, "EndRenderPass without a pass");
                    }
                    state.in_pass = false;
                }
                Command::SetPipeline { pipeline } => {
                    if pipeline.is_null() {
                        return err!(Code::InvalidArgument, "null pipeline");
                    }
                    child!(device, pipeline, Kind::Pipeline, "pipeline");
                    let pipeline = unsafe { &*pipeline };
                    state.pipeline = Some(pipeline_state(pipeline));
                    state.pipeline_rec = Some(pipeline as *const PipelineHandle as *mut PipelineHandle);
                }
                Command::SetVertexBuffer { stream: _, buffer, offset } => {
                    if buffer.is_null() {
                        return err!(Code::InvalidArgument, "null vertex buffer");
                    }
                    child!(device, buffer, Kind::Buffer, "buffer");
                    state.vertex = Some((buffer, offset));
                }
                Command::SetIndexBuffer { buffer, offset, format } => {
                    if buffer.is_null() {
                        return err!(Code::InvalidArgument, "null index buffer");
                    }
                    child!(device, buffer, Kind::Buffer, "buffer");
                    state.index = Some((buffer, offset, format));
                }
                Command::SetTexture { slot, texture, filter, wrap_u, wrap_v } => {
                    if !texture.is_null() {
                        child!(device, texture, Kind::Texture, "texture");
                    }
                    state.texture = Some((texture, filter, wrap_u, wrap_v));
                    let _ = slot;
                }
                Command::PushConstants { slot, data, size } => match slot {
                    PUSH_CONSTANT_VIEW_PROJ => {
                        if size < 64 {
                            return err!(Code::InvalidArgument, "push constant slot 0 takes a 4x4 matrix (64 bytes)");
                        }
                        state.view_proj = read_mat4(&data);
                    }
                    PUSH_CONSTANT_MODEL => {
                        if size < 64 {
                            return err!(Code::InvalidArgument, "push constant slot 1 takes a 4x4 matrix (64 bytes)");
                        }
                        state.model = read_mat4(&data);
                    }
                    other => {
                        return err!(Code::InvalidArgument, "unknown push constant slot {}", other);
                    }
                },
                Command::Draw { vertex_count, first_vertex } => {
                    if !state.in_pass {
                        return err!(Code::InvalidArgument, "Draw outside a render pass");
                    }
                    let (buffer, offset) = match state.vertex {
                        Some(v) => v,
                        None => return err!(Code::NotReady, "Draw without a vertex buffer"),
                    };
                    let buffer_ref = unsafe { &*buffer };
                    let stride = core::mem::size_of::<Vertex>() as u64;
                    let base = offset + first_vertex as u64 * stride;
                    let end = base + vertex_count as u64 * stride;
                    if end > buffer_ref.bytes.len() as u64 {
                        return err!(
                            Code::InvalidArgument,
                            "the draw reads {} bytes at {} but the buffer holds {}",
                            vertex_count as u64 * stride,
                            base,
                            buffer_ref.bytes.len()
                        );
                    }
                    let vertices = unsafe { buffer_ref.bytes.as_ptr().add(base as usize) as *const Vertex };
                    let pipeline = state.pipeline.unwrap_or_else(PipelineState::default);
                    let rec = state.pipeline_rec.map(|p| unsafe { &*p });
                    draws.push(DrawRecord {
                        vertices,
                        vertex_count,
                        indices: core::ptr::null(),
                        index_count: 0,
                        transform: math::mul(&state.view_proj, &state.model),
                        model: state.model,
                        pipeline,
                        shading: rec.map(|p| p.shading).unwrap_or(shading::UNLIT),
                        textured: rec.map(|p| p.texture_slots > 0).unwrap_or(false),
                        lit: rec.map(|p| p.shading == shading::LAMBERT || p.shading == shading::TEXTURED_LAMBERT).unwrap_or(false),
                        receives_shadow: rec.map(|p| p.receives_shadow).unwrap_or(false),
                        dynamic: true,
                        casts_shadow: rec.map(|p| p.casts_shadow).unwrap_or(false),
                    })?;
                    device.frame.triangles += (vertex_count / 3) as u64;
                }
                Command::DrawIndexed { index_count, first_index, vertex_offset } => {
                    if !state.in_pass {
                        return err!(Code::InvalidArgument, "DrawIndexed outside a render pass");
                    }
                    let (buffer, offset) = match state.vertex {
                        Some(v) => v,
                        None => return err!(Code::NotReady, "DrawIndexed without a vertex buffer"),
                    };
                    let (index_buffer, index_offset, index_format) = match state.index {
                        Some(i) => i,
                        None => return err!(Code::NotReady, "DrawIndexed without an index buffer"),
                    };
                    if index_format != 1 {
                        return err!(
                            Code::NotSupported,
                            "16-bit indices are not read directly in this release; upload uint32 indices"
                        );
                    }
                    let stride = core::mem::size_of::<Vertex>() as u64;
                    let buffer_ref = unsafe { &*buffer };
                    let index_ref = unsafe { &*index_buffer };
                    let index_bytes = index_count as u64 * 4;
                    if index_offset + first_index as u64 * 4 + index_bytes > index_ref.bytes.len() as u64 {
                        return err!(Code::InvalidArgument, "the indexed draw reads past the index buffer");
                    }
                    // The base the indices are relative to: the offset the host
                    // bound the buffer at, plus the draw's own first vertex. Both
                    // were previously dropped, so an indexed draw with a non-zero
                    // offset drew whatever sat at the start of the buffer.
                    let base = offset + vertex_offset.max(0) as u64 * stride;
                    let max_index = vertex_count_upper(index_ref, index_offset, first_index, index_count);
                    if base + (max_index + 1) * stride > buffer_ref.bytes.len() as u64 {
                        return err!(Code::InvalidArgument, "the indexed draw reads past the vertex buffer");
                    }
                    let indices = unsafe { index_ref.bytes.as_ptr().add(index_offset as usize).add(first_index as usize * 4) as *const u32 };
                    let vertices = unsafe { buffer_ref.bytes.as_ptr().add(base as usize) as *const Vertex };
                    let pipeline = state.pipeline.unwrap_or_else(PipelineState::default);
                    let rec = state.pipeline_rec.map(|p| unsafe { &*p });
                    draws.push(DrawRecord {
                        vertices,
                        vertex_count: 0,
                        indices,
                        index_count,
                        transform: math::mul(&state.view_proj, &state.model),
                        model: state.model,
                        pipeline,
                        shading: rec.map(|p| p.shading).unwrap_or(shading::UNLIT),
                        textured: rec.map(|p| p.texture_slots > 0).unwrap_or(false),
                        lit: rec.map(|p| p.shading == shading::LAMBERT || p.shading == shading::TEXTURED_LAMBERT).unwrap_or(false),
                        receives_shadow: rec.map(|p| p.receives_shadow).unwrap_or(false),
                        dynamic: true,
                        casts_shadow: rec.map(|p| p.casts_shadow).unwrap_or(false),
                    })?;
                    device.frame.triangles += (index_count / 3) as u64;
                }
            }
        }
        if state.in_pass {
            return err!(Code::InvalidArgument, "the command list ends inside a render pass");
        }
        if !saw_pass {
            return err!(Code::InvalidArgument, "the command list has no render pass");
        }

        device.frame.draws = draws;
        let mut frame = std::mem::replace(&mut device.frame, FrameRecord::new(device.alloc));
        frame.width = if frame.width == 0 { 1 } else { frame.width };
        frame.height = if frame.height == 0 { 1 } else { frame.height };

        // A hardware fault is reported here rather than returned: the frame it
        // interrupted is rendered again on the reference tier instead of being
        // lost (docs/offload.md).
        let mut faulted: Option<Error> = None;
        match &mut device.backend {
            BackendKind::SoftCpu(soft) => {
                let draw_items = build_draws(device.alloc, frame.draws.as_slice())?;
                let input = frame_input(&frame, draw_items.as_slice());
                soft.render(&input)?;
                frame.checksum = soft.color_checksum();
                frame.depth_checksum = soft.depth_checksum();

                // The audit: re-render the same frame and compare. It cannot
                // catch worker-count nondeterminism (the tests do that), but it
                // catches state-dependent nondeterminism - a stale shadow map, a
                // recycled buffer, a cache that changed the answer.
                if device.audit_every_frames > 0 && frame.index % device.audit_every_frames as u64 == 0 {
                    let second = soft.render(&input)?;
                    if second.frame_index != frame.index || soft.color_checksum() != frame.checksum {
                        device.stats.counters.audit_divergences += 1;
                        log_warn!("audit: frame {} rendered differently the second time", frame.index);
                    }
                }
                device.stats.shadows = soft.snapshot().shadows;
            }
            BackendKind::D3d11(gpu) => {
                let draw_items = build_draws(device.alloc, frame.draws.as_slice())?;
                let input = frame_input(&frame, draw_items.as_slice());
                // Targets are reserved between frames, so Submit only uploads
                // and draws. A failure is reported under the code the backend's
                // classifier chose; only a classified DEVICE_LOST is treated as
                // a fault, and the driver's own removal verdict must agree -
                // the code and the verdict are two readings of the same truth,
                // and a host must never see a loss over a rejected argument.
                if let Err(e) = gpu.prepare_frame(frame.width, frame.height).and_then(|()| gpu.render(&input)) {
                    if e.code == Code::DeviceLost && gpu.device_removed() {
                        faulted = Some(e);
                    } else {
                        return Err(e);
                    }
                } else {
                    frame.checksum = gpu.color_checksum();
                    frame.depth_checksum = gpu.depth_checksum();

                    // The same audit the reference runs: re-render the frame and
                    // compare. On hardware this is a determinism check, not a
                    // formality - a GPU that reshuffles its own scheduling must
                    // still produce the identical image twice.
                    if device.audit_every_frames > 0 && frame.index % device.audit_every_frames as u64 == 0 {
                        let second = gpu.render(&input)?;
                        if second.frame_index != frame.index || gpu.color_checksum() != frame.checksum {
                            device.stats.counters.audit_divergences += 1;
                            log_warn!("audit: frame {} rendered differently the second time", frame.index);
                        }
                    }
                    device.stats.shadows = gpu.snapshot().shadows;
                }
            }
            BackendKind::Null(null) => {
                null.begin_frame(frame.width, frame.height)?;
                for draw in frame.draws.iter() {
                    let count = if draw.index_count > 0 { draw.index_count } else { draw.vertex_count } as u64;
                    null.note_draw(count, count / 3);
                }
                null.submit()?;
                frame.checksum = null.last_checksum();
            }
        }

        if let Some(fault) = faulted {
            if !device.may_offload() {
                // The host left `RECONL_DOWNGRADE_TIER` clear: a fault is an
                // error it answers itself, and the frame is dropped exactly as
                // it was before the offload existed.
                return Err(fault);
            }
            let detail = format!("frame {}: {}", frame.index, fault.message.as_str());
            device.offload_on_fault(frame.index, &detail)?;
            // This frame, rendered on the tier that just took over. It is the
            // frame the GPU could not finish, not the next one: the recorded
            // draws are still here, so nothing about it is lost but GPU time.
            device.render_frame_on_soft(&mut frame)?;
        }

        let tier_now = match &device.backend {
            BackendKind::SoftCpu(d) => d.tier(),
            BackendKind::D3d11(d) => d.tier(),
            BackendKind::Null(d) => d.tier(),
        };
        if tier_now != device.tier {
            device.tier = tier_now;
            device.tier_reason = match &device.backend {
                BackendKind::SoftCpu(d) => d.tier_reason(),
                BackendKind::D3d11(d) => d.tier_reason(),
                BackendKind::Null(_) => TierReason::HostRequest,
            };
            device.stats.tier = device.tier;
            device.stats.tier_reason = device.tier_reason;
            device.stats.tier_reason_text.set(device.tier_reason.text());
            device.stats.counters.frames_since_tier_change = 0;
        }

        device.frame = frame;
        device.frame_state = FrameState::Submitted;
        if !fence.is_null() {
            // Validated above, before the commit point.
            let fence = unsafe { &mut *fence };
            fence.signaled = true;
            fence.frame_index = device.frame.index;
        }
        owner.keep = true;
        Ok(())
    })
}

fn vertex_count_upper(index_buffer: &BufferHandle, offset: u64, first_index: u32, index_count: u32) -> u64 {
    // The upper bound on the vertex index this draw can reach, so the vertex
    // buffer bound check is a bound and not a guess.
    let mut max = 0u64;
    let start = offset as usize + first_index as usize * 4;
    let bytes = index_buffer.bytes.as_slice();
    for i in 0..index_count as usize {
        let at = start + i * 4;
        if at + 4 > bytes.len() {
            break;
        }
        let value = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        if value as u64 > max {
            max = value as u64;
        }
    }
    max
}

fn read_mat4(data: &[u8; 64]) -> Mat4 {
    let mut out = [0.0f32; 16];
    for (i, slot) in out.iter_mut().enumerate() {
        let at = i * 4;
        *slot = f32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
    }
    out
}

/// Turns the recorded draws into rasteriser draws, allocated from `alloc`.
///
/// The host says *what to shade*; the backend supplies the lighting state. That
/// split is why this function only has to translate flags.
///
/// The allocator is a parameter, not a default: allocating this list from
/// anywhere but the host allocator means the draws vanish when the allocation is
/// refused, and the frame is presented empty while every call still returned OK.
/// The backend-facing description of one frame.
///
/// Built from the frame record and its draw items in one place because three
/// paths need it - Submit on either tier, and the re-render after a hardware
/// fault - and a frame that faulted must reach the other tier describing
/// exactly what the first tier was asked for.
fn frame_input<'a>(frame: &FrameRecord, draws: &'a [DrawItem<'a>]) -> FrameInput<'a> {
    FrameInput {
        frame_index: frame.index,
        width: frame.width,
        height: frame.height,
        camera_view: frame.camera_view,
        fov_y_deg: frame.fov_y_deg,
        aspect: frame.aspect,
        near: frame.near,
        far: frame.far,
        light_dir: frame.light_dir,
        light_hash: frame.light_hash,
        lights: frame.lights,
        shadow: frame.shadow,
        clear_color: frame.clear_color,
        clear_depth: frame.clear_depth,
        clear_color_enabled: frame.clear_color_on,
        clear_depth_enabled: frame.clear_depth_on,
        world_revision: frame.world_revision,
        static_geometry_revision: frame.static_revision,
        draws,
    }
}

/// Copies a tightly packed RGBA frame into the host's presentation buffer,
/// honouring the row pitch and the flip the present descriptor asked for.
fn blit(pixels: &mut [u8], tight: &[u8], width: u32, height: u32, pitch: u32, flip: u32) {
    for row in 0..height as usize {
        let source_row = if flip != 0 { height as usize - 1 - row } else { row };
        let src_at = source_row * width as usize * 4;
        let dst_at = row * pitch as usize;
        pixels[dst_at..dst_at + width as usize * 4].copy_from_slice(&tight[src_at..src_at + width as usize * 4]);
    }
}

fn build_draws<'a>(alloc: HostAlloc, records: &'a [DrawRecord]) -> Result<HostVec<DrawItem<'a>>> {
    let mut draws = HostVec::with_capacity(alloc, records.len())?;
    for record in records {
        let vertices: &'a [Vertex] = if record.indices.is_null() {
            unsafe { core::slice::from_raw_parts(record.vertices, record.vertex_count as usize) }
        } else {
            // Indexed draws still need the whole vertex buffer as a slice; the
            // bound check in Submit proved the indices stay inside it. Walking
            // the index list here would be a second, subtly different answer.
            let max = max_index(record);
            unsafe { core::slice::from_raw_parts(record.vertices, (max + 1) as usize) }
        };
        let indices: Option<&'a [u32]> = if record.indices.is_null() {
            None
        } else {
            Some(unsafe { core::slice::from_raw_parts(record.indices, record.index_count as usize) })
        };
        let shader = SurfaceShader {
            textured: record.textured,
            lit: record.lit,
            receives_shadow: record.receives_shadow,
            texture: None,
            lights: None,
            shadows: None,
            flip_normal: false,
        };
        draws.push(DrawItem {
            vertices,
            indices,
            transform: record.transform,
            model: record.model,
            pipeline: record.pipeline,
            shader: ShaderRef::Surface(shader),
            dynamic: record.dynamic,
            casts_shadow: record.casts_shadow,
        })?;
    }
    Ok(draws)
}

fn max_index(record: &DrawRecord) -> u32 {
    let mut max = 0u32;
    let indices = unsafe { core::slice::from_raw_parts(record.indices, record.index_count as usize) };
    for value in indices {
        if *value > max {
            max = *value;
        }
    }
    max
}

// -------------------------------------------------------------------- present

#[no_mangle]
pub unsafe extern "C" fn reconlPresent(device: *mut DeviceHandle, swapchain: *mut SwapchainHandle, desc: *mut ReconLPresentDesc) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if swapchain.is_null() {
            return err!(Code::InvalidArgument, "null swapchain");
        }
        child!(device, swapchain, Kind::Swapchain, "swapchain");
        if device.frame_state != FrameState::Submitted {
            return err!(Code::NoFrame, "there is no submitted frame to present");
        }
        // Past this point the frame is consumed: this call either presents it or
        // ends it. A present that fails for any later reason - an unreadable
        // descriptor, a buffer too small for the frame - therefore leaves the
        // device able to open the next frame rather than stuck in `Submitted`
        // until the host happens to present successfully. Same contract as a
        // failed submit, and the same counter.
        let mut owner = FrameOwner::new(device as *mut DeviceHandle);
        let swapchain = unsafe { &*swapchain };
        let (out_pixels, out_size, out_pitch, flip) = if desc.is_null() {
            (core::ptr::null_mut(), 0u64, 0u32, 0u32)
        } else {
            let d = unsafe {
                check_header::<ReconLPresentDesc>(
                    desc as *const reconl_core::StructHeader,
                    struct_type::PRESENT_DESC,
                    core::mem::size_of::<ReconLPresentDesc>() as u32,
                    "ReconLPresentDesc",
                )?
            };
            (d.out_pixels, d.out_pixels_size, d.out_row_pitch, d.flip)
        };

        let mut pixels = if !out_pixels.is_null() {
            Some(unsafe { core::slice::from_raw_parts_mut(out_pixels as *mut u8, out_size as usize) })
        } else {
            None
        };

        // A hardware fault during the readback is reported rather than returned:
        // the frame is already rendered when it happens, so it is re-rendered on
        // the reference tier and the host still gets its pixels (docs/offload.md).
        let mut faulted: Option<Error> = None;
        match &mut device.backend {
            BackendKind::SoftCpu(soft) => {
                let (width, height) = soft.frame_size();
                if let Some(pixels) = pixels.as_deref_mut() {
                    let pitch = if out_pitch == 0 { width * 4 } else { out_pitch };
                    let needed = (height.saturating_sub(1) as u64) * pitch as u64 + width as u64 * 4;
                    if out_size < needed {
                        return err!(Code::InvalidArgument, "the present buffer is too small for {}x{}", width, height);
                    }
                    let mut tight = vec![0u8; (width * height * 4) as usize];
                    soft.readback(&mut tight)?;
                    blit(pixels, &tight, width, height, pitch, flip);
                }
                soft.on_frame_end()?;
            }
            BackendKind::D3d11(gpu) => {
                let (width, height) = gpu.frame_size();
                let outcome = (|| -> Result<()> {
                    if let Some(pixels) = pixels.as_deref_mut() {
                        let pitch = if out_pitch == 0 { width * 4 } else { out_pitch };
                        let needed = (height.saturating_sub(1) as u64) * pitch as u64 + width as u64 * 4;
                        if out_size < needed {
                            return err!(Code::InvalidArgument, "the present buffer is too small for {}x{}", width, height);
                        }
                        let mut tight = vec![0u8; (width * height * 4) as usize];
                        gpu.readback(&mut tight)?;
                        blit(pixels, &tight, width, height, pitch, flip);
                    }
                    gpu.on_frame_end()
                })();
                if let Err(e) = outcome {
                    if e.code == Code::DeviceLost && gpu.device_removed() {
                        faulted = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
            BackendKind::Null(null) => {
                if let Some(pixels) = pixels.as_deref_mut() {
                    let pitch = if out_pitch == 0 { swapchain.width * 4 } else { out_pitch };
                    null.present(Some(pixels), pitch)?;
                } else {
                    null.present(None, 0)?;
                }
            }
        }

        if swapchain.present_to_memory && out_pixels.is_null() {
            return err!(Code::InvalidArgument, "this swapchain presents to memory: out_pixels is required");
        }

        if let Some(fault) = faulted {
            if !device.may_offload() {
                return Err(fault);
            }
            device.present_after_fault(&fault, pixels.as_deref_mut(), out_size, out_pitch, flip)?;
        }

        device.stats.counters.frames_presented += 1;
        device.stats.counters.frames_since_tier_change += 1;
        device.stats.last_frame = device.frame_numbers();
        let total = device.stats.last_frame.total_ns;
        device.stats.frames.push(total);
        device.frame_state = FrameState::Idle;
        owner.keep = true;

        // The ladder, driven by measurement, run with the frame closed so a
        // backend change here cannot be observed mid-present. A policy that
        // cannot rebuild a backend leaves the device where it is rather than
        // failing a present that has already succeeded.
        let (index, width, height) = (device.frame.index, device.frame.width, device.frame.height);
        if let Err(e) = device.apply_offload_policy(index, width, height) {
            log_warn!("the offload could not change backend: {}", e.message.as_str());
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------- stats

#[no_mangle]
pub unsafe extern "C" fn reconlGetStats(device: *mut DeviceHandle, out: *mut ReconLStats) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if out.is_null() {
            return err!(Code::InvalidArgument, "null stats output");
        }
        let mut stats = ReconLStats {
            base: reconl_core::StructHeader::new(core::mem::size_of::<ReconLStats>() as u32, struct_type::STATS),
            backend: device.backend_id(),
            tier: device.tier as u32,
            tier_reason: device.tier_reason as u32,
            tier_locked: if device.tier_locked { 1 } else { 0 },
            frames_presented: 0,
            frames_dropped: 0,
            downgrade_count: 0,
            downgrade_capacity: abi::RECONL_MAX_DOWNGRADES as u32,
            downgrades: [ReconLDowngrade::default(); abi::RECONL_MAX_DOWNGRADES],
            memory: memory_stats(device),
            shadows: shadow_stats(device),
            frame: frame_timing(device),
            caps: device.caps,
            failures: 0,
            safe_path_events: 0,
            audit_divergences: 0,
            last_result: result::OK,
            tier_reason_text: [0; abi::RECONL_MAX_MESSAGE],
            device_name: [0; abi::RECONL_MAX_NAME],
        };
        // The device owns these: it is the only thing that sees a present, a
        // dropped frame, or a failed call. Reading them off the backend left
        // `frames_presented` and `audit_divergences` stuck at zero, because the
        // reference backend never counts frames it did not receive.
        let counters = device.stats.counters;
        stats.frames_presented = counters.frames_presented;
        stats.frames_dropped = counters.frames_dropped;
        stats.failures = counters.failures;
        stats.safe_path_events = counters.safe_path_events;
        stats.audit_divergences = counters.audit_divergences;
        stats.last_result = counters.last_result.map(|c| c.as_i32()).unwrap_or(result::OK);
        set_str(&mut stats.tier_reason_text, device.tier_reason.text());
        set_str(&mut stats.device_name, device.device_name.as_str());
        let downgrades = device.downgrade_entries();
        stats.downgrade_count = downgrades.len() as u32;
        for (slot, d) in stats.downgrades.iter_mut().zip(downgrades.iter()) {
            slot.from = d.from as u32;
            slot.to = d.to as u32;
            slot.reason = d.reason as u32;
            slot.frame_index = d.frame_index;
            slot.at_ns = d.at_ns;
            set_str(&mut slot.detail, d.detail.as_str());
        }
        unsafe { out.write(stats) };
        Ok(())
    })
}

fn shadow_stats(device: &DeviceHandle) -> ReconLShadowStats {
    let s = device.shadow_counters();
    ReconLShadowStats {
        base: reconl_core::StructHeader::new(core::mem::size_of::<ReconLShadowStats>() as u32, struct_type::SHADOW_CONFIG),
        cascades_active: s.cascades_active,
        map_width: s.map_width,
        map_height: s.map_height,
        map_bytes: s.map_bytes.min(u32::MAX as u64) as u32,
        filter_active: s.filter_active as u32,
        filter_requested: s.filter_requested as u32,
        shadow_pass_ns: s.shadow_pass_ns,
        cascade_fit_ns: s.fit_ns,
        cache_bytes_read: s.cache_bytes_read,
        cache_bytes_hit: s.cache_bytes_hit,
        cache_hits: s.cache_hits,
        cache_misses: s.cache_misses,
        cache_corrupt: s.cache_corrupt,
        fail_safe_unshadowed: s.fail_safe_unshadowed,
        frozen_cascades: s.frozen_cascades,
        shadowed_lights: s.shadowed_lights,
        point_lights_capped: s.point_lights_capped,
    }
}

fn frame_timing(device: &DeviceHandle) -> ReconLFrameTiming {
    let f = device.frame_numbers();
    let timing = device.stats.frames;
    ReconLFrameTiming {
        base: reconl_core::StructHeader::new(core::mem::size_of::<ReconLFrameTiming>() as u32, struct_type::FRAME_DESC),
        frame_index: f.frame_index,
        total_ns: f.total_ns,
        shadow_ns: f.shadow_ns,
        raster_ns: f.raster_ns,
        bin_ns: f.bin_ns,
        upload_ns: f.upload_ns,
        spill_wait_ns: f.spill_wait_ns,
        min_ns: timing.min_ns,
        avg_ns: timing.avg_ns(),
        max_ns: timing.max_ns,
        tiles_total: f.tiles_total,
        tiles_rendered: f.tiles_rendered,
        triangles_in: f.triangles_in,
        triangles_binned: f.triangles_binned,
        triangles_culled: f.triangles_culled,
        pixels_shaded: f.pixels_shaded,
        worker_threads: f.worker_threads,
        resolution_scale: f.resolution_scale,
        reserved: 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn reconlResetStats(device: *mut DeviceHandle) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        device.stats.counters = Counters::default();
        device.stats.shadows = ShadowCounters::default();
        device.stats.frames.reset();
        device.stats.last_frame = FrameNumbers::default();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlGetLastError(device: *mut DeviceHandle, out: *mut ReconLErrorInfo) -> i32 {
    if out.is_null() {
        return result::INVALID_ARGUMENT;
    }
    let error = if device.is_null() {
        global_error().lock().ok().and_then(|slot| slot.clone())
    } else {
        match check_handle(device as *mut c_void, Kind::Device, "device") {
            Ok(header) => unsafe { (*header).device.as_ref() }.and_then(|d| d.last_error.clone()),
            Err(_) => None,
        }
    };
    let mut info = ReconLErrorInfo {
        base: reconl_core::StructHeader::new(core::mem::size_of::<ReconLErrorInfo>() as u32, struct_type::ERROR_INFO),
        result: result::OK,
        message: [0; abi::RECONL_MAX_MESSAGE],
        file: [0; abi::RECONL_MAX_PATH],
        line: 0,
        function_name_index: 0,
        function: [0; abi::RECONL_MAX_NAME],
    };        match error {
        Some(error) => {
            info.result = error.code.as_i32();
            set_str(&mut info.message, error.message.as_str());
            set_str(&mut info.file, error.file.as_str());
            set_str(&mut info.function, error.function.as_str());
            info.line = error.line;
        }
        None => {
            set_str(&mut info.message, "no error has been recorded");
        }
    }
    unsafe { out.write(info) };
    result::OK
}

// ---------------------------------------------------------------- tier tools

#[no_mangle]
pub unsafe extern "C" fn reconlRequestTier(device: *mut DeviceHandle, tier: u32, reason: u32) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        let requested = Tier::from_u32(tier);
        let reason = match reason {
            0 => TierReason::StartupProbe,
            1 => TierReason::HostRequest,
            2 => TierReason::AllocationOverBudget,
            3 => TierReason::FrameTimeOverTarget,
            4 => TierReason::DeviceRemoved,
            5 => TierReason::DeviceLost,
            6 => TierReason::MemoryPressure,
            7 => TierReason::DiskCacheFull,
            8 => TierReason::Build,
            _ => TierReason::NoGpuApi,
        };
        if requested == device.tier {
            return Ok(());
        }
        device.tier_locked = true;
        let mut current = device.tier;
        while current != requested && current < Tier::OutOfCore {
            let next = if current < requested { requested } else { current.step_down() };
            device.record_downgrade(current, next, reason, device.frame.index, "host requested a tier change");
            current = next;
        }
        device.tier = current;
        device.tier_reason = reason;
        device.stats.tier = current;
        device.stats.tier_reason = reason;
        device.stats.tier_reason_text.set(reason.text());
        device.stats.counters.frames_since_tier_change = 0;
        log_info!("tier requested: now {}", current.name());
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlRequestShadowFallback(device: *mut DeviceHandle, event_kind: u32, detail: *const i8) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        let detail = if detail.is_null() { String::from("host requested") } else { unsafe { cstr(detail) } };
        let event = reconl_core::tier::ShadowEvent::from_u32(event_kind);
        match event {
            reconl_core::tier::ShadowEvent::None => {
                return err!(Code::InvalidArgument, "event {} is not a shadow event", event_kind)
            }
            _ => {}
        }
        device.stats.counters.safe_path_events += 1;
        if let Some(soft) = device.softcpu_mut() {
            soft.note_shadow_event(event);
        }
        log_warn!("shadow fail-safe requested: {} ({})", event.name(), detail);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlAudit(device: *mut DeviceHandle, every_n_frames: u32, out_divergences: *mut u32) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        device.audit_every_frames = every_n_frames;
        if !out_divergences.is_null() {
            unsafe { *out_divergences = device.stats.counters.audit_divergences };
        }
        log_info!("audit interval set to {} frames", every_n_frames);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlConfigureShadows(device: *mut DeviceHandle, config: *const ReconLShadowConfig) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        if config.is_null() {
            device.shadow = reconl_backend_softcpu::ShadowRequest::default();
            return Ok(());
        }
        let config = unsafe {
            check_header::<ReconLShadowConfig>(
                config as *const reconl_core::StructHeader,
                struct_type::SHADOW_CONFIG,
                core::mem::size_of::<ReconLShadowConfig>() as u32,
                "ReconLShadowConfig",
            )?
        };
        device.shadow = shadow_request_from(Some(config));
        let plan = shadow_plan(device.tier, device.shadow.cascades, device.shadow.texel_budget_bytes, device.shadow.filter, device.caps);
        if plan.filter != plan.filter_requested {
            log_warn!(
                "shadow filter downgraded from {} to {} by tier {}",
                plan.filter_requested.name(),
                plan.filter.name(),
                device.tier.name()
            );
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn reconlBumpWorldRevision(device: *mut DeviceHandle, flags: u32) -> i32 {
    entry!(device, {
        let device = device_mut!(device);
        device.world.bump(flags);
        Ok(())
    })
}

// ------------------------------------------------------------------- lifetime

#[no_mangle]
pub unsafe extern "C" fn reconlRetain(handle: *mut c_void) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let header = handle as *mut HandleHeader;
    unsafe { (*header).refcount.fetch_add(1, Ordering::Relaxed) + 1 }
}

#[no_mangle]
pub unsafe extern "C" fn reconlRelease(handle: *mut c_void) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let header_ptr = handle as *mut HandleHeader;
    let kind = unsafe { Kind::from_u32((*header_ptr).kind) };
    let previous = unsafe { (*header_ptr).refcount.fetch_sub(1, Ordering::AcqRel) };
    if previous > 1 {
        return previous - 1;
    }
    let device_ptr = unsafe { (*header_ptr).device };
    match kind {
        Kind::Device => {
            let device = handle as *mut DeviceHandle;
            let alloc = unsafe { (*device).alloc };
            unsafe { handle_free(alloc, device) };
        }
        Kind::Buffer => {
            let ptr = handle as *mut BufferHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
        Kind::Texture => {
            let ptr = handle as *mut TextureHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
        Kind::Pipeline => {
            let ptr = handle as *mut PipelineHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
        Kind::Swapchain => {
            let ptr = handle as *mut SwapchainHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
        Kind::CommandList => {
            let ptr = handle as *mut CommandListHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
        Kind::Fence => {
            let ptr = handle as *mut FenceHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
    }
    0
}

unsafe fn device_alloc(device: *mut DeviceHandle) -> HostAlloc {
    if device.is_null() {
        return HostAlloc::system();
    }
    unsafe { (*device).alloc }
}

unsafe fn release_device_reference(device: *mut DeviceHandle) {
    if device.is_null() {
        return;
    }
    // SAFETY: the child held a reference, so the device is alive.
    unsafe { reconlRelease(device as *mut c_void) };
}

