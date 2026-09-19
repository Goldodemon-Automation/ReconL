//! The `#[repr(C)]` mirror of `include/reconl/reconl.h`.
//!
//! This module is the ABI, written twice: once in C so hosts can include it, and
//! once here so the implementation can read it. The two are kept honest by
//! `tests/abi_layout.rs`, which generates `_Static_assert`s from the Rust
//! `size_of`/`offset_of` values and asks a real C compiler to compile them
//! against the shipped header. If a field is added on one side, the C compiler
//! fails and CI stops - which is the only way an ABI stays a contract.
//!
//! Rules mirrored from the header:
//!   * every struct the library reads starts with `ReconLBase`
//!     (`struct_size`, `type`, `next`);
//!   * fields are never reordered, retyped or removed in a minor release;
//!   * `struct_size` smaller than [`ABIStruct::MIN_SIZE`] is refused with
//!     `RECONL_ERR_STRUCT_SIZE`, and a `type` that disagrees with the struct a
//!     pointer was passed as is refused with `RECONL_ERR_WRONG_STRUCT_TYPE`.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use core::ffi::c_void;
use reconl_core::{ABIStruct, StructHeader};

// ------------------------------------------------------------------ capacities

pub const RECONL_MAX_BACKENDS: usize = 16;
pub const RECONL_MAX_DOWNGRADES: usize = 16;
pub const RECONL_MAX_CASCADES: usize = 4;
pub const RECONL_MAX_LIGHTS: usize = 64;
pub const RECONL_MAX_TEXTURE_SLOTS: usize = 8;
pub const RECONL_MAX_ATTACHMENTS: usize = 4;
pub const RECONL_MAX_NAME: usize = 64;
pub const RECONL_MAX_MESSAGE: usize = 192;
pub const RECONL_MAX_PATH: usize = 260;
pub const RECONL_TIER_COUNT: usize = 5;

pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;
pub const ABI_VERSION: u32 = 100;

// --------------------------------------------------------------------- results

pub mod result {
    pub const OK: i32 = 0;
    pub const INVALID_ARGUMENT: i32 = -1;
    pub const OUT_OF_MEMORY: i32 = -2;
    pub const NOT_SUPPORTED: i32 = -3;
    pub const BACKEND_UNAVAILABLE: i32 = -4;
    pub const BUDGET_EXCEEDED: i32 = -5;
    pub const DEVICE_LOST: i32 = -6;
    pub const INVALID_HANDLE: i32 = -7;
    pub const STRUCT_SIZE: i32 = -8;
    pub const WRONG_STRUCT_TYPE: i32 = -9;
    pub const ABI_VERSION: i32 = -10;
    pub const FRAME_IN_PROGRESS: i32 = -11;
    pub const NO_FRAME: i32 = -12;
    pub const NOT_READY: i32 = -13;
    pub const IO: i32 = -14;
    pub const CORRUPT_CACHE: i32 = -15;
    pub const DEGRADED: i32 = -16;
    pub const PANIC: i32 = -17;
    pub const EMPTY_FRAME: i32 = -18;
}

// --------------------------------------------------------------- struct types

pub mod struct_type {
    pub const NONE: u32 = 0;
    pub const ALLOCATOR: u32 = 1;
    pub const PROBE_DESC: u32 = 2;
    pub const PROBE_INFO: u32 = 3;
    pub const MEMORY_BUDGET: u32 = 4;
    pub const DEVICE_LIMITS: u32 = 5;
    pub const DEVICE_DESC: u32 = 6;
    pub const BUFFER_DESC: u32 = 7;
    pub const TEXTURE_DESC: u32 = 8;
    pub const SAMPLER_DESC: u32 = 9;
    pub const PIPELINE_DESC: u32 = 10;
    pub const COMMAND_LIST_DESC: u32 = 11;
    pub const RENDER_PASS_DESC: u32 = 12;
    pub const SWAPCHAIN_DESC: u32 = 13;
    pub const FRAME_DESC: u32 = 14;
    pub const PRESENT_DESC: u32 = 15;
    pub const SHADOW_CONFIG: u32 = 16;
    pub const STATS: u32 = 17;
    pub const LIGHT: u32 = 18;
    pub const LIGHT_LIST: u32 = 19;
    pub const ERROR_INFO: u32 = 20;
    pub const CAMERA: u32 = 21;
    pub const SOFTCPU_DESC: u32 = 64;
    pub const NULL_DESC: u32 = 65;
}

// ------------------------------------------------------------------ backend ids

pub mod backend {
    pub const NONE: u32 = 0;
    pub const SOFT_CPU: u32 = 1;
    pub const NULL: u32 = 2;
    pub const D3D11: u32 = 3;
    pub const D3D12: u32 = 4;
    pub const VULKAN: u32 = 5;
    pub const GL: u32 = 6;
    pub const METAL: u32 = 7;
    pub const WEBGPU: u32 = 8;
    pub const WASM_WEBGL2: u32 = 9;
}

// ------------------------------------------------------------- downgrade flags

pub mod allow_downgrade {
    pub const NONE: u32 = 0;
    pub const TIER: u32 = 1 << 0;
    pub const SHADOWS: u32 = 1 << 1;
    pub const RESOLUTION: u32 = 1 << 2;
    pub const DISK: u32 = 1 << 3;
    pub const FREEZE_CACHE: u32 = 1 << 4;
    pub const ALL: u32 = 0x1F;
}

pub mod caps {
    pub const TEXTURES: u32 = 1 << 0;
    pub const MIPMAPS: u32 = 1 << 1;
    pub const SHADOWS: u32 = 1 << 2;
    pub const PCF_5X5: u32 = 1 << 3;
    pub const PCSS_LITE: u32 = 1 << 4;
    pub const DISK_SPILL: u32 = 1 << 5;
    pub const MULTITHREAD: u32 = 1 << 6;
    pub const OUT_OF_CORE: u32 = 1 << 7;
    pub const CACHED_CASCADE: u32 = 1 << 8;
    pub const SIMD_SSE2: u32 = 1 << 9;
    pub const SIMD_AVX2: u32 = 1 << 10;
    pub const SIMD_NEON: u32 = 1 << 11;
    pub const SIMD_WASM128: u32 = 1 << 12;
    pub const COMPUTE: u32 = 1 << 13;
    pub const PRESENT_TO_MEMORY: u32 = 1 << 14;
}

// ------------------------------------------------------------------ allocator

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLAllocator {
    pub alloc: Option<unsafe extern "C" fn(user: *mut c_void, size: usize, alignment: usize) -> *mut c_void>,
    pub realloc: Option<unsafe extern "C" fn(user: *mut c_void, ptr: *mut c_void, old_size: usize, new_size: usize, alignment: usize) -> *mut c_void>,
    pub free: Option<unsafe extern "C" fn(user: *mut c_void, ptr: *mut c_void, size: usize)>,
    pub user: *mut c_void,
}

impl ReconLAllocator {
    pub fn is_zeroed(&self) -> bool {
        self.alloc.is_none() && self.realloc.is_none() && self.free.is_none() && self.user.is_null()
    }
}

// ---------------------------------------------------------------------- probe

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLBackendProbe {
    pub backend: u32,
    pub name: [u8; RECONL_MAX_NAME],
    pub usable: i32,
    pub caps: u32,
    pub best_tier: u32,
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    pub max_cascades: u32,
    pub shadow_texel_budget: u64,
    pub device_name: [u8; RECONL_MAX_NAME],
    pub note: [u8; RECONL_MAX_MESSAGE],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLProbeDesc {
    pub base: StructHeader,
    pub flags: u32,
    pub spill_dir: *const i8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLProbeInfo {
    pub base: StructHeader,
    pub entry_count: u32,
    pub entries: [ReconLBackendProbe; RECONL_MAX_BACKENDS],
    pub recommended_tier: u32,
    pub recommended_backend: u32,
}

// ------------------------------------------------------------ limits and budget

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLDeviceLimits {
    pub base: StructHeader,
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    pub disk_bytes: u64,
    pub max_allocation_bytes: u64,
    pub shadow_texel_budget_bytes: u64,
    pub worker_threads_max: u32,
    pub tile_size_min: u32,
    pub max_cascades: u32,
    pub max_lights: u32,
    pub caps: u32,
    pub backend: u32,
    pub device_name: [u8; RECONL_MAX_NAME],
    pub driver: [u8; RECONL_MAX_NAME],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLMemoryBudget {
    pub base: StructHeader,
    pub vram_cap_bytes: u64,
    pub ram_cap_bytes: u64,
    pub disk_cap_bytes: u64,
    pub allow_disk_spill: u32,
    pub reserved: u32,
    pub spill_dir: *const i8,
}

// --------------------------------------------------------------------- device

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLDeviceDesc {
    pub base: StructHeader,
    pub backend_hint: u32,
    pub tier_hint: u32,
    pub allow_downgrade: u32,
    pub worker_threads: u32,
    pub target_frame_ms: u32,
    pub downgrade_after_frames: u32,
    pub seed: u32,
    pub flags: u32,
    pub budget: *const ReconLMemoryBudget,
    pub allocator: ReconLAllocator,
    pub backend_desc: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLSwapchainDesc {
    pub base: StructHeader,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub image_count: u32,
    pub present_to_memory: u32,
    pub depth_format: u32,
    pub flags: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLBufferDesc {
    pub base: StructHeader,
    pub size_bytes: u64,
    pub usage: u32,
    pub reserved: u32,
    pub data: *const c_void,
    pub data_size: u64,
    pub debug_name: *const i8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLTextureDesc {
    pub base: StructHeader,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub format: u32,
    pub usage: u32,
    pub reserved: u32,
    pub debug_name: *const i8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLTextureLevel {
    pub base: StructHeader,
    pub mip: u32,
    pub layer: u32,
    pub row_pitch: u32,
    pub row_count: u32,
    pub data: *const c_void,
    pub data_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLSamplerDesc {
    pub base: StructHeader,
    pub filter: u32,
    pub wrap_u: u32,
    pub wrap_v: u32,
    pub wrap_w: u32,
    pub max_anisotropy: u32,
    pub mip_lod_bias: f32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReconLVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

pub mod shading {
    pub const UNLIT: u32 = 0;
    pub const LAMBERT: u32 = 1;
    pub const TEXTURED: u32 = 2;
    pub const TEXTURED_LAMBERT: u32 = 3;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLPipelineDesc {
    pub base: StructHeader,
    pub shading: u32,
    pub blend: u32,
    pub cull: u32,
    pub depth_compare: u32,
    pub depth_write: u32,
    pub texture_slots: u32,
    pub texture_formats: [u32; RECONL_MAX_TEXTURE_SLOTS],
    pub receives_shadow: u32,
    pub casts_shadow: u32,
    pub flags: u32,
    pub reserved: u32,
    pub debug_name: *const i8,
}

// --------------------------------------------------------------------- lights

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLLight {
    pub base: StructHeader,
    pub r#type: u32,
    pub cast_shadow: u32,
    pub shadow_priority: u32,
    pub shadow_quality: u32,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub cone_inner_deg: f32,
    pub cone_outer_deg: f32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLLightList {
    pub base: StructHeader,
    pub count: u32,
    pub reserved: u32,
    pub lights: *const ReconLLight,
}

// -------------------------------------------------------------------- shadows

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLShadowConfig {
    pub base: StructHeader,
    pub enabled: u32,
    pub cascade_count: u32,
    pub texel_budget_bytes: u64,
    pub filter: u32,
    pub max_distance: f32,
    pub blend_band: f32,
    pub normal_bias: f32,
    pub depth_bias: f32,
    pub slope_bias: f32,
    pub resolution_scale: f32,
    pub allow_disk_cache: u32,
    pub freeze_static_cascade: u32,
    pub refresh_interval_frames: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ReconLShadowStats {
    pub base: StructHeader,
    pub cascades_active: u32,
    pub map_width: u32,
    pub map_height: u32,
    pub map_bytes: u32,
    pub filter_active: u32,
    pub filter_requested: u32,
    pub shadow_pass_ns: u64,
    pub cascade_fit_ns: u64,
    pub cache_bytes_read: u64,
    pub cache_bytes_hit: u64,
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub cache_corrupt: u32,
    pub fail_safe_unshadowed: u32,
    pub frozen_cascades: u32,
    pub shadowed_lights: u32,
    pub point_lights_capped: u32,
}

// --------------------------------------------------------------------- memory

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ReconLMemoryStats {
    pub base: StructHeader,
    pub ram_resident_bytes: u64,
    pub ram_budget_bytes: u64,
    pub ram_peak_bytes: u64,
    pub spill_resident_bytes: u64,
    pub spill_disk_bytes: u64,
    pub spill_disk_cap_bytes: u64,
    pub spill_evicted_bytes: u64,
    pub spill_cache_bytes: u64,
    pub spill_entries: u32,
    pub spill_evictions: u32,
    pub spill_compactions: u32,
    pub spill_recovered_entries: u32,
    pub spill_errors: u32,
    pub host_alloc_calls: u32,
    pub host_alloc_bytes: u64,
    pub host_free_calls: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ReconLFrameTiming {
    pub base: StructHeader,
    pub frame_index: u64,
    pub total_ns: u64,
    pub shadow_ns: u64,
    pub raster_ns: u64,
    pub bin_ns: u64,
    pub upload_ns: u64,
    pub spill_wait_ns: u64,
    pub min_ns: u64,
    pub avg_ns: u64,
    pub max_ns: u64,
    pub tiles_total: u32,
    pub tiles_rendered: u32,
    pub triangles_in: u32,
    pub triangles_binned: u32,
    pub triangles_culled: u32,
    pub pixels_shaded: u32,
    pub worker_threads: u32,
    pub resolution_scale: f32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLDowngrade {
    pub from: u32,
    pub to: u32,
    pub reason: u32,
    pub frame_index: u64,
    pub at_ns: u64,
    pub detail: [u8; RECONL_MAX_MESSAGE],
}

impl Default for ReconLDowngrade {
    fn default() -> Self {
        Self { from: 0, to: 0, reason: 0, frame_index: 0, at_ns: 0, detail: [0; RECONL_MAX_MESSAGE] }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLStatsDesc {
    pub base: StructHeader,
    pub include_downgrades: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLStats {
    pub base: StructHeader,
    pub backend: u32,
    pub tier: u32,
    pub tier_reason: u32,
    pub tier_locked: u32,
    pub frames_presented: u32,
    pub frames_dropped: u32,
    pub downgrade_count: u32,
    pub downgrade_capacity: u32,
    pub downgrades: [ReconLDowngrade; RECONL_MAX_DOWNGRADES],
    pub memory: ReconLMemoryStats,
    pub shadows: ReconLShadowStats,
    pub frame: ReconLFrameTiming,
    pub caps: u32,
    pub failures: u32,
    pub safe_path_events: u32,
    pub audit_divergences: u32,
    pub last_result: i32,
    pub tier_reason_text: [u8; RECONL_MAX_MESSAGE],
    pub device_name: [u8; RECONL_MAX_NAME],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLErrorInfo {
    pub base: StructHeader,
    pub result: i32,
    pub message: [u8; RECONL_MAX_MESSAGE],
    pub file: [u8; RECONL_MAX_PATH],
    pub line: u32,
    pub function_name_index: u32,
    pub function: [u8; RECONL_MAX_NAME],
}

// ----------------------------------------------------------------- command list

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLCommandListDesc {
    pub base: StructHeader,
    pub capacity_bytes: u32,
    pub reserved: u32,
    pub debug_name: *const i8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLColorAttachment {
    pub texture: *mut c_void,
    pub resolve: *mut c_void,
    pub mip: u32,
    pub layer: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLRenderPassDesc {
    pub base: StructHeader,
    pub color_count: u32,
    pub reserved: u32,
    pub color: [ReconLColorAttachment; RECONL_MAX_ATTACHMENTS],
    pub depth: *mut c_void,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub load_color: u32,
    pub load_depth: u32,
    pub clear_color: [f32; 4],
    pub clear_depth: f32,
    pub stencil_clear: u32,
    pub reserved2: u32,
}

// ---------------------------------------------------------------------- frame

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLCamera {
    pub base: StructHeader,
    pub view: [f32; 16],
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
    pub reserved: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLFrameDesc {
    pub base: StructHeader,
    pub width: u32,
    pub height: u32,
    pub seed: u32,
    pub reserved: u32,
    pub lights: *const ReconLLightList,
    pub shadows: *const ReconLShadowConfig,
    /// The frame camera, or null for the identity view and default frustum.
    /// This is the extension region: see the `ABIStruct` impl below.
    pub camera: *const ReconLCamera,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReconLPresentDesc {
    pub base: StructHeader,
    pub out_pixels: *mut c_void,
    pub out_pixels_size: u64,
    pub out_row_pitch: u32,
    pub out_format: u32,
    pub flip: u32,
}

// -------------------------------------------------------------------- markers

macro_rules! abi_struct {
    ($t:ty, $ty_id:expr) => {
        impl ABIStruct for $t {
            const STRUCT_TYPE: u32 = $ty_id;
            /// This revision of the library reads the whole struct, so an older
            /// caller with a shorter one is refused rather than misread.
            const MIN_SIZE: u32 = core::mem::size_of::<$t>() as u32;
        }
    };
}

abi_struct!(ReconLProbeDesc, struct_type::PROBE_DESC);
abi_struct!(ReconLProbeInfo, struct_type::PROBE_INFO);
abi_struct!(ReconLMemoryBudget, struct_type::MEMORY_BUDGET);
abi_struct!(ReconLDeviceLimits, struct_type::DEVICE_LIMITS);
abi_struct!(ReconLDeviceDesc, struct_type::DEVICE_DESC);
abi_struct!(ReconLBufferDesc, struct_type::BUFFER_DESC);
abi_struct!(ReconLTextureDesc, struct_type::TEXTURE_DESC);
abi_struct!(ReconLTextureLevel, struct_type::TEXTURE_DESC);
abi_struct!(ReconLSamplerDesc, struct_type::SAMPLER_DESC);
abi_struct!(ReconLPipelineDesc, struct_type::PIPELINE_DESC);
abi_struct!(ReconLCommandListDesc, struct_type::COMMAND_LIST_DESC);
abi_struct!(ReconLRenderPassDesc, struct_type::RENDER_PASS_DESC);
abi_struct!(ReconLSwapchainDesc, struct_type::SWAPCHAIN_DESC);
abi_struct!(ReconLPresentDesc, struct_type::PRESENT_DESC);
abi_struct!(ReconLShadowConfig, struct_type::SHADOW_CONFIG);
abi_struct!(ReconLStats, struct_type::STATS);
abi_struct!(ReconLLight, struct_type::LIGHT);
abi_struct!(ReconLLightList, struct_type::LIGHT_LIST);
abi_struct!(ReconLErrorInfo, struct_type::ERROR_INFO);
abi_struct!(ReconLCamera, struct_type::CAMERA);
abi_struct!(ReconLStatsDesc, struct_type::STATS);

/// `ReconLFrameDesc` grew a trailing `camera` field, so its prefix - the part
/// this revision reads unconditionally - stops after `shadows`. A caller whose
/// `struct_size` ends there is read with the identity camera, which is exactly
/// what hosts compiled before that field drew. Reading `camera` is gated on
/// `struct_size`, so the field is never touched past a shorter caller's struct.
impl ABIStruct for ReconLFrameDesc {
    const STRUCT_TYPE: u32 = struct_type::FRAME_DESC;
    const MIN_SIZE: u32 = Self::PREFIX_SIZE;
}

impl ReconLFrameDesc {
    /// The bytes every revision has carried: up to and including `shadows`.
    pub const PREFIX_SIZE: u32 =
        (core::mem::size_of::<Self>() - core::mem::size_of::<*const ReconLCamera>()) as u32;

    /// Whether the caller's struct reaches the `camera` field.
    pub fn has_camera(&self) -> bool {
        self.base.struct_size >= core::mem::size_of::<Self>() as u32
    }
}

// --------------------------------------------------------------- fixed buffers

/// A fixed-size NUL-terminated string field, written without allocating.
#[derive(Clone, Copy)]
pub struct FixedString<const N: usize> {
    pub bytes: [u8; N],
}

impl<const N: usize> FixedString<N> {
    pub const fn empty() -> Self {
        Self { bytes: [0u8; N] }
    }

    pub fn set(&mut self, text: &str) {
        let bytes = text.as_bytes();
        let n = bytes.len().min(N - 1);
        self.bytes[..n].copy_from_slice(&bytes[..n]);
        self.bytes[n] = 0;
        // Zero the tail so a shorter message followed by a longer one cannot
        // leave a stale suffix behind.
        for slot in self.bytes.iter_mut().skip(n + 1) {
            *slot = 0;
        }
    }
}

impl<const N: usize> Default for FixedString<N> {
    fn default() -> Self {
        Self::empty()
    }
}

pub fn set_str<const N: usize>(field: &mut [u8; N], text: &str) {
    let bytes = text.as_bytes();
    let n = bytes.len().min(N.saturating_sub(1));
    field[..n].copy_from_slice(&bytes[..n]);
    for slot in field.iter_mut().skip(n) {
        *slot = 0;
    }
}
