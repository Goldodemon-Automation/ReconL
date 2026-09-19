/* ReconL - the renderer that never says no.
 *
 * include/reconl/reconl.h - the canonical C11 ABI. This file is the product:
 * every language binding is a mechanical wrapper over exactly these symbols.
 *
 * Rules that hold for every declaration below:
 *   - C11, extern "C", header-first. Everything is callable from C with no wrapper.
 *   - Opaque, ref-counted handles; no global mutable singletons.
 *   - No exceptions, no panics, no unwinding across the boundary. Every entry
 *     point returns ReconLResult except the ones documented as infallible.
 *   - Versioned structs: `uint32_t struct_size; ReconLStructType type; const void* next;`
 *     is the first three members of every struct the library reads.
 *   - The device takes an allocator and never calls the C runtime allocator
 *     behind the host's back.
 *
 * Determinism contract (see docs/determinism.md): reversed-Z depth everywhere,
 * top-left fill rule, 8-bit subpixel vertex precision, no wall-clock input into
 * geometry. The software tier is the reference the GPU tiers are diffed against.
 */
#ifndef RECONL_H
#define RECONL_H

#include <stddef.h>
#include <stdint.h>

#include "reconl_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ linkage */

#if defined(_WIN32) && defined(RECONL_BUILD_SHARED)
#  define RECONL_API __declspec(dllexport)
#elif defined(_WIN32) && defined(RECONL_USE_SHARED)
#  define RECONL_API __declspec(dllimport)
#elif defined(__GNUC__) && defined(RECONL_BUILD_SHARED)
#  define RECONL_API __attribute__((visibility("default")))
#else
#  define RECONL_API
#endif

#define RECONL_CALL

/* Fixed capacities. These are ABI, not tunables: raising one is a breaking change. */
#define RECONL_MAX_BACKENDS        16u
#define RECONL_MAX_DOWNGRADES      16u
#define RECONL_MAX_CASCADES        4u
#define RECONL_MAX_LIGHTS          64u
#define RECONL_MAX_TEXTURE_SLOTS   8u
#define RECONL_MAX_ATTACHMENTS     4u
#define RECONL_MAX_VERTEX_STREAMS  4u
#define RECONL_MAX_CAPS            32u
#define RECONL_MAX_NAME            64u
#define RECONL_MAX_MESSAGE         192u
#define RECONL_MAX_PATH            260u
#define RECONL_TIER_COUNT          5u

/* ------------------------------------------------------------------- result */

/* The error taxonomy. Codes are ABI; the semantics below are the contract.
 *
 * What each class means, and what a host should do:
 *
 *   RECONL_ERR_DEVICE_LOST          The device is gone: a genuine removal,
 *                                  reset, hang or driver fault confirmed by
 *                                  the driver itself - never a rejected
 *                                  argument wearing a loss code. The host
 *                                  should destroy the device and create a
 *                                  new one; no call on this one will
 *                                  succeed. With RECONL_ALLOW_DOWNGRADE_TIER
 *                                  set, the device has already moved itself
 *                                  to the reference tier where the removal
 *                                  allowed it (docs/offload.md).
 *   RECONL_ERR_OUT_OF_MEMORY        The host allocator, the driver, or the
 *                                  device's budget refused a request. The
 *                                  device is healthy: free resources or
 *                                  shrink the request and retry.
 *   RECONL_ERR_INVALID_ARGUMENT     A descriptor, handle or value the call
 *                                  validates was rejected - including by the
 *                                  driver on a perfectly healthy device
 *                                  (an oversized target, a format it does
 *                                  not take). Fix the input and retry; the
 *                                  device needs no recreation.
 *   RECONL_ERR_NOT_SUPPORTED        Valid input the implementation does not
 *                                  offer. Do not retry.
 *   RECONL_ERR_BACKEND_UNAVAILABLE  A driver or OS failure whose cause the
 *                                  library cannot classify. The device's
 *                                  health is unknown; probe it (a trivial
 *                                  frame) before destroying it.
 *   other RECONL_ERR_* codes        No driver involvement: state-machine
 *                                  (FRAME_IN_PROGRESS, NO_FRAME), handle
 *                                  and struct validation, budget, IO.
 *                                  React as the call's own documentation
 *                                  says; the device is healthy.
 */
typedef enum ReconLResult {
    RECONL_OK = 0,
    RECONL_ERR_INVALID_ARGUMENT = -1,
    RECONL_ERR_OUT_OF_MEMORY = -2,
    RECONL_ERR_NOT_SUPPORTED = -3,
    RECONL_ERR_BACKEND_UNAVAILABLE = -4,
    RECONL_ERR_BUDGET_EXCEEDED = -5,
    RECONL_ERR_DEVICE_LOST = -6,
    RECONL_ERR_INVALID_HANDLE = -7,
    RECONL_ERR_STRUCT_SIZE = -8,
    RECONL_ERR_WRONG_STRUCT_TYPE = -9,
    RECONL_ERR_ABI_VERSION = -10,
    RECONL_ERR_FRAME_IN_PROGRESS = -11,
    RECONL_ERR_NO_FRAME = -12,
    RECONL_ERR_NOT_READY = -13,
    RECONL_ERR_IO = -14,
    RECONL_ERR_CORRUPT_CACHE = -15,
    RECONL_ERR_DEGRADED = -16, /* the call succeeded at a lower tier; see stats */
    RECONL_ERR_PANIC = -17,    /* an internal invariant broke; the safe path was taken */
    RECONL_ERR_EMPTY_FRAME = -18 /* the frame had no geometry and the policy requires some */
} ReconLResult;

/* ------------------------------------------------------------------- structs */

typedef enum ReconLStructType {
    RECONL_STRUCT_NONE = 0,
    RECONL_STRUCT_ALLOCATOR = 1,
    RECONL_STRUCT_PROBE_DESC = 2,
    RECONL_STRUCT_PROBE_INFO = 3,
    RECONL_STRUCT_MEMORY_BUDGET = 4,
    RECONL_STRUCT_DEVICE_LIMITS = 5,
    RECONL_STRUCT_DEVICE_DESC = 6,
    RECONL_STRUCT_BUFFER_DESC = 7,
    RECONL_STRUCT_TEXTURE_DESC = 8,
    RECONL_STRUCT_SAMPLER_DESC = 9,
    RECONL_STRUCT_PIPELINE_DESC = 10,
    RECONL_STRUCT_COMMAND_LIST_DESC = 11,
    RECONL_STRUCT_RENDER_PASS_DESC = 12,
    RECONL_STRUCT_SWAPCHAIN_DESC = 13,
    RECONL_STRUCT_FRAME_DESC = 14,
    RECONL_STRUCT_PRESENT_DESC = 15,
    RECONL_STRUCT_SHADOW_CONFIG = 16,
    RECONL_STRUCT_STATS = 17,
    RECONL_STRUCT_LIGHT = 18,
    RECONL_STRUCT_LIGHT_LIST = 19,
    RECONL_STRUCT_ERROR_INFO = 20,
    RECONL_STRUCT_CAMERA = 21,
    RECONL_STRUCT_SOFTCPU_DESC = 64,
    RECONL_STRUCT_NULL_DESC = 65,
    RECONL_STRUCT_D3D11_DESC = 66
} ReconLStructType;

typedef struct ReconLBase {
    uint32_t          struct_size; /* sizeof() of the whole struct, in bytes */
    ReconLStructType  type;        /* identifies the struct `next`/`this` points at */
    const void*       next;        /* extension chain, or NULL */
} ReconLBase;

/* ------------------------------------------------------------------ enums */

typedef enum ReconLBackendId {
    RECONL_BACKEND_NONE = 0,
    RECONL_BACKEND_SOFT_CPU = 1,
    RECONL_BACKEND_NULL = 2,
    RECONL_BACKEND_D3D11 = 3,
    RECONL_BACKEND_D3D12 = 4,
    RECONL_BACKEND_VULKAN = 5,
    RECONL_BACKEND_GL = 6,
    RECONL_BACKEND_METAL = 7,
    RECONL_BACKEND_WEBGPU = 8,
    RECONL_BACKEND_WASM_WEBGL2 = 9
} ReconLBackendId;

typedef enum ReconLTier {
    RECONL_TIER_T0_GPU_DISCRETE = 0, /* full VRAM                          */
    RECONL_TIER_T1_GPU_SHARED = 1,   /* VRAM + system RAM spill            */
    RECONL_TIER_T2_CPU_RAM = 2,      /* RAM, multi-threaded reference tier */
    RECONL_TIER_T3_CPU_THRIFTY = 3,  /* RAM, capped, scaled down           */
    RECONL_TIER_T4_OUT_OF_CORE = 4   /* RAM + disk-backed arena            */
} ReconLTier;

typedef enum ReconLTierReason {
    RECONL_TIER_REASON_STARTUP_PROBE = 0,
    RECONL_TIER_REASON_HOST_REQUEST = 1,
    RECONL_TIER_REASON_ALLOCATION_OVER_BUDGET = 2,
    RECONL_TIER_REASON_FRAME_TIME_OVER_TARGET = 3,
    RECONL_TIER_REASON_DEVICE_REMOVED = 4,
    RECONL_TIER_REASON_DEVICE_LOST = 5,
    RECONL_TIER_REASON_MEMORY_PRESSURE = 6,
    RECONL_TIER_REASON_DISK_CACHE_FULL = 7,
    RECONL_TIER_REASON_BUILD = 8,
    RECONL_TIER_REASON_NO_GPU_API = 9,
    /* The hardware was rebuilt after the settle window and measured inside
       the frame-time target: the device came back up the ladder. Distinct
       from STARTUP_PROBE, so a host auditing its tier log can tell how the
       device started from how it recovered (see the offload contract below). */
    RECONL_TIER_REASON_RECOVERY = 10
} ReconLTierReason;

typedef enum ReconLCaps {
    RECONL_CAP_NONE = 0u,
    RECONL_CAP_TEXTURES = 1u << 0,
    RECONL_CAP_MIPMAPS = 1u << 1,
    RECONL_CAP_SHADOWS = 1u << 2,
    RECONL_CAP_PCF_5X5 = 1u << 3,
    RECONL_CAP_PCSS_LITE = 1u << 4,
    RECONL_CAP_DISK_SPILL = 1u << 5,
    RECONL_CAP_MULTITHREAD = 1u << 6,
    RECONL_CAP_OUT_OF_CORE = 1u << 7,
    RECONL_CAP_CACHED_CASCADE = 1u << 8,
    RECONL_CAP_SIMD_SSE2 = 1u << 9,
    RECONL_CAP_SIMD_AVX2 = 1u << 10,
    RECONL_CAP_SIMD_NEON = 1u << 11,
    RECONL_CAP_SIMD_WASM128 = 1u << 12,
    RECONL_CAP_COMPUTE = 1u << 13,
    RECONL_CAP_PRESENT_TO_MEMORY = 1u << 14
} ReconLCaps;

typedef enum ReconLAllowDowngrade {
    RECONL_ALLOW_DOWNGRADE_NONE = 0u,
    RECONL_ALLOW_DOWNGRADE_TIER = 1u << 0,      /* re-create the tier below; see
                                                   the offload contract at the
                                                   frame calls: without it a
                                                   fault or an overload never
                                                   moves the device              */
    RECONL_ALLOW_DOWNGRADE_SHADOWS = 1u << 1,   /* fewer cascades / smaller maps */
    RECONL_ALLOW_DOWNGRADE_RESOLUTION = 1u << 2,/* resolution scale              */
    RECONL_ALLOW_DOWNGRADE_DISK = 1u << 3,      /* spill to disk, not just RAM   */
    RECONL_ALLOW_DOWNGRADE_FREEZE_CACHE = 1u << 4,
    RECONL_ALLOW_DOWNGRADE_ALL = 0x1Fu
} ReconLAllowDowngrade;

typedef enum ReconLLogLevel {
    RECONL_LOG_OFF = 0,
    RECONL_LOG_ERROR = 1,
    RECONL_LOG_WARN = 2,
    RECONL_LOG_INFO = 3,
    RECONL_LOG_DEBUG = 4,
    RECONL_LOG_TRACE = 5
} ReconLLogLevel;

typedef enum ReconLFormat {
    RECONL_FORMAT_UNKNOWN = 0,
    RECONL_FORMAT_R8G8B8A8_UNORM = 1,
    RECONL_FORMAT_B8G8R8A8_UNORM = 2,
    RECONL_FORMAT_R8G8B8A8_SRGB = 3,
    RECONL_FORMAT_R32_FLOAT = 4,
    RECONL_FORMAT_R32G32_FLOAT = 5,
    RECONL_FORMAT_R32G32B32_FLOAT = 6,
    RECONL_FORMAT_R32G32B32A32_FLOAT = 7,
    RECONL_FORMAT_D32_FLOAT = 8
} ReconLFormat;

typedef enum ReconLBufferUsage {
    RECONL_BUFFER_VERTEX = 1u << 0,
    RECONL_BUFFER_INDEX = 1u << 1,
    RECONL_BUFFER_UNIFORM = 1u << 2,
    RECONL_BUFFER_UPLOAD = 1u << 3,
    RECONL_BUFFER_READBACK = 1u << 4,
    RECONL_BUFFER_STATIC = 1u << 5, /* eligible for the disk cache arena */
    RECONL_BUFFER_DYNAMIC = 1u << 6
} ReconLBufferUsage;

typedef enum ReconLTextureUsage {
    RECONL_TEXTURE_SAMPLED = 1u << 0,
    RECONL_TEXTURE_RENDER_TARGET = 1u << 1,
    RECONL_TEXTURE_DEPTH_STENCIL = 1u << 2,
    RECONL_TEXTURE_SHADOW_MAP = 1u << 3,
    RECONL_TEXTURE_HOST_READBACK = 1u << 4,
    RECONL_TEXTURE_MIPMAPPED = 1u << 5,
    RECONL_TEXTURE_STATIC = 1u << 6
} ReconLTextureUsage;

typedef enum ReconLIndexFormat {
    RECONL_INDEX_UINT16 = 0,
    RECONL_INDEX_UINT32 = 1
} ReconLIndexFormat;

typedef enum ReconLSamplerFilter {
    RECONL_FILTER_NEAREST = 0,
    RECONL_FILTER_LINEAR = 1,
    RECONL_FILTER_NEAREST_MIP_LINEAR = 2,
    RECONL_FILTER_LINEAR_MIP_LINEAR = 3
} ReconLSamplerFilter;

typedef enum ReconLSamplerWrap {
    RECONL_WRAP_REPEAT = 0,
    RECONL_WRAP_CLAMP = 1,
    RECONL_WRAP_MIRROR = 2
} ReconLSamplerWrap;

typedef enum ReconLBlendMode {
    RECONL_BLEND_OPAQUE = 0,
    RECONL_BLEND_ALPHA = 1,
    RECONL_BLEND_ADDITIVE = 2,
    RECONL_BLEND_MULTIPLY = 3
} ReconLBlendMode;

typedef enum ReconLCompareFunc {
    RECONL_COMPARE_LESS = 0,   /* standard depth                */
    RECONL_COMPARE_GREATER = 1 /* reversed-Z: this is the default */
} ReconLCompareFunc;

typedef enum ReconLCullMode {
    RECONL_CULL_NONE = 0,
    RECONL_CULL_BACK = 1,
    RECONL_CULL_FRONT = 2
} ReconLCullMode;

typedef enum ReconLLightType {
    RECONL_LIGHT_DIRECTIONAL = 0,
    RECONL_LIGHT_SPOT = 1,
    RECONL_LIGHT_POINT = 2
} ReconLLightType;

typedef enum ReconLShadowFilter {
    RECONL_SHADOW_FILTER_HARD = 0,
    RECONL_SHADOW_FILTER_PCF3X3 = 1,
    RECONL_SHADOW_FILTER_PCF5X5 = 2,
    RECONL_SHADOW_FILTER_PCSS_LITE = 3 /* T0/T1 only; an approximation, never "soft" */
} ReconLShadowFilter;

typedef enum ReconLFrameState {
    RECONL_FRAME_IDLE = 0,
    RECONL_FRAME_OPEN = 1,
    RECONL_FRAME_SUBMITTED = 2,
    RECONL_FRAME_PRESENTED = 3
} ReconLFrameState;

/* Shadow fail-safe events. A stale/corrupt cache renders unshadowed and counts. */
typedef enum ReconLShadowEvent {
    RECONL_SHADOW_EVENT_NONE = 0,
    RECONL_SHADOW_EVENT_CASCADE_DROPPED = 1,
    RECONL_SHADOW_EVENT_MAP_BUDGET_CLAMPED = 2,
    RECONL_SHADOW_EVENT_CACHE_HIT = 3,
    RECONL_SHADOW_EVENT_CACHE_MISS = 4,
    RECONL_SHADOW_EVENT_CACHE_CORRUPT = 5,
    RECONL_SHADOW_EVENT_FALLBACK_UNSHADOWED = 6,
    RECONL_SHADOW_EVENT_FILTER_DOWNGRADED = 7,
    RECONL_SHADOW_EVENT_FROZEN_CASCADE = 8
} ReconLShadowEvent;

/* ---------------------------------------------------------------- allocator */

typedef struct ReconLAllocator {
    void* (*alloc)(void* user, size_t size, size_t alignment);
    void* (*realloc)(void* user, void* ptr, size_t old_size, size_t new_size, size_t alignment);
    void  (*free)(void* user, void* ptr, size_t size);
    void* user;
} ReconLAllocator;

/* -------------------------------------------------------------------- probe */

typedef struct ReconLBackendProbe {
    ReconLBackendId backend;
    char            name[RECONL_MAX_NAME];
    int             usable;          /* 1 if a device on this backend would start   */
    uint32_t        caps;            /* ReconLCaps bitmask                          */
    ReconLTier      best_tier;
    uint64_t        vram_bytes;
    uint64_t        ram_bytes;
    uint32_t        max_cascades;
    uint64_t        shadow_texel_budget; /* bytes                                   */
    char            device_name[RECONL_MAX_NAME];
    char            note[RECONL_MAX_MESSAGE]; /* why it is unusable, if it is not    */
} ReconLBackendProbe;

typedef struct ReconLProbeDesc {
    ReconLBase   base;
    uint32_t     flags;
    const char*  spill_dir;   /* read-only probe of the cache directory, may be NULL */
} ReconLProbeDesc;

typedef struct ReconLProbeInfo {
    ReconLBase         base;
    uint32_t           entry_count;
    ReconLBackendProbe entries[RECONL_MAX_BACKENDS]; /* indexed [0, entry_count)    */
    ReconLTier         recommended_tier;
    uint32_t           recommended_backend; /* ReconLBackendId of the best entry   */
} ReconLProbeInfo;

/* ------------------------------------------------------- limits and budget  */

/* One gate, one ceiling, before any memory is touched.
 *
 * Every allocation ReconL makes passes a size check first: a frame's colour and
 * depth targets, the cascade maps, a texture's whole level chain, a buffer, a
 * command list's slots, and a swapchain's implied images. A request above
 * ReconLDeviceLimits::max_allocation_bytes is refused with
 * RECONL_ERR_BUDGET_EXCEEDED and *nothing is allocated* - the check runs before
 * the host allocator, the driver or the spill arena is called, so an absurd
 * descriptor costs the host an error and not a machine that starts swapping.
 *
 * The ceiling is the lower of the host's ReconLMemoryBudget::ram_cap_bytes and
 * the library's own 512 MiB single-allocation limit, and it is reported back in
 * ReconLDeviceLimits::max_allocation_bytes. It bounds one allocation, not the
 * total, so a large-but-affordable request (a 4096x4096 frame, a 4K swapchain,
 * a full 4096x4096 texture) still succeeds; what the RAM cap bounds is the sum,
 * which now counts buffers, textures and command lists as well as frames and
 * cascade maps.
 *
 * Such a refusal is an argument error, not a device failure: the handle stays
 * valid and the call can be retried with a smaller descriptor. A refused
 * reconlBeginFrame leaves the device idle, per the frame contract below. */

typedef struct ReconLDeviceLimits {
    ReconLBase      base;
    uint64_t        vram_bytes;
    uint64_t        ram_bytes;               /* process-addressable, from the OS     */
    uint64_t        disk_bytes;              /* free bytes where the spill arena lives */
    uint64_t        max_allocation_bytes;    /* enforced single-allocation ceiling   */
    uint64_t        shadow_texel_budget_bytes;
    uint32_t        worker_threads_max;
    uint32_t        tile_size_min;
    uint32_t        max_cascades;
    uint32_t        max_lights;
    uint32_t        caps;                    /* ReconLCaps                           */
    ReconLBackendId backend;
    char            device_name[RECONL_MAX_NAME];
    char            driver[RECONL_MAX_NAME];
} ReconLDeviceLimits;

typedef struct ReconLMemoryBudget {
    ReconLBase  base;
    uint64_t    vram_cap_bytes;   /* 0 = unlimited within the device cap              */
    uint64_t    ram_cap_bytes;    /* 0 = unlimited; bounds the total, textures included */
    uint64_t    disk_cap_bytes;   /* 0 = no disk use at all                           */
    uint32_t    allow_disk_spill; /* opt-in: disk is written only when this is 1      */
    uint32_t    reserved;
    const char* spill_dir;        /* NULL -> RECONL_SPILL_DIR -> per-user cache dir    */
} ReconLMemoryBudget;

/* ------------------------------------------------------------------- device */

typedef struct ReconLDeviceDesc {
    ReconLBase          base;
    ReconLBackendId     backend_hint;    /* RECONL_BACKEND_NONE = let ReconL choose   */
    ReconLTier          tier_hint;       /* requested tier; ReconL may step down      */
    uint32_t            allow_downgrade; /* ReconLAllowDowngrade bitmask (0 = none)   */
    uint32_t            worker_threads;  /* 0 = choose; hard cap honoured             */
    uint32_t            target_frame_ms; /* downgrade trigger input                   */
    uint32_t            downgrade_after_frames; /* consecutive over-budget frames     */
    uint32_t            seed;            /* determinism seed, never wall-clock        */
    uint32_t            flags;
    ReconLMemoryBudget* budget;          /* optional, may be NULL                     */
    ReconLAllocator     allocator;       /* required; zeroed allocator = refuse start */
    const void*         backend_desc;    /* ReconLBase-derived backend descriptor     */
} ReconLDeviceDesc;

typedef struct ReconLSwapchainDesc {
    ReconLBase      base;
    uint32_t        width;
    uint32_t        height;
    ReconLFormat    format;
    uint32_t        image_count;      /* 1..3                                     */
    uint32_t        present_to_memory;/* 1 = headless; pixels come back in Present */
    uint32_t        depth_format;     /* ReconLFormat, 0 = none                   */
    uint32_t        flags;            /* 1 = allow resolution-scale downgreduction */
    uint32_t        reserved;
} ReconLSwapchainDesc;

typedef struct ReconLBufferDesc {
    ReconLBase      base;
    uint64_t        size_bytes;
    uint32_t        usage;          /* ReconLBufferUsage bitmask, required          */
    uint32_t        reserved;
    const void*     data;           /* optional initial contents                    */
    uint64_t        data_size;      /* must be <= size_bytes when data is non-NULL  */
    const char*     debug_name;
} ReconLBufferDesc;

typedef struct ReconLTextureDesc {
    ReconLBase      base;
    uint32_t        width;
    uint32_t        height;
    uint32_t        mip_levels;    /* 0 = full chain; requires MIPMAPPED usage      */
    uint32_t        array_layers;  /* 6 = cube; point light shadows                 */
    ReconLFormat    format;
    uint32_t        usage;         /* ReconLTextureUsage bitmask                    */
    uint32_t        reserved;
    const char*     debug_name;
} ReconLTextureDesc;

/* One mip level of one array slice. Rows are tightly packed. */
typedef struct ReconLTextureLevel {
    ReconLBase  base;
    uint32_t    mip;
    uint32_t    layer;
    uint32_t    row_pitch;   /* bytes per row                              */
    uint32_t    row_count;   /* rows in this level                          */
    const void* data;
    uint64_t    data_size;
} ReconLTextureLevel;

typedef struct ReconLSamplerDesc {
    ReconLBase          base;
    ReconLSamplerFilter filter;
    ReconLSamplerWrap   wrap_u;
    ReconLSamplerWrap   wrap_v;
    ReconLSamplerWrap   wrap_w;
    uint32_t            max_anisotropy; /* 1 = off                          */
    float               mip_lod_bias;
    uint32_t            reserved;
} ReconLSamplerDesc;

/* Vertex format the milestone-1 pipeline understands: position, normal, uv, colour. */
typedef struct ReconLVertex {
    float position[3];
    float normal[3];
    float uv[2];
    float color[4];
} ReconLVertex;

typedef enum ReconLPipelineShading {
    RECONL_SHADING_UNLIT = 0,      /* colour/vertex colours only                 */
    RECONL_SHADING_LAMBERT = 1,    /* Lambert + shadow lookup                    */
    RECONL_SHADING_TEXTURED = 2,   /* texture * vertex colour                    */
    RECONL_SHADING_TEXTURED_LAMBERT = 3
} ReconLPipelineShading;

typedef struct ReconLPipelineDesc {
    ReconLBase             base;
    ReconLPipelineShading  shading;
    ReconLBlendMode        blend;
    ReconLCullMode         cull;
    ReconLCompareFunc      depth_compare;   /* reversed-Z default: GREATER        */
    uint32_t               depth_write;
    uint32_t               texture_slots;   /* 0..RECONL_MAX_TEXTURE_SLOTS        */
    ReconLFormat           texture_formats[RECONL_MAX_TEXTURE_SLOTS];
    uint32_t               receives_shadow; /* participates in shadow lookup      */
    uint32_t               casts_shadow;
    uint32_t               flags;           /* 1 = twosided shadow render        */
    uint32_t               reserved;
    const char*            debug_name;
} ReconLPipelineDesc;

/* ------------------------------------------------------------------ lights */

typedef struct ReconLLight {
    ReconLBase      base;
    ReconLLightType type;
    uint32_t        cast_shadow;
    uint32_t        shadow_priority;  /* higher wins the texel budget              */
    uint32_t        shadow_quality;   /* 0..3 hint; tiers clamp it                 */
    float           position[3];
    float           direction[3];
    float           color[3];
    float           intensity;
    float           range;
    float           cone_inner_deg;
    float           cone_outer_deg;
    uint32_t        reserved;
} ReconLLight;

typedef struct ReconLLightList {
    ReconLBase  base;
    uint32_t    count;
    uint32_t    reserved;
    const ReconLLight* lights; /* array of `count` ReconLLight                  */
} ReconLLightList;

/* ---------------------------------------------------------------- shadows  */

typedef struct ReconLShadowConfig {
    ReconLBase         base;
    uint32_t           enabled;
    uint32_t           cascade_count;      /* 1..4, clamped per tier             */
    uint64_t           texel_budget_bytes; /* map memory, not "quality"          */
    ReconLShadowFilter filter;
    float              max_distance;       /* blend to unshadowed beyond this    */
    float              blend_band;         /* cascade crossfade width, world units */
    /* Shadow bias: the host pinning the policy. All three zero - the default,
     * and what a host that predates these fields sends - means the library picks
     * from the tier, the map size and the filter, as its bias table documents.
     * Any non-zero value means the host owns all three, and they are used as
     * given rather than rescaled, so pinning them is what makes one scene render
     * the same image on two tiers. Units are the ones the table documents:
     * texels for normal_bias, reversed-Z depth units for depth_bias (already
     * scaled for the map size it will be used at) and texel footprints for
     * slope_bias. */
    float              normal_bias;        /* documented, not magical            */
    float              depth_bias;
    float              slope_bias;
    float              resolution_scale;   /* map size multiplier, 0.25..1.0     */
    uint32_t           allow_disk_cache;
    uint32_t           freeze_static_cascade; /* T3: refresh every N frames      */
    uint32_t           refresh_interval_frames;
    uint32_t           reserved;
} ReconLShadowConfig;

typedef struct ReconLShadowStats {
    ReconLBase  base;
    uint32_t    cascades_active;
    uint32_t    map_width;
    uint32_t    map_height;
    uint32_t    map_bytes;
    ReconLShadowFilter filter_active;
    ReconLShadowFilter filter_requested;
    uint64_t    shadow_pass_ns;
    uint64_t    cascade_fit_ns;
    uint64_t    cache_bytes_read;
    uint64_t    cache_bytes_hit;
    uint32_t    cache_hits;
    uint32_t    cache_misses;
    uint32_t    cache_corrupt;
    uint32_t    fail_safe_unshadowed;
    uint32_t    frozen_cascades;
    uint32_t    shadowed_lights;
    uint32_t    point_lights_capped;
} ReconLShadowStats;

/* ------------------------------------------------------------------ memory */

typedef struct ReconLMemoryStats {
    ReconLBase  base;
    uint64_t    ram_resident_bytes;   /* ReconL-owned, in RAM as far as ReconL knows */
    uint64_t    ram_budget_bytes;
    uint64_t    ram_peak_bytes;
    uint64_t    spill_resident_bytes; /* currently mapped/spilled                    */
    uint64_t    spill_disk_bytes;
    uint64_t    spill_disk_cap_bytes;
    uint64_t    spill_evicted_bytes;
    uint64_t    spill_cache_bytes;    /* static-cascade cache on disk                */
    uint32_t    spill_entries;
    uint32_t    spill_evictions;
    uint32_t    spill_compactions;
    uint32_t    spill_recovered_entries; /* torn entries dropped on open             */
    uint32_t    spill_errors;
    uint32_t    host_alloc_calls;
    uint64_t    host_alloc_bytes;
    uint32_t    host_free_calls;
    uint32_t    reserved;
} ReconLMemoryStats;

typedef struct ReconLFrameTiming {
    ReconLBase  base;
    uint64_t    frame_index;
    uint64_t    total_ns;
    uint64_t    shadow_ns;
    uint64_t    raster_ns;
    uint64_t    bin_ns;
    uint64_t    upload_ns;
    uint64_t    spill_wait_ns;
    uint64_t    min_ns;
    uint64_t    avg_ns;
    uint64_t    max_ns;
    uint32_t    tiles_total;
    uint32_t    tiles_rendered;
    uint32_t    triangles_in;
    uint32_t    triangles_binned;
    uint32_t    triangles_culled;
    uint32_t    pixels_shaded;
    uint32_t    worker_threads;
    float       resolution_scale;
    uint32_t    reserved;
} ReconLFrameTiming;

typedef struct ReconLDowngrade {
    ReconLTier       from;
    ReconLTier       to;
    ReconLTierReason reason;
    uint64_t         frame_index;
    uint64_t         at_ns;
    char             detail[RECONL_MAX_MESSAGE];
} ReconLDowngrade;

typedef struct ReconLStatsDesc {
    ReconLBase  base;
    uint32_t    include_downgrades; /* 1 = fill the downgrade ring                 */
    uint32_t    reserved;
} ReconLStatsDesc;

typedef struct ReconLStats {
    ReconLBase          base;
    ReconLBackendId     backend;
    ReconLTier          tier;
    ReconLTierReason    tier_reason;
    uint32_t            tier_locked;      /* 1 = host pinned the tier              */
    uint32_t            frames_presented;
    uint32_t            frames_dropped;
    uint32_t            downgrade_count;  /* total ever, not just in the ring      */
    uint32_t            downgrade_capacity;
    ReconLDowngrade     downgrades[RECONL_MAX_DOWNGRADES];
    ReconLMemoryStats   memory;
    ReconLShadowStats   shadows;
    ReconLFrameTiming   frame;
    uint32_t            caps;
    uint32_t            failures;         /* calls that returned a non-OK code     */
    uint32_t            safe_path_events; /* fail-safe over fast-fail (§13)        */
    uint32_t            audit_divergences;
    ReconLResult        last_result;
    char                tier_reason_text[RECONL_MAX_MESSAGE];
    char                device_name[RECONL_MAX_NAME];
} ReconLStats;

typedef struct ReconLErrorInfo {
    ReconLBase  base;
    ReconLResult result;
    char        message[RECONL_MAX_MESSAGE];
    char        file[RECONL_MAX_PATH];
    uint32_t    line;
    uint32_t    function_name_index; /* 0 = unknown */
    char        function[RECONL_MAX_NAME];
} ReconLErrorInfo;

/* ------------------------------------------------------------------ handles */

typedef struct ReconLDevice     ReconLDevice;
typedef struct ReconLBuffer     ReconLBuffer;
typedef struct ReconLTexture    ReconLTexture;
typedef struct ReconLPipeline   ReconLPipeline;
typedef struct ReconLSwapchain  ReconLSwapchain;
typedef struct ReconLCommandList ReconLCommandList;
typedef struct ReconLFence      ReconLFence;

/* ---------------------------------------------------------------- commands */

/* One command's slot in a command list. A command list is a fixed array of
 * these, allocated at creation and never grown, so a host that knows how many
 * commands it will record can size the list exactly:
 *
 *     capacity_bytes = commands * RECONL_COMMAND_BYTES
 *
 * Recording past the capacity fails with RECONL_ERR_BUDGET_EXCEEDED, which in
 * this one case means "the list is full" rather than "the device is out of
 * memory" - the list is a reservation like any other, and it is counted against
 * the device budget from creation. */
#define RECONL_COMMAND_BYTES 64

typedef struct ReconLCommandListDesc {
    ReconLBase  base;
    uint32_t    capacity_bytes; /* 0 = the library's default (1024 commands);
                                 * the list never grows inside a frame, and a
                                 * full list refuses the next command rather
                                 * than reallocating mid-frame */
    uint32_t    reserved;
    const char* debug_name;
} ReconLCommandListDesc;

typedef struct ReconLColorAttachment {
    ReconLTexture* texture;
    ReconLTexture* resolve;      /* unused in v0.1, reserved */
    uint32_t       mip;
    uint32_t       layer;
} ReconLColorAttachment;

typedef struct ReconLRenderPassDesc {
    ReconLBase            base;
    uint32_t              color_count;
    uint32_t              reserved;
    ReconLColorAttachment color[RECONL_MAX_ATTACHMENTS];
    ReconLTexture*        depth;
    uint32_t              viewport_width;
    uint32_t              viewport_height;
    uint32_t              load_color;      /* 1 = clear on load        */
    uint32_t              load_depth;      /* 1 = clear on load        */
    float                 clear_color[4];
    float                 clear_depth;     /* reversed-Z: clear to 0.0 */
    uint32_t              stencil_clear;
    uint32_t              reserved2;
} ReconLRenderPassDesc;

/* ---------------------------------------------------------------- frame    */

/* ------------------------------------------------------------------ camera */

/* The frame's camera.
 *
 * This exists because a vertex transform cannot be un-multiplied. Push
 * constant slot 0 carries `view * projection` - that is what draws geometry,
 * and its meaning is unchanged. The shadow system needs the *view* matrix on
 * its own: fitting shadow cascades takes the world-space corners of the view
 * frustum, which needs a rigid world-to-camera transform, and the camera-space
 * depth a shadow lookup is gated on is read from the same matrix. Neither can
 * be recovered from a projection multiplied into a view.
 *
 * Supplying a camera does not change how vertices are transformed, and a host
 * that never sets one renders pixel-for-pixel as it did before.
 *
 * `view` must be rigid: a rotation and a translation. No scale, no shear, no
 * projection, and no dropped look direction - the fit inverts it as an
 * orthonormal basis. */
typedef struct ReconLCamera {
    ReconLBase base;
    float      view[16];   /* world -> camera space, column-major             */
    float      fov_y_deg;  /* vertical fov of the projection in slot 0        */
    float      near;       /* near plane, in camera-space units               */
    float      far;        /* far plane, in camera-space units                */
    float      reserved;
} ReconLCamera;

typedef struct ReconLFrameDesc {
    ReconLBase          base;
    uint32_t            width;
    uint32_t            height;
    uint32_t            seed;
    uint32_t            reserved;
    const ReconLLightList* lights;
    const ReconLShadowConfig* shadows; /* per-frame override, may be NULL */
    /* The frame camera, or NULL for an identity view and the default frustum
     * (`fov_y_deg` 60, `near` 0.1, `far` 100). Those defaults are the right
     * camera for a host that draws geometry already in clip space, with the
     * identity in slot 0, and they are what every revision before ABI 100
     * implied.
     *
     * `camera` is the end of the struct: a caller whose `struct_size` stops
     * after `shadows` is read as if this field were NULL. Raise `struct_size`
     * to sizeof(ReconLFrameDesc) to be read. */
    const ReconLCamera* camera;
} ReconLFrameDesc;

typedef struct ReconLPresentDesc {
    ReconLBase  base;
    void*       out_pixels;      /* headless readback, RGBA8, may be NULL        */
    uint64_t    out_pixels_size;
    uint32_t    out_row_pitch;
    ReconLFormat out_format;
    uint32_t    flip;            /* 1 = present bottom-up (D3D style)             */
} ReconLPresentDesc;

/* ---------------------------------------------------------------- functions */

/* Version query. Always safe, never allocates, never fails. */
RECONL_API void RECONL_CALL reconlVersion(uint32_t* major, uint32_t* minor, uint32_t* patch);
RECONL_API const char* RECONL_CALL reconlVersionString(void);
RECONL_API const char* RECONL_CALL reconlResultName(ReconLResult r);
RECONL_API const char* RECONL_CALL reconlTierName(ReconLTier tier);
RECONL_API const char* RECONL_CALL reconlBackendName(ReconLBackendId backend);

/* Logging: a single global sink, no per-device state, safe from any thread. */
RECONL_API void RECONL_CALL reconlSetLogLevel(ReconLLogLevel level);
RECONL_API ReconLLogLevel RECONL_CALL reconlGetLogLevel(void);
/* sink is called from the emitting thread; must not call back into ReconL. */
RECONL_API void RECONL_CALL reconlSetLogSink(
    void (*sink)(void* user, ReconLLogLevel level, const char* message, const char* file, uint32_t line),
    void* user);

/* Probe. Enumerates backends and one file-system check. Creates no device,
 * allocates nothing the host must free, and is safe to call at any time. */
RECONL_API ReconLResult RECONL_CALL reconlProbe(const ReconLProbeDesc* desc, ReconLProbeInfo* out);

RECONL_API ReconLResult RECONL_CALL reconlCreateDevice(const ReconLDeviceDesc* desc, ReconLDevice** out);
RECONL_API ReconLResult RECONL_CALL reconlGetDeviceLimits(ReconLDevice* device, ReconLDeviceLimits* out);
RECONL_API ReconLResult RECONL_CALL reconlGetMemoryStats(ReconLDevice* device, ReconLMemoryStats* out);

RECONL_API ReconLResult RECONL_CALL reconlCreateBuffer(ReconLDevice* device, const ReconLBufferDesc* desc, ReconLBuffer** out);
RECONL_API ReconLResult RECONL_CALL reconlWriteBuffer(ReconLDevice* device, ReconLBuffer* buffer, uint64_t offset, const void* data, uint64_t size);
RECONL_API ReconLResult RECONL_CALL reconlCreateTexture(ReconLDevice* device, const ReconLTextureDesc* desc, ReconLTexture** out);
RECONL_API ReconLResult RECONL_CALL reconlWriteTexture(ReconLDevice* device, ReconLTexture* texture, const ReconLTextureLevel* level);
RECONL_API ReconLResult RECONL_CALL reconlReadTexture(ReconLDevice* device, ReconLTexture* texture, uint32_t mip, uint32_t layer, void* out, uint64_t out_size, uint32_t out_row_pitch);
RECONL_API ReconLResult RECONL_CALL reconlGenerateMips(ReconLDevice* device, ReconLTexture* texture);
RECONL_API ReconLResult RECONL_CALL reconlCreatePipeline(ReconLDevice* device, const ReconLPipelineDesc* desc, ReconLPipeline** out);
RECONL_API ReconLResult RECONL_CALL reconlCreateSwapchain(ReconLDevice* device, const ReconLSwapchainDesc* desc, ReconLSwapchain** out);
RECONL_API ReconLResult RECONL_CALL reconlCreateCommandList(ReconLDevice* device, const ReconLCommandListDesc* desc, ReconLCommandList** out);
RECONL_API ReconLResult RECONL_CALL reconlCreateFence(ReconLDevice* device, uint32_t signaled, ReconLFence** out);
RECONL_API ReconLResult RECONL_CALL reconlWaitFence(ReconLDevice* device, ReconLFence* fence, uint64_t timeout_ns);
RECONL_API ReconLResult RECONL_CALL reconlConfigureShadows(ReconLDevice* device, const ReconLShadowConfig* config);

/* World revision: bumping it retires cached static shadow cascades and cached
 * static uploads in O(1). `flags`: 1 = shadow-relevant geometry changed. */
RECONL_API ReconLResult RECONL_CALL reconlBumpWorldRevision(ReconLDevice* device, uint32_t flags);

/* Command recording. A command list is single-threaded-on-a-recording-context. */
RECONL_API ReconLResult RECONL_CALL reconlCmdReset(ReconLCommandList* list);
RECONL_API ReconLResult RECONL_CALL reconlCmdBeginRenderPass(ReconLCommandList* list, const ReconLRenderPassDesc* desc);
RECONL_API ReconLResult RECONL_CALL reconlCmdEndRenderPass(ReconLCommandList* list);
RECONL_API ReconLResult RECONL_CALL reconlCmdSetPipeline(ReconLCommandList* list, ReconLPipeline* pipeline);
RECONL_API ReconLResult RECONL_CALL reconlCmdSetVertexBuffer(ReconLCommandList* list, uint32_t stream, ReconLBuffer* buffer, uint64_t offset);
RECONL_API ReconLResult RECONL_CALL reconlCmdSetIndexBuffer(ReconLCommandList* list, ReconLBuffer* buffer, uint64_t offset, ReconLIndexFormat format);
RECONL_API ReconLResult RECONL_CALL reconlCmdSetTexture(ReconLCommandList* list, uint32_t slot, ReconLTexture* texture, const ReconLSamplerDesc* sampler);
/* Push constants. A slot takes 64 bytes - one 4x4 column-major matrix - and is
 * sticky: it keeps its value until the list is reset, and applies to every
 * draw recorded after it.
 *
 *   slot 0  view * projection, the transform vertices are drawn with
 *   slot 1  model, composed with slot 0 per draw
 *
 * Slot 0 is the vertex transform and nothing else. It is not the shadow
 * system's camera: no projection can be undone from it, so the camera's view
 * matrix and frustum travel separately, in `ReconLFrameDesc::camera`. */
RECONL_API ReconLResult RECONL_CALL reconlCmdPushConstants(ReconLCommandList* list, uint32_t slot, const void* data, uint32_t size_bytes);
RECONL_API ReconLResult RECONL_CALL reconlCmdDraw(ReconLCommandList* list, uint32_t vertex_count, uint32_t first_vertex);
RECONL_API ReconLResult RECONL_CALL reconlCmdDrawIndexed(ReconLCommandList* list, uint32_t index_count, uint32_t first_index, int32_t vertex_offset);
RECONL_API uint32_t RECONL_CALL reconlCmdCount(const ReconLCommandList* list); /* infallible */

/* Frame. begin -> submit -> present, in that order, on one thread.
 *
 * One frame is in flight at a time, in one of three states:
 *
 *   idle       nothing open; only reconlBeginFrame is accepted
 *   open       begun and being recorded; only reconlSubmit is accepted
 *   submitted  rendered and waiting; only reconlPresent is accepted
 *
 * A call the state does not allow is RECONL_ERR_FRAME_IN_PROGRESS (a frame is
 * already open) or RECONL_ERR_NO_FRAME (there is none to submit or present).
 *
 * Recovery, so a host never has to guess which call clears a bad state:
 *
 * - A call refused for its arguments changes nothing. A null or foreign handle,
 *   a frame state the call does not accept, a missing descriptor or a camera
 *   the fit cannot use all return an error and leave the frame state as it was:
 *   fix the argument and call again. In particular, the handles passed to
 *   reconlSubmit (list and fence) and reconlPresent (swapchain) are checked
 *   before the call commits, so a bad handle never costs the host a frame.
 * - A call that fails *after* committing ends the frame. A submit whose command
 *   list is rejected and a present that cannot be delivered - an unreadable
 *   present descriptor, a buffer too small for the frame - both consume the
 *   submitted frame, increment ReconLStats.frames_dropped, and return the device
 *   to idle, so the next reconlBeginFrame opens a fresh frame. Nothing is
 *   dropped silently, and a failed reconlPresent is not retryable for the same
 *   frame: it is dropped, not held. Re-record and begin again.
 * - reconlBeginFrame itself drops nothing, but a begin whose target reservation
 *   is refused (RECONL_ERR_BUDGET_EXCEEDED or RECONL_ERR_OUT_OF_MEMORY, from
 *   the frame's colour/depth and shadow targets) also leaves the device idle, so
 *   the next reconlBeginFrame is accepted.
 *
 * The invariant underneath all of it: however a call fails, the state the
 * device is left in still accepts the call the state machine names next.
 *
 * Continuing on the reference tier (the offload; see RECONL_ALLOW_DOWNGRADE_TIER
 * and docs/offload.md).
 *
 * A device built on a hardware backend may continue its frames on the CPU
 * reference tier. It is a tier change like any other: visible in
 * ReconLStats.backend/tier/tier_reason, counted in safe_path_events, recorded in
 * the downgrade ring with its reason, and refused entirely when the host left
 * RECONL_ALLOW_DOWNGRADE_TIER out of ReconLDeviceDesc.allow_downgrade. It
 * happens for exactly two reasons:
 *
 * - A hardware fault: the driver reports the device removed or reset. The call
 *   that hit it does not fail - the frame is re-rendered on the reference tier
 *   and presented normally, so a fault costs the frame's GPU time and not the
 *   frame, and frames_dropped is not incremented. A driver failure the device
 *   survived (a rejected descriptor, an unsupported format) is an error as
 *   before and never moves the device.
 * - A measured overload: frames miss target_frame_ms for
 *   downgrade_after_frames consecutive frames *and* the reference tier is
 *   measured to render them faster. One frame on the CPU is the calibration that
 *   decides it, reported as a safe_path_event; if the reference tier was not
 *   faster the device returns to the hardware and remembers that measurement for
 *   that resolution and shadow plan, so the calibration is never paid twice.
 *
 * Handles stay valid across an offload: buffers, textures, pipelines and command
 * lists are host-owned bytes, and both tiers render from them. An offloaded
 * frame's pixels are the reference tier's pixels, byte for byte, exactly as if
 * the device had been created on that tier.
 *
 * Coming back up: a device offloaded because of a measured overload rebuilds the
 * hardware backend after downgrade_after_frames consecutive frames inside
 * target_frame_ms, and tries the hardware again - once per resolution and shadow
 * plan, so a workload that oscillates around the target does not thrash. A
 * fault-offloaded device recovers the same way, except that with
 * target_frame_ms = 0 there is no window to settle in: the offload then lasts
 * until the host destroys the device. Either return is logged with
 * RECONL_TIER_REASON_RECOVERY. */
RECONL_API ReconLResult RECONL_CALL reconlBeginFrame(ReconLDevice* device, ReconLFrameDesc* desc);
RECONL_API ReconLResult RECONL_CALL reconlSubmit(ReconLDevice* device, const ReconLCommandList* list, ReconLFence* fence);
RECONL_API ReconLResult RECONL_CALL reconlPresent(ReconLDevice* device, ReconLSwapchain* swapchain, ReconLPresentDesc* desc);

RECONL_API ReconLResult RECONL_CALL reconlGetStats(ReconLDevice* device, ReconLStats* out);
RECONL_API ReconLResult RECONL_CALL reconlResetStats(ReconLDevice* device);

/* Last error. `device` may be NULL, in which case a process-wide (mutex-guarded)
 * last error is returned. Never returns OK unless nothing has failed yet. */
RECONL_API ReconLResult RECONL_CALL reconlGetLastError(ReconLDevice* device, ReconLErrorInfo* out);

/* Lifetime. Every handle is ref-counted; release the last reference to destroy. */
RECONL_API uint32_t RECONL_CALL reconlRetain(void* handle);   /* returns new count */
RECONL_API uint32_t RECONL_CALL reconlRelease(void* handle);  /* returns new count */

/* Diagnostics: drive the tier ladder on purpose (tests, CI, demos). */
RECONL_API ReconLResult RECONL_CALL reconlRequestTier(ReconLDevice* device, ReconLTier tier, ReconLTierReason reason);
RECONL_API ReconLResult RECONL_CALL reconlRequestShadowFallback(ReconLDevice* device, uint32_t event_kind, const char* detail);
RECONL_API ReconLResult RECONL_CALL reconlAudit(ReconLDevice* device, uint32_t every_n_frames, uint32_t* out_divergences);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RECONL_H */
