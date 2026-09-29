/* ReconL - backend descriptor catalogue.
 *
 * A backend descriptor may be passed to reconlCreateDevice through
 * ReconLDeviceDesc::backend_desc. The D3D11 adapter selection fields are read
 * in ABI 101 and the compute backend's in ABI 102; SoftCPU and Null descriptors
 * remain reserved and are accepted without being read.
 *
 * A backend that is asked for a capability it does not have reports it in
 * ReconLBackendProbe / ReconLDeviceLimits; it never fails device creation over
 * a missing optional feature - the tier resolver falls back instead.
 *
 * No backend-specific type may appear in the portable core. This header is the
 * only place they are declared, and it is not included by include/reconl/reconl.h.
 */
#ifndef RECONL_BACKENDS_H
#define RECONL_BACKENDS_H

#include "reconl.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------- soft-cpu
 * Reserved this revision: accepted and never read.
 *
 * The reference tier (T2/T3/T4). Bit-identical output for a given
 * (scene, seed, tier, worker count) is the contract, so every knob here is part
 * of the run fingerprint.
 */
typedef struct ReconLSoftCpuDesc {
    ReconLBase  base;   /* type = RECONL_STRUCT_SOFTCPU_DESC */
    uint32_t    worker_threads;   /* 0 = min(limits.worker_threads_max, hw) */
    uint32_t    tile_size;        /* 0 = 64; must be a power of two, 16..256 */
    uint32_t    simd;             /* 0 = auto, 1 = force scalar (reference path) */
    uint32_t    deterministic;    /* 1 = fixed tile order (default; slower look, same output) */
    uint32_t    bin_order;        /* 0 = scanline, 1 = tile-major, 2 = morton */
    uint32_t    reserved;
    uint64_t    resident_tile_budget_bytes;
    uint64_t    arena_block_bytes;
    uint32_t    eviction_policy;
    uint32_t    prefetch_depth;
} ReconLSoftCpuDesc;

/* ---------------------------------------------------------------------- null
 * Reserved this revision: accepted and never read. The deterministic no-op
 * backend exists so ABI conformance, refcounting and failure paths run with no
 * GPU, no threads and no disk.
 */
typedef struct ReconLNullDesc {
    ReconLBase  base;   /* type = RECONL_STRUCT_NULL_DESC */
    ReconLTier  fake_tier;        /* tier to report; out-of-range = T2 */
    uint64_t    fake_vram_bytes;
    uint64_t    fake_ram_bytes;
    uint32_t    report_caps;
    uint32_t    present_checksum;
    uint32_t    fail_after_frames;
    uint32_t    reserved;
    uint32_t    command_capacity;
    uint32_t    reserved2;
} ReconLNullDesc;

/* --------------------------------------------------------------------- d3d11
 * The first hardware tier. The same passes, the same cascade, reversed-Z, so
 * the goldens are comparable with soft-cpu line for line.
 *
 * The original descriptor prefix ends at requested_vram_cap and remains
 * accepted. Fields appended after that prefix control adapter selection:
 * preference is AUTO, INTEGRATED, DISCRETE, INDEX, or LUID. AUTO and DISCRETE
 * prefer a discrete GPU, then fall back to any usable adapter; INTEGRATED does
 * the analogous integrated-first choice. `adapter_index` is used only by INDEX
 * and follows DXGI enumeration order. `adapter_luid` is required for LUID and
 * should come from ReconLAdapterInfo; it is stable until reboot or driver
 * restart. A LUID selects that exact adapter, with no fallback.
 *
 * Other original descriptor options are currently reserved and ignored.
 */
typedef struct ReconLD3D11Desc {
    ReconLBase  base;   /* type = RECONL_STRUCT_D3D11_DESC */
    int32_t     adapter_index;    /* zero-based DXGI index; used by INDEX preference */
    uint32_t    feature_level_min;/* 0 = 11_0 (only supported level this revision) */
    uint32_t    debug_layer;      /* reserved; currently ignored */
    uint32_t    allow_warp;       /* reserved; currently ignored */
    uint32_t    prefer_flip_model;/* reserved; currently ignored */
    uint32_t    reserved;
    uint64_t    requested_vram_cap; /* reserved; budget lives in ReconLMemoryBudget */
    ReconLAdapterPreference adapter_preference; /* AUTO = discrete first */
    uint32_t    reserved2;
    uint64_t    adapter_luid;     /* exact adapter_luid for LUID preference */
} ReconLD3D11Desc;

/* ----------------------------------------------------------------- gpu-compute
 * The GPU tiers through a vendor compute API: the CUDA driver API on NVIDIA,
 * the ROCm/HIP runtime on AMD. Opened at run time, so neither stack is a build
 * or link dependency; a machine with neither driver reports
 * RECONL_ERR_BACKEND_UNAVAILABLE rather than falling back to software.
 *
 * Adapters come from reconlEnumerateAdapters(RECONL_BACKEND_GPU_COMPUTE, ...),
 * which reports the device name, its total memory, whether the driver
 * classified it as integrated or discrete, and whether a compute context can
 * be created on it. `device_index` selects one by its position in that list;
 * `vendor_preference` picks which runtime to open when a machine has both.
 *
 * A device is created only if a context opens and the backend's kernels
 * compile; a device that is reported but cannot render is never returned.
 */
typedef enum ReconLComputeVendor {
    RECONL_COMPUTE_VENDOR_AUTO = 0,    /* NVIDIA first, then AMD               */
    RECONL_COMPUTE_VENDOR_NVIDIA = 1,  /* CUDA only                            */
    RECONL_COMPUTE_VENDOR_AMD = 2      /* ROCm/HIP only                        */
} ReconLComputeVendor;

typedef struct ReconLComputeDesc {
    ReconLBase           base;   /* type = RECONL_STRUCT_COMPUTE_DESC */
    ReconLComputeVendor  vendor_preference; /* AUTO = whichever runtime is here */
    int32_t              device_index;      /* < 0 = let ReconL choose          */
    uint32_t             reserved;
    uint64_t             reserved2;
} ReconLComputeDesc;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RECONL_BACKENDS_H */
