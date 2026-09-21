/* ReconL - backend descriptor catalogue.
 *
 * A backend descriptor may be passed to reconlCreateDevice through
 * ReconLDeviceDesc::backend_desc.
 *
 * RESERVED IN THIS REVISION (ABI 100): the library accepts the pointer and does
 * not read it. Every backend starts with its documented defaults, passing a
 * descriptor changes nothing, and passing a wild pointer is as harmless as
 * passing NULL - so a host may fill one in today. The structs below are the
 * contract for the revision that reads them: a field is never repurposed, so a
 * descriptor written against this header keeps meaning the same thing when the
 * knobs start to be honored. Until then their values are not observable.
 *
 * A backend that is asked for a capability it does not have reports it in
 * ReconLBackendProbe / ReconLDeviceLimits; it never fails the device creation
 * over a missing optional feature - the tier resolver falls back instead.
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
 * Reserved this revision: see the note at the top of this header.
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
    /* Out-of-core tuning. Ignored unless ReconLMemoryBudget::allow_disk_spill. */
    uint64_t    resident_tile_budget_bytes; /* 0 = derived from the RAM cap */
    uint64_t    arena_block_bytes;          /* 0 = 4 MiB */
    uint32_t    eviction_policy;            /* 0 = LRU by cost/size, 1 = LRU plain */
    uint32_t    prefetch_depth;
} ReconLSoftCpuDesc;

/* ---------------------------------------------------------------------- null
 * Reserved this revision: see the note at the top of this header.
 *
 * Deterministic no-op backend. It answers every query, records every command,
 * counts everything, and writes nothing. It exists so that ABI conformance,
 * refcounting, budget accounting and error-path tests run with no GPU, no
 * threads and no disk - the CI tier.
 *
 * With `fake_tier` set, it reports a chosen tier and lets the downgrade ladder
 * be exercised end to end without hardware.
 */
typedef struct ReconLNullDesc {
    ReconLBase  base;   /* type = RECONL_STRUCT_NULL_DESC */
    ReconLTier  fake_tier;        /* tier to report; out-of-range = T2 */
    uint64_t    fake_vram_bytes;
    uint64_t    fake_ram_bytes;
    uint32_t    report_caps;      /* ReconLCaps; 0 = the full set */
    uint32_t    present_checksum; /* 1 = fill out_pixels with the frame checksum pattern */
    uint32_t    fail_after_frames;/* N > 0: return RECONL_ERR_DEVICE_LOST on frame N */
    uint32_t    reserved;
    uint32_t    command_capacity;
    uint32_t    reserved2;
} ReconLNullDesc;

/* --------------------------------------------------------------------- d3d11
 * Reserved this revision: see the note at the top of this header.
 *
 * The first hardware tier. The same passes, the same cascade, reversed-Z, so
 * the goldens are comparable with soft-cpu line for line.
 */
typedef struct ReconLD3D11Desc {
    ReconLBase  base;   /* type = RECONL_STRUCT_D3D11_DESC */
    int32_t     adapter_index;    /* -1 = default adapter (highest VRAM)   */
    uint32_t    feature_level_min;/* e.g. 0xB000 for 11_0                 */
    uint32_t    debug_layer;      /* 1 = enable the SDK debug layer if present */
    uint32_t    allow_warp;       /* 1 = fall back to WARP (software D3D11)   */
    uint32_t    prefer_flip_model;/* 1 = DXGI flip-model swapchain            */
    uint32_t    reserved;
    uint64_t    requested_vram_cap;
} ReconLD3D11Desc;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RECONL_BACKENDS_H */
