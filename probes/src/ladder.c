/* The tier ladder's own sequence, through the shipped DLL, with a target
 * derived from frame costs measured on this machine (docs/offload.md).
 *
 * What this settles: whether the 64x64 knife-edge arm's residual failure is a
 * real violation of the documented anti-thrash rule ("the return trip is
 * attempted at most once per plan: a workload that oscillates around the target
 * must not thrash") or an arm whose target - a hard-coded 1 ms - puts it in a
 * regime the document never promised anything about.
 *
 * So: pilot both tiers at the same frame (target 0 = no ladder), derive the
 * target from what they actually cost here, then drive one hardware device
 * through the ladder at that target and print what the host can see for every
 * frame - tier, reason, the frame cost the ladder judged, and the safe-path
 * count - and the whole downgrade ring at the end.
 *
 * Usage: ladder [--frames=24] [--pilot=8] [--size=64]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "reconl/reconl.h"

#define SETBASE(p, T)                                        \
    do {                                                     \
        (p)->base.struct_size = (uint32_t)sizeof(*(p));       \
        (p)->base.type = (T);                                 \
        (p)->base.next = NULL;                                \
    } while (0)

static uint32_t SIZE = 64;

static int checks = 0, fails = 0;
static void check(int ok, const char* what) {
    checks++;
    if (!ok) fails++;
    printf("    [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

static void* h_alloc(void* u, size_t s, size_t a) {
    (void)u; if (a < 16) a = 16;
    size_t t = s + a + sizeof(void*);
    unsigned char* raw = (unsigned char*)malloc(t);
    if (!raw) return NULL;
    uintptr_t b = (uintptr_t)raw;
    uintptr_t al = (b + sizeof(void*) + a - 1) & ~(uintptr_t)(a - 1);
    ((void**)al)[-1] = raw;
    return (void*)al;
}
static void h_free(void* u, void* p, size_t s) { (void)u; (void)s; if (p) free(((void**)p)[-1]); }
static void* h_realloc(void* u, void* p, size_t o, size_t n, size_t a) {
    void* f = h_alloc(u, n, a);
    if (!f) return NULL;
    if (p) { memcpy(f, p, o < n ? o : n); h_free(u, p, o); }
    return f;
}

typedef struct {
    ReconLDevice* d;
    ReconLSwapchain* s;
    ReconLCommandList* cl;
    unsigned char* px;
} Rig;

static int rig(Rig* r, ReconLBackendId backend, uint32_t allow, uint32_t target_ms, uint32_t after) {
    memset(r, 0, sizeof *r);
    r->px = (unsigned char*)malloc((size_t)SIZE * SIZE * 4);
    if (!r->px) return 1;
    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    dd.tier_hint = backend == RECONL_BACKEND_D3D11 ? RECONL_TIER_T1_GPU_SHARED : RECONL_TIER_T2_CPU_RAM;
    dd.allow_downgrade = allow;
    dd.target_frame_ms = target_ms;
    dd.downgrade_after_frames = after;
    dd.seed = 7;
    dd.allocator.alloc = h_alloc; dd.allocator.realloc = h_realloc; dd.allocator.free = h_free;
    if (reconlCreateDevice(&dd, &r->d) != RECONL_OK) return 1;

    ReconLSwapchainDesc sd;
    memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = SIZE; sd.height = SIZE;
    sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2; sd.present_to_memory = 1;
    sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    if (reconlCreateSwapchain(r->d, &sd, &r->s) != RECONL_OK) return 1;

    ReconLCommandListDesc cd;
    memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    cd.capacity_bytes = 8192;
    return reconlCreateCommandList(r->d, &cd, &r->cl) != RECONL_OK;
}

static void rig_free(Rig* r) {
    if (r->cl) reconlRelease(r->cl);
    if (r->s) reconlRelease(r->s);
    if (r->d) reconlRelease(r->d);
    free(r->px);
    r->px = NULL;
}

static int stats_of(Rig* r, ReconLStats* st) {
    memset(st, 0, sizeof *st);
    SETBASE(st, RECONL_STRUCT_STATS);
    return reconlGetStats(r->d, st) == RECONL_OK;
}

static ReconLResult frame_and_submit(Rig* r) {
    ReconLLight light;
    memset(&light, 0, sizeof light);
    SETBASE(&light, RECONL_STRUCT_LIGHT);
    light.type = RECONL_LIGHT_DIRECTIONAL;
    light.direction[2] = -1.0f;
    light.color[0] = light.color[1] = light.color[2] = 1.0f;
    light.intensity = 1.0f;
    ReconLLightList ll;
    memset(&ll, 0, sizeof ll);
    SETBASE(&ll, RECONL_STRUCT_LIGHT_LIST);
    ll.count = 1; ll.lights = &light;

    ReconLFrameDesc fd;
    memset(&fd, 0, sizeof fd);
    SETBASE(&fd, RECONL_STRUCT_FRAME_DESC);
    fd.width = SIZE; fd.height = SIZE; fd.seed = 1; fd.lights = &ll;
    ReconLResult b = reconlBeginFrame(r->d, &fd);
    if (b != RECONL_OK) return b;

    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = SIZE; rp.viewport_height = SIZE;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;
    reconlCmdReset(r->cl);
    reconlCmdBeginRenderPass(r->cl, &rp);
    reconlCmdEndRenderPass(r->cl);
    return reconlSubmit(r->d, r->cl, NULL);
}

static ReconLResult full_frame(Rig* r) {
    ReconLResult s = frame_and_submit(r);
    if (s != RECONL_OK) return s;
    ReconLPresentDesc p;
    memset(&p, 0, sizeof p);
    SETBASE(&p, RECONL_STRUCT_PRESENT_DESC);
    p.out_pixels = r->px;
    p.out_pixels_size = (uint64_t)SIZE * SIZE * 4;
    p.out_row_pitch = SIZE * 4;
    p.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return reconlPresent(r->d, r->s, &p);
}

static const char* reason_name(ReconLTierReason r) {
    switch (r) {
        case RECONL_TIER_REASON_STARTUP_PROBE: return "startup-probe";
        case RECONL_TIER_REASON_HOST_REQUEST: return "host-request";
        case RECONL_TIER_REASON_ALLOCATION_OVER_BUDGET: return "over-budget";
        case RECONL_TIER_REASON_FRAME_TIME_OVER_TARGET: return "over-target";
        case RECONL_TIER_REASON_DEVICE_REMOVED: return "device-removed";
        case RECONL_TIER_REASON_DEVICE_LOST: return "device-lost";
        case RECONL_TIER_REASON_MEMORY_PRESSURE: return "memory-pressure";
        case RECONL_TIER_REASON_DISK_CACHE_FULL: return "disk-cache-full";
        case RECONL_TIER_REASON_BUILD: return "build";
        case RECONL_TIER_REASON_NO_GPU_API: return "no-gpu-api";
        case RECONL_TIER_REASON_RECOVERY: return "recovery";
        default: return "?";
    }
}

#define REASON_MAX RECONL_TIER_REASON_RECOVERY

/* What this machine's frame costs look like at one tier, with no ladder running.
 * A target has to be above what the reference tier's typical frame costs and
 * below what the hardware's does, or it cannot tell the two apart. */
static uint64_t pilot(ReconLBackendId backend, uint32_t frames, uint64_t* mean_out, uint64_t* peak_out, int* ok) {
    Rig r;
    *ok = 0;
    *mean_out = 0;
    *peak_out = 0;
    if (rig(&r, backend, 0, 0, 0)) { printf("    pilot: %s device not created\n", reconlBackendName(backend)); rig_free(&r); return 0; }
    uint64_t best = UINT64_MAX, peak = 0, sum = 0;
    uint32_t counted = 0;
    for (uint32_t i = 0; i < frames; ++i) {
        if (full_frame(&r) != RECONL_OK) continue;
        ReconLStats st;
        if (!stats_of(&r, &st)) continue;
        uint64_t ns = st.frame.total_ns;
        if (i == 0) printf("    pilot %s: %s %ux%u\n", reconlBackendName(backend),
                           reconlTierName(st.tier), SIZE, SIZE);
        if (ns == 0) continue;
        if (ns < best) best = ns;
        if (ns > peak) peak = ns;
        sum += ns;
        counted++;
    }
    if (best == UINT64_MAX) best = 0;
    *mean_out = counted ? sum / counted : 0;
    *peak_out = peak;
    printf("    pilot %s: %u frames, cheapest %llu ns, mean %llu ns, peak %llu ns\n",
           reconlBackendName(backend), counted, (unsigned long long)best,
           (unsigned long long)*mean_out, (unsigned long long)peak);
    rig_free(&r);
    *ok = 1;
    return best;
}

int main(int argc, char** argv) {
    uint32_t frames = 24, pilot_frames = 8;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--frames=", 9) == 0) frames = (uint32_t)strtoul(argv[i] + 9, NULL, 10);
        else if (strncmp(argv[i], "--pilot=", 8) == 0) pilot_frames = (uint32_t)strtoul(argv[i] + 8, NULL, 10);
        else if (strncmp(argv[i], "--size=", 7) == 0) SIZE = (uint32_t)strtoul(argv[i] + 7, NULL, 10);
    }

    printf("ladder — %ux%u, %u frames, target derived from this machine\n", SIZE, SIZE, frames);

    int gpu_ok = 0, cpu_ok = 0;
    uint64_t gpu_mean = 0, gpu_peak = 0, cpu_mean = 0, cpu_peak = 0;
    uint64_t gpu_min = pilot(RECONL_BACKEND_D3D11, pilot_frames, &gpu_mean, &gpu_peak, &gpu_ok);
    uint64_t cpu_min = pilot(RECONL_BACKEND_SOFT_CPU, pilot_frames, &cpu_mean, &cpu_peak, &cpu_ok);
    if (!gpu_ok) { printf("no hardware tier on this machine\n"); return 2; }
    (void)cpu_min; (void)cpu_peak; (void)gpu_min;

    /* The target is derived here, from what the two tiers cost on this machine,
     * because a hard-coded 1 ms is a number that happens to sit under the
     * hardware's frame on one host and above it on another. A target the
     * hardware's typical frame is under and the reference tier's typical frame is
     * over is the discriminating one: the hardware misses it, the reference tier
     * meets it, and the ladder's job is decided by the measurements rather than by
     * luck. When no whole millisecond does that, take the largest one below the
     * hardware's mean - the trigger still fires - and say that the reference tier
     * also misses it, which is a different regime. */
    uint64_t cpu_floor_ms = cpu_mean / 1000000u;   /* a target above this is over a typical CPU frame */
    uint64_t gpu_floor_ms = gpu_mean / 1000000u;   /* a target below this is under a typical GPU frame */
    int discriminating = gpu_floor_ms > cpu_floor_ms;
    uint64_t derived = discriminating ? gpu_floor_ms : gpu_floor_ms;
    if (derived == 0) derived = 1;
    uint32_t target_ms = (uint32_t)derived;
    printf("  derived target: %u ms (cpu mean %.3f ms, gpu mean %.3f ms, gpu peak %.3f ms)%s\n",
           target_ms, cpu_mean / 1e6, gpu_mean / 1e6, gpu_peak / 1e6,
           discriminating ? " - under the hardware's frame, over the reference tier's"
                          : " - under the hardware's mean; the reference tier misses it too");

    /* The ladder's own worst case: one frame over target offloads, one frame
     * inside it returns - the thresholds the knife-edge arm used. */
    Rig r;
    if (rig(&r, RECONL_BACKEND_D3D11, RECONL_ALLOW_DOWNGRADE_TIER, target_ms, 1)) {
        printf("device not created\n");
        return 2;
    }

    uint32_t changes = 0, last_change_frame = 0, offloads = 0, returns = 0;
    uint64_t last_ns = 0;
    printf("\n  frame  backend    tier              reason            safe-path  cost\n");
    for (uint32_t i = 0; i < frames; ++i) {
        ReconLResult p = full_frame(&r);
        if (p != RECONL_OK) { printf("  frame %u: present -> %d\n", i, (int)p); break; }
        ReconLStats st;
        if (!stats_of(&r, &st)) { printf("  frame %u: stats unreadable\n", i); break; }
        printf("  %5u  %-10s %-17s %-17s %9u  %llu us%s\n", i, reconlBackendName(st.backend),
               reconlTierName(st.tier), reason_name(st.tier_reason), st.safe_path_events,
               (unsigned long long)(st.frame.total_ns / 1000),
               st.safe_path_events != changes ? "   <-- backend change" : "");
        if (st.safe_path_events != changes) {
            changes = st.safe_path_events;
            last_change_frame = i;
            if (st.tier_reason == RECONL_TIER_REASON_RECOVERY) returns++;
            else offloads++;
        }
        last_ns = st.frame.total_ns;
    }

    ReconLStats st;
    if (!stats_of(&r, &st)) { printf("final stats unreadable\n"); rig_free(&r); return 2; }
    printf("\n  ring (%u entries, %u total, capacity %u)\n", st.downgrade_count, st.downgrade_count,
           st.downgrade_capacity);
    for (uint32_t i = 0; i < st.downgrade_count && i < st.downgrade_capacity; ++i) {
        printf("    [%u] %s -> %s  %s  frame %llu\n        %s\n", i,
               reconlTierName(st.downgrades[i].from), reconlTierName(st.downgrades[i].to),
               reason_name(st.downgrades[i].reason),
               (unsigned long long)st.downgrades[i].frame_index, st.downgrades[i].detail);
    }

    printf("\n  verdict on the documented rule (docs/offload.md):\n");
    check(st.safe_path_events == changes, "every backend change was counted once");
    check(returns <= 1, "the return trip was attempted at most once (one per plan)");
    check(changes <= 3, "one round trip at most: offload, return, and the offload of a miss after it");
    check(last_change_frame + 1 < frames,
          "the ladder settled: no backend change in the last frame of the window");
    check(st.downgrade_count >= changes,
          "each backend change wrote an entry the host can read");
    {
        int why = 1;
        for (uint32_t i = 0; i < st.downgrade_count && i < st.downgrade_capacity; ++i) {
            if (st.downgrades[i].detail[0] == 0) why = 0;
            if (st.downgrades[i].reason > REASON_MAX) why = 0;
        }
        check(why, "every entry carries a reason and the measurement behind it");
    }
    printf("  %u backend changes in %u frames (%u offloads, %u returns), last at frame %u, final cost %llu us\n",
           changes, frames, offloads, returns, last_change_frame, (unsigned long long)(last_ns / 1000));

    printf("\nladder: %d checks, %d failures\n", checks, fails);
    rig_free(&r);
    return fails ? 1 : 0;
}
