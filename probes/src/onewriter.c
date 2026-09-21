/* Who writes the tier ladder's ring, through the shipped DLL.
 *
 * Two layers touch a device's tier (docs/offload.md, "Two layers, one number"):
 * the backend's own relabel and the host-level offload that changes the backend.
 * This probe measures, on the artifact a host actually loads:
 *
 *   arm A  allow=0, so no backend change is permitted: the only writer left is
 *          the running backend's own relabel ladder. Does the hardware half of
 *          that ladder fire, or is its config path dead?
 *   arm B  allow=TIER: both writers live. What does the host see in
 *          ReconLStats - the entry count against safe_path_events, whether the
 *          entries are in the order they happened, and whether an entry the host
 *          already saw survives the backend that recorded it.
 *
 * Usage: onewriter [--frames=24] [--size=64]
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

static ReconLResult full_frame(Rig* r) {
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
    if (reconlBeginFrame(r->d, &fd) != RECONL_OK) return RECONL_ERR_BACKEND_UNAVAILABLE;

    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = SIZE; rp.viewport_height = SIZE;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;
    reconlCmdReset(r->cl);
    reconlCmdBeginRenderPass(r->cl, &rp);
    reconlCmdEndRenderPass(r->cl);
    if (reconlSubmit(r->d, r->cl, NULL) != RECONL_OK) return RECONL_ERR_BACKEND_UNAVAILABLE;

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

/* One entry the host has read, kept so a later frame's read can be compared
 * against it: "did an entry the host already saw stay put?". */
typedef struct {
    uint64_t frame_index;
    uint32_t from, to, reason;
    char detail[96];
} Seen;

static void print_ring(Rig* r) {
    ReconLStats st;
    if (!stats_of(r, &st)) { printf("      ring unreadable\n"); return; }
    printf("      ring: %u entries, safe_path_events %u\n", st.downgrade_count, st.safe_path_events);
    for (uint32_t i = 0; i < st.downgrade_count && i < st.downgrade_capacity; ++i) {
        printf("        [%u] %-17s -> %-17s %-14s frame %-4llu %s\n", i,
               reconlTierName(st.downgrades[i].from), reconlTierName(st.downgrades[i].to),
               reason_name(st.downgrades[i].reason),
               (unsigned long long)st.downgrades[i].frame_index, st.downgrades[i].detail);
    }
}

static uint32_t snapshot_ring(Rig* r, Seen* out, uint32_t cap) {
    ReconLStats st;
    if (!stats_of(r, &st)) return 0;
    uint32_t n = st.downgrade_count < st.downgrade_capacity ? st.downgrade_count : st.downgrade_capacity;
    if (n > cap) n = cap;
    for (uint32_t i = 0; i < n; ++i) {
        out[i].frame_index = st.downgrades[i].frame_index;
        out[i].from = st.downgrades[i].from; out[i].to = st.downgrades[i].to;
        out[i].reason = st.downgrades[i].reason;
        snprintf(out[i].detail, sizeof out[i].detail, "%s", st.downgrades[i].detail);
    }
    return n;
}

/* arm A: nothing may change the backend; the only writer left is the running
 * backend's own relabel ladder. */
static void arm_relabel_only(void) {
    printf("\n== arm A: the backend's own relabel, no backend change permitted (allow=0) ==\n");
    Rig r;
    if (rig(&r, RECONL_BACKEND_D3D11, 0, 1, 1)) { printf("    device not created\n"); rig_free(&r); return; }

    uint32_t relabels = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        if (full_frame(&r) != RECONL_OK) { printf("    frame %u: present failed\n", i); break; }
        ReconLStats st;
        if (!stats_of(&r, &st)) break;
        printf("    frame %u: %-10s %-17s %-14s entries %u, safe-path %u\n", i,
               reconlBackendName(st.backend), reconlTierName(st.tier), reason_name(st.tier_reason),
               st.downgrade_count, st.safe_path_events);
    }
    ReconLStats st;
    if (stats_of(&r, &st)) {
        relabels = st.downgrade_count;
        print_ring(&r);
        check(st.backend == RECONL_BACKEND_D3D11, "the opt-out kept the backend on the hardware");
        check(st.safe_path_events == 0, "no backend change was counted");
        check(relabels > 0, "the hardware backend's own relabel ladder fired");
        check(st.tier != RECONL_TIER_T1_GPU_SHARED, "the hardware device's tier stepped down");
    }
    rig_free(&r);
}

/* arm B: both writers live. What the host reads, and whether it is stable. */
static void arm_two_writers(void) {
    printf("\n== arm B: both layers live (allow=TIER, target 1 ms, threshold 1) ==\n");
    Rig r;
    if (rig(&r, RECONL_BACKEND_D3D11, RECONL_ALLOW_DOWNGRADE_TIER, 1, 1)) {
        printf("    device not created\n"); rig_free(&r); return;
    }

    Seen seen[32];
    uint32_t seen_n = 0;
    uint32_t lost = 0, order_breaks = 0, max_seen_frame = 0;
    uint32_t last_backend_changes = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        if (full_frame(&r) != RECONL_OK) { printf("    frame %u: present failed\n", i); break; }
        ReconLStats st;
        if (!stats_of(&r, &st)) break;
        printf("    frame %u: %-9s %-17s %-14s entries %u, safe-path %u, cost %llu us\n", i,
               reconlBackendName(st.backend), reconlTierName(st.tier), reason_name(st.tier_reason),
               st.downgrade_count, st.safe_path_events,
               (unsigned long long)(st.frame.total_ns / 1000));
        if (i < 7) print_ring(&r);
        Seen now[32];
        uint32_t now_n = snapshot_ring(&r, now, 32);
        /* An entry the host already read that is no longer there: the record of
         * something that happened, gone. */
        for (uint32_t s = 0; s < seen_n; ++s) {
            int found = 0;
            for (uint32_t k = 0; k < now_n; ++k) {
                if (now[k].frame_index == seen[s].frame_index && now[k].from == seen[s].from &&
                    now[k].to == seen[s].to && now[k].reason == seen[s].reason &&
                    strcmp(now[k].detail, seen[s].detail) == 0) { found = 1; break; }
            }
            if (!found) {
                lost++;
                printf("      LOST since frame %u: %s -> %s frame %llu (%s)\n", i,
                       reconlTierName(seen[s].from), reconlTierName(seen[s].to),
                       (unsigned long long)seen[s].frame_index, seen[s].detail);
            }
        }
        /* "in the order they happened": the frame indices a host reads must not
         * go backwards. */
        for (uint32_t k = 1; k < now_n; ++k) {
            if (now[k].frame_index < now[k - 1].frame_index) {
                order_breaks++;
                if (order_breaks == 1) {
                    printf("      OUT OF ORDER: entry %u is frame %llu, entry %u before it is frame %llu\n",
                           k, (unsigned long long)now[k].frame_index, k - 1,
                           (unsigned long long)now[k - 1].frame_index);
                }
            }
        }
        if (now_n > seen_n) { memcpy(seen, now, sizeof(Seen) * now_n); seen_n = now_n; }
        if (seen_n && seen[seen_n - 1].frame_index > max_seen_frame) max_seen_frame = (uint32_t)seen[seen_n - 1].frame_index;
        if (st.safe_path_events != last_backend_changes) last_backend_changes = st.safe_path_events;
    }

    ReconLStats st;
    if (stats_of(&r, &st)) {
        print_ring(&r);
        printf("    backend changes %u, entries in the ring %u, entries lost %u, order breaks %u\n",
               st.safe_path_events, st.downgrade_count, lost, order_breaks);
        check(st.safe_path_events > 0, "the offload layer fired (there is something to compare)");
        check(lost == 0, "an entry the host read was still there at the end");
        check(order_breaks == 0, "the entries are in the order they happened");
        check(st.downgrade_count >= st.safe_path_events,
              "every backend change wrote an entry the host can read");
    }
    rig_free(&r);
}

int main(int argc, char** argv) {
    uint32_t frames = 16;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--frames=", 9) == 0) frames = (uint32_t)strtoul(argv[i] + 9, NULL, 10);
        else if (strncmp(argv[i], "--size=", 7) == 0) SIZE = (uint32_t)strtoul(argv[i] + 7, NULL, 10);
    }
    (void)frames;
    printf("onewriter — %ux%u, who writes the tier ladder's ring\n", SIZE, SIZE);

    arm_relabel_only();
    arm_two_writers();

    printf("\nonewriter: %d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
