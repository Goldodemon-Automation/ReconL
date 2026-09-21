/* The over-target decision, as a host sees it, through the shipped DLL.
 *
 * One device, N frames, then everything the host can read about the decision:
 * the tier and its reason, the downgrade ring (detail strings included, because
 * which layer wrote an entry is in the text), the safe-path counter, and the
 * composed frame cost the ladder is supposed to decide on.
 *
 * The point is to discriminate the two definitions of "over target":
 *   device-local  = what the backend measures inside its own render()
 *   composed      = device-local + the readback the host waited for
 * which differ by the readback (most of a hardware frame at 1080p).
 *
 * Usage: overtarget --backend=d3d11|soft-cpu --size=WxH --target-ms=N
 *                   --after=N --frames=N --allow=0|1
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "reconl/reconl.h"

#define SETBASE(p, T)                                          \
    do {                                                       \
        (p)->base.struct_size = (uint32_t)sizeof(*(p));         \
        (p)->base.type = (T);                                  \
        (p)->base.next = NULL;                                  \
    } while (0)

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
    uint32_t w, h;
} Rig;

static int rig(Rig* r, ReconLBackendId backend, uint32_t allow, uint32_t target_ms,
               uint32_t after, uint32_t w, uint32_t h) {
    memset(r, 0, sizeof *r);
    r->w = w; r->h = h;
    r->px = (unsigned char*)malloc((size_t)w * h * 4);
    if (!r->px) return 1;
    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    dd.allow_downgrade = allow;
    dd.target_frame_ms = target_ms;
    dd.downgrade_after_frames = after;
    dd.seed = 7;
    dd.allocator.alloc = h_alloc; dd.allocator.realloc = h_realloc; dd.allocator.free = h_free;
    if (reconlCreateDevice(&dd, &r->d) != RECONL_OK) return 1;

    ReconLSwapchainDesc sd;
    memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = w; sd.height = h;
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
    fd.width = r->w; fd.height = r->h; fd.seed = 1; fd.lights = &ll;
    ReconLResult b = reconlBeginFrame(r->d, &fd);
    if (b != RECONL_OK) return b;

    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = r->w; rp.viewport_height = r->h;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;
    reconlCmdReset(r->cl);
    reconlCmdBeginRenderPass(r->cl, &rp);
    reconlCmdEndRenderPass(r->cl);
    ReconLResult s = reconlSubmit(r->d, r->cl, NULL);
    if (s != RECONL_OK) return s;

    ReconLPresentDesc p;
    memset(&p, 0, sizeof p);
    SETBASE(&p, RECONL_STRUCT_PRESENT_DESC);
    p.out_pixels = r->px; p.out_pixels_size = (uint64_t)r->w * r->h * 4;
    p.out_row_pitch = r->w * 4; p.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return reconlPresent(r->d, r->s, &p);
}

static const char* backend_name(uint32_t b) {
    switch (b) {
        case RECONL_BACKEND_D3D11: return "d3d11";
        case RECONL_BACKEND_SOFT_CPU: return "soft-cpu";
        case RECONL_BACKEND_NULL: return "null";
        default: return "?";
    }
}

static void show_stats(const ReconLStats* st) {
    printf("  backend %s tier %u/%s reason %u/%s\n",
           backend_name(st->backend), st->tier, reconlTierName((ReconLTier)st->tier),
           st->tier_reason, st->tier_reason_text);
    printf("  presented %u dropped %u failures %u safe-path %u  downgrades %u\n",
           st->frames_presented, st->frames_dropped, st->failures,
           st->safe_path_events, st->downgrade_count);
    printf("  frame cost (host-visible, composed): last %llu us  min %llu us  avg %llu us  max %llu us\n",
           (unsigned long long)(st->frame.total_ns / 1000),
           (unsigned long long)(st->frame.min_ns / 1000),
           (unsigned long long)(st->frame.avg_ns / 1000),
           (unsigned long long)(st->frame.max_ns / 1000));
    for (uint32_t i = 0; i < st->downgrade_count && i < st->downgrade_capacity; ++i) {
        const ReconLDowngrade* e = &st->downgrades[i];
        printf("  downgrade[%u] tier %u -> %u, reason %u/%s, frame %llu: %s\n",
               i, e->from, e->to, e->reason, reconlTierName((ReconLTier)e->to),
               (unsigned long long)e->frame_index, e->detail);
    }
}

static uint32_t num(const char* s, uint32_t dflt) { return s && *s ? (uint32_t)strtoul(s, NULL, 10) : dflt; }

static const char* value_of(int argc, char** argv, const char* key) {
    size_t n = strlen(key);
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], key, n) == 0 && argv[i][n] == '=') return argv[i] + n + 1;
    }
    return NULL;
}

int main(int argc, char** argv) {
    const char* be = value_of(argc, argv, "--backend");
    ReconLBackendId backend = (be && strcmp(be, "soft-cpu") == 0)
        ? RECONL_BACKEND_SOFT_CPU
        : (be && strcmp(be, "null") == 0 ? RECONL_BACKEND_NULL : RECONL_BACKEND_D3D11);
    uint32_t w = num(value_of(argc, argv, "--w"), 1920), h = num(value_of(argc, argv, "--h"), 1080);
    uint32_t target = num(value_of(argc, argv, "--target-ms"), 0);
    uint32_t after = num(value_of(argc, argv, "--after"), 1);
    uint32_t frames = num(value_of(argc, argv, "--frames"), 12);
    uint32_t allow = num(value_of(argc, argv, "--allow"), RECONL_ALLOW_DOWNGRADE_TIER);

    printf("== %s %ux%u target %u ms, threshold %u frames, frames %u, allow %u\n",
           backend_name(backend), w, h, target, after, frames, allow);
    Rig r;
    if (rig(&r, backend, allow, target, after, w, h)) {
        printf("  (unavailable)\n");
        return 2;
    }
    ReconLResult last = RECONL_OK;
    for (uint32_t i = 0; i < frames; ++i) {
        last = full_frame(&r);
        if (last != RECONL_OK) {
            printf("  frame %u -> %d (%s)\n", i, last, reconlResultName(last));
            break;
        }
    }
    ReconLStats st;
    if (!stats_of(&r, &st)) {
        printf("  stats unreadable\n");
    } else {
        show_stats(&st);
    }
    rig_free(&r);
    return 0;
}
