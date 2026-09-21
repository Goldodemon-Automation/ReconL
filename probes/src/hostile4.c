/* Is the frame-state dead end recoverable, and how?
 *
 * Sequence: frame -> submit -> present(too small) fails -> then try, in
 * isolation, each plausible recovery (retry present correctly, submit again,
 * begin). Exactly one of them should say "this is what a host must do".
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

typedef struct { ReconLDevice* d; ReconLSwapchain* s; ReconLCommandList* cl; } Rig;

static int rig(Rig* r, ReconLBackendId backend) {
    memset(r, 0, sizeof *r);
    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    dd.tier_hint = backend == RECONL_BACKEND_D3D11 ? RECONL_TIER_T1_GPU_SHARED : RECONL_TIER_T2_CPU_RAM;
    dd.allow_downgrade = 0;
    dd.target_frame_ms = 1000; dd.downgrade_after_frames = 16; dd.seed = 7;
    dd.allocator.alloc = h_alloc; dd.allocator.realloc = h_realloc; dd.allocator.free = h_free;
    if (reconlCreateDevice(&dd, &r->d) != RECONL_OK) return 1;
    ReconLSwapchainDesc sd;
    memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = 32; sd.height = 32;
    sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2; sd.present_to_memory = 1;
    sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    if (reconlCreateSwapchain(r->d, &sd, &r->s) != RECONL_OK) return 1;
    ReconLCommandListDesc cd;
    memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    cd.capacity_bytes = 1024;
    return reconlCreateCommandList(r->d, &cd, &r->cl) != RECONL_OK;
}

static void rig_free(Rig* r) {
    if (r->cl) reconlRelease(r->cl);
    if (r->s) reconlRelease(r->s);
    if (r->d) reconlRelease(r->d);
}

static ReconLResult frame_and_submit(Rig* r) {
    ReconLLight light; memset(&light, 0, sizeof light);
    SETBASE(&light, RECONL_STRUCT_LIGHT);
    light.type = RECONL_LIGHT_DIRECTIONAL;
    light.direction[2] = -1.0f;
    light.color[0] = light.color[1] = light.color[2] = 1.0f;
    light.intensity = 1.0f;
    ReconLLightList ll; memset(&ll, 0, sizeof ll);
    SETBASE(&ll, RECONL_STRUCT_LIGHT_LIST);
    ll.count = 1; ll.lights = &light;
    ReconLFrameDesc fd; memset(&fd, 0, sizeof fd);
    SETBASE(&fd, RECONL_STRUCT_FRAME_DESC);
    fd.width = 32; fd.height = 32; fd.seed = 1; fd.lights = &ll;
    ReconLResult b = reconlBeginFrame(r->d, &fd);
    if (b != RECONL_OK) return b;
    ReconLRenderPassDesc rp; memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = 32; rp.viewport_height = 32;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;
    reconlCmdReset(r->cl);
    reconlCmdBeginRenderPass(r->cl, &rp);
    reconlCmdEndRenderPass(r->cl);
    return reconlSubmit(r->d, r->cl, NULL);
}

static ReconLPresentDesc present_desc(unsigned char* buf, uint64_t size) {
    ReconLPresentDesc p;
    memset(&p, 0, sizeof p);
    SETBASE(&p, RECONL_STRUCT_PRESENT_DESC);
    p.out_pixels = buf; p.out_pixels_size = size;
    p.out_row_pitch = 32 * 4; p.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return p;
}

static void arm(ReconLBackendId backend, const char* name) {
    printf("  %s:\n", name);
    unsigned char px[32 * 32 * 4];

    /* Recovery 1: retry Present with a correct buffer. */
    {
        Rig r;
        if (rig(&r, backend)) { printf("    (unavailable)\n"); return; }
        ReconLResult s0 = frame_and_submit(&r);
        ReconLPresentDesc bad = present_desc(px, 16);
        ReconLResult p1 = reconlPresent(r.d, r.s, &bad);
        ReconLPresentDesc good = present_desc(px, sizeof px);
        ReconLResult p2 = reconlPresent(r.d, r.s, &good);
        ReconLResult b2 = reconlBeginFrame(r.d, &(ReconLFrameDesc){ 0 });
        printf("    submit %d; present(too small) %d; present(correct) %d; begin(mismatched desc) %d\n",
               (int)s0, (int)p1, (int)p2, (int)b2);
        ReconLResult b3 = frame_and_submit(&r);
        printf("    a full frame afterwards: %d %s\n", (int)b3, b3 == 0 ? "(recovered)" : "(still refused)");
        rig_free(&r);
    }

    /* Recovery 2: submit again. */
    {
        Rig r;
        if (rig(&r, backend)) { printf("    (unavailable)\n"); return; }
        frame_and_submit(&r);
        ReconLPresentDesc bad = present_desc(px, 16);
        ReconLResult p1 = reconlPresent(r.d, r.s, &bad);
        ReconLResult s2 = reconlSubmit(r.d, r.cl, NULL);
        ReconLResult b2 = frame_and_submit(&r);
        printf("    present(too small) %d; submit again %d; frame afterwards %d %s\n",
               (int)p1, (int)s2, (int)b2, b2 == 0 ? "(recovered)" : "(still refused)");
        rig_free(&r);
    }
}

int main(void) {
    printf("frame-state recovery\n");
    arm(RECONL_BACKEND_SOFT_CPU, "soft-cpu");
    arm(RECONL_BACKEND_D3D11, "d3d11");

    /* What each backend reports for limits, side by side. */
    const ReconLBackendId kinds[2] = { RECONL_BACKEND_SOFT_CPU, RECONL_BACKEND_D3D11 };
    const char* names[2] = { "soft-cpu", "d3d11" };
    for (int i = 0; i < 2; ++i) {
        Rig r;
        if (rig(&r, kinds[i])) continue;
        ReconLDeviceLimits lim;
        memset(&lim, 0, sizeof lim);
        SETBASE(&lim, RECONL_STRUCT_DEVICE_LIMITS);
        if (reconlGetDeviceLimits(r.d, &lim) == RECONL_OK) {
            printf("  %s limits: vram=%llu ram=%llu max_allocation=%llu threads_max=%u tile_min=%u max_lights=%u\n",
                   names[i], (unsigned long long)lim.vram_bytes, (unsigned long long)lim.ram_bytes,
                   (unsigned long long)lim.max_allocation_bytes, lim.worker_threads_max,
                   lim.tile_size_min, lim.max_lights);
        }
        rig_free(&r);
    }
    return 0;
}
