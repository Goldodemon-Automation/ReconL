/* Recovery after a failed call: can the host start the next frame?
 *
 * A host that passes a bad size, a bad present buffer, or an out-of-memory
 * frame has to be able to carry on. If a failure leaves the device believing a
 * frame is open, every later BeginFrame is refused and the host is wedged with
 * no way to clear it - so this walks each failure and then tries to render a
 * normal frame again.
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

static void* h_alloc(void* user, size_t size, size_t alignment) {
    (void)user;
    if (alignment < 16) alignment = 16;
    if (size > 512u * 1024u * 1024u) return NULL;
    size_t total = size + alignment + sizeof(void*);
    unsigned char* raw = (unsigned char*)malloc(total);
    if (!raw) return NULL;
    uintptr_t base = (uintptr_t)raw;
    uintptr_t aligned = (base + sizeof(void*) + alignment - 1) & ~(uintptr_t)(alignment - 1);
    ((void**)aligned)[-1] = raw;
    return (void*)aligned;
}
static void h_free(void* user, void* ptr, size_t size) {
    (void)user; (void)size;
    if (ptr) free(((void**)ptr)[-1]);
}
static void* h_realloc(void* user, void* ptr, size_t o, size_t n, size_t a) {
    void* f = h_alloc(user, n, a);
    if (!f) return NULL;
    if (ptr) { memcpy(f, ptr, o < n ? o : n); h_free(user, ptr, o); }
    return f;
}

static const char* code_name(ReconLResult r) {
    switch (r) {
    case 0: return "OK";
    case -1: return "INVALID_ARGUMENT";
    case -2: return "OUT_OF_MEMORY";
    case -3: return "NOT_SUPPORTED";
    case -4: return "BACKEND_UNAVAILABLE";
    case -5: return "BUDGET_EXCEEDED";
    case -6: return "DEVICE_LOST";
    case -7: return "INVALID_HANDLE";
    case -8: return "STRUCT_SIZE";
    case -11: return "FRAME_IN_PROGRESS";
    case -12: return "NO_FRAME";
    case -13: return "NOT_READY";
    case -17: return "PANIC";
    case -18: return "EMPTY_FRAME";
    default: return "?";
    }
}

typedef struct {
    ReconLDevice* d;
    ReconLSwapchain* s;
    ReconLPipeline* p;
    ReconLCommandList* cl;
} Rig;

static int rig(Rig* r, ReconLBackendId backend, uint64_t ram_cap) {
    memset(r, 0, sizeof *r);
    ReconLMemoryBudget b;
    memset(&b, 0, sizeof b);
    SETBASE(&b, RECONL_STRUCT_MEMORY_BUDGET);
    b.ram_cap_bytes = ram_cap;

    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    dd.tier_hint = backend == RECONL_BACKEND_D3D11 ? RECONL_TIER_T1_GPU_SHARED : RECONL_TIER_T2_CPU_RAM;
    dd.allow_downgrade = 0;
    dd.target_frame_ms = 1000;
    dd.downgrade_after_frames = 16;
    dd.seed = 7;
    dd.budget = ram_cap ? &b : NULL;
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

    ReconLPipelineDesc pd;
    memset(&pd, 0, sizeof pd);
    SETBASE(&pd, RECONL_STRUCT_PIPELINE_DESC);
    pd.shading = RECONL_SHADING_UNLIT;
    pd.depth_compare = RECONL_COMPARE_GREATER;
    if (reconlCreatePipeline(r->d, &pd, &r->p) != RECONL_OK) return 1;

    ReconLCommandListDesc cd;
    memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    cd.capacity_bytes = 1024;
    if (reconlCreateCommandList(r->d, &cd, &r->cl) != RECONL_OK) return 1;
    return 0;
}

static void rig_free(Rig* r) {
    if (r->cl) reconlRelease(r->cl);
    if (r->p) reconlRelease(r->p);
    if (r->s) reconlRelease(r->s);
    if (r->d) reconlRelease(r->d);
}

static ReconLResult begin(Rig* r, uint32_t w, uint32_t h) {
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
    fd.width = w; fd.height = h; fd.seed = 1; fd.lights = &ll;
    return reconlBeginFrame(r->d, &fd);
}

/* Begin, one empty pass, submit. Returns the submit result (0 = fine). */
static ReconLResult render(Rig* r, uint32_t w, uint32_t h) {
    ReconLResult b = begin(r, w, h);
    if (b != RECONL_OK) return b;
    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = w; rp.viewport_height = h;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;
    reconlCmdReset(r->cl);
    reconlCmdBeginRenderPass(r->cl, &rp);
    reconlCmdEndRenderPass(r->cl);
    return reconlSubmit(r->d, r->cl, NULL);
}

static int checks = 0, findings = 0;
#define CHECK(label, cond)                                     \
    do {                                                       \
        ++checks;                                              \
        if (cond) printf("  ok   %s\n", label);                 \
        else { printf("  FIND %s\n", label); ++findings; }      \
    } while (0)

static void arm(ReconLBackendId backend, uint64_t ram_cap, const char* name) {
    printf("  %s:\n", name);
    Rig r;
    if (rig(&r, backend, ram_cap)) { printf("    (device unavailable)\n"); return; }

    unsigned char px[32 * 32 * 4];
    ReconLPresentDesc prd;
    memset(&prd, 0, sizeof prd);
    SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
    prd.out_pixels = px; prd.out_pixels_size = sizeof px;
    prd.out_row_pitch = 32 * 4; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;

    /* Baseline: a normal frame works. */
    ReconLResult r0 = render(&r, 32, 32);
    /* End the submitted frame, so every case below starts from idle: a
     * submitted frame is a frame the state machine is entitled to keep, and
     * testing recovery on top of one measures the probe, not the library. */
    if (r0 == RECONL_OK) reconlPresent(r.d, r.s, &prd);
    if (r0 != RECONL_OK) {
        printf("    baseline frame failed: %d %s\n", (int)r0, code_name(r0));
        CHECK("baseline frame renders", 0);
        rig_free(&r);
        return;
    }
    printf("    baseline frame: %d %s\n", (int)r0, code_name(r0));

    /* A. failed present (buffer too small), then a normal frame. */
    reconlPresent(r.d, r.s, &(ReconLPresentDesc){ 0 }); /* nonsense desc: expect a refusal */
    ReconLPresentDesc small = prd;
    small.out_pixels_size = 16;
    ReconLResult a1 = reconlPresent(r.d, r.s, &small);
    ReconLResult a2 = render(&r, 32, 32);
    if (a2 == RECONL_OK) reconlPresent(r.d, r.s, &prd);
    printf("    after a refused present: present %d %s, next frame %d %s\n",
           (int)a1, code_name(a1), (int)a2, code_name(a2));
    CHECK("a refused present does not wedge the next frame", a2 == RECONL_OK);

    /* B. an absurd frame, then a normal frame. Whether the refusal lands at
     * BeginFrame (the reference tier reserves targets there) or at Submit (the
     * hardware tier reserves between frames), the device must be able to carry
     * on. The absurd frame is finished first if it opened at all.
     *
     * The rig runs with a 64 MiB RAM cap on purpose. 8192x8192 is 512 MiB of
     * colour+depth on the reference tier and 1 GiB on the hardware one; the cap
     * bounds it whatever the library's own ceiling does, so a regression here
     * still cannot page-thrash the machine this probe runs on. An absurd size
     * used to reach the driver - a 16384x16384 frame is 2 GiB - which is what
     * hung this probe. */
    ReconLResult b1 = render(&r, 8192, 8192);
    ReconLPresentDesc big = prd;
    reconlPresent(r.d, r.s, &big);
    ReconLResult b2 = render(&r, 32, 32);
    if (b2 == RECONL_OK) reconlPresent(r.d, r.s, &prd);
    printf("    after an absurd frame: %d %s, next frame %d %s\n",
           (int)b1, code_name(b1), (int)b2, code_name(b2));
    CHECK("an absurd frame is refused with BUDGET_EXCEEDED", b1 == RECONL_ERR_BUDGET_EXCEEDED);
    CHECK("an absurd frame does not wedge the next frame", b2 == RECONL_OK);

    /* C. present with no frame open at all. */
    ReconLResult c1 = reconlPresent(r.d, r.s, &prd);
    ReconLResult c2 = render(&r, 32, 32);
    if (c2 == RECONL_OK) reconlPresent(r.d, r.s, &prd);
    printf("    after present with no frame: present %d %s, next frame %d %s\n",
           (int)c1, code_name(c1), (int)c2, code_name(c2));
    CHECK("present with no frame does not wedge the next frame", c2 == RECONL_OK);

    /* D. begin twice without submitting. */
    ReconLResult d1 = begin(&r, 32, 32);
    ReconLResult d2 = begin(&r, 32, 32);
    /* The frame d1 opened is still open: end it the documented way, then a
     * fresh frame must work. */
    if (d1 == RECONL_OK) {
        ReconLRenderPassDesc rp;
        memset(&rp, 0, sizeof rp);
        SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
        rp.viewport_width = 32; rp.viewport_height = 32;
        rp.load_color = 1; rp.load_depth = 1;
        reconlCmdReset(r.cl);
        reconlCmdBeginRenderPass(r.cl, &rp);
        reconlCmdEndRenderPass(r.cl);
        reconlSubmit(r.d, r.cl, NULL);
        reconlPresent(r.d, r.s, &prd);
    }
    ReconLResult d3 = render(&r, 32, 32);
    if (d3 == RECONL_OK) reconlPresent(r.d, r.s, &prd);
    printf("    begin twice: %d %s then %d %s; next frame %d %s\n",
           (int)d1, code_name(d1), (int)d2, code_name(d2), (int)d3, code_name(d3));
    CHECK("a doubled BeginFrame is refused and the device recovers", d2 != RECONL_OK && d3 == RECONL_OK);

    rig_free(&r);
}

int main(void) {
    printf("recovery after failure\n");
    /* 64 MiB of budget for every rig below: the absurd sizes then stay cheap and
     * a regression cannot commit gigabytes of this machine. */
    const uint64_t cap = 64ull << 20;
    arm(RECONL_BACKEND_SOFT_CPU, cap, "soft-cpu");
    arm(RECONL_BACKEND_D3D11, cap, "d3d11");

    /* What the device says its own limits are, next to what it does with them. */
    {
        Rig r;
        if (rig(&r, RECONL_BACKEND_D3D11, 0) == 0) {
            ReconLDeviceLimits lim;
            memset(&lim, 0, sizeof lim);
            SETBASE(&lim, RECONL_STRUCT_DEVICE_LIMITS);
            if (reconlGetDeviceLimits(r.d, &lim) == RECONL_OK) {
                printf("  d3d11 limits: vram=%llu ram=%llu max_allocation=%llu worker_threads_max=%u\n",
                       (unsigned long long)lim.vram_bytes, (unsigned long long)lim.ram_bytes,
                       (unsigned long long)lim.max_allocation_bytes, lim.worker_threads_max);
                printf("  (a 16384x16384 frame needs 2 GiB of colour+depth, refused by the ceiling)\n");
                CHECK("an uncapped device reports the concrete ceiling",
                      lim.max_allocation_bytes == (512ull << 20));
            }
            rig_free(&r);
        }
    }

    printf("hostile3: %d checks, %d findings\n", checks, findings);
    return 0;
}
