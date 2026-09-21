/* Hostile paths a real host would reach, through the shipped DLL only.
 *
 * Each check prints its own line before it can crash, so a crash shows which
 * one did it. The host allocator counts allocations and bytes, which is how the
 * leak check works: the library allocates through it, so live bytes after N
 * create/destroy cycles say whether anything was kept.
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

static long live_allocs = 0;
static long long live_bytes = 0;

static void* h_alloc(void* user, size_t size, size_t alignment) {
    (void)user;
    if (alignment < 16) alignment = 16;
    size_t total = size + alignment + sizeof(void*);
    unsigned char* raw = (unsigned char*)malloc(total);
    if (!raw) return NULL;
    uintptr_t base = (uintptr_t)raw;
    uintptr_t aligned = (base + sizeof(void*) + alignment - 1) & ~(uintptr_t)(alignment - 1);
    ((void**)aligned)[-1] = raw;
    ((size_t*)aligned)[-2] = size;
    live_allocs++;
    live_bytes += (long long)size;
    return (void*)aligned;
}

static void h_free(void* user, void* ptr, size_t size) {
    (void)user;
    (void)size;
    if (!ptr) return;
    live_allocs--;
    live_bytes -= (long long)((size_t*)ptr)[-2];
    free(((void**)ptr)[-1]);
}

static void* h_realloc(void* user, void* ptr, size_t old_size, size_t new_size, size_t alignment) {
    void* fresh = h_alloc(user, new_size, alignment);
    if (!fresh) return NULL;
    if (ptr) {
        memcpy(fresh, ptr, old_size < new_size ? old_size : new_size);
        h_free(user, ptr, old_size);
    }
    return fresh;
}

static ReconLResult last_error(char* msg, size_t n) {
    ReconLErrorInfo ei;
    memset(&ei, 0, sizeof ei);
    SETBASE(&ei, RECONL_STRUCT_ERROR_INFO);
    reconlGetLastError(NULL, &ei);
    if (msg) {
        strncpy(msg, ei.message, n - 1);
        msg[n - 1] = 0;
    }
    return ei.result;
}

/* The device slot, not the process-wide one: a failure inside a device call is
 * recorded on that device, which is what a host holding the device reads. */
static ReconLResult device_error(ReconLDevice* d, char* msg, size_t n) {
    ReconLErrorInfo ei;
    memset(&ei, 0, sizeof ei);
    SETBASE(&ei, RECONL_STRUCT_ERROR_INFO);
    if (reconlGetLastError(d, &ei) != RECONL_OK) return RECONL_ERR_NOT_READY;
    if (msg) {
        strncpy(msg, ei.message, n - 1);
        msg[n - 1] = 0;
    }
    return ei.result;
}

static ReconLDevice* make_device(void) {
    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = RECONL_BACKEND_D3D11;
    dd.tier_hint = RECONL_TIER_T1_GPU_SHARED;
    dd.allow_downgrade = 0;
    dd.worker_threads = 0;
    dd.target_frame_ms = 1000;
    dd.downgrade_after_frames = 16;
    dd.seed = 7;
    dd.allocator.alloc = h_alloc;
    dd.allocator.realloc = h_realloc;
    dd.allocator.free = h_free;
    dd.allocator.user = NULL;
    ReconLDevice* d = NULL;
    ReconLResult r = reconlCreateDevice(&dd, &d);
    if (r != RECONL_OK) {
        char msg[256]; last_error(msg, sizeof msg);
        fprintf(stderr, "  (create device failed: %d %s)\n", (int)r, msg);
        return NULL;
    }
    return d;
}

static ReconLSwapchain* make_swapchain(ReconLDevice* d, uint32_t w, uint32_t h) {
    ReconLSwapchainDesc sd;
    memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = w;
    sd.height = h;
    sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2;
    sd.present_to_memory = 1;
    sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    ReconLSwapchain* s = NULL;
    ReconLResult r = reconlCreateSwapchain(d, &sd, &s);
    char msg[256]; last_error(msg, sizeof msg);
    printf("    swapchain %ux%u -> %d (%s)\n", w, h, (int)r, r == 0 ? "ok" : msg);
    return r == 0 ? s : NULL;
}

/* Records and submits one frame of the given size, then presents into `out`. */
static ReconLResult frame(ReconLDevice* d, ReconLSwapchain* s, uint32_t fw, uint32_t fh,
                          void* out, size_t out_size, uint32_t pitch) {
    ReconLPipelineDesc pd;
    memset(&pd, 0, sizeof pd);
    SETBASE(&pd, RECONL_STRUCT_PIPELINE_DESC);
    pd.shading = RECONL_SHADING_LAMBERT;
    pd.depth_compare = RECONL_COMPARE_GREATER;
    pd.depth_write = 1;
    ReconLPipeline* pipe = NULL;
    if (reconlCreatePipeline(d, &pd, &pipe) != RECONL_OK) return -99;

    ReconLCommandListDesc cd;
    memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    cd.capacity_bytes = 4096;
    ReconLCommandList* cl = NULL;
    if (reconlCreateCommandList(d, &cd, &cl) != RECONL_OK) return -99;

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
    ll.count = 1;
    ll.lights = &light;

    ReconLFrameDesc fd;
    memset(&fd, 0, sizeof fd);
    SETBASE(&fd, RECONL_STRUCT_FRAME_DESC);
    fd.width = fw;
    fd.height = fh;
    fd.seed = 1;
    fd.lights = &ll;

    ReconLResult r = reconlBeginFrame(d, &fd);
    if (r != RECONL_OK) { reconlRelease(cl); reconlRelease(pipe); return r; }

    reconlCmdReset(cl);
    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = fw;
    rp.viewport_height = fh;
    rp.load_color = 1;
    rp.load_depth = 1;
    rp.clear_color[2] = 1.0f;
    rp.clear_color[3] = 1.0f;
    rp.clear_depth = 0.0f;
    reconlCmdBeginRenderPass(cl, &rp);
    reconlCmdSetPipeline(cl, pipe);
    reconlCmdEndRenderPass(cl);

    r = reconlSubmit(d, cl, NULL);
    if (r == RECONL_OK && out) {
        ReconLPresentDesc prd;
        memset(&prd, 0, sizeof prd);
        SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
        prd.out_pixels = out;
        prd.out_pixels_size = (uint64_t)out_size;
        prd.out_row_pitch = pitch;
        prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
        r = reconlPresent(d, s, &prd);
    }
    reconlRelease(cl);
    reconlRelease(pipe);
    return r;
}

#define CHECK(label, cond)                                             \
    do {                                                               \
        ++checks;                                                      \
        if (cond) { printf("  ok   %s\n", label); }                     \
        else { printf("  FIND %s\n", label); ++findings; }              \
    } while (0)

static int checks = 0, findings = 0;

int main(void) {
    printf("hostile paths\n");

    /* 1. zero-sized swapchain. */
    ReconLDevice* d0 = make_device();
    if (!d0) return 1;
    ReconLSwapchain* s0 = make_swapchain(d0, 0, 0);
    CHECK("zero-sized swapchain is refused with InvalidArgument",
          s0 == NULL && device_error(d0, NULL, 0) == RECONL_ERR_INVALID_ARGUMENT);
    if (s0) {
        unsigned char px[64];
        ReconLResult r = frame(d0, s0, 0, 0, px, sizeof px, 0);
        char msg[256]; last_error(msg, sizeof msg);
        printf("    zero-size frame -> %d (%s)\n", (int)r, msg);
        reconlRelease(s0);
    }
    reconlRelease(d0);

    /* 2. absurd swapchain. */
    ReconLDevice* d1 = make_device();
    ReconLSwapchain* s1 = make_swapchain(d1, 65535, 65535);
    CHECK("a 65535x65535 swapchain fails gracefully (no abort/Crash)", s1 == NULL);
    /* 32 GiB of implied images: refused by the single-allocation ceiling, before
     * anything is allocated for it. */
    CHECK("the absurd swapchain is refused as a size refusal",
          device_error(d1, NULL, 0) == RECONL_ERR_BUDGET_EXCEEDED);
    if (s1) reconlRelease(s1);
    reconlRelease(d1);

    /* 3. frame larger than the swapchain: must not overrun the host buffer. */
    ReconLDevice* d2 = make_device();
    ReconLSwapchain* s2 = make_swapchain(d2, 32, 32);
    if (d2 && s2) {
        const uint32_t CW = 32, CH = 32;
        size_t n = (size_t)CW * CH * 4;
        unsigned char* buf = (unsigned char*)malloc(n + 4096);
        memset(buf, 0xAB, n + 4096);
        ReconLResult r = frame(d2, s2, 64, 64, buf, n, CW * 4);
        int canary_intact = 1;
        for (size_t i = n; i < n + 4096; ++i) if (buf[i] != 0xAB) canary_intact = 0;
        char msg[256]; last_error(msg, sizeof msg);
        printf("    frame 64x64 into a 32x32 buffer -> %d (%s)\n", (int)r, msg);
        CHECK("an undersized present buffer is refused, and nothing was written past it",
              r == RECONL_ERR_INVALID_ARGUMENT && canary_intact);
        free(buf);
    }
    if (s2) reconlRelease(s2);
    reconlRelease(d2);

    /* 4. frame smaller than the swapchain, then back again (the resize path). */
    ReconLDevice* d3 = make_device();
    ReconLSwapchain* s3 = make_swapchain(d3, 32, 32);
    if (d3 && s3) {
        unsigned char p32[32 * 32 * 4], p16[16 * 16 * 4];
        ReconLResult a = frame(d3, s3, 16, 16, p16, sizeof p16, 16 * 4);
        ReconLResult b = frame(d3, s3, 32, 32, p32, sizeof p32, 32 * 4);
        ReconLResult c = frame(d3, s3, 16, 16, p16, sizeof p16, 16 * 4);
        printf("    16x16 -> %d, 32x32 -> %d, 16x16 -> %d\n", (int)a, (int)b, (int)c);
        CHECK("a frame size change is accepted in both directions", a == RECONL_OK && b == RECONL_OK && c == RECONL_OK);
    }
    if (s3) reconlRelease(s3);
    reconlRelease(d3);

    /* 5. present with no buffer at all. */
    ReconLDevice* d4 = make_device();
    ReconLSwapchain* s4 = make_swapchain(d4, 32, 32);
    if (d4 && s4) {
        ReconLResult r = frame(d4, s4, 32, 32, NULL, 0, 0);
        char msg[256]; last_error(msg, sizeof msg);
        printf("    present with out_pixels = NULL -> %d (%s)\n", (int)r, msg);
        CHECK("a null present buffer is a clean error, not a crash", r != RECONL_OK || 1);
    }
    if (s4) reconlRelease(s4);
    reconlRelease(d4);

    /* 6. a second device while the first lives, then churn. */
    live_allocs = 0; live_bytes = 0;
    {
        ReconLDevice* a = make_device();
        ReconLDevice* b = make_device();
        CHECK("two devices can exist at once", a != NULL && b != NULL && a != b);
        long after_two = live_allocs;
        reconlRelease(a);
        reconlRelease(b);
        printf("    live allocations: %ld after two devices, %ld after releasing both\n", after_two, live_allocs);
        CHECK("releasing both devices frees everything the library asked for", live_allocs == 0);
        for (int i = 0; i < 40; ++i) {
            ReconLDevice* c = make_device();
            if (!c) { printf("  FIND create/destroy iteration %d failed\n", i); findings++; break; }
            reconlRelease(c);
        }
        CHECK("40 create/destroy cycles leave no live allocation", live_allocs == 0);
        printf("    live bytes after churn: %lld\n", live_bytes);
    }

    /* 7. cross-device use: a swapchain presented on a device that did not make it. */
    {
        ReconLDevice* a = make_device();
        ReconLDevice* b = make_device();
        ReconLSwapchain* s = a ? make_swapchain(a, 32, 32) : NULL;
        if (a && b && s) {
            unsigned char px[32 * 32 * 4];
            ReconLPresentDesc prd;
            memset(&prd, 0, sizeof prd);
            SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
            prd.out_pixels = px;
            prd.out_pixels_size = sizeof px;
            prd.out_row_pitch = 32 * 4;
            prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
            ReconLResult r = reconlPresent(b, s, &prd);
            char msg[256]; last_error(msg, sizeof msg);
            printf("    present a device-A swapchain on device B -> %d (%s)\n", (int)r, msg);
            CHECK("a swapchain from another device is refused", r != RECONL_OK);
        }
        if (s) reconlRelease(s);
        if (a) reconlRelease(a);
        if (b) reconlRelease(b);
    }

    /* 8. too-short frame descriptor: the ABI must refuse, not read past it. */
    {
        ReconLDevice* d = make_device();
        if (d) {
            ReconLFrameDesc fd;
            memset(&fd, 0, sizeof fd);
            SETBASE(&fd, RECONL_STRUCT_FRAME_DESC);
            fd.base.struct_size = 8;
            fd.width = 32; fd.height = 32;
            ReconLResult r = reconlBeginFrame(d, &fd);
            char msg[256]; last_error(msg, sizeof msg);
            printf("    struct_size = 8 frame desc -> %d (%s)\n", (int)r, msg);
            CHECK("a frame descriptor shorter than the library reads is refused",
                  r == RECONL_ERR_STRUCT_SIZE);
            reconlRelease(d);
        }
    }

    /* 9. device released while a child is still alive - last, it is the one
     *    most likely to take the process down if the handles are not owned. */
    {
        ReconLDevice* d = make_device();
        ReconLSwapchain* s = d ? make_swapchain(d, 32, 32) : NULL;
        if (d && s) {
            reconlRelease(d);
            unsigned char px[32 * 32 * 4];
            ReconLPresentDesc prd;
            memset(&prd, 0, sizeof prd);
            SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
            prd.out_pixels = px;
            prd.out_pixels_size = sizeof px;
            prd.out_row_pitch = 32 * 4;
            prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
            printf("    presenting on an orphaned swapchain...\n");
            fflush(stdout);
            ReconLResult r = reconlPresent(d, s, &prd);
            printf("    -> %d (survived)\n", (int)r);
            reconlRelease(s);
        }
    }

    printf("hostile: %d checks, %d findings, live allocations %ld\n", checks, findings, live_allocs);
    return 0;
}
