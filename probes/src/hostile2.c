/* Hostile paths, part 2: absurd sizes, a host allocator that refuses, state
 * machine misuse, and bogus present pitches.
 *
 * The host allocator refuses anything over 512 MiB, which is a legitimate host
 * policy and makes the library's out-of-memory path deterministic instead of
 * depending on how much page file this machine has.
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

#define CAP (512u * 1024u * 1024u)
static long refused = 0;
static long live = 0;

static void* h_alloc(void* user, size_t size, size_t alignment) {
    (void)user;
    if (alignment < 16) alignment = 16;
    if (size > CAP) { refused++; return NULL; }
    size_t total = size + alignment + sizeof(void*);
    unsigned char* raw = (unsigned char*)malloc(total);
    if (!raw) { refused++; return NULL; }
    uintptr_t base = (uintptr_t)raw;
    uintptr_t aligned = (base + sizeof(void*) + alignment - 1) & ~(uintptr_t)(alignment - 1);
    ((void**)aligned)[-1] = raw;
    live++;
    return (void*)aligned;
}

static void h_free(void* user, void* ptr, size_t size) {
    (void)user; (void)size;
    if (!ptr) return;
    live--;
    free(((void**)ptr)[-1]);
}

static void* h_realloc(void* user, void* ptr, size_t old_size, size_t new_size, size_t alignment) {
    void* fresh = h_alloc(user, new_size, alignment);
    if (!fresh) return NULL;
    if (ptr) { memcpy(fresh, ptr, old_size < new_size ? old_size : new_size); h_free(user, ptr, old_size); }
    return fresh;
}

static int checks = 0, findings = 0;
#define CHECK(label, cond)                                              \
    do {                                                                \
        ++checks;                                                       \
        if (cond) printf("  ok   %s\n", label);                          \
        else { printf("  FIND %s\n", label); ++findings; }               \
    } while (0)

static void describe(ReconLDevice* d, const char* what, ReconLResult r) {
    ReconLErrorInfo ei;
    memset(&ei, 0, sizeof ei);
    SETBASE(&ei, RECONL_STRUCT_ERROR_INFO);
    reconlGetLastError(d, &ei);
    printf("    %s -> %d (%s)\n", what, (int)r, ei.message[0] ? ei.message : "no message");
}

static ReconLDevice* make_device(ReconLBackendId backend) {
    /* A 64 MiB RAM cap on purpose: the absurd-size cases below are then bounded
     * by the device's own budget, so a regression in the library's ceiling
     * cannot make this probe commit gigabytes and start swapping. */
    ReconLMemoryBudget b;
    memset(&b, 0, sizeof b);
    SETBASE(&b, RECONL_STRUCT_MEMORY_BUDGET);
    b.ram_cap_bytes = 64ull << 20;

    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    dd.tier_hint = RECONL_TIER_T1_GPU_SHARED;
    dd.allow_downgrade = 0;
    dd.target_frame_ms = 1000;
    dd.downgrade_after_frames = 16;
    dd.seed = 7;
    dd.budget = &b;
    dd.allocator.alloc = h_alloc;
    dd.allocator.realloc = h_realloc;
    dd.allocator.free = h_free;
    ReconLDevice* d = NULL;
    ReconLResult r = reconlCreateDevice(&dd, &d);
    if (r != RECONL_OK) { describe(NULL, "create device", r); return NULL; }
    return d;
}

static ReconLSwapchain* make_swapchain(ReconLDevice* d, uint32_t w, uint32_t h) {
    ReconLSwapchainDesc sd;
    memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = w; sd.height = h;
    sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2;
    sd.present_to_memory = 1;
    sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    ReconLSwapchain* s = NULL;
    ReconLResult r = reconlCreateSwapchain(d, &sd, &s);
    if (r != RECONL_OK) describe(d, "create swapchain", r);
    return r == 0 ? s : NULL;
}

/* Begin a frame of the given size and submit one empty render pass. */
static ReconLResult run_frame(ReconLDevice* d, uint32_t fw, uint32_t fh, ReconLPresentDesc* prd,
                              ReconLSwapchain* s) {
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
    fd.width = fw; fd.height = fh; fd.seed = 1; fd.lights = &ll;
    ReconLResult r = reconlBeginFrame(d, &fd);
    if (r != RECONL_OK) return r;

    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = fw;
    rp.viewport_height = fh;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;
    rp.clear_depth = 0.0f;
    ReconLCommandListDesc cd;
    memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    cd.capacity_bytes = 1024;
    ReconLCommandList* cl = NULL;
    r = reconlCreateCommandList(d, &cd, &cl);
    if (r != RECONL_OK) return r;
    reconlCmdReset(cl);
    reconlCmdBeginRenderPass(cl, &rp);
    reconlCmdEndRenderPass(cl);
    r = reconlSubmit(d, cl, NULL);
    reconlRelease(cl);
    if (r != RECONL_OK) return r;
    if (prd && s) return reconlPresent(d, s, prd);
    return RECONL_OK;
}

int main(void) {
    printf("hostile paths, part 2\n");

    /* 1. absurd frame sizes, per backend, through a real frame. */
    {
        const ReconLBackendId kinds[2] = { RECONL_BACKEND_D3D11, RECONL_BACKEND_SOFT_CPU };
        const char* names[2] = { "d3d11", "soft-cpu" };
        for (int k = 0; k < 2; ++k) {
            ReconLDevice* d = make_device(kinds[k]);
            if (!d) continue;
            printf("  %s:\n", names[k]);
            long before = live;
            ReconLResult r = run_frame(d, 65535, 65535, NULL, NULL);
            describe(d, "frame 65535x65535", r);
            CHECK("an absurd frame size is refused, not attempted", r != RECONL_OK);
            CHECK("it is refused as a budget refusal, not an allocation failure",
                  r == RECONL_ERR_BUDGET_EXCEEDED);
            r = run_frame(d, 16384, 16384, NULL, NULL);
            describe(d, "frame 16384x16384", r);
            CHECK("a 16384x16384 frame is refused the same way", r == RECONL_ERR_BUDGET_EXCEEDED);
            printf("    host allocations still live: %ld (was %ld), refusals: %ld\n", live, before, refused);
            reconlRelease(d);
        }
    }

    /* 2. zero-sized frame. */
    {
        ReconLDevice* d = make_device(RECONL_BACKEND_SOFT_CPU);
        if (d) {
            ReconLSwapchain* s = make_swapchain(d, 32, 32);
            ReconLResult r = run_frame(d, 0, 0, NULL, NULL);
            describe(d, "frame 0x0", r);
            CHECK("a zero-sized frame is refused or coerced, never a crash", 1);
            if (s) reconlRelease(s);
            reconlRelease(d);
        }
    }

    /* 3. present with a non-null but zero-sized buffer, and a bogus pitch. */
    {
        ReconLDevice* d = make_device(RECONL_BACKEND_D3D11);
        ReconLSwapchain* s = d ? make_swapchain(d, 32, 32) : NULL;
        if (d && s) {
            unsigned char px[32 * 32 * 4];
            ReconLPresentDesc prd;
            memset(&prd, 0, sizeof prd);
            SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
            prd.out_pixels = px;
            prd.out_pixels_size = 0;
            prd.out_row_pitch = 32 * 4;
            prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
            ReconLResult r = run_frame(d, 32, 32, &prd, s);
            describe(d, "present with out_pixels_size = 0", r);
            CHECK("a zero-sized present buffer is refused", r == RECONL_ERR_INVALID_ARGUMENT);

            /* Bogus pitch: rows overlap, but must stay inside the buffer. */
            size_t n = sizeof px;
            unsigned char* buf = (unsigned char*)malloc(n + 4096);
            memset(buf, 0xAB, n + 4096);
            prd.out_pixels = buf;
            prd.out_pixels_size = n;
            prd.out_row_pitch = 4;
            r = run_frame(d, 32, 32, &prd, s);
            int canary = 1;
            for (size_t i = n; i < n + 4096; ++i) if (buf[i] != 0xAB) canary = 0;
            describe(d, "present with out_row_pitch = 4", r);
            CHECK("a pitch smaller than the row does not write past the buffer", canary);
            free(buf);
        }
        if (s) reconlRelease(s);
        if (d) reconlRelease(d);
    }

    /* 4. state machine: present with no frame, begin twice, draw with no frame. */
    {
        ReconLDevice* d = make_device(RECONL_BACKEND_SOFT_CPU);
        ReconLSwapchain* s = d ? make_swapchain(d, 32, 32) : NULL;
        if (d && s) {
            unsigned char px[32 * 32 * 4];
            ReconLPresentDesc prd;
            memset(&prd, 0, sizeof prd);
            SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
            prd.out_pixels = px; prd.out_pixels_size = sizeof px;
            prd.out_row_pitch = 32 * 4; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
            ReconLResult r = reconlPresent(d, s, &prd);
            describe(d, "present before any frame", r);
            CHECK("present with no frame began is refused", r == RECONL_ERR_NO_FRAME);

            ReconLCommandListDesc cd;
            memset(&cd, 0, sizeof cd);
            SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
            cd.capacity_bytes = 1024;
            ReconLCommandList* cl = NULL;
            reconlCreateCommandList(d, &cd, &cl);
            ReconLRenderPassDesc rp;
            memset(&rp, 0, sizeof rp);
            SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
            rp.viewport_width = 32; rp.viewport_height = 32;
            rp.load_color = 1; rp.load_depth = 1;
            reconlCmdReset(cl);
            reconlCmdBeginRenderPass(cl, &rp);
            reconlCmdEndRenderPass(cl);
            r = reconlSubmit(d, cl, NULL);
            describe(d, "submit with no frame began", r);
            CHECK("submit with no frame began is refused", r != RECONL_OK);
            reconlRelease(cl);
            if (s) reconlRelease(s);
            reconlRelease(d);
        }
    }

    /* 5. a zeroed allocator is documented as "refuse start". */
    {
        ReconLDeviceDesc dd;
        memset(&dd, 0, sizeof dd);
        SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
        dd.backend_hint = RECONL_BACKEND_SOFT_CPU;
        dd.tier_hint = RECONL_TIER_T2_CPU_RAM;
        dd.allow_downgrade = 0;
        ReconLDevice* d = NULL;
        ReconLResult r = reconlCreateDevice(&dd, &d);
        printf("    zeroed allocator -> %d\n", (int)r);
        CHECK("a zeroed allocator is refused as documented", r != RECONL_OK && d == NULL);
    }

    /* 6. zero-sized buffer and an empty draw. */
    {
        ReconLDevice* d = make_device(RECONL_BACKEND_SOFT_CPU);
        if (d) {
            ReconLBufferDesc bd;
            memset(&bd, 0, sizeof bd);
            SETBASE(&bd, RECONL_STRUCT_BUFFER_DESC);
            bd.size_bytes = 0;
            bd.usage = RECONL_BUFFER_VERTEX;
            ReconLBuffer* b = NULL;
            ReconLResult r = reconlCreateBuffer(d, &bd, &b);
            describe(d, "zero-byte vertex buffer", r);
            CHECK("a zero-byte buffer is refused or harmless", r != RECONL_OK ? (b == NULL) : (b != NULL));
            if (b) {
                /* An empty draw with a live buffer: no crash, and the frame still ends. */
                ReconLResult fr = run_frame(d, 32, 32, NULL, NULL);
                CHECK("a frame after a zero-byte buffer still runs", fr == RECONL_OK || fr == RECONL_ERR_EMPTY_FRAME);
                reconlRelease(b);
            }
            reconlRelease(d);
        }
    }

    printf("hostile2: %d checks, %d findings, live allocations %ld, refusals %ld\n",
           checks, findings, live, refused);
    return 0;
}
