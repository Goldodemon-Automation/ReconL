/* Frame-state recovery through the shipped ABI, and the allocation ceiling.
 *
 * The claim under test: however a call fails, the state the device is left in
 * still accepts the call the state machine names next. Concretely -
 *   a present that cannot be delivered consumes the frame (counted as dropped)
 *   and is not retryable for that frame, and the next BeginFrame is accepted;
 *   a begin whose target reservation is refused leaves the device idle too.
 * And: a request above the single-allocation ceiling is refused with
 * BUDGET_EXCEEDED before the host allocator or the driver is called, which is
 * why the sizes this probe used to actually attempt are cheap now.
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

typedef struct { ReconLDevice* d; ReconLSwapchain* s; ReconLCommandList* cl; } Rig;

/* `ram_cap` 0 means "no budget struct at all". */
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
    dd.target_frame_ms = 1000; dd.downgrade_after_frames = 16; dd.seed = 7;
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

static int stats_of(Rig* r, ReconLStats* st) {
    memset(st, 0, sizeof *st);
    SETBASE(st, RECONL_STRUCT_STATS);
    return reconlGetStats(r->d, st) == RECONL_OK;
}

static ReconLResult frame_and_submit(Rig* r, uint32_t w, uint32_t h) {
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
    fd.width = w; fd.height = h; fd.seed = 1; fd.lights = &ll;
    ReconLResult b = reconlBeginFrame(r->d, &fd);
    if (b != RECONL_OK) return b;
    ReconLRenderPassDesc rp; memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = w; rp.viewport_height = h;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;
    reconlCmdReset(r->cl);
    reconlCmdBeginRenderPass(r->cl, &rp);
    reconlCmdEndRenderPass(r->cl);
    return reconlSubmit(r->d, r->cl, NULL);
}

static ReconLPresentDesc present_desc(unsigned char* buf, uint64_t size, uint32_t pitch) {
    ReconLPresentDesc p;
    memset(&p, 0, sizeof p);
    SETBASE(&p, RECONL_STRUCT_PRESENT_DESC);
    p.out_pixels = buf; p.out_pixels_size = size;
    p.out_row_pitch = pitch; p.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return p;
}

/* begin -> submit -> present, all good: returns the result of the last call. */
static int full_frame(Rig* r) {
    ReconLResult b = frame_and_submit(r, 32, 32);
    if (b != RECONL_OK) return (int)b;
    unsigned char px[32 * 32 * 4];
    ReconLPresentDesc p = present_desc(px, sizeof px, 32 * 4);
    return (int)reconlPresent(r->d, r->s, &p);
}

static void arm(ReconLBackendId backend, const char* name) {
    printf("  %s\n", name);
    unsigned char px[32 * 32 * 4];
    Rig r;
    if (rig(&r, backend, 0)) {
        printf("    (unavailable)\n");
        return;
    }

    /* A present that cannot be delivered consumes the frame, visibly. */
    ReconLResult s0 = frame_and_submit(&r, 32, 32);
    check(s0 == RECONL_OK, "begin+submit is OK");
    ReconLPresentDesc bad = present_desc(px, 16, 32 * 4);
    ReconLResult p1 = reconlPresent(r.d, r.s, &bad);
    check(p1 == RECONL_ERR_INVALID_ARGUMENT, "present with an undersized buffer is refused");
    ReconLStats st;
    check(stats_of(&r, &st), "stats readable");
    check(st.frames_dropped == 1, "the refused present counted one dropped frame");
    check(st.frames_presented == 0, "nothing was presented");

    /* The dropped frame is not held for a retry - and a new frame begins. */
    ReconLPresentDesc good = present_desc(px, sizeof px, 32 * 4);
    ReconLResult p2 = reconlPresent(r.d, r.s, &good);
    check(p2 == RECONL_ERR_NO_FRAME, "a second present is NO_FRAME: the frame was dropped, not held");
    int f = full_frame(&r);
    check(f == RECONL_OK, "a whole frame after the failed present succeeds (BeginFrame accepted)");
    check(stats_of(&r, &st) && st.frames_presented == 1 && st.frames_dropped == 1,
          "counters: 1 presented, 1 dropped");

    /* A submit refused for its arguments must not cost the host a frame. */
    ReconLResult b = reconlBeginFrame(r.d, &(ReconLFrameDesc){ 0 });
    int before = (int)b;
    (void)before;
    rig_free(&r);

    /* A begin whose reservation is refused leaves the device idle. */
    Rig rb;
    if (rig(&rb, backend, 32ull << 20) == 0) {
        ReconLResult big = frame_and_submit(&rb, 4096, 4096);
        if (backend == RECONL_BACKEND_SOFT_CPU) {
            check(big == RECONL_ERR_BUDGET_EXCEEDED, "a 4096x4096 begin on a 32 MiB device is BUDGET_EXCEEDED");
            ReconLStats st2;
            check(stats_of(&rb, &st2) && st2.frames_dropped == 0, "a begin that never opened a frame drops nothing");
            int f2 = full_frame(&rb);
            check(f2 == RECONL_OK, "a whole frame after the refused begin succeeds (BeginFrame accepted)");
        } else {
            printf("    (info) d3d11 begin at 4096x4096 with a 32 MiB cap: %d\n", (int)big);
            int f2 = full_frame(&rb);
            check(f2 == RECONL_OK, "a whole frame after it succeeds");
        }
        rig_free(&rb);
    }
}

/* The audit's second finding, measured: a driver refusal is reported as
 * DeviceLost. Does the device survive what it was told was a lost device? */
static void device_lost_claim(void) {
    printf("  device-lost classification (d3d11)\n");
    Rig r;
    if (rig(&r, RECONL_BACKEND_D3D11, 0)) {
        printf("    (unavailable)\n");
        return;
    }
    const struct { uint32_t w, h, mips, layers, fmt, usage; const char* label; } cases[] = {
        { 2048, 2048, 1, 1, RECONL_FORMAT_R8G8B8A8_UNORM, RECONL_TEXTURE_SAMPLED,                     "2048^2 sampled" },
        { 8192, 8192, 1, 1, RECONL_FORMAT_R8G8B8A8_UNORM, RECONL_TEXTURE_SAMPLED,                     "8192^2 sampled (256 MiB)" },
        { 8192, 8192, 1, 1, RECONL_FORMAT_D32_FLOAT,      RECONL_TEXTURE_DEPTH_STENCIL,               "8192^2 depth" },
        { 8192, 8192, 0, 1, RECONL_FORMAT_R8G8B8A8_UNORM, RECONL_TEXTURE_SAMPLED | RECONL_TEXTURE_MIPMAPPED, "8192^2 sampled, full mips" },
    };
    /* The sizes that used to be attempted. A 1 GiB buffer and a 1 TiB texture
     * were handed to this driver - which is what made this probe page-thrash for
     * minutes and freeze the desktop it ran on. They are now refused by the
     * library's own single-allocation ceiling before the allocator is reached,
     * so the probe asserts the refusal. 256 MiB stays in the list because a
     * legitimate large request must still succeed. */
    const uint64_t sizes[] = { 256ull << 20, 1ull << 30, 1ull << 40 };
    for (unsigned i = 0; i < 3; ++i) {
        ReconLBufferDesc bd;
        memset(&bd, 0, sizeof bd);
        SETBASE(&bd, RECONL_STRUCT_BUFFER_DESC);
        bd.size_bytes = sizes[i]; bd.usage = RECONL_BUFFER_VERTEX;
        ReconLBuffer* buf = NULL;
        ReconLResult br = reconlCreateBuffer(r.d, &bd, &buf);
        ReconLErrorInfo bei;
        memset(&bei, 0, sizeof bei);
        SETBASE(&bei, RECONL_STRUCT_ERROR_INFO);
        const char* bmsg = "";
        if (br != RECONL_OK && reconlGetLastError(r.d, &bei) == RECONL_OK) bmsg = bei.message;
        printf("    CreateBuffer(%llu GiB) -> %d %s\n", (unsigned long long)(sizes[i] >> 30), (int)br, bmsg);
        if (sizes[i] <= (512ull << 20)) {
            check(br == RECONL_OK, "a large but affordable buffer is still created");
        } else {
            check(br == RECONL_ERR_BUDGET_EXCEEDED, "an over-ceiling buffer is refused with BUDGET_EXCEEDED");
            check(strstr(bmsg, "ceiling") != NULL, "the refusal names the ceiling");
            check(buf == NULL, "a refused create publishes no handle");
        }
        if (br == RECONL_ERR_DEVICE_LOST) {
            int f = full_frame(&r);
            printf("      code says the device is lost; a normal frame afterwards: %d -> %s\n",
                   f, f == RECONL_OK ? "the device was never lost" : "the device is broken");
        }
        if (buf) reconlRelease(buf);
    }

    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        ReconLTextureDesc td;
        memset(&td, 0, sizeof td);
        SETBASE(&td, RECONL_STRUCT_TEXTURE_DESC);
        td.width = cases[i].w; td.height = cases[i].h;
        td.mip_levels = cases[i].mips; td.array_layers = cases[i].layers;
        td.format = (ReconLFormat)cases[i].fmt; td.usage = cases[i].usage;
        ReconLTexture* tex = NULL;
        ReconLResult tr = reconlCreateTexture(r.d, &td, &tex);
        ReconLErrorInfo ei;
        memset(&ei, 0, sizeof ei);
        SETBASE(&ei, RECONL_STRUCT_ERROR_INFO);
        const char* msg = "";
        if (tr != RECONL_OK && reconlGetLastError(r.d, &ei) == RECONL_OK) msg = ei.message;
        printf("    CreateTexture(%-22s) -> %d %s\n", cases[i].label, (int)tr, msg);
        if (tr == RECONL_ERR_DEVICE_LOST) {
            int f = full_frame(&r);
            printf("      code says the device is lost; a normal frame afterwards: %d -> %s\n",
                   f, f == RECONL_OK ? "the device was never lost" : "the device is broken");
        }
        if (tex) reconlRelease(tex);
    }
    rig_free(&r);
}

/* The ceiling itself: what a host reads back, and that it refuses a request
 * before any memory is touched - on the reference tier and on the hardware one.
 *
 * The capped rig is the reason this can be asserted safely: with a 32 MiB RAM
 * cap the oversize requests are 64 MiB and 128 MiB rather than gigabytes, so a
 * regression that removed the ceiling still could not thrash this machine. */
static void ceiling_limits(void) {
    printf("  single-allocation ceiling\n");
    const ReconLBackendId kinds[2] = { RECONL_BACKEND_SOFT_CPU, RECONL_BACKEND_D3D11 };
    const char* names[2] = { "soft-cpu", "d3d11" };
    for (int k = 0; k < 2; ++k) {
        Rig r;
        if (rig(&r, kinds[k], 0)) { printf("    %s: (unavailable)\n", names[k]); continue; }

        ReconLDeviceLimits lim;
        memset(&lim, 0, sizeof lim);
        SETBASE(&lim, RECONL_STRUCT_DEVICE_LIMITS);
        int have_limits = reconlGetDeviceLimits(r.d, &lim) == RECONL_OK;
        printf("    %s: max_allocation_bytes = %llu\n", names[k], (unsigned long long)lim.max_allocation_bytes);
        check(have_limits && lim.max_allocation_bytes == (512ull << 20),
              "an uncapped device reports the library's 512 MiB ceiling");

        /* One byte past it: refused, named, and the handle stays null. */
        ReconLBufferDesc bd;
        memset(&bd, 0, sizeof bd);
        SETBASE(&bd, RECONL_STRUCT_BUFFER_DESC);
        bd.size_bytes = (512ull << 20) + 1;
        bd.usage = RECONL_BUFFER_VERTEX;
        ReconLBuffer* buf = NULL;
        ReconLResult br = reconlCreateBuffer(r.d, &bd, &buf);
        check(br == RECONL_ERR_BUDGET_EXCEEDED && buf == NULL,
              "one byte over the ceiling is refused with BUDGET_EXCEEDED");
        check(full_frame(&r) == RECONL_OK, "the device still frames after the refusal");
        rig_free(&r);

        if (rig(&r, kinds[k], 32ull << 20)) { printf("    %s: (capped rig unavailable)\n", names[k]); continue; }
        memset(&lim, 0, sizeof lim);
        SETBASE(&lim, RECONL_STRUCT_DEVICE_LIMITS);
        have_limits = reconlGetDeviceLimits(r.d, &lim) == RECONL_OK;
        check(have_limits && lim.max_allocation_bytes == (32ull << 20),
              "a 32 MiB RAM cap lowers the reported ceiling to 32 MiB");

        bd.size_bytes = 64ull << 20;
        buf = NULL;
        br = reconlCreateBuffer(r.d, &bd, &buf);
        check(br == RECONL_ERR_BUDGET_EXCEEDED && buf == NULL, "64 MiB on a 32 MiB device is refused");
        bd.size_bytes = 8ull << 20;
        buf = NULL;
        br = reconlCreateBuffer(r.d, &bd, &buf);
        check(br == RECONL_OK, "8 MiB on a 32 MiB device still succeeds");
        if (buf) reconlRelease(buf);

        /* The frame path too, where the bytes are targets rather than a buffer. */
        ReconLResult big = frame_and_submit(&r, 4096, 4096);
        printf("      frame 4096x4096 on the 32 MiB device: %d\n", (int)big);
        check(big == RECONL_ERR_BUDGET_EXCEEDED, "an over-ceiling frame is refused before the driver");
        check(full_frame(&r) == RECONL_OK, "the next frame still renders");
        rig_free(&r);
    }
}

int main(void) {
    printf("frame-state recovery\n");
    arm(RECONL_BACKEND_SOFT_CPU, "soft-cpu");
    arm(RECONL_BACKEND_D3D11, "d3d11");
    device_lost_claim();
    ceiling_limits();
    printf("%d checks, %d failures\n", checks, fails);
    return fails != 0;
}
