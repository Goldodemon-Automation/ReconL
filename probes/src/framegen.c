/* Frame generation, as a host sees it, through the shipped DLL.
 *
 * This is the surface a game's quality toggle sits on, so the probe checks the
 * three things a toggle needs to be true, all through the exported C functions:
 *
 *   1. a frame that asks for generation can be warped forward, and one that does
 *      not is refused - the request is per frame, not a device mode;
 *   2. a generated image is not a rendered frame: it is counted in
 *      ReconLStats.framegen and never in frames_presented, which is the number
 *      the tier ladder judges;
 *   3. a refused generated frame leaves the device able to render.
 *
 * The scene is drawn in clip space with the identity transform, and the frame
 * declares a camera that moves between frames - so the generator has real motion
 * to warp along. That is the plumbing under test here; the reprojection's
 * accuracy is pinned by the ABI tests, which render world space.
 *
 * Usage: framegen --backend=soft-cpu|d3d11 [--size=64]
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
        (p)->base.next = NULL;                                 \
    } while (0)

static int checks = 0;
static int failures = 0;

static void check(int condition, const char* what) {
    ++checks;
    if (!condition) {
        ++failures;
        printf("  FAIL  %s\n", what);
    }
}

static void* h_alloc(void* user, size_t size, size_t alignment) {
    (void)user;
    size_t align = alignment < 16 ? 16 : (alignment > 4096 ? 4096 : alignment);
    char* base = (char*)malloc(size + align + 64);
    if (!base) return NULL;
    uintptr_t aligned = ((uintptr_t)base + 64 + align - 1) & ~(uintptr_t)(align - 1);
    ((uintptr_t*)aligned)[-1] = (uintptr_t)base;
    return (void*)aligned;
}
static void h_free(void* user, void* ptr, size_t size) {
    (void)user; (void)size;
    if (!ptr) return;
    free((void*)(((uintptr_t*)ptr)[-1]));
}
static void* h_realloc(void* user, void* ptr, size_t old, size_t fresh, size_t alignment) {
    void* moved = h_alloc(user, fresh, alignment);
    if (!moved) return NULL;
    if (ptr) {
        memcpy(moved, ptr, old < fresh ? old : fresh);
        h_free(user, ptr, old);
    }
    return moved;
}

static ReconLBackendId backend = RECONL_BACKEND_SOFT_CPU;
static uint32_t size = 64;

static const char* arg(const char* name, int argc, char** argv) {
    size_t n = strlen(name);
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], name, n) == 0 && argv[i][n] == '=') return argv[i] + n + 1;
    }
    return NULL;
}

/* A world -> camera matrix that is a pure translation: the camera looks down -z
 * from `eye`, which is all the motion the reprojection needs and exactly what a
 * host declares when it slides its camera. Column-major, as the header says. */
static void view_at(float eye_x, float out[16]) {
    memset(out, 0, 16 * sizeof(float));
    out[0] = 1.0f;
    out[5] = 1.0f;
    out[10] = 1.0f;
    out[15] = 1.0f;
    out[12] = -eye_x;
}

static const float IDENTITY[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
};

typedef struct {
    ReconLDevice* device;
    ReconLSwapchain* swapchain;
    ReconLPipeline* pipeline;
    ReconLCommandList* commands;
    ReconLBuffer* vb;
    ReconLBuffer* ib;
} Rig;

static int build(Rig* rig) {
    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    dd.tier_hint = RECONL_TIER_T2_CPU_RAM;
    dd.allow_downgrade = RECONL_ALLOW_DOWNGRADE_NONE;
    dd.worker_threads = 1;
    dd.seed = 1;
    dd.allocator.alloc = h_alloc;
    dd.allocator.realloc = h_realloc;
    dd.allocator.free = h_free;
    if (reconlCreateDevice(&dd, &rig->device) != RECONL_OK) {
        printf("device creation failed\n");
        return 0;
    }

    ReconLSwapchainDesc sd;
    memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = size;
    sd.height = size;
    sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2;
    sd.present_to_memory = 1;
    sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    if (reconlCreateSwapchain(rig->device, &sd, &rig->swapchain) != RECONL_OK) {
        printf("swapchain creation failed\n");
        return 0;
    }

    ReconLPipelineDesc pd;
    memset(&pd, 0, sizeof pd);
    SETBASE(&pd, RECONL_STRUCT_PIPELINE_DESC);
    pd.shading = RECONL_SHADING_LAMBERT;
    pd.blend = RECONL_BLEND_OPAQUE;
    pd.cull = RECONL_CULL_NONE;
    pd.depth_compare = RECONL_COMPARE_GREATER;
    pd.depth_write = 1;
    pd.receives_shadow = 0;
    pd.casts_shadow = 0;
    if (reconlCreatePipeline(rig->device, &pd, &rig->pipeline) != RECONL_OK) {
        printf("pipeline creation failed\n");
        return 0;
    }

    /* A quad that covers the clip-space square, so every pixel has depth and the
     * generator has something to warp. */
    static ReconLVertex verts[6];
    static uint32_t indices[6] = {0, 1, 2, 3, 4, 5};
    const float corners[6][2] = {
        {-1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, -1.0f},
        {-1.0f, 1.0f}, {1.0f, -1.0f}, {-1.0f, -1.0f},
    };
    for (int i = 0; i < 6; ++i) {
        memset(&verts[i], 0, sizeof verts[i]);
        verts[i].position[0] = corners[i][0];
        verts[i].position[1] = corners[i][1];
        verts[i].position[2] = 0.5f;
        verts[i].normal[2] = 1.0f;
        verts[i].color[0] = (i < 3) ? 1.0f : 0.3f;
        verts[i].color[1] = 1.0f;
        verts[i].color[2] = (i < 3) ? 1.0f : 0.3f;
        verts[i].color[3] = 1.0f;
    }

    ReconLBufferDesc vd;
    memset(&vd, 0, sizeof vd);
    SETBASE(&vd, RECONL_STRUCT_BUFFER_DESC);
    vd.size_bytes = sizeof verts;
    vd.usage = RECONL_BUFFER_VERTEX;
    vd.data = verts;
    vd.data_size = sizeof verts;
    if (reconlCreateBuffer(rig->device, &vd, &rig->vb) != RECONL_OK) return 0;

    ReconLBufferDesc id;
    memset(&id, 0, sizeof id);
    SETBASE(&id, RECONL_STRUCT_BUFFER_DESC);
    id.size_bytes = sizeof indices;
    id.usage = RECONL_BUFFER_INDEX;
    id.data = indices;
    id.data_size = sizeof indices;
    if (reconlCreateBuffer(rig->device, &id, &rig->ib) != RECONL_OK) return 0;

    ReconLCommandListDesc cd;
    memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    if (reconlCreateCommandList(rig->device, &cd, &rig->commands) != RECONL_OK) return 0;
    return 1;
}

/* Renders one frame at `eye_x`, asking for generation when `keep` is set, and
 * presents it into `out`. Returns the present result. */
static ReconLResult render(Rig* rig, float eye_x, int keep, unsigned char* out) {
    ReconLLight light;
    memset(&light, 0, sizeof light);
    SETBASE(&light, RECONL_STRUCT_LIGHT);
    light.type = RECONL_LIGHT_DIRECTIONAL;
    light.direction[2] = -1.0f;
    light.color[0] = light.color[1] = light.color[2] = 1.0f;
    light.intensity = 1.0f;

    ReconLLightList lights;
    memset(&lights, 0, sizeof lights);
    SETBASE(&lights, RECONL_STRUCT_LIGHT_LIST);
    lights.count = 1;
    lights.lights = &light;

    ReconLCamera camera;
    memset(&camera, 0, sizeof camera);
    SETBASE(&camera, RECONL_STRUCT_CAMERA);
    view_at(eye_x, camera.view);
    camera.fov_y_deg = 60.0f;
    camera.near = 0.1f;
    camera.far = 100.0f;

    ReconLFrameGenDesc framegen;
    memset(&framegen, 0, sizeof framegen);
    SETBASE(&framegen, RECONL_STRUCT_FRAME_GEN);
    framegen.enabled = keep ? 1u : 0u;

    ReconLFrameDesc fd;
    memset(&fd, 0, sizeof fd);
    SETBASE(&fd, RECONL_STRUCT_FRAME_DESC);
    fd.width = size;
    fd.height = size;
    fd.seed = 1;
    fd.lights = &lights;
    fd.camera = &camera;
    fd.framegen = keep ? &framegen : NULL;

    ReconLResult r = reconlBeginFrame(rig->device, &fd);
    if (r != RECONL_OK) return r;

    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.load_color = 1;
    rp.load_depth = 1;
    rp.clear_color[2] = 1.0f;
    rp.clear_color[3] = 1.0f;
    rp.clear_depth = 0.0f;
    reconlCmdReset(rig->commands);
    if ((r = reconlCmdBeginRenderPass(rig->commands, &rp)) != RECONL_OK) return r;
    if ((r = reconlCmdSetPipeline(rig->commands, rig->pipeline)) != RECONL_OK) return r;
    if ((r = reconlCmdPushConstants(rig->commands, 0, IDENTITY, 64)) != RECONL_OK) return r;
    if ((r = reconlCmdPushConstants(rig->commands, 1, IDENTITY, 64)) != RECONL_OK) return r;
    if ((r = reconlCmdSetVertexBuffer(rig->commands, 0, rig->vb, 0)) != RECONL_OK) return r;
    if ((r = reconlCmdSetIndexBuffer(rig->commands, rig->ib, 0, RECONL_INDEX_UINT32)) != RECONL_OK) return r;
    if ((r = reconlCmdDrawIndexed(rig->commands, 6, 0, 0)) != RECONL_OK) return r;
    if ((r = reconlCmdEndRenderPass(rig->commands)) != RECONL_OK) return r;
    if ((r = reconlSubmit(rig->device, rig->commands, NULL)) != RECONL_OK) return r;

    ReconLPresentDesc prd;
    memset(&prd, 0, sizeof prd);
    SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
    prd.out_pixels = out;
    prd.out_pixels_size = (uint64_t)size * size * 4;
    prd.out_row_pitch = size * 4;
    prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return reconlPresent(rig->device, rig->swapchain, &prd);
}

static ReconLResult generate(Rig* rig, float ahead, unsigned char* out) {
    ReconLPresentDesc prd;
    memset(&prd, 0, sizeof prd);
    SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
    prd.out_pixels = out;
    prd.out_pixels_size = (uint64_t)size * size * 4;
    prd.out_row_pitch = size * 4;
    prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return reconlPresentGenerated(rig->device, rig->swapchain, &prd, ahead);
}

static ReconLStats stats_of(Rig* rig) {
    ReconLStats st;
    memset(&st, 0, sizeof st);
    SETBASE(&st, RECONL_STRUCT_STATS);
    reconlGetStats(rig->device, &st);
    return st;
}

/* How many pixels differ, so "the image moved" is a number and not a hope. */
static long differing(const unsigned char* a, const unsigned char* b) {
    long n = 0;
    for (uint32_t i = 0; i < size * size; ++i) {
        const unsigned char* pa = a + (size_t)i * 4;
        const unsigned char* pb = b + (size_t)i * 4;
        if (pa[0] != pb[0] || pa[1] != pb[1] || pa[2] != pb[2]) ++n;
    }
    return n;
}

int main(int argc, char** argv) {
    const char* name = arg("--backend", argc, argv);
    if (name && strcmp(name, "d3d11") == 0) backend = RECONL_BACKEND_D3D11;
    const char* sz = arg("--size", argc, argv);
    if (sz) size = (uint32_t)strtoul(sz, NULL, 10);

    Rig rig;
    memset(&rig, 0, sizeof rig);
    if (!build(&rig)) return 2;

    ReconLDeviceLimits limits;
    memset(&limits, 0, sizeof limits);
    SETBASE(&limits, RECONL_STRUCT_DEVICE_LIMITS);
    reconlGetDeviceLimits(rig.device, &limits);
    ReconLStats st = stats_of(&rig);
    printf("framegen — %s, %s, %ux%u\n", reconlBackendName(limits.backend), reconlTierName(st.tier), size, size);

    unsigned char* frame = (unsigned char*)malloc((size_t)size * size * 4);
    unsigned char* image = (unsigned char*)malloc((size_t)size * size * 4);
    unsigned char* first = (unsigned char*)malloc((size_t)size * size * 4);
    unsigned char* second = (unsigned char*)malloc((size_t)size * size * 4);

    /* 1. The toggle, per frame. Off first, so the refusal is the first thing a
     *    host that never asks would ever see. */
    ReconLResult r = render(&rig, 0.0f, 0, frame);
    check(r == RECONL_OK, "a frame that asks for nothing still renders");
    check(generate(&rig, 0.5f, image) == RECONL_ERR_NO_FRAME,
          "generation from a frame that never asked is RECONL_ERR_NO_FRAME");
    st = stats_of(&rig);
    check(st.framegen.ready == 0, "a frame that did not ask left nothing ready");

    /* 2. On: the same frame, now keeping its pixels, its depth and its camera. */
    r = render(&rig, 0.5f, 1, frame);
    check(r == RECONL_OK, "a frame that asks for generation renders");
    st = stats_of(&rig);
    check(st.framegen.ready == 1, "the asking frame left an image ready");
    const uint32_t presented_before = st.frames_presented;

    r = generate(&rig, 1.0f, first);
    check(r == RECONL_OK, "a generated image is delivered");
    st = stats_of(&rig);
    check(st.framegen.generated == 1, "the generated image was counted");
    check(st.framegen.generated_ns > 0, "the cost of the generated image was recorded");
    check(st.frames_presented == presented_before,
          "generating an image did NOT move frames_presented (the ladder's number)");
    check(st.framegen.last_ahead == 1.0f, "the look-ahead was reported back");

    /* 3. The motion is real: another frame, the camera moved, and the image the
     *    generator produces is not the frame it came from. */
    r = render(&rig, 1.0f, 1, frame);
    check(r == RECONL_OK, "a second asking frame renders");
    r = generate(&rig, 1.0f, second);
    check(r == RECONL_OK, "a generated image is delivered after a camera move");
    long moved = differing(frame, second);
    printf("  warp            a moving camera changed %ld of %u generated pixels\n", moved, size * size);
    check(moved > 0, "the generated image follows the camera's motion");

    /* 4. Every refusal is an error, and none of them wedges the device. */
    check(generate(&rig, 0.0f, image) == RECONL_ERR_INVALID_ARGUMENT, "ahead 0 is refused");
    check(generate(&rig, -1.0f, image) == RECONL_ERR_INVALID_ARGUMENT, "ahead -1 is refused");
    check(generate(&rig, 2.0f, image) == RECONL_ERR_INVALID_ARGUMENT, "ahead 2 is refused");
    st = stats_of(&rig);
    check(st.framegen.generated == 2, "refused images were not counted as generated");
    check(render(&rig, 1.5f, 0, frame) == RECONL_OK,
          "the device still renders after refused generated frames");

    /* 5. Off again: the game's toggle down, with the descriptor present. */
    r = render(&rig, 2.0f, 0, frame);
    check(r == RECONL_OK, "a frame that asks for nothing renders after an asking one");
    st = stats_of(&rig);
    check(st.framegen.ready == 0, "the toggle down cleared what the last frame kept");
    check(generate(&rig, 0.5f, image) == RECONL_ERR_NO_FRAME,
          "generation is refused once the toggle is down");

    st = stats_of(&rig);
    printf("  counters        frames_presented %u  generated %u  generated_ns %llu  ahead %.2f\n",
           st.frames_presented, st.framegen.generated,
           (unsigned long long)st.framegen.generated_ns, (double)st.framegen.last_ahead);
    printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
