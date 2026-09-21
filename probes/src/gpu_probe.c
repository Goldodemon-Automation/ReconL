/* A C host against the shipped DLL: the same surface a user links.
 *
 * Checks the three things the backend has to deliver, through the public
 * header only:
 *   1. the probe advertises the hardware tier it can actually create,
 *   2. a frame recorded through the ABI comes back with pixels that are not
 *      the clear colour, and reports the triangles it drew,
 *   3. two identical frames produce identical bytes.
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

/* The host allocator: header-slot base pointer, 16-byte alignment. */
static void* h_alloc(void* user, size_t size, size_t alignment) {
    (void)user;
    if (alignment < 16) alignment = 16;
    size_t total = size + alignment + sizeof(void*);
    unsigned char* raw = (unsigned char*)malloc(total);
    if (!raw) return NULL;
    uintptr_t base = (uintptr_t)raw;
    uintptr_t aligned = (base + sizeof(void*) + alignment - 1) & ~(uintptr_t)(alignment - 1);
    ((void**)aligned)[-1] = raw;
    return (void*)aligned;
}

static void h_free(void* user, void* ptr, size_t size) {
    (void)user;
    (void)size;
    if (!ptr) return;
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

#define W 32u
#define H 32u

typedef struct Rig {
    ReconLDevice* device;
    ReconLSwapchain* swapchain;
    ReconLPipeline* pipeline;
    ReconLCommandList* commands;
    unsigned char pixels[W * H * 4];
} Rig;

static int fail(const char* what, ReconLResult r) {
    fprintf(stderr, "FAIL %s: %d\n", what, (int)r);
    return 1;
}

static int rig_init(Rig* rig) {
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

    ReconLResult r = reconlCreateDevice(&dd, &rig->device);
    if (r != RECONL_OK) {
        ReconLErrorInfo ei;
        memset(&ei, 0, sizeof ei);
        SETBASE(&ei, RECONL_STRUCT_ERROR_INFO);
        reconlGetLastError(NULL, &ei);
        fprintf(stderr, "create device: %d %s\n", (int)r, ei.message);
        return 1;
    }

    ReconLSwapchainDesc sd;
    memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = W;
    sd.height = H;
    sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2;
    sd.present_to_memory = 1;
    sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    r = reconlCreateSwapchain(rig->device, &sd, &rig->swapchain);
    if (r != RECONL_OK) return fail("create swapchain", r);

    ReconLPipelineDesc pd;
    memset(&pd, 0, sizeof pd);
    SETBASE(&pd, RECONL_STRUCT_PIPELINE_DESC);
    pd.shading = RECONL_SHADING_LAMBERT;
    pd.blend = RECONL_BLEND_OPAQUE;
    pd.cull = RECONL_CULL_NONE;
    pd.depth_compare = RECONL_COMPARE_GREATER;
    pd.depth_write = 1;
    r = reconlCreatePipeline(rig->device, &pd, &rig->pipeline);
    if (r != RECONL_OK) return fail("create pipeline", r);

    ReconLCommandListDesc cd;
    memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    cd.capacity_bytes = 4096;
    r = reconlCreateCommandList(rig->device, &cd, &rig->commands);
    if (r != RECONL_OK) return fail("create command list", r);
    return 0;
}

static void rig_free(Rig* rig) {
    if (rig->commands) reconlRelease(rig->commands);
    if (rig->pipeline) reconlRelease(rig->pipeline);
    if (rig->swapchain) reconlRelease(rig->swapchain);
    if (rig->device) reconlRelease(rig->device);
}

/* One white triangle covering most of the frame, with a headlight so the
 * Lambert path is lit; positions carry z>0 for reversed-Z. */
/* camera_mode: 0 = no camera, 1 = a camera that describes this frame exactly,
 * 2 = the same frame from a caller whose struct_size predates the camera field.
 * All three have to produce the same image. */
static int draw_frame(Rig* rig, int camera_mode) {
    ReconLVertex verts[3];
    memset(verts, 0, sizeof verts);
    verts[0].position[0] = 0.0f;  verts[0].position[1] = 0.6f;  verts[0].position[2] = 0.5f;
    verts[1].position[0] = 0.6f;  verts[1].position[1] = -0.6f; verts[1].position[2] = 0.5f;
    verts[2].position[0] = -0.6f; verts[2].position[1] = -0.6f; verts[2].position[2] = 0.5f;
    for (int i = 0; i < 3; ++i) {
        verts[i].normal[2] = 1.0f;
        for (int c = 0; c < 4; ++c) verts[i].color[c] = 1.0f;
    }
    uint32_t indices[3] = {0, 1, 2};

    ReconLBufferDesc vbd;
    memset(&vbd, 0, sizeof vbd);
    SETBASE(&vbd, RECONL_STRUCT_BUFFER_DESC);
    vbd.size_bytes = sizeof verts;
    vbd.usage = RECONL_BUFFER_VERTEX;
    vbd.data = verts;
    vbd.data_size = sizeof verts;
    ReconLBuffer* vb = NULL;
    ReconLResult r = reconlCreateBuffer(rig->device, &vbd, &vb);
    if (r != RECONL_OK) return fail("create vertex buffer", r);

    ReconLBufferDesc ibd;
    memset(&ibd, 0, sizeof ibd);
    SETBASE(&ibd, RECONL_STRUCT_BUFFER_DESC);
    ibd.size_bytes = sizeof indices;
    ibd.usage = RECONL_BUFFER_INDEX;
    ibd.data = indices;
    ibd.data_size = sizeof indices;
    ReconLBuffer* ib = NULL;
    r = reconlCreateBuffer(rig->device, &ibd, &ib);
    if (r != RECONL_OK) return fail("create index buffer", r);

    ReconLLight light;
    memset(&light, 0, sizeof light);
    SETBASE(&light, RECONL_STRUCT_LIGHT);
    light.type = RECONL_LIGHT_DIRECTIONAL;
    light.direction[2] = -1.0f;
    light.color[0] = light.color[1] = light.color[2] = 1.0f;
    light.intensity = 1.0f;
    light.cast_shadow = 0;

    ReconLLightList ll;
    memset(&ll, 0, sizeof ll);
    SETBASE(&ll, RECONL_STRUCT_LIGHT_LIST);
    ll.count = 1;
    ll.lights = &light;

    ReconLFrameDesc fd;
    memset(&fd, 0, sizeof fd);
    SETBASE(&fd, RECONL_STRUCT_FRAME_DESC);
    fd.width = W;
    fd.height = H;
    fd.seed = 1;
    fd.lights = &ll;
    fd.shadows = NULL;

    /* The camera. The shadow system needs the view matrix on its own, and the
       vertex transform in constant slot 0 - view * projection - cannot give it
       back. This geometry is already in clip space, so the identity view and the
       header's default frustum are exactly what it means, and the image must
       match the no-camera path byte for byte. */
    ReconLCamera cam;
    memset(&cam, 0, sizeof cam);
    SETBASE(&cam, RECONL_STRUCT_CAMERA);
    cam.view[0] = 1.0f;
    cam.view[5] = 1.0f;
    cam.view[10] = 1.0f;
    cam.view[15] = 1.0f;
    cam.fov_y_deg = 60.0f;
    cam.near = 0.1f;
    cam.far = 100.0f;

    if (camera_mode == 1) {
        fd.camera = &cam;
    } else if (camera_mode == 2) {
        /* A host compiled before the camera field existed: struct_size stops
           after `shadows`, so the library has to read the prefix it has always
           had instead of refusing the frame. */
        fd.base.struct_size = (uint32_t)(sizeof(ReconLFrameDesc) - sizeof(ReconLCamera*));
        fd.camera = &cam;
    }
    r = reconlBeginFrame(rig->device, &fd);
    if (r != RECONL_OK) return fail("begin frame", r);

    reconlCmdReset(rig->commands);
    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.color_count = 0;
    rp.viewport_width = W;
    rp.viewport_height = H;
    rp.load_color = 1;
    rp.load_depth = 1;
    rp.clear_color[2] = 1.0f;   /* blue */
    rp.clear_color[3] = 1.0f;
    rp.clear_depth = 0.0f;
    reconlCmdBeginRenderPass(rig->commands, &rp);
    reconlCmdSetPipeline(rig->commands, rig->pipeline);

    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    reconlCmdPushConstants(rig->commands, 0, identity, sizeof identity);
    reconlCmdPushConstants(rig->commands, 1, identity, sizeof identity);
    reconlCmdSetVertexBuffer(rig->commands, 0, vb, 0);
    reconlCmdSetIndexBuffer(rig->commands, ib, 0, RECONL_INDEX_UINT32);
    r = reconlCmdDrawIndexed(rig->commands, 3, 0, 0);
    reconlCmdEndRenderPass(rig->commands);
    if (r != RECONL_OK) return fail("draw", r);

    r = reconlSubmit(rig->device, rig->commands, NULL);
    if (r != RECONL_OK) {
        ReconLErrorInfo ei;
        memset(&ei, 0, sizeof ei);
        SETBASE(&ei, RECONL_STRUCT_ERROR_INFO);
        reconlGetLastError(rig->device, &ei);
        fprintf(stderr, "submit: %d %s\n", (int)r, ei.message);
        return 1;
    }

    ReconLPresentDesc prd;
    memset(&prd, 0, sizeof prd);
    SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
    prd.out_pixels = rig->pixels;
    prd.out_pixels_size = sizeof rig->pixels;
    prd.out_row_pitch = W * 4;
    prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    r = reconlPresent(rig->device, rig->swapchain, &prd);
    if (r != RECONL_OK) return fail("present", r);

    reconlRelease(vb);
    reconlRelease(ib);
    return 0;
}

int main(void) {
    /* 1. the probe. */
    ReconLProbeInfo info;
    memset(&info, 0, sizeof info);
    SETBASE(&info, RECONL_STRUCT_PROBE_INFO);
    ReconLResult r = reconlProbe(NULL, &info);
    if (r != RECONL_OK) return fail("probe", r);

    int gpu_usable = 0;
    printf("probe: %u entries, recommends backend=%u tier=%u\n",
           info.entry_count, (unsigned)info.recommended_backend, (unsigned)info.recommended_tier);
    for (uint32_t i = 0; i < info.entry_count; ++i) {
        const ReconLBackendProbe* e = &info.entries[i];
        printf("  [%u] %-9s usable=%d tier=%u vram=%llu device='%s'\n",
               i, e->name, e->usable, (unsigned)e->best_tier,
               (unsigned long long)e->vram_bytes, e->device_name);
        if (e->backend == RECONL_BACKEND_D3D11) gpu_usable = e->usable;
    }
    if (!gpu_usable) {
        printf("no usable D3D11 device: nothing further to check\n");
        return 0;
    }
    if (info.recommended_backend != RECONL_BACKEND_D3D11 || info.recommended_tier != RECONL_TIER_T1_GPU_SHARED) {
        fprintf(stderr, "FAIL: a usable GPU must be recommended\n");
        return 1;
    }

    /* 2. a frame through the ABI. */
    Rig rig;
    memset(&rig, 0, sizeof rig);
    if (rig_init(&rig)) { rig_free(&rig); return 1; }
    if (draw_frame(&rig, 0)) { rig_free(&rig); return 1; }

    unsigned blue = 0, white = 0;
    for (size_t i = 0; i < sizeof rig.pixels; i += 4) {
        if (rig.pixels[i] > 200 && rig.pixels[i + 2] > 200) white++;
        else if (rig.pixels[i + 2] > 200 && rig.pixels[i] < 50) blue++;
    }
    printf("frame: %u lit pixels, %u clear pixels\n", white, blue);
    if (white <= 100) {
        fprintf(stderr, "FAIL: the frame is only the clear colour\n");
        rig_free(&rig);
        return 1;
    }

    ReconLStats st;
    memset(&st, 0, sizeof st);
    SETBASE(&st, RECONL_STRUCT_STATS);
    if (reconlGetStats(rig.device, &st) != RECONL_OK) { rig_free(&rig); return fail("stats", r); }
    printf("stats: backend=%u tier=%u frames_presented=%u triangles_in=%u shadow_ns=%llu\n",
           st.backend, st.tier, st.frames_presented, st.frame.triangles_in,
           (unsigned long long)st.frame.shadow_ns);
    if (st.backend != RECONL_BACKEND_D3D11) { rig_free(&rig); return fail("backend id", RECONL_ERR_PANIC); }
    if (st.frame.triangles_in == 0) {
        fprintf(stderr, "FAIL: a triangle was drawn but none was counted\n");
        rig_free(&rig);
        return 1;
    }

    /* 3. the same frame twice is the same image. */
    unsigned char first[W * H * 4];
    memcpy(first, rig.pixels, sizeof first);
    if (draw_frame(&rig, 0)) { rig_free(&rig); return 1; }
    if (memcmp(first, rig.pixels, sizeof first) != 0) {
        fprintf(stderr, "FAIL: two identical frames produced different images\n");
        rig_free(&rig);
        return 1;
    }
    printf("determinism: two identical frames, identical bytes\n");

    /* 4. the camera field, and the struct_size that predates it. */
    if (draw_frame(&rig, 1)) { rig_free(&rig); return 1; }
    if (memcmp(first, rig.pixels, sizeof first) != 0) {
        fprintf(stderr, "FAIL: declaring the camera changed the image\n");
        rig_free(&rig);
        return 1;
    }
    if (draw_frame(&rig, 2)) { rig_free(&rig); return 1; }
    if (memcmp(first, rig.pixels, sizeof first) != 0) {
        fprintf(stderr, "FAIL: a caller with the pre-camera struct_size rendered differently\n");
        rig_free(&rig);
        return 1;
    }
    printf("camera: declared, null and pre-camera struct_size all render identically\n");

    rig_free(&rig);
    printf("OK\n");
    return 0;
}
