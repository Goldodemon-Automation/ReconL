/* The camera and shadow config through the C ABI, including cameras that are
 * degenerate: the camera is the newest ABI field, so it is the one most likely
 * to be handed something a host got wrong. A shadow config is present for every
 * case, because that is what actually consumes the camera.
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

#define W 32u
#define H 32u

static const char* code_name(ReconLResult r) {
    switch (r) {
    case 0: return "OK"; case -1: return "INVALID_ARGUMENT"; case -2: return "OUT_OF_MEMORY";
    case -3: return "NOT_SUPPORTED"; case -4: return "BACKEND_UNAVAILABLE"; case -5: return "BUDGET_EXCEEDED";
    case -6: return "DEVICE_LOST"; case -7: return "INVALID_HANDLE"; case -8: return "STRUCT_SIZE";
    case -9: return "WRONG_STRUCT_TYPE"; case -10: return "ABI_VERSION"; case -11: return "FRAME_IN_PROGRESS";
    case -12: return "NO_FRAME"; case -13: return "NOT_READY"; case -16: return "DEGRADED";
    case -17: return "PANIC"; case -18: return "EMPTY_FRAME"; default: return "?";
    }
}

static void describe(ReconLDevice* d, const char* what, ReconLResult r) {
    ReconLErrorInfo ei;
    memset(&ei, 0, sizeof ei);
    SETBASE(&ei, RECONL_STRUCT_ERROR_INFO);
    reconlGetLastError(d, &ei);
    printf("    %-34s -> %2d %-18s %s\n", what, (int)r, code_name(r),
           r == 0 ? "" : (ei.message[0] ? ei.message : "(no message)"));
}

/* camera_kind: 0 none, 1 valid, 2 near == far, 3 fov 0, 4 fov 1000,
 * 5 zeroed view, 6 scaled (non-rigid) view, 7 identity view of a scene behind it. */
static ReconLResult one_frame(ReconLDevice* d, ReconLSwapchain* s, int camera_kind,
                              int shadows_enabled, unsigned char* out) {
    ReconLLight light;
    memset(&light, 0, sizeof light);
    SETBASE(&light, RECONL_STRUCT_LIGHT);
    light.type = RECONL_LIGHT_DIRECTIONAL;
    light.direction[0] = 0.1f; light.direction[1] = -1.0f; light.direction[2] = 1.55f;
    light.color[0] = light.color[1] = light.color[2] = 1.0f;
    light.intensity = 1.0f;
    light.cast_shadow = 1;
    ReconLLightList ll;
    memset(&ll, 0, sizeof ll);
    SETBASE(&ll, RECONL_STRUCT_LIGHT_LIST);
    ll.count = 1; ll.lights = &light;

    ReconLShadowConfig sc;
    memset(&sc, 0, sizeof sc);
    SETBASE(&sc, RECONL_STRUCT_SHADOW_CONFIG);
    sc.enabled = shadows_enabled;
    sc.cascade_count = 2;
    sc.texel_budget_bytes = 8u << 20;
    sc.filter = RECONL_SHADOW_FILTER_PCF3X3;
    sc.max_distance = 77.0f;
    sc.blend_band = 3.0f;
    sc.normal_bias = 1.25f; sc.depth_bias = 5.0e-4f; sc.slope_bias = 1.75f;
    sc.refresh_interval_frames = 1;

    ReconLCamera cam;
    memset(&cam, 0, sizeof cam);
    SETBASE(&cam, RECONL_STRUCT_CAMERA);
    cam.view[0] = cam.view[5] = cam.view[10] = cam.view[15] = 1.0f;
    cam.view[13] = -13.0f; /* eye above the origin looking down -Z */
    cam.fov_y_deg = 60.0f; cam.near = 0.1f; cam.far = 100.0f;
    switch (camera_kind) {
    case 2: cam.near = 0.5f; cam.far = 0.5f; break;
    case 3: cam.fov_y_deg = 0.0f; break;
    case 4: cam.fov_y_deg = 1000.0f; break;
    case 5: memset(cam.view, 0, sizeof cam.view); break;
    case 6: for (int i = 0; i < 16; ++i) cam.view[i] = (float)i * 0.5f; break;
    case 7: break;
    default: break;
    }

    ReconLFrameDesc fd;
    memset(&fd, 0, sizeof fd);
    SETBASE(&fd, RECONL_STRUCT_FRAME_DESC);
    fd.width = W; fd.height = H; fd.seed = 1;
    fd.lights = &ll;
    fd.shadows = &sc;
    fd.camera = camera_kind == 0 ? NULL : &cam;

    ReconLResult r = reconlBeginFrame(d, &fd);
    if (r != RECONL_OK) return r;

    ReconLVertex verts[3];
    memset(verts, 0, sizeof verts);
    verts[0].position[0] = -0.7f; verts[0].position[1] = 3.0f;  verts[0].position[2] = -4.0f;
    verts[1].position[0] = 0.7f;  verts[1].position[1] = 3.0f;  verts[1].position[2] = -4.5f;
    verts[2].position[0] = 0.0f;  verts[2].position[1] = 3.0f;  verts[2].position[2] = -1.0f;
    for (int i = 0; i < 3; ++i) { verts[i].normal[1] = 1.0f;
        verts[i].color[0] = verts[i].color[1] = verts[i].color[2] = verts[i].color[3] = 1.0f; }
    uint32_t idx[3] = { 0, 1, 2 };

    ReconLBufferDesc vbd;
    memset(&vbd, 0, sizeof vbd);
    SETBASE(&vbd, RECONL_STRUCT_BUFFER_DESC);
    vbd.size_bytes = sizeof verts; vbd.usage = RECONL_BUFFER_VERTEX;
    vbd.data = verts; vbd.data_size = sizeof verts;
    ReconLBuffer* vb = NULL;
    r = reconlCreateBuffer(d, &vbd, &vb);
    if (r != RECONL_OK) return r;
    ReconLBufferDesc ibd;
    memset(&ibd, 0, sizeof ibd);
    SETBASE(&ibd, RECONL_STRUCT_BUFFER_DESC);
    ibd.size_bytes = sizeof idx; ibd.usage = RECONL_BUFFER_INDEX;
    ibd.data = idx; ibd.data_size = sizeof idx;
    ReconLBuffer* ib = NULL;
    r = reconlCreateBuffer(d, &ibd, &ib);
    if (r != RECONL_OK) { reconlRelease(vb); return r; }

    ReconLPipelineDesc pd;
    memset(&pd, 0, sizeof pd);
    SETBASE(&pd, RECONL_STRUCT_PIPELINE_DESC);
    pd.shading = RECONL_SHADING_LAMBERT;
    pd.depth_compare = RECONL_COMPARE_GREATER;
    pd.depth_write = 1;
    pd.receives_shadow = 1;
    pd.casts_shadow = 1;
    ReconLPipeline* pipe = NULL;
    r = reconlCreatePipeline(d, &pd, &pipe);
    if (r != RECONL_OK) { reconlRelease(vb); reconlRelease(ib); return r; }

    ReconLCommandListDesc cd;
    memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    cd.capacity_bytes = 4096;
    ReconLCommandList* cl = NULL;
    r = reconlCreateCommandList(d, &cd, &cl);
    if (r != RECONL_OK) { reconlRelease(vb); reconlRelease(ib); reconlRelease(pipe); return r; }

    reconlCmdReset(cl);
    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = W; rp.viewport_height = H;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;
    rp.clear_depth = 0.0f;
    reconlCmdBeginRenderPass(cl, &rp);
    reconlCmdSetPipeline(cl, pipe);
    const float identity[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    reconlCmdPushConstants(cl, 0, identity, sizeof identity);
    reconlCmdPushConstants(cl, 1, identity, sizeof identity);
    reconlCmdSetVertexBuffer(cl, 0, vb, 0);
    reconlCmdSetIndexBuffer(cl, ib, 0, RECONL_INDEX_UINT32);
    reconlCmdDrawIndexed(cl, 3, 0, 0);
    reconlCmdEndRenderPass(cl);
    r = reconlSubmit(d, cl, NULL);
    if (r == RECONL_OK && out) {
        ReconLPresentDesc prd;
        memset(&prd, 0, sizeof prd);
        SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
        prd.out_pixels = out; prd.out_pixels_size = W * H * 4;
        prd.out_row_pitch = W * 4; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
        r = reconlPresent(d, s, &prd);
    }
    reconlRelease(cl); reconlRelease(pipe); reconlRelease(vb); reconlRelease(ib);
    return r;
}

static int findings = 0, checks = 0;
#define CHECK(label, cond)                                  \
    do { ++checks; if (cond) printf("  ok   %s\n", label);  \
         else { printf("  FIND %s\n", label); ++findings; } } while (0)

static void arm(ReconLBackendId backend, const char* name) {
    printf("  %s:\n", name);
    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    dd.tier_hint = backend == RECONL_BACKEND_D3D11 ? RECONL_TIER_T1_GPU_SHARED : RECONL_TIER_T2_CPU_RAM;
    dd.allow_downgrade = 0;
    dd.target_frame_ms = 1000; dd.downgrade_after_frames = 16; dd.seed = 7;
    dd.allocator.alloc = h_alloc; dd.allocator.realloc = h_realloc; dd.allocator.free = h_free;
    ReconLDevice* d = NULL;
    if (reconlCreateDevice(&dd, &d) != RECONL_OK) { printf("    (unavailable)\n"); return; }
    ReconLSwapchainDesc sd;
    memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = W; sd.height = H;
    sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2; sd.present_to_memory = 1;
    sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    ReconLSwapchain* s = NULL;
    if (reconlCreateSwapchain(d, &sd, &s) != RECONL_OK) { reconlRelease(d); return; }

    unsigned char px[W * H * 4];
    const char* labels[8] = { "no camera", "valid camera", "near == far", "fov 0",
                              "fov 1000", "zeroed view", "scaled view", "identity view" };
    for (int kind = 0; kind <= 7; ++kind) {
        memset(px, 0xAB, sizeof px);
        ReconLResult r = one_frame(d, s, kind, 1, px);
        describe(d, labels[kind], r);
        if (kind == 2 || kind == 3 || kind == 4) {
            CHECK("a degenerate camera is refused, not used", r != RECONL_OK);
        } else {
            CHECK("the frame completes without taking the process down", r == RECONL_OK || r == RECONL_ERR_EMPTY_FRAME);
        }
    }
    reconlRelease(s);
    reconlRelease(d);
}

int main(void) {
    printf("camera and shadow config edge cases\n");
    arm(RECONL_BACKEND_SOFT_CPU, "soft-cpu");
    arm(RECONL_BACKEND_D3D11, "d3d11");
    printf("hostile5: %d checks, %d findings\n", checks, findings);
    return 0;
}
