/* The render-pass viewport, as a host sees it, through the shipped DLL.
 *
 * Renders a triangle that covers the whole clip-space cube (identity
 * view-projection, identity model) into a square frame, asks for a
 * `--vp=N` viewport, presents, and reports where the lit pixels landed. If the
 * viewport means anything, the lit pixels are confined to the top-left N x N of
 * the frame; if it is ignored, they cover the whole frame.
 *
 * Usage: viewport --backend=soft-cpu|d3d11 --size=64 --vp=16
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

static const char* backend_name(uint32_t b) {
    switch (b) {
        case RECONL_BACKEND_D3D11: return "d3d11";
        case RECONL_BACKEND_SOFT_CPU: return "soft-cpu";
        case RECONL_BACKEND_NULL: return "null";
        default: return "?";
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
    if (!be) be = "soft-cpu";
    ReconLBackendId backend = strcmp(be, "d3d11") == 0 ? RECONL_BACKEND_D3D11
        : (strcmp(be, "null") == 0 ? RECONL_BACKEND_NULL : RECONL_BACKEND_SOFT_CPU);
    const uint32_t size = num(value_of(argc, argv, "--size"), 64);
    const uint32_t vp = num(value_of(argc, argv, "--vp"), 16);
    const uint32_t vpw = num(value_of(argc, argv, "--vpw"), vp);
    const uint32_t vph = num(value_of(argc, argv, "--vph"), vpw);

    ReconLDevice* dev = NULL;
    ReconLSwapchain* sc = NULL;
    ReconLCommandList* cl = NULL;
    ReconLBuffer *vb = NULL, *ib = NULL;
    ReconLPipeline* pipe = NULL;
    unsigned char* px = (unsigned char*)calloc((size_t)size * size * 4, 1);
    int rc = 0;

    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    /* `--tier=N` pins a tier for the scaled-target case; 0 is what the probe
     * always passed before (memset), so the default run is unchanged. */
    dd.tier_hint = (ReconLTier)num(value_of(argc, argv, "--tier"), 0);
    dd.seed = 7;
    dd.allocator.alloc = h_alloc; dd.allocator.realloc = h_realloc; dd.allocator.free = h_free;
    if (reconlCreateDevice(&dd, &dev) != RECONL_OK) { printf("device unavailable\n"); free(px); return 2; }

    ReconLSwapchainDesc sd;
    memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = size; sd.height = size;
    sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2; sd.present_to_memory = 1;
    sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    rc |= reconlCreateSwapchain(dev, &sd, &sc) != RECONL_OK;

    ReconLCommandListDesc cd;
    memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    cd.capacity_bytes = 8192;
    rc |= reconlCreateCommandList(dev, &cd, &cl) != RECONL_OK;

    /* One triangle covering the whole clip-space cube, unlit white. */
    ReconLVertex verts[3];
    memset(verts, 0, sizeof verts);
    const float pos[3][3] = {{-1.0f, -1.0f, 0.5f}, {3.0f, -1.0f, 0.5f}, {-1.0f, 3.0f, 0.5f}};
    for (int i = 0; i < 3; ++i) {
        memcpy(verts[i].position, pos[i], sizeof pos[i]);
        verts[i].normal[2] = 1.0f;
        verts[i].color[0] = verts[i].color[1] = verts[i].color[2] = verts[i].color[3] = 1.0f;
    }
    const uint32_t idx[3] = {0, 1, 2};

    ReconLBufferDesc bd;
    memset(&bd, 0, sizeof bd);
    SETBASE(&bd, RECONL_STRUCT_BUFFER_DESC);
    bd.size_bytes = sizeof verts; bd.usage = RECONL_BUFFER_VERTEX;
    bd.data = verts; bd.data_size = sizeof verts;
    rc |= reconlCreateBuffer(dev, &bd, &vb) != RECONL_OK;
    memset(&bd, 0, sizeof bd);
    SETBASE(&bd, RECONL_STRUCT_BUFFER_DESC);
    bd.size_bytes = sizeof idx; bd.usage = RECONL_BUFFER_INDEX;
    bd.data = idx; bd.data_size = sizeof idx;
    rc |= reconlCreateBuffer(dev, &bd, &ib) != RECONL_OK;

    ReconLPipelineDesc pd;
    memset(&pd, 0, sizeof pd);
    SETBASE(&pd, RECONL_STRUCT_PIPELINE_DESC);
    pd.shading = RECONL_SHADING_UNLIT;
    pd.blend = RECONL_BLEND_OPAQUE;
    pd.cull = RECONL_CULL_NONE;
    pd.depth_compare = RECONL_COMPARE_GREATER;
    pd.depth_write = 1;
    rc |= reconlCreatePipeline(dev, &pd, &pipe) != RECONL_OK;

    ReconLFrameDesc fd;
    memset(&fd, 0, sizeof fd);
    SETBASE(&fd, RECONL_STRUCT_FRAME_DESC);
    fd.width = size; fd.height = size; fd.seed = 1;
    ReconLResult begin = reconlBeginFrame(dev, &fd);

    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = vpw;
    rp.viewport_height = vph;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[0] = 0.0f; rp.clear_color[1] = 0.0f;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;   /* blue */
    rp.clear_depth = 0.0f;

    reconlCmdReset(cl);
    ReconLResult pass = reconlCmdBeginRenderPass(cl, &rp);
    const float ident[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    reconlCmdSetPipeline(cl, pipe);
    reconlCmdPushConstants(cl, 0, ident, 64);
    reconlCmdPushConstants(cl, 1, ident, 64);
    reconlCmdSetVertexBuffer(cl, 0, vb, 0);
    reconlCmdSetIndexBuffer(cl, ib, 0, RECONL_INDEX_UINT32);
    ReconLResult draw = reconlCmdDrawIndexed(cl, 3, 0, 0);
    reconlCmdEndRenderPass(cl);
    ReconLResult submit = reconlSubmit(dev, cl, NULL);

    ReconLPresentDesc pr;
    memset(&pr, 0, sizeof pr);
    SETBASE(&pr, RECONL_STRUCT_PRESENT_DESC);
    pr.out_pixels = px; pr.out_pixels_size = (uint64_t)size * size * 4;
    pr.out_row_pitch = size * 4; pr.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    ReconLResult present = reconlPresent(dev, sc, &pr);

    uint64_t white = 0, outside = 0;
    uint32_t maxx = 0, maxy = 0, minx = size, miny = size;
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            const unsigned char* p = px + ((size_t)y * size + x) * 4;
            if (p[0] > 200 && p[1] > 200 && p[2] > 200) {
                ++white;
                if (x < minx) minx = x;
                if (y < miny) miny = y;
                if (x > maxx) maxx = x;
                if (y > maxy) maxy = y;
                if (x >= vpw || y >= vph) ++outside;
            }
        }
    }

    ReconLStats st;
    memset(&st, 0, sizeof st);
    SETBASE(&st, RECONL_STRUCT_STATS);
    reconlGetStats(dev, &st);
    printf("%s tier T%d %ux%u, requested viewport %ux%u\n",
           backend_name(st.backend), (int)st.tier, size, size, vpw, vph);
    printf("  begin %d  pass %d  draw %d  submit %d  present %d\n",
           begin, pass, draw, submit, present);
    printf("  lit pixels %llu of %u  (%.1f%% of the frame)\n",
           (unsigned long long)white, size * size, 100.0 * (double)white / (double)(size * size));
    if (white) {
        printf("  lit bounding box x[%u..%u] y[%u..%u]\n", minx, maxx, miny, maxy);
    }
    printf("  lit pixels outside the requested %ux%u rect: %llu\n",
           vpw, vph, (unsigned long long)outside);

    if (pipe) reconlRelease(pipe);
    if (vb) reconlRelease(vb);
    if (ib) reconlRelease(ib);
    if (cl) reconlRelease(cl);
    if (sc) reconlRelease(sc);
    if (dev) reconlRelease(dev);
    free(px);
    return rc ? 1 : 0;
}
