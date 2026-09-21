/* The verification order of the two present paths, measured through the shipped
 * DLL. Read-only: this probe decides nothing, it records what the library does.
 *
 * Every cell is measured on its own fresh rig, with the frame state and the
 * feature's liveness established explicitly, so no cell can be contaminated by
 * the one before it (an earlier version of this probe re-used one rig and its own
 * health-check presents left the feature live, which made the "state" column a
 * lie).
 *
 * For each cell it records the code the call returns, whether any byte of the
 * host's buffer was written, and whether the device is still in the state it was
 * - which is what the header promises ("a call refused for its arguments changes
 * nothing").
 *
 * Usage: fgstate --backend=soft-cpu|d3d11 [--size=64]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "reconl/reconl.h"

#define SETBASE(p, T)                                          \
    do {                                                       \
        (p)->base.struct_size = (uint32_t)sizeof(*(p));         \
        (p)->base.type = (T);                                  \
        (p)->base.next = NULL;                                 \
    } while (0)

static long live_allocs = 0;
static long long live_bytes = 0;

static void* h_alloc(void* u, size_t size, size_t a) {
    (void)u;
    if (a < 16) a = 16;
    size_t total = size + a + 2 * sizeof(size_t);
    char* base = (char*)malloc(total);
    if (!base) return NULL;
    uintptr_t aligned = ((uintptr_t)base + 2 * sizeof(size_t) + a - 1) & ~(uintptr_t)(a - 1);
    ((size_t*)aligned)[-1] = size;
    ((size_t*)aligned)[-2] = (size_t)base;
    live_allocs++;
    live_bytes += (long long)size;
    return (void*)aligned;
}
static void h_free(void* u, void* ptr, size_t size) {
    (void)u; (void)size;
    if (!ptr) return;
    live_allocs--;
    live_bytes -= (long long)((size_t*)ptr)[-1];
    free((void*)((uintptr_t*)ptr)[-2]);
}
static void* h_realloc(void* u, void* ptr, size_t o, size_t n, size_t a) {
    void* f = h_alloc(u, n, a);
    if (!f) return NULL;
    if (ptr) { memcpy(f, ptr, o < n ? o : n); h_free(u, ptr, o); }
    return f;
}

static ReconLBackendId backend = RECONL_BACKEND_SOFT_CPU;
static uint32_t size = 64;
static const char* bname = "soft-cpu";

static const char* arg(const char* name, int argc, char** argv) {
    size_t n = strlen(name);
    for (int i = 1; i < argc; ++i)
        if (strncmp(argv[i], name, n) == 0 && argv[i][n] == '=') return argv[i] + n + 1;
    return NULL;
}

static const float IDENTITY[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

typedef struct { ReconLDevice* dev; ReconLSwapchain* sc; ReconLPipeline* pipe; ReconLCommandList* cl; ReconLBuffer* vb; ReconLBuffer* ib; } Rig;

static void view_at(float x, float out[16]) {
    memset(out, 0, 16 * sizeof(float));
    out[0] = out[5] = out[10] = out[15] = 1.0f;
    out[12] = -x;
}

static int build(Rig* r) {
    ReconLDeviceDesc dd; memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    dd.tier_hint = backend == RECONL_BACKEND_D3D11 ? RECONL_TIER_T1_GPU_SHARED
                 : backend == RECONL_BACKEND_NULL   ? RECONL_TIER_T3_CPU_THRIFTY
                                                    : RECONL_TIER_T2_CPU_RAM;
    dd.worker_threads = backend == RECONL_BACKEND_D3D11 ? 0 : 1;
    dd.seed = 1;
    dd.allocator.alloc = h_alloc; dd.allocator.realloc = h_realloc; dd.allocator.free = h_free;
    if (reconlCreateDevice(&dd, &r->dev) != RECONL_OK) return 0;
    ReconLSwapchainDesc sd; memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = size; sd.height = size; sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2; sd.present_to_memory = 1; sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    if (reconlCreateSwapchain(r->dev, &sd, &r->sc) != RECONL_OK) return 0;
    ReconLPipelineDesc pd; memset(&pd, 0, sizeof pd);
    SETBASE(&pd, RECONL_STRUCT_PIPELINE_DESC);
    pd.shading = RECONL_SHADING_LAMBERT; pd.depth_compare = RECONL_COMPARE_GREATER; pd.depth_write = 1;
    if (reconlCreatePipeline(r->dev, &pd, &r->pipe) != RECONL_OK) return 0;
    static ReconLVertex verts[6];
    static uint32_t idx[6] = {0,1,2,3,4,5};
    const float c[6][2] = {{-1,1},{1,1},{1,-1},{-1,1},{1,-1},{-1,-1}};
    for (int i = 0; i < 6; ++i) {
        memset(&verts[i], 0, sizeof verts[i]);
        verts[i].position[0] = c[i][0]; verts[i].position[1] = c[i][1]; verts[i].position[2] = 0.5f;
        verts[i].normal[2] = 1.0f;
        verts[i].color[0] = verts[i].color[1] = verts[i].color[2] = verts[i].color[3] = 1.0f;
    }
    ReconLBufferDesc vd; memset(&vd, 0, sizeof vd);
    SETBASE(&vd, RECONL_STRUCT_BUFFER_DESC);
    vd.size_bytes = sizeof verts; vd.usage = RECONL_BUFFER_VERTEX; vd.data = verts; vd.data_size = sizeof verts;
    if (reconlCreateBuffer(r->dev, &vd, &r->vb) != RECONL_OK) return 0;
    ReconLBufferDesc id; memset(&id, 0, sizeof id);
    SETBASE(&id, RECONL_STRUCT_BUFFER_DESC);
    id.size_bytes = sizeof idx; id.usage = RECONL_BUFFER_INDEX; id.data = idx; id.data_size = sizeof idx;
    if (reconlCreateBuffer(r->dev, &id, &r->ib) != RECONL_OK) return 0;
    ReconLCommandListDesc cd; memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    if (reconlCreateCommandList(r->dev, &cd, &r->cl) != RECONL_OK) return 0;
    return 1;
}

static void destroy(Rig* r) {
    if (r->cl) reconlRelease(r->cl);
    if (r->vb) reconlRelease(r->vb);
    if (r->ib) reconlRelease(r->ib);
    if (r->pipe) reconlRelease(r->pipe);
    if (r->sc) reconlRelease(r->sc);
    if (r->dev) reconlRelease(r->dev);
}

static ReconLResult open_frame(Rig* r, float eye, int keep) {
    ReconLLight light; memset(&light, 0, sizeof light);
    SETBASE(&light, RECONL_STRUCT_LIGHT);
    light.type = RECONL_LIGHT_DIRECTIONAL; light.direction[2] = -1.0f;
    light.color[0] = light.color[1] = light.color[2] = 1.0f; light.intensity = 1.0f;
    ReconLLightList lights; memset(&lights, 0, sizeof lights);
    SETBASE(&lights, RECONL_STRUCT_LIGHT_LIST);
    lights.count = 1; lights.lights = &light;
    ReconLCamera cam; memset(&cam, 0, sizeof cam);
    SETBASE(&cam, RECONL_STRUCT_CAMERA);
    view_at(eye, cam.view); cam.fov_y_deg = 60.0f; cam.near = 0.1f; cam.far = 100.0f;
    ReconLFrameGenDesc fg; memset(&fg, 0, sizeof fg);
    SETBASE(&fg, RECONL_STRUCT_FRAME_GEN);
    fg.enabled = keep ? 1u : 0u;
    ReconLFrameDesc fd; memset(&fd, 0, sizeof fd);
    SETBASE(&fd, RECONL_STRUCT_FRAME_DESC);
    fd.width = size; fd.height = size; fd.seed = 1; fd.lights = &lights; fd.camera = &cam;
    fd.framegen = keep ? &fg : NULL;
    ReconLResult rr = reconlBeginFrame(r->dev, &fd);
    if (rr != RECONL_OK) return rr;
    ReconLRenderPassDesc rp; memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.load_color = 1; rp.load_depth = 1; rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;
    reconlCmdReset(r->cl);
    if ((rr = reconlCmdBeginRenderPass(r->cl, &rp)) != RECONL_OK) return rr;
    if ((rr = reconlCmdSetPipeline(r->cl, r->pipe)) != RECONL_OK) return rr;
    if ((rr = reconlCmdPushConstants(r->cl, 0, IDENTITY, 64)) != RECONL_OK) return rr;
    if ((rr = reconlCmdPushConstants(r->cl, 1, IDENTITY, 64)) != RECONL_OK) return rr;
    if ((rr = reconlCmdSetVertexBuffer(r->cl, 0, r->vb, 0)) != RECONL_OK) return rr;
    if ((rr = reconlCmdSetIndexBuffer(r->cl, r->ib, 0, RECONL_INDEX_UINT32)) != RECONL_OK) return rr;
    if ((rr = reconlCmdDrawIndexed(r->cl, 6, 0, 0)) != RECONL_OK) return rr;
    return reconlCmdEndRenderPass(r->cl);
}

static ReconLResult present_good(Rig* r, unsigned char* out) {
    ReconLPresentDesc prd; memset(&prd, 0, sizeof prd);
    SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
    prd.out_pixels = out; prd.out_pixels_size = (uint64_t)size * size * 4;
    prd.out_row_pitch = size * 4; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return reconlPresent(r->dev, r->sc, &prd);
}

/* One whole frame, presented. `keep` decides whether it asks for generation. */
static int whole_frame(Rig* r, unsigned char* out, float eye, int keep) {
    ReconLResult rr = open_frame(r, eye, keep);
    if (rr == RECONL_OK) rr = reconlSubmit(r->dev, r->cl, NULL);
    if (rr == RECONL_OK) rr = present_good(r, out);
    return rr == RECONL_OK;
}

/* Descriptor and value variants. */
enum {
    V_VALID, V_WRONGTYPE, V_NULLPIX, V_ZEROSIZE, V_SHORT, V_NARROW,
    V_AHEAD_HIGH, V_AHEAD_NAN, V_COUNT
};
static const char* vname[V_COUNT] = {
    "valid", "wrong-type", "null-pixels", "zero-size", "short-buffer", "narrow-pitch",
    "ahead>1", "ahead-NaN"
};

static void make_desc(ReconLPresentDesc* prd, int v, unsigned char* buf) {
    memset(prd, 0, sizeof *prd);
    prd->base.struct_size = (uint32_t)sizeof *prd;
    prd->base.type = v == V_WRONGTYPE ? RECONL_STRUCT_FRAME_DESC : RECONL_STRUCT_PRESENT_DESC;
    prd->out_pixels = buf;
    prd->out_pixels_size = (uint64_t)size * size * 4;
    prd->out_row_pitch = size * 4;
    prd->out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    if (v == V_NULLPIX) prd->out_pixels = NULL;
    if (v == V_ZEROSIZE) prd->out_pixels_size = 0;
    if (v == V_SHORT) prd->out_pixels_size = (uint64_t)size * size * 4 - 4;
    if (v == V_NARROW) prd->out_row_pitch = size * 4 - 4;
}

static float ahead_for(int v) {
    if (v == V_AHEAD_HIGH) return 1.0000001f;
    if (v == V_AHEAD_NAN) return (float)(0.0 / 0.0);
    return 1.0f;
}

/* What a host has done before the call: the frame machine's state, and whether
 * anything has asked for generation (which is what the feature's history is). */
enum { L_NONE, L_NOASK, L_ASKED, L_COUNT };
static const char* lname[L_COUNT] = { "nothing-asked", "last-asked=no", "last-asked=yes" };
enum { F_IDLE, F_OPEN, F_SUBMITTED, F_COUNT };
static const char* fname[F_COUNT] = { "idle", "open", "submitted" };

/* A fresh rig in the requested state, or NULL. */
static Rig* setup(int liveness, int frame, unsigned char* scratch) {
    Rig* r = (Rig*)malloc(sizeof(Rig));
    memset(r, 0, sizeof *r);
    if (!build(r)) { destroy(r); free(r); return NULL; }
    if (liveness != L_NONE) {
        if (!whole_frame(r, scratch, 0.2f, liveness == L_ASKED)) { destroy(r); free(r); return NULL; }
    }
    if (frame != F_IDLE) {
        if (open_frame(r, 0.4f, 1) != RECONL_OK) { destroy(r); free(r); return NULL; }
        if (frame == F_SUBMITTED && reconlSubmit(r->dev, r->cl, NULL) != RECONL_OK) {
            destroy(r); free(r); return NULL;
        }
    }
    return r;
}

static int any_written(const unsigned char* buf, size_t n) {
    for (size_t i = 0; i < n; ++i) if (buf[i] != 0xCD) return 1;
    return 0;
}

/* Is the device still where the cell's state says it should be? */
static int still_usable(Rig* r, int liveness, int frame, unsigned char* scratch) {
    ReconLPresentDesc prd; make_desc(&prd, V_VALID, scratch);
    switch (frame) {
        case F_OPEN: return reconlSubmit(r->dev, r->cl, NULL) == RECONL_OK;
        case F_SUBMITTED: return present_good(r, scratch) == RECONL_OK;
        default:
            /* Idle: a tier that keeps no depth never generates, whatever the
             * state; otherwise a valid descriptor generates when there is a
             * history and says there is nothing to generate from when there is
             * not. */
            return reconlPresentGenerated(r->dev, r->sc, &prd, 1.0f)
                   == (backend == RECONL_BACKEND_NULL ? RECONL_ERR_NOT_SUPPORTED
                       : liveness == L_ASKED           ? RECONL_OK
                                                      : RECONL_ERR_NO_FRAME);
    }
}

/* [liveness][frame][variant] */
static int gcode[L_COUNT][F_COUNT][V_COUNT];
static int gwrote[L_COUNT][F_COUNT][V_COUNT];
static int gusable[L_COUNT][F_COUNT][V_COUNT];
static int pcode[F_COUNT][V_COUNT];
static int pwrote[F_COUNT][V_COUNT];

static int cells_ran = 0, cells_failed = 0;
static int refused_but_wrote = 0, wedged = 0;

int main(int argc, char** argv) {
    const char* nm = arg("--backend", argc, argv);
    if (nm && strcmp(nm, "d3d11") == 0) { backend = RECONL_BACKEND_D3D11; bname = "d3d11"; }
    if (nm && strcmp(nm, "null") == 0) { backend = RECONL_BACKEND_NULL; bname = "null"; }
    const char* sz = arg("--size", argc, argv);
    if (sz) size = (uint32_t)strtoul(sz, NULL, 10);

    printf("fgstate - %s, %ux%u, one fresh rig per cell\n", bname, size, size);
    unsigned char* scratch = (unsigned char*)malloc((size_t)size * size * 4);
    unsigned char* buf = (unsigned char*)malloc((size_t)size * size * 4);
    const size_t buflen = (size_t)size * size * 4;

    printf("\nreconlPresentGenerated, by what the host did before the call\n");
    for (int l = 0; l < L_COUNT; ++l) {
        for (int f = 0; f < F_COUNT; ++f) {
            for (int v = 0; v < V_COUNT; ++v) {
                Rig* r = setup(l, f, scratch);
                if (!r) { printf("  %s/%s unreachable\n", lname[l], fname[f]); ++cells_failed; break; }
                memset(buf, 0xCD, buflen);
                ReconLPresentDesc prd; make_desc(&prd, v, buf);
                ReconLResult res = reconlPresentGenerated(r->dev, r->sc, &prd, ahead_for(v));
                gcode[l][f][v] = (int)res;
                gwrote[l][f][v] = any_written(buf, buflen);
                gusable[l][f][v] = still_usable(r, l, f, scratch);
                if (res != RECONL_OK && gwrote[l][f][v]) ++refused_but_wrote;
                if (!gusable[l][f][v]) ++wedged;
                ++cells_ran;
                printf("  %-14s/%-9s %-13s -> %3d  wrote=%-3s device-usable=%s\n",
                       lname[l], fname[f], vname[v], (int)res,
                       gwrote[l][f][v] ? "YES" : "no", gusable[l][f][v] ? "yes" : "NO");
                destroy(r); free(r);
            }
        }
    }

    printf("\nreconlPresent, by frame state (a fresh rig per cell)\n");
    for (int f = 0; f < F_COUNT; ++f) {
        for (int v = 0; v < V_COUNT - 2; ++v) {
            Rig* r = setup(L_NONE, f, scratch);
            if (!r) { printf("  %s unreachable\n", fname[f]); ++cells_failed; break; }
            memset(buf, 0xCD, buflen);
            ReconLPresentDesc prd; make_desc(&prd, v, buf);
            ReconLResult res = reconlPresent(r->dev, r->sc, &prd);
            pcode[f][v] = (int)res;
            pwrote[f][v] = any_written(buf, buflen);
            if (res != RECONL_OK && pwrote[f][v]) ++refused_but_wrote;
            ++cells_ran;
            printf("  %-9s %-13s -> %3d  wrote=%s\n", fname[f], vname[v], (int)res,
                   pwrote[f][v] ? "YES" : "no");
            destroy(r); free(r);
        }
    }

    printf("\nwhat the matrix says\n");
    printf("  the generated path, per descriptor: the code in every cell (the\n  variation below is with whether a frame is kept, not with the frame state)\n");
    int vary = 0;
    for (int v = 0; v < V_COUNT; ++v) {
        int first = gcode[0][0][v], same = 1;
        for (int l = 0; l < L_COUNT; ++l)
            for (int f = 0; f < F_COUNT; ++f)
                if (gcode[l][f][v] != first) same = 0;
        if (!same) {
            ++vary;
            printf("    varies %-13s:", vname[v]);
            for (int l = 0; l < L_COUNT; ++l)
                for (int f = 0; f < F_COUNT; ++f) printf(" %s/%s=%d", lname[l], fname[f], gcode[l][f][v]);
            printf("\n");
        }
    }
    if (!vary) printf("    (every descriptor answers the same code in every cell)\n");

    printf("  the pitch rule per cell: what the documented order says it answers\n");
    int pitch_bad = 0;
    for (int l = 0; l < L_COUNT; ++l)
        for (int f = 0; f < F_COUNT; ++f) {
            /* The tier speaks before both; then, with nothing to generate from,
             * the gate does; then the descriptor does. */
            int want = backend == RECONL_BACKEND_NULL ? RECONL_ERR_NOT_SUPPORTED
                     : l == L_ASKED ? RECONL_ERR_INVALID_ARGUMENT
                                    : RECONL_ERR_NO_FRAME;
            if (gcode[l][f][V_NARROW] != want) {
                ++pitch_bad;
                printf("    FIND %s/%s: %d, expected %d\n", lname[l], fname[f], gcode[l][f][V_NARROW], want);
            }
        }
    if (!pitch_bad) printf("    (the documented order holds in every cell)\n");

    printf("  refusals that wrote into the host's buffer: %d\n", refused_but_wrote);
    printf("  cells that left the device somewhere unusable: %d\n", wedged);
    printf("  cells that could not be set up: %d of %d\n", cells_failed, cells_ran + cells_failed);

    printf("\n  reconlPresent by descriptor (documented: the state machine decides first)\n");
    for (int v = 0; v < V_COUNT - 2; ++v)
        printf("    %-13s: idle=%3d open=%3d submitted=%3d%s\n", vname[v],
               pcode[0][v], pcode[1][v], pcode[2][v],
               (pcode[0][v] == pcode[1][v] && pcode[1][v] == pcode[2][v]) ? "" : "   <- state decides");

    /* The order documented in reconl.h puts the tier and the kept-frame gate
     * before the descriptor. That is only safe if neither gate dereferences the
     * descriptor - so hand it descriptors a host really can pass and that the
     * pin above never builds: a NULL pointer, a header no ABI call accepts, and a
     * header that stops after the prefix. Nothing here may crash, and whatever
     * the call answers, the device must be exactly where it was. */
    printf("\nthe descriptor pointer itself, in the states the order gates on\n");
    int ptrbad = 0;
    for (int li = 0; li < 2; ++li) {
        Rig* r = setup(li ? L_ASKED : L_NONE, F_IDLE, scratch);
        if (!r) { printf("  %s unreachable\n", lname[li ? L_ASKED : L_NONE]); ++cells_failed; continue; }
        ReconLPresentDesc h;
        int codes[4];
        codes[0] = (int)reconlPresentGenerated(r->dev, r->sc, NULL, 1.0f);
        memset(&h, 0, sizeof h);
        h.base.struct_size = 0xFFFFFFFFu; h.base.type = 0xDEADBEEFu;
        codes[1] = (int)reconlPresentGenerated(r->dev, r->sc, &h, 1.0f);
        memset(&h, 0, sizeof h);
        h.base.struct_size = 4; h.base.type = RECONL_STRUCT_PRESENT_DESC;
        codes[2] = (int)reconlPresentGenerated(r->dev, r->sc, &h, 1.0f);
        codes[3] = (int)reconlPresent(r->dev, r->sc, NULL);
        int usable = still_usable(r, li ? L_ASKED : L_NONE, F_IDLE, scratch);

        /* The documented order, for the tier running: the tier speaks first, then
         * whether a frame is kept; the present's gate is its own frame state. */
        int gen_ok;
        if (backend == RECONL_BACKEND_NULL) {
            /* The tier speaks before everything, so the descriptor is never read. */
            gen_ok = codes[0] == RECONL_ERR_NOT_SUPPORTED && codes[1] == RECONL_ERR_NOT_SUPPORTED
                  && codes[2] == RECONL_ERR_NOT_SUPPORTED;
        } else if (li == 0) {
            /* Nothing to generate from: one answer, and no descriptor was read. */
            gen_ok = codes[0] == RECONL_ERR_NO_FRAME && codes[1] == RECONL_ERR_NO_FRAME
                  && codes[2] == RECONL_ERR_NO_FRAME;
        } else {
            /* A frame is kept, so the descriptor is the gate, and each of these
             * is refused for what it is: no descriptor, a header no ABI call
             * accepts, one that stops after the prefix. */
            gen_ok = codes[0] == RECONL_ERR_INVALID_ARGUMENT && codes[1] != 0 && codes[2] != 0;
        }
        int pres_ok = codes[3] == RECONL_ERR_NO_FRAME;   /* no frame to present */
        if (!gen_ok || !pres_ok || !usable) ++ptrbad;
        printf("  %-14s gen(NULL)=%3d gen(garbage-hdr)=%3d gen(prefix-only)=%3d present(NULL)=%3d usable=%s  %s\n",
               lname[li ? L_ASKED : L_NONE], codes[0], codes[1], codes[2], codes[3],
               usable ? "yes" : "NO", (gen_ok && pres_ok && usable) ? "" : "<- FIND");
        destroy(r); free(r);
        ++cells_ran;
    }
    printf("  descriptors that crashed or wedged the device: %d\n", ptrbad);

    printf("\nbytes live at exit: %ld (%lld bytes)\n", live_allocs, live_bytes);
    free(scratch); free(buf);
    return 0;
}
