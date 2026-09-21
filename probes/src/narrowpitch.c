/* The row-layout contract, through the shipped DLL, route by route.
 *
 * reconl.h (out_row_pitch, l.881-891): "`out_row_pitch < width * 4` is refused
 * with RECONL_ERR_INVALID_ARGUMENT and nothing is written."
 *
 * Each case hands the library a buffer big enough that the *size* check cannot
 * be what answers, and a pitch one pixel narrower than a row (and, separately,
 * pitch=4). Buffers are pre-filled with 0xAB so unwritten bytes are visible.
 *
 * Routes covered:
 *   [1] reconlWriteTexture        (upload; the control with a real check)
 *   [2] reconlReadTexture         (texture/cache readback)
 *   [3] reconlPresent             (framegen off -> backend readback)
 *   [4] reconlPresent             (framegen on  -> the keep path)
 *   [5] reconlPresent             (after reconlAudit -> the cached readback)
 *   [6] reconlPresentGenerated    (the generated-frame path)
 *
 * Usage: narrowpitch --backend=soft-cpu|d3d11|null [--size=64]
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

static int checks = 0, findings = 0;

static void CHECK(const char* what, int ok) {
    ++checks;
    printf("  %s  %s\n", ok ? "ok  " : "FIND", what);
    if (!ok) ++findings;
}

static ReconLBackendId backend = RECONL_BACKEND_SOFT_CPU;
static uint32_t size = 64;

static const char* arg(const char* name, int argc, char** argv) {
    size_t n = strlen(name);
    for (int i = 1; i < argc; ++i)
        if (strncmp(argv[i], name, n) == 0 && argv[i][n] == '=') return argv[i] + n + 1;
    return NULL;
}

static const float IDENTITY[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

static void* h_alloc(void* u, size_t size, size_t a) {
    (void)u;
    if (a < 16) a = 16;
    size_t total = size + a + 2 * sizeof(size_t);
    char* base = (char*)malloc(total);
    if (!base) return NULL;
    uintptr_t aligned = ((uintptr_t)base + 2 * sizeof(size_t) + a - 1) & ~(uintptr_t)(a - 1);
    ((size_t*)aligned)[-1] = size;
    ((size_t*)aligned)[-2] = (size_t)base;
    return (void*)aligned;
}
static void h_free(void* u, void* ptr, size_t size) {
    (void)u; (void)size;
    if (!ptr) return;
    free((void*)((uintptr_t*)ptr)[-2]);
}
static void* h_realloc(void* u, void* ptr, size_t o, size_t n, size_t a) {
    void* f = h_alloc(u, n, a);
    if (!f) return NULL;
    if (ptr) { memcpy(f, ptr, o < n ? o : n); h_free(u, ptr, o); }
    return f;
}

typedef struct { ReconLDevice* dev; ReconLSwapchain* sc; ReconLPipeline* pipe; ReconLCommandList* cl; ReconLBuffer* vb; ReconLBuffer* ib; } Rig;

static void view_at(float x, float out[16]) {
    memset(out, 0, 16 * sizeof(float));
    out[0] = out[5] = out[10] = out[15] = 1.0f;
    out[12] = -x;
}

static int build(Rig* r) {
    ReconLDeviceDesc dd; memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend; dd.tier_hint = RECONL_TIER_T2_CPU_RAM; dd.worker_threads = 1; dd.seed = 1;
    dd.allocator.alloc = h_alloc; dd.allocator.realloc = h_realloc; dd.allocator.free = h_free;
    ReconLResult rr = reconlCreateDevice(&dd, &r->dev);
    if (rr != RECONL_OK) { printf("    create device -> %d\n", (int)rr); return 0; }
    ReconLSwapchainDesc sd; memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = size; sd.height = size; sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2; sd.present_to_memory = 1; sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    if ((rr = reconlCreateSwapchain(r->dev, &sd, &r->sc)) != RECONL_OK) { printf("    swapchain -> %d\n", (int)rr); return 0; }
    ReconLPipelineDesc pd; memset(&pd, 0, sizeof pd);
    SETBASE(&pd, RECONL_STRUCT_PIPELINE_DESC);
    pd.shading = RECONL_SHADING_LAMBERT; pd.depth_compare = RECONL_COMPARE_GREATER; pd.depth_write = 1;
    if ((rr = reconlCreatePipeline(r->dev, &pd, &r->pipe)) != RECONL_OK) { printf("    pipeline -> %d\n", (int)rr); return 0; }
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
    if ((rr = reconlCreateBuffer(r->dev, &vd, &r->vb)) != RECONL_OK) return 0;
    ReconLBufferDesc id; memset(&id, 0, sizeof id);
    SETBASE(&id, RECONL_STRUCT_BUFFER_DESC);
    id.size_bytes = sizeof idx; id.usage = RECONL_BUFFER_INDEX; id.data = idx; id.data_size = sizeof idx;
    if ((rr = reconlCreateBuffer(r->dev, &id, &r->ib)) != RECONL_OK) return 0;
    ReconLCommandListDesc cd; memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    if ((rr = reconlCreateCommandList(r->dev, &cd, &r->cl)) != RECONL_OK) return 0;
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

/* Opens a frame, records the quad, submits, and optionally audits it. The eye
 * is fixed so two renders of "the same frame" really are the same frame. */
static ReconLResult record_and_submit(Rig* r, int keep, int audit) {
    ReconLLight light; memset(&light, 0, sizeof light);
    SETBASE(&light, RECONL_STRUCT_LIGHT);
    light.type = RECONL_LIGHT_DIRECTIONAL; light.direction[2] = -1.0f;
    light.color[0] = light.color[1] = light.color[2] = 1.0f; light.intensity = 1.0f;
    ReconLLightList lights; memset(&lights, 0, sizeof lights);
    SETBASE(&lights, RECONL_STRUCT_LIGHT_LIST);
    lights.count = 1; lights.lights = &light;
    ReconLCamera cam; memset(&cam, 0, sizeof cam);
    SETBASE(&cam, RECONL_STRUCT_CAMERA);
    view_at(0.0f, cam.view); cam.fov_y_deg = 60.0f; cam.near = 0.1f; cam.far = 100.0f;
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
    if ((rr = reconlCmdEndRenderPass(r->cl)) != RECONL_OK) return rr;
    if ((rr = reconlSubmit(r->dev, r->cl, NULL)) != RECONL_OK) return rr;
    if (audit) {
        uint32_t divergences = 0;
        ReconLResult a = reconlAudit(r->dev, 1, &divergences);
        printf("    (reconlAudit -> %d, divergences %u)\n", (int)a, divergences);
    }
    return RECONL_OK;
}

static ReconLResult present_into(Rig* r, void* buf, uint64_t buf_size, uint32_t pitch, int keep) {
    ReconLResult rr = record_and_submit(r, keep, 0);
    if (rr != RECONL_OK) { printf("    (record+submit -> %d)\n", (int)rr); return rr; }
    ReconLPresentDesc prd; memset(&prd, 0, sizeof prd);
    SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
    prd.out_pixels = buf; prd.out_pixels_size = buf_size;
    prd.out_row_pitch = pitch; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return reconlPresent(r->dev, r->sc, &prd);
}

/* The audited route: record, submit, audit (which reads the frame out of the
 * driver), then present - the driver is not asked again for the same frame. */
static ReconLResult present_audited_into(Rig* r, void* buf, uint64_t buf_size, uint32_t pitch) {
    ReconLResult rr = record_and_submit(r, 0, 1);
    if (rr != RECONL_OK) { printf("    (record+submit+audit -> %d)\n", (int)rr); return rr; }
    ReconLPresentDesc prd; memset(&prd, 0, sizeof prd);
    SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
    prd.out_pixels = buf; prd.out_pixels_size = buf_size;
    prd.out_row_pitch = pitch; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return reconlPresent(r->dev, r->sc, &prd);
}

static ReconLResult generate_into(Rig* r, float ahead, void* buf, uint64_t buf_size, uint32_t pitch) {
    ReconLPresentDesc prd; memset(&prd, 0, sizeof prd);
    SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
    prd.out_pixels = buf; prd.out_pixels_size = buf_size;
    prd.out_row_pitch = pitch; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return reconlPresentGenerated(r->dev, r->sc, &prd, ahead);
}

static size_t unwritten(const unsigned char* buf, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) if (buf[i] == 0xAB) ++k;
    return k;
}

static size_t differs(const unsigned char* a, const unsigned char* b, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) ++k;
    return k;
}

/* What the host's buffer looks like for one accepted narrow pitch, measured
 * against the same frame laid out tightly. */
static void describe(Rig* r, const char* what, unsigned char* buf, size_t n,
                     unsigned char* ref, uint32_t pitch, int keep) {
    memset(buf, 0xAB, n);
    ReconLResult rr = present_into(r, buf, (uint64_t)n, pitch, keep);
    if (rr != RECONL_OK) {
        printf("      %s: refused (%d), nothing written: %zu bytes still 0xAB\n",
               what, (int)rr, unwritten(buf, n));
        return;
    }
    printf("      %s: accepted (%d); never written %zu of %zu bytes;\n",
           what, (int)rr, unwritten(buf, n), n);
    printf("        %zu of %zu bytes differ from the same frame laid out tightly\n",
           differs(buf, ref, n), n);
}

int main(int argc, char** argv) {
    const char* name = arg("--backend", argc, argv);
    if (name && strcmp(name, "d3d11") == 0) backend = RECONL_BACKEND_D3D11;
    else if (name && strcmp(name, "null") == 0) backend = RECONL_BACKEND_NULL;
    const char* sz = arg("--size", argc, argv);
    if (sz) size = (uint32_t)strtoul(sz, NULL, 10);

    Rig rig; memset(&rig, 0, sizeof rig);
    if (!build(&rig)) { printf("build failed\n"); return 2; }

    printf("narrowpitch — asked %s, %ux%u\n",
           backend == RECONL_BACKEND_D3D11 ? "d3d11"
           : backend == RECONL_BACKEND_NULL ? "null" : "soft-cpu", size, size);

    const uint32_t tight = size * 4;            /* one row of RGBA8 */
    const uint32_t narrow = tight - 4;          /* one pixel narrower than a row */
    const uint64_t generous = (uint64_t)size * tight;
    const size_t n = (size_t)generous;

    unsigned char* buf = (unsigned char*)malloc(n);
    unsigned char* ref = (unsigned char*)malloc(n);

    /* A reference: this frame, laid out tightly and legally. */
    {
        ReconLResult rr = present_into(&rig, ref, generous, tight, 0);
        printf("    reference (tight, framegen off) -> %d\n", (int)rr);
    }

    printf("\n[1] reconlWriteTexture, row_pitch=%u (row is %u) — the control\n", narrow, tight);
    {
        ReconLTextureDesc td; memset(&td, 0, sizeof td);
        SETBASE(&td, RECONL_STRUCT_TEXTURE_DESC);
        td.width = size; td.height = size; td.format = RECONL_FORMAT_R8G8B8A8_UNORM;
        td.usage = RECONL_TEXTURE_SAMPLED;
        ReconLTexture* tex = NULL;
        ReconLResult cr = reconlCreateTexture(rig.dev, &td, &tex);
        if (cr != RECONL_OK) { printf("    create texture -> %d\n", (int)cr); }
        else {
            ReconLTextureLevel lv; memset(&lv, 0, sizeof lv);
            SETBASE(&lv, RECONL_STRUCT_TEXTURE_DESC);
            lv.mip = 0; lv.layer = 0; lv.data = ref; lv.data_size = generous;
            lv.row_pitch = narrow; lv.row_count = size;
            ReconLResult w = reconlWriteTexture(rig.dev, tex, &lv);
            printf("    write -> %d\n", (int)w);
            CHECK("upload refuses a row_pitch narrower than a row", w == RECONL_ERR_INVALID_ARGUMENT);

            printf("\n[2] reconlReadTexture, out_row_pitch=%u — the cached readback\n", narrow);
            ReconLTextureLevel ok; memset(&ok, 0, sizeof ok);
            SETBASE(&ok, RECONL_STRUCT_TEXTURE_DESC);
            ok.mip = 0; ok.layer = 0; ok.data = ref; ok.data_size = generous;
            ok.row_pitch = tight; ok.row_count = size;
            ReconLResult wo = reconlWriteTexture(rig.dev, tex, &ok);
            printf("    (a legal upload first -> %d)\n", (int)wo);
            memset(buf, 0xAB, n);
            ReconLResult rd = reconlReadTexture(rig.dev, tex, 0, 0, buf, generous, narrow);
            printf("    read -> %d; never written %zu of %zu bytes\n", (int)rd, unwritten(buf, n), n);
            CHECK("texture readback refuses an out_row_pitch narrower than a row",
                  rd == RECONL_ERR_INVALID_ARGUMENT);
            reconlRelease(tex);
        }
    }

    printf("\n[3] reconlPresent, framegen OFF, out_row_pitch=%u — the backend readback\n", narrow);
    {
        ReconLResult p = present_into(&rig, buf, generous, narrow, 0);
        printf("    present -> %d\n", (int)p);
        CHECK("present refuses a pitch narrower than a row", p == RECONL_ERR_INVALID_ARGUMENT);
        ReconLResult t = present_into(&rig, buf, generous, tight, 0);
        CHECK("the device still presents afterwards", t == RECONL_OK);
    }

    printf("\n[4] reconlPresent, framegen ON, out_row_pitch=%u — the keep path\n", narrow);
    {
        ReconLResult p = present_into(&rig, buf, generous, narrow, 1);
        printf("    present -> %d\n", (int)p);
        CHECK("present refuses a narrow pitch when the frame is kept", p == RECONL_ERR_INVALID_ARGUMENT);
        ReconLResult t = present_into(&rig, buf, generous, tight, 1);
        CHECK("the device still presents afterwards", t == RECONL_OK);
    }

    printf("\n[5] reconlPresent after reconlAudit, out_row_pitch=%u — the cached readback\n", narrow);
    {
        ReconLResult p = present_audited_into(&rig, buf, generous, narrow);
        printf("    present -> %d\n", (int)p);
        CHECK("an audited present refuses a pitch narrower than a row",
              p == RECONL_ERR_INVALID_ARGUMENT);
        ReconLResult t = present_into(&rig, buf, generous, tight, 0);
        CHECK("the device still presents afterwards", t == RECONL_OK);
    }

    printf("\n[6] reconlPresentGenerated, out_row_pitch=%u — the generated path\n", narrow);
    {
        ReconLResult s = present_into(&rig, ref, generous, tight, 1);
        printf("    (a tight kept frame first -> %d)\n", (int)s);
        /* The null tier cannot generate at all, so this route is closed there
         * and the pitch never reaches a layout: NOT_SUPPORTED is the answer a
         * host should get, on any pitch. */
        const int null_tier = backend == RECONL_BACKEND_NULL;
        const int refused = null_tier ? RECONL_ERR_NOT_SUPPORTED : RECONL_ERR_INVALID_ARGUMENT;
        ReconLResult g = generate_into(&rig, 1.0f, buf, generous, narrow);
        printf("    generated -> %d\n", (int)g);
        CHECK("a generated frame refuses a pitch narrower than a row", g == refused);
        ReconLResult g4 = generate_into(&rig, 1.0f, buf, generous, 4);
        printf("    generated with pitch=4 -> %d\n", (int)g4);
        CHECK("an absurd generated pitch is refused", g4 == refused);
        ReconLResult gt = generate_into(&rig, 1.0f, buf, generous, tight);
        CHECK("generation works with a tight pitch (or is unsupported, on null)",
              gt == (null_tier ? RECONL_ERR_NOT_SUPPORTED : RECONL_OK));
    }

    if (backend != RECONL_BACKEND_NULL) {
        printf("\n[7] what the host's buffer looks like, this frame, at pitch=%u and pitch=4\n", narrow);
        unsigned char* scratch = (unsigned char*)malloc(n);
        describe(&rig, "present (framegen off)", scratch, n, ref, narrow, 0);
        describe(&rig, "present (framegen on) ", scratch, n, ref, narrow, 1);
        {
            ReconLResult s = present_into(&rig, ref, generous, tight, 1);
            if (s == RECONL_OK) {
                memset(scratch, 0xAB, n);
                ReconLResult g = generate_into(&rig, 1.0f, scratch, generous, 4);
                printf("      generated, pitch=4: -> %d; never written %zu of %zu bytes\n",
                       (int)g, unwritten(scratch, n), n);
            }
        }
        free(scratch);
    }

    printf("\nnarrowpitch: %d checks, %d findings\n", checks, findings);
    free(buf);
    free(ref);
    destroy(&rig);
    return findings ? 1 : 0;
}
