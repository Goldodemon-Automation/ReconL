/* Frame generation under hostile and unexercised conditions, through the
 * shipped DLL: the paths the ABI tests and the `framegen` probe never reach.
 *
 * Read-only audit: nothing here is meant to fix anything, it exists to find out
 * what the shipped library actually does.
 *
 * Usage: fghostile --backend=soft-cpu|d3d11 [--size=64]
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
static long live_allocs = 0;
static long long live_bytes = 0;

static void CHECK(const char* what, int ok) {
    ++checks;
    printf("  %s  %s\n", ok ? "ok  " : "FIND", what);
    if (!ok) ++findings;
}

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
    dd.backend_hint = backend; dd.tier_hint = RECONL_TIER_T2_CPU_RAM; dd.worker_threads = 1; dd.seed = 1;
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

/* Opens a frame with generation asked for, records the quad and stops. */
static ReconLResult begin_and_record(Rig* r, float eye, int keep) {
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

static ReconLResult present_frame(Rig* r, unsigned char* out, uint32_t pitch, uint32_t flip) {
    ReconLPresentDesc prd; memset(&prd, 0, sizeof prd);
    SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
    prd.out_pixels = out; prd.out_pixels_size = (uint64_t)size * pitch;
    prd.out_row_pitch = pitch; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM; prd.flip = flip;
    return reconlPresent(r->dev, r->sc, &prd);
}

static ReconLResult gen(Rig* r, float ahead, unsigned char* out, uint32_t pitch, uint32_t flip) {
    ReconLPresentDesc prd; memset(&prd, 0, sizeof prd);
    SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
    prd.out_pixels = out; prd.out_pixels_size = (uint64_t)size * pitch;
    prd.out_row_pitch = pitch; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM; prd.flip = flip;
    return reconlPresentGenerated(r->dev, r->sc, &prd, ahead);
}

static int same(const unsigned char* a, const unsigned char* b, size_t n) {
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) return 0;
    return 1;
}

int main(int argc, char** argv) {
    const char* name = arg("--backend", argc, argv);
    if (name && strcmp(name, "d3d11") == 0) backend = RECONL_BACKEND_D3D11;
    const char* sz = arg("--size", argc, argv);
    if (sz) size = (uint32_t)strtoul(sz, NULL, 10);

    Rig rig; memset(&rig, 0, sizeof rig);
    if (!build(&rig)) { printf("build failed\n"); return 2; }
    printf("fghostile — %s, %ux%u\n", backend == RECONL_BACKEND_D3D11 ? "d3d11" : "soft-cpu", size, size);

    unsigned char* frame = (unsigned char*)malloc((size_t)size * size * 4);
    unsigned char* tight = (unsigned char*)malloc((size_t)size * size * 4);
    unsigned char* other = (unsigned char*)malloc((size_t)size * size * 4);
    const uint32_t pitch = size * 4 + 16;
    unsigned char* padded = (unsigned char*)malloc((size_t)pitch * size);

    /* 1. The order both present calls decide their answers in, on the descriptor
     *    type: the tier, then whether there is anything to present or generate
     *    from, and only then the descriptor. So with nothing available both calls
     *    answer the same thing for a valid descriptor and for a wrong one - which
     *    is the like-for-like comparison, and the one the section-3b pair below
     *    completes once each call is in the state that accepts it. */
    {
        ReconLPresentDesc prd; memset(&prd, 0, sizeof prd);
        prd.base.struct_size = sizeof prd;
        prd.base.type = RECONL_STRUCT_FRAME_DESC; /* wrong on purpose */
        prd.out_pixels = tight;
        prd.out_pixels_size = (uint64_t)size * size * 4;
        prd.out_row_pitch = size * 4;
        prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
        ReconLResult r = reconlPresentGenerated(rig.dev, rig.sc, &prd, 1.0f);
        printf("    generated with a wrong desc type, nothing asked yet -> %d\n", (int)r);
        CHECK("with nothing to generate from, the generated call reads no descriptor",
              r == RECONL_ERR_NO_FRAME);
        prd.base.type = RECONL_STRUCT_PRESENT_DESC;
        ReconLResult pv = reconlPresent(rig.dev, rig.sc, &prd);
        prd.base.type = RECONL_STRUCT_FRAME_DESC;
        ReconLResult pw = reconlPresent(rig.dev, rig.sc, &prd);
        printf("    nothing to present: reconlPresent with the right type -> %d, with the wrong type -> %d\n",
               (int)pv, (int)pw);
        CHECK("and the present path reads none either with no frame to present",
              pv == RECONL_ERR_NO_FRAME && pw == RECONL_ERR_NO_FRAME && r == pv);
    }

    /* 2. The OPEN and SUBMITTED states: generating must not touch the frame
     *    state machine, and the frame must still complete. */
    {
        ReconLResult r = begin_and_record(&rig, 0.0f, 1);
        CHECK("a frame opens", r == RECONL_OK);
        r = gen(&rig, 0.5f, tight, size * 4, 0);
        printf("    generate in the OPEN state -> %d\n", (int)r);
        r = reconlSubmit(rig.dev, rig.cl, NULL);
        CHECK("the frame still submits after a generated frame in the OPEN state", r == RECONL_OK);
        ReconLResult g = gen(&rig, 0.5f, other, size * 4, 0);
        printf("    generate in the SUBMITTED state -> %d\n", (int)g);
        r = present_frame(&rig, frame, size * 4, 0);
        CHECK("the frame still presents after a generated frame in the SUBMITTED state", r == RECONL_OK);
    }

    /* 3. A second asking frame, so there is a real camera pair, then the layout
     *    contract of a generated present: a padded bottom-up buffer. */
    {
        ReconLResult r = begin_and_record(&rig, 0.7f, 1);
        if (r == RECONL_OK) r = reconlSubmit(rig.dev, rig.cl, NULL);
        if (r == RECONL_OK) r = present_frame(&rig, frame, size * 4, 0);
        CHECK("a second asking frame renders", r == RECONL_OK);

        CHECK("a generated frame is delivered", gen(&rig, 0.75f, tight, size * 4, 0) == RECONL_OK);
        /* Determinism: the same pair, the same look-ahead, the same bytes. */
        CHECK("two generated frames of one pair are identical",
              gen(&rig, 0.75f, other, size * 4, 0) == RECONL_OK && same(tight, other, (size_t)size * size * 4));

        memset(padded, 0xAB, (size_t)pitch * size);
        ReconLResult rp = gen(&rig, 0.75f, padded, pitch, 1);
        CHECK("a generated frame into a padded, bottom-up buffer is delivered", rp == RECONL_OK);
        int flipped_ok = 1, padding_ok = 1;
        for (uint32_t y = 0; y < size && flipped_ok; ++y) {
            const unsigned char* src = tight + (size_t)(size - 1 - y) * size * 4;
            const unsigned char* dst = padded + (size_t)y * pitch;
            if (memcmp(src, dst, (size_t)size * 4) != 0) flipped_ok = 0;
            for (uint32_t x = size * 4; x < pitch; ++x)
                if (dst[x] != 0xAB) padding_ok = 0;
        }
        CHECK("flip = 1 writes the image bottom-up", flipped_ok);
        CHECK("a generated present leaves the padding untouched", padding_ok);
    }

    /* 3b. Descriptor type validation, with the feature live (so the call is not
     *     refused earlier for having nothing to generate from) and with a frame
     *     submitted (so the present state machine accepts the call). */
    {
        ReconLPresentDesc prd; memset(&prd, 0, sizeof prd);
        prd.base.struct_size = sizeof prd;
        prd.base.type = RECONL_STRUCT_FRAME_DESC; /* wrong on purpose */
        prd.out_pixels = tight;
        prd.out_pixels_size = (uint64_t)size * size * 4;
        prd.out_row_pitch = size * 4;
        prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
        ReconLResult g = reconlPresentGenerated(rig.dev, rig.sc, &prd, 1.0f);
        printf("    generated, wrong desc type, feature live -> %d\n", (int)g);
        CHECK("a generated present checks the descriptor type", g == RECONL_ERR_WRONG_STRUCT_TYPE);

        ReconLResult open = begin_and_record(&rig, 1.1f, 1);
        if (open == RECONL_OK) open = reconlSubmit(rig.dev, rig.cl, NULL);
        CHECK("a third asking frame renders", open == RECONL_OK);
        ReconLResult p = reconlPresent(rig.dev, rig.sc, &prd);
        printf("    reconlPresent, wrong desc type, frame submitted -> %d\n", (int)p);
        CHECK("reconlPresent checks the descriptor type", p == RECONL_ERR_WRONG_STRUCT_TYPE);
        /* The refused present consumed the frame, per the documented contract. */
        ReconLPresentDesc good; memset(&good, 0, sizeof good);
        SETBASE(&good, RECONL_STRUCT_PRESENT_DESC);
        good.out_pixels = frame; good.out_pixels_size = (uint64_t)size * size * 4;
        good.out_row_pitch = size * 4; good.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
        printf("    the same present with a good descriptor -> %d\n", (int)reconlPresent(rig.dev, rig.sc, &good));
    }

    /* 3c. The pitch rule, on both calls, each with a frame the state machine
     *     accepts: the same descriptor type, so the same documented rule. */
    {
        ReconLResult r = begin_and_record(&rig, 1.4f, 1);
        if (r == RECONL_OK) r = reconlSubmit(rig.dev, rig.cl, NULL);
        CHECK("a fourth asking frame renders", r == RECONL_OK);
        ReconLPresentDesc narrow; memset(&narrow, 0, sizeof narrow);
        SETBASE(&narrow, RECONL_STRUCT_PRESENT_DESC);
        narrow.out_pixels = padded;
        narrow.out_pixels_size = (uint64_t)size * size * 4;
        narrow.out_row_pitch = size * 4 - 4;
        narrow.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
        ReconLResult pr = reconlPresent(rig.dev, rig.sc, &narrow);
        printf("    reconlPresent, narrow pitch, frame submitted -> %d\n", (int)pr);
        CHECK("reconlPresent refuses a row pitch narrower than a row", pr == RECONL_ERR_INVALID_ARGUMENT);

        /* A fresh frame, presented normally so the feature keeps it, then the
         * same narrow descriptor through the generated call. */
        r = begin_and_record(&rig, 1.6f, 1);
        if (r == RECONL_OK) r = reconlSubmit(rig.dev, rig.cl, NULL);
        if (r == RECONL_OK) r = present_frame(&rig, frame, size * 4, 0);
        CHECK("a fifth asking frame renders", r == RECONL_OK);
        ReconLResult gr = reconlPresentGenerated(rig.dev, rig.sc, &narrow, 0.5f);
        printf("    reconlPresentGenerated, same narrow pitch -> %d\n", (int)gr);
        CHECK("reconlPresentGenerated refuses a row pitch narrower than a row",
              gr == RECONL_ERR_INVALID_ARGUMENT);
        /* Like for like: the same descriptor through both calls, each in the
         * state that accepts it - a submitted frame for one, a kept frame for
         * the other. Comparing either with a call the state does not accept is
         * comparing two different questions, which is what this probe used to
         * do (and reported as a finding). */
        CHECK("in the state that accepts each call, both refuse the same narrow pitch",
              pr == RECONL_ERR_INVALID_ARGUMENT && gr == RECONL_ERR_INVALID_ARGUMENT);
    }

    /* 4. Look-ahead boundaries. */
    {
        float tiny = 1.0e-30f;
        printf("    ahead: 1.0 -> %d, 1.0000001 -> %d, 1e-30 -> %d, -0.0 -> %d, nan -> %d\n",
               (int)gen(&rig, 1.0f, tight, size * 4, 0),
               (int)gen(&rig, 1.0000001f, tight, size * 4, 0),
               (int)gen(&rig, tiny, tight, size * 4, 0),
               (int)gen(&rig, -0.0f, tight, size * 4, 0),
               (int)gen(&rig, 0.0f / 0.0f, tight, size * 4, 0));
        CHECK("ahead of exactly 1.0 is accepted", gen(&rig, 1.0f, tight, size * 4, 0) == RECONL_OK);
        CHECK("ahead above 1.0 is refused", gen(&rig, 1.0000001f, tight, size * 4, 0) == RECONL_ERR_INVALID_ARGUMENT);
        CHECK("ahead of -0.0 is refused", gen(&rig, -0.0f, tight, size * 4, 0) == RECONL_ERR_INVALID_ARGUMENT);
        CHECK("a NaN ahead is refused", gen(&rig, 0.0f / 0.0f, tight, size * 4, 0) == RECONL_ERR_INVALID_ARGUMENT);
    }

    /* 5. Buffers that cannot hold the frame. */
    {
        CHECK("a null out_pixels is refused", gen(&rig, 0.5f, NULL, size * 4, 0) == RECONL_ERR_INVALID_ARGUMENT);
        ReconLPresentDesc prd; memset(&prd, 0, sizeof prd);
        SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
        prd.out_pixels = tight; prd.out_pixels_size = 0; prd.out_row_pitch = size * 4;
        prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
        CHECK("a zero-sized buffer is refused", reconlPresentGenerated(rig.dev, rig.sc, &prd, 0.5f) == RECONL_ERR_INVALID_ARGUMENT);
        prd.out_pixels_size = (uint64_t)size * size * 4 - 4;
        CHECK("a buffer one row short is refused", reconlPresentGenerated(rig.dev, rig.sc, &prd, 0.5f) == RECONL_ERR_INVALID_ARGUMENT);
        prd.out_pixels_size = (uint64_t)size * size * 4;
        prd.out_row_pitch = size * 4 - 4; /* narrower than a row */
        unsigned char* narrow_buf = (unsigned char*)malloc((size_t)size * size * 4);
        memset(narrow_buf, 0xCD, (size_t)size * size * 4);
        prd.out_pixels = narrow_buf;
        ReconLResult narrow = reconlPresentGenerated(rig.dev, rig.sc, &prd, 0.5f);
        printf("    out_row_pitch narrower than a row -> %d\n", (int)narrow);
        CHECK("a generated present with a row pitch narrower than a row is refused", narrow == RECONL_ERR_INVALID_ARGUMENT);
        if (narrow == RECONL_OK) {
            gen(&rig, 0.5f, tight, size * 4, 0);
            long smeared = 0;
            for (uint32_t y = 0; y < size; ++y) {
                const unsigned char* want = tight + (size_t)y * size * 4;
                const unsigned char* got = narrow_buf + (size_t)y * (size * 4 - 4);
                for (uint32_t x = 0; x < size * 4 - 8; ++x) if (want[x] != got[x]) ++smeared;
            }
            printf("    the accepted narrow pitch smeared %ld bytes of the image\n", smeared);
        }
        prd.out_pixels = tight;
        prd.out_row_pitch = size * 4;
        ReconLPresentDesc rpd; memset(&rpd, 0, sizeof rpd);
        SETBASE(&rpd, RECONL_STRUCT_PRESENT_DESC);
        rpd.out_pixels = tight; rpd.out_pixels_size = (uint64_t)size * size * 4;
        rpd.out_row_pitch = size * 4 - 4; rpd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
        ReconLResult present_narrow = reconlPresent(rig.dev, rig.sc, &rpd);
        printf("    reconlPresent with the same narrow pitch, no frame to present -> %d\n",
               (int)present_narrow);
        /* This call is in the idle state, where the machine answers before the
         * descriptor does - the order the header states and section 1 pins. The
         * like-for-like pitch comparison is the one above, both calls in the
         * state that accepts each. */
        CHECK("with nothing to present, a narrow pitch is the state's answer",
              present_narrow == RECONL_ERR_NO_FRAME);
    }

    /* 6. A swapchain from another device, and a handle of the wrong kind.
     *
     * Deliberately *not* an already-released handle. What a pointer to freed
     * memory does is undefined by the ABI - the header's contract is ref-counted
     * lifetime, not "a stale pointer is refused" - so asserting a code for one
     * asks the library to read memory its host has handed back, and reads it
     * whether or not the page is still mapped. That cell segfaulted about one
     * run in three on d3d11 (the allocator is free to return the page), which is
     * a defect in the probe, not in the library. A live handle of the wrong kind
     * is the sound version of the property this cell is for: the library must
     * reject a handle it was not given for this parameter by its kind word, and
     * never follow it. */
    {
        Rig other_rig; memset(&other_rig, 0, sizeof other_rig);
        if (build(&other_rig)) {
            ReconLPresentDesc prd; memset(&prd, 0, sizeof prd);
            SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
            prd.out_pixels = tight; prd.out_pixels_size = (uint64_t)size * size * 4;
            prd.out_row_pitch = size * 4; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
            ReconLResult r = reconlPresentGenerated(rig.dev, other_rig.sc, &prd, 0.5f);
            printf("    another device's swapchain -> %d\n", (int)r);
            CHECK("another device's swapchain is refused, not confused", r != RECONL_OK);
            /* A live command list where the swapchain belongs: same check, wrong
             * kind word. The cast is the point of the cell. */
            ReconLResult wrong_kind =
                reconlPresentGenerated(rig.dev, (ReconLSwapchain*)other_rig.cl, &prd, 0.5f);
            printf("    a command list where a swapchain belongs -> %d\n", (int)wrong_kind);
            CHECK("a handle of the wrong kind is refused, not followed",
                  wrong_kind == RECONL_ERR_INVALID_HANDLE);
            destroy(&other_rig);
        }
    }

    /* 7. The history must be given back: the device holds three host buffers
     *    while the feature is on. */
    {
        long before = live_allocs;
        destroy(&rig);
        printf("    live allocations before destroy %ld, after %ld (%lld bytes live)\n", before, live_allocs, live_bytes);
        CHECK("releasing a device with a retained history frees every byte it took", live_allocs == 0);
    }
    {
        live_allocs = 0; live_bytes = 0;
        for (int i = 0; i < 20; ++i) {
            Rig r; memset(&r, 0, sizeof r);
            if (!build(&r)) break;
            if (begin_and_record(&r, 0.0f, 1) == RECONL_OK) {
                reconlSubmit(r.dev, r.cl, NULL);
                present_frame(&r, tight, size * 4, 0);
                gen(&r, 1.0f, other, size * 4, 0);
            }
            destroy(&r);
        }
        printf("    live allocations after 20 create/keep/release cycles: %ld\n", live_allocs);
        CHECK("20 create/keep/release cycles leave nothing live", live_allocs == 0);
    }

    printf("fghostile: %d checks, %d findings\n", checks, findings);
    return findings ? 1 : 0;
}
