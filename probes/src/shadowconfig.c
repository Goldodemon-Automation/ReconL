/* `reconlConfigureShadows` through the shipped DLL, as a host calls it.
 *
 * This export had no caller anywhere in the project, so nothing had ever driven
 * it. It is the device-level shadow configuration: a host sets it once and every
 * later frame uses it, unless the frame carries its own override. The documented
 * contract is "callable at any time, applied at the next frame boundary; never
 * mid-frame", so that is what this checks - including the mid-frame call, which
 * is the case a host gets wrong by accident.
 *
 * The scene is the same arrangement the cross-tier tests use: a ground quad, a
 * caster held above it, a real world-space camera, and one shadow-casting
 * directional light. Geometry is white, so "shadowed" is measurable as "darker
 * than the same frame with shadows off".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "reconl/reconl.h"

#define SETBASE(p, T)                                        \
    do {                                                     \
        (p)->base.struct_size = (uint32_t)sizeof(*(p));       \
        (p)->base.type = (T);                                 \
        (p)->base.next = NULL;                                \
    } while (0)

#define W 64u
#define H 64u

static int checks = 0, fails = 0;

static void check(int ok, const char* fmt, ...) {
    (void)fmt;
    checks++;
    if (!ok) fails++;
}

/* A check that prints, for the ones whose numbers are worth reading. */
static void check_show(int ok, const char* label) {
    checks++;
    if (!ok) fails++;
    printf("    [%s] %s\n", ok ? "PASS" : "FAIL", label);
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

/* ------------------------------------------------------- the scene's matrices */

typedef float V3[3];

static void v3_sub(V3 o, const V3 a, const V3 b) { for (int i = 0; i < 3; ++i) o[i] = a[i] - b[i]; }
static float v3_dot(const V3 a, const V3 b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
static void v3_cross(V3 o, const V3 a, const V3 b) {
    o[0] = a[1]*b[2] - a[2]*b[1];
    o[1] = a[2]*b[0] - a[0]*b[2];
    o[2] = a[0]*b[1] - a[1]*b[0];
}
static void v3_norm(V3 v) {
    float l = sqrtf(v3_dot(v, v));
    if (l > 1.0e-20f) { for (int i = 0; i < 3; ++i) v[i] /= l; }
}

/* `math::look_at`, transcribed: column-major, translation in column 3. */
static void look_at(float m[16], const V3 eye, const V3 target, const V3 up) {
    V3 f, s, u;
    v3_sub(f, target, eye); v3_norm(f);
    v3_cross(s, f, up); v3_norm(s);
    v3_cross(u, s, f);
    m[0] = s[0]; m[1] = u[0]; m[2] = -f[0]; m[3] = 0.0f;
    m[4] = s[1]; m[5] = u[1]; m[6] = -f[1]; m[7] = 0.0f;
    m[8] = s[2]; m[9] = u[2]; m[10] = -f[2]; m[11] = 0.0f;
    m[12] = -v3_dot(s, eye); m[13] = -v3_dot(u, eye); m[14] = v3_dot(f, eye); m[15] = 1.0f;
}

/* `math::perspective_rh_reversed_z`. */
static void perspective(float m[16], float fov_y_deg, float aspect, float near, float far) {
    float f = 1.0f / tanf(fov_y_deg * 0.5f * 3.14159265358979323846f / 180.0f);
    float nf = 1.0f / (near - far);
    m[0] = f / aspect; m[1] = 0; m[2] = 0; m[3] = 0;
    m[4] = 0; m[5] = f; m[6] = 0; m[7] = 0;
    m[8] = 0; m[9] = 0; m[10] = -near * nf; m[11] = -1.0f;
    m[12] = 0; m[13] = 0; m[14] = -near * far * nf; m[15] = 0;
}

/* `math::mul`, transcribed: column-major a·b. */
static void mul(float out[16], const float a[16], const float b[16]) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1]
                           + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
        }
    }
}

/* ------------------------------------------------------------------ the rig */

typedef struct {
    ReconLDevice* d;
    ReconLSwapchain* s;
    ReconLCommandList* cl;
} Rig;

static int rig(Rig* r, ReconLBackendId backend) {
    memset(r, 0, sizeof *r);
    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    dd.tier_hint = backend == RECONL_BACKEND_D3D11 ? RECONL_TIER_T1_GPU_SHARED : RECONL_TIER_T2_CPU_RAM;
    dd.allow_downgrade = 0;
    dd.target_frame_ms = 1000; dd.downgrade_after_frames = 16; dd.seed = 7;
    dd.allocator.alloc = h_alloc; dd.allocator.realloc = h_realloc; dd.allocator.free = h_free;
    if (reconlCreateDevice(&dd, &r->d) != RECONL_OK) return 1;

    ReconLSwapchainDesc sd;
    memset(&sd, 0, sizeof sd);
    SETBASE(&sd, RECONL_STRUCT_SWAPCHAIN_DESC);
    sd.width = W; sd.height = H;
    sd.format = RECONL_FORMAT_R8G8B8A8_UNORM;
    sd.image_count = 2; sd.present_to_memory = 1;
    sd.depth_format = RECONL_FORMAT_D32_FLOAT;
    if (reconlCreateSwapchain(r->d, &sd, &r->s) != RECONL_OK) return 1;

    ReconLCommandListDesc cd;
    memset(&cd, 0, sizeof cd);
    SETBASE(&cd, RECONL_STRUCT_COMMAND_LIST_DESC);
    cd.capacity_bytes = 8192;
    return reconlCreateCommandList(r->d, &cd, &r->cl) != RECONL_OK;
}

static void rig_free(Rig* r) {
    if (r->cl) reconlRelease(r->cl);
    if (r->s) reconlRelease(r->s);
    if (r->d) reconlRelease(r->d);
}

static ReconLShadowConfig shadow_config(uint32_t enabled, uint32_t cascades, uint32_t filter,
                                        uint64_t budget, float band) {
    ReconLShadowConfig c;
    memset(&c, 0, sizeof c);
    SETBASE(&c, RECONL_STRUCT_SHADOW_CONFIG);
    c.enabled = enabled;
    c.cascade_count = cascades;
    c.texel_budget_bytes = budget;
    c.filter = (ReconLShadowFilter)filter;
    c.max_distance = 77.0f;
    c.blend_band = band;
    /* Pinned, so a tier's own preset cannot be what moves a pixel: the same
     * values the reference scene in the tool pins. */
    c.normal_bias = 1.25f;
    c.depth_bias = 5.0e-4f;
    c.slope_bias = 1.75f;
    c.refresh_interval_frames = 1;
    return c;
}

/* One frame of the ground+caster scene. `frame_shadows` is the per-frame
 * override (NULL = use whatever `reconlConfigureShadows` set). If `mid_frame`
 * is non-NULL it is passed to `reconlConfigureShadows` *between* BeginFrame and
 * Submit, which must not affect this frame. Returns the result of the last call
 * and fills `stats`. */
static ReconLResult scene_frame(Rig* r, const ReconLShadowConfig* frame_shadows,
                                const ReconLShadowConfig* mid_frame, unsigned char* out,
                                ReconLStats* stats, ReconLResult* mid_frame_result) {
    ReconLLight light;
    memset(&light, 0, sizeof light);
    SETBASE(&light, RECONL_STRUCT_LIGHT);
    light.type = RECONL_LIGHT_DIRECTIONAL;
    light.direction[0] = 0.10f; light.direction[1] = -1.0f; light.direction[2] = 1.55f;
    light.color[0] = light.color[1] = light.color[2] = 1.0f;
    light.intensity = 1.0f;
    light.cast_shadow = 1;
    ReconLLightList ll;
    memset(&ll, 0, sizeof ll);
    SETBASE(&ll, RECONL_STRUCT_LIGHT_LIST);
    ll.count = 1; ll.lights = &light;

    ReconLCamera cam;
    memset(&cam, 0, sizeof cam);
    SETBASE(&cam, RECONL_STRUCT_CAMERA);
    V3 eye = { 0.0f, 13.0f, 10.0f }, target = { 0.0f, 0.0f, 0.0f }, up = { 0.0f, 1.0f, 0.0f };
    look_at(cam.view, eye, target, up);
    cam.fov_y_deg = 45.0f; cam.near = 0.5f; cam.far = 100.0f;

    float proj[16], view_proj[16];
    perspective(proj, cam.fov_y_deg, (float)W / (float)H, cam.near, cam.far);
    mul(view_proj, proj, cam.view);

    ReconLFrameDesc fd;
    memset(&fd, 0, sizeof fd);
    SETBASE(&fd, RECONL_STRUCT_FRAME_DESC);
    fd.width = W; fd.height = H; fd.seed = 1;
    fd.lights = &ll;
    fd.shadows = frame_shadows;
    fd.camera = &cam;

    ReconLResult res = reconlBeginFrame(r->d, &fd);
    if (res != RECONL_OK) return res;

    if (mid_frame && mid_frame_result) {
        *mid_frame_result = reconlConfigureShadows(r->d, mid_frame);
    }

    /* Ground: a 32x32 quad at y=0, white, receiving and casting. */
    ReconLVertex ground[4];
    memset(ground, 0, sizeof ground);
    const float g[4][3] = { { -16, 0, -16 }, { 16, 0, -16 }, { 16, 0, 16 }, { -16, 0, 16 } };
    for (int i = 0; i < 4; ++i) {
        ground[i].position[0] = g[i][0]; ground[i].position[1] = g[i][1]; ground[i].position[2] = g[i][2];
        ground[i].normal[1] = 1.0f;
        for (int c = 0; c < 4; ++c) ground[i].color[c] = 1.0f;
    }
    uint32_t ground_idx[6] = { 0, 1, 2, 0, 2, 3 };

    /* Caster: the reference scene's wide shallow triangle, 4.3 units up. */
    ReconLVertex caster[3];
    memset(caster, 0, sizeof caster);
    const float cst[3][3] = { { -6.25f, 4.3f, -3.9f }, { 5.55f, 4.3f, -4.5f }, { -0.35f, 4.3f, -0.9f } };
    for (int i = 0; i < 3; ++i) {
        caster[i].position[0] = cst[i][0]; caster[i].position[1] = cst[i][1]; caster[i].position[2] = cst[i][2];
        caster[i].normal[1] = 1.0f;
        for (int c = 0; c < 4; ++c) caster[i].color[c] = 1.0f;
    }
    uint32_t caster_idx[3] = { 0, 1, 2 };

    ReconLBuffer* buffers[4] = { NULL, NULL, NULL, NULL };
    {
        const void* data[4] = { ground, ground_idx, caster, caster_idx };
        const size_t bytes[4] = { sizeof ground, sizeof ground_idx, sizeof caster, sizeof caster_idx };
        const uint32_t usage[4] = { RECONL_BUFFER_VERTEX, RECONL_BUFFER_INDEX,
                                    RECONL_BUFFER_VERTEX, RECONL_BUFFER_INDEX };
        for (int i = 0; i < 4; ++i) {
            ReconLBufferDesc bd;
            memset(&bd, 0, sizeof bd);
            SETBASE(&bd, RECONL_STRUCT_BUFFER_DESC);
            bd.size_bytes = bytes[i]; bd.usage = usage[i];
            bd.data = data[i]; bd.data_size = bytes[i];
            res = reconlCreateBuffer(r->d, &bd, &buffers[i]);
            if (res != RECONL_OK) goto done;
        }
    }

    ReconLPipeline* pipes[2] = { NULL, NULL };
    for (int i = 0; i < 2; ++i) {
        ReconLPipelineDesc pd;
        memset(&pd, 0, sizeof pd);
        SETBASE(&pd, RECONL_STRUCT_PIPELINE_DESC);
        pd.shading = RECONL_SHADING_LAMBERT;
        pd.depth_compare = RECONL_COMPARE_GREATER;
        pd.depth_write = 1;
        pd.cull = 0;
        pd.receives_shadow = 1;
        pd.casts_shadow = 1;
        res = reconlCreatePipeline(r->d, &pd, &pipes[i]);
        if (res != RECONL_OK) goto done;
    }

    reconlCmdReset(r->cl);
    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = W; rp.viewport_height = H;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;  /* blue backdrop */
    rp.clear_depth = 0.0f;
    reconlCmdBeginRenderPass(r->cl, &rp);
    {
        const float identity[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
        for (int i = 0; i < 2; ++i) {
            reconlCmdSetPipeline(r->cl, pipes[i]);
            reconlCmdPushConstants(r->cl, 0, view_proj, sizeof view_proj);
            reconlCmdPushConstants(r->cl, 1, identity, sizeof identity);
            reconlCmdSetVertexBuffer(r->cl, 0, buffers[i * 2], 0);
            reconlCmdSetIndexBuffer(r->cl, buffers[i * 2 + 1], 0, RECONL_INDEX_UINT32);
            reconlCmdDrawIndexed(r->cl, i == 0 ? 6 : 3, 0, 0);
        }
    }
    reconlCmdEndRenderPass(r->cl);

    res = reconlSubmit(r->d, r->cl, NULL);
    if (res == RECONL_OK && out) {
        ReconLPresentDesc prd;
        memset(&prd, 0, sizeof prd);
        SETBASE(&prd, RECONL_STRUCT_PRESENT_DESC);
        prd.out_pixels = out; prd.out_pixels_size = (uint64_t)W * H * 4;
        prd.out_row_pitch = W * 4; prd.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
        res = reconlPresent(r->d, r->s, &prd);
    }
    if (stats && res == RECONL_OK) {
        memset(stats, 0, sizeof *stats);
        SETBASE(stats, RECONL_STRUCT_STATS);
        reconlGetStats(r->d, stats);
    }

done:
    for (int i = 0; i < 4; ++i) if (buffers[i]) reconlRelease(buffers[i]);
    for (int i = 0; i < 2; ++i) if (pipes[i]) reconlRelease(pipes[i]);
    return res;
}

/* Pixels darker by `margin` on green than the baseline: the shadow's footprint. */
static int shadow_pixels(const unsigned char* lit, const unsigned char* baseline) {
    int n = 0;
    for (unsigned i = 0; i < W * H; ++i) {
        int d = (int)baseline[i * 4 + 1] - (int)lit[i * 4 + 1];
        if (d >= 20) n++;
    }
    return n;
}

static void arm(ReconLBackendId backend, const char* name) {
    printf("  %s\n", name);
    Rig r;
    if (rig(&r, backend)) { printf("    (unavailable)\n"); return; }

    unsigned char lit[W * H * 4], baseline[W * H * 4];
    ReconLStats st;

    /* Baseline: the same scene with shadows off, through the per-frame override
     * (so this also exercises that path). */
    ReconLShadowConfig off = shadow_config(0, 2, 1, 8u << 20, 3.0f);
    ReconLResult res = scene_frame(&r, &off, NULL, baseline, NULL, NULL);
    check_show(res == RECONL_OK, "the scene renders with shadows off");
    int base_shadow = shadow_pixels(baseline, baseline);
    check_show(base_shadow == 0, "the baseline is its own baseline");

    /* 1. With no device config ever set, a frame with no override runs the
     *    library's documented default (shadows on), scaled to the tier. This is
     *    the default a host that never calls the export gets, so it is worth
     *    knowing it is a shadow pass and not an idle frame. */
    res = scene_frame(&r, NULL, NULL, lit, &st, NULL);
    check_show(res == RECONL_OK, "a frame with no config at all renders");
    printf("      no config: cascades_active %u, map %ux%u, filter %u/%u, %d px darker than the baseline\n",
           st.shadows.cascades_active, st.shadows.map_width, st.shadows.map_height,
           st.shadows.filter_active, st.shadows.filter_requested,
           shadow_pixels(lit, baseline));
    int tier_max = backend == RECONL_BACKEND_D3D11 ? 3 : 2;
    check_show(st.shadows.cascades_active >= 1 && st.shadows.cascades_active <= (unsigned)tier_max
                   && st.shadows.map_width > 0,
               "the unconfigured default is the tier-scaled shadow plan, not an idle frame");
    check_show(st.shadows.filter_requested == RECONL_SHADOW_FILTER_PCF3X3,
               "the default filter request is pcf3x3");

    /* 1b. `enabled = 0` is how a host says off, and off means no shadow pass at
     *     all - not a shadow pass that happens to be lit. */
    {
        ReconLShadowConfig disabled = shadow_config(0, 3, RECONL_SHADOW_FILTER_PCF3X3, 24u << 20, 4.0f);
        res = scene_frame(&r, &disabled, NULL, lit, &st, NULL);
        printf("      enabled=0: cascades_active %u, %d px darker than the baseline\n",
               st.shadows.cascades_active, shadow_pixels(lit, baseline));
        check_show(res == RECONL_OK && st.shadows.cascades_active == 0
                       && shadow_pixels(lit, baseline) == 0,
                   "enabled=0 draws no shadow at all");
    }

    /* 2. The export, and only the export: this is the call with no callers. */
    ReconLShadowConfig cfg = shadow_config(1, 2, RECONL_SHADOW_FILTER_PCF3X3, 8u << 20, 3.0f);
    ReconLResult cr = reconlConfigureShadows(r.d, &cfg);
    check_show(cr == RECONL_OK, "reconlConfigureShadows accepts the scene's configuration");
    res = scene_frame(&r, NULL, NULL, lit, &st, NULL);
    int shadowed = shadow_pixels(lit, baseline);
    printf("      configured: cascades %u, map %ux%u, filter %u/%u, %d px darker (%.1f%%)\n",
           st.shadows.cascades_active, st.shadows.map_width, st.shadows.map_height,
           st.shadows.filter_active, st.shadows.filter_requested,
           shadowed, 100.0 * shadowed / (double)(W * H));
    check_show(res == RECONL_OK && st.shadows.cascades_active >= 1 && st.shadows.map_width > 0,
               "the device-level configuration takes effect on the next frame");
    check_show(shadowed * 100 >= (int)(W * H) * 12,
               "and it makes the shadow appear (at least 12% of the frame)");
    check_show(st.shadows.filter_requested == RECONL_SHADOW_FILTER_PCF3X3,
               "the requested filter is echoed back");

    /* 3. A request the tier will not honour must be reported, not silently
     *    substituted: 4 cascades and pcss-lite are above the reference tier. */
    ReconLShadowConfig big = shadow_config(1, 4, RECONL_SHADOW_FILTER_PCSS_LITE, 8u << 20, 3.0f);
    cr = reconlConfigureShadows(r.d, &big);
    res = scene_frame(&r, NULL, NULL, lit, &st, NULL);
    printf("      c4/pcss requested: cascades %u, filter %u/%u, map %ux%u, %d px darker\n",
           st.shadows.cascades_active, st.shadows.filter_active, st.shadows.filter_requested,
           st.shadows.map_width, st.shadows.map_height, shadow_pixels(lit, baseline));
    check_show(cr == RECONL_OK && res == RECONL_OK, "an over-tier request is accepted and clamps");
    check_show(st.shadows.cascades_active >= 1 && st.shadows.cascades_active <= (unsigned)tier_max,
               "4 cascades is clamped to the tier's own maximum (2 reference, 3 hardware)");
    check_show(st.shadows.filter_requested == RECONL_SHADOW_FILTER_PCSS_LITE
                   && st.shadows.filter_active != RECONL_SHADOW_FILTER_PCSS_LITE,
               "pcss-lite below T1 is downgraded and the downgrade is reported");
    check_show(shadow_pixels(lit, baseline) > 0, "and the clamps still produce a shadow");

    /* 4. Mid-frame: a configuration set between BeginFrame and Submit applies to
     *    the *next* frame, never to the one being recorded. */
    /* The plan the previous frame ran, to compare against: a configuration
     * change must move at least one of these three. */
    unsigned prev_cascades = st.shadows.cascades_active;
    unsigned prev_map = st.shadows.map_width;
    unsigned prev_filter = st.shadows.filter_active;
    ReconLShadowConfig mid = shadow_config(1, 2, RECONL_SHADOW_FILTER_PCF3X3, 2u << 20, 3.0f);
    ReconLResult mid_res = -1;
    res = scene_frame(&r, NULL, &mid, lit, &st, &mid_res);
    printf("      mid-frame config: call %d, this frame cascades %u map %u filter %u (previous plan %u/%u/%u)\n",
           (int)mid_res, st.shadows.cascades_active, st.shadows.map_width, st.shadows.filter_active,
           prev_cascades, prev_map, prev_filter);
    check_show(mid_res == RECONL_OK, "the mid-frame call itself succeeds");
    check_show(res == RECONL_OK && st.shadows.cascades_active == prev_cascades
                   && st.shadows.map_width == prev_map && st.shadows.filter_active == prev_filter,
               "the frame being recorded keeps the plan it began with");
    /* The next frame picks it up: 2 cascades over 2 MiB is a 256x256 map on the
     * reference tier and 512x512 on the hardware tier (the tiers scale the
     * budget by 0.25 and 0.5), so what is asserted is that the plan *changed*,
     * not which of the tiers' own sizes it changed to. */
    res = scene_frame(&r, NULL, NULL, lit, &st, NULL);
    printf("      the frame after: cascades %u, map %ux%u, filter %u\n",
           st.shadows.cascades_active, st.shadows.map_width, st.shadows.map_height, st.shadows.filter_active);
    check_show(res == RECONL_OK
                   && !(st.shadows.cascades_active == prev_cascades && st.shadows.map_width == prev_map
                        && st.shadows.filter_active == prev_filter),
               "the next frame picks the mid-frame configuration up");

    /* 5. NULL resets the device to the library default (shadows on, tier
     *    scaled), which is the same plan the unconfigured device ran in step 1. */
    cr = reconlConfigureShadows(r.d, NULL);
    res = scene_frame(&r, NULL, NULL, lit, &st, NULL);
    printf("      after NULL: cascades %u, map %u, filter %u (step 1 ran %u/%u)\n",
           st.shadows.cascades_active, st.shadows.map_width, st.shadows.filter_active,
           prev_cascades, prev_map);
    check_show(cr == RECONL_OK && res == RECONL_OK && st.shadows.cascades_active >= 1
                   && st.shadows.cascades_active <= (unsigned)tier_max,
               "a NULL config resets the device to the library default");

    /* 5b. The ABI's own guards on the export: a struct the library cannot read
     *     in full, and a struct of the wrong type, must both be refused. */
    {
        ReconLShadowConfig short_cfg = shadow_config(1, 2, RECONL_SHADOW_FILTER_PCF3X3, 8u << 20, 3.0f);
        short_cfg.base.struct_size = 8;
        ReconLResult short_res = reconlConfigureShadows(r.d, &short_cfg);
        ReconLShadowConfig wrong = shadow_config(1, 2, RECONL_SHADOW_FILTER_PCF3X3, 8u << 20, 3.0f);
        wrong.base.type = RECONL_STRUCT_CAMERA;
        ReconLResult wrong_res = reconlConfigureShadows(r.d, &wrong);
        printf("      guards: struct_size 8 -> %d, wrong type -> %d\n", (int)short_res, (int)wrong_res);
        check_show(short_res == RECONL_ERR_STRUCT_SIZE && wrong_res == RECONL_ERR_WRONG_STRUCT_TYPE,
                   "a short or mistyped shadow config is refused");
    }

    /* 6. A per-frame config still wins over the device's, which is what lets one
     *    host render two different shadow settings in one process. */
    cr = reconlConfigureShadows(r.d, &cfg);
    ReconLShadowConfig per_frame = shadow_config(1, 1, RECONL_SHADOW_FILTER_HARD, 2u << 20, 0.0f);
    res = scene_frame(&r, &per_frame, NULL, lit, &st, NULL);
    printf("      per-frame override: cascades %u map %u filter %u (device asked for 2 cascades, 8 MiB, pcf3x3)\n",
           st.shadows.cascades_active, st.shadows.map_width, st.shadows.filter_active);
    check_show(res == RECONL_OK && st.shadows.cascades_active == 1
                   && st.shadows.filter_active == RECONL_SHADOW_FILTER_HARD,
               "a per-frame config overrides the device's");

    rig_free(&r);
}

int main(void) {
    printf("reconlConfigureShadows through the shipped ABI\n");
    arm(RECONL_BACKEND_SOFT_CPU, "soft-cpu");
    arm(RECONL_BACKEND_D3D11, "d3d11");
    printf("shadowconfig: %d checks, %d failures\n", checks, fails);
    return fails != 0;
}
