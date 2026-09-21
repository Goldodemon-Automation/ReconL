/* The GPU offload through the shipped DLL, as a host drives it (docs/offload.md).
 *
 * Four arms:
 *   1. a host that left RECONL_ALLOW_DOWNGRADE_TIER out is never moved, however
 *      badly its frames miss the target - the existing opt-out;
 *   2. a host that allowed it has its next frame rendered by the reference tier
 *      once the target is missed, and the *calibration* frame's measurement
 *      decides whether the device stays there or comes back. The numbers are
 *      printed, because which way that goes is this machine's business, not the
 *      contract's;
 *   3. no miss, no offload: frames inside a generous target keep the hardware;
 *   4. a frame the driver refuses - a target past D3D11's 16384 texel limit -
 *      does NOT move the device, even though that failure reports
 *      RECONL_ERR_DEVICE_LOST. The trigger is the driver's own
 *      GetDeviceRemovedReason verdict, not the code: a device that offloaded on
 *      the code would abandon a healthy GPU over the host's own bad argument.
 *
 * The frame is a single empty render pass: the offload is a policy about tiers,
 * not about pixels, so nothing here depends on what is drawn.
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

#define W 64u
#define H 64u

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

typedef struct {
    ReconLDevice* d;
    ReconLSwapchain* s;
    ReconLCommandList* cl;
    unsigned char px[W * H * 4];
} Rig;

static int rig(Rig* r, ReconLBackendId backend, uint32_t allow, uint32_t target_ms, uint32_t after) {
    memset(r, 0, sizeof *r);
    ReconLDeviceDesc dd;
    memset(&dd, 0, sizeof dd);
    SETBASE(&dd, RECONL_STRUCT_DEVICE_DESC);
    dd.backend_hint = backend;
    dd.tier_hint = backend == RECONL_BACKEND_D3D11 ? RECONL_TIER_T1_GPU_SHARED : RECONL_TIER_T2_CPU_RAM;
    dd.allow_downgrade = allow;
    dd.target_frame_ms = target_ms;
    dd.downgrade_after_frames = after;
    dd.seed = 7;
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

static int stats_of(Rig* r, ReconLStats* st) {
    memset(st, 0, sizeof *st);
    SETBASE(st, RECONL_STRUCT_STATS);
    return reconlGetStats(r->d, st) == RECONL_OK;
}

/* Begin + one empty pass + submit. `w`/`h` set the frame's targets, which is
 * what the driver refuses in arm 4. */
static ReconLResult frame_and_submit(Rig* r, uint32_t w, uint32_t h) {
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
    fd.width = w; fd.height = h; fd.seed = 1; fd.lights = &ll;
    ReconLResult b = reconlBeginFrame(r->d, &fd);
    if (b != RECONL_OK) return b;

    ReconLRenderPassDesc rp;
    memset(&rp, 0, sizeof rp);
    SETBASE(&rp, RECONL_STRUCT_RENDER_PASS_DESC);
    rp.viewport_width = w; rp.viewport_height = h;
    rp.load_color = 1; rp.load_depth = 1;
    rp.clear_color[2] = 1.0f; rp.clear_color[3] = 1.0f;
    reconlCmdReset(r->cl);
    reconlCmdBeginRenderPass(r->cl, &rp);
    reconlCmdEndRenderPass(r->cl);
    return reconlSubmit(r->d, r->cl, NULL);
}

/* A whole frame: begin, submit, present. Returns the result of the last call. */
static ReconLResult full_frame(Rig* r) {
    ReconLResult s = frame_and_submit(r, W, H);
    if (s != RECONL_OK) return s;
    ReconLPresentDesc p;
    memset(&p, 0, sizeof p);
    SETBASE(&p, RECONL_STRUCT_PRESENT_DESC);
    p.out_pixels = r->px; p.out_pixels_size = sizeof r->px;
    p.out_row_pitch = W * 4; p.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    return reconlPresent(r->d, r->s, &p);
}

static const char* backend_name(uint32_t b) {
    switch (b) {
        case RECONL_BACKEND_D3D11: return "d3d11";
        case RECONL_BACKEND_SOFT_CPU: return "soft-cpu";
        case RECONL_BACKEND_NULL: return "null";
        default: return "?";
    }
}

static void show_stats(const ReconLStats* st) {
    printf("      backend %s tier %u reason %u  presented %u dropped %u failures %u safe-path %u  "
           "downgrades %u  last frame %llu us\n",
           backend_name(st->backend), st->tier, st->tier_reason,
           st->frames_presented, st->frames_dropped, st->failures, st->safe_path_events,
           st->downgrade_count, (unsigned long long)(st->frame.total_ns / 1000));
    for (uint32_t i = 0; i < st->downgrade_count && i < st->downgrade_capacity; ++i) {
        const ReconLDowngrade* e = &st->downgrades[i];
        printf("      downgrade[%u] tier %u -> %u, reason %u, frame %llu: %s\n",
               i, e->from, e->to, e->reason, (unsigned long long)e->frame_index, e->detail);
    }
}

/* 1. The host's opt-out: no tier changes allowed, target missed anyway. */
static void arm_opted_out(void) {
    printf("  allow_downgrade = 0 (the opt-out), target 1 ms, 1 frame over target\n");
    Rig r;
    if (rig(&r, RECONL_BACKEND_D3D11, RECONL_ALLOW_DOWNGRADE_NONE, 1, 1)) {
        printf("    (d3d11 unavailable)\n");
        return;
    }
    ReconLResult p = full_frame(&r);
    check(p == RECONL_OK, "the frame is presented");
    ReconLStats st;
    check(stats_of(&r, &st), "stats readable");
    show_stats(&st);
    check(st.backend == RECONL_BACKEND_D3D11, "the device never left the hardware");
    check(st.safe_path_events == 0, "no safe path was taken");
    check(st.tier_reason != RECONL_TIER_REASON_RECOVERY, "no return trip: nothing left");
    rig_free(&r);
}

/* What a tier's frames cost on this machine, measured with the ladder off.
 *
 * The target this arm used to hard-code (1 ms) is a number that happens to sit
 * under a hardware frame here and above one on a faster host, which turns the
 * arm's verdict into a property of the machine. It is derived from these
 * numbers instead, and from the *median* rather than the peak: one slow frame on
 * a busy machine would otherwise push the target up a whole millisecond and the
 * arm would stop firing, which is the same class of accident as hard-coding it.
 * The first frame is measured and reported but not derived from - a device's
 * first frame everywhere in this codebase is its cold one. */
static int pilot(ReconLBackendId backend, uint32_t frames, uint64_t* cold_out, uint64_t* min_out,
                 uint64_t* median_out, uint64_t* max_out) {
    Rig m;
    uint64_t warm[32];
    uint32_t n = 0;
    *cold_out = 0;
    *min_out = 0;
    *median_out = 0;
    *max_out = 0;
    if (frames > 32) frames = 32;
    if (rig(&m, backend, RECONL_ALLOW_DOWNGRADE_NONE, 0, 1)) {
        rig_free(&m);
        return 1;
    }
    uint32_t counted = 0;
    for (uint32_t i = 0; i < frames; ++i) {
        if (full_frame(&m) != RECONL_OK) continue;
        ReconLStats st;
        if (!stats_of(&m, &st) || st.frame.total_ns == 0) continue;
        if (i == 0) {
            *cold_out = st.frame.total_ns;
        } else {
            warm[n++] = st.frame.total_ns;
            if (*min_out == 0 || st.frame.total_ns < *min_out) *min_out = st.frame.total_ns;
            if (st.frame.total_ns > *max_out) *max_out = st.frame.total_ns;
        }
        counted++;
    }
    rig_free(&m);
    for (uint32_t a = 1; a < n; ++a) {
        uint64_t v = warm[a];
        uint32_t b = a;
        while (b > 0 && warm[b - 1] > v) { warm[b] = warm[b - 1]; --b; }
        warm[b] = v;
    }
    if (n) *median_out = warm[n / 2];
    return counted ? 0 : 1;
}

/* The derived target: the smallest whole millisecond the reference tier is
 * *demonstrably* able to meet, which is what its cheapest measured frame says.
 *
 * The settle window's arm is a run of reference-tier frames inside the target,
 * so a target that tier has actually produced a frame under is the difference
 * between an arm that is a fact and an arm that waits for luck - which is the
 * knife edge this arm was on, and the reason its verdict used to depend on how
 * many frames the run happened to observe. The *fastest* frame rather than the
 * typical one, because the typical one moves with whatever else the machine is
 * doing: under load the median can rise a whole millisecond and then the
 * hardware's first frame - inflated by the same load, but from a much lower
 * base - stops being over the target at all, and the arm has nothing to
 * exercise. A millisecond is the granularity the ABI offers, so "the smallest
 * whole millisecond the tier's own cheapest frame is inside" is as tight as the
 * derivation can be, and the value is a measurement rather than a taste. */
static uint32_t derive_target_ms(uint64_t cpu_warm_min) {
    uint64_t ms = (cpu_warm_min + 999999u) / 1000000u;
    return ms ? (uint32_t)ms : 1;
}

/* How many frames the ladder may take to quit, and how many quiet frames prove
 * it has. The policy's state bounds the changes (one return per plan, one
 * remembered offload after it), so three quiet frames after the last change is
 * a verdict and not a guess - but the run is driven to quiescence rather than
 * stopped at a fixed window, which is what the old arm got wrong: a window of
 * six frames cut the round trip in half and read the cut as a failure. */
#define ARM_MAX_FRAMES 24u
#define ARM_QUIET_FRAMES 3u

/* 2. The measured overload: offload, calibrate, and act on the measurement.
 *
 * First in `main`, and that order is load-bearing: the trigger fires on a
 * hardware frame over the target, and the frame that is reliably over it here is
 * the *first* frame of a D3D11 device in this process. A device created after
 * other arms have run starts warm, inside the derived target, and the arm would
 * then have nothing to measure. The reference tier's pilot does not warm the
 * hardware path. */
static void arm_overload(void) {
    uint64_t cpu_cold = 0, cpu_min = 0, cpu_median = 0, cpu_max = 0;
    int cpu_ok = pilot(RECONL_BACKEND_SOFT_CPU, 8, &cpu_cold, &cpu_min, &cpu_median, &cpu_max) == 0;
    if (!cpu_ok) {
        printf("    (the reference tier could not be piloted)\n");
        return;
    }
    uint32_t target_ms = derive_target_ms(cpu_min);
    printf("  allow_downgrade = TIER, target %u ms (derived here: the reference tier's frames "
           "cost %llu ns cold, %llu ns at their cheapest, %llu ns median and %llu ns at their "
           "slowest, so %u ms is the smallest target the tier is demonstrably able to meet), "
           "1 frame over target\n",
           target_ms, (unsigned long long)cpu_cold, (unsigned long long)cpu_min,
           (unsigned long long)cpu_median, (unsigned long long)cpu_max, target_ms);

    Rig r;
    if (rig(&r, RECONL_BACKEND_D3D11, RECONL_ALLOW_DOWNGRADE_TIER, target_ms, 1)) {
        printf("    (d3d11 unavailable)\n");
        return;
    }
    ReconLStats st;

    /* The first frame is the hardware's, and its coldest: over target, so the
     * offload happens at its end and the next frame is the reference tier's.
     * Whether it is over target on *this* host is observed, not assumed. */
    check(full_frame(&r) == RECONL_OK, "the frame that missed the target is presented");
    check(stats_of(&r, &st), "stats readable");
    if (st.safe_path_events == 0) {
        printf("    (this host's first hardware frame cost %llu ns, inside the derived %u ms "
               "target: there is no miss here for the trigger to act on)\n",
               (unsigned long long)st.frame.total_ns, target_ms);
        rig_free(&r);
        return;
    }
    check(st.backend == RECONL_BACKEND_SOFT_CPU, "the next frame is the reference tier's");
    check(st.tier == RECONL_TIER_T2_CPU_RAM, "reported at the reference tier");
    check(st.safe_path_events == 1, "one backend change: the offload");
    check(st.downgrades[0].reason == RECONL_TIER_REASON_FRAME_TIME_OVER_TARGET,
          "logged with the reason that caused it");
    check(st.frames_presented == 1 && st.frames_dropped == 0 && st.failures == 0,
          "the host saw a frame, not an error");

    /* The second frame is the calibration, rendered by the reference tier, and
     * the decision stands on its own measurement. */
    check(full_frame(&r) == RECONL_OK, "the calibration frame is presented");
    check(stats_of(&r, &st), "stats readable");
    check(st.frames_presented == 2 && st.frames_dropped == 0 && st.failures == 0,
          "two frames presented, nothing dropped");
    int returned = st.backend == RECONL_BACKEND_D3D11;
    if (returned) {
        check(st.tier_reason == RECONL_TIER_REASON_RECOVERY, "the return records itself");
        check(st.safe_path_events == 2, "the return is a second backend change");
        unsigned i = 0, found = 0;
        for (; i < st.downgrade_count && i < st.downgrade_capacity; ++i) {
            if (st.downgrades[i].reason == RECONL_TIER_REASON_RECOVERY) found = 1;
        }
        check(found, "the tier log carries the return");
        printf("      (this host: the reference tier measured slower, so the device came back)\n");
    } else {
        check(st.backend == RECONL_BACKEND_SOFT_CPU, "the reference tier measured faster and kept it");
        printf("      (this host: the reference tier measured faster, so the device stayed)\n");
    }

    /* Drive the ladder until it quits, and print every decision it took. What
     * the rule bounds is the *return*, once per plan: one round trip and no more
     * - the offload, the return the settle window bought, and, if the hardware
     * missed again after it, the offload that remembers the calibration instead
     * of paying for it twice. That is three backend changes; a fourth, or a
     * second return, is the thrash the rule exists to prevent. Where those
     * changes land depends on how quickly the reference tier produces a frame
     * inside the target, so the arm runs until three quiet frames say the ladder
     * has quit instead of sampling a window and calling the sample the answer. */
    /* `rendered` is the index of the last frame rendered - what the policy
     * itself calls the frame - so the count is one more than it. */
    uint32_t changes = st.safe_path_events, quiet = 0, rendered = 2, frames = 2,
             returns = returned ? 1 : 0, offloads = 1;
    int settled = 0;
    for (; rendered < ARM_MAX_FRAMES; ++rendered) {
        if (full_frame(&r) != RECONL_OK) {
            check(0, "every frame of the ladder run is presented");
            break;
        }
        if (!stats_of(&r, &st)) {
            check(0, "stats readable");
            break;
        }
        frames = rendered + 1;
        if (st.safe_path_events != changes) {
            changes = st.safe_path_events;
            quiet = 0;
            if (st.tier_reason == RECONL_TIER_REASON_RECOVERY) {
                returns++;
            } else {
                offloads++;
            }
            printf("      change %u: after frame %u, frame %u renders on %s - `%s`\n",
                   changes, rendered, rendered + 1, backend_name(st.backend), st.tier_reason_text);
        } else if (++quiet >= ARM_QUIET_FRAMES) {
            settled = 1;
            break;
        }
    }
    show_stats(&st);
    printf("      %u frames driven, %u backend changes (%u offloads, %u returns), %s\n",
           frames, changes, offloads, returns,
           settled ? "settled" : "still changing at the cap");
    check(st.frames_presented == frames && st.frames_dropped == 0 && st.failures == 0,
          "every frame was presented, nothing dropped");
    check(settled, "the ladder quit: three consecutive frames changed no backend");
    check(returns <= 1, "the return trip was attempted at most once: one per plan");
    check(changes <= 3,
          "one round trip at most: offload, return, and the offload of a miss after it");
    /* The measurement is remembered: the calibration is paid once per plan, so a
     * second offload at the same plan must not carry the calibration's wording. */
    unsigned calibrations = 0, remembered = 0, k = 0;
    for (; k < st.downgrade_count && k < st.downgrade_capacity; ++k) {
        const ReconLDowngrade* e = &st.downgrades[k];
        if (e->reason != RECONL_TIER_REASON_FRAME_TIME_OVER_TARGET) continue;
        if (strstr(e->detail, "calibrating the reference tier")) calibrations++;
        if (strstr(e->detail, "already measured")) remembered++;
    }
    check(calibrations <= 1, "the calibration was paid once, not once per offload");
    check(offloads <= 1 || remembered >= 1,
          "a second offload at this plan reused the measurement instead of recalibrating");
    rig_free(&r);
}

/* 3. No miss, no offload. */
static void arm_within_target(void) {
    printf("  allow_downgrade = TIER, target 5000 ms (never missed)\n");
    Rig r;
    if (rig(&r, RECONL_BACKEND_D3D11, RECONL_ALLOW_DOWNGRADE_TIER, 5000, 1)) {
        printf("    (d3d11 unavailable)\n");
        return;
    }
    int i;
    for (i = 0; i < 3; ++i) check(full_frame(&r) == RECONL_OK, "the frame is presented");
    ReconLStats st;
    check(stats_of(&r, &st), "stats readable");
    show_stats(&st);
    check(st.backend == RECONL_BACKEND_D3D11, "the hardware is kept");
    check(st.safe_path_events == 0, "no safe path was taken");
    rig_free(&r);
}

/* 4. A frame the driver refuses is not a lost device. */
static void arm_refused_frame(void) {
    printf("  allow_downgrade = TIER, target 1 ms, a 32768x1 frame the driver refuses\n");
    Rig r;
    if (rig(&r, RECONL_BACKEND_D3D11, RECONL_ALLOW_DOWNGRADE_TIER, 1, 1)) {
        printf("    (d3d11 unavailable)\n");
        return;
    }
    ReconLResult s = frame_and_submit(&r, 32768, 1);
    printf("      submit -> %d\n", (int)s);
    check(s == RECONL_ERR_INVALID_ARGUMENT,
          "the driver's E_INVALIDARG surfaces as the argument code, not DEVICE_LOST");
    ReconLStats st;
    check(stats_of(&r, &st), "stats readable");
    show_stats(&st);
    check(st.backend == RECONL_BACKEND_D3D11, "the device is still on the hardware");
    check(st.safe_path_events == 0, "a refused frame took no safe path");
    check(st.frames_dropped == 1, "the frame was dropped, as before the offload existed");
    rig_free(&r);
}

int main(void) {
    printf("reconl offload probe (DLL %d)\n", RECONL_ABI_VERSION);
    arm_overload();
    arm_opted_out();
    arm_within_target();
    arm_refused_frame();
    printf("%d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
