//! Animation: easing curves, tweens, and a spring - the motion vocabulary the
//! widgets draw on.
//!
//! Everything here is *time-in, value-out*: the host supplies `dt`, nothing
//! reads a wall clock. That keeps the frontend on the same determinism rule
//! the renderer keeps (`docs/determinism.md`): the same sequence of inputs
//! produces the same frames, which is what lets the showcase be diffed.
//!
//! Three ways to move a value:
//! * [`Tween`]   - explicit from/to/duration with an easing curve. UI that
//!                  animates *between two known states* (a panel opening,
//!                  a colour changing) uses this.
//! * [`Spring`]  - a critically-configurable damped spring toward a target.
//!                  UI that follows an *unbounded input* (a dragged slider, a
//!                  scroll offset) uses this, because a spring never overshoot
//!                  budget questions - it just settles.
//! * [`ease`]    - a bare `t in [0,1] -> eased t`, for the one-off cases.

const std = @import("std");

/// The easing curves, all pure functions on the unit interval with
/// `f(0) = 0` and `f(1) = 1` (asserted by the tests below - a curve that
/// fails that either never lands on its target or lands on it late, and both
/// read as a bug in every widget that uses it).
pub const Easing = enum {
    linear,
    quad_in,
    quad_out,
    quad_in_out,
    cubic_in,
    cubic_out,
    cubic_in_out,
    expo_out,
    back_out,
    elastic_out,

    pub fn apply(e: Easing, t: f32) f32 {
        const x = std.math.clamp(t, 0.0, 1.0);
        return switch (e) {
            .linear => x,
            .quad_in => x * x,
            .quad_out => 1.0 - (1.0 - x) * (1.0 - x),
            .quad_in_out => if (x < 0.5) 2.0 * x * x else 1.0 - std.math.pow(f32, -2.0 * x + 2.0, 2.0) / 2.0,
            .cubic_in => x * x * x,
            .cubic_out => 1.0 - std.math.pow(f32, 1.0 - x, 3.0),
            .cubic_in_out => blk: {
                if (x < 0.5) break :blk 4.0 * x * x * x;
                break :blk 1.0 - std.math.pow(f32, -2.0 * x + 2.0, 3.0) / 2.0;
            },
            .expo_out => if (x >= 1.0) 1.0 else 1.0 - std.math.pow(f32, 2.0, -10.0 * x),
            // Slight overshoot - the "arrives with personality" curve for
            // panels and toasts. Never used where overshoot could clip a
            // hit target, because the *value* exceeds 1 mid-flight.
            .back_out => blk: {
                const c1 = 1.70158;
                const c3 = c1 + 1.0;
                const p = x - 1.0;
                break :blk 1.0 + c3 * p * p * p + c1 * p * p;
            },
            .elastic_out => blk: {
                if (x <= 0.0) break :blk 0.0;
                if (x >= 1.0) break :blk 1.0;
                const c4 = (2.0 * std.math.pi) / 3.0;
                break :blk std.math.pow(f32, 2.0, -10.0 * x) * @sin((x * 10.0 - 0.75) * c4) + 1.0;
            },
        };
    }
};

/// A finite tween: drive [`Tween.update`] each frame, read [`Tween.value`].
/// When `t` reaches 1 the value stays pinned at `to` - a finished tween is
/// not a per-frame allocation or a source of drift, it is just the number.
pub const Tween = struct {
    from: f32 = 0.0,
    to: f32 = 0.0,
    elapsed_ms: f32 = 0.0,
    duration_ms: f32 = 200.0,
    easing: Easing = .cubic_out,

    pub fn init(from: f32, to: f32, duration_ms: f32, easing: Easing) Tween {
        return .{ .from = from, .to = to, .duration_ms = duration_ms, .easing = easing };
    }

    /// Retarget: starts a new tween from *where we are now*, so interrupting
    /// an in-flight animation never jumps. `reset=true` replays from `from`
    /// (used when a widget re-enters a state rather than transitioning).
    pub fn retarget(tw: *Tween, to: f32, duration_ms: f32, easing: Easing, reset: bool) void {
        if (reset) {
            tw.from = tw.from;
        } else {
            tw.from = tw.value();
        }
        tw.to = to;
        tw.elapsed_ms = 0.0;
        tw.duration_ms = duration_ms;
        tw.easing = easing;
    }

    pub fn update(tw: *Tween, dt_ms: f32) void {
        if (tw.duration_ms <= 0.0) {
            tw.elapsed_ms = tw.duration_ms;
            return;
        }
        tw.elapsed_ms = @min(tw.elapsed_ms + dt_ms, tw.duration_ms);
    }

    pub fn t(tw: Tween) f32 {
        if (tw.duration_ms <= 0.0) return 1.0;
        return std.math.clamp(tw.elapsed_ms / tw.duration_ms, 0.0, 1.0);
    }

    pub fn value(tw: Tween) f32 {
        const k = tw.easing.apply(tw.t());
        return tw.from + (tw.to - tw.from) * k;
    }

    pub fn done(tw: Tween) bool {
        return tw.elapsed_ms >= tw.duration_ms and tw.duration_ms > 0.0;
    }
};

/// A damped spring chasing a target. Semi-implicit Euler at `dt`, which is
/// stable for the stiffness/damping pairs the theme ships and is a pure
/// function of (state, target, dt) - no clock, no accumulation order
/// dependence beyond the frame itself.
pub const Spring = struct {
    value: f32 = 0.0,
    velocity: f32 = 0.0,
    target: f32 = 0.0,
    stiffness: f32 = 320.0,
    damping: f32 = 26.0,

    pub fn init(value: f32, stiffness: f32, damping: f32) Spring {
        return .{ .value = value, .target = value, .stiffness = stiffness, .damping = damping };
    }

    pub fn step(sp: *Spring, dt_s: f32) void {
        // Sub-step if the host hands us a long frame: one explicit Euler step
        // at dt > ~8ms with stiffness 320 goes visibly wrong, and a UI that
        // stutters once stutters in every recording of it.
        var remaining = dt_s;
        const max_step = 1.0 / 120.0;
        while (remaining > 0.0) {
            const h = @min(remaining, max_step);
            const accel = sp.stiffness * (sp.target - sp.value) - sp.damping * sp.velocity;
            sp.velocity += accel * h;
            sp.value += sp.velocity * h;
            remaining -= h;
        }
    }

    pub fn settled(sp: Spring, epsilon: f32) bool {
        return @abs(sp.target - sp.value) < epsilon and @abs(sp.velocity) < epsilon;
    }
};

test "every easing hits its endpoints" {
    inline for (std.meta.fields(Easing)) |f| {
        const e: Easing = @enumFromInt(f.value);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), e.apply(0.0), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), e.apply(1.0), 1e-5);
    }
}

test "the in-out curves are monotonic and symmetric" {
    const e = Easing.cubic_in_out;
    var prev = e.apply(0.0);
    var i: f32 = 1.0;
    while (i <= 20.0) : (i += 1.0) {
        const v = e.apply(i / 20.0);
        try std.testing.expect(v >= prev - 1e-6);
        prev = v;
    }
    try std.testing.expectApproxEqAbs(e.apply(0.25), 1.0 - e.apply(0.75), 1e-5);
}

test "tween lands exactly on its target and stays" {
    var tw = Tween.init(10.0, 40.0, 100.0, .quad_out);
    var i: u32 = 0;
    while (i < 30) : (i += 1) tw.update(16.0);
    try std.testing.expect(tw.done());
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), tw.value(), 1e-6);
    // And it does not drift afterwards.
    tw.update(16.0);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), tw.value(), 1e-6);
}

test "retargeting an in-flight tween starts from the current value" {
    var tw = Tween.init(0.0, 100.0, 1000.0, .linear);
    tw.update(500.0);
    const mid = tw.value();
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), mid, 1e-4);
    tw.retarget(0.0, 1000.0, .linear, false);
    try std.testing.expectApproxEqAbs(mid, tw.from, 1e-6);
}

test "spring settles at its target without ringing forever" {
    var sp = Spring.init(0.0, 320.0, 26.0);
    sp.target = 100.0;
    var i: u32 = 0;
    while (i < 240) : (i += 1) sp.step(1.0 / 60.0);
    try std.testing.expect(sp.settled(0.05));
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), sp.value, 0.05);
}

test "a long frame is sub-stepped rather than exploded" {
    var sp = Spring.init(0.0, 320.0, 26.0);
    sp.target = 1.0;
    sp.step(0.5); // a half-second hitch
    try std.testing.expect(@abs(sp.value) < 10.0);
}
