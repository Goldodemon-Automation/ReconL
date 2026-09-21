//! The offload ladder: tiers T0..T4, what each one means, and the rules that
//! pick one.
//!
//! Selection is explicit and logged. A tier is never inferred quietly at the end
//! of a frame: [`resolve_tier`] returns the tier *and* the reason, and the device
//! (the only owner of a tier change and of the log that records one) appends to
//! [`crate::stats::DowngradeLog`]. [`FrameLadder`] is that owner's arithmetic -
//! the run of frames that decides a step - so both layers that act on a tier
//! answer "was this frame over target?" from one counter rather than one each.

use crate::text::Text;

pub const TIER_COUNT: usize = 5;

/// Mirrors `ReconLBackendId`.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    None = 0,
    SoftCpu = 1,
    Null = 2,
    D3d11 = 3,
    D3d12 = 4,
    Vulkan = 5,
    Gl = 6,
    Metal = 7,
    WebGpu = 8,
    WasmWebGl2 = 9,
}

impl Backend {
    pub fn from_u32(v: u32) -> Backend {
        match v {
            1 => Backend::SoftCpu,
            2 => Backend::Null,
            3 => Backend::D3d11,
            4 => Backend::D3d12,
            5 => Backend::Vulkan,
            6 => Backend::Gl,
            7 => Backend::Metal,
            8 => Backend::WebGpu,
            9 => Backend::WasmWebGl2,
            _ => Backend::None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Backend::None => "none",
            Backend::SoftCpu => "soft-cpu",
            Backend::Null => "null",
            Backend::D3d11 => "d3d11",
            Backend::D3d12 => "d3d12",
            Backend::Vulkan => "vulkan",
            Backend::Gl => "gl",
            Backend::Metal => "metal",
            Backend::WebGpu => "webgpu",
            Backend::WasmWebGl2 => "wasm-webgl2",
        }
    }
}

/// Mirrors `ReconLTier`.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Tier {
    GpuDiscrete = 0,
    GpuShared = 1,
    CpuRam = 2,
    CpuThrifty = 3,
    OutOfCore = 4,
}

impl Tier {
    pub fn from_u32(v: u32) -> Tier {
        match v {
            0 => Tier::GpuDiscrete,
            1 => Tier::GpuShared,
            2 => Tier::CpuRam,
            3 => Tier::CpuThrifty,
            _ => Tier::OutOfCore,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    pub const fn name(self) -> &'static str {
        match self {
            Tier::GpuDiscrete => "T0/gpu-discrete",
            Tier::GpuShared => "T1/gpu-shared",
            Tier::CpuRam => "T2/cpu-ram",
            Tier::CpuThrifty => "T3/cpu-thrifty",
            Tier::OutOfCore => "T4/out-of-core",
        }
    }

    /// The tier one step down, saturating at T4.
    pub fn step_down(self) -> Tier {
        match self {
            Tier::GpuDiscrete => Tier::GpuShared,
            Tier::GpuShared => Tier::CpuRam,
            Tier::CpuRam => Tier::CpuThrifty,
            Tier::CpuThrifty | Tier::OutOfCore => Tier::OutOfCore,
        }
    }
}

/// Mirrors `ReconLTierReason`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TierReason {
    StartupProbe = 0,
    HostRequest = 1,
    AllocationOverBudget = 2,
    FrameTimeOverTarget = 3,
    DeviceRemoved = 4,
    DeviceLost = 5,
    MemoryPressure = 6,
    DiskCacheFull = 7,
    Build = 8,
    NoGpuApi = 9,
    /// The hardware was rebuilt after the settle window and measured inside the
    /// frame-time target: the device came back up the ladder. Distinct from
    /// `StartupProbe` because a host auditing its tier log needs to tell "this is
    /// how the device started" from "this is how it recovered".
    Recovery = 10,
}

impl TierReason {
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    pub const fn text(self) -> &'static str {
        match self {
            TierReason::StartupProbe => "startup probe chose the highest usable tier",
            TierReason::HostRequest => "host requested this tier",
            TierReason::AllocationOverBudget => "allocation attempt exceeded the budget",
            TierReason::FrameTimeOverTarget => "frame time over target for N consecutive frames",
            TierReason::DeviceRemoved => "device removed (DXGI_ERROR_DEVICE_REMOVED)",
            TierReason::DeviceLost => "device lost (VK_ERROR_DEVICE_LOST)",
            TierReason::MemoryPressure => "OS memory pressure",
            TierReason::DiskCacheFull => "disk cache full",
            TierReason::Build => "compiled without this backend",
            TierReason::NoGpuApi => "no usable GPU API on this host",
            TierReason::Recovery => "the hardware was rebuilt after the settle window",
        }
    }
}

/// The frame-time ladder's arithmetic: a target, a threshold, and the run of
/// consecutive frames over the target that decides a step down.
///
/// Two layers act on a device's tier (`docs/offload.md`, "Two layers, one
/// number"): the *relabel*, which keeps the backend and lowers its quality tier,
/// and the *offload*, which changes the backend. They answer the same question
/// from the same number, so they count the same run: this type is the one
/// counter, and the device holds the one instance of it. What to do when the run
/// reaches the threshold is the device's policy (`ffi/src/offload.rs`); this
/// type only keeps the count and says when the ladder acts.
///
/// The run is the *device's*, over the costs it composed in frame order: it is
/// not restarted when the renderer changes, because the frames a rebuilt backend
/// presents were paid for by the device that handed them over. The one thing
/// that ends a run is a frame inside the target.
pub struct FrameLadder {
    target_ms: u32,
    threshold: u32,
    over: u32,
    last_over_ns: u64,
}

impl FrameLadder {
    pub const fn new(target_ms: u32, threshold: u32) -> Self {
        Self { target_ms, threshold, over: 0, last_over_ns: 0 }
    }

    /// The target a frame is judged against, in milliseconds. Zero means the
    /// ladder is off: a host that set no target has said "do not decide for me".
    pub const fn target_ms(&self) -> u32 {
        self.target_ms
    }

    /// `desc.downgrade_after_frames`: how many consecutive over-target frames
    /// make the ladder act. Zero disables it as well.
    pub const fn threshold(&self) -> u32 {
        self.threshold
    }

    pub const fn target_ns(&self) -> u64 {
        self.target_ms as u64 * 1_000_000
    }

    /// Consecutive frames over the target so far - the number a relabel's own
    /// detail reports.
    pub const fn over_target(&self) -> u32 {
        self.over
    }

    /// The cost of the most recent frame the run counted. The run is a run *of a
    /// number*, so the record of it names that number: a relabel's detail reports
    /// it, which is how a host reads the frame cost the ladder acted on instead of
    /// taking the device's word for it. Zero when no frame is counted.
    pub const fn last_over_ns(&self) -> u64 {
        self.last_over_ns
    }

    /// Records one frame by its complete cost - the number the host waited for,
    /// readback and all, which only the boundary that made the host wait can
    /// compose. Returns true on the frame the run reaches the threshold: the one
    /// the offload layer acts on. A frame inside the target, or a ladder that is
    /// off, ends the run.
    pub fn observe(&mut self, cost_ns: u64) -> bool {
        if self.target_ms == 0 || cost_ns <= self.target_ns() {
            self.over = 0;
            self.last_over_ns = 0;
            return false;
        }
        self.over = self.over.saturating_add(1);
        self.last_over_ns = cost_ns;
        self.acting()
    }

    /// Whether the run already stands at the threshold - without counting `cost`.
    ///
    /// This is the question the relabel layer answers, and it is asked before the
    /// frame's own observation: the device's own ladder acts on the frame *after*
    /// the cost that armed it (`docs/offload.md`, "Two layers, one number"), so a
    /// host sees a tier change on the frame that follows the over-target frame,
    /// at the render the backend it names then performs.
    pub const fn acting(&self) -> bool {
        self.target_ms > 0 && self.threshold > 0 && self.over >= self.threshold
    }
}

/// Mirrors `ReconLShadowFilter`.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShadowFilter {
    Hard = 0,
    Pcf3x3 = 1,
    Pcf5x5 = 2,
    PcssLite = 3,
}

impl Default for ShadowFilter {
    fn default() -> Self {
        ShadowFilter::Pcf3x3
    }
}

impl ShadowFilter {
    pub fn from_u32(v: u32) -> ShadowFilter {
        match v {
            0 => ShadowFilter::Hard,
            1 => ShadowFilter::Pcf3x3,
            2 => ShadowFilter::Pcf5x5,
            _ => ShadowFilter::PcssLite,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            ShadowFilter::Hard => "hard",
            ShadowFilter::Pcf3x3 => "pcf3x3",
            ShadowFilter::Pcf5x5 => "pcf5x5",
            ShadowFilter::PcssLite => "pcss-lite",
        }
    }

    pub fn taps(self) -> u32 {
        match self {
            ShadowFilter::Hard => 1,
            ShadowFilter::Pcf3x3 => 9,
            ShadowFilter::Pcf5x5 => 25,
            ShadowFilter::PcssLite => 25,
        }
    }
}

/// Mirrors the `ReconLCaps` bitmask.
pub mod caps {
    pub const TEXTURES: u32 = 1 << 0;
    pub const MIPMAPS: u32 = 1 << 1;
    pub const SHADOWS: u32 = 1 << 2;
    pub const PCF_5X5: u32 = 1 << 3;
    pub const PCSS_LITE: u32 = 1 << 4;
    pub const DISK_SPILL: u32 = 1 << 5;
    pub const MULTITHREAD: u32 = 1 << 6;
    pub const OUT_OF_CORE: u32 = 1 << 7;
    pub const CACHED_CASCADE: u32 = 1 << 8;
    pub const SIMD_SSE2: u32 = 1 << 9;
    pub const SIMD_AVX2: u32 = 1 << 10;
    pub const SIMD_NEON: u32 = 1 << 11;
    pub const SIMD_WASM128: u32 = 1 << 12;
    pub const COMPUTE: u32 = 1 << 13;
    pub const PRESENT_TO_MEMORY: u32 = 1 << 14;
}

/// What a tier is allowed to spend, and where its work happens.
#[derive(Clone, Copy)]
pub struct TierRules {
    pub tier: Tier,
    pub uses_gpu: bool,
    pub resident: &'static str,
    pub frame_path: &'static str,
    /// Cascades the tier may keep live.
    pub max_cascades: u32,
    /// Multiplier applied to the shadow texel budget asked for by the host.
    pub shadow_budget_scale: f32,
    pub shadow_filter_cap: ShadowFilter,
    /// Static cascade lives in RAM or on disk.
    pub static_cache_on_disk: bool,
    /// Multiplier applied to the requested resolution.
    pub resolution_scale: f32,
    /// How the tier treats its worker pool.
    pub worker_scale: f32,
    /// What this tier does when pressure arrives anyway.
    pub fallback: &'static str,
}

pub fn rules(tier: Tier) -> TierRules {
    match tier {
        Tier::GpuDiscrete => TierRules {
            tier,
            uses_gpu: true,
            resident: "full VRAM",
            frame_path: "hardware raster + compute",
            max_cascades: 4,
            shadow_budget_scale: 1.0,
            shadow_filter_cap: ShadowFilter::PcssLite,
            static_cache_on_disk: false,
            resolution_scale: 1.0,
            worker_scale: 0.0,
            fallback: "step to T1",
        },
        Tier::GpuShared => TierRules {
            tier,
            uses_gpu: true,
            resident: "VRAM + system RAM spill",
            frame_path: "hardware raster, streamed resources",
            max_cascades: 3,
            shadow_budget_scale: 0.5,
            shadow_filter_cap: ShadowFilter::Pcf5x5,
            static_cache_on_disk: false,
            resolution_scale: 1.0,
            worker_scale: 0.5,
            fallback: "drop to 2 cascades",
        },
        Tier::CpuRam => TierRules {
            tier,
            uses_gpu: false,
            resident: "RAM, multi-threaded",
            frame_path: "tiled software rasteriser, worker pool",
            max_cascades: 2,
            shadow_budget_scale: 0.25,
            shadow_filter_cap: ShadowFilter::Pcf3x3,
            static_cache_on_disk: false,
            resolution_scale: 1.0,
            worker_scale: 1.0,
            fallback: "1 cascade at lower resolution",
        },
        Tier::CpuThrifty => TierRules {
            tier,
            uses_gpu: false,
            resident: "RAM, capped",
            frame_path: "T2 with resolution scale, LOD, frozen caches",
            max_cascades: 1,
            shadow_budget_scale: 0.0625,
            shadow_filter_cap: ShadowFilter::Pcf3x3,
            static_cache_on_disk: false,
            resolution_scale: 0.5,
            worker_scale: 0.5,
            fallback: "freeze the cascade, refresh every N frames",
        },
        Tier::OutOfCore => TierRules {
            tier,
            uses_gpu: false,
            resident: "RAM + DISK",
            frame_path: "T2/T3 with tile + mip streaming from a disk cache",
            max_cascades: 1,
            shadow_budget_scale: 0.0625,
            shadow_filter_cap: ShadowFilter::Pcf3x3,
            static_cache_on_disk: true,
            resolution_scale: 0.5,
            worker_scale: 0.5,
            fallback: "shadowed-atlas region only, unshadowed outside",
        },
    }
}

/// Everything the resolver is allowed to know.
#[derive(Clone, Copy, Debug)]
pub struct TierInputs {
    /// A hardware backend could create a device on this host.
    pub gpu_usable: bool,
    /// The GPU shares system memory (iGPU) or the budget is tight.
    pub gpu_shared: bool,
    /// Free space where the spill arena would live.
    pub disk_bytes: u64,
    pub vram_cap: u64,
    pub ram_cap: u64,
    pub disk_cap: u64,
    pub allow_disk_spill: bool,
    /// Bytes the working set needs resident to render without streaming.
    pub working_set_bytes: u64,
    pub caps: u32,
}

#[derive(Clone, Copy)]
pub struct TierStep {
    pub tier: Tier,
    pub reason: TierReason,
}

impl TierStep {
    pub fn text(&self) -> Text<192> {
        let mut t = Text::new();
        t.push(self.tier.name());
        t.push(": ");
        t.push(self.reason.text());
        t
    }
}

/// Picks the highest usable tier at or below `requested`.
///
/// `requested` is a ceiling, never a floor: asking for T0 on a machine with no
/// GPU API yields T2 with `NoGpuApi`, because the API does not hard-fail on the
/// machine being weak.
pub fn resolve_tier(requested: Tier, inputs: &TierInputs) -> TierStep {
    // 1. What could this host actually run?
    let best_available = if inputs.gpu_usable && !inputs.gpu_shared {
        Tier::GpuDiscrete
    } else if inputs.gpu_usable {
        Tier::GpuShared
    } else if inputs.ram_cap > 0 && inputs.working_set_bytes > inputs.ram_cap {
        // Working set does not fit RAM: disk is the only place left for the part
        // that does not fit in the cap.
        if inputs.allow_disk_spill && inputs.disk_cap > 0 && inputs.disk_bytes > 0 {
            Tier::OutOfCore
        } else {
            Tier::CpuThrifty
        }
    } else if (inputs.caps & caps::MULTITHREAD) != 0 {
        Tier::CpuRam
    } else {
        Tier::CpuThrifty
    };

    // T0 is the ceiling with the *lowest* enum value: stepping down means the
    // value grows. `requested` is therefore a cap, and the tier actually used is
    // the numerically larger (lower-quality) of the two.
    let mut reason = if best_available > requested {
        if inputs.gpu_usable {
            TierReason::AllocationOverBudget
        } else {
            TierReason::NoGpuApi
        }
    } else if requested > best_available {
        TierReason::HostRequest
    } else {
        TierReason::StartupProbe
    };

    // 2. A disk-backed tier without an opt-in is not a tier we may use.
    let mut tier = requested.max(best_available);
    if tier == Tier::OutOfCore && !(inputs.allow_disk_spill && inputs.disk_cap > 0 && inputs.disk_bytes > 0) {
        tier = Tier::CpuThrifty;
        reason = TierReason::AllocationOverBudget;
    }
    if tier >= Tier::CpuRam && (inputs.caps & caps::MULTITHREAD) == 0 && tier != Tier::OutOfCore {
        // Single-threaded host: T3 semantics (capped, scaled) instead of T2.
        tier = Tier::CpuThrifty;
    }

    TierStep { tier, reason }
}

/// The shadow policy a tier is allowed to run, before per-frame pressure.
#[derive(Clone, Copy)]
pub struct ShadowPlan {
    pub cascades: u32,
    /// Texel budget after the tier's scale: what the cascade set may occupy.
    pub map_bytes: u64,
    /// The per-cascade map edge that budget buys - the one number a backend
    /// needs to allocate the set, decided here so both tiers and the host's
    /// probes answer alike instead of each deriving it.
    pub map_size: u32,
    /// What the set actually occupies, `map_size^2 * 4 * cascades`. Never above
    /// `map_bytes`: that is what `map_size` is chosen for.
    pub resident_bytes: u64,
    pub filter: ShadowFilter,
    pub filter_requested: ShadowFilter,
    pub disk_backed: bool,
    pub freeze_static: bool,
    pub clamp_event: Option<ShadowEvent>,
    pub note: &'static str,
}

/// Mirrors `ReconLShadowEvent`.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShadowEvent {
    None = 0,
    CascadeDropped = 1,
    MapBudgetClamped = 2,
    CacheHit = 3,
    CacheMiss = 4,
    CacheCorrupt = 5,
    FallbackUnshadowed = 6,
    FilterDowngraded = 7,
    FrozenCascade = 8,
}

impl ShadowEvent {
    pub fn from_u32(v: u32) -> ShadowEvent {
        match v {
            1 => ShadowEvent::CascadeDropped,
            2 => ShadowEvent::MapBudgetClamped,
            3 => ShadowEvent::CacheHit,
            4 => ShadowEvent::CacheMiss,
            5 => ShadowEvent::CacheCorrupt,
            6 => ShadowEvent::FallbackUnshadowed,
            7 => ShadowEvent::FilterDowngraded,
            8 => ShadowEvent::FrozenCascade,
            _ => ShadowEvent::None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            ShadowEvent::None => "none",
            ShadowEvent::CascadeDropped => "cascade-dropped",
            ShadowEvent::MapBudgetClamped => "map-budget-clamped",
            ShadowEvent::CacheHit => "cache-hit",
            ShadowEvent::CacheMiss => "cache-miss",
            ShadowEvent::CacheCorrupt => "cache-corrupt",
            ShadowEvent::FallbackUnshadowed => "fallback-unshadowed",
            ShadowEvent::FilterDowngraded => "filter-downgraded",
            ShadowEvent::FrozenCascade => "frozen-cascade",
        }
    }
}

/// Applies the §7 shadow ladder: the tier decides how good the shadows are, and
/// every clamp is reported rather than substituted silently.
pub fn shadow_plan(
    tier: Tier,
    requested_cascades: u32,
    requested_budget_bytes: u64,
    requested_filter: ShadowFilter,
    caps: u32,
) -> ShadowPlan {
    let r = rules(tier);
    let mut event: Option<ShadowEvent> = None;

    let cascades = requested_cascades.clamp(1, 4).min(r.max_cascades);
    if cascades < requested_cascades.clamp(1, 4) {
        event = Some(ShadowEvent::CascadeDropped);
    }

    let scaled = if r.shadow_budget_scale >= 1.0 {
        requested_budget_bytes
    } else {
        ((requested_budget_bytes as f64) * (r.shadow_budget_scale as f64)) as u64
    };
    let map_bytes = scaled.max(256 * 1024);
    if map_bytes < requested_budget_bytes {
        event = Some(ShadowEvent::MapBudgetClamped);
    }

    // Budget to map edge happens here and only here. It used to be a function
    // each backend carried a copy of, kept in step by a comment, which is one
    // policy with two owners; now the plan *is* the decision, and a backend
    // that allocates the set reads `map_size` and `resident_bytes` off it.
    let map_size = map_size_for(map_bytes, cascades);
    let resident_bytes = u64::from(map_size) * u64::from(map_size) * 4 * u64::from(cascades);

    // Filter: the tier caps the filter; pcss-lite is not available below T1 and
    // is never silently replaced by something the host did not ask for - the
    // downgrade is recorded.
    let mut filter = requested_filter;
    if matches!(filter, ShadowFilter::PcssLite) && (caps & caps::PCSS_LITE) == 0 {
        filter = ShadowFilter::Pcf5x5;
        event = Some(ShadowEvent::FilterDowngraded);
    }
    if (filter as u32) > (r.shadow_filter_cap as u32) {
        filter = r.shadow_filter_cap;
        event = Some(ShadowEvent::FilterDowngraded);
    }

    ShadowPlan {
        cascades,
        map_bytes,
        map_size,
        resident_bytes,
        filter,
        filter_requested: requested_filter,
        disk_backed: r.static_cache_on_disk,
        freeze_static: tier >= Tier::CpuThrifty,
        clamp_event: event,
        note: r.fallback,
    }
}

/// The largest power-of-two map side whose four-byte texels fit a share of
/// `map_bytes` while staying inside `cascades` maps.
///
/// Private on purpose: a caller with a budget wants the whole decision
/// ([`shadow_plan`]), not another way to turn bytes into an edge. The floor is
/// one 128-texel map, so a starved budget still yields a usable one; the
/// ceiling is 4096 per side, which no tier has the budget to reach anyway.
fn map_size_for(map_bytes: u64, cascades: u32) -> u32 {
    let per_map = map_bytes / u64::from(cascades.max(1)) / 4;
    let side = (per_map as f64).sqrt() as u64;
    let mut size: u64 = 128;
    while size * 2 <= side && size < 4096 {
        size *= 2;
    }
    size as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> TierInputs {
        TierInputs {
            gpu_usable: false,
            gpu_shared: false,
            disk_bytes: 1 << 40,
            vram_cap: 0,
            ram_cap: 0,
            disk_cap: 0,
            allow_disk_spill: false,
            working_set_bytes: 1 << 20,
            caps: caps::MULTITHREAD | caps::SHADOWS | caps::TEXTURES | caps::MIPMAPS,
        }
    }

    #[test]
    fn the_ladder_acts_on_the_frame_its_run_reaches_the_threshold() {
        let mut ladder = FrameLadder::new(10, 3);
        // Inside the target: no run to speak of.
        assert!(!ladder.observe(9_000_000));
        assert_eq!(ladder.over_target(), 0);
        // Three consecutive misses, and the third is the frame it acts on.
        assert!(!ladder.observe(11_000_000));
        assert!(!ladder.observe(12_000_000));
        assert!(ladder.observe(11_000_000));
        assert_eq!(ladder.over_target(), 3);
        // The frame exactly at the target is inside it: the ladder steps for a
        // frame the host was told was *over* budget, not one that met it.
        assert!(!ladder.observe(10_000_000));
        assert_eq!(ladder.over_target(), 0);
    }

    #[test]
    fn the_run_remembers_the_cost_it_is_a_run_of() {
        let mut ladder = FrameLadder::new(10, 2);
        // Nothing counted: no number to report.
        assert_eq!(ladder.last_over_ns(), 0);
        assert!(!ladder.observe(11_000_000));
        assert_eq!(ladder.last_over_ns(), 11_000_000);
        // The number follows the run: the most recent frame it counted.
        assert!(ladder.observe(13_000_000));
        assert_eq!(ladder.over_target(), 2);
        assert_eq!(ladder.last_over_ns(), 13_000_000);
        // A frame inside the target ends the run, and the run's number with it -
        // the entry a relabel writes names the frame it acted on, not a stale one.
        assert!(!ladder.observe(9_000_000));
        assert_eq!(ladder.over_target(), 0);
        assert_eq!(ladder.last_over_ns(), 0);
    }

    #[test]
    fn a_ladder_with_no_target_or_no_threshold_never_acts() {
        let mut no_target = FrameLadder::new(0, 4);
        for _ in 0..8 {
            assert!(!no_target.observe(1_000_000_000));
        }
        let mut no_threshold = FrameLadder::new(1, 0);
        for _ in 0..8 {
            assert!(!no_threshold.observe(1_000_000_000));
        }
        // The run is counted either way - "how many frames over target" is a
        // question a host may ask of a device that was told not to act on it -
        // but a ladder that does not act is a ladder that does not act.
        assert_eq!(no_threshold.over_target(), 8);
    }

    #[test]
    fn the_run_answers_the_relabel_before_the_frame_that_arms_it_is_counted() {
        let mut ladder = FrameLadder::new(10, 2);
        // Two misses: the second is the frame the offload layer acts on, and it
        // is only after it that the relabel layer's question is true. So the
        // relabel lands on frame 2 while the frame the host read over target was
        // frame 1 - the order every backend's own ladder used before these two
        // layers shared one run.
        assert!(!ladder.observe(20_000_000));
        assert!(!ladder.acting(), "one miss does not arm a two-frame threshold");
        assert!(ladder.observe(20_000_000));
        assert!(ladder.acting(), "the second miss arms it");
        // A frame inside the target ends the run: the relabel has nothing to
        // answer either.
        assert!(!ladder.observe(1_000_000));
        assert!(!ladder.acting());
    }

    #[test]
    fn no_gpu_api_resolves_to_t2_not_an_error() {
        let step = resolve_tier(Tier::GpuDiscrete, &inputs());
        assert_eq!(step.tier, Tier::CpuRam);
        assert_eq!(step.reason, TierReason::NoGpuApi);
    }

    #[test]
    fn host_may_pin_a_lower_tier() {
        let mut i = inputs();
        i.gpu_usable = true;
        let step = resolve_tier(Tier::CpuRam, &i);
        assert_eq!(step.tier, Tier::CpuRam);
        assert_eq!(step.reason, TierReason::HostRequest);
    }

    #[test]
    fn ram_cap_overflow_with_disk_opt_in_is_t4() {
        let mut i = inputs();
        i.ram_cap = 64 << 20;
        i.working_set_bytes = 96 << 20;
        i.allow_disk_spill = true;
        i.disk_cap = 512 << 20;
        let step = resolve_tier(Tier::CpuRam, &i);
        assert_eq!(step.tier, Tier::OutOfCore);
    }

    #[test]
    fn ram_cap_overflow_without_disk_opt_in_is_thrifty() {
        let mut i = inputs();
        i.ram_cap = 64 << 20;
        i.working_set_bytes = 96 << 20;
        let step = resolve_tier(Tier::CpuRam, &i);
        assert_eq!(step.tier, Tier::CpuThrifty);
    }

    #[test]
    fn single_threaded_host_cannot_claim_t2() {
        let mut i = inputs();
        i.caps = caps::SHADOWS;
        let step = resolve_tier(Tier::CpuRam, &i);
        assert_eq!(step.tier, Tier::CpuThrifty);
    }

    #[test]
    fn shadow_ladder_clamps_and_reports() {
        let plan = shadow_plan(Tier::OutOfCore, 4, 32 << 20, ShadowFilter::PcssLite, 0);
        assert_eq!(plan.cascades, 1);
        assert_eq!(plan.filter, ShadowFilter::Pcf3x3);
        assert!(plan.disk_backed);
        assert!(plan.freeze_static);
        assert!(plan.clamp_event.is_some());

        let t0 = shadow_plan(Tier::GpuDiscrete, 4, 32 << 20, ShadowFilter::PcssLite, caps::PCSS_LITE);
        assert_eq!(t0.cascades, 4);
        assert_eq!(t0.filter, ShadowFilter::PcssLite);
        assert!(t0.clamp_event.is_none());
        assert!(!t0.disk_backed);
    }

    #[test]
    fn the_plan_owns_the_map_size_and_what_it_costs() {
        // Unscaled tier (T0), two cascades: the budget is spent on the set.
        for (budget_mib, cascades, expected) in [(2u64, 2u32, 512u32), (8, 2, 1024), (32, 2, 2048)] {
            let plan = shadow_plan(Tier::GpuDiscrete, cascades, budget_mib << 20, ShadowFilter::Pcf3x3, caps::SHADOWS);
            assert_eq!(plan.map_size, expected, "{budget_mib} MiB across {cascades} cascades");
            assert_eq!(
                plan.resident_bytes,
                u64::from(expected) * u64::from(expected) * 4 * u64::from(cascades)
            );
        }

        // A budget that is not a power of two stops at the largest edge that
        // fits, and the set is priced at what it occupies - not at the budget.
        let odd = shadow_plan(Tier::GpuDiscrete, 2, 3 << 20, ShadowFilter::Pcf3x3, caps::SHADOWS);
        assert_eq!(odd.map_size, 512);
        assert_eq!(odd.resident_bytes, 2 << 20);
        assert!(odd.resident_bytes < odd.map_bytes);
    }

    #[test]
    fn every_tier_sizes_a_set_that_fits_its_budget() {
        for tier in [Tier::GpuDiscrete, Tier::GpuShared, Tier::CpuRam, Tier::CpuThrifty, Tier::OutOfCore] {
            for cascades in 1..=4u32 {
                for budget_mib in [0u64, 1, 2, 8, 24, 32, 128] {
                    let plan = shadow_plan(tier, cascades, budget_mib << 20, ShadowFilter::PcssLite, caps::PCSS_LITE);
                    assert!(plan.map_size.is_power_of_two(), "{tier:?} {cascades} {budget_mib} MiB");
                    assert!(plan.map_size >= 128);
                    assert_eq!(
                        plan.resident_bytes,
                        u64::from(plan.map_size) * u64::from(plan.map_size) * 4 * u64::from(plan.cascades)
                    );
                    assert!(
                        plan.resident_bytes <= plan.map_bytes,
                        "{tier:?} {cascades} cascades from {budget_mib} MiB: {} > {}",
                        plan.resident_bytes,
                        plan.map_bytes
                    );
                }
            }
        }
    }

    #[test]
    fn tier_step_down_saturates() {
        assert_eq!(Tier::GpuDiscrete.step_down(), Tier::GpuShared);
        assert_eq!(Tier::OutOfCore.step_down(), Tier::OutOfCore);
    }
}
