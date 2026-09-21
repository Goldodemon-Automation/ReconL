//! Stats: what actually happened, in numbers the host can query.
//!
//! Everything here is written by the frame path and read by `reconlGetStats`.
//! Counters never allocate and never block, so recording a downgrade inside a
//! frame costs a store.

use crate::budget::BudgetSnapshot;
use crate::error::Code;
use crate::text::Text;
use crate::tier::{ShadowFilter, Tier, TierReason};

pub const MAX_DOWNGRADES: usize = 16;
pub const DOWNGRADE_DETAIL_CAP: usize = 192;

#[derive(Clone, Copy)]
pub struct Downgrade {
    pub from: Tier,
    pub to: Tier,
    pub reason: TierReason,
    pub frame_index: u64,
    pub at_ns: u64,
    pub detail: Text<DOWNGRADE_DETAIL_CAP>,
}

impl Downgrade {
    pub fn new(from: Tier, to: Tier, reason: TierReason, frame_index: u64, at_ns: u64, detail: &str) -> Self {
        Self { from, to, reason, frame_index, at_ns, detail: Text::from_str(detail) }
    }
}

/// Fixed ring of the most recent steps, plus the total ever taken.
///
/// A ring rather than a queue because the frame path may not allocate, and
/// because the first downgrade matters more than the ten-thousandth: the count
/// is unbounded, the history is not.
pub struct DowngradeLog {
    ring: [Option<Downgrade>; MAX_DOWNGRADES],
    next: usize,
    len: usize,
    total: u32,
}

impl Default for DowngradeLog {
    fn default() -> Self {
        Self::new()
    }
}

impl DowngradeLog {
    pub const fn new() -> Self {
        Self { ring: [None; MAX_DOWNGRADES], next: 0, len: 0, total: 0 }
    }

    pub fn record(&mut self, d: Downgrade) -> Option<Downgrade> {
        let evicted = self.ring[self.next].take();
        self.ring[self.next] = Some(d);
        self.next = (self.next + 1) % MAX_DOWNGRADES;
        if self.len < MAX_DOWNGRADES {
            self.len += 1;
        }
        self.total = self.total.saturating_add(1);
        evicted
    }

    pub fn total(&self) -> u32 {
        self.total
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Oldest retained entry first.
    pub fn iter(&self) -> impl Iterator<Item = &Downgrade> {
        let start = if self.len == MAX_DOWNGRADES { self.next } else { 0 };
        (0..self.len).filter_map(move |i| self.ring[(start + i) % MAX_DOWNGRADES].as_ref())
    }

    pub fn clear(&mut self) {
        self.ring = [None; MAX_DOWNGRADES];
        self.next = 0;
        self.len = 0;
        self.total = 0;
    }
}

/// Rolling min/avg/max. A peak is never quoted as "the" result: the recorder
/// writes all three and the run fingerprint says what produced them.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct TimingAccum {
    pub count: u64,
    pub last_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub sum_ns: u64,
}

impl TimingAccum {
    pub fn push(&mut self, ns: u64) {
        if self.count == 0 {
            self.min_ns = ns;
            self.max_ns = ns;
        } else {
            if ns < self.min_ns {
                self.min_ns = ns;
            }
            if ns > self.max_ns {
                self.max_ns = ns;
            }
        }
        self.last_ns = ns;
        self.sum_ns = self.sum_ns.saturating_add(ns);
        self.count += 1;
    }

    pub fn avg_ns(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.sum_ns / self.count
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ShadowCounters {
    pub cascades_active: u32,
    pub cascades_rendered: u32,
    pub map_width: u32,
    pub map_height: u32,
    pub map_bytes: u64,
    pub filter_active: ShadowFilter,
    pub filter_requested: ShadowFilter,
    pub shadow_pass_ns: u64,
    pub fit_ns: u64,
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub cache_corrupt: u32,
    pub cache_bytes_read: u64,
    pub cache_bytes_hit: u64,
    pub fail_safe_unshadowed: u32,
    pub frozen_cascades: u32,
    pub shadowed_lights: u32,
    pub point_lights_capped: u32,
    pub atlas_entries: u32,
    pub atlas_evictions: u32,
    pub filter_taps: u32,
}

impl ShadowCounters {
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses + self.cache_corrupt;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct FrameNumbers {
    pub frame_index: u64,
    pub total_ns: u64,
    pub shadow_ns: u64,
    pub raster_ns: u64,
    pub bin_ns: u64,
    pub upload_ns: u64,
    pub spill_wait_ns: u64,
    pub tiles_total: u32,
    pub tiles_rendered: u32,
    pub triangles_in: u32,
    pub triangles_binned: u32,
    pub triangles_culled: u32,
    pub pixels_shaded: u32,
    pub worker_threads: u32,
    pub resolution_scale: f32,
    /// Bytes ReconL moved to or from the arena during this frame.
    pub spill_io_bytes: u64,
    pub allocations_in_frame: u32,
}

/// Cross-frame counters a host can read.
///
/// Every field here is counted by the layer that owns the event and is read by
/// someone - a host through `ReconLStats`, the offload policy, or a test. The
/// frame-time ladder's own memory (how many consecutive frames were over the
/// target) is deliberately *not* here: it is per-device state owned by the
/// ladder, and a second copy in these counters would be a second definition of
/// "over target" that nothing reads and nothing keeps in step.
#[derive(Clone, Copy, Default, Debug)]
pub struct Counters {
    pub frames_presented: u32,
    pub frames_dropped: u32,
    pub failures: u32,
    pub safe_path_events: u32,
    pub audit_divergences: u32,
    pub last_result: Option<Code>,
    pub device_losses: u32,
    pub frames_since_tier_change: u32,
}

/// The whole device's observable state.
pub struct Stats {
    pub backend: u32,
    pub tier: Tier,
    pub tier_reason: TierReason,
    pub tier_locked: bool,
    pub tier_reason_text: Text<192>,
    pub device_name: Text<64>,
    pub budget: BudgetSnapshot,
    pub host: crate::alloc::HostStats,
    pub shadows: ShadowCounters,
    pub counters: Counters,
    pub frames: TimingAccum,
    pub last_frame: FrameNumbers,
    pub downgrades: DowngradeLog,
    pub caps: u32,
}

impl Default for Stats {
    fn default() -> Self {
        Self::new(0, Tier::CpuRam, TierReason::StartupProbe)
    }
}

impl Stats {
    pub fn new(backend: u32, tier: Tier, reason: TierReason) -> Self {
        Self {
            backend,
            tier,
            tier_reason: reason,
            tier_locked: false,
            tier_reason_text: Text::new(),
            device_name: Text::new(),
            budget: BudgetSnapshot::default(),
            host: crate::alloc::HostStats::default(),
            shadows: ShadowCounters::default(),
            counters: Counters::default(),
            frames: TimingAccum::default(),
            last_frame: FrameNumbers::default(),
            downgrades: DowngradeLog::new(),
            caps: 0,
        }
    }

    pub fn reset(&mut self) {
        let tier = self.tier;
        let reason = self.tier_reason;
        let backend = self.backend;
        let caps = self.caps;
        self.shadows = ShadowCounters::default();
        self.counters = Counters::default();
        self.frames.reset();
        self.last_frame = FrameNumbers::default();
        self.budget = BudgetSnapshot::default();
        self.tier = tier;
        self.tier_reason = reason;
        self.backend = backend;
        self.caps = caps;
        self.tier_locked = false;
        self.tier_reason_text.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_accum_reports_min_avg_max_not_last() {
        let mut t = TimingAccum::default();
        for v in [10u64, 4, 30, 6] {
            t.push(v);
        }
        assert_eq!(t.min_ns, 4);
        assert_eq!(t.max_ns, 30);
        assert_eq!(t.avg_ns(), 12);
        assert_eq!(t.last_ns, 6);
        assert_eq!(t.count, 4);
    }

    #[test]
    fn downgrade_ring_keeps_the_newest_and_counts_the_rest() {
        let mut log = DowngradeLog::new();
        for i in 0..40u32 {
            log.record(Downgrade::new(
                Tier::GpuDiscrete,
                Tier::GpuShared,
                TierReason::FrameTimeOverTarget,
                i as u64,
                0,
                "over target",
            ));
        }
        assert_eq!(log.total(), 40);
        assert_eq!(log.len(), MAX_DOWNGRADES);
        let first = log.iter().next().unwrap();
        assert_eq!(first.frame_index, 24);
        let last = log.iter().last().unwrap();
        assert_eq!(last.frame_index, 39);
    }

    #[test]
    fn empty_ring_iterates_nothing() {
        let log = DowngradeLog::new();
        assert_eq!(log.iter().count(), 0);
        assert_eq!(log.total(), 0);
    }
}
