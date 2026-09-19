//! Budget accounting: an explicit cap, honoured or refused, never silently
//! exceeded.
//!
//! Every resident byte ReconL owns is reserved through this module before it is
//! allocated, and released when it is dropped. A failed reservation is a
//! tier-relevant event: it is counted, logged, and handed to the tier resolver,
//! which is how T2 becomes T3 and T4 rather than an out-of-memory crash.

use crate::error::{Code, Error, Result};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// The library's own ceiling on any single allocation.
///
/// This is a *sizing* limit, not an accounting one, and it exists because an
/// absurd descriptor is otherwise not refused at all: it is handed to the host
/// allocator or the driver, which starts committing gigabytes and page-thrashes
/// the machine. A request above this is refused with `BudgetExceeded` before any
/// of that is touched. The host's `ram_cap_bytes` can lower it and nothing can
/// raise it.
pub const MAX_SINGLE_ALLOCATION_BYTES: u64 = 512 << 20;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct BudgetCaps {
    /// 0 = unlimited within the device cap.
    pub vram: u64,
    /// 0 = unlimited. The T4 out-of-core trigger.
    pub ram: u64,
    /// 0 = no disk use at all.
    pub disk: u64,
    pub allow_disk_spill: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Ram,
    Vram,
    Disk,
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub ram_used: u64,
    pub ram_peak: u64,
    pub ram_cap: u64,
    pub spill_resident: u64,
    pub spill_disk: u64,
    pub spill_disk_peak: u64,
    pub spill_disk_cap: u64,
    /// The enforced single-allocation ceiling: what a request must stay under
    /// to be attempted at all. See [`Budget::max_allocation_bytes`].
    pub max_allocation_bytes: u64,
    pub over_budget_events: u32,
    pub refusals: u32,
    pub disk_full_events: u32,
    pub reservations: u32,
}

pub struct Budget {
    caps: BudgetCaps,
    ram_used: AtomicU64,
    ram_peak: AtomicU64,
    spill_resident: AtomicU64,
    spill_disk: AtomicU64,
    spill_disk_peak: AtomicU64,
    over_budget: AtomicU32,
    refusals: AtomicU32,
    disk_full: AtomicU32,
    reservations: AtomicU32,
}

impl Budget {
    pub fn new(caps: BudgetCaps) -> Self {
        Self {
            caps,
            ram_used: AtomicU64::new(0),
            ram_peak: AtomicU64::new(0),
            spill_resident: AtomicU64::new(0),
            spill_disk: AtomicU64::new(0),
            spill_disk_peak: AtomicU64::new(0),
            over_budget: AtomicU32::new(0),
            refusals: AtomicU32::new(0),
            disk_full: AtomicU32::new(0),
            reservations: AtomicU32::new(0),
        }
    }

    pub fn caps(&self) -> BudgetCaps {
        self.caps
    }

    pub fn set_caps(self: &Arc<Self>, caps: BudgetCaps) -> Arc<Budget> {
        let b = Budget::new(caps);
        b.ram_used.store(self.ram_used.load(Ordering::Relaxed), Ordering::Relaxed);
        b.ram_peak.store(self.ram_peak.load(Ordering::Relaxed), Ordering::Relaxed);
        b.spill_resident.store(self.spill_resident.load(Ordering::Relaxed), Ordering::Relaxed);
        b.spill_disk.store(self.spill_disk.load(Ordering::Relaxed), Ordering::Relaxed);
        b.spill_disk_peak.store(self.spill_disk_peak.load(Ordering::Relaxed), Ordering::Relaxed);
        Arc::new(b)
    }

    /// Bytes of RAM still inside the cap (u64::MAX when uncapped).
    pub fn ram_headroom(&self) -> u64 {
        if self.caps.ram == 0 {
            u64::MAX
        } else {
            self.caps.ram.saturating_sub(self.ram_used.load(Ordering::Relaxed))
        }
    }

    pub fn disk_headroom(&self) -> u64 {
        if self.caps.disk == 0 {
            0
        } else {
            self.caps.disk.saturating_sub(self.spill_disk.load(Ordering::Relaxed))
        }
    }

    pub fn ram_used(&self) -> u64 {
        self.ram_used.load(Ordering::Relaxed)
    }

    pub fn disk_used(&self) -> u64 {
        self.spill_disk.load(Ordering::Relaxed)
    }

    pub fn snapshot(&self) -> BudgetSnapshot {
        BudgetSnapshot {
            ram_used: self.ram_used.load(Ordering::Relaxed),
            ram_peak: self.ram_peak.load(Ordering::Relaxed),
            ram_cap: self.caps.ram,
            spill_resident: self.spill_resident.load(Ordering::Relaxed),
            spill_disk: self.spill_disk.load(Ordering::Relaxed),
            spill_disk_peak: self.spill_disk_peak.load(Ordering::Relaxed),
            spill_disk_cap: self.caps.disk,
            max_allocation_bytes: self.max_allocation_bytes(),
            over_budget_events: self.over_budget.load(Ordering::Relaxed),
            refusals: self.refusals.load(Ordering::Relaxed),
            disk_full_events: self.disk_full.load(Ordering::Relaxed),
            reservations: self.reservations.load(Ordering::Relaxed),
        }
    }

    /// The largest single allocation this budget will attempt: the host's RAM
    /// cap when it set one and it is the lower of the two, never above the
    /// library's own [`MAX_SINGLE_ALLOCATION_BYTES`].
    pub fn max_allocation_bytes(&self) -> u64 {
        if self.caps.ram == 0 {
            MAX_SINGLE_ALLOCATION_BYTES
        } else {
            self.caps.ram.min(MAX_SINGLE_ALLOCATION_BYTES)
        }
    }

    /// The single gate every allocation path passes before it touches the host
    /// allocator, the driver, or the spill arena: the single-allocation ceiling
    /// first, then whatever the caller does with the bytes.
    ///
    /// `what` names the request, so a refusal says which descriptor to shrink
    /// and not only how many bytes it implied.
    pub fn check_allocation(&self, bytes: u64, what: &str) -> Result<()> {
        let ceiling = self.max_allocation_bytes();
        if bytes > ceiling {
            self.over_budget.fetch_add(1, Ordering::Relaxed);
            self.refusals.fetch_add(1, Ordering::Relaxed);
            return crate::err!(
                Code::BudgetExceeded,
                "{} needs {} bytes in a single allocation, above this device's {}-byte ceiling; nothing was allocated",
                what,
                bytes,
                ceiling
            );
        }
        Ok(())
    }

    /// Reserves resident RAM for an allocation that is about to be made.
    /// `Err(BudgetExceeded)` when the ceiling or the cap would break.
    pub fn admit_ram(self: &Arc<Self>, bytes: u64, what: &str) -> Result<Reservation> {
        self.check_allocation(bytes, what)?;
        self.reserve_ram(bytes)
    }

    /// Reserves resident RAM. `Err(BudgetExceeded)` when the cap would break, or
    /// when the request is above the single-allocation ceiling.
    pub fn reserve_ram(self: &Arc<Self>, bytes: u64) -> Result<Reservation> {
        self.check_allocation(bytes, "a RAM reservation")?;
        if self.caps.ram != 0 {
            let used = self.ram_used.load(Ordering::Relaxed);
            if used.saturating_add(bytes) > self.caps.ram {
                self.over_budget.fetch_add(1, Ordering::Relaxed);
                self.refusals.fetch_add(1, Ordering::Relaxed);
                return crate::err!(
                    Code::BudgetExceeded,
                    "RAM reservation of {} bytes would exceed the {}-byte cap ({} in use)",
                    bytes,
                    self.caps.ram,
                    used
                );
            }
        }
        self.ram_used.fetch_add(bytes, Ordering::Relaxed);
        self.bump_peak(bytes, true);
        self.reservations.fetch_add(1, Ordering::Relaxed);
        Ok(Reservation { budget: Arc::clone(self), class: Class::Ram, bytes })
    }

    /// Reserves resident RAM that is allowed to live in the spill arena.
    pub fn reserve_spill(self: &Arc<Self>, bytes: u64) -> Result<Reservation> {
        self.spill_resident.fetch_add(bytes, Ordering::Relaxed);
        self.reservations.fetch_add(1, Ordering::Relaxed);
        Ok(Reservation { budget: Arc::clone(self), class: Class::Vram, bytes })
    }

    /// Reserves disk bytes in the arena. A full arena is a downgrade trigger,
    /// not a failure: callers fall back to the RAM-only path.
    pub fn reserve_disk(self: &Arc<Self>, bytes: u64) -> Result<Reservation> {
        if !self.caps.allow_disk_spill {
            return crate::err!(Code::NotSupported, "disk spill was not opted in");
        }
        let used = self.spill_disk.load(Ordering::Relaxed);
        let cap = if self.caps.disk == 0 { u64::MAX } else { self.caps.disk };
        if used.saturating_add(bytes) > cap {
            self.over_budget.fetch_add(1, Ordering::Relaxed);
            self.refusals.fetch_add(1, Ordering::Relaxed);
            return crate::err!(
                Code::BudgetExceeded,
                "disk reservation of {} bytes would exceed the {}-byte cache cap",
                bytes,
                cap
            );
        }
        self.spill_disk.fetch_add(bytes, Ordering::Relaxed);
        let peak = self.spill_disk.load(Ordering::Relaxed);
        let mut cur = self.spill_disk_peak.load(Ordering::Relaxed);
        while peak > cur {
            match self.spill_disk_peak.compare_exchange_weak(cur, peak, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
        self.reservations.fetch_add(1, Ordering::Relaxed);
        Ok(Reservation { budget: Arc::clone(self), class: Class::Disk, bytes })
    }

    fn bump_peak(&self, bytes: u64, ram: bool) {
        if !ram {
            return;
        }
        let used = self.ram_used.load(Ordering::Relaxed);
        let mut peak = self.ram_peak.load(Ordering::Relaxed);
        while used > peak {
            match self.ram_peak.compare_exchange_weak(peak, used, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(now) => peak = now,
            }
        }
        let _ = bytes;
    }

    /// The file system said no. Counted, and the resolver may step down.
    pub fn note_disk_full(&self) {
        self.disk_full.fetch_add(1, Ordering::Relaxed);
        self.over_budget.fetch_add(1, Ordering::Relaxed);
    }

    fn release(&self, class: Class, bytes: u64) {
        match class {
            Class::Ram => {
                self.ram_used.fetch_sub(bytes, Ordering::Relaxed);
            }
            Class::Vram => {
                self.spill_resident.fetch_sub(bytes, Ordering::Relaxed);
            }
            Class::Disk => {
                self.spill_disk.fetch_sub(bytes, Ordering::Relaxed);
            }
        }
    }
}

/// RAII reservation: dropping it gives the bytes back, so there is no path that
/// leaks budget without also leaking memory.
pub struct Reservation {
    budget: Arc<Budget>,
    class: Class,
    bytes: u64,
}

impl core::fmt::Debug for Reservation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Reservation({:?}, {} bytes)", self.class, self.bytes)
    }
}

impl Reservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn class(&self) -> Class {
        self.class
    }
    pub fn resize(&mut self, new_bytes: u64) {
        if new_bytes >= self.bytes {
            self.budget.reserve_add(self.class, new_bytes - self.bytes);
        } else {
            self.budget.release(self.class, self.bytes - new_bytes);
        }
        self.bytes = new_bytes;
    }
}

impl Budget {
    fn reserve_add(&self, class: Class, extra: u64) {
        match class {
            Class::Ram => {
                self.ram_used.fetch_add(extra, Ordering::Relaxed);
                self.bump_peak(extra, true);
            }
            Class::Vram => {
                self.spill_resident.fetch_add(extra, Ordering::Relaxed);
            }
            Class::Disk => {
                self.spill_disk.fetch_add(extra, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.release(self.class, self.bytes);
        self.budget.reservations.fetch_sub(1, Ordering::Relaxed);
    }
}

pub fn exceeded(err: &Error) -> bool {
    err.code == Code::BudgetExceeded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(caps: BudgetCaps) -> Arc<Budget> {
        Arc::new(Budget::new(caps))
    }

    #[test]
    fn uncapped_reservation_always_succeeds() {
        let b = arc(BudgetCaps::default());
        let r = b.reserve_ram(256 << 20).unwrap();
        assert_eq!(b.snapshot().ram_used, 256 << 20);
        drop(r);
        assert_eq!(b.snapshot().ram_used, 0);
    }

    #[test]
    fn a_single_allocation_above_the_ceiling_is_refused_untouched() {
        let b = arc(BudgetCaps::default());
        let err = b
            .admit_ram(MAX_SINGLE_ALLOCATION_BYTES + 1, "a 16384x16384 frame")
            .unwrap_err();
        assert!(exceeded(&err));
        // Nothing was counted as resident, so nothing was allocated either.
        assert_eq!(b.snapshot().ram_used, 0);
        assert_eq!(b.snapshot().ram_peak, 0);
        assert_eq!(b.snapshot().reservations, 0);
        assert_eq!(b.snapshot().refusals, 1);
        assert_eq!(b.snapshot().over_budget_events, 1);
        // And the refusal names the descriptor and the ceiling.
        assert!(err.message_str().contains("16384x16384"), "{err}");
        assert!(err.message_str().contains("-byte ceiling"), "{err}");
    }

    #[test]
    fn the_backends_reservation_path_is_bounded_too() {
        // The frame and shadow-map paths call `reserve_ram` directly, so the
        // ceiling has to hold there and not only on the labelled entry point.
        let b = arc(BudgetCaps::default());
        let err = b.reserve_ram(u64::from(u32::MAX) * u64::from(u32::MAX)).unwrap_err();
        assert!(exceeded(&err));
        assert_eq!(b.snapshot().ram_used, 0);
    }

    #[test]
    fn a_large_but_affordable_allocation_still_succeeds() {
        let b = arc(BudgetCaps::default());
        let r = b.admit_ram(256 << 20, "a 4096x4096 frame").unwrap();
        assert_eq!(b.snapshot().ram_used, 256 << 20);
        assert_eq!(b.snapshot().max_allocation_bytes, MAX_SINGLE_ALLOCATION_BYTES);
        drop(r);
        assert_eq!(b.snapshot().ram_used, 0);
        assert_eq!(b.snapshot().refusals, 0);
    }

    #[test]
    fn a_ram_cap_lowers_the_ceiling_and_never_raises_it() {
        let low = arc(BudgetCaps { ram: 4 << 20, ..Default::default() });
        assert_eq!(low.max_allocation_bytes(), 4 << 20);
        let err = low.admit_ram((4 << 20) + 1, "a texture").unwrap_err();
        assert!(exceeded(&err));
        let high = arc(BudgetCaps { ram: 256 << 30, ..Default::default() });
        assert_eq!(high.max_allocation_bytes(), MAX_SINGLE_ALLOCATION_BYTES);
    }

    #[test]
    fn cap_is_honoured_and_refusal_is_counted() {
        let b = arc(BudgetCaps { ram: 1000, ..Default::default() });
        let _r = b.reserve_ram(900).unwrap();
        let err = b.reserve_ram(200).unwrap_err();
        assert!(exceeded(&err));
        assert_eq!(b.snapshot().refusals, 1);
        assert_eq!(b.snapshot().over_budget_events, 1);
        assert_eq!(b.snapshot().ram_peak, 900);
    }

    #[test]
    fn disk_needs_the_opt_in() {
        let b = arc(BudgetCaps { disk: 1 << 20, allow_disk_spill: false, ..Default::default() });
        assert_eq!(b.reserve_disk(4096).unwrap_err().code, Code::NotSupported);
        let b = arc(BudgetCaps { disk: 1 << 20, allow_disk_spill: true, ..Default::default() });
        let r = b.reserve_disk(4096).unwrap();
        assert_eq!(b.snapshot().spill_disk, 4096);
        assert_eq!(b.disk_headroom(), (1 << 20) - 4096);
        drop(r);
        assert_eq!(b.snapshot().spill_disk, 0);
    }

    #[test]
    fn resize_grows_and_shrinks_without_leaking() {
        let b = arc(BudgetCaps::default());
        let mut r = b.reserve_ram(1000).unwrap();
        r.resize(4000);
        assert_eq!(b.snapshot().ram_used, 4000);
        r.resize(512);
        assert_eq!(b.snapshot().ram_used, 512);
        drop(r);
        assert_eq!(b.snapshot().ram_used, 0);
        assert_eq!(b.snapshot().reservations, 0);
    }

    #[test]
    fn peak_tracks_the_high_water_mark_not_the_current() {
        let b = arc(BudgetCaps::default());
        {
            let _a = b.reserve_ram(5_000_000).unwrap();
            let _c = b.reserve_ram(2_000_000).unwrap();
        }
        assert_eq!(b.snapshot().ram_peak, 7_000_000);
        assert_eq!(b.snapshot().ram_used, 0);
    }
}
