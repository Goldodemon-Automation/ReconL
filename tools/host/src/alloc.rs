//! The host allocator the tools hand to ReconL, with counters.
//!
//! A conforming allocator is a hard requirement of `reconlCreateDevice` - the
//! library refuses to start on a zeroed one - and it is also the only place a
//! host sees every block the library really takes. `reconlGetMemoryStats`
//! already reports the library's own view of that traffic; these counters are
//! the host side of the same ledger, so the two can be compared rather than
//! trusted.
//!
//! Blocks carry their bookkeeping in 16 bytes of slack before the aligned
//! pointer - the allocation's base and the layout size - so `free` can hand the
//! allocator back exactly the layout it was handed, which is what
//! `std::alloc::dealloc` requires and what a size-ignoring free only pretends to
//! do.

use core::ffi::c_void;
use core::sync::atomic::{AtomicU64, Ordering};
use reconl::abi;

/// Slack in front of every aligned pointer: `[base, layout_size]`.
const SLACK: usize = 16;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static FREE_CALLS: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Counters {
    pub alloc_calls: u64,
    pub alloc_bytes: u64,
    pub free_calls: u64,
    pub live_bytes: u64,
    pub peak_bytes: u64,
}

impl Counters {
    /// Allocations and frees still outstanding: zero after every handle is
    /// released is what proves the host handed back all the memory.
    pub fn live_blocks(&self) -> i64 {
        self.alloc_calls as i64 - self.free_calls as i64
    }
}

/// What this allocator has been asked for since the process started, or since
/// the last [`reset`].
pub fn counters() -> Counters {
    Counters {
        alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        free_calls: FREE_CALLS.load(Ordering::Relaxed),
        live_bytes: LIVE_BYTES.load(Ordering::Relaxed),
        peak_bytes: PEAK_BYTES.load(Ordering::Relaxed),
    }
}

/// Zeroes the counters. A tool calls this immediately before the region it wants
/// to measure, so "0 allocations during the frames" is about those frames.
///
/// Only the *counts* are trustworthy after a reset; the live-bytes figure is
/// relative to the reset point, because blocks taken earlier and released later
/// are not attributable to either window.
pub fn reset() {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    FREE_CALLS.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_BYTES.store(0, Ordering::Relaxed);
}

fn add(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_add(value)));
}

fn sub(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(value)));
}

/// Books an allocation. Called from inside the `extern "C"` callback, so every
/// step is saturating and lock-free: a counter that panics here aborts the
/// process, which is a spectacular way for a measurement to fail, and a counter
/// that wraps would report a steady state as billions of bytes live.
///
/// The live figure saturates at both ends for one specific case: a block taken
/// *before* a [`reset`] and released after it. That free subtracts nothing rather
/// than driving the figure negative, which is why a claim about a measurement
/// window is made from [`Counters::alloc_calls`] and not from the live total.
fn note_alloc(size: u64) {
    add(&ALLOC_CALLS, 1);
    add(&ALLOC_BYTES, size);
    add(&LIVE_BYTES, size);
    let live = LIVE_BYTES.load(Ordering::Relaxed);
    let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
    while live > peak {
        match PEAK_BYTES.compare_exchange_weak(peak, live, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(now) => peak = now,
        }
    }
}

fn note_free(size: u64) {
    add(&FREE_CALLS, 1);
    sub(&LIVE_BYTES, size);
}

extern "C" fn tool_alloc(_user: *mut c_void, size: usize, alignment: usize) -> *mut c_void {
    let align = alignment.clamp(16, 4096);
    let size = size.max(1);
    let total = match size.checked_add(align).and_then(|v| v.checked_add(SLACK)) {
        Some(t) => t,
        None => return core::ptr::null_mut(),
    };
    let layout = match std::alloc::Layout::from_size_align(total, 16) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };
    // SAFETY: `layout` has a non-zero size.
    let raw = unsafe { std::alloc::alloc(layout) };
    if raw.is_null() {
        return core::ptr::null_mut();
    }
    let aligned = (raw as usize + SLACK + align - 1) & !(align - 1);
    // SAFETY: SLACK bytes sit between `raw` and `aligned`, and `align` keeps the
    // pointer far enough in for both words to be inside this block.
    unsafe {
        ((aligned - 8) as *mut usize).write_unaligned(total);
        ((aligned - 16) as *mut usize).write_unaligned(raw as usize);
    }
    note_alloc(size as u64);
    aligned as *mut c_void
}

extern "C" fn tool_free(_user: *mut c_void, ptr: *mut c_void, size: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: every pointer handed out by `tool_alloc` carries these two words.
    let (base, total) = unsafe {
        (
            ((ptr as usize - 16) as *const usize).read_unaligned(),
            ((ptr as usize - 8) as *const usize).read_unaligned(),
        )
    };
    note_free(size.max(1) as u64);
    // SAFETY: `base`/`total` are exactly what `tool_alloc` passed to `alloc`.
    unsafe {
        std::alloc::dealloc(base as *mut u8, std::alloc::Layout::from_size_align_unchecked(total, 16));
    }
}

extern "C" fn tool_realloc(
    _user: *mut c_void,
    ptr: *mut c_void,
    old_size: usize,
    new_size: usize,
    alignment: usize,
) -> *mut c_void {
    let fresh = tool_alloc(core::ptr::null_mut(), new_size, alignment);
    if fresh.is_null() {
        return core::ptr::null_mut();
    }
    if !ptr.is_null() {
        // SAFETY: both blocks are live for their own sizes and do not overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr as *const u8, fresh as *mut u8, old_size.min(new_size));
        }
        tool_free(core::ptr::null_mut(), ptr, old_size);
    }
    fresh
}

/// The allocator every tool passes to `reconlCreateDevice`.
pub fn allocator() -> abi::ReconLAllocator {
    abi::ReconLAllocator {
        alloc: Some(tool_alloc),
        realloc: Some(tool_realloc),
        free: Some(tool_free),
        user: core::ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A free of something never allocated saturates instead of wrapping: the
    /// one shape that used to drive the ledger to `u64::MAX` and then abort the
    /// process on the next allocation.
    #[test]
    #[serial_test::serial]
    fn a_free_of_an_unknown_block_saturates_rather_than_wrapping() {
        let alloc = allocator();
        let f = alloc.alloc.unwrap();
        let free = alloc.free.unwrap();
        reset();
        // SAFETY: allocator contract, driven directly.
        let a = unsafe { f(core::ptr::null_mut(), 64, 16) };
        // SAFETY: a block that was taken before the reset, released after it.
        unsafe {
            free(core::ptr::null_mut(), a, 64);
        }
        let after_release = counters();
        assert_eq!(after_release.live_bytes, 0, "no negative live bytes in an unsigned ledger");
        // SAFETY: and the next allocation still books correctly.
        let b = unsafe { f(core::ptr::null_mut(), 1024, 16) };
        assert_eq!(counters().live_bytes, 1024);
        // SAFETY: `b` is live with this size.
        unsafe { free(core::ptr::null_mut(), b, 1024) };
    }

    // The ledger is process-wide, so these three tests read and reset counters
    // that every other test in this binary books into concurrently: exact
    // assertions here are only honest when nothing else runs at the same time
    // (the same reason the CLI tools measure in their own process). One test
    // thread for the tests that assert the counts themselves.
    #[test]
    #[serial_test::serial]
    fn blocks_round_trip_and_the_counters_balance() {
        let alloc = allocator();
        let f = alloc.alloc.unwrap();
        let free = alloc.free.unwrap();
        reset();
        // SAFETY: this is the allocator's own contract, driven directly.
        let a = unsafe { f(core::ptr::null_mut(), 1000, 64) };
        let b = unsafe { f(core::ptr::null_mut(), 4096, 16) };
        assert!(!a.is_null() && !b.is_null());
        assert_eq!(a as usize % 64, 0, "alignment is honoured");
        let after = counters();
        assert_eq!(after.alloc_calls, 2);
        assert_eq!(after.alloc_bytes, 5096);
        assert_eq!(after.live_bytes, 5096);
        assert_eq!(after.peak_bytes, 5096);
        // SAFETY: both pointers came from the allocator above, with these sizes.
        unsafe {
            free(core::ptr::null_mut(), a, 1000);
            free(core::ptr::null_mut(), b, 4096);
        }
        let end = counters();
        assert_eq!(end.free_calls, 2);
        assert_eq!(end.live_bytes, 0);
        assert_eq!(end.live_blocks(), 0);
        assert_eq!(end.peak_bytes, 5096, "the peak is a high-water mark, not the current");
    }

    #[test]
    #[serial_test::serial]
    fn realloc_moves_the_bytes_and_keeps_the_ledger_straight() {
        let alloc = allocator();
        let f = alloc.alloc.unwrap();
        let r = alloc.realloc.unwrap();
        let free = alloc.free.unwrap();
        reset();
        // SAFETY: allocator contract, driven directly.
        let a = unsafe { f(core::ptr::null_mut(), 8, 16) };
        // SAFETY: `a` holds 8 writable bytes.
        unsafe { core::ptr::write_bytes(a as *mut u8, 0xAB, 8) };
        // SAFETY: growing a live 8-byte block to 64.
        let b = unsafe { r(core::ptr::null_mut(), a, 8, 64, 16) };
        assert!(!b.is_null());
        // SAFETY: the first 8 bytes were copied by the allocator.
        assert_eq!(unsafe { (b as *const u8).read() }, 0xAB);
        let mid = counters();
        assert_eq!(mid.live_bytes, 64, "the old block was given back");
        // SAFETY: `b` came from this allocator with size 64.
        unsafe { free(core::ptr::null_mut(), b, 64) };
        assert_eq!(counters().live_bytes, 0);
    }
}
