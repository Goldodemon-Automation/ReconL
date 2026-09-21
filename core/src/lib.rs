//! ReconL core: tier resolver, budget accounting, error model, stats, logging
//! and the host-allocator plumbing every other crate builds on.
//!
//! No graphics here, no platform code here, no external dependencies here.
//! Anything that knows how to put a pixel somewhere lives in a backend.
//!
//! Handles are deliberately *not* here. A handle's header is part of the C
//! surface - `reconlRetain`/`reconlRelease` read it off a `void*` - and it
//! carries the device back-pointer the FFI uses to prove a child belongs to its
//! device, so it lives in `ffi/src/handle.rs` with the rest of the ABI's object
//! model. This crate used to carry a second, unused handle table; two
//! representations of one concern is how a reader learns the wrong one.
#![forbid(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]

pub mod alloc;
pub mod budget;
pub mod error;
pub mod hash;
pub mod log;
pub mod stats;
pub mod text;
pub mod tier;

pub use alloc::{HostAlloc, HostAllocatorA, HostBox, HostStats, HostVec};
pub use budget::{Budget, BudgetCaps, Reservation};
pub use error::{Code, Error, Result};
pub use hash::{xxh64, xxh64_seeded, XxHash64};
pub use log::Level;
pub use stats::{Counters, Downgrade, DowngradeLog, ShadowCounters, Stats, TimingAccum};
pub use text::Text;
pub use tier::{resolve_tier, shadow_plan, Backend, Tier, TierRules, TierStep, TIER_COUNT};

/// `struct_size` / `type` / `next` triple that opens every ABI struct.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StructHeader {
    pub struct_size: u32,
    pub struct_type: u32,
    pub next: *const core::ffi::c_void,
}

impl StructHeader {
    pub const fn new(struct_size: u32, struct_type: u32) -> Self {
        Self { struct_size, struct_type, next: core::ptr::null() }
    }
}

impl Default for StructHeader {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

/// Validates the common header of an incoming struct.
///
/// Returns `Err(Code::StructSize)` when the caller's `struct_size` is too small
/// to contain the fields this library reads, and `Err(Code::WrongStructType)`
/// when `type` names a different struct. Both are recoverable and reported, not
/// crashed on - a newer library must never hard-fail an older header.
///
/// # Safety
/// `header` must point at a readable struct whose first three members are the
/// `ReconLBase` triple, and the returned reference must not outlive the ABI call
/// that provided the pointer.
pub unsafe fn check_header<'a, H>(
    header: *const StructHeader,
    expected_type: u32,
    min_size: u32,
    what: &str,
) -> Result<&'a H>
where
    H: ABIStruct,
{
    if header.is_null() {
        return err!(Code::InvalidArgument, "null {} descriptor", what);
    }
    // SAFETY: caller guarantees `header` points at a readable StructHeader.
    let base = unsafe { &*header };
    if base.struct_size < min_size {
        return err!(
            Code::StructSize,
            "{} struct_size is {} but this library reads {}",
            what,
            base.struct_size,
            min_size
        );
    }
    if base.struct_type != expected_type && base.struct_type != 0 {
        return err!(Code::WrongStructType, "{} type is {} but {} was expected", what, base.struct_type, expected_type);
    }
    // SAFETY: `min_size` was validated above, so H's prefix is present in the
    // caller's struct; the caller keeps it alive for the duration of the call.
    Ok(unsafe { &*(header as *const H) })
}

/// Marker for `#[repr(C)]` structs that begin with [`StructHeader`].
pub trait ABIStruct {
    const STRUCT_TYPE: u32;
    /// Size of the prefix of the struct that this revision of the library reads.
    const MIN_SIZE: u32;
}
