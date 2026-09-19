//! Host-side plumbing shared by the ReconL tools.
//!
//! Every tool here is a *host*: it drives the same C ABI an application does,
//! through the same `reconl-ffi` entry points a C program calls. So they all
//! need the same six things before they can measure anything - a conforming
//! allocator, a device, the camera and scene the reference frame is made of,
//! the frame plumbing that encodes it, readable names for the ABI's enums, and
//! units a human can read. That is what this crate is, and nothing else: it has
//! no `main`, no scene of its own beyond the documented reference one, and no
//! opinion about what a tool reports.
//!
//! Keeping it here rather than copy-pasted into each tool is the point, and it
//! is why the reference scene has exactly one owner: `reconl-diff` renders it,
//! `reconl-bench` times it, and a cross-tier comparison between the two is a
//! comparison of the *same* scene rather than of two scenes that happen to
//! agree today.
//!
//! The allocator is also the only place a host sees every block the library
//! really takes, which is what makes "no allocations during a frame in steady
//! state" a measurement instead of a belief.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod alloc;
pub mod device;
pub mod frame;
pub mod math;
pub mod names;
pub mod scene;
pub mod units;

use reconl::abi;
use reconl_core::{ABIStruct, StructHeader};

/// The header every ABI descriptor this crate builds starts with.
///
/// Derived from the Rust type, so a struct that gains a field on the Rust side
/// and not in the header fails to compile at the call site rather than
/// mis-declaring its own size.
pub fn hdr<T: ABIStruct>() -> StructHeader {
    StructHeader::new(core::mem::size_of::<T>() as u32, T::STRUCT_TYPE)
}

/// Reads a `*const c_char` the ABI owns into a `String` (lossy, NUL-terminated).
///
/// # Safety
/// `p` must be null or a pointer to a NUL-terminated string that outlives the
/// call, as every `reconl*Name` accessor promises.
pub unsafe fn cstr_of(p: *const core::ffi::c_char) -> String {
    if p.is_null() {
        return "?".into();
    }
    // SAFETY: the caller's contract.
    let bytes = unsafe { core::ffi::CStr::from_ptr(p) };
    bytes.to_string_lossy().into_owned()
}

/// The name `reconlResultName` gives a result code - the same string a host
/// logs, so a tool and its host agree on the vocabulary.
pub fn result_name(r: i32) -> String {
    // SAFETY: reconlResultName returns a static NUL-terminated string for any input.
    unsafe { cstr_of(reconl::reconlResultName(r)) }
}

/// The ABI's own name for a descriptor's error, for a call that failed.
pub fn failed(what: &str, r: i32) -> String {
    format!("{what}: {} ({r})", result_name(r))
}

/// Whether a descriptor the library refused was refused because it is older
/// than this revision's minimum readable prefix.
pub fn is_short_struct(r: i32) -> bool {
    r == abi::result::STRUCT_SIZE || r == abi::result::WRONG_STRUCT_TYPE
}
