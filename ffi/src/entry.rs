//! The boundary glue every entry point uses: how a call gets in, what happens
//! when one fails, and how a C string is read.
//!
//! There are three pieces, and each exists to make a class of entry-point bug
//! impossible rather than to be written correctly in sixty places:
//!
//! * [`FrameOwner`] - a failed `Submit` or `Present` ends the frame it was given
//!   instead of stranding the device in a state only a later success can clear.
//! * [`guarded_entry`] - the panic boundary, so a bug in the renderer is a
//!   return code in a `panic = "unwind"` build rather than a crashed host.
//! * the entry macros - the validation every call repeats: resolve the device
//!   handle, prove a child handle belongs to it, and return the documented code
//!   when either fails.

use crate::{record_error, DeviceHandle, FrameState};
use reconl_core::error::Result;
// Only the panic boundary needs these: in the shipped profile there is no
// unwinding to catch, so a panic is an abort and nothing here names a code.
#[cfg(panic = "unwind")]
use reconl_core::error::{Code, Error};


/// Drops the open frame if the call that owns it fails.
///
/// A device left in `Open` or `Submitted` rejects every later `BeginFrame` with
/// `FrameInProgress`, and there is no ABI call that abandons a frame, so without
/// this a host that records one bad draw - or whose present cannot be delivered -
/// can never render again. Both `Submit` and `Present` take one of these once the
/// call has committed to the frame and mark it `keep` only on success, so every
/// failure in between ends the frame instead of stranding it. The counted drop is
/// what `ReconLStats.frames_dropped` is for.
///
/// Holds the device as a raw pointer on purpose: the caller keeps using its own
/// `&mut`, and the two borrows are disjoint by construction (this one is only
/// read in `drop`, after the caller's last use).
pub(crate) struct FrameOwner {
    device: *mut DeviceHandle,
    /// Set once the frame reached a state worth keeping.
    pub(crate) keep: bool,
}

impl FrameOwner {
    pub(crate) fn new(device: *mut DeviceHandle) -> Self {
        Self { device, keep: false }
    }
}

impl Drop for FrameOwner {
    fn drop(&mut self) {
        if self.keep {
            return;
        }
        // SAFETY: the device outlives the entry point that created this.
        let device = unsafe { &mut *self.device };
        device.frame_state = FrameState::Idle;
        device.stats.counters.frames_dropped += 1;
    }
}

/// Runs an entry point body, converting a `Result` into an ABI code and
/// recording the error where the host will look for it.
pub fn run_entry(device: *mut DeviceHandle, body: impl FnOnce() -> Result<()>) -> i32 {
    match body() {
        Ok(()) => crate::abi::result::OK,
        Err(err) => record_error(device, err),
    }
}

/// An entry point with a panic boundary in `panic = "unwind"` builds. In the
/// shipped profile (`panic = "abort"`) a panic aborts the process, which is the
/// documented contract for `panic` across FFI.
#[cfg(panic = "unwind")]
pub fn guarded_entry(device: *mut DeviceHandle, body: impl FnOnce() -> Result<()>) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(r) => run_entry(device, || r),
        Err(_) => {
            record_error(device, Error::new(Code::Panic, "a ReconL invariant broke; the safe path was taken"))
        }
    }
}

#[cfg(not(panic = "unwind"))]
pub fn guarded_entry(device: *mut DeviceHandle, body: impl FnOnce() -> Result<()>) -> i32 {
    run_entry(device, body)
}

macro_rules! entry {
    ($device:expr, $body:block) => {{
        let device_ptr: *mut $crate::DeviceHandle = $device;
        // The body is wrapped in its own block so the `block` fragment can be
        // used in closure-body position at all.
        $crate::entry::guarded_entry(device_ptr, move || -> ::reconl_core::Result<()> { $body })
    }};
}

/// Validates the device handle and yields `&mut DeviceHandle`.
macro_rules! device_mut {
    ($ptr:expr) => {{
        let header =
            unsafe { $crate::handle::check_handle($ptr as *mut core::ffi::c_void, $crate::handle::Kind::Device, "device")? };
        let device = header as *mut $crate::DeviceHandle;
        unsafe { &mut *device }
    }};
}

/// Validates a child handle and proves it belongs to `device`.
macro_rules! child {
    ($device:expr, $ptr:expr, $kind:expr, $what:expr) => {{
        let header = unsafe { $crate::handle::check_handle($ptr as *mut core::ffi::c_void, $kind, $what)? };
        let device_header = &$device.header;
        if unsafe { (*header).device } != (device_header as *const $crate::handle::HandleHeader as *mut $crate::DeviceHandle) {
            return reconl_core::err!(
                reconl_core::error::Code::InvalidHandle,
                "{} belongs to a different device",
                $what
            );
        }
        header
    }};
}

pub(crate) use child;
pub(crate) use device_mut;
pub(crate) use entry;

/// # Safety
/// `ptr` must point at a NUL-terminated string.
pub(crate) unsafe fn cstr(ptr: *const i8) -> String {
    let mut len = 0usize;
    // SAFETY: the caller guarantees a NUL-terminated string.
    unsafe {
        while *ptr.add(len) != 0 && len < 4096 {
            len += 1;
        }
        let slice = core::slice::from_raw_parts(ptr as *const u8, len);
        String::from_utf8_lossy(slice).into_owned()
    }
}
