//! The ABI's object model: what a handle *is*, how one is allocated and freed,
//! and the ref-counted lifetime rules a host drives through `reconlRetain` and
//! `reconlRelease`.
//!
//! Every object a host holds is one block from the host's own allocator whose
//! first member is a [`HandleHeader`]. That header is what makes the `void*`
//! calls work: `reconlRelease` reads the kind out of it and dispatches to the
//! right destructor without knowing any concrete type, and `child!` reads the
//! `device` back-pointer to prove a child belongs to the device it was passed
//! with, rather than misreading someone else's object.
//!
//! A child holds one implicit reference to its device, so a host that releases
//! its device while a buffer it made is still alive keeps the device alive too -
//! the alternative is a child whose destructor runs against freed memory.
//!
//! Field visibility: the child types below are declared here and *used* by the
//! modules that implement them (`lib.rs`, `order.rs`, ...), so their
//! fields are `pub(crate)`. What stays here is the part that must not be written
//! by hand: a header is stamped by [`header_of`], which takes the kind word from
//! the handle's own type, so a kind and a type cannot drift apart.

use crate::DeviceHandle;
use reconl_core::alloc::{HostAlloc, HostVec};
use reconl_core::error::{Code, Result};
use reconl_core::{err, Text};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Device = 1,
    Buffer = 2,
    Texture = 3,
    Pipeline = 4,
    Swapchain = 5,
    CommandList = 6,
    Fence = 7,
}

impl Kind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Kind::Device => "device",
            Kind::Buffer => "buffer",
            Kind::Texture => "texture",
            Kind::Pipeline => "pipeline",
            Kind::Swapchain => "swapchain",
            Kind::CommandList => "command list",
            Kind::Fence => "fence",
        }
    }

    pub(crate) fn from_u32(v: u32) -> Kind {
        match v {
            1 => Kind::Device,
            2 => Kind::Buffer,
            3 => Kind::Texture,
            4 => Kind::Pipeline,
            5 => Kind::Swapchain,
            6 => Kind::CommandList,
            7 => Kind::Fence,
            _ => Kind::Device,
        }
    }
}

/// First member of every handle. `void*`-typed ABI calls (`reconlRetain`,
/// `reconlRelease`) read this to find out what they were handed.
#[repr(C)]
pub struct HandleHeader {
    pub kind: u32,
    /// 1 for a live handle. The device holds one implicit reference per live
    /// child, so a child outliving the host's device reference keeps the device
    /// alive rather than dangling.
    pub refcount: AtomicU32,
    pub device: *mut DeviceHandle,
}

/// Anything with a [`HandleHeader`] first.
pub unsafe trait Handle {
    fn header(&self) -> &HandleHeader;
    fn kind() -> Kind;
}

macro_rules! impl_handle {
    ($t:ty, $kind:expr) => {
        unsafe impl $crate::handle::Handle for $t {
            fn header(&self) -> &$crate::handle::HandleHeader {
                &self.header
            }
            fn kind() -> $crate::handle::Kind {
                $kind
            }
        }
    };
}

// Declared here, used by every module that defines a handle type.
pub(crate) use impl_handle;

/// Allocates a handle with the host allocator. ReconL never uses the Rust global
/// allocator for objects the host owns.
pub(crate) unsafe fn handle_new<T: Handle>(alloc: HostAlloc, value: T) -> *mut T {
    match unsafe { alloc.alloc_bytes(core::mem::size_of::<T>(), core::mem::align_of::<T>().max(16)) } {
        Ok(p) => {
            let ptr = p.as_ptr() as *mut T;
            unsafe { ptr.write(value) };
            ptr
        }
        Err(_) => core::ptr::null_mut(),
    }
}

pub(crate) unsafe fn handle_free<T: Handle>(alloc: HostAlloc, ptr: *mut T) {
    if ptr.is_null() {
        return;
    }
    let bytes = core::mem::size_of::<T>();
    unsafe { core::ptr::drop_in_place(ptr) };
    let nonnull = unsafe { core::ptr::NonNull::new_unchecked(ptr as *mut u8) };
    unsafe { alloc.free_bytes(nonnull, bytes, core::mem::align_of::<T>().max(16)) };
}

/// Validates a handle pointer of a known kind and returns its header.
pub(crate) unsafe fn check_handle(ptr: *mut c_void, expected: Kind, what: &str) -> Result<*mut HandleHeader> {
    if ptr.is_null() {
        return err!(Code::InvalidArgument, "null {} handle", what);
    }
    let header = ptr as *mut HandleHeader;
    // SAFETY: every handle this library hands out begins with a HandleHeader.
    let kind = unsafe { (*header).kind };
    if kind != expected as u32 {
        return err!(
            Code::InvalidHandle,
            "{} handle was passed where a {} was expected",
            Kind::from_u32(kind).name(),
            what
        );
    }
    Ok(header)
}

/// The header a live handle of type `T` starts with: its kind, a reference of
/// one, and the device that owns it.
///
/// The kind comes from `T`, not from the call site, so the pair cannot get out
/// of step. The one handle whose `device` is null is the device itself: it owns
/// no other handle and is the end of every child's reference chain.
pub(crate) fn header_of<T: Handle>(device: *mut DeviceHandle) -> HandleHeader {
    HandleHeader { kind: T::kind() as u32, refcount: AtomicU32::new(1), device }
}

// -------------------------------------------------------------- child handles
//
// One block each, allocated and freed through `handle_new`/`handle_free` by the
// module that implements the object. The header is first, as it is for every
// handle, and `reconlRetain`/`reconlRelease` reach the `ram` reservations below
// only through the destructor this module dispatches to.

#[repr(C)]
pub struct BufferHandle {
    pub(crate) header: HandleHeader,
    pub(crate) bytes: HostVec<u8>,
    /// Held for the buffer's lifetime, so its bytes stay counted against the
    /// device budget until the host releases it.
    pub(crate) ram: reconl_core::budget::Reservation,
    pub(crate) usage: u32,
    pub(crate) name: Text<64>,
}
impl_handle!(BufferHandle, Kind::Buffer);

#[repr(C)]
pub struct TextureHandle {
    pub(crate) header: HandleHeader,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: u32,
    pub(crate) usage: u32,
    /// RGBA8 mip chain, tightly packed.
    pub(crate) levels: HostVec<HostVec<u8>>,
    pub(crate) level_sizes: HostVec<(u32, u32)>,
    /// The whole chain's bytes, counted against the device budget for as long
    /// as the texture lives.
    pub(crate) ram: reconl_core::budget::Reservation,
    pub(crate) name: Text<64>,
}
impl_handle!(TextureHandle, Kind::Texture);

#[repr(C)]
pub struct PipelineHandle {
    pub(crate) header: HandleHeader,
    pub(crate) shading: u32,
    pub(crate) blend: u32,
    pub(crate) cull: u32,
    pub(crate) depth_compare: u32,
    pub(crate) depth_write: bool,
    pub(crate) texture_slots: u32,
    pub(crate) receives_shadow: bool,
    pub(crate) casts_shadow: bool,
    pub(crate) two_sided_shadow: bool,
    pub(crate) name: Text<64>,
}
impl_handle!(PipelineHandle, Kind::Pipeline);

#[repr(C)]
pub struct SwapchainHandle {
    pub(crate) header: HandleHeader,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: u32,
    pub(crate) present_to_memory: bool,
    pub(crate) depth: bool,
}
impl_handle!(SwapchainHandle, Kind::Swapchain);

#[repr(C)]
pub struct FenceHandle {
    pub(crate) header: HandleHeader,
    pub(crate) signaled: bool,
    pub(crate) frame_index: u64,
}
impl_handle!(FenceHandle, Kind::Fence);

// ------------------------------------------------------------------- lifetime

#[no_mangle]
pub unsafe extern "C" fn reconlRetain(handle: *mut c_void) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let header = handle as *mut HandleHeader;
    unsafe { (*header).refcount.fetch_add(1, Ordering::Relaxed) + 1 }
}

#[no_mangle]
pub unsafe extern "C" fn reconlRelease(handle: *mut c_void) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let header_ptr = handle as *mut HandleHeader;
    let kind = unsafe { Kind::from_u32((*header_ptr).kind) };
    let previous = unsafe { (*header_ptr).refcount.fetch_sub(1, Ordering::AcqRel) };
    if previous > 1 {
        return previous - 1;
    }
    let device_ptr = unsafe { (*header_ptr).device };
    match kind {
        Kind::Device => {
            let device = handle as *mut DeviceHandle;
            let alloc = unsafe { (*device).alloc };
            unsafe { handle_free(alloc, device) };
        }
        Kind::Buffer => {
            let ptr = handle as *mut BufferHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
        Kind::Texture => {
            let ptr = handle as *mut TextureHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
        Kind::Pipeline => {
            let ptr = handle as *mut PipelineHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
        Kind::Swapchain => {
            let ptr = handle as *mut SwapchainHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
        Kind::CommandList => {
            let ptr = handle as *mut crate::CommandListHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
        Kind::Fence => {
            let ptr = handle as *mut FenceHandle;
            let alloc = unsafe { device_alloc(device_ptr) };
            unsafe { handle_free(alloc, ptr) };
            release_device_reference(device_ptr);
        }
    }
    0
}

pub(crate) unsafe fn device_alloc(device: *mut DeviceHandle) -> HostAlloc {
    if device.is_null() {
        return HostAlloc::system();
    }
    unsafe { (*device).alloc }
}

unsafe fn release_device_reference(device: *mut DeviceHandle) {
    if device.is_null() {
        return;
    }
    // SAFETY: the child held a reference, so the device is alive.
    unsafe { reconlRelease(device as *mut c_void) };
}
