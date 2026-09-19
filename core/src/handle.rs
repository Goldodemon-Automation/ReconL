//! Handles: opaque, ref-counted, type-tagged, allocated from the host.
//!
//! A handle's first three words are always the same header
//! (`magic`, `kind`, `refs`) so that `reconlRetain`/`reconlRelease` can be
//! called with a `void*` of unknown type, validate it, and dispatch to the right
//! destructor without the core knowing any concrete type.

use crate::alloc::{HostAlloc, HostBox};
use crate::error::{Code, Error, Result};
use core::sync::atomic::{AtomicU32, Ordering};

/// "RCLH".
pub const HANDLE_MAGIC: u32 = 0x5243_4C48;
/// Written over `magic` when a handle is destroyed, so a stale pointer is caught
/// instead of silently dereferencing freed memory.
pub const HANDLE_DEAD: u32 = 0x4445_4144;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HandleKind {
    Device = 1,
    Buffer = 2,
    Texture = 3,
    Pipeline = 4,
    Swapchain = 5,
    CommandList = 6,
    Fence = 7,
}

impl HandleKind {
    pub fn from_u32(v: u32) -> Option<HandleKind> {
        match v {
            1 => Some(HandleKind::Device),
            2 => Some(HandleKind::Buffer),
            3 => Some(HandleKind::Texture),
            4 => Some(HandleKind::Pipeline),
            5 => Some(HandleKind::Swapchain),
            6 => Some(HandleKind::CommandList),
            7 => Some(HandleKind::Fence),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            HandleKind::Device => "device",
            HandleKind::Buffer => "buffer",
            HandleKind::Texture => "texture",
            HandleKind::Pipeline => "pipeline",
            HandleKind::Swapchain => "swapchain",
            HandleKind::CommandList => "command list",
            HandleKind::Fence => "fence",
        }
    }
}

#[repr(C)]
pub struct HandleHeader {
    pub magic: AtomicU32,
    pub kind: u32,
    pub refs: AtomicU32,
}

impl HandleHeader {
    pub fn new(kind: HandleKind) -> Self {
        Self {
            magic: AtomicU32::new(HANDLE_MAGIC),
            kind: kind as u32,
            refs: AtomicU32::new(1),
        }
    }

    /// Validates magic and kind. Never panics: a bad handle is a returned error.
    pub fn validate(&self, expected: HandleKind) -> Result<()> {
        let magic = self.magic.load(Ordering::Acquire);
        if magic == HANDLE_DEAD {
            return crate::err!(Code::InvalidHandle, "use after release");
        }
        if magic != HANDLE_MAGIC {
            return crate::err!(Code::InvalidHandle, "not a ReconL handle");
        }
        if self.kind != expected as u32 {
            return crate::err!(
                Code::InvalidHandle,
                "expected a {}, got a {}",
                expected.name(),
                HandleKind::from_u32(self.kind).map(|k| k.name()).unwrap_or("unknown handle")
            );
        }
        if self.refs.load(Ordering::Acquire) == 0 {
            return crate::err!(Code::InvalidHandle, "handle has no live references");
        }
        Ok(())
    }

    pub fn retain(&self) -> u32 {
        self.refs.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Returns the count after the decrement. `0` means the final reference was
    /// just dropped and the caller must destroy the object.
    pub fn release(&self) -> u32 {
        let prev = self.refs.fetch_sub(1, Ordering::AcqRel);
        prev.saturating_sub(1)
    }

    pub fn refs(&self) -> u32 {
        self.refs.load(Ordering::Acquire)
    }

    pub fn mark_dead(&self) {
        self.magic.store(HANDLE_DEAD, Ordering::Release);
    }
}

/// A ref-counted object: a validated header followed by the backend's state.
#[repr(C)]
pub struct RefHandle<T> {
    pub header: HandleHeader,
    pub inner: T,
}

impl<T> RefHandle<T> {
    /// Allocates a handle from the host allocator and returns the opaque pointer.
    pub fn create(alloc: HostAlloc, kind: HandleKind, inner: T) -> Result<*mut Self> {
        let boxed = HostBox::new(alloc, RefHandle { header: HandleHeader::new(kind), inner })?;
        Ok(boxed.into_raw())
    }

    /// # Safety
    /// `ptr` must be a live handle of `kind`, and the returned reference must not
    /// outlive the ABI call. Handles are single-threaded-on-a-context in v0.1
    /// (docs/abi.md, "Threading"), so no two `&mut` to the same handle exist.
    pub unsafe fn as_ref<'a>(ptr: *mut Self, kind: HandleKind) -> Result<&'a mut Self> {
        let h = unsafe { Self::peek(ptr, kind)? };
        Ok(unsafe { &mut *h })
    }

    /// # Safety
    /// As [`RefHandle::as_ref`], but for read-only access.
    pub unsafe fn peek<'a>(ptr: *mut Self, kind: HandleKind) -> Result<*mut Self> {
        if ptr.is_null() {
            return crate::err!(Code::InvalidHandle, "null {} handle", kind.name());
        }
        // SAFETY: the caller guarantees `ptr` is a live ReconL handle; we only
        // read the header, which every live handle begins with.
        let header = unsafe { &(*ptr).header };
        header.validate(kind)?;
        Ok(ptr)
    }

    /// Decrements the ref count and destroys the object when it reaches zero.
    ///
    /// # Safety
    /// `ptr` must be a live handle of `kind` created by [`RefHandle::create`].
    pub unsafe fn release(ptr: *mut Self, kind: HandleKind, alloc: HostAlloc) -> u32 {
        if ptr.is_null() {
            return 0;
        }
        // SAFETY: live handle per the caller's contract.
        let header = unsafe { &(*ptr).header };
        if header.validate(kind).is_err() {
            return 0;
        }
        let remaining = header.release();
        if remaining == 0 {
            header.mark_dead();
            // SAFETY: the last reference is gone, so the object can be dropped
            // through the same allocator that produced it.
            unsafe {
                if let Some(b) = HostBox::<Self>::from_raw(alloc, ptr) {
                    drop(b);
                }
            }
        }
        remaining
    }

    /// Retains and returns the new count.
    ///
    /// # Safety
    /// `ptr` must be a live handle.
    pub unsafe fn retain(ptr: *mut Self, kind: HandleKind) -> Result<u32> {
        if ptr.is_null() {
            return crate::err!(Code::InvalidHandle, "null {} handle", kind.name());
        }
        // SAFETY: live handle per the caller's contract.
        let header = unsafe { &(*ptr).header };
        header.validate(kind)?;
        Ok(header.retain())
    }
}

/// Reads the kind word out of any live handle, for `reconlRetain`/`reconlRelease`.
///
/// # Safety
/// `ptr` must be a live ReconL handle (or a dead one, whose magic we detect).
pub unsafe fn kind_of(ptr: *const core::ffi::c_void) -> Result<HandleKind> {
    if ptr.is_null() {
        return Err(Error::new(Code::InvalidHandle, "null handle"));
    }
    // SAFETY: the caller guarantees a ReconL handle; we read only the header.
    let header = unsafe { &*(ptr as *const HandleHeader) };
    if header.magic.load(Ordering::Acquire) == HANDLE_DEAD {
        return Err(Error::new(Code::InvalidHandle, "use after release"));
    }
    if header.magic.load(Ordering::Acquire) != HANDLE_MAGIC {
        return Err(Error::new(Code::InvalidHandle, "not a ReconL handle"));
    }
    HandleKind::from_u32(header.kind).ok_or_else(|| Error::new(Code::InvalidHandle, "unknown handle kind"))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Thing {
        value: u32,
    }

    fn expect_err<T>(r: Result<T>) -> Error {
        match r {
            Err(e) => e,
            Ok(_) => panic!("expected an error"),
        }
    }

    #[test]
    fn refcount_lifecycle_frees_on_last_release() {
        let _g = crate::alloc::test_guard();
        crate::alloc::reset_host_stats();
        let alloc = HostAlloc::system();
        let ptr = RefHandle::create(alloc, HandleKind::Buffer, Thing { value: 7 }).unwrap();
        let h = unsafe { RefHandle::as_ref(ptr, HandleKind::Buffer) }.unwrap();
        assert_eq!(h.inner.value, 7);
        assert_eq!(unsafe { RefHandle::retain(ptr, HandleKind::Buffer) }.unwrap(), 2);
        assert_eq!(unsafe { RefHandle::release(ptr, HandleKind::Buffer, alloc) }, 1);
        assert_eq!(crate::alloc::host_stats().free_calls, 0, "still one reference alive");
        assert_eq!(unsafe { RefHandle::release(ptr, HandleKind::Buffer, alloc) }, 0);
        assert_eq!(crate::alloc::host_stats().live_bytes, 0);
    }

    #[test]
    fn wrong_kind_is_an_error_not_a_crash() {
        let _g = crate::alloc::test_guard();
        let alloc = HostAlloc::system();
        let ptr = RefHandle::create(alloc, HandleKind::Buffer, Thing { value: 1 }).unwrap();
        let err = expect_err(unsafe { RefHandle::as_ref(ptr, HandleKind::Texture) });
        assert_eq!(err.code, Code::InvalidHandle);
        assert!(err.message_str().contains("texture"));
        unsafe { RefHandle::release(ptr, HandleKind::Buffer, alloc) };
    }

    #[test]
    fn use_after_release_is_caught_by_the_dead_magic() {
        let _g = crate::alloc::test_guard();
        let alloc = HostAlloc::system();
        let ptr = RefHandle::create(alloc, HandleKind::Fence, Thing { value: 1 }).unwrap();
        unsafe { RefHandle::release(ptr, HandleKind::Fence, alloc) };
        let err = expect_err(unsafe { RefHandle::as_ref(ptr, HandleKind::Fence) });
        assert_eq!(err.message_str(), "use after release");
    }

    #[test]
    fn null_and_garbage_are_reported() {
        assert_eq!(expect_err(unsafe { kind_of(core::ptr::null()) }).code, Code::InvalidHandle);
        let x = 0u64;
        let e = expect_err(unsafe { kind_of(&x as *const u64 as *const core::ffi::c_void) });
        assert_eq!(e.code, Code::InvalidHandle);
    }
}
