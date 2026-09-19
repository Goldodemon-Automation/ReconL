//! Host allocator plumbing.
//!
//! ReconL never calls the C runtime allocator behind the host's back. The host
//! passes a [`HostAllocatorA`] at device creation and every byte ReconL owns for
//! bulk storage (buffers, textures, tiles, shadow maps, command lists, spill
//! blocks) comes from it, through [`HostVec`] and [`HostBox`].
//!
//! Two counters exist so the promise is measurable rather than asserted:
//! [`HostStats`] counts calls and bytes that went to the host allocator, and the
//! frame-loop test asserts it stops moving once the frame is running.
//!
//! Alignment contract: `alloc`/`realloc` are called with a power-of-two
//! alignment in `[16, 64]`. An allocator that only handles `malloc` alignment
//! fails the device-creation self-check (see [`HostAlloc::self_check`]).

use crate::error::{Code, Error, Result};
use core::ffi::c_void;
use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU64, Ordering};

/// The ABI allocator, byte-for-byte the layout of `ReconLAllocator`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HostAllocatorA {
    pub alloc: Option<unsafe extern "C" fn(user: *mut c_void, size: usize, alignment: usize) -> *mut c_void>,
    pub realloc: Option<unsafe extern "C" fn(user: *mut c_void, ptr: *mut c_void, old_size: usize, new_size: usize, alignment: usize) -> *mut c_void>,
    pub free: Option<unsafe extern "C" fn(user: *mut c_void, ptr: *mut c_void, size: usize)>,
    pub user: *mut c_void,
}

impl HostAllocatorA {
    pub const fn zeroed() -> Self {
        Self { alloc: None, realloc: None, free: None, user: core::ptr::null_mut() }
    }

    pub fn is_zeroed(&self) -> bool {
        self.alloc.is_none() || self.free.is_none()
    }
}

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static FREE_CALLS: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct HostStats {
    pub alloc_calls: u64,
    pub alloc_bytes: u64,
    pub free_calls: u64,
    pub live_bytes: u64,
    pub peak_bytes: u64,
}

pub fn host_stats() -> HostStats {
    HostStats {
        alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        free_calls: FREE_CALLS.load(Ordering::Relaxed),
        live_bytes: LIVE_BYTES.load(Ordering::Relaxed),
        peak_bytes: PEAK_BYTES.load(Ordering::Relaxed),
    }
}

pub fn reset_host_stats() {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    FREE_CALLS.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_BYTES.store(0, Ordering::Relaxed);
}

fn note_alloc(bytes: u64) {
    ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
    ALLOC_BYTES.fetch_add(bytes, Ordering::Relaxed);
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
    while live > peak {
        match PEAK_BYTES.compare_exchange_weak(peak, live, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(now) => peak = now,
        }
    }
}

fn note_free(bytes: u64) {
    FREE_CALLS.fetch_add(1, Ordering::Relaxed);
    LIVE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

/// A copyable handle to the host's allocator.
#[derive(Clone, Copy)]
pub struct HostAlloc {
    a: HostAllocatorA,
}

impl HostAlloc {
    pub const fn zeroed() -> Self {
        Self { a: HostAllocatorA::zeroed() }
    }

    pub fn from_abi(a: HostAllocatorA) -> Self {
        Self { a }
    }

    pub fn abi(&self) -> &HostAllocatorA {
        &self.a
    }

    pub fn is_zeroed(&self) -> bool {
        self.a.is_zeroed()
    }

    /// A `malloc`-backed allocator, for tools, tests and examples that are the
    /// host. A library never installs this on the host's behalf.
    ///
    /// It is deliberately *not* `std::alloc`: the ABI's `free` is told the size
    /// but not the alignment, exactly like `free(3)`, so a conforming allocator
    /// either tracks the real base pointer itself or cannot free an over-aligned
    /// block at all. This one stores the base pointer in a header slot, which is
    /// the smallest thing that satisfies the contract - and it is why the
    /// alignment self-check at device creation is not a formality.
    pub fn system() -> Self {
        extern "C" {
            fn malloc(size: usize) -> *mut c_void;
            fn free(ptr: *mut c_void);
        }

        unsafe extern "C" fn sys_alloc(_u: *mut c_void, size: usize, alignment: usize) -> *mut c_void {
            let align = alignment.clamp(16, 4096);
            let total = match size.max(1).checked_add(align).and_then(|v| v.checked_add(16)) {
                Some(t) => t,
                None => return core::ptr::null_mut(),
            };
            // SAFETY: malloc is called with a checked size.
            let raw = unsafe { malloc(total) };
            if raw.is_null() {
                return core::ptr::null_mut();
            }
            let base = raw as usize;
            let aligned = (base + 16 + align - 1) & !(align - 1);
            // SAFETY: `aligned - 8` is inside the block (16 bytes of slack).
            unsafe { ((aligned - 8) as *mut usize).write_unaligned(base) };
            aligned as *mut c_void
        }

        unsafe extern "C" fn sys_realloc(
            _u: *mut c_void,
            ptr: *mut c_void,
            old: usize,
            new: usize,
            alignment: usize,
        ) -> *mut c_void {
            // SAFETY: `ptr` came from sys_alloc with this alignment, per the ABI contract.
            let fresh = unsafe { sys_alloc(core::ptr::null_mut(), new, alignment) };
            if !ptr.is_null() && !fresh.is_null() {
                let copy = old.min(new);
                // SAFETY: both blocks are at least `copy` bytes and do not overlap.
                unsafe { core::ptr::copy_nonoverlapping(ptr as *const u8, fresh as *mut u8, copy) };
                // SAFETY: `ptr` came from sys_alloc.
                unsafe { sys_free(core::ptr::null_mut(), ptr, old) };
            }
            fresh
        }

        unsafe extern "C" fn sys_free(_u: *mut c_void, ptr: *mut c_void, _size: usize) {
            if ptr.is_null() {
                return;
            }
            // SAFETY: sys_alloc wrote the malloc base pointer into the 8 bytes
            // immediately before the pointer it handed out.
            let base = unsafe { ((ptr as usize - 8) as *const usize).read_unaligned() } as *mut c_void;
            // SAFETY: `base` is what malloc returned for this block.
            unsafe { free(base) };
        }
        Self {
            a: HostAllocatorA {
                alloc: Some(sys_alloc),
                realloc: Some(sys_realloc),
                free: Some(sys_free),
                user: core::ptr::null_mut(),
            },
        }
    }

    /// Proves the allocator answers a 16-, 32- and 64-byte aligned request and
    /// that `free` accepts what `alloc` returned. Called once at device creation
    /// so a non-conforming allocator is reported, not discovered later.
    pub fn self_check(&self) -> Result<()> {
        if self.is_zeroed() {
            return Err(Error::new(Code::InvalidArgument, "no allocator supplied; ReconL will not use the C runtime allocator"));
        }
        for align in [16usize, 32, 64] {
            let p = unsafe { self.alloc_bytes(64, align) }?;
            if p.as_ptr() as usize % align != 0 {
                unsafe { self.free_bytes(p, 64, align) };
                return Err(Error::new(Code::InvalidArgument, "allocator returned a misaligned pointer"));
            }
            unsafe { p.as_ptr().write(0xAB) };
            unsafe { self.free_bytes(p, 64, align) };
        }
        if self.a.realloc.is_none() {
            return Err(Error::new(Code::InvalidArgument, "allocator has no realloc"));
        }
        Ok(())
    }

    /// # Safety
    /// `alignment` must be a power of two in `[16, 64]`.
    pub unsafe fn alloc_bytes(&self, size: usize, alignment: usize) -> Result<NonNull<u8>> {
        let f = match self.a.alloc {
            Some(f) => f,
            None => return Err(Error::new(Code::InvalidArgument, "allocator has no alloc")),
        };
        let bytes = size.max(1);
        let p = unsafe { f(self.a.user, bytes, alignment) } as *mut u8;
        match NonNull::new(p) {
            Some(p) => {
                note_alloc(bytes as u64);
                Ok(p)
            }
            None => Err(Error::new(Code::OutOfMemory, "host allocator returned null")),
        }
    }

    /// # Safety
    /// `ptr` must come from this allocator's `alloc`/`realloc` with `old_size`
    /// and `alignment`; `alignment` must be a power of two in `[16, 64]`.
    pub unsafe fn realloc_bytes(
        &self,
        ptr: NonNull<u8>,
        old_size: usize,
        new_size: usize,
        alignment: usize,
    ) -> Result<NonNull<u8>> {
        let f = match self.a.realloc {
            Some(f) => f,
            None => return Err(Error::new(Code::InvalidArgument, "allocator has no realloc")),
        };
        let p = unsafe { f(self.a.user, ptr.as_ptr() as *mut c_void, old_size.max(1), new_size.max(1), alignment) } as *mut u8;
        match NonNull::new(p) {
            Some(p) => {
                if new_size > old_size {
                    note_alloc((new_size - old_size) as u64);
                } else {
                    note_free((old_size - new_size) as u64);
                }
                Ok(p)
            }
            None => Err(Error::new(Code::OutOfMemory, "host allocator realloc returned null")),
        }
    }

    /// # Safety
    /// `ptr` must have come from this allocator, with the same `size`.
    pub unsafe fn free_bytes(&self, ptr: NonNull<u8>, size: usize, _alignment: usize) {
        if let Some(f) = self.a.free {
            unsafe { f(self.a.user, ptr.as_ptr() as *mut c_void, size.max(1)) };
            note_free(size.max(1) as u64);
        }
    }
}

fn align_for<T>() -> usize {
    align_of::<T>().clamp(16, 64).next_power_of_two()
}

/// A growable array whose storage is the host allocator's.
///
/// Growth returns `Err` instead of aborting: inside a frame, a failed push is a
/// tier-relevant event, not a crash.
pub struct HostVec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
    alloc: HostAlloc,
}

impl<T> HostVec<T> {
    pub const fn new(alloc: HostAlloc) -> Self {
        Self { ptr: core::ptr::null_mut(), len: 0, cap: 0, alloc }
    }

    pub fn with_capacity(alloc: HostAlloc, n: usize) -> Result<Self> {
        let mut v = Self::new(alloc);
        v.try_reserve(n)?;
        Ok(v)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn alloc(&self) -> HostAlloc {
        self.alloc
    }

    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: `ptr` has room for `len` initialised elements.
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: as above, and `&mut self` makes the borrow exclusive.
            unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.as_mut_slice().get_mut(index)
    }

    pub fn clear(&mut self) {
        // SAFETY: elements are initialised; dropping them in place is correct.
        unsafe { core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(self.ptr, self.len)) };
        self.len = 0;
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<()> {
        if size_of::<T>() == 0 {
            self.cap = usize::MAX;
            self.len = self.len.saturating_add(additional);
            return Ok(());
        }
        let needed = self.len.checked_add(additional).ok_or_else(|| {
            Error::new(Code::OutOfMemory, "host vector length overflow")
        })?;
        if needed <= self.cap {
            return Ok(());
        }
        let new_cap = needed.max(self.cap.saturating_mul(2)).max(8);
        let align = align_for::<T>();
        let bytes = new_cap.checked_mul(size_of::<T>()).ok_or_else(|| {
            Error::new(Code::OutOfMemory, "host vector capacity overflow")
        })?;
        let new_ptr = if self.ptr.is_null() {
            // SAFETY: alignment is a power of two in [16, 64].
            unsafe { self.alloc.alloc_bytes(bytes, align) }?.as_ptr() as *mut T
        } else {
            let old_bytes = self.cap * size_of::<T>();
            let old = NonNull::new(self.ptr as *mut u8).expect("non-null ptr with non-zero cap");
            // SAFETY: ptr/cap came from this allocator with this alignment.
            unsafe { self.alloc.realloc_bytes(old, old_bytes, bytes, align) }?.as_ptr() as *mut T
        };
        self.ptr = new_ptr;
        self.cap = new_cap;
        Ok(())
    }

    pub fn push(&mut self, value: T) -> Result<()> {
        if size_of::<T>() == 0 {
            self.len = self.len.saturating_add(1);
            return Ok(());
        }
        self.try_reserve(1)?;
        // SAFETY: capacity was ensured above; this slot is uninitialised.
        unsafe {
            self.ptr.add(self.len).write(value);
        }
        self.len += 1;
        Ok(())
    }

    /// `push`, returning the element just written.
    pub fn push_get(&mut self, value: T) -> Result<&mut T> {
        self.push(value)?;
        let idx = self.len - 1;
        Ok(&mut self.as_mut_slice()[idx])
    }

    pub fn extend_from_slice(&mut self, values: &[T]) -> Result<()>
    where
        T: Copy,
    {
        if values.is_empty() {
            return Ok(());
        }
        self.try_reserve(values.len())?;
        // SAFETY: capacity was ensured; src and dst do not overlap because the
        // source is an independent slice.
        unsafe {
            core::ptr::copy_nonoverlapping(values.as_ptr(), self.ptr.add(self.len), values.len());
        }
        self.len += values.len();
        Ok(())
    }

    pub fn resize_with(&mut self, new_len: usize, mut f: impl FnMut() -> T) -> Result<()> {
        if new_len < self.len {
            // SAFETY: dropping the tail elements, which are initialised.
            unsafe {
                core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                    self.ptr.add(new_len),
                    self.len - new_len,
                ))
            };
            self.len = new_len;
            return Ok(());
        }
        self.try_reserve(new_len - self.len)?;
        while self.len < new_len {
            // SAFETY: capacity was ensured for `new_len`.
            unsafe {
                self.ptr.add(self.len).write(f());
            }
            self.len += 1;
        }
        Ok(())
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let mut write = 0usize;
        for read in 0..self.len {
            // SAFETY: indices are in bounds and elements are initialised.
            unsafe {
                if keep(&*self.ptr.add(read)) {
                    if read != write {
                        core::ptr::copy_nonoverlapping(self.ptr.add(read), self.ptr.add(write), 1);
                    }
                    write += 1;
                } else {
                    core::ptr::drop_in_place(self.ptr.add(read));
                }
            }
        }
        self.len = write;
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T> core::ops::Index<usize> for HostVec<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.as_slice()[index]
    }
}

impl<T> core::ops::IndexMut<usize> for HostVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.as_mut_slice()[index]
    }
}

impl<T> HostVec<T> {
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T> Drop for HostVec<T> {
    fn drop(&mut self) {
        self.clear();
        if !self.ptr.is_null() && size_of::<T>() != 0 {
            let bytes = self.cap * size_of::<T>();
            let align = align_for::<T>();
            let nn = NonNull::new(self.ptr as *mut u8).expect("non-null ptr checked above");
            // SAFETY: ptr/cap came from this allocator with the same alignment.
            unsafe { self.alloc.free_bytes(nn, bytes, align) };
        }
    }
}

/// Single-value host allocation: `Box`, but on the host's allocator.
pub struct HostBox<T> {
    ptr: NonNull<T>,
    alloc: HostAlloc,
}

impl<T> HostBox<T> {
    pub fn new(alloc: HostAlloc, value: T) -> Result<Self> {
        let align = align_for::<T>();
        // SAFETY: alignment is a power of two in [16, 64].
        let raw = unsafe { alloc.alloc_bytes(size_of::<T>().max(1), align) }?;
        let ptr = raw.as_ptr() as *mut T;
        // SAFETY: fresh allocation of at least size_of::<T>() bytes, aligned.
        unsafe { ptr.write(value) };
        Ok(Self { ptr: NonNull::new(ptr).expect("non-null"), alloc })
    }

    pub fn as_ref(&self) -> &T {
        // SAFETY: the box owns an initialised T.
        unsafe { self.ptr.as_ref() }
    }

    pub fn as_mut(&mut self) -> &mut T {
        // SAFETY: as above; `&mut self` gives exclusive access.
        unsafe { self.ptr.as_mut() }
    }

    pub fn into_raw(self) -> *mut T {
        let p = self.ptr.as_ptr();
        core::mem::forget(self);
        p
    }

    /// # Safety
    /// `ptr` must have come from `HostBox::into_raw` with the same `alloc`.
    pub unsafe fn from_raw(alloc: HostAlloc, ptr: *mut T) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self { ptr, alloc })
    }
}

impl<T> Drop for HostBox<T> {
    fn drop(&mut self) {
        let align = align_for::<T>();
        // SAFETY: the box owns an initialised T; drop it, then hand the storage back.
        unsafe {
            core::ptr::drop_in_place(self.ptr.as_ptr());
            self.alloc.free_bytes(
                NonNull::new_unchecked(self.ptr.as_ptr() as *mut u8),
                size_of::<T>().max(1),
                align,
            );
        }
    }
}

/// Serialises tests that read the process-wide allocation counters.
///
/// The counters are global on purpose (the host wants one number for "what did
/// ReconL allocate"), so tests that assert on them must not run concurrently with
/// other tests that allocate.
#[cfg(test)]
pub fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    match GUARD.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_allocator_self_check_passes() {
        assert!(HostAlloc::system().self_check().is_ok());
    }

    #[test]
    fn zeroed_allocator_is_refused() {
        let err = HostAlloc::zeroed().self_check().unwrap_err();
        assert_eq!(err.code, Code::InvalidArgument);
    }

    #[test]
    fn host_vec_grows_through_the_host_allocator_only() {
        let _g = test_guard();
        reset_host_stats();
        let alloc = HostAlloc::system();
        let mut v: HostVec<u32> = HostVec::with_capacity(alloc, 4).unwrap();
        for i in 0..1000u32 {
            v.push(i).unwrap();
        }
        assert_eq!(v.len(), 1000);
        assert_eq!(v.as_slice()[999], 999);
        let stats = host_stats();
        assert!(stats.alloc_calls >= 1);
        assert!(stats.alloc_bytes >= 4000);
        drop(v);
        assert_eq!(host_stats().live_bytes, 0);
    }

    #[test]
    fn host_box_round_trips_and_frees() {
        let _g = test_guard();
        reset_host_stats();
        let alloc = HostAlloc::system();
        let b = HostBox::new(alloc, 0x1234_5678u64).unwrap();
        assert_eq!(*b.as_ref(), 0x1234_5678);
        let raw = b.into_raw();
        let b = unsafe { HostBox::from_raw(alloc, raw) }.unwrap();
        drop(b);
        assert_eq!(host_stats().live_bytes, 0);
        assert_eq!(host_stats().free_calls, 1);
    }

    #[test]
    fn retain_drops_and_compacts() {
        let _g = test_guard();
        let alloc = HostAlloc::system();
        let mut v: HostVec<u32> = HostVec::with_capacity(alloc, 8).unwrap();
        for i in 0..10u32 {
            v.push(i).unwrap();
        }
        v.retain(|x| x % 2 == 0);
        assert_eq!(v.as_slice(), &[0, 2, 4, 6, 8]);
    }

    #[test]
    fn host_vec_indexing_and_iteration() {
        let _g = test_guard();
        let alloc = HostAlloc::system();
        let mut v: HostVec<u32> = HostVec::with_capacity(alloc, 4).unwrap();
        for i in 0..5u32 {
            v.push(i).unwrap();
        }
        assert_eq!(v[3], 3);
        v[3] = 30;
        assert_eq!(v[3], 30);
        assert_eq!(v.iter().copied().max(), Some(30));
        for x in v.iter_mut() {
            *x += 1;
        }
        assert_eq!(v.as_slice(), &[1, 2, 3, 31, 5]);
    }
}
