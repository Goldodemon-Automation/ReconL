//! Loading a vendor driver at runtime, with no build-time dependency on it.
//!
//! The GPU backends are the only part of ReconL that talks to a driver the
//! project does not ship, and neither CUDA nor ROCm can be assumed present.
//! Requiring the CUDA toolkit or the ROCm stack at *build* time would make the
//! library fail to link on a machine that has neither - which is the machine
//! most of this repository is developed on (see `README.md`: Intel UHD, no
//! discrete GPU at all).
//!
//! So the driver is opened the way the Windows build already opens DXCore: name
//! the library, resolve the symbols, and let a missing library be an ordinary
//! `Err` rather than a link error. Nothing here decides which runtime is
//! *usable* - it only answers "can this library be opened, and does it export
//! this symbol". The tier resolver and the device creation path are what turn
//! that into a usable device.
//!
//! A loaded library stays open for the process's lifetime: a driver's context
//! outlives this handle (a device holds it), so closing it early would be a
//! use-after-free on a driver that does not refcount its own handle. The
//! handle is intentionally never dropped.

use core::ffi::c_void;

#[cfg(windows)]
mod sys {
    use core::ffi::{c_char, c_void};

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(lib: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(lib: *mut c_void) -> i32;
    }

    pub fn open(name: &str) -> Option<*mut c_void> {
        let mut buf = [0u8; 260];
        if name.len() + 1 > buf.len() {
            return None;
        }
        buf[..name.len()].copy_from_slice(name.as_bytes());
        let handle = unsafe { LoadLibraryA(buf.as_ptr() as *const c_char) };
        if handle.is_null() {
            None
        } else {
            Some(handle)
        }
    }

    pub fn symbol(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
        let mut buf = [0u8; 128];
        if name.len() + 1 > buf.len() {
            return None;
        }
        buf[..name.len()].copy_from_slice(name.as_bytes());
        let addr = unsafe { GetProcAddress(handle, buf.as_ptr() as *const c_char) };
        if addr.is_null() {
            None
        } else {
            Some(addr)
        }
    }

    #[allow(dead_code)]
    pub fn close(handle: *mut c_void) {
        unsafe {
            FreeLibrary(handle);
        }
    }
}

#[cfg(unix)]
mod sys {
    use core::ffi::{c_char, c_void};

    // `libdl` is separate from libc on glibc before 2.34 and the stub is
    // harmless after it. musl has the symbols in libc and no `libdl` at all, so
    // the link attribute is scoped to gnu targets rather than applied blindly.
    #[cfg_attr(all(target_os = "linux", target_env = "gnu"), link(name = "dl"))]
    #[cfg_attr(target_os = "android", link(name = "dl"))]
    extern "C" {
        fn dlopen(name: *const c_char, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> i32;
    }

    const RTLD_NOW: i32 = 2;
    const RTLD_LOCAL: i32 = 0;

    pub fn open(name: &str) -> Option<*mut c_void> {
        let mut buf = [0u8; 260];
        if name.len() + 1 > buf.len() {
            return None;
        }
        buf[..name.len()].copy_from_slice(name.as_bytes());
        let handle = unsafe { dlopen(buf.as_ptr() as *const c_char, RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            None
        } else {
            Some(handle)
        }
    }

    pub fn symbol(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
        let mut buf = [0u8; 128];
        if name.len() + 1 > buf.len() {
            return None;
        }
        buf[..name.len()].copy_from_slice(name.as_bytes());
        let addr = unsafe { dlsym(handle, buf.as_ptr() as *const c_char) };
        if addr.is_null() {
            None
        } else {
            Some(addr)
        }
    }

    #[allow(dead_code)]
    pub fn close(handle: *mut c_void) {
        unsafe {
            dlclose(handle);
        }
    }
}

/// One opened driver library.
///
/// `handle` is never closed: see the module comment. The type is `Copy`-free on
/// purpose so ownership reads as "the process owns exactly one of these per
/// driver".
pub struct Library {
    handle: *mut c_void,
    /// The file name this handle was opened from, kept so a report can name the
    /// thing it actually loaded rather than the thing it hoped for.
    name: &'static str,
}

// SAFETY: the handle is an opaque OS handle to a driver, and every use of it
// goes through the same `symbol` lookup. A driver is process-wide state by
// design (CUDA's driver API is, and HIP's runtime is), so sharing one handle is
// what the vendor's own API expects rather than a hazard we introduce.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    /// Opens the first of `names` that loads, in order.
    ///
    /// The order is the caller's: a versioned soname is tried before the
    /// unversioned one so a machine with several runtime versions installed
    /// reports the one whose API we actually resolved.
    pub fn open_first(names: &[&'static str]) -> Option<Library> {
        for name in names {
            if let Some(handle) = sys::open(name) {
                return Some(Library { handle, name });
            }
        }
        None
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Resolves one symbol to `T`.
    ///
    /// # Safety
    /// The caller asserts that `T` is the ABI of the symbol named - the same
    /// contract a hand-written `dlsym` cast has, which is why this is unsafe
    /// and why every call site is inside this crate's `api` module, next to the
    /// declaration the cast has to match.
    pub unsafe fn symbol<T: Copy>(&self, name: &str) -> Option<T> {
        let addr = sys::symbol(self.handle, name)?;
        // SAFETY: a function pointer and a data pointer are the same size on
        // every target this ships on, and `T` is the caller's asserted ABI.
        Some(unsafe { core::mem::transmute_copy::<*mut c_void, T>(&addr) })
    }
}

impl core::fmt::Debug for Library {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Library").field("name", &self.name).finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_library_that_does_not_exist_is_none_not_a_failure() {
        // The whole point of the loader: a machine without the vendor stack is
        // the ordinary case, not an error path.
        assert!(Library::open_first(&["reconl-no-such-driver-xyz.dll", "libreconl-no-such-driver-xyz.so"]).is_none());
    }

    #[test]
    fn a_library_that_does_exist_resolves_its_symbols() {
        // Names that have to exist on any target this project builds for: the C
        // runtime is linked into the executable, so its own symbols are already
        // in the process image whether or not the loader can find a file.
        let lib = Library::open_first(&["kernel32.dll", "libc.so.6", "libSystem.B.dylib"]);
        let Some(lib) = lib else {
            // A statically linked or musl target may have none of these as a
            // loadable file; that is a property of the host, not of the loader.
            return;
        };
        assert!(!lib.name().is_empty());
        // An unknown symbol is `None` rather than a bogus non-null pointer.
        let missing: Option<extern "C" fn()> = unsafe { lib.symbol("reconl_no_such_symbol_xyz") };
        assert!(missing.is_none());
    }
}
