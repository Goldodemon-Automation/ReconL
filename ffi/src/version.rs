//! What the library calls itself and its own values: version numbers, the
//! spelling of every code, tier and backend name, and the log controls.
//!
//! Nothing here touches a device, and a host calls these to *find out* what it
//! is linked against, so they are plain `extern "C"` functions with no entry
//! wrapper and no error path - a pointer into a static string is the whole
//! contract. `*_name` for an unknown value returns `"unknown"` rather than
//! nothing, so a host can always print what it was handed.

use crate::abi;
use reconl_core::log::{self, Level};
use reconl_core::tier::TIER_COUNT;
use std::ffi::c_void;

#[no_mangle]
pub extern "C" fn reconlVersion(major: *mut u32, minor: *mut u32, patch: *mut u32) {
    unsafe {
        if !major.is_null() {
            *major = abi::VERSION_MAJOR;
        }
        if !minor.is_null() {
            *minor = abi::VERSION_MINOR;
        }
        if !patch.is_null() {
            *patch = abi::VERSION_PATCH;
        }
    }
}

const VERSION_STRING: &[u8] = b"0.1.0\0";

#[no_mangle]
pub extern "C" fn reconlVersionString() -> *const i8 {
    VERSION_STRING.as_ptr() as *const i8
}

const UNKNOWN_NAME: &[u8] = b"unknown\0";
const RESULT_OK: &[u8] = b"RECONL_OK\0";
const RESULT_INVALID_ARGUMENT: &[u8] = b"RECONL_ERR_INVALID_ARGUMENT\0";
const RESULT_OUT_OF_MEMORY: &[u8] = b"RECONL_ERR_OUT_OF_MEMORY\0";
const RESULT_NOT_SUPPORTED: &[u8] = b"RECONL_ERR_NOT_SUPPORTED\0";
const RESULT_BACKEND_UNAVAILABLE: &[u8] = b"RECONL_ERR_BACKEND_UNAVAILABLE\0";
const RESULT_BUDGET_EXCEEDED: &[u8] = b"RECONL_ERR_BUDGET_EXCEEDED\0";
const RESULT_DEVICE_LOST: &[u8] = b"RECONL_ERR_DEVICE_LOST\0";
const RESULT_INVALID_HANDLE: &[u8] = b"RECONL_ERR_INVALID_HANDLE\0";
const RESULT_STRUCT_SIZE: &[u8] = b"RECONL_ERR_STRUCT_SIZE\0";
const RESULT_WRONG_STRUCT_TYPE: &[u8] = b"RECONL_ERR_WRONG_STRUCT_TYPE\0";
const RESULT_ABI_VERSION: &[u8] = b"RECONL_ERR_ABI_VERSION\0";
const RESULT_FRAME_IN_PROGRESS: &[u8] = b"RECONL_ERR_FRAME_IN_PROGRESS\0";
const RESULT_NO_FRAME: &[u8] = b"RECONL_ERR_NO_FRAME\0";
const RESULT_NOT_READY: &[u8] = b"RECONL_ERR_NOT_READY\0";
const RESULT_IO: &[u8] = b"RECONL_ERR_IO\0";
const RESULT_CORRUPT_CACHE: &[u8] = b"RECONL_ERR_CORRUPT_CACHE\0";
const RESULT_DEGRADED: &[u8] = b"RECONL_ERR_DEGRADED\0";
const RESULT_PANIC: &[u8] = b"RECONL_ERR_PANIC\0";
const RESULT_EMPTY_FRAME: &[u8] = b"RECONL_ERR_EMPTY_FRAME\0";

#[no_mangle]
pub extern "C" fn reconlResultName(r: i32) -> *const i8 {
    let name: &[u8] = match r {
        abi::result::OK => RESULT_OK,
        abi::result::INVALID_ARGUMENT => RESULT_INVALID_ARGUMENT,
        abi::result::OUT_OF_MEMORY => RESULT_OUT_OF_MEMORY,
        abi::result::NOT_SUPPORTED => RESULT_NOT_SUPPORTED,
        abi::result::BACKEND_UNAVAILABLE => RESULT_BACKEND_UNAVAILABLE,
        abi::result::BUDGET_EXCEEDED => RESULT_BUDGET_EXCEEDED,
        abi::result::DEVICE_LOST => RESULT_DEVICE_LOST,
        abi::result::INVALID_HANDLE => RESULT_INVALID_HANDLE,
        abi::result::STRUCT_SIZE => RESULT_STRUCT_SIZE,
        abi::result::WRONG_STRUCT_TYPE => RESULT_WRONG_STRUCT_TYPE,
        abi::result::ABI_VERSION => RESULT_ABI_VERSION,
        abi::result::FRAME_IN_PROGRESS => RESULT_FRAME_IN_PROGRESS,
        abi::result::NO_FRAME => RESULT_NO_FRAME,
        abi::result::NOT_READY => RESULT_NOT_READY,
        abi::result::IO => RESULT_IO,
        abi::result::CORRUPT_CACHE => RESULT_CORRUPT_CACHE,
        abi::result::DEGRADED => RESULT_DEGRADED,
        abi::result::PANIC => RESULT_PANIC,
        abi::result::EMPTY_FRAME => RESULT_EMPTY_FRAME,
        _ => UNKNOWN_NAME,
    };
    name.as_ptr() as *const i8
}

#[no_mangle]
pub extern "C" fn reconlTierName(tier: u32) -> *const i8 {
    static NAMES: [&[u8]; 5] = [
        b"T0/gpu-discrete\0",
        b"T1/gpu-shared\0",
        b"T2/cpu-ram\0",
        b"T3/cpu-thrifty\0",
        b"T4/out-of-core\0",
    ];
    let index = (tier as usize).min(TIER_COUNT - 1);
    NAMES[index].as_ptr() as *const i8
}

#[no_mangle]
pub extern "C" fn reconlBackendName(backend: u32) -> *const i8 {
    let name: &[u8] = match backend {
        1 => b"soft-cpu\0",
        2 => b"null\0",
        3 => b"d3d11\0",
        4 => b"d3d12\0",
        5 => b"vulkan\0",
        6 => b"gl\0",
        7 => b"metal\0",
        8 => b"webgpu\0",
        9 => b"wasm-webgl2\0",
        _ => b"none\0",
    };
    name.as_ptr() as *const i8
}

#[no_mangle]
pub extern "C" fn reconlSetLogLevel(level: u32) {
    log::set_level(match level {
        0 => Level::Off,
        1 => Level::Error,
        2 => Level::Warn,
        3 => Level::Info,
        4 => Level::Debug,
        _ => Level::Trace,
    });
}

#[no_mangle]
pub extern "C" fn reconlGetLogLevel() -> u32 {
    match log::level() {
        Level::Off => 0,
        Level::Error => 1,
        Level::Warn => 2,
        Level::Info => 3,
        Level::Debug => 4,
        Level::Trace => 5,
    }
}

/// Installs a host log sink. Called from the emitting thread; the sink must not
/// call back into ReconL.
#[no_mangle]
pub extern "C" fn reconlSetLogSink(
    sink: Option<unsafe extern "C" fn(user: *mut c_void, level: u32, message: *const i8, file: *const i8, line: u32)>,
    user: *mut c_void,
) {
    log::set_sink(sink, user);
}
