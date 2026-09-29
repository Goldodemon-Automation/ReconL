//! The two vendor APIs this backend speaks, and the one shape they are used in.
//!
//! CUDA and ROCm are different libraries with different names for the same
//! three ideas: enumerate devices, run a kernel, move bytes. This module
//! declares the exact symbols each driver exports, resolves them through
//! [`crate::loader`], and hides the difference behind [`Runtime`], so the tier
//! resolver, the probe and the device talk to one API rather than two.
//!
//! Every declaration here is the vendor's own C signature. Nothing is
//! reinterpreted: a `CUdeviceptr` stays a `u64` and a `hipDeviceptr_t` stays a
//! pointer, because those are the types the vendors document and a cast between
//! them would be a silent ABI assumption rather than a declaration.

use core::ffi::{c_char, c_void};

use reconl_core::error::{Code, Error, Result};

use crate::loader::Library;

// ------------------------------------------------------------------------- CUDA
//
// The CUDA *driver* API, not the runtime API: it needs no toolkit at build time
// and no `cudart` at run time, and its entry points are `_v2`-suffixed where the
// driver exports a new ABI for an old name. Attribute numbers are the values
// `CUdevice_attribute` documents and they have been stable since CUDA 4; the
// ones used here are listed next to the constant so a reader can check them
// against the header rather than trusting the number.

pub mod cuda {
    use super::*;

    pub const SUCCESS: i32 = 0;
    pub const ATTR_MAX_THREADS_PER_BLOCK: i32 = 1;
    pub const ATTR_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
    pub const ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
    pub const ATTR_INTEGRATED: i32 = 18;
    pub const ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    pub const ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;

    pub type CuResult = i32;
    pub type CuDevice = i32;
    pub type CuContext = *mut c_void;
    pub type CuModule = *mut c_void;
    pub type CuFunction = *mut c_void;
    pub type CuStream = *mut c_void;
    pub type CuDevicePtr = u64;

    /// Every entry point the backend requires. A library that opens but does not
    /// export all of these is treated as unusable rather than half-loaded: a
    /// device built on a partial API would fail inside a frame, where the
    /// failure is a lost frame instead of a refusal.
    #[derive(Clone, Copy)]
    pub struct Api {
        pub init: unsafe extern "C" fn(u32) -> CuResult,
        pub driver_get_version: unsafe extern "C" fn(*mut i32) -> CuResult,
        pub device_get_count: unsafe extern "C" fn(*mut i32) -> CuResult,
        pub device_get: unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult,
        pub device_get_name: unsafe extern "C" fn(*mut c_char, i32, CuDevice) -> CuResult,
        pub device_total_mem: unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult,
        pub device_get_attribute: unsafe extern "C" fn(*mut i32, i32, CuDevice) -> CuResult,
        pub get_error_name: unsafe extern "C" fn(CuResult) -> *const c_char,
        pub get_error_string: unsafe extern "C" fn(CuResult) -> *const c_char,
        pub ctx_create: unsafe extern "C" fn(*mut CuContext, u32, CuDevice) -> CuResult,
        pub ctx_destroy: unsafe extern "C" fn(CuContext) -> CuResult,
        pub ctx_synchronize: unsafe extern "C" fn() -> CuResult,
        pub module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void, u32, *mut i32, *mut *mut c_void) -> CuResult,
        pub module_get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
        pub launch_kernel: unsafe extern "C" fn(
            CuFunction,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            CuStream,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> CuResult,
        pub mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
        pub mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
        pub memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult,
        pub memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    }

    impl Api {
        pub fn load(lib: &Library) -> Option<Api> {
            // SAFETY: each symbol is resolved into a signature that is copied
            // verbatim from `cuda.h`; a symbol whose real ABI differed would be
            // a wrong cast, which is why the signatures are kept in one block
            // next to the attribute table they belong to.
            unsafe {
                Some(Api {
                    init: lib.symbol("cuInit")?,
                    driver_get_version: lib.symbol("cuDriverGetVersion")?,
                    device_get_count: lib.symbol("cuDeviceGetCount")?,
                    device_get: lib.symbol("cuDeviceGet")?,
                    device_get_name: lib.symbol("cuDeviceGetName")?,
                    device_total_mem: lib
                        .symbol("cuDeviceTotalMem_v2")
                        .or_else(|| lib.symbol("cuDeviceTotalMem"))?,
                    device_get_attribute: lib
                        .symbol("cuDeviceGetAttribute")
                        .or_else(|| lib.symbol("cuDeviceGetAttribute_"))?,
                    get_error_name: lib.symbol("cuGetErrorName")?,
                    get_error_string: lib.symbol("cuGetErrorString")?,
                    ctx_create: lib
                        .symbol("cuCtxCreate_v2")
                        .or_else(|| lib.symbol("cuCtxCreate"))?,
                    ctx_destroy: lib
                        .symbol("cuCtxDestroy_v2")
                        .or_else(|| lib.symbol("cuCtxDestroy"))?,
                    ctx_synchronize: lib.symbol("cuCtxSynchronize")?,
                    module_load_data: lib
                        .symbol("cuModuleLoadDataEx")
                        .or_else(|| lib.symbol("cuModuleLoadData"))?,
                    module_get_function: lib.symbol("cuModuleGetFunction")?,
                    launch_kernel: lib.symbol("cuLaunchKernel")?,
                    mem_alloc: lib.symbol("cuMemAlloc_v2").or_else(|| lib.symbol("cuMemAlloc"))?,
                    mem_free: lib.symbol("cuMemFree_v2").or_else(|| lib.symbol("cuMemFree"))?,
                    memcpy_htod: lib
                        .symbol("cuMemcpyHtoD_v2")
                        .or_else(|| lib.symbol("cuMemcpyHtoD"))?,
                    memcpy_dtoh: lib
                        .symbol("cuMemcpyDtoH_v2")
                        .or_else(|| lib.symbol("cuMemcpyDtoH"))?,
                })
            }
        }

        /// The driver's own words for a failure, so a refusal quotes the vendor
        /// rather than inventing a reason.
        pub fn describe(&self, code: CuResult) -> String {
            let mut out = String::new();
            // SAFETY: both entry points take the result code and return a
            // static NUL-terminated string owned by the driver.
            unsafe {
                let name = (self.get_error_name)(code);
                if !name.is_null() {
                    out.push_str(&cstr_lossy(name));
                }
                let text = (self.get_error_string)(code);
                if !text.is_null() {
                    if !out.is_empty() {
                        out.push_str(": ");
                    }
                    out.push_str(&cstr_lossy(text));
                }
            }
            if out.is_empty() {
                out = format!("CUDA error {code}");
            }
            out
        }
    }
}

// --------------------------------------------------------------------------- HIP
//
// The ROCm runtime, which is what a host gets from `amdhip64`. The attribute
// enum is *not* CUDA's: HIP numbers its attributes in `cudaDeviceProp` field
// order, so `hipDeviceAttributeIntegrated` is 12 where CUDA's is 18. Only the
// attribute this backend actually reads is named here, and a value outside
// `{0, 1}` is reported as unknown rather than interpreted - see
// `crate::device`, which is where that decision matters.

pub mod hip {
    use super::*;

    pub const SUCCESS: i32 = 0;
    /// `hipDeviceAttributeIntegrated`: 0 = discrete, 1 = integrated APU.
    pub const ATTR_INTEGRATED: i32 = 12;
    pub const ATTR_MAX_THREADS_PER_BLOCK: i32 = 52;
    pub const ATTR_MULTIPROCESSOR_COUNT: i32 = 58;

    pub type HipError = i32;
    pub type HipContext = *mut c_void;
    pub type HipModule = *mut c_void;
    pub type HipFunction = *mut c_void;
    pub type HipStream = *mut c_void;
    pub type HipDevicePtr = *mut c_void;

    /// `hipMemcpyKind`.
    pub const MEMCPY_HOST_TO_DEVICE: i32 = 1;
    pub const MEMCPY_DEVICE_TO_HOST: i32 = 2;

    #[derive(Clone, Copy)]
    pub struct Api {
        pub init: unsafe extern "C" fn(u32) -> HipError,
        pub driver_get_version: unsafe extern "C" fn(*mut i32) -> HipError,
        pub device_get_count: unsafe extern "C" fn(*mut i32) -> HipError,
        pub device_get: unsafe extern "C" fn(*mut i32, i32) -> HipError,
        pub device_get_name: unsafe extern "C" fn(*mut c_char, i32, i32) -> HipError,
        pub device_total_mem: unsafe extern "C" fn(*mut usize, i32) -> HipError,
        pub device_get_attribute: unsafe extern "C" fn(*mut i32, i32, i32) -> HipError,
        pub get_error_name: unsafe extern "C" fn(HipError) -> *const c_char,
        pub get_error_string: unsafe extern "C" fn(HipError) -> *const c_char,
        pub ctx_create: unsafe extern "C" fn(*mut HipContext, u32, i32) -> HipError,
        pub ctx_destroy: unsafe extern "C" fn(HipContext) -> HipError,
        pub device_synchronize: unsafe extern "C" fn() -> HipError,
        pub module_load_data: unsafe extern "C" fn(*mut HipModule, *const c_void, u32, *mut i32, *mut *mut c_void) -> HipError,
        pub module_get_function: unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError,
        pub module_launch_kernel: unsafe extern "C" fn(
            HipFunction,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            HipStream,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> HipError,
        pub malloc: unsafe extern "C" fn(*mut HipDevicePtr, usize) -> HipError,
        pub free: unsafe extern "C" fn(HipDevicePtr) -> HipError,
        /// `hipMemcpy` with an explicit kind: the kind-explicit form is the one
        /// exported across every ROCm release this backend might meet.
        pub memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> HipError,
    }

    impl Api {
        pub fn load(lib: &Library) -> Option<Api> {
            // SAFETY: as for CUDA above - signatures copied from `hip_runtime_api.h`.
            unsafe {
                Some(Api {
                    init: lib.symbol("hipInit")?,
                    driver_get_version: lib.symbol("hipDriverGetVersion")?,
                    device_get_count: lib.symbol("hipGetDeviceCount")?,
                    device_get: lib.symbol("hipDeviceGet")?,
                    device_get_name: lib.symbol("hipDeviceGetName")?,
                    device_total_mem: lib.symbol("hipDeviceTotalMem")?,
                    device_get_attribute: lib.symbol("hipDeviceGetAttribute")?,
                    get_error_name: lib.symbol("hipGetErrorName")?,
                    get_error_string: lib.symbol("hipGetErrorString")?,
                    ctx_create: lib.symbol("hipCtxCreate")?,
                    ctx_destroy: lib.symbol("hipCtxDestroy")?,
                    device_synchronize: lib.symbol("hipDeviceSynchronize")?,
                    module_load_data: lib
                        .symbol("hipModuleLoadDataEx")
                        .or_else(|| lib.symbol("hipModuleLoadData"))?,
                    module_get_function: lib.symbol("hipModuleGetFunction")?,
                    module_launch_kernel: lib.symbol("hipModuleLaunchKernel")?,
                    malloc: lib.symbol("hipMalloc")?,
                    free: lib.symbol("hipFree")?,
                    memcpy: lib.symbol("hipMemcpy")?,
                })
            }
        }

        pub fn describe(&self, code: HipError) -> String {
            if code == SUCCESS {
                return "ok".into();
            }
            let mut out = String::new();
            // SAFETY: both take the error code and return driver-owned statics.
            unsafe {
                let name = (self.get_error_name)(code);
                if !name.is_null() {
                    out.push_str(&cstr_lossy(name));
                }
                let text = (self.get_error_string)(code);
                if !text.is_null() {
                    if !out.is_empty() {
                        out.push_str(": ");
                    }
                    out.push_str(&cstr_lossy(text));
                }
            }
            if out.is_empty() {
                out = format!("ROCm error {code}");
            }
            out
        }
    }
}

/// One device as the driver describes it, before any ReconL policy is applied.
#[derive(Clone, Debug, Default)]
pub struct DeviceInfo {
    pub index: i32,
    pub name: String,
    pub vram_bytes: u64,
    /// `Some(true)` integrated, `Some(false)` discrete, `None` when the runtime
    /// does not answer - which is the honest report and the one the D3D11 path
    /// makes when DXCore classification is unavailable.
    pub integrated: Option<bool>,
    pub compute_capability: Option<(u32, u32)>,
    pub max_threads_per_block: Option<u32>,
    pub multiprocessor_count: Option<u32>,
}

/// Which vendor runtime is loaded, and the API resolved from it.
pub enum Runtime {
    Cuda { lib: &'static Library, api: cuda::Api },
    Rocm { lib: &'static Library, api: hip::Api },
}

/// A live driver context, borrowed from a `'static` runtime.
///
/// The context is what every later driver call in a thread is relative to: a
/// module, an allocation and a kernel launch all belong to one. It is closed
/// explicitly rather than in `Drop` because the two vendors disagree about what
/// "the context was destroyed while a module was loaded" means, and both are
/// loud about it - an explicit close makes the order a caller's decision.
pub struct Context {
    runtime: &'static Runtime,
    handle: ContextHandle,
}

#[derive(Clone, Copy, Debug)]
pub enum ContextHandle {
    Cuda(cuda::CuContext),
    Rocm(hip::HipContext),
}

impl Context {
    pub fn handle(&self) -> ContextHandle {
        self.handle
    }

    pub fn runtime(&self) -> &'static Runtime {
        self.runtime
    }

    /// Releases the context. The handle is unusable afterwards.
    pub fn close(self) -> Result<()> {
        match (self.runtime, self.handle) {
            (Runtime::Cuda { api, .. }, ContextHandle::Cuda(handle)) => {
                // SAFETY: `handle` came from this API's `ctx_create` and is
                // closed exactly once, because `close` takes `self`.
                let rc = unsafe { (api.ctx_destroy)(handle) };
                if rc != cuda::SUCCESS {
                    return Err(Error::new(
                        Code::BackendUnavailable,
                        &format!("cuCtxDestroy refused: {}", api.describe(rc)),
                    ));
                }
            }
            (Runtime::Rocm { api, .. }, ContextHandle::Rocm(handle)) => {
                // SAFETY: as above, for HIP's own destroy.
                let rc = unsafe { (api.ctx_destroy)(handle) };
                if rc != hip::SUCCESS {
                    return Err(Error::new(
                        Code::BackendUnavailable,
                        &format!("hipCtxDestroy refused: {}", api.describe(rc)),
                    ));
                }
            }
            // A handle tagged with the wrong runtime is a bug in this module,
            // not a driver condition: it can only be produced by the two
            // constructors below, which pair them.
            _ => unreachable!("context handle and runtime disagree"),
        }
        Ok(())
    }
}

impl Runtime {
    /// The vendor name a report prints: the runtime that was actually opened.
    pub fn vendor(&self) -> &'static str {
        match self {
            Runtime::Cuda { .. } => "nvidia",
            Runtime::Rocm { .. } => "amd",
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Runtime::Cuda { .. } => "cuda",
            Runtime::Rocm { .. } => "rocm",
        }
    }

    pub fn library_name(&self) -> &'static str {
        match self {
            Runtime::Cuda { lib, .. } => lib.name(),
            Runtime::Rocm { lib, .. } => lib.name(),
        }
    }

    /// Opens a context on one device.
    ///
    /// The context becomes current on the calling thread, which is what the
    /// module load and the launches after it rely on.
    pub fn open_context(self: &'static Runtime, device_index: i32) -> Result<Context> {
        match self {
            Runtime::Cuda { api, .. } => {
                let mut dev: cuda::CuDevice = 0;
                // SAFETY: `cuDeviceGet` writes one `CUdevice` for the index.
                let rc = unsafe { (api.device_get)(&mut dev, device_index) };
                if rc != cuda::SUCCESS {
                    return Err(Error::new(
                        Code::BackendUnavailable,
                        &format!("cuDeviceGet({device_index}) refused: {}", api.describe(rc)),
                    ));
                }
                let mut handle: cuda::CuContext = core::ptr::null_mut();
                // SAFETY: `cuCtxCreate_v2` writes one `CUcontext`. `0` is
                // `CU_CTX_SCHED_AUTO`, the vendor's own default.
                let rc = unsafe { (api.ctx_create)(&mut handle, 0, dev) };
                if rc != cuda::SUCCESS {
                    return Err(Error::new(
                        Code::BackendUnavailable,
                        &format!("cuCtxCreate refused: {}", api.describe(rc)),
                    ));
                }
                Ok(Context { runtime: self, handle: ContextHandle::Cuda(handle) })
            }
            Runtime::Rocm { api, .. } => {
                let mut dev: i32 = 0;
                // SAFETY: `hipDeviceGet` writes one device ordinal.
                let rc = unsafe { (api.device_get)(&mut dev, device_index) };
                if rc != hip::SUCCESS {
                    return Err(Error::new(
                        Code::BackendUnavailable,
                        &format!("hipDeviceGet({device_index}) refused: {}", api.describe(rc)),
                    ));
                }
                let mut handle: hip::HipContext = core::ptr::null_mut();
                // SAFETY: `hipCtxCreate` writes one `hipContext_t`.
                let rc = unsafe { (api.ctx_create)(&mut handle, 0, dev) };
                if rc != hip::SUCCESS {
                    return Err(Error::new(
                        Code::BackendUnavailable,
                        &format!("hipCtxCreate refused: {}", api.describe(rc)),
                    ));
                }
                Ok(Context { runtime: self, handle: ContextHandle::Rocm(handle) })
            }
        }
    }

    /// Whether a context can be opened on this device, without keeping one.
    ///
    /// This is what makes an adapter's `usable` a *measurement* rather than a
    /// claim made from enumeration alone: a device whose driver reports it but
    /// whose context refuses (out of memory, a driver mid-reset, a device in an
    /// exclusive compute mode another process holds) is reported unusable with
    /// the driver's own words as the note.
    pub fn context_works(self: &'static Runtime, device_index: i32) -> core::result::Result<(), String> {
        match self.open_context(device_index) {
            Ok(ctx) => match ctx.close() {
                Ok(()) => Ok(()),
                Err(e) => Err(e.message.as_str().to_string()),
            },
            Err(e) => Err(e.message.as_str().to_string()),
        }
    }

    /// `cuInit` / `hipInit`. Idempotent on both, so calling it once per process
    /// is safe and calling it again costs a driver-side check.
    pub fn init(&self) -> Result<()> {
        match self {
            Runtime::Cuda { api, .. } => {
                // SAFETY: `cuInit` takes a flags word and touches no host memory.
                let rc = unsafe { (api.init)(0) };
                if rc != cuda::SUCCESS {
                    return Err(Error::new(
                        Code::BackendUnavailable,
                        &format!("cuInit refused: {}", api.describe(rc)),
                    ));
                }
            }
            Runtime::Rocm { api, .. } => {
                // SAFETY: `hipInit` as above.
                let rc = unsafe { (api.init)(0) };
                if rc != hip::SUCCESS {
                    return Err(Error::new(
                        Code::BackendUnavailable,
                        &format!("hipInit refused: {}", api.describe(rc)),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn driver_version(&self) -> Option<u32> {
        let mut v: i32 = 0;
        match self {
            // SAFETY: both write one `int` through the pointer they are given.
            Runtime::Cuda { api, .. } => {
                let rc = unsafe { (api.driver_get_version)(&mut v) };
                (rc == cuda::SUCCESS).then_some(v.max(0) as u32)
            }
            Runtime::Rocm { api, .. } => {
                let rc = unsafe { (api.driver_get_version)(&mut v) };
                (rc == hip::SUCCESS).then_some(v.max(0) as u32)
            }
        }
    }

    /// Every device the driver reports, in the driver's own enumeration order.
    ///
    /// A device whose name cannot be read is still reported - with an empty
    /// name - rather than dropped, because "the driver has N devices" and "I
    /// could name N of them" are different facts and only the first one decides
    /// whether a device can be created.
    pub fn probe_devices(&self) -> Result<Vec<DeviceInfo>> {
        self.init()?;
        match self {
            Runtime::Cuda { api, .. } => {
                let mut count: i32 = 0;
                // SAFETY: `cuDeviceGetCount` writes one `int`.
                let rc = unsafe { (api.device_get_count)(&mut count) };
                if rc != cuda::SUCCESS {
                    return Err(Error::new(
                        Code::BackendUnavailable,
                        &format!("cuDeviceGetCount refused: {}", api.describe(rc)),
                    ));
                }
                let mut out = Vec::new();
                for index in 0..count.max(0) {
                    let mut dev: cuda::CuDevice = 0;
                    // SAFETY: `cuDeviceGet` writes one `CUdevice` for an index
                    // the driver itself reported as in range.
                    if unsafe { (api.device_get)(&mut dev, index) } != cuda::SUCCESS {
                        continue;
                    }
                    out.push(DeviceInfo {
                        index,
                        name: read_name_cuda(api, dev),
                        vram_bytes: read_total_mem_cuda(api, dev),
                        integrated: read_attr_cuda(api, dev, cuda::ATTR_INTEGRATED).map(|v| v != 0),
                        compute_capability: match (
                            read_attr_cuda(api, dev, cuda::ATTR_COMPUTE_CAPABILITY_MAJOR),
                            read_attr_cuda(api, dev, cuda::ATTR_COMPUTE_CAPABILITY_MINOR),
                        ) {
                            (Some(major), Some(minor)) => Some((major as u32, minor as u32)),
                            _ => None,
                        },
                        max_threads_per_block: read_attr_cuda(api, dev, cuda::ATTR_MAX_THREADS_PER_BLOCK)
                            .map(|v| v as u32),
                        multiprocessor_count: read_attr_cuda(api, dev, cuda::ATTR_MULTIPROCESSOR_COUNT)
                            .map(|v| v as u32),
                    });
                }
                Ok(out)
            }
            Runtime::Rocm { api, .. } => {
                let mut count: i32 = 0;
                // SAFETY: `hipGetDeviceCount` writes one `int`.
                let rc = unsafe { (api.device_get_count)(&mut count) };
                if rc != hip::SUCCESS {
                    return Err(Error::new(
                        Code::BackendUnavailable,
                        &format!("hipGetDeviceCount refused: {}", api.describe(rc)),
                    ));
                }
                let mut out = Vec::new();
                for index in 0..count.max(0) {
                    let mut dev: i32 = 0;
                    // SAFETY: `hipDeviceGet` writes one device ordinal.
                    if unsafe { (api.device_get)(&mut dev, index) } != hip::SUCCESS {
                        continue;
                    }
                    out.push(DeviceInfo {
                        index,
                        name: read_name_hip(api, dev),
                        vram_bytes: read_total_mem_hip(api, dev),
                        // HIP's attribute enum is not CUDA's (`hip` module above),
                        // and an answer that is not 0 or 1 is reported as unknown
                        // rather than turned into a claim about the adapter.
                        integrated: match read_attr_hip(api, dev, hip::ATTR_INTEGRATED) {
                            Some(0) => Some(false),
                            Some(1) => Some(true),
                            _ => None,
                        },
                        compute_capability: None,
                        max_threads_per_block: read_attr_hip(api, dev, hip::ATTR_MAX_THREADS_PER_BLOCK)
                            .map(|v| v as u32),
                        multiprocessor_count: read_attr_hip(api, dev, hip::ATTR_MULTIPROCESSOR_COUNT)
                            .map(|v| v as u32),
                    });
                }
                Ok(out)
            }
        }
    }

    pub fn describe_error(&self, code: i32) -> String {
        match self {
            Runtime::Cuda { api, .. } => api.describe(code),
            Runtime::Rocm { api, .. } => api.describe(code),
        }
    }
}

fn read_name_cuda(api: &cuda::Api, dev: cuda::CuDevice) -> String {
    let mut buf = [0i8; 128];
    // SAFETY: `cuDeviceGetName` writes at most `len` bytes into `buf`.
    let rc = unsafe { (api.device_get_name)(buf.as_mut_ptr(), buf.len() as i32, dev) };
    if rc != cuda::SUCCESS {
        return String::new();
    }
    unsafe { cstr_lossy(buf.as_ptr()) }
}

fn read_total_mem_cuda(api: &cuda::Api, dev: cuda::CuDevice) -> u64 {
    let mut bytes: usize = 0;
    // SAFETY: `cuDeviceTotalMem` writes one `size_t`.
    let rc = unsafe { (api.device_total_mem)(&mut bytes, dev) };
    if rc != cuda::SUCCESS {
        0
    } else {
        bytes as u64
    }
}

fn read_attr_cuda(api: &cuda::Api, dev: cuda::CuDevice, attr: i32) -> Option<i32> {
    let mut value: i32 = 0;
    // SAFETY: `cuDeviceGetAttribute` writes one `int`.
    let rc = unsafe { (api.device_get_attribute)(&mut value, attr, dev) };
    (rc == cuda::SUCCESS).then_some(value)
}

fn read_name_hip(api: &hip::Api, dev: i32) -> String {
    let mut buf = [0i8; 128];
    // SAFETY: `hipDeviceGetName` writes at most `len` bytes into `buf`.
    let rc = unsafe { (api.device_get_name)(buf.as_mut_ptr(), buf.len() as i32, dev) };
    if rc != hip::SUCCESS {
        return String::new();
    }
    unsafe { cstr_lossy(buf.as_ptr()) }
}

fn read_total_mem_hip(api: &hip::Api, dev: i32) -> u64 {
    let mut bytes: usize = 0;
    // SAFETY: `hipDeviceTotalMem` writes one `size_t`.
    let rc = unsafe { (api.device_total_mem)(&mut bytes, dev) };
    if rc != hip::SUCCESS {
        0
    } else {
        bytes as u64
    }
}

fn read_attr_hip(api: &hip::Api, dev: i32, attr: i32) -> Option<i32> {
    let mut value: i32 = 0;
    // SAFETY: `hipDeviceGetAttribute` writes one `int`.
    let rc = unsafe { (api.device_get_attribute)(&mut value, attr, dev) };
    (rc == hip::SUCCESS).then_some(value)
}

/// Reads a NUL-terminated C string into an owned `String`.
///
/// # Safety
/// `ptr` must point at a NUL-terminated string, or be null.
unsafe fn cstr_lossy(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    let mut len = 0usize;
    // SAFETY: the caller guarantees a NUL-terminated string, so this scan stops.
    while unsafe { *ptr.add(len) } != 0 && len < 4096 {
        len += 1;
    }
    let bytes = unsafe { core::slice::from_raw_parts(ptr as *const u8, len) };
    String::from_utf8_lossy(bytes).into_owned()
}
