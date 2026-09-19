//! Throwaway probe: what code does each driver-reachable failure carry today?
#![allow(non_snake_case)]
use reconl::abi;
use reconl_core::{ABIStruct, StructHeader};

fn hdr<H: ABIStruct>() -> StructHeader {
    StructHeader::new(core::mem::size_of::<H>() as u32, H::STRUCT_TYPE)
}

fn error_text(device: *mut reconl::DeviceHandle) -> String {
    let mut ei: abi::ReconLErrorInfo = unsafe { core::mem::zeroed() };
    ei.base = hdr::<abi::ReconLErrorInfo>();
    if unsafe { reconl::reconlGetLastError(device, &mut ei) } != abi::result::OK {
        return "<none>".into();
    }
    let len = ei.message.iter().position(|&b| b == 0).unwrap_or(ei.message.len());
    String::from_utf8_lossy(&ei.message[..len]).into_owned()
}

unsafe extern "C" fn h_alloc(_u: *mut core::ffi::c_void, s: usize, a: usize) -> *mut core::ffi::c_void {
    let align = a.clamp(16, 4096);
    let total = match s.max(1).checked_add(align).and_then(|v| v.checked_add(16)) {
        Some(t) => t,
        None => return core::ptr::null_mut(),
    };
    let layout = match std::alloc::Layout::from_size_align(total, 16) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };
    let raw = unsafe { std::alloc::alloc(layout) };
    if raw.is_null() {
        return core::ptr::null_mut();
    }
    let base_ptr = raw as usize;
    // 16 bytes of slack before the aligned block, like the library's own
    // `HostAlloc::system`, so `aligned - 8` is always inside the allocation.
    let aligned = (base_ptr + 16 + align - 1) & !(align - 1);
    unsafe { ((aligned - 8) as *mut usize).write_unaligned(base_ptr) };
    aligned as *mut core::ffi::c_void
}
unsafe extern "C" fn h_free(_u: *mut core::ffi::c_void, p: *mut core::ffi::c_void, _s: usize) {
    if !p.is_null() {
        let base = unsafe { ((p as usize - 8) as *const usize).read_unaligned() } as *mut u8;
        unsafe { std::alloc::dealloc(base, std::alloc::Layout::from_size_align_unchecked(1, 16)) }
    }
}
unsafe extern "C" fn h_realloc(u: *mut core::ffi::c_void, p: *mut core::ffi::c_void, o: usize, n: usize, a: usize) -> *mut core::ffi::c_void {
    unsafe {
        let np = h_alloc(u, n, a);
        if !np.is_null() {
            if !p.is_null() { std::ptr::copy_nonoverlapping(p as *const u8, np as *mut u8, o.min(n)); }
            h_free(u, p, o);
        }
        np
    }
}

fn main() {
    unsafe {
        let mut dd: abi::ReconLDeviceDesc = core::mem::zeroed();
        dd.base = hdr::<abi::ReconLDeviceDesc>();
        dd.backend_hint = abi::backend::D3D11;
        dd.tier_hint = 1;
        dd.allow_downgrade = abi::allow_downgrade::TIER;
        dd.target_frame_ms = 1;
        dd.downgrade_after_frames = 1;
        dd.seed = 7;
        dd.allocator = abi::ReconLAllocator {
            alloc: Some(h_alloc), realloc: Some(h_realloc), free: Some(h_free), user: core::ptr::null_mut(),
        };
        let mut device: *mut reconl::DeviceHandle = core::ptr::null_mut();
        let r = reconl::reconlCreateDevice(&dd, &mut device);
        println!("create -> {r}");
        if r != abi::result::OK { println!("  {}", error_text(core::ptr::null_mut())); return; }

        let mut cl: *mut reconl::CommandListHandle = core::ptr::null_mut();
        let mut cd: abi::ReconLCommandListDesc = core::mem::zeroed();
        cd.base = hdr::<abi::ReconLCommandListDesc>();
        cd.capacity_bytes = 4096;
        assert_eq!(reconl::reconlCreateCommandList(device, &cd, &mut cl), abi::result::OK);

        // Frame 1: 32768x1 — past D3D11's 16384 texel limit. What code comes back?
        let mut fd: abi::ReconLFrameDesc = core::mem::zeroed();
        fd.base = hdr::<abi::ReconLFrameDesc>();
        fd.width = 32768; fd.height = 1; fd.seed = 1;
        let b = reconl::reconlBeginFrame(device, &fd as *const _ as *mut _);
        println!("begin(32768x1)   -> {b}");
        if b == abi::result::OK {
            reconl::reconlCmdReset(cl);
            let mut rp: abi::ReconLRenderPassDesc = core::mem::zeroed();
            rp.base = hdr::<abi::ReconLRenderPassDesc>();
            rp.viewport_width = 32768; rp.viewport_height = 1;
            rp.load_color = 1; rp.load_depth = 1;
            rp.clear_color = [0.0, 0.0, 0.0, 1.0];
            reconl::reconlCmdBeginRenderPass(cl, &rp);
            reconl::reconlCmdEndRenderPass(cl);
            let s = reconl::reconlSubmit(device, cl, core::ptr::null_mut());
            println!("submit(32768x1)  -> {s}");
            println!("  err: {}", error_text(device));
        }

        // Frame 2: a normal 64x64 frame, to show the device still works.
        fd.width = 64; fd.height = 64;
        let b2 = reconl::reconlBeginFrame(device, &fd as *const _ as *mut _);
        println!("begin(64x64)     -> {b2}");
        if b2 == abi::result::OK {
            reconl::reconlCmdReset(cl);
            let mut rp: abi::ReconLRenderPassDesc = core::mem::zeroed();
            rp.base = hdr::<abi::ReconLRenderPassDesc>();
            rp.viewport_width = 64; rp.viewport_height = 64;
            rp.load_color = 1; rp.load_depth = 1;
            rp.clear_color = [0.0, 0.0, 1.0, 1.0];
            reconl::reconlCmdBeginRenderPass(cl, &rp);
            reconl::reconlCmdEndRenderPass(cl);
            let s = reconl::reconlSubmit(device, cl, core::ptr::null_mut());
            println!("submit(64x64)    -> {s}");
        }
        reconl::reconlRelease(device as *mut core::ffi::c_void);
    }
}
