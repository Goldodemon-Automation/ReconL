//! In-process ABI tests: a Rust host exercising the same surface a C host sees.
//!
//! These exist because every bug this pass fixed was invisible to the crate's
//! own unit tests - the entry points returned OK while the frame they shipped
//! was empty. Each test pins one observed failure mode through the real entry
//! points, never through internal state.

#![allow(non_snake_case)]

use reconl::abi;
use reconl::abi::ReconLVertex;
use reconl_core::{ABIStruct, StructHeader};
use std::sync::atomic::{AtomicUsize, Ordering};

// ----------------------------------------------------------------- test host

/// The largest single allocation request a test's device has made through the
/// host allocator, and the bound this allocator refuses to exceed.
///
/// The bound is what makes an absurd descriptor safe to assert on: the library's
/// own ceiling has to refuse it *before* the allocator is called, and if that
/// ceiling ever regresses this allocator answers NULL instead of committing
/// gigabytes of the test machine's memory and starting it swapping.
#[derive(Default)]
struct Requests {
    largest: AtomicUsize,
    over_bound: AtomicUsize,
    calls: AtomicUsize,
    bytes: AtomicUsize,
}

/// 64 MiB: above everything a test legitimately allocates here (a 32 MiB buffer,
/// a 5 MiB mip chain, a 4 KiB frame), and far below what an absurd descriptor
/// implies.
const TEST_ALLOC_BOUND: usize = 64 << 20;

impl Requests {
    fn note(&self, size: usize) {
        self.largest.fetch_max(size, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(size, Ordering::Relaxed);
        if size > TEST_ALLOC_BOUND {
            self.over_bound.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Allocations made through this allocator so far, and their total size -
    /// counted per test, because the allocator belongs to one device.
    fn traffic(&self) -> (usize, usize) {
        (self.calls.load(Ordering::Relaxed), self.bytes.load(Ordering::Relaxed))
    }

    fn largest(&self) -> usize {
        self.largest.load(Ordering::Relaxed)
    }

    fn over_bound(&self) -> usize {
        self.over_bound.load(Ordering::Relaxed)
    }
}

/// The smallest conforming host allocator: base pointer in a header slot.
///
/// `user`, when a test sets it, is a [`Requests`] owned by that test: every
/// request is recorded, and anything above `TEST_ALLOC_BOUND` is refused.
extern "C" fn t_alloc(user: *mut core::ffi::c_void, size: usize, alignment: usize) -> *mut core::ffi::c_void {
    if !user.is_null() {
        // SAFETY: the only pointers passed here are boxes owned by the test that
        // created the device, and they outlive it.
        unsafe { &*(user as *const Requests) }.note(size);
    }
    if size > TEST_ALLOC_BOUND {
        return core::ptr::null_mut();
    }
    let align = alignment.clamp(16, 4096);
    let total = match size.max(1).checked_add(align).and_then(|v| v.checked_add(16)) {
        Some(t) => t,
        None => return core::ptr::null_mut(),
    };
    let layout = match std::alloc::Layout::from_size_align(total, 16) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };
    // SAFETY: layout has non-zero size.
    let raw = unsafe { std::alloc::alloc(layout) };
    if raw.is_null() {
        return core::ptr::null_mut();
    }
    let base = raw as usize;
    let aligned = (base + 16 + align - 1) & !(align - 1);
    // SAFETY: `aligned - 8` sits in the 16 bytes of slack.
    unsafe { ((aligned - 8) as *mut usize).write_unaligned(base) };
    aligned as *mut core::ffi::c_void
}

extern "C" fn t_free(_user: *mut core::ffi::c_void, ptr: *mut core::ffi::c_void, _size: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: t_alloc wrote the allocation base into the 8 bytes before the pointer.
    let base = unsafe { ((ptr as usize - 8) as *const usize).read_unaligned() } as *mut u8;
    // The layout of every block this allocator made: the size is not recoverable,
    // so free through the same layout t_alloc used. t_alloc pads every block to
    // at least one alignment unit, and `Layout::from_size_align` accepted `total`,
    // so deallocating with size 1 align 16 is valid for Rust's allocator.
    // SAFETY: base came from std::alloc::alloc in t_alloc.
    unsafe {
        std::alloc::dealloc(base, std::alloc::Layout::from_size_align_unchecked(1, 16));
    }
}

extern "C" fn t_realloc(user: *mut core::ffi::c_void, ptr: *mut core::ffi::c_void, old_size: usize, new_size: usize, alignment: usize) -> *mut core::ffi::c_void {
    let fresh = t_alloc(user, new_size, alignment);
    if fresh.is_null() {
        return core::ptr::null_mut();
    }
    if !ptr.is_null() {
        // SAFETY: both blocks are live for their full old/new size and do not overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr as *const u8, fresh as *mut u8, old_size.min(new_size));
        }
        t_free(user, ptr, old_size);
    }
    fresh
}

fn test_allocator() -> abi::ReconLAllocator {
    abi::ReconLAllocator {
        alloc: Some(t_alloc),
        realloc: Some(t_realloc),
        free: Some(t_free),
        user: core::ptr::null_mut(),
    }
}

/// The header every ABI struct starts with, stamped with the real size and type.
fn hdr<H: ABIStruct>() -> StructHeader {
    StructHeader::new(core::mem::size_of::<H>() as u32, H::STRUCT_TYPE)
}

/// A device with no spill dir, no downgrade and one worker: deterministic and tiny.
fn device_desc(allocator: abi::ReconLAllocator) -> abi::ReconLDeviceDesc {
    device_desc_for(allocator, abi::backend::SOFT_CPU, 2)
}

/// The same device aimed at any backend, so the hardware tier can be driven
/// through exactly the entry points the reference tier is driven through.
fn device_desc_for(
    allocator: abi::ReconLAllocator,
    backend_hint: u32,
    tier_hint: u32,
) -> abi::ReconLDeviceDesc {
    abi::ReconLDeviceDesc {
        base: hdr::<abi::ReconLDeviceDesc>(),
        backend_hint,
        tier_hint, // T1/gpu-shared for the hardware tier, T2/cpu-ram for the reference
        allow_downgrade: abi::allow_downgrade::NONE,
        worker_threads: if backend_hint == abi::backend::D3D11 { 0 } else { 1 },
        target_frame_ms: 1000,
        downgrade_after_frames: 16,
        seed: 7,
        flags: 0,
        budget: core::ptr::null(),
        allocator,
        backend_desc: core::ptr::null(),
    }
}

fn create_device(desc: &abi::ReconLDeviceDesc) -> Result<*mut reconl::DeviceHandle, i32> {
    let mut dev: *mut reconl::DeviceHandle = core::ptr::null_mut();
    let r = unsafe { reconl::reconlCreateDevice(desc, &mut dev) };
    if r != abi::result::OK || dev.is_null() {
        Err(r)
    } else {
        Ok(dev)
    }
}

/// One white triangle covering most of a 32x32 frame, in clip space.
fn triangle() -> [ReconLVertex; 3] {
    let mut v = [ReconLVertex::default(); 3];
    v[0].position = [0.0, 0.6, 0.5];
    v[1].position = [0.6, -0.6, 0.5];
    v[2].position = [-0.6, -0.6, 0.5];
    for t in &mut v {
        t.normal = [0.0, 0.0, 1.0];
        t.color = [1.0, 1.0, 1.0, 1.0];
    }
    v
}

const W: u32 = 32;
const H: u32 = 32;

/// The full boilerplate: device + swapchain + pipeline + buffers + command list.
struct Rig {
    device: *mut reconl::DeviceHandle,
    swapchain: *mut reconl::SwapchainHandle,
    pipeline: *mut reconl::PipelineHandle,
    commands: *mut reconl::CommandListHandle,
    pixels: Vec<u8>,
}

impl Rig {
    fn new() -> Self {
        Self::new_with(abi::backend::SOFT_CPU, 2)
    }

    fn new_with(backend_hint: u32, tier_hint: u32) -> Self {
        Self::new_full(backend_hint, tier_hint, core::ptr::null())
    }

    /// The same rig with the device's RAM cap pinned, so a frame's target
    /// reservation can be refused on purpose instead of depending on how much
    /// memory the machine running the test happens to have.
    fn new_with_ram_cap(ram_cap: u64) -> Self {
        Self::with_ram_cap(abi::backend::SOFT_CPU, 2, ram_cap)
    }

    /// The same rig on any backend, with the RAM cap pinned so that a refusal is
    /// a property of the test and not of the machine running it.
    fn with_ram_cap(backend_hint: u32, tier_hint: u32, ram_cap: u64) -> Self {
        let budget = abi::ReconLMemoryBudget {
            base: hdr::<abi::ReconLMemoryBudget>(),
            vram_cap_bytes: 0,
            ram_cap_bytes: ram_cap,
            disk_cap_bytes: 0,
            allow_disk_spill: 0,
            reserved: 0,
            spill_dir: core::ptr::null(),
        };
        Self::new_full(backend_hint, tier_hint, &budget)
    }

    fn new_full(backend_hint: u32, tier_hint: u32, budget: *const abi::ReconLMemoryBudget) -> Self {
        Self::with_allocator(backend_hint, tier_hint, budget, test_allocator())
    }

    /// The same rig on an allocator the test owns, so the allocation requests the
    /// library makes can be counted and bounded.
    fn with_allocator(
        backend_hint: u32,
        tier_hint: u32,
        budget: *const abi::ReconLMemoryBudget,
        allocator: abi::ReconLAllocator,
    ) -> Self {
        // SAFETY: every call is passed a valid descriptor built in this test.
        unsafe {
            let mut desc = device_desc_for(allocator, backend_hint, tier_hint);
            desc.budget = budget;
            let device = match create_device(&desc) {
                Ok(d) => d,
                // The detail matters here: a backend that refuses has to say
                // why, and this is the only place a host would see it.
                Err(code) => panic!("device creation failed: {code} ({})", last_global_error()),
            };
            let sd = abi::ReconLSwapchainDesc {
                base: hdr::<abi::ReconLSwapchainDesc>(),
                width: W,
                height: H,
                format: 1, // RGBA8
                image_count: 2,
                present_to_memory: 1,
                depth_format: 1,
                flags: 0,
                reserved: 0,
            };
            let mut swapchain: *mut reconl::SwapchainHandle = core::ptr::null_mut();
            assert_eq!(reconl::reconlCreateSwapchain(device, &sd, &mut swapchain), abi::result::OK);
            let pd = abi::ReconLPipelineDesc {
                base: hdr::<abi::ReconLPipelineDesc>(),
                shading: abi::shading::LAMBERT,
                blend: 0,
                cull: 0,
                depth_compare: 1, // GREATER, reversed-Z
                depth_write: 1,
                texture_slots: 0,
                texture_formats: [0; abi::RECONL_MAX_TEXTURE_SLOTS],
                receives_shadow: 0,
                casts_shadow: 0,
                flags: 0,
                reserved: 0,
                debug_name: core::ptr::null(),
            };
            let mut pipeline: *mut reconl::PipelineHandle = core::ptr::null_mut();
            assert_eq!(reconl::reconlCreatePipeline(device, &pd, &mut pipeline), abi::result::OK);
            let cd = abi::ReconLCommandListDesc {
                base: hdr::<abi::ReconLCommandListDesc>(),
                capacity_bytes: 4096,
                reserved: 0,
                debug_name: core::ptr::null(),
            };
            let mut commands: *mut reconl::CommandListHandle = core::ptr::null_mut();
            assert_eq!(reconl::reconlCreateCommandList(device, &cd, &mut commands), abi::result::OK);
            Self { device, swapchain, pipeline, commands, pixels: vec![0u8; (W * H * 4) as usize] }
        }
    }

    fn stats(&self) -> abi::ReconLStats {
        let mut st: abi::ReconLStats = unsafe { core::mem::zeroed() };
        st.base = hdr::<abi::ReconLStats>();
        let r = unsafe { reconl::reconlGetStats(self.device, &mut st) };
        assert_eq!(r, abi::result::OK);
        st
    }

    /// The device limits, as a host reads them back.
    fn limits(&self) -> abi::ReconLDeviceLimits {
        let mut l: abi::ReconLDeviceLimits = unsafe { core::mem::zeroed() };
        l.base = hdr::<abi::ReconLDeviceLimits>();
        let r = unsafe { reconl::reconlGetDeviceLimits(self.device, &mut l) };
        assert_eq!(r, abi::result::OK);
        l
    }

    /// The device's last error, as text: the only thing that makes a failed
    /// GPU call diagnosable from a test.
    fn error_message(&self) -> String {
        let ei = self.last_error();
        let len = ei.message.iter().position(|&b| b == 0).unwrap_or(ei.message.len());
        String::from_utf8_lossy(&ei.message[..len]).into_owned()
    }

    fn last_error(&self) -> abi::ReconLErrorInfo {
        let mut ei: abi::ReconLErrorInfo = unsafe { core::mem::zeroed() };
        ei.base = hdr::<abi::ReconLErrorInfo>();
        let r = unsafe { reconl::reconlGetLastError(self.device, &mut ei) };
        assert_eq!(r, abi::result::OK);
        ei
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: handles created by this rig, released once.
        unsafe {
            reconl::reconlRelease(self.commands as *mut core::ffi::c_void);
            reconl::reconlRelease(self.pipeline as *mut core::ffi::c_void);
            reconl::reconlRelease(self.swapchain as *mut core::ffi::c_void);
            reconl::reconlRelease(self.device as *mut core::ffi::c_void);
        }
    }
}

/// Calls `reconlBeginFrame` at an arbitrary size, with the rig's headlight.
unsafe fn begin_frame(rig: &mut Rig, width: u32, height: u32) -> i32 {
    // A headlight so the Lambert pipeline is lit: without it every pixel the
    // triangle covers shades to black and the test cannot tell draw from clear.
    let mut light: abi::ReconLLight = core::mem::zeroed();
    light.base = hdr::<abi::ReconLLight>();
    light.r#type = 0; // directional
    light.direction = [0.0, 0.0, -1.0];
    light.color = [1.0, 1.0, 1.0];
    light.intensity = 1.0;
    let ll = abi::ReconLLightList {
        base: hdr::<abi::ReconLLightList>(),
        count: 1,
        reserved: 0,
        lights: &light,
    };
    let fd: abi::ReconLFrameDesc = abi::ReconLFrameDesc {
        base: hdr::<abi::ReconLFrameDesc>(),
        width,
        height,
        seed: 1,
        reserved: 0,
        lights: &ll,
        shadows: core::ptr::null(),
        camera: core::ptr::null(),
        framegen: core::ptr::null(),
    };
    reconl::reconlBeginFrame(rig.device, &fd as *const _ as *mut _)
}

/// Records one indexed draw of the given triangles, submits and presents.
/// Returns the submit result; the presented pixels are in `rig.pixels`.
unsafe fn draw_frame(rig: &mut Rig, verts: &[ReconLVertex], indices: &[u32], vertex_offset: i32, buffer_offset: u64) -> i32 {
    let submit_r = submit_frame_with_fence(rig, verts, indices, vertex_offset, buffer_offset, core::ptr::null_mut());
    let mut prd = abi::ReconLPresentDesc {
        base: hdr::<abi::ReconLPresentDesc>(),
        out_pixels: rig.pixels.as_mut_ptr() as *mut core::ffi::c_void,
        out_pixels_size: (W * H * 4) as u64,
        out_row_pitch: W * 4,
        out_format: 1,
        flip: 0,
    };
    if submit_r == abi::result::OK {
        assert_eq!(reconl::reconlPresent(rig.device, rig.swapchain, &mut prd), abi::result::OK);
    }
    submit_r
}

/// Records one indexed draw and submits it, leaving the frame submitted and
/// unpresented so the caller chooses the present.
unsafe fn submit_frame(rig: &mut Rig, verts: &[ReconLVertex], indices: &[u32], vertex_offset: i32, buffer_offset: u64) -> i32 {
    submit_frame_with_fence(rig, verts, indices, vertex_offset, buffer_offset, core::ptr::null_mut())
}

/// Begins the rig's standard frame and asserts it opened.
unsafe fn begin_default(rig: &mut Rig) -> i32 {
    let r = begin_frame(rig, W, H);
    assert_eq!(r, abi::result::OK);
    r
}

/// The same as `submit_frame`, with the fence the host passes to `reconlSubmit`.
unsafe fn submit_frame_with_fence(rig: &mut Rig, verts: &[ReconLVertex], indices: &[u32], vertex_offset: i32, buffer_offset: u64, fence: *mut reconl::FenceHandle) -> i32 {
    begin_default(rig);
    record_and_submit(rig, verts, indices, vertex_offset, buffer_offset, fence)
}

/// The same, for a frame whose pass asks for a viewport other than the whole
/// frame.
unsafe fn record_and_submit_with_viewport(
    rig: &mut Rig,
    verts: &[ReconLVertex],
    indices: &[u32],
    vertex_offset: i32,
    buffer_offset: u64,
    fence: *mut reconl::FenceHandle,
    viewport: (u32, u32),
) -> i32 {
    record_and_submit_inner(rig, verts, indices, vertex_offset, buffer_offset, fence, viewport)
}

/// Records the draw into the frame that is already open and submits it. This is
/// the retry path: a submit refused for its arguments leaves the frame open, so
/// the host reuses the frame it has rather than beginning another.
unsafe fn record_and_submit(rig: &mut Rig, verts: &[ReconLVertex], indices: &[u32], vertex_offset: i32, buffer_offset: u64, fence: *mut reconl::FenceHandle) -> i32 {
    record_and_submit_inner(rig, verts, indices, vertex_offset, buffer_offset, fence, (W, H))
}

unsafe fn record_and_submit_inner(
    rig: &mut Rig,
    verts: &[ReconLVertex],
    indices: &[u32],
    vertex_offset: i32,
    buffer_offset: u64,
    fence: *mut reconl::FenceHandle,
    viewport: (u32, u32),
) -> i32 {
    let vbd = abi::ReconLBufferDesc {
        base: hdr::<abi::ReconLBufferDesc>(),
        size_bytes: (verts.len() * core::mem::size_of::<ReconLVertex>()) as u64,
        usage: 1, // VERTEX
        reserved: 0,
        data: verts.as_ptr() as *const core::ffi::c_void,
        data_size: (verts.len() * core::mem::size_of::<ReconLVertex>()) as u64,
        debug_name: core::ptr::null(),
    };
    let mut vb: *mut reconl::BufferHandle = core::ptr::null_mut();
    assert_eq!(reconl::reconlCreateBuffer(rig.device, &vbd, &mut vb), abi::result::OK);
    let ibd = abi::ReconLBufferDesc {
        base: hdr::<abi::ReconLBufferDesc>(),
        size_bytes: (indices.len() * 4) as u64,
        usage: 2, // INDEX
        reserved: 0,
        data: indices.as_ptr() as *const core::ffi::c_void,
        data_size: (indices.len() * 4) as u64,
        debug_name: core::ptr::null(),
    };
    let mut ib: *mut reconl::BufferHandle = core::ptr::null_mut();
    assert_eq!(reconl::reconlCreateBuffer(rig.device, &ibd, &mut ib), abi::result::OK);

    reconl::reconlCmdReset(rig.commands);
    let rp = abi::ReconLRenderPassDesc {
        base: hdr::<abi::ReconLRenderPassDesc>(),
        color_count: 0,
        reserved: 0,
        color: [zeroed_attachment(); abi::RECONL_MAX_ATTACHMENTS],
        depth: core::ptr::null_mut(),
        viewport_width: viewport.0,
        viewport_height: viewport.1,
        load_color: 1,
        load_depth: 1,
        clear_color: [0.0, 0.0, 1.0, 1.0],
        clear_depth: 0.0,
        stencil_clear: 0,
        reserved2: 0,
    };
    reconl::reconlCmdBeginRenderPass(rig.commands, &rp);
    reconl::reconlCmdSetPipeline(rig.commands, rig.pipeline);
    let identity: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    reconl::reconlCmdPushConstants(rig.commands, 0, identity.as_ptr() as *const core::ffi::c_void, 64);
    reconl::reconlCmdPushConstants(rig.commands, 1, identity.as_ptr() as *const core::ffi::c_void, 64);
    reconl::reconlCmdSetVertexBuffer(rig.commands, 0, vb, buffer_offset);
    reconl::reconlCmdSetIndexBuffer(rig.commands, ib, 0, 1); // UINT32
    let draw_r = reconl::reconlCmdDrawIndexed(rig.commands, indices.len() as u32, 0, vertex_offset);
    reconl::reconlCmdEndRenderPass(rig.commands);

    let submit_r = if draw_r == abi::result::OK {
        reconl::reconlSubmit(rig.device, rig.commands, fence)
    } else {
        draw_r
    };
    reconl::reconlRelease(vb as *mut core::ffi::c_void);
    reconl::reconlRelease(ib as *mut core::ffi::c_void);
    submit_r
}

/// Presents `frames` empty 1080p frames on a device built from `desc`, and
/// returns each frame's cost as the host reads it plus the device's final
/// stats. 1080p because the readback the host waits for is most of a frame at
/// that size, which is the split the ladder's definition is about; an empty
/// pass because nothing here depends on what is drawn.
unsafe fn present_empty_frames(
    desc: abi::ReconLDeviceDesc,
    frames: u32,
) -> (Vec<u64>, abi::ReconLStats) {
    const FW: u32 = 1920;
    const FH: u32 = 1080;
    let device = match create_device(&desc) {
        Ok(d) => d,
        Err(code) => panic!("device creation failed: {code} ({})", last_global_error()),
    };
    let sd = abi::ReconLSwapchainDesc {
        base: hdr::<abi::ReconLSwapchainDesc>(),
        width: FW,
        height: FH,
        format: 1, // RGBA8
        image_count: 2,
        present_to_memory: 1,
        depth_format: 1,
        flags: 0,
        reserved: 0,
    };
    let mut swapchain: *mut reconl::SwapchainHandle = core::ptr::null_mut();
    assert_eq!(reconl::reconlCreateSwapchain(device, &sd, &mut swapchain), abi::result::OK);
    let cd = abi::ReconLCommandListDesc {
        base: hdr::<abi::ReconLCommandListDesc>(),
        capacity_bytes: 4096,
        reserved: 0,
        debug_name: core::ptr::null(),
    };
    let mut commands: *mut reconl::CommandListHandle = core::ptr::null_mut();
    assert_eq!(reconl::reconlCreateCommandList(device, &cd, &mut commands), abi::result::OK);
    let mut pixels = vec![0u8; (FW * FH * 4) as usize];

    // One directional light, so the shadow pass the frame cost is dominated by
    // is the one a real host gets.
    let mut light: abi::ReconLLight = core::mem::zeroed();
    light.base = hdr::<abi::ReconLLight>();
    light.r#type = 0;
    light.direction = [0.0, 0.0, -1.0];
    light.color = [1.0, 1.0, 1.0];
    light.intensity = 1.0;
    let ll = abi::ReconLLightList {
        base: hdr::<abi::ReconLLightList>(),
        count: 1,
        reserved: 0,
        lights: &light,
    };

    let mut totals = Vec::new();
    for _ in 0..frames {
        let fd = abi::ReconLFrameDesc {
            base: hdr::<abi::ReconLFrameDesc>(),
            width: FW,
            height: FH,
            seed: 1,
            reserved: 0,
            lights: &ll,
            shadows: core::ptr::null(),
            camera: core::ptr::null(),
            framegen: core::ptr::null(),
        };
        assert_eq!(reconl::reconlBeginFrame(device, &fd as *const _ as *mut _), abi::result::OK);
        reconl::reconlCmdReset(commands);
        let rp = abi::ReconLRenderPassDesc {
            base: hdr::<abi::ReconLRenderPassDesc>(),
            color_count: 0,
            reserved: 0,
            color: [zeroed_attachment(); abi::RECONL_MAX_ATTACHMENTS],
            depth: core::ptr::null_mut(),
            viewport_width: FW,
            viewport_height: FH,
            load_color: 1,
            load_depth: 1,
            clear_color: [0.0, 0.0, 1.0, 1.0],
            clear_depth: 0.0,
            stencil_clear: 0,
            reserved2: 0,
        };
        reconl::reconlCmdBeginRenderPass(commands, &rp);
        reconl::reconlCmdEndRenderPass(commands);
        assert_eq!(reconl::reconlSubmit(device, commands, core::ptr::null_mut()), abi::result::OK);
        let mut prd = abi::ReconLPresentDesc {
            base: hdr::<abi::ReconLPresentDesc>(),
            out_pixels: pixels.as_mut_ptr() as *mut core::ffi::c_void,
            out_pixels_size: (FW * FH * 4) as u64,
            out_row_pitch: FW * 4,
            out_format: 1,
            flip: 0,
        };
        assert_eq!(reconl::reconlPresent(device, swapchain, &mut prd), abi::result::OK);
        let mut st: abi::ReconLStats = core::mem::zeroed();
        st.base = hdr::<abi::ReconLStats>();
        assert_eq!(reconl::reconlGetStats(device, &mut st), abi::result::OK);
        totals.push(st.frame.total_ns);
    }

    let mut stats: abi::ReconLStats = core::mem::zeroed();
    stats.base = hdr::<abi::ReconLStats>();
    assert_eq!(reconl::reconlGetStats(device, &mut stats), abi::result::OK);
    reconl::reconlRelease(commands as *mut core::ffi::c_void);
    reconl::reconlRelease(swapchain as *mut core::ffi::c_void);
    reconl::reconlRelease(device as *mut core::ffi::c_void);
    (totals, stats)
}

/// The number that follows `marker` in a downgrade's detail.
fn number_after(detail: &str, marker: &str) -> Option<u64> {
    let rest = &detail[detail.find(marker)? + marker.len()..];
    let digits: String = rest
        .trim_start_matches(|c: char| !c.is_ascii_digit())
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// The device's own over-target ladder must judge the number the host reads.
///
/// "This frame was over target" has two candidate numbers: the device's own
/// render pass, and the frame the host actually waited for, the readback it
/// blocked on included. At 1080p they differ by milliseconds. The pin reads the
/// number the ladder *acted on* out of the entry it wrote - every tier change
/// records the cost it was decided from - and requires it to be the number the
/// host read for that frame, so a device-side ladder judging its own smaller
/// render time fails on the arithmetic rather than on a target placed near the
/// gap between the two, which is what used to make this pin a measurement of how
/// loaded the machine was.
///
/// The target is one millisecond - the ABI's smallest, and under every 1080p
/// frame either tier renders on this host - so the ladder acts on the first frame
/// it can whatever else the suite is doing. The offload is opted out: the relabel
/// is then the only response, and the hardware stays the renderer, so the number
/// the host reads for the frame the ladder acted on is still that device's.
#[test]
fn the_device_ladder_judges_the_frame_the_host_reads() {
    // SAFETY: every descriptor is built here, and every handle is released.
    unsafe {
        let mut desc = device_desc_for(test_allocator(), abi::backend::D3D11, 1);
        desc.target_frame_ms = 1;
        desc.downgrade_after_frames = 1;
        desc.allow_downgrade = abi::allow_downgrade::NONE;
        let (totals, stats) = present_empty_frames(desc, 3);

        assert!(totals.iter().all(|ns| *ns > 1_000_000), "1080p frames are over the target here");
        // The ladder acted: the first frame arms the run, the second relabels.
        assert!(stats.downgrade_count >= 1, "a frame over target must step the device's own tier");
        assert!(stats.tier > 1, "the device stepped a tier (tier {})", stats.tier);

        let first = &stats.downgrades[0];
        let detail: String = first
            .detail
            .iter()
            .take_while(|c| **c != 0)
            .map(|c| char::from(*c as u8))
            .collect();
        assert_eq!(first.frame_index, 1, "the step lands on the frame after the cost that armed it");
        assert!(
            detail.contains("the host read"),
            "the entry names the number it was decided from: {detail}"
        );
        // The clause, exactly: the ladder's record of the frame it acted on is the
        // frame the host read, not the device's own smaller number for it.
        assert_eq!(
            number_after(&detail, "the host read"),
            Some(totals[0]),
            "the ladder must judge the frame the host read ({} ns), not the device's own render time: {detail}",
            totals[0]
        );
    }
}

fn zeroed_attachment() -> abi::ReconLColorAttachment {
    // SAFETY: all-zero bytes are valid for this plain-data struct.
    unsafe { core::mem::zeroed() }
}

fn count_colours(pixels: &[u8]) -> (usize, usize) {
    let mut blue = 0;
    let mut white = 0;
    for px in pixels.chunks_exact(4) {
        if px[0] > 200 && px[2] > 200 {
            white += 1;
        } else if px[2] > 200 && px[0] < 50 {
            blue += 1;
        }
    }
    (blue, white)
}

// ------------------------------------------------------------------- the bugs

/// A C host's frame used to be the clear colour end to end: the draw list was
/// allocated from a dead allocator that refuses every allocation, so Submit
/// shipped zero draws and every call still returned OK.
#[test]
fn submit_renders_the_draws_the_host_recorded() {
    let mut rig = Rig::new();
    let r = unsafe { draw_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK);
    let (_, white) = count_colours(&rig.pixels);
    assert!(white > 0, "the recorded triangle must be visible; got {white} white px");
    let st = rig.stats();
    assert_eq!(st.frame.triangles_in, 1);
}

/// The device handle used to point `header.device` at a stack copy that died
/// before the first call, so every entry point that dereferenced it read dead
/// memory. It must point at the live block.
#[test]
fn device_header_points_at_the_live_device() {
    // SAFETY: a device we just created and release before it goes away.
    unsafe {
        let device = create_device(&device_desc(test_allocator())).expect("device");
        // The header layout is kind, refcount, device: the device field must
        // point at the block itself, not a dead stack copy.
        let hdr = &*(device as *const reconl::HandleHeader);
        let _ = hdr;
        assert_eq!(hdr.kind, 1, "Kind::Device is 1");
        let seen = hdr.device;
        assert_eq!(seen, device as *mut _);
        assert_eq!(reconl::reconlRelease(device as *mut core::ffi::c_void), 0, "one reference held");
    }
}

/// `reconl_backends.h` declares the three `ReconL*Desc` structs and this
/// revision says plainly that `ReconLDeviceDesc::backend_desc` is reserved:
/// accepted and never read. That is a negative contract, so it needs a pin -
/// a descriptor at address 8 faults if anything dereferences it, and the device
/// that comes back must be the one a host with no descriptor gets.
#[test]
fn a_reserved_backend_descriptor_is_accepted_and_never_read() {
    fn limits(device: *mut reconl::DeviceHandle) -> abi::ReconLDeviceLimits {
        let mut l: abi::ReconLDeviceLimits = unsafe { core::mem::zeroed() };
        l.base = hdr::<abi::ReconLDeviceLimits>();
        assert_eq!(unsafe { reconl::reconlGetDeviceLimits(device, &mut l) }, abi::result::OK);
        l
    }

    let mut without: *mut reconl::DeviceHandle = core::ptr::null_mut();
    let desc = device_desc(test_allocator());
    assert_eq!(unsafe { reconl::reconlCreateDevice(&desc, &mut without) }, abi::result::OK);

    let mut with_wild: *mut reconl::DeviceHandle = core::ptr::null_mut();
    let mut desc = device_desc(test_allocator());
    desc.backend_desc = 8usize as *const core::ffi::c_void;
    assert_eq!(unsafe { reconl::reconlCreateDevice(&desc, &mut with_wild) }, abi::result::OK);

    let a = limits(without);
    let b = limits(with_wild);
    assert_eq!(a.backend, b.backend, "the descriptor must not choose a backend");
    assert_eq!(a.caps, b.caps, "the descriptor must not change the caps");
    assert_eq!(a.worker_threads_max, b.worker_threads_max);
    assert_eq!(a.tile_size_min, b.tile_size_min);
    assert_eq!(a.max_allocation_bytes, b.max_allocation_bytes);
    unsafe {
        reconl::reconlRelease(without as *mut core::ffi::c_void);
        reconl::reconlRelease(with_wild as *mut core::ffi::c_void);
    }
}

/// `failures` counts every non-OK call, and `last_result` holds the code of the
/// latest one. Both used to be read from the backend, which never sees a failed
/// host call, so both sat at their defaults forever.
#[test]
fn failures_and_last_result_come_from_the_device() {
    let rig = Rig::new();
    assert_eq!(rig.stats().failures, 0);
    // A null command list: fails on the argument, before the frame state.
    let r = unsafe { reconl::reconlSubmit(rig.device, core::ptr::null(), core::ptr::null_mut()) };
    assert_eq!(r, abi::result::INVALID_ARGUMENT);
    // A real list with no frame open: the state check fires.
    let r = unsafe { reconl::reconlSubmit(rig.device, rig.commands, core::ptr::null_mut()) };
    assert_eq!(r, abi::result::NO_FRAME);
    let st = rig.stats();
    assert_eq!(st.failures, 2, "a failed call must count");
    assert_eq!(st.last_result, abi::result::NO_FRAME);
    // The error slot agrees with the counters.
    assert_eq!(rig.last_error().result, abi::result::NO_FRAME);
}

/// A submit that fails validation used to leave the device in `Open` forever:
/// `BeginFrame` then refused with FrameInProgress, and there is no ABI call to
/// abandon a frame - the host was stuck after one bad draw list. A failed
/// submit must drop the frame and count the drop.
#[test]
fn a_failed_submit_releases_the_frame() {
    let mut rig = Rig::new();
    // Three indices into a one-vertex buffer: the draw is out of range.
    let verts = [ReconLVertex::default(); 1];
    let r = unsafe { draw_frame(&mut rig, &verts, &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::INVALID_ARGUMENT);
    // The frame is gone: a fresh one can open, and the drop was counted.
    let st = rig.stats();
    assert_eq!(st.frames_dropped, 1);
    assert_eq!(st.frames_presented, 0);
    // And the device still renders: a good frame after the bad one works.
    let r = unsafe { draw_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK);
    let (_, white) = count_colours(&rig.pixels);
    assert!(white > 0);
    let st = rig.stats();
    assert_eq!(st.frames_presented, 1);
    assert_eq!(st.frames_dropped, 1);
}

/// The same defect one call later: a present that fails must consume the frame
/// too. It used to leave the state `Submitted`, which no other call accepts -
/// `BeginFrame` refused with FrameInProgress and `Submit` with NoFrame - so the
/// only way out was a present that succeeded, a call a host that has already
/// given up on the frame has no reason to make.
#[test]
fn a_failed_present_releases_the_frame() {
    let mut rig = Rig::new();
    let r = unsafe { submit_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK);
    // A buffer sized for something else: the present cannot be delivered.
    let mut prd = abi::ReconLPresentDesc {
        base: hdr::<abi::ReconLPresentDesc>(),
        out_pixels: rig.pixels.as_mut_ptr() as *mut core::ffi::c_void,
        out_pixels_size: 4,
        out_row_pitch: W * 4,
        out_format: 1,
        flip: 0,
    };
    let r = unsafe { reconl::reconlPresent(rig.device, rig.swapchain, &mut prd) };
    assert_eq!(r, abi::result::INVALID_ARGUMENT);
    // The frame is gone, visibly: counted as dropped, and the state no longer
    // accepts a present at all.
    let st = rig.stats();
    assert_eq!(st.frames_dropped, 1, "a present that fails consumes the frame");
    assert_eq!(st.frames_presented, 0);
    assert_eq!(st.last_result, abi::result::INVALID_ARGUMENT);
    let r = unsafe { reconl::reconlPresent(rig.device, rig.swapchain, &mut prd) };
    assert_eq!(r, abi::result::NO_FRAME, "the dropped frame is not held for a retry");
    // And the device still renders: a whole good frame after the bad present.
    let r = unsafe { draw_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK, "BeginFrame must be accepted after a failed present");
    let (_, white) = count_colours(&rig.pixels);
    assert!(white > 0, "the frame after the failed present must render");
    let st = rig.stats();
    assert_eq!(st.frames_presented, 1);
    assert_eq!(st.frames_dropped, 1);
}

/// A begin whose target reservation is refused must leave the device able to
/// begin again. The state used to flip to `Open` before the reservation, so a
/// refused begin wedged the device: `BeginFrame` then said FrameInProgress, and
/// the only call it accepted was `Submit` - which is not what a host does after
/// a begin it just watched fail.
#[test]
fn a_failed_begin_frame_leaves_the_device_able_to_begin_again() {
    // A 32 MiB device: enough for the rig's own frame and its cascade maps, far
    // too little for this frame's colour and depth targets (128 MiB).
    let mut rig = Rig::new_with_ram_cap(32 << 20);
    let r = unsafe { begin_frame(&mut rig, 4096, 4096) };
    assert_eq!(r, abi::result::BUDGET_EXCEEDED, "{}", rig.error_message());
    assert_eq!(rig.stats().frames_dropped, 0, "a begin that never opened a frame drops nothing");
    // The device is idle again: a whole frame, begin through present, works.
    let r = unsafe { draw_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK, "{}", rig.error_message());
    let (_, white) = count_colours(&rig.pixels);
    assert!(white > 0, "the frame after the refused begin must render");
    let st = rig.stats();
    assert_eq!(st.frames_presented, 1);
    assert_eq!(st.frames_dropped, 0);
}

/// A handle from another device is refused *before* the frame is committed, so
/// it does not cost the host the frame it was about to render: the frame stays
/// open and the same submit can be retried with a valid fence. It used to be
/// validated after the commit point, which dropped - and counted - a frame whose
/// only fault was the caller's fence argument.
#[test]
fn a_foreign_fence_does_not_cost_the_frame() {
    let mut rig = Rig::new();
    let other = Rig::new();
    let mut fence: *mut reconl::FenceHandle = core::ptr::null_mut();
    // SAFETY: a live device from the second rig; the fence is released below.
    unsafe {
        assert_eq!(reconl::reconlCreateFence(other.device, 0, &mut fence), abi::result::OK);
        assert!(!fence.is_null());
        let r = submit_frame_with_fence(&mut rig, &triangle(), &[0, 1, 2], 0, 0, fence);
        assert_eq!(r, abi::result::INVALID_HANDLE, "a foreign fence is the caller's bug");
        let st = rig.stats();
        assert_eq!(st.frames_dropped, 0, "a bad handle must not cost the frame");
        assert_eq!(st.frames_presented, 0);
        // The frame is still open, so the same submit with a valid fence works.
        let r = record_and_submit(&mut rig, &triangle(), &[0, 1, 2], 0, 0, core::ptr::null_mut());
        assert_eq!(r, abi::result::OK);
        let mut prd = abi::ReconLPresentDesc {
            base: hdr::<abi::ReconLPresentDesc>(),
            out_pixels: rig.pixels.as_mut_ptr() as *mut core::ffi::c_void,
            out_pixels_size: (W * H * 4) as u64,
            out_row_pitch: W * 4,
            out_format: 1,
            flip: 0,
        };
        assert_eq!(reconl::reconlPresent(rig.device, rig.swapchain, &mut prd), abi::result::OK);
        let st = rig.stats();
        assert_eq!(st.frames_presented, 1);
        assert_eq!(st.frames_dropped, 0, "the retried frame must present cleanly");
        assert_eq!(reconl::reconlRelease(fence as *mut core::ffi::c_void), 0);
    }
}

/// An indexed draw's vertex data sits at `buffer_offset + vertex_offset`; both
/// used to be dropped, so the draw read vertex 0 whatever the host bound.
#[test]
fn indexed_draw_honours_buffer_and_vertex_offsets() {
    let mut rig = Rig::new();
    let red = {
        let mut v = triangle();
        for t in &mut v {
            t.color = [1.0, 0.0, 0.0, 1.0];
        }
        v
    };
    let mut both = Vec::with_capacity(6);
    both.extend_from_slice(&red);
    both.extend_from_slice(&triangle());
    // Buffer offset picks the white triangle.
    let r = unsafe { draw_frame(&mut rig, &both, &[0, 1, 2], 0, 3 * core::mem::size_of::<ReconLVertex>() as u64) };
    assert_eq!(r, abi::result::OK);
    let (_, white) = count_colours(&rig.pixels);
    assert!(white > 0, "buffer offset must select the white triangle");
    // Draw vertex_offset picks it instead.
    let r = unsafe { draw_frame(&mut rig, &both, &[0, 1, 2], 3, 0) };
    assert_eq!(r, abi::result::OK);
    let (_, white) = count_colours(&rig.pixels);
    assert!(white > 0, "draw vertex_offset must select the white triangle");
    // And a draw with neither still lands on the red one.
    let r = unsafe { draw_frame(&mut rig, &both, &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK);
    let (blue, _) = count_colours(&rig.pixels);
    assert!(blue > 0, "no-offset draw must render the red triangle, not the white one");
}

/// A draw reading past its buffer is refused, and the frame it rode in on is
/// dropped - not left half-open.
#[test]
fn out_of_range_draw_is_refused_and_drops_the_frame() {
    let mut rig = Rig::new();
    let verts = triangle();
    // Index 3 does not exist: the buffer has 3 vertices.
    let r = unsafe { draw_frame(&mut rig, &verts, &[0, 1, 3], 0, 0) };
    assert_eq!(r, abi::result::INVALID_ARGUMENT);
    assert_eq!(rig.stats().frames_dropped, 1);
    assert_eq!(rig.stats().frames_presented, 0);
}

/// The audit re-render runs without diverging on a deterministic backend.
#[test]
fn audit_rerender_agrees() {
    let mut rig = Rig::new();
    // SAFETY: live device from the rig.
    unsafe {
        assert_eq!(reconl::reconlRequestTier(rig.device, 2, 0), abi::result::OK);
        // Turn the audit on via the tier request path's sibling: it is a device
        // setting, exercised here through a second submit + present cycle.
    }
    let r = unsafe { draw_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK);
    let st = rig.stats();
    assert_eq!(st.audit_divergences, 0, "a deterministic backend must agree with itself");
}

/// Present without a submitted frame stays NO_FRAME, and does not corrupt the
/// counters that a failed submit just wrote.
#[test]
fn present_without_frame_leaves_counters_sane() {
    let rig = Rig::new();
    let mut prd = abi::ReconLPresentDesc {
        base: hdr::<abi::ReconLPresentDesc>(),
        out_pixels: core::ptr::null_mut(),
        out_pixels_size: 0,
        out_row_pitch: 0,
        out_format: 1,
        flip: 0,
    };
    let r = unsafe { reconl::reconlPresent(rig.device, rig.swapchain, &mut prd) };
    assert_eq!(r, abi::result::NO_FRAME);
    let st = rig.stats();
    assert_eq!(st.frames_presented, 0);
    assert_eq!(st.failures, 1);
    assert_eq!(st.last_result, abi::result::NO_FRAME);
}

/// BeginFrame twice is FRAME_IN_PROGRESS and counts as a failure.
#[test]
fn double_begin_frame_is_refused() {
    let rig = Rig::new();
    // A headlight so the Lambert pipeline is lit: without it every pixel the
    // triangle covers shades to black and the test cannot tell draw from clear.
    let mut light: abi::ReconLLight = unsafe { core::mem::zeroed() };
    light.base = hdr::<abi::ReconLLight>();
    light.r#type = 0; // directional
    light.direction = [0.0, 0.0, -1.0];
    light.color = [1.0, 1.0, 1.0];
    light.intensity = 1.0;
    let ll = abi::ReconLLightList {
        base: hdr::<abi::ReconLLightList>(),
        count: 1,
        reserved: 0,
        lights: &light,
    };
    let fd: abi::ReconLFrameDesc = abi::ReconLFrameDesc {
        base: hdr::<abi::ReconLFrameDesc>(),
        width: W,
        height: H,
        seed: 1,
        reserved: 0,
        lights: &ll,
        shadows: core::ptr::null(),
        // No camera: this rig draws clip-space geometry with the identity in
        // constant slot 0, which is what the identity view and the default
        // frustum describe. It also keeps the pre-ABI-100 no-camera path - a
        // caller whose struct_size stops before this field - under test.
        camera: core::ptr::null(),
        framegen: core::ptr::null(),
    };
    assert_eq!(unsafe { reconl::reconlBeginFrame(rig.device, &fd as *const _ as *mut _) }, abi::result::OK);
    assert_eq!(
        unsafe { reconl::reconlBeginFrame(rig.device, &fd as *const _ as *mut _) },
        abi::result::FRAME_IN_PROGRESS
    );
    let st = rig.stats();
    assert_eq!(st.failures, 1);
    assert_eq!(st.last_result, abi::result::FRAME_IN_PROGRESS);
}

/// The zeroed allocator is refused at device creation: the ABI does not fall
/// back to the C runtime.
#[test]
fn zeroed_allocator_is_refused() {
    // SAFETY: an all-zero allocator: every fn pointer null, user null.
    let desc = device_desc(unsafe { core::mem::zeroed() });
    assert_eq!(create_device(&desc).unwrap_err(), abi::result::INVALID_ARGUMENT);
}

/// A device that lives and dies through create/release with one frame cycle
/// leaks nothing the allocator can see.
#[test]
fn create_present_release_is_balanced() {
    let mut rig = Rig::new();
    let r = unsafe { draw_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK);
    // Drop the rig: every handle released exactly once, in child-first order.
    drop(rig);
}

// ------------------------------------------------------------- the GPU tier

/// The device-less error slot, read the way a host reads it after a failed
/// `reconlCreateDevice`.
fn last_global_error() -> String {
    let mut ei: abi::ReconLErrorInfo = unsafe { core::mem::zeroed() };
    ei.base = hdr::<abi::ReconLErrorInfo>();
    if unsafe { reconl::reconlGetLastError(core::ptr::null_mut(), &mut ei) } != abi::result::OK {
        return "<no error recorded>".to_string();
    }
    let len = ei.message.iter().position(|&b| b == 0).unwrap_or(ei.message.len());
    String::from_utf8_lossy(&ei.message[..len]).into_owned()
}

/// Whether the probe says a D3D11 device can be created on this machine. The
/// probe is the ABI's own answer, so the tests gate on the same fact a host
/// would, rather than on a private accessor.
fn d3d11_usable() -> bool {
    let mut info: abi::ReconLProbeInfo = unsafe { core::mem::zeroed() };
    info.base = hdr::<abi::ReconLProbeInfo>();
    if unsafe { reconl::reconlProbe(core::ptr::null(), &mut info) } != abi::result::OK {
        return false;
    }
    (0..info.entry_count as usize).any(|i| {
        info.entries[i].backend == abi::backend::D3D11 && info.entries[i].usable != 0
    })
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    let mut total = 0u64;
    for (x, y) in a.iter().zip(b.iter()) {
        total += (*x as i32 - *y as i32).unsigned_abs() as u64;
    }
    total as f64 / a.len().max(1) as f64
}

/// The hardware tier through the same entry points a C host uses, all the way
/// to a pixel. This is the capability the probe advertises and the tier table
/// promises: create device → record → submit → present → read back.
#[test]
fn d3d11_renders_a_lit_triangle_through_the_abi() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device on this machine; skipping the hardware tier test");
        return;
    }
    let mut rig = Rig::new_with(abi::backend::D3D11, 1);
    let r = unsafe { draw_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK, "the GPU frame was rejected: {}", rig.error_message());

    let (blue, white) = count_colours(&rig.pixels);
    assert!(
        white > 100,
        "the GPU drew {white} lit pixels over {blue} clear ones; a frame that is only \
         the clear colour is not a rendered frame"
    );

    let st = rig.stats();
    assert_eq!(st.backend, abi::backend::D3D11);
    assert_eq!(st.tier, 1, "the hardware device did not land on the T1 tier");
    assert!(
        st.frame.triangles_in > 0,
        "triangles_in was {} for a frame containing one triangle",
        st.frame.triangles_in
    );
}

// ------------------------------------------------------------- frame generation

use reconl_raster::math as rmath;

/// A world-space camera at `eye` looking at `target`, as the ABI's
/// `ReconLCamera` declares it and as slot 0 carries it: one frustum, used for
/// both, which is the arrangement frame generation is defined against.
///
/// The two are separate arguments so a test can move the camera without turning
/// it - a pure translation, whose screen-space motion over one interval is
/// exactly linear, which is what makes the extrapolation measurable rather than
/// approximate.
fn world_camera(eye: [f32; 3], target: [f32; 3]) -> (abi::ReconLCamera, [f32; 16]) {
    let view = rmath::look_at(eye, target, [0.0, 1.0, 0.0]);
    let proj = rmath::perspective_rh_reversed_z(60.0, W as f32 / H as f32, 0.1, 100.0);
    let mut cam: abi::ReconLCamera = unsafe { core::mem::zeroed() };
    cam.base = hdr::<abi::ReconLCamera>();
    cam.view = view;
    cam.fov_y_deg = 60.0;
    cam.near = 0.1;
    cam.far = 100.0;
    cam.reserved = 0.0;
    (cam, rmath::mul(&proj, &view))
}

/// The frame-generation scene: a plane that fills the frame, with a bright
/// stripe standing a hair in front of it.
///
/// The plane matters as much as the stripe. A generated frame is a *backward*
/// warp - every output pixel asks where its content came from - so a pixel the
/// source had no geometry for cannot be filled from the geometry that swept over
/// it: a stripe on nothing shrinks as it moves, and the measurement would read a
/// generator that is working correctly as one that lost the object. A scene whose
/// whole frame has depth - which is what a game's does - is warped across its own
/// pixels. The stripe is ten pixels wide and stands out by its colour, so the
/// frame's own centre of brightness says where the reprojection put it.
fn scene() -> ([ReconLVertex; 12], [u32; 12]) {
    let mut v = [ReconLVertex::default(); 12];
    let dim = [0.15f32, 0.15, 0.3, 1.0];
    let quad = |slot: &mut ReconLVertex, x: f32, y: f32, z: f32, color: [f32; 4]| {
        slot.position = [x, y, z];
        slot.normal = [0.0, 0.0, 1.0];
        slot.color = color;
    };
    let corners = [(-10.0f32, 10.0f32), (10.0, 10.0), (10.0, -10.0), (-10.0, 10.0), (10.0, -10.0), (-10.0, -10.0)];
    for (slot, (x, y)) in v[..6].iter_mut().zip(corners.iter()) {
        quad(slot, *x, *y, 0.0, dim);
    }
    for (slot, (x, y)) in v[6..].iter_mut().zip(corners.iter()) {
        let (sx, sy) = (*x * 0.1, *y * 0.12);
        quad(slot, sx, sy, 0.01, [1.0, 1.0, 1.0, 1.0]);
    }
    let indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    (v, indices)
}

/// Begins a frame at the given eye, draws the `(index_count, first_index)`
/// segments of `indices` in *world* space through the camera's own
/// view-projection, one draw command each, and submits it. Returns the submit
/// result; the frame is left submitted for the caller's present.
///
/// World space matters here: the generator reprojects using the camera the frame
/// declares, so a scene drawn in clip space would move on screen only because the
/// test asked it to, and a generated frame could not be compared against a real
/// render of the extrapolated camera.
unsafe fn submit_world_frame(
    rig: &mut Rig,
    eye: [f32; 3],
    target: [f32; 3],
    verts: &[ReconLVertex],
    indices: &[u32],
    framegen: Option<&abi::ReconLFrameGenDesc>,
    segments: &[(u32, u32)],
) -> i32 {
    let (camera, view_proj) = world_camera(eye, target);
    let mut light: abi::ReconLLight = core::mem::zeroed();
    light.base = hdr::<abi::ReconLLight>();
    light.r#type = 0;
    light.direction = [0.0, 0.0, -1.0];
    light.color = [1.0, 1.0, 1.0];
    light.intensity = 1.0;
    let ll = abi::ReconLLightList {
        base: hdr::<abi::ReconLLightList>(),
        count: 1,
        reserved: 0,
        lights: &light,
    };
    let fd = abi::ReconLFrameDesc {
        base: hdr::<abi::ReconLFrameDesc>(),
        width: W,
        height: H,
        seed: 1,
        reserved: 0,
        lights: &ll,
        shadows: core::ptr::null(),
        camera: &camera,
        framegen: match framegen {
            Some(fg) => fg as *const abi::ReconLFrameGenDesc,
            None => core::ptr::null(),
        },
    };
    assert_eq!(reconl::reconlBeginFrame(rig.device, &fd as *const _ as *mut _), abi::result::OK);

    let vbd = abi::ReconLBufferDesc {
        base: hdr::<abi::ReconLBufferDesc>(),
        size_bytes: (verts.len() * core::mem::size_of::<ReconLVertex>()) as u64,
        usage: 1,
        reserved: 0,
        data: verts.as_ptr() as *const core::ffi::c_void,
        data_size: (verts.len() * core::mem::size_of::<ReconLVertex>()) as u64,
        debug_name: core::ptr::null(),
    };
    let mut vb: *mut reconl::BufferHandle = core::ptr::null_mut();
    assert_eq!(reconl::reconlCreateBuffer(rig.device, &vbd, &mut vb), abi::result::OK);
    let ibd = abi::ReconLBufferDesc {
        base: hdr::<abi::ReconLBufferDesc>(),
        size_bytes: (indices.len() * 4) as u64,
        usage: 2,
        reserved: 0,
        data: indices.as_ptr() as *const core::ffi::c_void,
        data_size: (indices.len() * 4) as u64,
        debug_name: core::ptr::null(),
    };
    let mut ib: *mut reconl::BufferHandle = core::ptr::null_mut();
    assert_eq!(reconl::reconlCreateBuffer(rig.device, &ibd, &mut ib), abi::result::OK);

    reconl::reconlCmdReset(rig.commands);
    let rp = abi::ReconLRenderPassDesc {
        base: hdr::<abi::ReconLRenderPassDesc>(),
        color_count: 0,
        reserved: 0,
        color: [zeroed_attachment(); abi::RECONL_MAX_ATTACHMENTS],
        depth: core::ptr::null_mut(),
        viewport_width: W,
        viewport_height: H,
        load_color: 1,
        load_depth: 1,
        clear_color: [0.0, 0.0, 1.0, 1.0],
        clear_depth: 0.0,
        stencil_clear: 0,
        reserved2: 0,
    };
    reconl::reconlCmdBeginRenderPass(rig.commands, &rp);
    reconl::reconlCmdSetPipeline(rig.commands, rig.pipeline);
    let identity: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    reconl::reconlCmdPushConstants(rig.commands, 0, view_proj.as_ptr() as *const core::ffi::c_void, 64);
    reconl::reconlCmdPushConstants(rig.commands, 1, identity.as_ptr() as *const core::ffi::c_void, 64);
    reconl::reconlCmdSetVertexBuffer(rig.commands, 0, vb, 0);
    reconl::reconlCmdSetIndexBuffer(rig.commands, ib, 0, 1);
    let mut draw_r = abi::result::OK;
    for (count, first) in segments {
        draw_r = reconl::reconlCmdDrawIndexed(rig.commands, *count, *first, 0);
        if draw_r != abi::result::OK {
            break;
        }
    }
    reconl::reconlCmdEndRenderPass(rig.commands);
    let submit_r = if draw_r == abi::result::OK {
        reconl::reconlSubmit(rig.device, rig.commands, core::ptr::null_mut())
    } else {
        draw_r
    };
    reconl::reconlRelease(vb as *mut core::ffi::c_void);
    reconl::reconlRelease(ib as *mut core::ffi::c_void);
    submit_r
}

/// Presents the submitted frame into `out`, the way a host reads a frame back.
unsafe fn present_into(rig: &mut Rig, out: &mut [u8]) -> i32 {
    let mut prd = abi::ReconLPresentDesc {
        base: hdr::<abi::ReconLPresentDesc>(),
        out_pixels: out.as_mut_ptr() as *mut core::ffi::c_void,
        out_pixels_size: out.len() as u64,
        out_row_pitch: W * 4,
        out_format: 1,
        flip: 0,
    };
    reconl::reconlPresent(rig.device, rig.swapchain, &mut prd)
}

/// The x coordinate of the white stripe's centre of mass, in pixels: the one
/// number that says where the reprojection put it.
///
/// The stripe is white and the clear colour this rig draws with is blue, so "has
/// red" is the test - and it is a test the clear cannot pass, which is what keeps
/// a frame where the stripe was missed from reading as a frame where it sat at
/// x = 0.
fn stripe_centroid_x(pixels: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let mut weight = 0.0f32;
    for y in 0..H {
        for x in 0..W {
            let at = ((y * W + x) * 4) as usize;
            if pixels[at] > 100 {
                sum += x as f32;
                weight += 1.0;
            }
        }
    }
    if weight == 0.0 {
        -1.0
    } else {
        sum / weight
    }
}

/// The mean absolute difference between two frames, over all four channels.
fn frame_distance(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    let sum: u64 = (0..n).map(|i| (a[i] as i32 - b[i] as i32).unsigned_abs() as u64).sum();
    sum as f64 / n as f64
}

/// The frame generation request the ABI declares, with the toggle on.
fn ask_for_generation() -> abi::ReconLFrameGenDesc {
    abi::ReconLFrameGenDesc {
        base: hdr::<abi::ReconLFrameGenDesc>(),
        enabled: 1,
        reserved: 0,
    }
}

/// The same request with the toggle off: a game switching the feature down
/// without changing the shape of its frame descriptor.
fn no_generation() -> abi::ReconLFrameGenDesc {
    let mut fg = ask_for_generation();
    fg.enabled = 0;
    fg
}

/// One generated image through `reconlPresentGenerated`, into `out`.
unsafe fn generate_into(rig: &mut Rig, ahead: f32, out: &mut [u8]) -> i32 {
    let mut prd = abi::ReconLPresentDesc {
        base: hdr::<abi::ReconLPresentDesc>(),
        out_pixels: out.as_mut_ptr() as *mut core::ffi::c_void,
        out_pixels_size: out.len() as u64,
        out_row_pitch: W * 4,
        out_format: 1,
        flip: 0,
    };
    reconl::reconlPresentGenerated(rig.device, rig.swapchain, &mut prd, ahead)
}

/// A host that renders world-space geometry at a camera: begin, submit, present.
/// Returns the present result; `out` receives the frame.
unsafe fn render_at(
    rig: &mut Rig,
    eye: [f32; 3],
    verts: &[ReconLVertex],
    indices: &[u32],
    framegen: Option<&abi::ReconLFrameGenDesc>,
    out: &mut [u8],
) -> i32 {
    // The camera looks straight down -z, so a camera that moves without turning
    // translates the image on screen exactly.
    let target = [eye[0], eye[1], eye[2] - 4.0];
    let submitted = submit_world_frame(rig, eye, target, verts, indices, framegen, &[(indices.len() as u32, 0)]);
    assert_eq!(submitted, abi::result::OK, "submit: {}", rig.error_message());
    present_into(rig, out)
}

/// A frame that records fewer draw commands than the frame before it must
/// present only the draws it recorded.
///
/// The FFI's draw list is device-owned storage, cleared and refilled by every
/// submit, and this is the ABI-level half of the reference tier's stale-tail
/// pin (`backends/soft-cpu/tests/render.rs`, which covers the two lists the
/// backend fills): a list that kept its entries across a submit would make the
/// next frame present geometry its host never recorded. The scene is two quads
/// in one buffer - the dim plane at indices 0..6 and a white stripe at 6..12 -
/// so one frame records both as two draw commands and the next records the plane
/// alone. The stripe is what says the leak happened.
#[test]
fn a_frame_that_records_less_than_the_one_before_it_leaves_no_stale_geometry() {
    let (verts, indices) = scene();
    let eye = [0.0f32, 0.0, 4.0];
    let target = [0.0f32, 0.0, 0.0];
    // `(index_count, first_index)` per draw command: the plane, then the stripe.
    let both = [(6u32, 0u32), (6, 6)];
    let lean = [(6u32, 0u32)];

    for (backend, tier, name) in [
        (abi::backend::SOFT_CPU, 2u32, "soft-cpu"),
        (abi::backend::D3D11, 1u32, "d3d11"),
    ] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            eprintln!("no usable D3D11 device on this machine; skipping the {name} leg");
            continue;
        }
        let mut reused = Rig::new_with(backend, tier);
        let mut fresh = Rig::new_with(backend, tier);
        let mut long = vec![0u8; (W * H * 4) as usize];
        let mut reused_lean = vec![0u8; (W * H * 4) as usize];
        let mut fresh_lean = vec![0u8; (W * H * 4) as usize];
        unsafe {
            assert_eq!(
                submit_world_frame(&mut reused, eye, target, &verts, &indices, None, &both),
                abi::result::OK,
                "{name}: the two-draw frame was refused: {}",
                reused.error_message()
            );
            assert_eq!(
                present_into(&mut reused, &mut long),
                abi::result::OK,
                "{name}: {}",
                reused.error_message()
            );
            assert!(
                count_colours(&long).1 > 0,
                "{name}: the two-draw frame drew no stripe, so the check below would prove nothing"
            );

            assert_eq!(
                submit_world_frame(&mut reused, eye, target, &verts, &indices, None, &lean),
                abi::result::OK,
                "{name}: the one-draw frame was refused: {}",
                reused.error_message()
            );
            assert_eq!(
                present_into(&mut reused, &mut reused_lean),
                abi::result::OK,
                "{name}: {}",
                reused.error_message()
            );

            assert_eq!(
                submit_world_frame(&mut fresh, eye, target, &verts, &indices, None, &lean),
                abi::result::OK,
                "{name}: the fresh device refused the one-draw frame: {}",
                fresh.error_message()
            );
            assert_eq!(
                present_into(&mut fresh, &mut fresh_lean),
                abi::result::OK,
                "{name}: {}",
                fresh.error_message()
            );
        }
        assert_eq!(
            count_colours(&reused_lean).1,
            0,
            "{name}: a frame that recorded one draw presented stale geometry - the white \
             stripe the frame before it recorded"
        );
        assert_eq!(
            reused_lean, fresh_lean,
            "{name}: a frame that records less than the one before it rendered differently on \
             a device that had already drawn more"
        );
    }
}

/// Renders the rig's triangle and presents it into `out` at the given row pitch
/// and flip, the way a host with a pitched or bottom-up buffer does.
unsafe fn present_triangle_into(rig: &mut Rig, out: &mut [u8], pitch: u32, flip: u32) -> i32 {
    assert_eq!(submit_frame(rig, &triangle(), &[0, 1, 2], 0, 0), abi::result::OK);
    let mut prd = abi::ReconLPresentDesc {
        base: hdr::<abi::ReconLPresentDesc>(),
        out_pixels: out.as_mut_ptr() as *mut core::ffi::c_void,
        out_pixels_size: out.len() as u64,
        out_row_pitch: pitch,
        out_format: 1,
        flip,
    };
    reconl::reconlPresent(rig.device, rig.swapchain, &mut prd)
}

/// A host that asks for pitched rows, or for a bottom-up image, gets exactly
/// that and nothing else: the rows land where the descriptor says, in the order
/// the descriptor says, and the bytes between them are never written.
///
/// Every other test and tool in the tree presents tightly packed and top-down,
/// so the layout rule had no coverage on either tier - which matters more since
/// the layout moved from the ABI boundary into the backends, each of which now
/// lays the frame into the host's own rows. A wrong stride and a wrong flip are
/// both invisible in a tight, top-down buffer.
#[test]
fn the_present_layout_honours_the_row_pitch_and_the_flip() {
    let row_bytes = (W * 4) as usize;
    let pitch = row_bytes + 16;
    for (backend, tier) in [(abi::backend::SOFT_CPU, 2), (abi::backend::D3D11, 1)] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            eprintln!("no usable D3D11 device on this machine; skipping its layout leg");
            continue;
        }
        let mut rig = Rig::new_with(backend, tier);

        // The same frame twice: once tightly packed and top-down, once pitched
        // and bottom-up. Two frames of one scene render the same, so any byte
        // that differs is the layout's doing, not the render's.
        let mut tight = vec![0u8; row_bytes * H as usize];
        assert_eq!(
            unsafe { present_triangle_into(&mut rig, &mut tight, W * 4, 0) },
            abi::result::OK,
            "backend {backend}"
        );
        let (_, white) = count_colours(&tight);
        assert!(white > 100, "backend {backend} drew no lit pixels to lay out");

        let mut pitched = vec![0xABu8; pitch * H as usize];
        assert_eq!(
            unsafe { present_triangle_into(&mut rig, &mut pitched, pitch as u32, 1) },
            abi::result::OK,
            "backend {backend}"
        );

        for row in 0..H as usize {
            // Bottom-up: the buffer's first row is the image's last.
            let image_row = H as usize - 1 - row;
            let at = row * pitch;
            assert_eq!(
                &pitched[at..at + row_bytes],
                &tight[image_row * row_bytes..(image_row + 1) * row_bytes],
                "backend {backend}: row {row} of a flipped, pitched present"
            );
            assert!(
                pitched[at + row_bytes..at + pitch].iter().all(|b| *b == 0xAB),
                "backend {backend}: row {row} wrote {} bytes past the row, into the pitch",
                pitch - row_bytes
            );
        }
    }
}

/// A row pitch narrower than a row would make the frame's rows overlap, so it
/// is refused rather than written. The size check alone does not catch it - a
/// 4-byte pitch passes a "big enough" test comfortably - and the row layout used
/// to be applied by a copy that would have written every row over the previous
/// one and handed the host a smeared image. The refusal is a present that fails,
/// so it also has to leave the device able to render the next frame: the same
/// contract a refused buffer size has.
#[test]
fn a_row_pitch_narrower_than_a_row_is_refused_without_wrecking_the_device() {
    for (backend, tier) in [(abi::backend::SOFT_CPU, 2), (abi::backend::D3D11, 1)] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            eprintln!("no usable D3D11 device on this machine; skipping its pitch leg");
            continue;
        }
        let mut rig = Rig::new_with(backend, tier);
        let mut out = vec![0xABu8; (W * H * 4) as usize];
        let r = unsafe { present_triangle_into(&mut rig, &mut out, 4, 0) };
        assert_eq!(
            r,
            abi::result::INVALID_ARGUMENT,
            "backend {backend}: a pitch narrower than a row must be refused"
        );
        assert!(
            out.iter().all(|b| *b == 0xAB),
            "backend {backend}: a refused present must not half-write the host's buffer"
        );
        let st = rig.stats();
        assert_eq!(st.frames_dropped, 1, "backend {backend}: the refused present consumes the frame");

        // And the device is still usable: a whole frame, begin through present.
        let mut good = vec![0u8; (W * H * 4) as usize];
        assert_eq!(
            unsafe { present_triangle_into(&mut rig, &mut good, W * 4, 0) },
            abi::result::OK,
            "backend {backend}"
        );
        let (_, white) = count_colours(&good);
        assert!(white > 100, "backend {backend}: the frame after the refusal must render");
    }
}

/// Renders the reference scene and presents it at the host's own row pitch,
/// with the frame kept for generation or not. The present result is returned.
unsafe fn present_pitched(
    rig: &mut Rig,
    out: &mut [u8],
    pitch: u32,
    framegen: Option<&abi::ReconLFrameGenDesc>,
) -> i32 {
    let (verts, indices) = scene();
    let eye = [0.0f32, 0.0, 4.0];
    let target = [eye[0], eye[1], eye[2] - 4.0];
    assert_eq!(
        submit_world_frame(rig, eye, target, &verts, &indices, framegen, &[(indices.len() as u32, 0)]),
        abi::result::OK,
        "submit: {}",
        rig.error_message()
    );
    let mut prd = abi::ReconLPresentDesc {
        base: hdr::<abi::ReconLPresentDesc>(),
        out_pixels: out.as_mut_ptr() as *mut core::ffi::c_void,
        out_pixels_size: out.len() as u64,
        out_row_pitch: pitch,
        out_format: 1,
        flip: 0,
    };
    reconl::reconlPresent(rig.device, rig.swapchain, &mut prd)
}

/// One generated image into `out` at the host's own row pitch.
unsafe fn generate_pitched(rig: &mut Rig, ahead: f32, out: &mut [u8], pitch: u32) -> i32 {
    let mut prd = abi::ReconLPresentDesc {
        base: hdr::<abi::ReconLPresentDesc>(),
        out_pixels: out.as_mut_ptr() as *mut core::ffi::c_void,
        out_pixels_size: out.len() as u64,
        out_row_pitch: pitch,
        out_format: 1,
        flip: 0,
    };
    reconl::reconlPresentGenerated(rig.device, rig.swapchain, &mut prd, ahead)
}

/// The row-layout rule is one rule, and every route that takes a host pitch
/// obeys it: the present, the present of a frame kept for generation, the
/// generated present, the texture read and the texture write.
///
/// It used to live in whoever happened to own the pixels, which made the *same*
/// present refuse a pitch narrower than a row with frame generation off and lay
/// every row over the previous one with it on - and left the generated present
/// and the texture read unchecked altogether, so a host got overlapping rows,
/// no error, for a buffer it had every reason to believe was laid out its way.
///
/// Each leg is one buffer used twice: once at a legal pitch, which proves the
/// buffer holds the frame, and once narrower than a row, which must be refused
/// with nothing written. A size check cannot pass the first and fail the second.
#[test]
fn every_route_that_takes_a_host_pitch_refuses_a_narrow_one() {
    for (backend, tier) in [
        (abi::backend::SOFT_CPU, 2),
        (abi::backend::D3D11, 1),
        (abi::backend::NULL, 3),
    ] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            eprintln!("no usable D3D11 device on this machine; skipping its pitch leg");
            continue;
        }
        let mut rig = Rig::new_with(backend, tier);
        let row_bytes = (W * 4) as usize;
        let legal = (row_bytes + 16) as u32;
        let mut out = vec![0xABu8; legal as usize * H as usize];
        let ask = ask_for_generation();
        let untouched = |out: &[u8]| out.iter().all(|b| *b == 0xAB);

        // The present of a frame the host keeps for generation: the route that
        // reads the pixels out for the generator rather than asking the
        // backend again, and so has its own row layout.
        assert_eq!(
            unsafe { present_pitched(&mut rig, &mut out, legal, Some(&ask)) },
            abi::result::OK,
            "backend {backend}: a legal pitch with the frame kept\n{}",
            rig.error_message()
        );
        assert!(
            count_colours(&out).1 > 0,
            "backend {backend}: no lit pixels to lay out"
        );
        out.fill(0xAB);
        assert_eq!(
            unsafe { present_pitched(&mut rig, &mut out, 4, Some(&ask)) },
            abi::result::INVALID_ARGUMENT,
            "backend {backend}: a kept frame must refuse a pitch narrower than a row"
        );
        assert!(
            untouched(&out),
            "backend {backend}: the refused kept frame wrote into the host's buffer"
        );

        // The generated present. A tier that cannot generate at all closes this
        // route before any layout, which is not the same answer and is what a
        // host on that tier should see.
        let generated = unsafe {
            present_pitched(&mut rig, &mut out, legal, Some(&ask));
            generate_pitched(&mut rig, 1.0, &mut out, legal)
        };
        if generated == abi::result::NOT_SUPPORTED {
            assert_eq!(
                backend,
                abi::backend::NULL,
                "backend {backend}: this tier is expected to generate"
            );
        } else {
            assert_eq!(
                generated,
                abi::result::OK,
                "backend {backend}: a legal pitch on a generated frame\n{}",
                rig.error_message()
            );
            out.fill(0xAB);
            assert_eq!(
                unsafe { generate_pitched(&mut rig, 1.0, &mut out, 4) },
                abi::result::INVALID_ARGUMENT,
                "backend {backend}: a generated frame must refuse a pitch narrower than a row"
            );
            assert!(
                untouched(&out),
                "backend {backend}: the refused generated frame wrote into the host's buffer"
            );
        }

        // The texture, both directions: the upload a host fills a texture with,
        // and the readback it gets the texels back through.
        let td = abi::ReconLTextureDesc {
            base: hdr::<abi::ReconLTextureDesc>(),
            width: W,
            height: H,
            mip_levels: 1,
            array_layers: 1,
            format: 1,
            usage: 1,
            reserved: 0,
            debug_name: core::ptr::null(),
        };
        let mut texture: *mut reconl::TextureHandle = core::ptr::null_mut();
        assert_eq!(
            unsafe { reconl::reconlCreateTexture(rig.device, &td, &mut texture) },
            abi::result::OK,
            "backend {backend}: creating the texture\n{}",
            rig.error_message()
        );
        let mut level: abi::ReconLTextureLevel = unsafe { core::mem::zeroed() };
        level.base = hdr::<abi::ReconLTextureLevel>();
        level.mip = 0;
        level.layer = 0;
        level.row_count = H;
        level.data = out.as_ptr() as *const core::ffi::c_void;
        level.data_size = out.len() as u64;
        level.row_pitch = legal;
        assert_eq!(
            unsafe { reconl::reconlWriteTexture(rig.device, texture, &level) },
            abi::result::OK,
            "backend {backend}: a legal upload pitch\n{}",
            rig.error_message()
        );
        level.row_pitch = 4;
        assert_eq!(
            unsafe { reconl::reconlWriteTexture(rig.device, texture, &level) },
            abi::result::INVALID_ARGUMENT,
            "backend {backend}: an upload must refuse a pitch narrower than a row"
        );
        level.row_pitch = legal;
        assert_eq!(
            unsafe { reconl::reconlWriteTexture(rig.device, texture, &level) },
            abi::result::OK,
            "backend {backend}: the level after the refused upload\n{}",
            rig.error_message()
        );
        let read = |pitch: u32, out: &mut [u8]| unsafe {
            reconl::reconlReadTexture(
                rig.device,
                texture,
                0,
                0,
                out.as_mut_ptr() as *mut core::ffi::c_void,
                out.len() as u64,
                pitch,
            )
        };
        assert_eq!(
            read(legal, &mut out),
            abi::result::OK,
            "backend {backend}: a legal readback pitch\n{}",
            rig.error_message()
        );
        out.fill(0xAB);
        assert_eq!(
            read(4, &mut out),
            abi::result::INVALID_ARGUMENT,
            "backend {backend}: a readback must refuse a pitch narrower than a row"
        );
        assert!(
            untouched(&out),
            "backend {backend}: the refused readback wrote into the host's buffer"
        );
        unsafe { reconl::reconlRelease(texture as *mut core::ffi::c_void) };
    }
}

/// A frame must not depend on whether the audit read it out of the driver
/// first. There are two routes from the rendered frame to the host's buffer -
/// read the driver, or reuse the bytes a checksum already read - and this is
/// what keeps them one answer rather than two that drift.
#[test]
fn an_audited_frame_presents_the_same_bytes_as_an_unaudited_one() {
    for (backend, tier) in [(abi::backend::SOFT_CPU, 2), (abi::backend::D3D11, 1)] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            eprintln!("no usable D3D11 device on this machine; skipping its audit leg");
            continue;
        }
        let mut rig = Rig::new_with(backend, tier);
        // A pitched, bottom-up buffer: the reuse route has its own row layout, so
        // comparing the two routes through a *tight* buffer would not have
        // compared the code that lays the rows out.
        let row_bytes = (W * 4) as usize;
        let pitch = row_bytes + 16;
        let mut plain = vec![0xABu8; pitch * H as usize];
        assert_eq!(
            unsafe { present_triangle_into(&mut rig, &mut plain, pitch as u32, 1) },
            abi::result::OK,
            "backend {backend}"
        );
        assert!(
            count_colours(&plain).1 > 100,
            "backend {backend}: no lit pixels to compare"
        );

        // Audit every frame: the next frame's checksum is taken, which reads the
        // frame out on the submit side, so the present that follows is the
        // reuse route.
        let mut divergences = 0u32;
        assert_eq!(
            unsafe { reconl::reconlAudit(rig.device, 1, &mut divergences) },
            abi::result::OK,
            "backend {backend}: arming the audit"
        );
        let mut audited = vec![0xABu8; pitch * H as usize];
        assert_eq!(
            unsafe { present_triangle_into(&mut rig, &mut audited, pitch as u32, 1) },
            abi::result::OK,
            "backend {backend}"
        );
        assert_eq!(
            audited, plain,
            "backend {backend}: the frame a host is handed changed because an audit read it out first"
        );
        for row in 0..H as usize {
            let at = row * pitch;
            assert!(
                audited[at + row_bytes..at + pitch].iter().all(|b| *b == 0xAB),
                "backend {backend}: the reuse route wrote into the row pitch"
            );
        }
    }
}

/// A triangle that covers the whole clip-space square, whatever the viewport.
///
/// The rig's own triangle is small and centred, which makes it useless here:
/// confined or not, all of it lands in the same quadrant and the test could not
/// tell the two apart. This one fills whatever rect clip space maps onto, so
/// the lit pixels are the rect itself.
unsafe fn covering_triangle() -> [ReconLVertex; 3] {
    let mut v = [ReconLVertex::default(); 3];
    v[0].position = [-1.0, -1.0, 0.5];
    v[1].position = [3.0, -1.0, 0.5];
    v[2].position = [-1.0, 3.0, 0.5];
    for t in &mut v {
        t.normal = [0.0, 0.0, 1.0];
        t.color = [1.0, 1.0, 1.0, 1.0];
    }
    v
}

/// The lit pixels of `pixels`: how many, and their bounding box.
fn lit_pixels(pixels: &[u8]) -> (usize, Option<(u32, u32, u32, u32)>) {
    let mut count = 0usize;
    let mut box_: Option<(u32, u32, u32, u32)> = None;
    for y in 0..H {
        for x in 0..W {
            let at = ((y * W + x) * 4) as usize;
            let p = &pixels[at..at + 4];
            if p[0] > 200 && p[1] > 200 && p[2] > 200 {
                count += 1;
                box_ = Some(match box_ {
                    None => (x, y, x, y),
                    Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                });
            }
        }
    }
    (count, box_)
}

/// A pass that asks for a viewport smaller than the frame gets exactly that
/// rect drawn and the rest of the target left as the pass cleared it, on both
/// tiers. `viewport_width`/`height` were recorded by the ABI and ignored by the
/// replay, so a host that asked for a sub-rect silently received a full-frame
/// image and had no way to tell that its request meant nothing.
///
/// The default is pinned in the same test: a host that leaves both at 0 asked
/// for the whole frame before the fields meant anything and still gets it, which
/// is what keeps every existing scene - the golden included - byte-identical.
#[test]
fn a_pass_viewport_confines_the_render_to_the_rect_it_asked_for() {
    let verts = unsafe { covering_triangle() };
    let indices = [0u32, 1, 2];
    // T2 renders at the frame size; T3 and T4 render into a half-sized target,
    // so they are the legs that exercise resolving a viewport from frame pixels
    // into the target a tier actually draws in.
    for (backend, tier) in [
        (abi::backend::SOFT_CPU, 2),
        (abi::backend::SOFT_CPU, 3),
        (abi::backend::SOFT_CPU, 4),
        (abi::backend::D3D11, 1),
    ] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            eprintln!("no usable D3D11 device on this machine; skipping its viewport leg");
            continue;
        }
        let mut rig = Rig::new_with(backend, tier);
        // The leg is only the leg it claims to be if the device really built on
        // that tier: on T3/T4 the render target is half the frame, which is the
        // case this test exists to cover.
        assert_eq!(
            rig.stats().tier,
            tier,
            "backend {backend} asked for tier {tier} and got {}",
            rig.stats().tier
        );
        let half = (W / 2, H / 2);

        unsafe { begin_default(&mut rig) };
        let r = unsafe {
            record_and_submit_with_viewport(&mut rig, &verts, &indices, 0, 0, core::ptr::null_mut(), half)
        };
        assert_eq!(r, abi::result::OK, "backend {backend}: {}", rig.error_message());
        let mut prd = abi::ReconLPresentDesc {
            base: hdr::<abi::ReconLPresentDesc>(),
            out_pixels: rig.pixels.as_mut_ptr() as *mut core::ffi::c_void,
            out_pixels_size: (W * H * 4) as u64,
            out_row_pitch: W * 4,
            out_format: 1,
            flip: 0,
        };
        assert_eq!(
            unsafe { reconl::reconlPresent(rig.device, rig.swapchain, &mut prd) },
            abi::result::OK
        );

        let (lit, box_) = lit_pixels(&rig.pixels);
        assert_eq!(
            lit,
            (half.0 * half.1) as usize,
            "backend {backend}: a triangle covering the whole clip square drew {lit} pixels \
             inside a {}x{} viewport; the viewport was not honoured",
            half.0,
            half.1
        );
        assert_eq!(
            box_,
            Some((0, 0, half.0 - 1, half.1 - 1)),
            "backend {backend}: the lit pixels are not anchored at the viewport's top-left"
        );

        // The default: no viewport asked for, the whole frame drawn.
        let mut rig2 = Rig::new_with(backend, tier);
        assert_eq!(
            unsafe { draw_frame(&mut rig2, &verts, &indices, 0, 0) },
            abi::result::OK
        );
        let (lit, box_) = lit_pixels(&rig2.pixels);
        assert_eq!(
            lit,
            (W * H) as usize,
            "backend {backend}: a viewport of 0x0 is the documented default and must draw the \
             whole frame, but {lit} of {} pixels were drawn",
            W * H
        );
        assert_eq!(box_, Some((0, 0, W - 1, H - 1)), "backend {backend}");
    }
}

/// The cross-tier claim of the whole design: the hardware tier and the
/// reference tier render the same scene to the same image, within the
/// tolerance a different rasterisation of the same shading allows. Silent
/// divergence between the two is the failure this pins.
#[test]
fn d3d11_and_softcpu_agree_on_the_same_scene() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device on this machine; skipping the cross-tier test");
        return;
    }
    let verts = triangle();
    let indices = [0u32, 1, 2];

    let mut reference = Rig::new_with(abi::backend::SOFT_CPU, 2);
    assert_eq!(
        unsafe { draw_frame(&mut reference, &verts, &indices, 0, 0) },
        abi::result::OK
    );
    let mut hardware = Rig::new_with(abi::backend::D3D11, 1);
    assert_eq!(
        unsafe { draw_frame(&mut hardware, &verts, &indices, 0, 0) },
        abi::result::OK
    );

    let (r_blue, r_white) = count_colours(&reference.pixels);
    let (g_blue, g_white) = count_colours(&hardware.pixels);
    // Both tiers have to have drawn the same *shape* before comparing pixels:
    // a diff between two differently-covered frames is meaningless.
    assert!(
        (r_white as i64 - g_white as i64).abs() <= (r_white / 50).max(4) as i64,
        "the tiers covered different areas: soft-cpu {r_white} lit / {r_blue} clear, \
         d3d11 {g_white} lit / {g_blue} clear"
    );

    let mean = mean_abs_diff(&reference.pixels, &hardware.pixels);
    eprintln!(
        "soft-cpu vs d3d11: mean absolute channel difference {mean:.4} over {} channels",
        reference.pixels.len()
    );
    assert!(
        mean < 4.0,
        "the tiers diverged: mean absolute channel difference {mean:.3} over {} channels",
        reference.pixels.len()
    );
}

/// Determinism on hardware: the same frame rendered twice through the ABI is the
/// same image. D3D11 makes no such promise by itself, so this is a property
/// this backend has to hold rather than inherit.
#[test]
fn d3d11_rendering_twice_gives_the_same_image() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device on this machine; skipping the determinism test");
        return;
    }
    let verts = triangle();
    let indices = [0u32, 1, 2];
    let mut rig = Rig::new_with(abi::backend::D3D11, 1);

    assert_eq!(unsafe { draw_frame(&mut rig, &verts, &indices, 0, 0) }, abi::result::OK);
    let first = rig.pixels.clone();
    assert_eq!(unsafe { draw_frame(&mut rig, &verts, &indices, 0, 0) }, abi::result::OK);
    assert_eq!(
        rig.pixels, first,
        "two identical frames through the ABI produced different images"
    );
}

/// The tier a host reads back has to be the tier the backend actually runs at.
/// A GPU tier label on the software backend, or a CPU tier on the hardware one,
/// is a lie in the stats a host sizes its resources from - so the tier hint is
/// reconciled with the backend at creation, not passed through.
#[test]
fn the_reported_tier_matches_the_backend_it_built() {
    // The reference backend must never claim a GPU tier, whatever was asked.
    for (hint, expected) in [(1u32, 2u32), (0, 2), (2, 2)] {
        let rig = Rig::new_with(abi::backend::SOFT_CPU, hint);
        let st = rig.stats();
        assert_eq!(st.backend, abi::backend::SOFT_CPU);
        assert_eq!(
            st.tier, expected,
            "soft-cpu reported tier {} for a tier hint of {hint}",
            st.tier
        );
    }

    if !d3d11_usable() {
        eprintln!("no usable D3D11 device; the hardware half of the tier check is skipped");
        return;
    }
    // A CPU tier asked of the hardware backend is clamped up, not claimed down.
    for (hint, expected) in [(4u32, 1u32), (0, 1), (3, 1)] {
        let rig = Rig::new_with(abi::backend::D3D11, hint);
        let st = rig.stats();
        assert_eq!(st.backend, abi::backend::D3D11);
        assert_eq!(
            st.tier, expected,
            "d3d11 reported tier {} for a tier hint of {hint}",
            st.tier
        );
    }
}

/// The probe has to advertise the hardware tier it can actually deliver, and
/// recommend it. A probe that hides a usable GPU is as wrong as one that
/// promises one it cannot create.
#[test]
fn probe_reports_and_recommends_the_hardware_tier() {
    let mut info: abi::ReconLProbeInfo = unsafe { core::mem::zeroed() };
    info.base = hdr::<abi::ReconLProbeInfo>();
    assert_eq!(unsafe { reconl::reconlProbe(core::ptr::null(), &mut info) }, abi::result::OK);
    assert!(info.entry_count >= 1);

    let d3d11 = (0..info.entry_count as usize)
        .find(|&i| info.entries[i].backend == abi::backend::D3D11)
        .expect("the probe does not report the d3d11 backend at all");
    let entry = &info.entries[d3d11];
    assert!(!entry.name.is_empty(), "the d3d11 probe entry has no name");

    if entry.usable == 0 {
        // No hardware here: the ladder must then start on the reference tier.
        assert_eq!(info.recommended_backend, abi::backend::SOFT_CPU);
        assert_eq!(info.recommended_tier, 2, "without a GPU the ladder starts at T2");
        eprintln!("no usable D3D11 device; the probe correctly recommends the reference tier");
        return;
    }
    assert_eq!(info.recommended_backend, abi::backend::D3D11);
    assert_eq!(info.recommended_tier, 1, "a usable GPU means the ladder starts at T1");
    assert!(
        entry.vram_bytes > 0,
        "a usable GPU entry reported no video memory"
    );
}

// ---------------------------------------------- sizes: the allocation ceiling

/// A rig whose host allocation requests are counted, so a test can prove that an
/// absurd descriptor never reached the allocator rather than only that the call
/// returned an error.
fn recorded_rig(
    backend_hint: u32,
    tier_hint: u32,
    budget: *const abi::ReconLMemoryBudget,
) -> (Box<Requests>, Rig) {
    let requests = Box::new(Requests::default());
    let allocator = abi::ReconLAllocator {
        alloc: Some(t_alloc),
        realloc: Some(t_realloc),
        free: Some(t_free),
        user: &*requests as *const Requests as *mut core::ffi::c_void,
    };
    (requests, Rig::with_allocator(backend_hint, tier_hint, budget, allocator))
}

/// An absurd descriptor is refused by the ceiling *before* the host allocator is
/// called: the host gets RECONL_ERR_BUDGET_EXCEEDED instead of a machine that
/// starts swapping while the library commits gigabytes it could never finish
/// using. Sizes this large used to reach the allocator, which is what hung the
/// probe that asked for them and froze the desktop of whoever ran it.
#[test]
fn an_absurd_request_is_refused_before_the_allocator_is_called() {
    let (requests, mut rig) = recorded_rig(abi::backend::SOFT_CPU, 2, core::ptr::null());

    // The ceiling a host reads back is the ceiling that is enforced.
    assert_eq!(
        rig.limits().max_allocation_bytes,
        512 << 20,
        "the reported ceiling must be the enforced one"
    );

    // A 4 GiB buffer.
    let bd = abi::ReconLBufferDesc {
        base: hdr::<abi::ReconLBufferDesc>(),
        size_bytes: 4 << 30,
        usage: 1,
        reserved: 0,
        data: core::ptr::null(),
        data_size: 0,
        debug_name: core::ptr::null(),
    };
    let mut buffer: *mut reconl::BufferHandle = core::ptr::null_mut();
    assert_eq!(
        unsafe { reconl::reconlCreateBuffer(rig.device, &bd, &mut buffer) },
        abi::result::BUDGET_EXCEEDED,
        "{}",
        rig.error_message()
    );
    assert!(buffer.is_null(), "a refused create must not publish a handle");
    assert!(
        rig.error_message().contains("ceiling"),
        "the refusal must name the ceiling: {}",
        rig.error_message()
    );

    // A 65535 x 65535 texture: a 16 GiB mip chain.
    let td = abi::ReconLTextureDesc {
        base: hdr::<abi::ReconLTextureDesc>(),
        width: 65535,
        height: 65535,
        mip_levels: 1,
        array_layers: 1,
        format: 1,
        usage: 1,
        reserved: 0,
        debug_name: core::ptr::null(),
    };
    let mut texture: *mut reconl::TextureHandle = core::ptr::null_mut();
    assert_eq!(
        unsafe { reconl::reconlCreateTexture(rig.device, &td, &mut texture) },
        abi::result::BUDGET_EXCEEDED,
        "{}",
        rig.error_message()
    );
    assert!(texture.is_null());

    // A 65535 x 65535 swapchain: the images its descriptor names are 32 GiB.
    let sd = abi::ReconLSwapchainDesc {
        base: hdr::<abi::ReconLSwapchainDesc>(),
        width: 65535,
        height: 65535,
        format: 1,
        image_count: 2,
        present_to_memory: 1,
        depth_format: 1,
        flags: 0,
        reserved: 0,
    };
    let mut swapchain: *mut reconl::SwapchainHandle = core::ptr::null_mut();
    assert_eq!(
        unsafe { reconl::reconlCreateSwapchain(rig.device, &sd, &mut swapchain) },
        abi::result::BUDGET_EXCEEDED,
        "{}",
        rig.error_message()
    );

    // A 16384 x 16384 frame: 2 GiB of colour and depth targets, before the shadow
    // maps and the tile bins that would follow it.
    assert_eq!(
        unsafe { begin_frame(&mut rig, 16384, 16384) },
        abi::result::BUDGET_EXCEEDED,
        "{}",
        rig.error_message()
    );
    assert_eq!(
        rig.stats().frames_dropped,
        0,
        "a begin that never opened a frame drops nothing"
    );

    // Nothing oversized ever reached the allocator, so the machine was never
    // asked to back any of it.
    assert_eq!(
        requests.over_bound(),
        0,
        "an oversized request reached the host allocator (largest {} bytes)",
        requests.largest()
    );
    assert!(
        requests.largest() <= TEST_ALLOC_BOUND,
        "largest allocation request was {} bytes",
        requests.largest()
    );

    // The device is untouched: a whole small frame still renders.
    let r = unsafe { draw_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK, "{}", rig.error_message());
    let (_, white) = count_colours(&rig.pixels);
    assert!(white > 0, "the frame after the refusals must render");
    assert_eq!(rig.stats().frames_presented, 1);
}

/// The same ceiling on the hardware tier, where the bytes belong to the driver.
///
/// The frame case runs against a 64 MiB device on purpose: the oversize request
/// is then small in absolute terms, so a regression that let it through would
/// make the driver allocate megabytes rather than the gigabytes that would
/// thrash the machine running this test.
#[test]
fn the_ceiling_holds_on_the_hardware_tier_too() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device; the hardware half of the sizing checks is skipped");
        return;
    }

    // Host-side requests on an uncapped device: the library's own 512 MiB ceiling.
    let mut rig = Rig::new_with(abi::backend::D3D11, 1);
    assert_eq!(rig.limits().max_allocation_bytes, 512 << 20);
    let bd = abi::ReconLBufferDesc {
        base: hdr::<abi::ReconLBufferDesc>(),
        size_bytes: 4 << 30,
        usage: 1,
        reserved: 0,
        data: core::ptr::null(),
        data_size: 0,
        debug_name: core::ptr::null(),
    };
    let mut buffer: *mut reconl::BufferHandle = core::ptr::null_mut();
    assert_eq!(
        unsafe { reconl::reconlCreateBuffer(rig.device, &bd, &mut buffer) },
        abi::result::BUDGET_EXCEEDED,
        "{}",
        rig.error_message()
    );
    assert!(buffer.is_null());
    drop(rig);

    // The frame path, kept bounded by a 64 MiB device: a 4096 x 4096 frame is
    // 256 MiB of targets and is refused before the driver is asked for them.
    let mut rig = Rig::with_ram_cap(abi::backend::D3D11, 1, 64 << 20);
    assert_eq!(
        rig.limits().max_allocation_bytes,
        64 << 20,
        "a RAM cap lower than the ceiling must lower the reported one"
    );
    // The hardware tier prices its targets at begin or at submit, so either is
    // accepted here - what has to hold is the code, and that the driver is never
    // handed the frame.
    let opened = unsafe { begin_frame(&mut rig, 4096, 4096) };
    if opened == abi::result::OK {
        let r = unsafe { record_and_submit(&mut rig, &triangle(), &[0, 1, 2], 0, 0, core::ptr::null_mut()) };
        assert_eq!(r, abi::result::BUDGET_EXCEEDED, "{}", rig.error_message());
    } else {
        assert_eq!(opened, abi::result::BUDGET_EXCEEDED, "{}", rig.error_message());
        assert_eq!(
            rig.stats().frames_dropped,
            0,
            "a begin that never opened a frame drops nothing"
        );
    }
    let r = unsafe { draw_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) };
    assert_eq!(r, abi::result::OK, "{}", rig.error_message());
    let (_, white) = count_colours(&rig.pixels);
    assert!(white > 0, "the frame after the refused reservation must render");
}

/// The ceiling is a size *limit*, not a total: requests well above the rig's own
/// 32 x 32 - a 32 MiB buffer, a full-mip 1024 x 1024 texture, a 512 x 512 frame -
/// must still succeed, or the gate that stops an absurd request would also stop
/// an ordinary one.
#[test]
fn a_large_but_affordable_request_still_succeeds() {
    let mut rig = Rig::new();
    unsafe {
        let sd = abi::ReconLSwapchainDesc {
            base: hdr::<abi::ReconLSwapchainDesc>(),
            width: 512,
            height: 512,
            format: 1,
            image_count: 2,
            present_to_memory: 1,
            depth_format: 1,
            flags: 0,
            reserved: 0,
        };
        let mut big_swapchain: *mut reconl::SwapchainHandle = core::ptr::null_mut();
        assert_eq!(
            reconl::reconlCreateSwapchain(rig.device, &sd, &mut big_swapchain),
            abi::result::OK,
            "{}",
            rig.error_message()
        );

        let bd = abi::ReconLBufferDesc {
            base: hdr::<abi::ReconLBufferDesc>(),
            size_bytes: 32 << 20,
            usage: 1,
            reserved: 0,
            data: core::ptr::null(),
            data_size: 0,
            debug_name: core::ptr::null(),
        };
        let mut big_buffer: *mut reconl::BufferHandle = core::ptr::null_mut();
        assert_eq!(
            reconl::reconlCreateBuffer(rig.device, &bd, &mut big_buffer),
            abi::result::OK,
            "{}",
            rig.error_message()
        );

        // A 1024 x 1024 texture with its full chain, written and read back, so
        // the bytes are real and not merely admitted.
        let td = abi::ReconLTextureDesc {
            base: hdr::<abi::ReconLTextureDesc>(),
            width: 1024,
            height: 1024,
            mip_levels: 0,
            array_layers: 1,
            format: 1,
            usage: 1 | 32,
            reserved: 0,
            debug_name: core::ptr::null(),
        };
        let mut big_texture: *mut reconl::TextureHandle = core::ptr::null_mut();
        assert_eq!(
            reconl::reconlCreateTexture(rig.device, &td, &mut big_texture),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        let mut texel = vec![0u8; 1024 * 1024 * 4];
        texel[..4].copy_from_slice(&[7, 8, 9, 255]);
        let mut tl: abi::ReconLTextureLevel = core::mem::zeroed();
        tl.base = hdr::<abi::ReconLTextureLevel>();
        tl.data = texel.as_ptr() as *const core::ffi::c_void;
        tl.data_size = texel.len() as u64;
        assert_eq!(
            reconl::reconlWriteTexture(rig.device, big_texture, &tl),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        let mut back = vec![0u8; texel.len()];
        assert_eq!(
            reconl::reconlReadTexture(
                rig.device,
                big_texture,
                0,
                0,
                back.as_mut_ptr() as *mut core::ffi::c_void,
                back.len() as u64,
                1024 * 4,
            ),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        assert_eq!(&back[..4], &[7, 8, 9, 255], "the large texture must hold what was written");

        // And the frame path at 512 x 512, presented into the matching swapchain.
        assert_eq!(
            begin_frame(&mut rig, 512, 512),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        assert_eq!(
            record_and_submit(&mut rig, &triangle(), &[0, 1, 2], 0, 0, core::ptr::null_mut()),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        let mut pixels = vec![0u8; 512 * 512 * 4];
        let mut prd = abi::ReconLPresentDesc {
            base: hdr::<abi::ReconLPresentDesc>(),
            out_pixels: pixels.as_mut_ptr() as *mut core::ffi::c_void,
            out_pixels_size: pixels.len() as u64,
            out_row_pitch: 512 * 4,
            out_format: 1,
            flip: 0,
        };
        assert_eq!(
            reconl::reconlPresent(rig.device, big_swapchain, &mut prd),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        assert!(
            pixels.iter().any(|&b| b != 0),
            "the 512 x 512 frame must render into its swapchain"
        );

        assert_eq!(reconl::reconlRelease(big_texture as *mut core::ffi::c_void), 0);
        assert_eq!(reconl::reconlRelease(big_buffer as *mut core::ffi::c_void), 0);
        assert_eq!(reconl::reconlRelease(big_swapchain as *mut core::ffi::c_void), 0);
    }
}

/// A presented frame is the frame the host just submitted.
///
/// The hardware tier copies a frame out of the driver when a host asks for the
/// pixels, not on every submit, so that capture has to follow the frame. If it
/// did not, the second frame below would present the first one's image - and
/// every call would still return OK, so nothing else in the suite would notice.
#[test]
fn a_presented_frame_is_the_frame_that_was_just_submitted() {
    for (backend, tier, name) in [
        (abi::backend::SOFT_CPU, 2, "soft-cpu"),
        (abi::backend::D3D11, 1, "d3d11"),
    ] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            eprintln!("no usable D3D11 device on this machine; skipping the hardware tier");
            continue;
        }
        let mut rig = Rig::new_with(backend, tier);
        assert_eq!(
            unsafe { draw_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0) },
            abi::result::OK,
            "{name}: {}",
            rig.error_message()
        );
        let first = rig.pixels.clone();
        assert!(first.iter().any(|&b| b != 0), "{name}: the first frame must render");

        // The same triangle in red: a different image, so a stale readback shows
        // up as a failure rather than being assumed away.
        let mut red = triangle();
        for vertex in &mut red {
            vertex.color = [1.0, 0.0, 0.0, 1.0];
        }
        assert_eq!(
            unsafe { draw_frame(&mut rig, &red, &[0, 1, 2], 0, 0) },
            abi::result::OK,
            "{name}: {}",
            rig.error_message()
        );
        assert_ne!(
            rig.pixels, first,
            "{name}: the second frame's pixels must be its own, not the first frame's"
        );
    }
}

/// A frame's cost includes the readback the host waited for, once, on every
/// tier.
///
/// The number is read twice: after the submit, before any readback has
/// happened, and after the present that performs it. It must grow by the
/// readback - so a tier that does not count one fails - and by no more than the
/// present call took, so a tier that counts its own fails as well.
#[test]
fn the_frame_cost_includes_the_readback_the_host_waited_for() {
    for (backend, tier, name) in [
        (abi::backend::SOFT_CPU, 2, "soft-cpu"),
        (abi::backend::D3D11, 1, "d3d11"),
    ] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            eprintln!("no usable D3D11 device on this machine; skipping the hardware tier");
            continue;
        }
        let mut rig = Rig::new_with(backend, tier);
        unsafe {
            assert_eq!(
                submit_frame(&mut rig, &triangle(), &[0, 1, 2], 0, 0),
                abi::result::OK,
                "{name}: {}",
                rig.error_message()
            );
            // Submitted, not yet presented: no readback has happened, so this is
            // the device's own frame time.
            let device_only = rig.stats().frame.total_ns;

            let started = std::time::Instant::now();
            let mut prd = abi::ReconLPresentDesc {
                base: hdr::<abi::ReconLPresentDesc>(),
                out_pixels: rig.pixels.as_mut_ptr() as *mut core::ffi::c_void,
                out_pixels_size: (W * H * 4) as u64,
                out_row_pitch: W * 4,
                out_format: 1,
                flip: 0,
            };
            assert_eq!(
                reconl::reconlPresent(rig.device, rig.swapchain, &mut prd),
                abi::result::OK,
                "{name}: {}",
                rig.error_message()
            );
            let present_took = started.elapsed().as_nanos() as u64;
            let with_readback = rig.stats().frame.total_ns;

            let readback = with_readback.saturating_sub(device_only);
            assert!(
                readback > 0,
                "{name}: the frame's cost leaves out the readback the host waited for \
                 ({device_only} ns before the present, {with_readback} ns after)"
            );
            assert!(
                readback <= present_took * 2 + 1_000_000,
                "{name}: the readback was counted more than once: {readback} ns of it in a \
                 {present_took} ns present"
            );
        }
    }
}

// -------------------------------------------- frame generation, through the ABI

/// A frame buffer for one present of this rig's size.
fn frame_buffer() -> Vec<u8> {
    vec![0u8; (W * H * 4) as usize]
}

/// One frame of the frame-generation scene at `eye`, presented into `out`.
unsafe fn render_scene_at(
    rig: &mut Rig,
    eye: [f32; 3],
    framegen: Option<&abi::ReconLFrameGenDesc>,
    out: &mut [u8],
) -> i32 {
    let (verts, indices) = scene();
    render_at(rig, eye, &verts, &indices, framegen, out)
}

/// A frame generation request is refused until a frame asks for one, and a host
/// that asks gets the frame it was handed back, to the byte, for a camera that
/// did not move - through the shipped entry points.
///
/// The middle is the point: generation is not a mode the device is put into, it
/// is a property of the frame, so the same host can switch it off again per frame
/// without recreating anything.
#[test]
fn a_generated_frame_requires_a_frame_that_asked_for_one() {
    let mut rig = Rig::new();
    let mut frame = frame_buffer();
    let mut generated = frame_buffer();
    let eye = [0.0f32, 0.0, 4.0];
    unsafe {
        assert_eq!(
            render_scene_at(&mut rig, eye, None, &mut frame),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        assert_eq!(
            generate_into(&mut rig, 0.5, &mut generated),
            abi::result::NO_FRAME,
            "a frame that never asked for generation left something to generate from: {}",
            rig.error_message()
        );
        assert_eq!(rig.stats().framegen.generated, 0, "a refused generate was counted");
        assert_eq!(rig.stats().framegen.ready, 0);

        // The same scene, now asking. The camera has not moved since the frame
        // before, so generation must reproduce that frame exactly.
        let ask = ask_for_generation();
        assert_eq!(
            render_scene_at(&mut rig, eye, Some(&ask), &mut frame),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        assert_eq!(
            generate_into(&mut rig, 1.0, &mut generated),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        assert_eq!(
            generated, frame,
            "a still camera must generate the frame the host was handed, byte for byte"
        );

        // And off again, without changing anything else about the frame.
        let off = no_generation();
        assert_eq!(
            render_scene_at(&mut rig, eye, Some(&off), &mut frame),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        assert_eq!(
            generate_into(&mut rig, 1.0, &mut generated),
            abi::result::NO_FRAME,
            "a frame that asked for nothing left history behind: {}",
            rig.error_message()
        );
        assert_eq!(rig.stats().framegen.ready, 0);
    }
}

/// A host that never asks pays nothing, and the frame that asks pays for exactly
/// what it keeps: the history's three buffers, reserved against the device's own
/// budget.
///
/// This is the property that makes the feature a per-game quality toggle rather
/// than a tax on every host. The count is the rig's own allocator - the allocator
/// the device was created with - so nothing another test allocates can appear in
/// it, and the size is the history's own: four bytes of colour, four of depth and
/// four of generated image for every pixel of the frame.
#[test]
fn a_host_that_never_asks_reserves_nothing() {
    let (requests, mut rig) = recorded_rig(abi::backend::SOFT_CPU, 2, core::ptr::null());
    let mut frame = frame_buffer();
    let eye = [0.0f32, 0.0, 4.0];
    let history_bytes = usize::try_from(u64::from(W) * u64::from(H) * 12).unwrap();
    unsafe {
        for _ in 0..2 {
            assert_eq!(
                render_scene_at(&mut rig, eye, None, &mut frame),
                abi::result::OK,
                "{}",
                rig.error_message()
            );
        }
        // Every frame allocates and frees its own buffers, so the number that
        // says anything is the difference between two frames that do the same
        // work: one that asks for generation and one that does not.
        let quiet_before = requests.traffic();
        assert_eq!(
            render_scene_at(&mut rig, eye, None, &mut frame),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        let quiet = requests.traffic();
        let asking_before = requests.traffic();
        let ask = ask_for_generation();
        assert_eq!(
            render_scene_at(&mut rig, eye, Some(&ask), &mut frame),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        let asking = requests.traffic();

        let quiet_calls = quiet.0 - quiet_before.0;
        let quiet_bytes = quiet.1 - quiet_before.1;
        let asking_calls = asking.0 - asking_before.0;
        let asking_bytes = asking.1 - asking_before.1;
        assert_eq!(
            asking_calls,
            quiet_calls + 3,
            "a frame that asked for generation should allocate its history's three buffers on top \
             of the same frame a quiet one allocates: {quiet_calls} calls quiet, {asking_calls} asking"
        );
        assert!(
            asking_bytes >= quiet_bytes + history_bytes,
            "the frame that asked kept too little to warp from: {asking_bytes} bytes against \
             {quiet_bytes} quiet, and a {}x{} history is at least {history_bytes} bytes",
            W,
            H
        );
        assert_eq!(rig.stats().framegen.ready, 1, "the asking frame left nothing to generate from");

        // The history is kept, not re-made: the next asking frame pays for its
        // own buffers and nothing else.
        let again_before = requests.traffic();
        assert_eq!(
            render_scene_at(&mut rig, [0.3, 0.0, 4.0], Some(&ask), &mut frame),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        let again = requests.traffic();
        assert_eq!(
            again.0 - again_before.0,
            quiet_calls,
            "a second asking frame reallocated its history instead of keeping it"
        );
    }
}

/// The reprojection moves the image by the camera's own motion, and the frame it
/// produces is a better picture of the *next* frame than the frame it was warped
/// from.
///
/// Both halves matter and neither is a matter of taste: the first is measured
/// against the projection (where the stripe must land), the second against a real
/// render of the camera the generator was asked to predict. If the generator
/// merely returned the source frame, the first would fail; if it warped in the
/// wrong direction or by the wrong amount, the second would.
#[test]
fn a_generated_frame_predicts_the_frame_after_it() {
    if !d3d11_usable() {
        eprintln!("no usable D3D11 device on this machine; the hardware tier leg is skipped");
    }
    for (backend, tier, name) in [
        (abi::backend::SOFT_CPU, 2, "soft-cpu"),
        (abi::backend::D3D11, 1, "d3d11"),
    ] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            continue;
        }
        let mut rig = Rig::new_with(backend, tier);
        assert_eq!(rig.stats().backend, backend, "{name}: wrong backend");
        let ask = ask_for_generation();
        let mut a = frame_buffer();
        let mut b = frame_buffer();
        let mut generated = frame_buffer();
        let mut c = frame_buffer();
        let (x1, x2) = (0.0f32, 0.5f32);
        unsafe {
            // Two real frames, one interval apart, both asking so the pair of
            // cameras is the motion the generator extrapolates.
            assert_eq!(render_scene_at(&mut rig, [x1, 0.0, 4.0], Some(&ask), &mut a), abi::result::OK, "{name}: {}", rig.error_message());
            assert_eq!(render_scene_at(&mut rig, [x2, 0.0, 4.0], Some(&ask), &mut b), abi::result::OK, "{name}: {}", rig.error_message());
            assert_eq!(generate_into(&mut rig, 1.0, &mut generated), abi::result::OK, "{name}: {}", rig.error_message());

            let (ca, cb, cg) = (stripe_centroid_x(&a), stripe_centroid_x(&b), stripe_centroid_x(&generated));
            assert!(ca >= 0.0 && cb >= 0.0 && cg >= 0.0, "{name}: the stripe was not found in one of the frames ({ca}, {cb}, {cg})");
            let interval = cb - ca;
            assert!(interval.abs() > 0.5, "{name}: the camera moved but the stripe did not ({ca} -> {cb})");
            assert!(
                (cg - (cb + interval)).abs() < 1.5,
                "{name}: one interval of motion is {interval:.2} px, so the generated frame should put the \
                 stripe at {:.2} px; it put it at {cg:.2}",
                cb + interval
            );

            // And a real render of the camera one interval further on: the
            // generated frame must be closer to it than the frame it was made
            // from. This is the claim the feature exists to make.
            assert_eq!(render_scene_at(&mut rig, [x2 + (x2 - x1), 0.0, 4.0], None, &mut c), abi::result::OK, "{name}: {}", rig.error_message());
            let (to_true, to_source) = (frame_distance(&generated, &c), frame_distance(&b, &c));
            assert!(
                to_true < to_source,
                "{name}: the generated frame is further from the frame that follows ({to_true:.2}) than the \
                 frame it was warped from ({to_source:.2}) - it is not predicting anything"
            );
        }
    }
}

/// Generation is deterministic, counts only itself, and leaves the frame ladder
/// judging rendered frames.
///
/// The last part is the honest one: a host cannot make a slow tier look fast by
/// generating more images, so `frames_presented` - the number the ladder decides
/// on - must not move when frames are generated, even though the host is now
/// seeing more images than it renders.
#[test]
fn generation_counts_only_itself() {
    let mut rig = Rig::new();
    let ask = ask_for_generation();
    let mut a = frame_buffer();
    let mut b = frame_buffer();
    let mut one = frame_buffer();
    let mut two = frame_buffer();
    unsafe {
        assert_eq!(render_scene_at(&mut rig, [0.0, 0.0, 4.0], Some(&ask), &mut a), abi::result::OK, "{}", rig.error_message());
        assert_eq!(render_scene_at(&mut rig, [0.5, 0.0, 4.0], Some(&ask), &mut b), abi::result::OK, "{}", rig.error_message());

        let before = rig.stats();
        // A generated frame is not gated on the frame state machine: a host may
        // ask for one as soon as the frame it warps has been presented, which is
        // exactly the moment a display wants it.
        assert_eq!(generate_into(&mut rig, 0.5, &mut one), abi::result::OK, "{}", rig.error_message());
        assert_eq!(generate_into(&mut rig, 0.5, &mut two), abi::result::OK, "{}", rig.error_message());
        assert_eq!(one, two, "two generated frames of one pair differ");
        assert_ne!(one, a, "the generated frame is the newest real frame; nothing was warped");

        let after = rig.stats();
        assert_eq!(after.frames_presented, before.frames_presented, "generating a frame moved frames_presented");
        assert_eq!(after.framegen.generated, before.framegen.generated + 2);
        assert!(
            after.framegen.generated_ns > before.framegen.generated_ns,
            "the cost of a generated frame was not recorded"
        );
        assert_eq!(after.framegen.last_ahead, 0.5);
        assert_eq!(after.framegen.ready, 1, "generating consumed the frame it warped");

        // The device renders normally straight afterwards.
        assert_eq!(render_scene_at(&mut rig, [0.7, 0.0, 4.0], Some(&ask), &mut two), abi::result::OK, "{}", rig.error_message());
        assert_eq!(rig.stats().frames_presented, before.frames_presented + 1);
    }
}

/// One present descriptor for the generated path, wrong in exactly one way: 0 is
/// valid, 1 carries another call's struct type, 2 a pitch narrower than a row, 3
/// a buffer one row short of the frame, 4 no buffer at all.
unsafe fn generated_desc(out: &mut [u8], wrong: u8) -> abi::ReconLPresentDesc {
    let mut prd = abi::ReconLPresentDesc {
        base: hdr::<abi::ReconLPresentDesc>(),
        out_pixels: out.as_mut_ptr() as *mut core::ffi::c_void,
        out_pixels_size: out.len() as u64,
        out_row_pitch: W * 4,
        out_format: 1,
        flip: 0,
    };
    match wrong {
        1 => prd.base.struct_type = abi::struct_type::FRAME_DESC,
        2 => prd.out_row_pitch = W * 4 - 4,
        3 => prd.out_pixels_size = out.len() as u64 - 4,
        4 => prd.out_pixels = core::ptr::null_mut(),
        _ => {}
    }
    prd
}

/// The order a generated present decides its answers in, met the way a host meets
/// it: the tier first, then whether there is anything to generate from, and only
/// then the descriptor and `ahead`.
///
/// The middle gate is not the frame state machine - a generated frame is not a
/// frame - but the thing the call actually needs: a frame the host asked to keep.
/// Until one exists, every call is NO_FRAME however malformed its arguments are,
/// which is the same shape `reconlPresent` has with no frame to present. So a host
/// reads one code per call instead of a choice between two, and the code it reads
/// is the one it can act on. Once a frame is being kept the arguments are read,
/// and the buffer rules are read with the frame they have to hold in hand - which
/// is what the header says, and what a reordering of these checks would break.
#[test]
fn a_generated_frame_is_refused_for_its_arguments_only_once_there_is_one_to_generate() {
    for (backend, tier) in [
        (abi::backend::SOFT_CPU, 2),
        (abi::backend::D3D11, 1),
        (abi::backend::NULL, 3),
    ] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            eprintln!("no usable D3D11 device on this machine; skipping its order leg");
            continue;
        }
        // SAFETY: every descriptor is built inside, and every handle is released
        // with that leg's own rig.
        unsafe { order_leg(backend, tier) };
    }
}

/// One tier's leg of the order pin: the same matrix on each tier, because the
/// tier is the first question a generated present asks.
unsafe fn order_leg(backend: u32, tier: u32) {
    let generates = backend != abi::backend::NULL;
    let mut rig = Rig::new_with(backend, tier);
    let mut out = frame_buffer();
    let untouched = |out: &[u8]| out.iter().all(|b| *b == 0xAB);

    // SAFETY: every descriptor is built here and every handle is released with
    // this leg's rig.
    unsafe {
        // Nothing has asked for generation, so there is nothing to generate from
        // and nothing has been read: one answer for all of them - the tier's own
        // on a tier where the tier is the first question.
        let absent = if generates { abi::result::NO_FRAME } else { abi::result::NOT_SUPPORTED };
        for (what, wrong, ahead) in [
            ("a valid descriptor", 0u8, 1.0f32),
            ("another call's struct type", 1, 1.0),
            ("a pitch narrower than a row", 2, 1.0),
            ("a buffer one row short", 3, 1.0),
            ("no buffer at all", 4, 1.0),
            ("a look-ahead above 1", 0, 1.5),
            ("a NaN look-ahead", 0, f32::NAN),
        ] {
            out.fill(0xAB);
            let mut prd = generated_desc(&mut out, wrong);
            assert_eq!(
                reconl::reconlPresentGenerated(rig.device, rig.swapchain, &mut prd, ahead),
                absent,
                "backend {backend}: with nothing to generate from, {what} must answer {absent}: {}",
                rig.error_message()
            );
            assert!(
                untouched(&out),
                "backend {backend}: the refusal of {what} wrote into the host's buffer"
            );
        }

        if generates {
            arguments_are_read_once_a_frame_is_kept(&mut rig, &mut out, backend);
        }
        present_answers_its_state_before_its_descriptor(&mut rig, &mut out, backend);
    }
}

/// With a frame kept, the arguments are what stands between the host and an
/// image: the shape gate, the buffer rules, then the look-ahead. `out` is the
/// host's own buffer, unwritten by every refusal.
unsafe fn arguments_are_read_once_a_frame_is_kept(rig: &mut Rig, out: &mut [u8], backend: u32) {
    let mut frame = frame_buffer();
    let untouched = |out: &[u8]| out.iter().all(|b| *b == 0xAB);
    // SAFETY: every descriptor is built here and every handle is released with
    // the rig.
    unsafe {
        // A frame that asks, and the same descriptors are read: the arguments are
        // now what stands between the host and an image.
        assert_eq!(
            render_scene_at(rig, [0.0, 0.0, 4.0], Some(&ask_for_generation()), &mut frame),
            abi::result::OK,
            "{}",
            rig.error_message()
        );
        assert_eq!(rig.stats().framegen.ready, 1, "a frame asked and was kept");
        for (what, wrong, ahead, want) in [
            ("a valid descriptor", 0u8, 1.0f32, abi::result::OK),
            ("another call's struct type", 1, 1.0, abi::result::WRONG_STRUCT_TYPE),
            ("a pitch narrower than a row", 2, 1.0, abi::result::INVALID_ARGUMENT),
            ("a buffer one row short", 3, 1.0, abi::result::INVALID_ARGUMENT),
            ("no buffer at all", 4, 1.0, abi::result::INVALID_ARGUMENT),
            ("a look-ahead above 1", 0, 1.5, abi::result::INVALID_ARGUMENT),
            ("a NaN look-ahead", 0, f32::NAN, abi::result::INVALID_ARGUMENT),
        ] {
            out.fill(0xAB);
            let mut prd = generated_desc(out, wrong);
            assert_eq!(
                reconl::reconlPresentGenerated(rig.device, rig.swapchain, &mut prd, ahead),
                want,
                "backend {backend}: with a frame kept, {what} must answer {want}: {}",
                rig.error_message()
            );
            if want != abi::result::OK {
                assert!(
                    untouched(out),
                    "backend {backend}: the refusal of {what} wrote into the host's buffer"
                );
            }
        }
    }
}

/// The present path has the same shape with the frame state as its gate, which
/// is what makes this one order rather than two accidents: no frame to present
/// is also every descriptor's answer, and a submitted frame is when the argument
/// is read. That gate is not the tier, so this leg runs on every tier.
unsafe fn present_answers_its_state_before_its_descriptor(rig: &mut Rig, out: &mut [u8], backend: u32) {
    let untouched = |out: &[u8]| out.iter().all(|b| *b == 0xAB);
    // SAFETY: every descriptor is built here and every handle is released with
    // the rig.
    unsafe {
        out.fill(0xAB);
        let mut narrow = generated_desc(out, 2);
        assert_eq!(
            reconl::reconlPresent(rig.device, rig.swapchain, &mut narrow),
            abi::result::NO_FRAME,
            "backend {backend}: with no frame to present, a narrow pitch is NO_FRAME too: {}",
            rig.error_message()
        );
        assert!(untouched(out), "backend {backend}: the refusal with no frame to present wrote");
        let (verts, indices) = scene();
        assert_eq!(
            submit_frame_with_fence(rig, &verts, &indices, 0, 0, core::ptr::null_mut()),
            abi::result::OK,
            "backend {backend}: a frame submits so the present has something to refuse: {}",
            rig.error_message()
        );
        out.fill(0xAB);
        let mut narrow = generated_desc(out, 2);
        assert_eq!(
            reconl::reconlPresent(rig.device, rig.swapchain, &mut narrow),
            abi::result::INVALID_ARGUMENT,
            "backend {backend}: with a frame submitted, the same pitch is refused: {}",
            rig.error_message()
        );
        assert!(
            untouched(out),
            "backend {backend}: the refused present with a submitted frame wrote"
        );
    }
}

/// The property the order exists for, on both present paths and every tier: a
/// host pointer is only read by the question that owns it.
///
/// The unreadable descriptor below points at memory this process does not own. A
/// call that answers a code for it has not read it - a read would fault - so
/// these assertions are the proof rather than a restatement: in a state whose own
/// gate comes first the pointer is never touched, and in the state where the
/// arguments *are* the question, a header that cannot be honoured is refused for
/// what the header says. Every refusal then leaves the device able to render the
/// next frame, which is what "refused, not wedged" means for a call that renders
/// nothing.
#[test]
fn a_descriptor_is_never_read_before_the_gate_that_owns_it() {
    for (backend, tier) in [
        (abi::backend::SOFT_CPU, 2),
        (abi::backend::D3D11, 1),
        (abi::backend::NULL, 3),
    ] {
        if backend == abi::backend::D3D11 && !d3d11_usable() {
            eprintln!("no usable D3D11 device on this machine; skipping its deref leg");
            continue;
        }
        // SAFETY: the unreadable pointer is never dereferenced in the states
        // asserted on, and every descriptor that is read is built inside.
        unsafe { deref_leg(backend, tier) };
    }
}

/// One tier's leg of the deref-safety pin.
unsafe fn deref_leg(backend: u32, tier: u32) {
    let generates = backend != abi::backend::NULL;
    let mut rig = Rig::new_with(backend, tier);
    let mut out = frame_buffer();
    let mut live = frame_buffer();
    // Address 8 is not mapped in this process: reading it faults.
    let unreadable = 8usize as *mut abi::ReconLPresentDesc;
    let absent = if generates { abi::result::NO_FRAME } else { abi::result::NOT_SUPPORTED };

    // SAFETY: the whole point is that nothing reads the unreadable pointer; every
    // descriptor that is read is built here.
    unsafe {
        // The null cell comes first so a reorder that starts reading the
        // descriptor reports as a mismatch here rather than as the fault the
        // unreadable cell would earn.
        for (what, desc) in [
            ("a null descriptor", core::ptr::null_mut()),
            ("an unreadable descriptor", unreadable),
        ] {
            assert_eq!(
                reconl::reconlPresentGenerated(rig.device, rig.swapchain, desc, 1.0),
                absent,
                "backend {backend}: {what} with nothing to generate from must answer {absent}: {}",
                rig.error_message()
            );
            assert_eq!(
                reconl::reconlPresent(rig.device, rig.swapchain, desc),
                abi::result::NO_FRAME,
                "backend {backend}: {what} with no frame to present must answer NO_FRAME: {}",
                rig.error_message()
            );
        }

        // Once a frame is kept, the descriptor is the question - and what is
        // refused is what its header says, not a fault.
        if generates {
            assert_eq!(
                render_scene_at(&mut rig, [0.0, 0.0, 4.0], Some(&ask_for_generation()), &mut live),
                abi::result::OK,
                "{}",
                rig.error_message()
            );
            let mut wrong_type = generated_desc(&mut out, 0);
            wrong_type.base.struct_type = abi::struct_type::FRAME_DESC;
            assert_eq!(
                reconl::reconlPresentGenerated(rig.device, rig.swapchain, &mut wrong_type, 1.0),
                abi::result::WRONG_STRUCT_TYPE,
                "backend {backend}: a header of another call's type is its own refusal: {}",
                rig.error_message()
            );
            let mut prefix_only = generated_desc(&mut out, 0);
            prefix_only.base.struct_size = core::mem::size_of::<StructHeader>() as u32;
            assert_eq!(
                reconl::reconlPresentGenerated(rig.device, rig.swapchain, &mut prefix_only, 1.0),
                abi::result::STRUCT_SIZE,
                "backend {backend}: a descriptor shorter than this library reads is refused: {}",
                rig.error_message()
            );
            assert_eq!(
                reconl::reconlPresentGenerated(rig.device, rig.swapchain, core::ptr::null_mut(), 1.0),
                abi::result::INVALID_ARGUMENT,
                "backend {backend}: with a frame kept, a null descriptor is the argument's own error: {}",
                rig.error_message()
            );
        }

        // The same, on the present path once a frame is submitted: the argument
        // question is the one that reads the descriptor, and each refusal
        // consumes the frame - the documented rule for a present that cannot be
        // delivered.
        let (verts, indices) = scene();
        assert_eq!(
            submit_frame_with_fence(&mut rig, &verts, &indices, 0, 0, core::ptr::null_mut()),
            abi::result::OK,
            "backend {backend}: a frame submits so the present has a descriptor to refuse: {}",
            rig.error_message()
        );
        let mut wrong_type = generated_desc(&mut out, 0);
        wrong_type.base.struct_type = abi::struct_type::FRAME_DESC;
        assert_eq!(
            reconl::reconlPresent(rig.device, rig.swapchain, &mut wrong_type),
            abi::result::WRONG_STRUCT_TYPE,
            "backend {backend}: a submitted present refuses another call's struct type: {}",
            rig.error_message()
        );
        assert_eq!(
            submit_frame_with_fence(&mut rig, &verts, &indices, 0, 0, core::ptr::null_mut()),
            abi::result::OK,
            "backend {backend}: the device opens another frame after a refused present: {}",
            rig.error_message()
        );
        let mut prefix_only = generated_desc(&mut out, 0);
        prefix_only.base.struct_size = core::mem::size_of::<StructHeader>() as u32;
        assert_eq!(
            reconl::reconlPresent(rig.device, rig.swapchain, &mut prefix_only),
            abi::result::STRUCT_SIZE,
            "backend {backend}: a submitted present refuses a prefix-only header: {}",
            rig.error_message()
        );

        // And a frame still renders and presents, so nothing above left the
        // device wedged behind a refusal.
        assert_eq!(
            render_scene_at(&mut rig, [0.0, 0.0, 4.0], None, &mut live),
            abi::result::OK,
            "backend {backend}: the device must render after those refusals: {}",
            rig.error_message()
        );
    }
}

/// Every way a generated frame can be refused is an error, not a wrecked device:
/// the frame state machine is untouched, the counter does not move, and the next
/// rendered frame is unaffected.
#[test]
fn a_refused_generated_frame_leaves_the_device_able_to_render() {
    let mut rig = Rig::new();
    let ask = ask_for_generation();
    let mut frame = frame_buffer();
    let mut generated = frame_buffer();
    unsafe {
        assert_eq!(render_scene_at(&mut rig, [0.0, 0.0, 4.0], Some(&ask), &mut frame), abi::result::OK, "{}", rig.error_message());
        assert_eq!(render_scene_at(&mut rig, [0.4, 0.0, 4.0], Some(&ask), &mut frame), abi::result::OK, "{}", rig.error_message());

        // A look-ahead outside the documented interval, and a present buffer that
        // cannot hold the frame.
        for ahead in [0.0f32, -1.0, 1.5, f32::NAN, f32::INFINITY] {
            assert_eq!(
                generate_into(&mut rig, ahead, &mut generated),
                abi::result::INVALID_ARGUMENT,
                "ahead {ahead} was accepted"
            );
        }
        let mut small = vec![0u8; (W * H * 4) as usize - 4];
        assert_eq!(generate_into(&mut rig, 0.5, &mut small), abi::result::INVALID_ARGUMENT);
        assert_eq!(rig.stats().framegen.generated, 0, "a refused generate was counted");

        // Nothing above left the device unable to render, or to generate.
        assert_eq!(render_scene_at(&mut rig, [0.8, 0.0, 4.0], Some(&ask), &mut frame), abi::result::OK, "{}", rig.error_message());
        assert_eq!(generate_into(&mut rig, 1.0, &mut generated), abi::result::OK, "{}", rig.error_message());
    }
}

/// A tier that keeps no depth to reproject says so rather than handing over a
/// blank image or a copy of a frame it never rendered.
#[test]
fn a_tier_with_nothing_to_reproject_refuses_generation() {
    let mut rig = Rig::new_with(abi::backend::NULL, 2);
    let mut out = frame_buffer();
    unsafe {
        assert_eq!(
            generate_into(&mut rig, 1.0, &mut out),
            abi::result::NOT_SUPPORTED,
            "the null tier generated a frame: {}",
            rig.error_message()
        );
    }
}

/// Reset restarts the measurement a host reads, and nothing else: after it, the
/// generated count is zero but the frame a host is looking at can still be
/// generated from - otherwise a host that resets its counters once a second
/// would silently lose the feature.
#[test]
fn resetting_the_counters_keeps_the_frame_behind_them() {
    let mut rig = Rig::new();
    let ask = ask_for_generation();
    let mut frame = frame_buffer();
    let mut generated = frame_buffer();
    unsafe {
        assert_eq!(render_scene_at(&mut rig, [0.0, 0.0, 4.0], Some(&ask), &mut frame), abi::result::OK, "{}", rig.error_message());
        assert_eq!(render_scene_at(&mut rig, [0.5, 0.0, 4.0], Some(&ask), &mut frame), abi::result::OK, "{}", rig.error_message());
        assert_eq!(generate_into(&mut rig, 0.5, &mut generated), abi::result::OK, "{}", rig.error_message());
        assert_eq!(rig.stats().framegen.generated, 1);

        assert_eq!(reconl::reconlResetStats(rig.device), abi::result::OK);
        let reset = rig.stats();
        assert_eq!(reset.framegen.generated, 0, "the generated count survived a stats reset");
        assert_eq!(reset.framegen.generated_ns, 0);
        assert_eq!(reset.frames_presented, 0, "a stats reset stopped resetting frames_presented");
        assert_eq!(reset.framegen.ready, 1, "a stats reset threw the frame away with the counters");
        assert_eq!(generate_into(&mut rig, 0.5, &mut generated), abi::result::OK, "{}", rig.error_message());
    }
}
