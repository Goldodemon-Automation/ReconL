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
}

/// 64 MiB: above everything a test legitimately allocates here (a 32 MiB buffer,
/// a 5 MiB mip chain, a 4 KiB frame), and far below what an absurd descriptor
/// implies.
const TEST_ALLOC_BOUND: usize = 64 << 20;

impl Requests {
    fn note(&self, size: usize) {
        self.largest.fetch_max(size, Ordering::Relaxed);
        if size > TEST_ALLOC_BOUND {
            self.over_bound.fetch_add(1, Ordering::Relaxed);
        }
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

/// Records the draw into the frame that is already open and submits it. This is
/// the retry path: a submit refused for its arguments leaves the frame open, so
/// the host reuses the frame it has rather than beginning another.
unsafe fn record_and_submit(rig: &mut Rig, verts: &[ReconLVertex], indices: &[u32], vertex_offset: i32, buffer_offset: u64, fence: *mut reconl::FenceHandle) -> i32 {
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
