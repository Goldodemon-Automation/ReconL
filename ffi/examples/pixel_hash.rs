//! The canonical conformance frame, rendered through the Rust side of the ABI.
//!
//! This exists so the Java FFM suite has something to be equal to. Both sides
//! build the *same* scene from the same header - a full-clip unlit white
//! triangle over a blue clear, 64x64, presented to memory on the reference
//! tier - and print the SHA-256 of the RGBA8 bytes. If the two digests differ,
//! one of the two FFI layers disagrees with the header, and a conformance suite
//! that could not catch that would not be worth running.
//!
//! The reference tier is deliberate: it defines correctness, it is
//! deterministic, and it needs no GPU, so a digest mismatch means the binding is
//! wrong rather than the driver.
//!
//! The hash is implemented here rather than pulled in: the workspace has no
//! external dependencies on purpose, and a conformance digest is not worth one.
//!
//! Usage: cargo run -p reconl-ffi --example pixel_hash [--backend=soft-cpu] [--size=64]
#![allow(non_snake_case)]
use reconl::abi;
use reconl_core::{ABIStruct, StructHeader};

fn hdr<H: ABIStruct>() -> StructHeader {
    StructHeader::new(core::mem::size_of::<H>() as u32, H::STRUCT_TYPE)
}

// ------------------------------------------------------------------ SHA-256

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, hh].into_iter().enumerate() {
            h[i] = h[i].wrapping_add(v);
        }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

// ------------------------------------------------------------------ allocator
//
// The library's `free` callback carries a size but no alignment, so a layout
// cannot be reconstructed at free time - the C probe sidesteps this by using
// malloc, which ignores both. A registry is the honest equivalent: every
// allocation remembers the exact `Layout` it was made with, and nothing is
// inferred at free time.
use std::alloc::Layout;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

static ALLOCS: LazyLock<Mutex<HashMap<usize, Layout>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

unsafe extern "C" fn h_alloc(_u: *mut core::ffi::c_void, s: usize, a: usize) -> *mut core::ffi::c_void {
    let align = a.clamp(16, 4096).max(core::mem::align_of::<usize>());
    let Ok(layout) = Layout::from_size_align(s.max(1), align) else {
        return core::ptr::null_mut();
    };
    let p = std::alloc::alloc(layout);
    if p.is_null() {
        return core::ptr::null_mut();
    }
    ALLOCS.lock().expect("allocator registry").insert(p as usize, layout);
    p as *mut core::ffi::c_void
}

unsafe extern "C" fn h_free(_u: *mut core::ffi::c_void, p: *mut core::ffi::c_void, _s: usize) {
    if p.is_null() {
        return;
    }
    if let Some(layout) = ALLOCS.lock().expect("allocator registry").remove(&(p as usize)) {
        std::alloc::dealloc(p as *mut u8, layout);
    }
}

unsafe extern "C" fn h_realloc(
    _u: *mut core::ffi::c_void,
    p: *mut core::ffi::c_void,
    o: usize,
    n: usize,
    a: usize,
) -> *mut core::ffi::c_void {
    let fresh = h_alloc(_u, n, a);
    if !p.is_null() && !fresh.is_null() {
        core::ptr::copy_nonoverlapping(p as *const u8, fresh as *mut u8, o.min(n));
        h_free(_u, p, o);
    }
    fresh
}

// ---------------------------------------------------------------------- scene

fn check(rc: i32, what: &str) -> Result<(), String> {
    if rc == abi::result::OK {
        Ok(())
    } else {
        Err(format!("{what}: {rc}"))
    }
}

fn render(backend: u32, size: u32) -> Result<Vec<u8>, String> {
    let mut device: *mut reconl::DeviceHandle = core::ptr::null_mut();
    let mut dd: abi::ReconLDeviceDesc = unsafe { core::mem::zeroed() };
    dd.base = hdr::<abi::ReconLDeviceDesc>();
    dd.backend_hint = backend;
    dd.seed = 7;
    dd.allocator.alloc = Some(h_alloc);
    dd.allocator.realloc = Some(h_realloc);
    dd.allocator.free = Some(h_free);
    check(unsafe { reconl::reconlCreateDevice(&dd, &mut device) }, "create device")?;

    let mut sc: *mut reconl::SwapchainHandle = core::ptr::null_mut();
    let mut sd: abi::ReconLSwapchainDesc = unsafe { core::mem::zeroed() };
    sd.base = hdr::<abi::ReconLSwapchainDesc>();
    sd.width = size;
    sd.height = size;
    sd.format = abi::format::R8G8B8A8_UNORM;
    sd.image_count = 2;
    sd.present_to_memory = 1;
    sd.depth_format = abi::format::D32_FLOAT;
    check(unsafe { reconl::reconlCreateSwapchain(device, &sd, &mut sc) }, "create swapchain")?;

    let mut cl: *mut reconl::CommandListHandle = core::ptr::null_mut();
    let mut cd: abi::ReconLCommandListDesc = unsafe { core::mem::zeroed() };
    cd.base = hdr::<abi::ReconLCommandListDesc>();
    cd.capacity_bytes = 8192;
    check(unsafe { reconl::reconlCreateCommandList(device, &cd, &mut cl) }, "create command list")?;

    // One triangle covering the whole clip-space cube, unlit white.
    let pos: [[f32; 3]; 3] = [[-1.0, -1.0, 0.5], [3.0, -1.0, 0.5], [-1.0, 3.0, 0.5]];
    let mut verts: [abi::ReconLVertex; 3] = unsafe { core::mem::zeroed() };
    for (i, v) in verts.iter_mut().enumerate() {
        v.position = pos[i];
        v.normal = [0.0, 0.0, 1.0];
        v.uv = [0.0, 0.0];
        v.color = [1.0, 1.0, 1.0, 1.0];
    }
    let idx: [u32; 3] = [0, 1, 2];

    let mut vb: *mut reconl::BufferHandle = core::ptr::null_mut();
    let mut bd: abi::ReconLBufferDesc = unsafe { core::mem::zeroed() };
    bd.base = hdr::<abi::ReconLBufferDesc>();
    bd.size_bytes = core::mem::size_of_val(&verts) as u64;
    bd.usage = abi::buffer_usage::VERTEX;
    bd.data = verts.as_ptr() as *const core::ffi::c_void;
    bd.data_size = bd.size_bytes;
    check(unsafe { reconl::reconlCreateBuffer(device, &bd, &mut vb) }, "create vertex buffer")?;

    let mut ib: *mut reconl::BufferHandle = core::ptr::null_mut();
    let mut bd: abi::ReconLBufferDesc = unsafe { core::mem::zeroed() };
    bd.base = hdr::<abi::ReconLBufferDesc>();
    bd.size_bytes = core::mem::size_of_val(&idx) as u64;
    bd.usage = abi::buffer_usage::INDEX;
    bd.data = idx.as_ptr() as *const core::ffi::c_void;
    bd.data_size = bd.size_bytes;
    check(unsafe { reconl::reconlCreateBuffer(device, &bd, &mut ib) }, "create index buffer")?;

    let mut pipe: *mut reconl::PipelineHandle = core::ptr::null_mut();
    let mut pd: abi::ReconLPipelineDesc = unsafe { core::mem::zeroed() };
    pd.base = hdr::<abi::ReconLPipelineDesc>();
    pd.shading = abi::shading::UNLIT;
    pd.blend = 0; // RECONL_BLEND_OPAQUE
    pd.cull = 0; // RECONL_CULL_NONE
    pd.depth_compare = 1; // RECONL_COMPARE_GREATER, reversed-Z
    pd.depth_write = 1;
    check(unsafe { reconl::reconlCreatePipeline(device, &pd, &mut pipe) }, "create pipeline")?;

    let mut fd: abi::ReconLFrameDesc = unsafe { core::mem::zeroed() };
    fd.base = hdr::<abi::ReconLFrameDesc>();
    fd.width = size;
    fd.height = size;
    fd.seed = 1;
    check(unsafe { reconl::reconlBeginFrame(device, &mut fd) }, "begin frame")?;

    let mut rp: abi::ReconLRenderPassDesc = unsafe { core::mem::zeroed() };
    rp.base = hdr::<abi::ReconLRenderPassDesc>();
    rp.load_color = 1;
    rp.load_depth = 1;
    rp.clear_color = [0.0, 0.0, 1.0, 1.0];
    rp.clear_depth = 0.0;

    unsafe { reconl::reconlCmdReset(cl) };
    check(unsafe { reconl::reconlCmdBeginRenderPass(cl, &rp) }, "begin render pass")?;
    let ident: [f32; 16] = core::array::from_fn(|i| if i % 5 == 0 { 1.0 } else { 0.0 });
    unsafe { reconl::reconlCmdSetPipeline(cl, pipe) };
    unsafe { reconl::reconlCmdPushConstants(cl, 0, ident.as_ptr() as *const core::ffi::c_void, 64) };
    unsafe { reconl::reconlCmdPushConstants(cl, 1, ident.as_ptr() as *const core::ffi::c_void, 64) };
    check(unsafe { reconl::reconlCmdSetVertexBuffer(cl, 0, vb, 0) }, "set vertex buffer")?;
    check(
        unsafe { reconl::reconlCmdSetIndexBuffer(cl, ib, 0, abi::index_format::UINT32) },
        "set index buffer",
    )?;
    check(unsafe { reconl::reconlCmdDrawIndexed(cl, 3, 0, 0) }, "draw")?;
    check(unsafe { reconl::reconlCmdEndRenderPass(cl) }, "end render pass")?;
    check(unsafe { reconl::reconlSubmit(device, cl, core::ptr::null_mut()) }, "submit")?;

    let mut pixels = vec![0u8; (size as usize) * (size as usize) * 4];
    let mut pr: abi::ReconLPresentDesc = unsafe { core::mem::zeroed() };
    pr.base = hdr::<abi::ReconLPresentDesc>();
    pr.out_pixels = pixels.as_mut_ptr() as *mut core::ffi::c_void;
    pr.out_pixels_size = pixels.len() as u64;
    pr.out_row_pitch = size * 4;
    pr.out_format = abi::format::R8G8B8A8_UNORM;
    check(unsafe { reconl::reconlPresent(device, sc, &mut pr) }, "present")?;

    unsafe {
        reconl::reconlRelease(pipe as *mut core::ffi::c_void);
        reconl::reconlRelease(vb as *mut core::ffi::c_void);
        reconl::reconlRelease(ib as *mut core::ffi::c_void);
        reconl::reconlRelease(cl as *mut core::ffi::c_void);
        reconl::reconlRelease(sc as *mut core::ffi::c_void);
        reconl::reconlRelease(device as *mut core::ffi::c_void);
    }
    Ok(pixels)
}

fn main() {
    let mut backend = abi::backend::SOFT_CPU;
    let mut size = 64u32;
    let mut sizes_only = false;
    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("--backend=") {
            backend = match v {
                "null" => abi::backend::NULL,
                "d3d11" => abi::backend::D3D11,
                _ => abi::backend::SOFT_CPU,
            };
        } else if let Some(v) = arg.strip_prefix("--size=") {
            size = v.parse().unwrap_or(64);
        } else if arg == "--sizes" {
            sizes_only = true;
        }
    }
    // The struct sizes are part of the ABI: a caller whose `struct_size` does
    // not match is refused with ERR_STRUCT_SIZE, so any other FFI layer binding
    // this header has to agree with these numbers exactly.
    if sizes_only {
        println!("ReconLBase {}", core::mem::size_of::<reconl_core::StructHeader>());
        println!("ReconLAllocator {}", core::mem::size_of::<abi::ReconLAllocator>());
        println!("ReconLDeviceDesc {}", core::mem::size_of::<abi::ReconLDeviceDesc>());
        println!("ReconLSwapchainDesc {}", core::mem::size_of::<abi::ReconLSwapchainDesc>());
        println!("ReconLCommandListDesc {}", core::mem::size_of::<abi::ReconLCommandListDesc>());
        println!("ReconLBufferDesc {}", core::mem::size_of::<abi::ReconLBufferDesc>());
        println!("ReconLPipelineDesc {}", core::mem::size_of::<abi::ReconLPipelineDesc>());
        println!("ReconLRenderPassDesc {}", core::mem::size_of::<abi::ReconLRenderPassDesc>());
        println!("ReconLFrameDesc {}", core::mem::size_of::<abi::ReconLFrameDesc>());
        println!("ReconLPresentDesc {}", core::mem::size_of::<abi::ReconLPresentDesc>());
        println!("ReconLVertex {}", core::mem::size_of::<abi::ReconLVertex>());
        println!("ReconLStats {}", core::mem::size_of::<abi::ReconLStats>());
        println!("ReconLErrorInfo {}", core::mem::size_of::<abi::ReconLErrorInfo>());
        return;
    }
    match render(backend, size) {
        Ok(pixels) => {
            println!("size={size} backend={backend} bytes={}", pixels.len());
            println!("sha256={}", hex(&sha256(&pixels)));
        }
        Err(e) => {
            eprintln!("pixel_hash: {e}");
            std::process::exit(2);
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
