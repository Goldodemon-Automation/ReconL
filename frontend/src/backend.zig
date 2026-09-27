//! The rendering half of the frontend: it takes the finished mesh the widget
//! layer produced and presents pixels through the ReconL C ABI.
//!
//! This module deliberately knows nothing about widgets. It knows one thing:
//! the frame contract documented in `include/reconl/reconl.h` - begin, reset
//! the command list, open a pass, push the transforms, draw, end the pass,
//! submit, present - and it performs that sequence exactly the way
//! `tools/host/src/frame.rs` does, so a diff between a Zig-driven frame and a
//! Rust-driven frame can only contain UI.
//!
//! It is a separate module from the core on purpose: `zig build test` never
//! links libreconl, while everything here does (see `build.zig`'s module
//! graph). Types are shared through the `reconl-frontend` module import, so
//! `geom.Vec2` here is the same type widgets used.
//!
//! The device uses automatic backend selection with the tier ladder disabled:
//! the hardware renderer is used when available, with the reference backend as
//! the fallback. The showcase still cannot quietly change tiers between frames.

const std = @import("std");
const front = @import("reconl-frontend");
const geom = front.geom;
const tess = front.tess;
const ui_mod = front.ui;

const c = @cImport({
    @cInclude("reconl/reconl.h");
});

pub const Error = error{
    OutOfMemory,
    ReconLError,
};

/// Creation failures have no `Renderer` to record into yet, so they land in
/// this module-level slot; `createError()` reads it back for the C layer.
threadlocal var g_create_error: [256]u8 = undefined;
threadlocal var g_create_error_len: usize = 0;

pub fn createError() []const u8 {
    return g_create_error[0..g_create_error_len];
}

fn recordCreate(comptime what: []const u8, r: c_int) void {
    const name: []const u8 = if (r == 0) "unknown error" else std.mem.span(c.reconlResultName(r));
    const n = std.fmt.bufPrint(&g_create_error, "{s} failed: {s} ({d})", .{ what, name, r }) catch return;
    g_create_error_len = n.len;
}

/// The versioned-struct header every descriptor carries
/// (`struct_size`, `type`, `next = null`).
fn hdr(comptime T: type, comptime stype: c_int) c.ReconLBase {
    return .{
        .struct_size = @sizeOf(T),
        .type = @intCast(stype),
        .next = null,
    };
}

// ---- host allocator -------------------------------------------------------
//
// A conforming allocator is a hard requirement of `reconlCreateDevice` - the
// library refuses a zeroed one. The C contract gives `free` no alignment, so
// every block carries its bookkeeping in 16 bytes of slack in front of the
// returned pointer: the base pointer and the total size, exactly the layout
// `tools/host/src/alloc.rs` uses, so both hosts round-trip the same way.

const slack: usize = 16;

fn allocatorOf(user: ?*anyopaque) ?*std.mem.Allocator {
    const u = user orelse return null;
    const p: *std.mem.Allocator = @ptrCast(@alignCast(u));
    return p;
}

fn abiAlloc(user: ?*anyopaque, size: usize, alignment: usize) callconv(.c) ?*anyopaque {
    const gpa = allocatorOf(user) orelse return null;
    const align_wanted = std.math.clamp(alignment, 16, 4096);
    const sz = @max(size, 1);
    const total = std.math.add(usize, sz, align_wanted) catch return null;
    const total2 = std.math.add(usize, total, slack) catch return null;
    const base = gpa.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(16), total2) catch return null;
    const raw = @intFromPtr(base.ptr);
    // `align_wanted >= 16` makes the aligned interior pointer at least
    // `raw + slack` in, so both words below land inside the block.
    const aligned = std.mem.alignForward(usize, raw + slack, align_wanted);
    @as(*align(1) usize, @ptrFromInt(aligned - 16)).* = raw;
    @as(*align(1) usize, @ptrFromInt(aligned - 8)).* = total2;
    return @ptrFromInt(aligned);
}

fn abiFree(user: ?*anyopaque, ptr: ?*anyopaque, size: usize) callconv(.c) void {
    _ = size; // the slack header is the source of truth, as in alloc.rs
    const p = ptr orelse return;
    const gpa = allocatorOf(user) orelse return;
    const aligned = @intFromPtr(p);
    const raw = @as(*align(1) usize, @ptrFromInt(aligned - 16)).*;
    const total2 = @as(*align(1) usize, @ptrFromInt(aligned - 8)).*;
    const typed: [*]align(16) u8 = @ptrFromInt(raw);
    gpa.free(typed[0..total2]);
}

fn abiRealloc(
    user: ?*anyopaque,
    ptr: ?*anyopaque,
    old_size: usize,
    new_size: usize,
    alignment: usize,
) callconv(.c) ?*anyopaque {
    const fresh = abiAlloc(user, new_size, alignment) orelse return null;
    if (ptr) |old| {
        const n = @min(old_size, new_size);
        if (n > 0) {
            const dst: [*]u8 = @ptrCast(fresh);
            const src: [*]const u8 = @ptrCast(old);
            @memcpy(dst[0..n], src[0..n]);
        }
        abiFree(user, ptr, old_size);
    }
    return fresh;
}

// ---- renderer -------------------------------------------------------------

/// One UI frame rendered through ReconL: the resources a frame reuses and the
/// pixel buffer contract. Owned handles are released on `deinit`.
///
/// The vertex/index buffers are rewritten every frame (`DYNAMIC`) because the
/// widget layer is immediate-mode: the mesh is new each frame, the pipeline is
/// not. Buffers grow geometrically and are only recreated when the mesh
/// outgrows them.
pub const Renderer = struct {
    gpa: std.mem.Allocator,
    /// Heap box the device's allocator callbacks point at (`user` must stay
    /// valid for the device's life; a stack address would not).
    alloc_box: *std.mem.Allocator,
    device: ?*c.ReconLDevice,
    swapchain: ?*c.ReconLSwapchain,
    pipeline: ?*c.ReconLPipeline,
    commands: ?*c.ReconLCommandList,
    vb: ?*c.ReconLBuffer = null,
    ib: ?*c.ReconLBuffer = null,
    vb_bytes: u64 = 0,
    ib_bytes: u64 = 0,
    width: u32,
    height: u32,
    clear: [4]f32,
    frame_index: u32 = 0,
    err_buf: [256]u8 = undefined,
    err_len: usize = 0,

    /// Creates the device and frame resources using the highest available
    /// backend; the tier ladder is off to keep the selected backend fixed.
    pub fn init(gpa: std.mem.Allocator, width: u32, height: u32, clear: geom.Color) Error!Renderer {
        const w = @max(width, 1);
        const h = @max(height, 1);

        const box = try gpa.create(std.mem.Allocator);
        errdefer gpa.destroy(box);
        box.* = gpa;

        var device: ?*c.ReconLDevice = null;
        var dd: c.ReconLDeviceDesc = .{
            .base = hdr(c.ReconLDeviceDesc, c.RECONL_STRUCT_DEVICE_DESC),
            .backend_hint = @intCast(c.RECONL_BACKEND_NONE), // prefer available hardware; otherwise reference
            .tier_hint = 0, // let ReconL select the highest usable tier
            .allow_downgrade = 0, // no ladder: two frames of a demo must match
            .worker_threads = 0, // 0 = the library chooses
            .target_frame_ms = 0, // 0 = no frame-time ladder either
            .downgrade_after_frames = 16,
            .seed = 7, // never wall-clock (determinism contract)
            .flags = 0,
            .budget = null,
            .allocator = .{
                .alloc = abiAlloc,
                .realloc = abiRealloc,
                .free = abiFree,
                .user = box,
            },
            .backend_desc = null,
        };
        var r = c.reconlCreateDevice(&dd, &device);
        if (r != 0 or device == null) {
            recordCreate("reconlCreateDevice", r);
            return error.ReconLError;
        }
        errdefer _ = c.reconlRelease(device);

        var swapchain: ?*c.ReconLSwapchain = null;
        r = c.reconlCreateSwapchain(device, &swapDesc(w, h), &swapchain);
        if (r != 0) {
            recordCreate("reconlCreateSwapchain", r);
            return error.ReconLError;
        }
        errdefer _ = c.reconlRelease(swapchain);

        var pipeline: ?*c.ReconLPipeline = null;
        r = c.reconlCreatePipeline(device, &pipelineDesc(), &pipeline);
        if (r != 0) {
            recordCreate("reconlCreatePipeline", r);
            return error.ReconLError;
        }
        errdefer _ = c.reconlRelease(pipeline);

        var commands: ?*c.ReconLCommandList = null;
        r = c.reconlCreateCommandList(device, &cmdListDesc(), &commands);
        if (r != 0) {
            recordCreate("reconlCreateCommandList", r);
            return error.ReconLError;
        }
        errdefer _ = c.reconlRelease(commands);

        return .{
            .gpa = gpa,
            .alloc_box = box,
            .device = device,
            .swapchain = swapchain,
            .pipeline = pipeline,
            .commands = commands,
            .width = w,
            .height = h,
            .clear = .{ clear.r, clear.g, clear.b, clear.a },
        };
    }

    pub fn deinit(self: *Renderer) void {
        if (self.vb) |b| _ = c.reconlRelease(b);
        if (self.ib) |b| _ = c.reconlRelease(b);
        if (self.commands) |l| _ = c.reconlRelease(l);
        if (self.pipeline) |p| _ = c.reconlRelease(p);
        if (self.swapchain) |s| _ = c.reconlRelease(s);
        if (self.device) |d| _ = c.reconlRelease(d);
        const gpa = self.gpa;
        gpa.destroy(self.alloc_box);
        self.* = undefined;
    }

    /// Recreates the swapchain for a new surface size. The pipeline and the
    /// buffers are size-independent, so they survive.
    pub fn resize(self: *Renderer, width: u32, height: u32) Error!void {
        const w = @max(width, 1);
        const h = @max(height, 1);
        if (w == self.width and h == self.height) return;
        const old = self.swapchain;
        var swapchain: ?*c.ReconLSwapchain = null;
        const r = c.reconlCreateSwapchain(self.device, &swapDesc(w, h), &swapchain);
        if (r != 0) {
            self.record("reconlCreateSwapchain", r);
            return error.ReconLError;
        }
        // Swap only after the new one exists: a failed resize keeps the old
        // swapchain and the old size, never a dangling pair.
        _ = c.reconlRelease(old);
        self.swapchain = swapchain;
        self.width = w;
        self.height = h;
    }

    pub fn setClear(self: *Renderer, clear: geom.Color) void {
        self.clear = .{ clear.r, clear.g, clear.b, clear.a };
    }

    pub fn lastError(self: *const Renderer) []const u8 {
        return self.err_buf[0..self.err_len];
    }

    /// Renders `mesh` and presents into `pixels` (RGBA8, `width*height*4`).
    ///
    /// An empty mesh never touches the ABI - a frame with no geometry can be
    /// refused by the library's empty-frame policy, and the honest answer to
    /// "the UI drew nothing" is the clear colour anyway.
    pub fn render(self: *Renderer, mesh: ui_mod.Mesh, pixels: []u8) Error!void {
        const need: usize = @as(usize, self.width) * @as(usize, self.height) * 4;
        if (pixels.len < need) return error.ReconLError;

        if (mesh.indices.len == 0) {
            self.fillClear(pixels[0..need]);
            self.frame_index +%= 1;
            return;
        }

        try self.ensureBuffers(mesh);

        const vbytes: u64 = mesh.vertices.len * @sizeOf(tess.Vertex);
        const ibytes: u64 = mesh.indices.len * @sizeOf(u32);
        var r = c.reconlWriteBuffer(self.device, self.vb, 0, @ptrCast(mesh.vertices.ptr), vbytes);
        if (r != 0) {
            self.record("reconlWriteBuffer (vertices)", r);
            return error.ReconLError;
        }
        r = c.reconlWriteBuffer(self.device, self.ib, 0, @ptrCast(mesh.indices.ptr), ibytes);
        if (r != 0) {
            self.record("reconlWriteBuffer (indices)", r);
            return error.ReconLError;
        }

        const seed = self.frame_index;

        // ---- begin ------------------------------------------------------
        var fd: c.ReconLFrameDesc = .{
            .base = hdr(c.ReconLFrameDesc, c.RECONL_STRUCT_FRAME_DESC),
            .width = self.width,
            .height = self.height,
            .seed = seed,
            .reserved = 0,
            // Null camera = identity view + default frustum: right for
            // geometry that is already in clip space via the ortho push.
            .lights = null,
            .shadows = null,
            .camera = null,
            .framegen = null,
        };
        r = c.reconlBeginFrame(self.device, &fd);
        if (r != 0) {
            self.record("reconlBeginFrame", r);
            return error.ReconLError;
        }

        // ---- record -----------------------------------------------------
        const rp = self.renderPassDesc();
        _ = c.reconlCmdReset(self.commands);
        r = c.reconlCmdBeginRenderPass(self.commands, &rp);
        if (r != 0) {
            self.abortFrame("reconlCmdBeginRenderPass", r);
            return error.ReconLError;
        }

        // Slot 0: view-projection - the ortho that maps pixel space (y down)
        // to clip space (y up), constant z = 0.5, which passes reversed-Z
        // GREATER against the 0.0 clear. Slot 1: the model, identity - the
        // mesh is already in surface pixels.
        const view_proj = geom.ortho(@floatFromInt(self.width), @floatFromInt(self.height));
        if (!self.cmdPush(0, &view_proj)) return error.ReconLError;
        if (!self.cmdPush(1, &geom.IDENTITY)) return error.ReconLError;

        r = c.reconlCmdSetPipeline(self.commands, self.pipeline);
        if (r != 0) {
            self.abortFrame("reconlCmdSetPipeline", r);
            return error.ReconLError;
        }
        r = c.reconlCmdSetVertexBuffer(self.commands, 0, self.vb, 0);
        if (r != 0) {
            self.abortFrame("reconlCmdSetVertexBuffer", r);
            return error.ReconLError;
        }
        r = c.reconlCmdSetIndexBuffer(self.commands, self.ib, 0, @intCast(c.RECONL_INDEX_UINT32));
        if (r != 0) {
            self.abortFrame("reconlCmdSetIndexBuffer", r);
            return error.ReconLError;
        }
        r = c.reconlCmdDrawIndexed(self.commands, @intCast(mesh.indices.len), 0, 0);
        if (r != 0) {
            self.abortFrame("reconlCmdDrawIndexed", r);
            return error.ReconLError;
        }
        r = c.reconlCmdEndRenderPass(self.commands);
        if (r != 0) {
            // Close the frame as best we can, then report: an open frame
            // would poison every subsequent BeginFrame.
            self.record("reconlCmdEndRenderPass", r);
            _ = c.reconlSubmit(self.device, self.commands, null);
            return error.ReconLError;
        }

        // ---- submit -----------------------------------------------------
        r = c.reconlSubmit(self.device, self.commands, null);
        if (r != 0) {
            self.record("reconlSubmit", r);
            return error.ReconLError;
        }

        // ---- present ----------------------------------------------------
        var pd: c.ReconLPresentDesc = .{
            .base = hdr(c.ReconLPresentDesc, c.RECONL_STRUCT_PRESENT_DESC),
            .out_pixels = @ptrCast(pixels.ptr),
            .out_pixels_size = pixels.len,
            .out_row_pitch = self.width * 4,
            .out_format = @intCast(c.RECONL_FORMAT_R8G8B8A8_UNORM),
            .flip = 0,
        };
        r = c.reconlPresent(self.device, self.swapchain, &pd);
        if (r != 0) {
            self.record("reconlPresent", r);
            return error.ReconLError;
        }
        self.frame_index +%= 1;
    }

    // ---- internals ------------------------------------------------------

    fn cmdPush(self: *Renderer, slot: u32, data: anytype) bool {
        const r = c.reconlCmdPushConstants(self.commands, slot, data, 64);
        if (r != 0) {
            self.abortFrame("reconlCmdPushConstants", r);
            return false;
        }
        return true;
    }

    /// A failure between BeginFrame and Submit must leave the device in the
    /// state the state machine accepts next: close the pass, drop the frame
    /// with a best-effort submit, and never present half of it.
    fn abortFrame(self: *Renderer, comptime what: []const u8, r: c_int) void {
        _ = c.reconlCmdEndRenderPass(self.commands);
        self.record(what, r);
        _ = c.reconlSubmit(self.device, self.commands, null);
    }

    fn ensureBuffers(self: *Renderer, mesh: ui_mod.Mesh) Error!void {
        const vneed: u64 = @as(u64, @intCast(mesh.vertices.len)) * @sizeOf(tess.Vertex);
        const ineed: u64 = @as(u64, @intCast(mesh.indices.len)) * @sizeOf(u32);
        if (self.vb == null or self.vb_bytes < vneed) {
            const cap = grow(vneed);
            if (self.vb) |b| _ = c.reconlRelease(b);
            self.vb = null;
            self.vb_bytes = 0;
            var buf: ?*c.ReconLBuffer = null;
            const bd = bufferDesc(cap, c.RECONL_BUFFER_VERTEX);
            const r = c.reconlCreateBuffer(self.device, &bd, &buf);
            if (r != 0) {
                self.record("reconlCreateBuffer (vertices)", r);
                return error.ReconLError;
            }
            self.vb = buf;
            self.vb_bytes = cap;
        }
        if (self.ib == null or self.ib_bytes < ineed) {
            const cap = grow(ineed);
            if (self.ib) |b| _ = c.reconlRelease(b);
            self.ib = null;
            self.ib_bytes = 0;
            var buf: ?*c.ReconLBuffer = null;
            const bd = bufferDesc(cap, c.RECONL_BUFFER_INDEX);
            const r = c.reconlCreateBuffer(self.device, &bd, &buf);
            if (r != 0) {
                self.record("reconlCreateBuffer (indices)", r);
                return error.ReconLError;
            }
            self.ib = buf;
            self.ib_bytes = cap;
        }
    }

    fn fillClear(self: *Renderer, pixels: []u8) void {
        const r = channel(self.clear[0]);
        const g = channel(self.clear[1]);
        const b = channel(self.clear[2]);
        const a = channel(self.clear[3]);
        var i: usize = 0;
        while (i + 4 <= pixels.len) : (i += 4) {
            pixels[i] = r;
            pixels[i + 1] = g;
            pixels[i + 2] = b;
            pixels[i + 3] = a;
        }
    }

    fn record(self: *Renderer, comptime what: []const u8, r: c_int) void {
        const name: []const u8 = if (r == 0)
            "unknown error"
        else
            std.mem.span(c.reconlResultName(r));
        const n = std.fmt.bufPrint(&self.err_buf, "{s} failed: {s} ({d})", .{ what, name, r }) catch return;
        self.err_len = n.len;
    }

    fn swapDesc(w: u32, h: u32) c.ReconLSwapchainDesc {
        return .{
            .base = hdr(c.ReconLSwapchainDesc, c.RECONL_STRUCT_SWAPCHAIN_DESC),
            .width = w,
            .height = h,
            .format = @intCast(c.RECONL_FORMAT_R8G8B8A8_UNORM),
            .image_count = 2,
            .present_to_memory = 1, // fully headless: pixels come back in Present
            .depth_format = 1, // as tools/host/src/frame.rs creates it
            .flags = 0,
            .reserved = 0,
        };
    }

    /// The UI pipeline: vertex colours only, alpha blended, no culling (the
    /// tessellator emits either winding), reversed-Z GREATER with **depth
    /// writes off** - all UI sits at a constant z = 0.5, so a depth write from
    /// the first triangle would reject every triangle after it.
    fn pipelineDesc() c.ReconLPipelineDesc {
        return .{
            .base = hdr(c.ReconLPipelineDesc, c.RECONL_STRUCT_PIPELINE_DESC),
            .shading = @intCast(c.RECONL_SHADING_UNLIT),
            .blend = @intCast(c.RECONL_BLEND_ALPHA),
            .cull = @intCast(c.RECONL_CULL_NONE),
            .depth_compare = @intCast(c.RECONL_COMPARE_GREATER),
            .depth_write = 0,
            .texture_slots = 0,
            .texture_formats = @splat(0),
            .receives_shadow = 0,
            .casts_shadow = 0,
            .flags = 0,
            .reserved = 0,
            .debug_name = null,
        };
    }

    fn cmdListDesc() c.ReconLCommandListDesc {
        return .{
            .base = hdr(c.ReconLCommandListDesc, c.RECONL_STRUCT_COMMAND_LIST_DESC),
            // begin + 2 pushes + pipeline + vb + ib + draw + end = 8, doubled
            // for headroom; the list never grows inside a frame.
            .capacity_bytes = 32 * c.RECONL_COMMAND_BYTES,
            .reserved = 0,
            .debug_name = null,
        };
    }

    fn bufferDesc(cap: u64, usage: c_int) c.ReconLBufferDesc {
        return .{
            .base = hdr(c.ReconLBufferDesc, c.RECONL_STRUCT_BUFFER_DESC),
            .size_bytes = cap,
            .usage = @as(c_uint, @intCast(usage)) | @as(c_uint, @intCast(c.RECONL_BUFFER_DYNAMIC)),
            .reserved = 0,
            .data = null,
            .data_size = 0,
            .debug_name = null,
        };
    }

    fn renderPassDesc(self: *const Renderer) c.ReconLRenderPassDesc {
        const empty_att = c.ReconLColorAttachment{
            .texture = null,
            .resolve = null,
            .mip = 0,
            .layer = 0,
        };
        return .{
            .base = hdr(c.ReconLRenderPassDesc, c.RECONL_STRUCT_RENDER_PASS_DESC),
            .color_count = 0, // the swapchain's colour, as frame.rs does it
            .reserved = 0,
            .color = .{ empty_att, empty_att, empty_att, empty_att },
            .depth = null,
            .viewport_width = self.width,
            .viewport_height = self.height,
            .load_color = 1,
            .load_depth = 1,
            .clear_color = self.clear,
            .clear_depth = 0.0, // reversed-Z far
            .stencil_clear = 0,
            .reserved2 = 0,
        };
    }

    fn channel(x: f32) u8 {
        return @intFromFloat(std.math.clamp(x, 0.0, 1.0) * 255.0 + 0.5);
    }
};

/// Geometric growth with a floor: a 1-widget frame doesn't recreate buffers
/// every time a label wraps one line wider.
fn grow(need: u64) u64 {
    var cap: u64 = 64 * 1024;
    while (cap < need) cap *|= 2;
    return cap;
}
