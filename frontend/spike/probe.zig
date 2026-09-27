//! Type-shape probe for reconl.h under Zig 0.16 translate-c.
const std = @import("std");
const c = @cImport({ @cInclude("reconl/reconl.h"); });


pub fn main() void {
    var major: u32 = 0;
    var minor: u32 = 0;
    var patch: u32 = 0;
    c.reconlVersion(&major, &minor, &patch);
    std.debug.print("reconl {d}.{d}.{d}\n", .{ major, minor, patch });
    // Allocator callback types as translate-c sees them:
    inline for (.{ "reconlCreateDevice", "reconlCreateSwapchain", "reconlCreatePipeline", "reconlCreateCommandList", "reconlCreateBuffer", "reconlWriteBuffer", "reconlBeginFrame", "reconlCmdReset", "reconlCmdBeginRenderPass", "reconlCmdPushConstants", "reconlCmdDrawIndexed", "reconlSubmit", "reconlPresent", "reconlRelease" }) |name| {
        const f = @field(c, name);
        std.debug.print("{s}: {s}\n", .{ name, @typeName(@TypeOf(f)) });
    }
    const A = @TypeOf(@as(c.ReconLDeviceDesc, undefined).allocator);
    std.debug.print("allocator field: {s}\n", .{@typeName(A)});
    std.debug.print("alloc member: {s}\n", .{@typeName(@TypeOf(@as(A, undefined).alloc))});
    std.debug.print("free member: {s}\n", .{@typeName(@TypeOf(@as(A, undefined).free))});
    std.debug.print("realloc member: {s}\n", .{@typeName(@TypeOf(@as(A, undefined).realloc))});
}
