//! PNG output: RGBA8 in, a standard PNG on disk out, compressed by the
//! standard library's deflate - no image dependency, and files that stay in
//! the low hundreds of kilobytes because UI frames are flat colour runs.
//!
//! Filter choice is `None` per row: deflate already eats the repeats, and a
//! filter that predicts would cost encode time for little gain on this
//! specific workload (large runs of identical RGBA).

const std = @import("std");
const flate = std.compress.flate;

pub const signature = [_]u8{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };

/// Encodes `rgba` (width*height*4, straight alpha, top-down) as a PNG into
/// `out`.
pub fn encode(
    gpa: std.mem.Allocator,
    width: u32,
    height: u32,
    rgba: []const u8,
    out: *std.Io.Writer,
) !void {
    const px = @as(usize, width) * @as(usize, height);
    if (rgba.len != px * 4) return error.InvalidImageSize;
    if (width == 0 or height == 0) return error.InvalidImageSize;

    try out.writeAll(&signature);

    var ihdr: [13]u8 = undefined;
    writeBe32(ihdr[0..4], width);
    writeBe32(ihdr[4..8], height);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 6; // colour type: truecolour with alpha
    ihdr[10] = 0; // deflate
    ihdr[11] = 0; // adaptive filtering
    ihdr[12] = 0; // no interlace
    try chunk(out, "IHDR", &ihdr);

    // ---- IDAT: zlib of (filter byte + row) * height ---------------------
    const stride = @as(usize, width) * 4;
    const raw_len = (stride + 1) * @as(usize, height);
    // Worst case deflate output is raw + stored-block overhead; UI frames
    // compress far below raw, but the bound has to be a bound.
    const comp_buf = try gpa.alloc(u8, raw_len + raw_len / 8 + 65536);
    defer gpa.free(comp_buf);
    const window = try gpa.alloc(u8, flate.max_window_len);
    defer gpa.free(window);

    var comp_w: std.Io.Writer = .fixed(comp_buf);
    var comp = try flate.Compress.init(&comp_w, window, .zlib, .default);
    var row: u32 = 0;
    while (row < height) : (row += 1) {
        try comp.writer.writeAll(&.{0}); // filter: None
        const start = @as(usize, row) * stride;
        try comp.writer.writeAll(rgba[start .. start + stride]);
    }
    try comp.finish();
    try chunk(out, "IDAT", comp_w.buffered());

    try chunk(out, "IEND", &.{});
}

/// Encodes and writes `path` (relative to cwd) in one step.
pub fn writeRgba(
    gpa: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    width: u32,
    height: u32,
    rgba: []const u8,
) !void {
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try encode(gpa, width, height, rgba, &aw.writer);
    try std.Io.Dir.cwd().writeFile(io, path, .{ .data = aw.writer.buffered() });
}

fn chunk(out: *std.Io.Writer, comptime typ: []const u8, data: []const u8) !void {
    var len: [4]u8 = undefined;
    writeBe32(&len, @intCast(data.len));
    try out.writeAll(&len);
    try out.writeAll(typ);
    try out.writeAll(data);
    var crc = std.hash.Crc32.init();
    crc.update(typ);
    crc.update(data);
    var c: [4]u8 = undefined;
    writeBe32(&c, crc.final());
    try out.writeAll(&c);
}

fn writeBe32(dst: []u8, v: u32) void {
    dst[0] = @truncate(v >> 24);
    dst[1] = @truncate(v >> 16);
    dst[2] = @truncate(v >> 8);
    dst[3] = @truncate(v);
}

fn readBe32(b: []const u8) u32 {
    return (@as(u32, b[0]) << 24) | (@as(u32, b[1]) << 16) | (@as(u32, b[2]) << 8) | b[3];
}

// ---------------------------------------------------------------------------
// tests

test "an encoded PNG has valid structure, CRCs, and decompresses to the input" {
    const gpa = std.testing.allocator;
    // 4x2 test pattern.
    const w: u32 = 4;
    const h: u32 = 2;
    const rgba = [_]u8{
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    };

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try encode(gpa, w, h, &rgba, &aw.writer);
    const bytes = aw.writer.buffered();

    try std.testing.expectEqualSlices(u8, &signature, bytes[0..8]);

    // Walk chunks; each must carry a valid CRC.
    var off: usize = 8;
    var idat: ?[]const u8 = null;
    var saw_iend = false;
    while (off + 8 <= bytes.len) {
        const len = readBe32(bytes[off .. off + 4]);
        const typ = bytes[off + 4 .. off + 8];
        const data = bytes[off + 8 .. off + 8 + len];
        const stored_crc = readBe32(bytes[off + 8 + len .. off + 12 + len]);
        var crc = std.hash.Crc32.init();
        crc.update(typ);
        crc.update(data);
        try std.testing.expectEqual(stored_crc, crc.final());
        if (std.mem.eql(u8, typ, "IHDR")) {
            try std.testing.expectEqual(w, readBe32(data[0..4]));
            try std.testing.expectEqual(h, readBe32(data[4..8]));
            try std.testing.expectEqual(@as(u8, 8), data[8]);
            try std.testing.expectEqual(@as(u8, 6), data[9]);
        } else if (std.mem.eql(u8, typ, "IDAT")) {
            idat = data;
        } else if (std.mem.eql(u8, typ, "IEND")) {
            saw_iend = true;
        }
        off += 12 + len;
    }
    try std.testing.expect(saw_iend);
    const idat_bytes = idat.?;

    // Decompress the zlib stream and expect exactly the filtered rows back.
    var src: std.Io.Reader = .fixed(idat_bytes);
    var window: [flate.max_window_len]u8 = undefined;
    var d: flate.Decompress = .init(&src, .zlib, &window);
    var out_buf: [64]u8 = undefined;
    var got: usize = 0;
    while (got < out_buf.len) {
        const n = try d.reader.readSliceShort(out_buf[got..]);
        if (n == 0) break;
        got += n;
    }
    const stride = @as(usize, w) * 4;
    try std.testing.expectEqual((stride + 1) * @as(usize, h), got);
    var row: usize = 0;
    while (row < h) : (row += 1) {
        const base = row * (stride + 1);
        try std.testing.expectEqual(@as(u8, 0), out_buf[base]); // filter None
        try std.testing.expectEqualSlices(
            u8,
            rgba[row * stride .. (row + 1) * stride],
            out_buf[base + 1 .. base + 1 + stride],
        );
    }
}

test "encode rejects mismatched sizes" {
    const gpa = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try std.testing.expectError(error.InvalidImageSize, encode(gpa, 4, 2, &[_]u8{1, 2, 3}, &aw.writer));
}
