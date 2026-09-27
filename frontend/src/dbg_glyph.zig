//! Temporary diagnostic: dumps contour/triangulation numbers for glyphs the
//! font tests found suspicious. Not part of the build; run manually with
//! `zig run src/dbg_glyph.zig -I ../include -lc`.
const std = @import("std");
const font_mod = @import("font.zig");
const polygon = @import("polygon.zig");

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    const gpa = gpa_state.allocator();

    var f = try font_mod.loadBundled(gpa, "Inter-Variable.ttf");
    defer f.deinit();

    // Plain 'e' vs the composite: does the composite keep both children's
    // contours?
    {
        const ge = try f.glyph(f.glyphIndex('e'));
        std.debug.print("plain 'e': gid={d} contours={d} points={d}\n", .{ f.glyphIndex('e'), ge.contours.len, ge.points.len });
        {
            const ed = try f.glyphData(f.glyphIndex('e'));
            const enc = try font_mod.ri16(ed, 0);
            std.debug.print("  'e' header: nc={d} bbox=({d},{d})-({d},{d})", .{
                enc,
                try font_mod.ri16(ed, 2),
                try font_mod.ri16(ed, 4),
                try font_mod.ri16(ed, 6),
                try font_mod.ri16(ed, 8),
            });
            if (enc >= 1) {
                var ci: i32 = 0;
                while (ci < enc) : (ci += 1) {
                    std.debug.print(" endPt[{d}]={d}", .{ ci, try font_mod.ru16(ed, 10 + @as(usize, @intCast(ci)) * 2) });
                }
            }
            std.debug.print("\n", .{});
        }
        const data = try f.glyphData(f.glyphIndex(0x00E9));
        var off: usize = 10;
        var more = true;
        while (more) {
            const flags = try font_mod.ru16(data, off);
            off += 2;
            const cgid = try font_mod.ru16(data, off);
            off += 2;
            if (flags & 0x0001 != 0) off += 4 else off += 2;
            if (flags & 0x0008 != 0) off += 2 else if (flags & 0x0040 != 0) off += 4 else if (flags & 0x0080 != 0) off += 8;
            more = flags & 0x0020 != 0;
            std.debug.print("  component gid={d} flags=0x{X}\n", .{ cgid, flags });
            const cg = try f.glyph(cgid);
            std.debug.print("    child contours={d} points={d}\n", .{ cg.contours.len, cg.points.len });
        }
    }
    const cases = [_]u32{ f.glyphIndex('o'), f.glyphIndex('I'), f.glyphIndex(0x00E9) };
    // Raw bytes of the 'I' glyph for hand-decoding.
    {
        const data = try f.glyphData(f.glyphIndex('I'));
        std.debug.print("'I' glyph data ({d} bytes):\n  ", .{data.len});
        for (data[0..@min(data.len, 96)], 0..) |b, bi| {
            std.debug.print("{X:0>2}{s}", .{ b, if (bi % 16 == 15) "\n  " else " " });
        }
        std.debug.print("\n", .{});
    }
    for (cases, 0..) |gid, i| {
        const g = try f.glyph(gid);
        std.debug.print("case {d}: gid={d} points={d} contours={d} tris={d}\n", .{ i, gid, g.points.len, g.contours.len, g.tris.len / 3 });
        var max_a: f32 = 0;
        var rest: f32 = 0;
        for (g.contours, 0..) |ct, ci| {
            const a = polygon.contourArea(g.points, ct);
            std.debug.print("  contour {d}: start={d} len={d} area={d:.1}\n", .{ ci, ct.start, ct.len, a });
            if (ct.len <= 8) {
                var k: u32 = 0;
                while (k < ct.len) : (k += 1) {
                    const p = g.points[ct.start + k];
                    std.debug.print("    [{d}] = ({d:.1}, {d:.1})\n", .{ k, p.x, p.y });
                }
            }
            if (a > max_a) {
                rest += max_a;
                max_a = a;
            } else rest += a;
        }
        const filled = @abs(polygon.signedArea(g.points, g.tris));
        std.debug.print("  max={d:.1} rest={d:.1} expect={d:.1} filled={d:.1} ratio={d:.3}\n", .{ max_a, rest, max_a - rest, filled, filled / @max(max_a - rest, 1) });

        // Points extent vs the glyph's declared glyf bbox.
        const data = try f.glyphData(gid);
        const nc = try font_mod.ri16(data, 0);
        const x0: f32 = @floatFromInt(try font_mod.ri16(data, 2));
        const y0: f32 = @floatFromInt(try font_mod.ri16(data, 4));
        const x1: f32 = @floatFromInt(try font_mod.ri16(data, 6));
        const y1: f32 = @floatFromInt(try font_mod.ri16(data, 8));
        var minx: f32 = 1e9;
        var miny: f32 = 1e9;
        var maxx: f32 = -1e9;
        var maxy: f32 = -1e9;
        for (g.points) |p| {
            minx = @min(minx, p.x);
            miny = @min(miny, p.y);
            maxx = @max(maxx, p.x);
            maxy = @max(maxy, p.y);
        }
        std.debug.print("  nc={d} declared=({d:.0},{d:.0})-({d:.0},{d:.0}) actual=({d:.1},{d:.1})-({d:.1},{d:.1})\n", .{ nc, x0, y0, x1, y1, minx, miny, maxx, maxy });
    }
}
