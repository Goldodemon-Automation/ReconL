"""Behavioral verification of the showcase PNG sequence.

Decodes every frame (no PIL: raw zlib + PNG unfiltering) and asserts:
  1. 60 valid 960x540 RGBA8 frames exist;
  2. content is actually drawn (widgets cover a large share of the frame);
  3. the sequence animates (frame-to-frame pixel motion nearly everywhere,
     which is the hover ramps, progress fill, sparkline shift and springs);
  4. the scripted interactions land: the Deploy press (f10) and release
     (f11) change pixels, the toggle click (f19) keeps the knob springing
     for several frames, and frame 59 differs from frame 0 (end state).
"""
import struct, sys, zlib, os

# zig-out/showcase relative to this script (frontend/spike/..) or to cwd.
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zig-out", "showcase")
if not os.path.isdir(DIR):
    DIR = os.path.join(os.getcwd(), "zig-out", "showcase")

W, H, N = 960, 540, 60
BG = (11, 13, 12)

def read_png(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: not a PNG"
    pos, idat, ihdr = 8, b"", None
    while pos < len(data):
        (ln,) = struct.unpack(">I", data[pos:pos + 4])
        typ = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", chunk)
        elif typ == b"IDAT":
            idat += chunk
        elif typ == b"IEND":
            break
        pos += 12 + ln
    w, h, depth, ctype, comp, filt, inter = ihdr
    assert (w, h, depth, ctype, inter) == (W, H, 8, 6, 0), f"{path}: IHDR {ihdr}"
    raw = zlib.decompress(idat)
    stride = W * 4
    out = bytearray(h * stride)
    prev = bytearray(stride)
    p = 0
    for y in range(h):
        ft = raw[p]; p += 1
        line = bytearray(raw[p:p + stride]); p += stride
        if ft == 1:
            for i in range(4, stride):
                line[i] = (line[i] + line[i - 4]) & 0xFF
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ft == 3:
            for i in range(stride):
                a = line[i - 4] if i >= 4 else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = line[i - 4] if i >= 4 else 0
                b = prev[i]
                cc = prev[i - 4] if i >= 4 else 0
                pp = a + b - cc
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - cc)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else cc)
                line[i] = (line[i] + pr) & 0xFF
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return out

def diff_px(a, b):
    if a == b:
        return 0
    # Channel slices make the zip a C-speed extract; count differing pixels.
    ra, rb = a[0::4], b[0::4]
    ga, gb = a[1::4], b[1::4]
    ba, bb = a[2::4], b[2::4]
    return sum(1 for x, y, u, v, w, z in zip(ra, rb, ga, gb, ba, bb)
               if x != y or u != v or w != z)

def nonbg(px):
    r, g, b = px[0::4], px[1::4], px[2::4]
    br, bg_, bb = bytes([BG[0]]) * (W * H), bytes([BG[1]]) * (W * H), bytes([BG[2]]) * (W * H)
    return sum(1 for x, y, u, v, w, z in zip(r, br, g, bg_, b, bb)
               if x != y or u != v or w != z)

def main():
    frames = []
    for i in range(N):
        p = os.path.join(DIR, f"frame_{i:03}.png")
        assert os.path.isfile(p), f"missing {p}"
        frames.append(read_png(p))
    print(f"decoded {len(frames)} frames of {W}x{H} RGBA8")

    # 2. content drawn: widgets cover most of the surface
    cov = nonbg(frames[0]) / (W * H)
    assert cov > 0.5, f"frame 0 covers only {cov:.1%} of the surface"
    cov59 = nonbg(frames[59]) / (W * H)
    print(f"coverage: frame0={cov:.1%} frame59={cov59:.1%}")

    # 3. continuous animation: compute every consecutive-pair diff once.
    diffs = [None] + [diff_px(frames[i], frames[i - 1]) for i in range(1, N)]
    moving = sum(1 for d in diffs[1:] if d > 200)
    assert moving >= 45, f"only {moving}/{N-1} consecutive frame pairs animate"
    print(f"consecutive pairs with >200 changed px: {moving}/{N-1}")

    # 4a. Deploy press at f10, release at f11: both visibly change pixels
    d10, d11 = diffs[10], diffs[11]
    assert d10 > 100, f"press frame f10 changed only {d10} px"
    assert d11 > 100, f"release frame f11 changed only {d11} px"
    # press+release must be localized (not a whole-screen redraw)
    assert d10 < W * H * 0.15, f"press diff too global: {d10} px"

    # 4b. toggle click at f19: the knob spring keeps moving for several frames
    spring = diffs[20:26]
    assert all(s > 30 for s in spring), f"toggle spring stalled: {spring}"
    print(f"toggle spring frames f20..f25 diffs: {spring}")

    # 4c. slider drag f26..f40: motion in every drag frame
    drag = diffs[27:41]
    assert all(s > 30 for s in drag), f"slider drag stalled: {drag}"

    # 4d. wheel scroll f44..f50: log content moves
    scroll = diffs[45:51]
    assert all(s > 30 for s in scroll), f"scroll spring stalled: {scroll}"

    # 4e. end state differs from start state (toggle on, quality full, etc.)
    end = diff_px(frames[59], frames[0])
    assert end > W * H * 0.01, f"end state equals start ({end} px)"
    print(f"frame59 vs frame0: {end} px differ")

    print("showcase behavior: ALL ASSERTIONS PASSED")

if __name__ == "__main__":
    main()
