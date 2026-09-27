"""Lifecycle-edge verification of reconl_ui.dll - the paths that are easy
to get wrong, each traced end to end from a foreign process:

  a. render BEFORE any endFrame        -> 0, every pixel is the clear colour
     (the empty-mesh guard: the ABI's empty-frame policy is never fed an
     empty frame, pixels come from the host-side clear fill);
  b. begin/endFrame with zero widgets  -> endFrame says 0, render says 0,
     pixels still exactly the clear colour;
  c. wheel-scroll through the DLL      -> the content vertices move while the
     viewport rect stays put (spring + clamp, asserted on mesh bytes);
  d. Create after changing to an unrelated cwd -> succeeds because bundled
     fonts are embedded rather than found via process-relative paths;
  e. create -> render -> destroy -> create again -> render: two independent
     lifecycles, no leaked device state between them.
"""
import ctypes as ct
import os
import shutil
import tempfile

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "zig-out", "bin"))
os.environ["PATH"] = BASE + os.pathsep + os.environ.get("PATH", "")
try:
    os.add_dll_directory(BASE)
except (AttributeError, OSError):
    pass

W, H = 160, 120
BG = (0x0B, 0x0D, 0x0C, 0xFF)


class Input(ct.Structure):
    _fields_ = [("px", ct.c_float), ("py", ct.c_float),
                ("down", ct.c_uint32), ("pressed", ct.c_uint32),
                ("released", ct.c_uint32),
                ("wheel", ct.c_float), ("dt_ms", ct.c_float)]


class Rect(ct.Structure):
    _fields_ = [("x", ct.c_float), ("y", ct.c_float),
                ("w", ct.c_float), ("h", ct.c_float)]


def load():
    return ct.CDLL(os.path.join(BASE, "reconl_ui.dll"))


def bind(ui):
    ui.reconlUiCreate.argtypes = [ct.c_uint32, ct.c_uint32]
    ui.reconlUiCreate.restype = ct.c_void_p
    ui.reconlUiDestroy.argtypes = [ct.c_void_p]
    ui.reconlUiLastError.argtypes = [ct.c_void_p]
    ui.reconlUiLastError.restype = ct.c_char_p
    ui.reconlUiBegin.argtypes = [ct.c_void_p, ct.POINTER(Input)]
    ui.reconlUiEnd.argtypes = [ct.c_void_p]
    ui.reconlUiEndFrame.argtypes = [ct.c_void_p]
    ui.reconlUiEndFrame.restype = ct.c_uint32
    ui.reconlUiRender.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint64]
    ui.reconlUiRender.restype = ct.c_int32
    ui.reconlUiLabel.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int32, ct.c_void_p]
    ui.reconlUiLabel.restype = ct.c_float
    ui.reconlUiBeginScroll.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_float, ct.c_float]
    ui.reconlUiBeginScroll.restype = Rect
    ui.reconlUiEndFrame.argtypes = [ct.c_void_p]
    ui.reconlUiVertexCount.argtypes = [ct.c_void_p]
    ui.reconlUiVertexCount.restype = ct.c_uint32
    ui.reconlUiVertices.argtypes = [ct.c_void_p]
    ui.reconlUiVertices.restype = ct.c_void_p
    ui.reconlUiIndexCount.argtypes = [ct.c_void_p]
    ui.reconlUiIndexCount.restype = ct.c_uint32
    ui.reconlUiLastRect.argtypes = [ct.c_void_p]
    ui.reconlUiLastRect.restype = Rect
    ui.reconlUiFrameIndex.argtypes = [ct.c_void_p]
    ui.reconlUiFrameIndex.restype = ct.c_uint32


def begin(ui, h, px=-1e6, py=-1e6, wheel=0.0):
    inp = Input(px, py, 0, 0, 0, wheel, 1000.0 / 60.0)
    ui.reconlUiBegin(h, ct.byref(inp))


def main():
    ui = load()
    bind(ui)

    # ---- (a) render before any endFrame ---------------------------------
    h = ui.reconlUiCreate(W, H)
    assert h, f"create failed: {ui.reconlUiLastError(None)!r}"
    buf = ct.create_string_buffer(W * H * 4)
    rc = ui.reconlUiRender(h, buf, W * H * 4)
    assert rc == 0, f"render-before-endFrame refused: {ui.reconlUiLastError(h)!r}"
    assert buf.raw == bytes(BG) * (W * H), "empty mesh did not fill the clear colour"
    print("(a) render-before-endFrame: 0, pixels exactly clear colour")

    # ---- (b) a frame with no widgets ------------------------------------
    begin(ui, h)
    assert ui.reconlUiEndFrame(h) == 0, "empty frame claimed a mesh"
    rc = ui.reconlUiRender(h, buf, W * H * 4)
    assert rc == 0
    assert buf.raw == bytes(BG) * (W * H), "empty frame did not fill clear colour"
    assert ui.reconlUiVertexCount(h) == 0
    print("(b) empty begin/endFrame: endFrame=0, render=0, clear pixels")

    # ---- (c) wheel-scroll through the DLL -------------------------------
    def scroll_frame(wheel, px, py):
        begin(ui, h, px=px, py=py, wheel=wheel)
        content = ui.reconlUiBeginScroll(h, b"log", 60, 300)
        # last_rect right after beginScroll is the viewport (labels below
        # overwrite it with their own slots).
        viewport = ui.reconlUiLastRect(h)
        # 16 lines ~490px of content: the visible slice must contain text at
        # EVERY scroll position, including the settled clamp at -240.
        for i in range(16):
            ui.reconlUiLabel(h, f"line {i}".encode(), 3, None)
        ui.reconlUiEnd(h)
        assert ui.reconlUiEndFrame(h) == 1
        n = ui.reconlUiVertexCount(h) * 48
        ptr = ui.reconlUiVertices(h)
        assert ptr, "no vertices"
        return content, viewport, ct.string_at(ptr, n)

    rest_c, rest_v, mesh0 = scroll_frame(0.0, -1e6, -1e6)      # rest
    away_c, away_v, mesh1 = scroll_frame(1.0, 500, 500)        # wheel, pointer AWAY
    assert mesh1 == mesh0, "wheel scrolled with the pointer outside the viewport"
    assert away_v.y == rest_v.y and away_c.y == rest_c.y
    # beginScroll returns the fixed viewport; its height is not content height.
    assert rest_c.y == rest_v.y and rest_c.h == 60, \
        "beginScroll did not return the viewport rect"

    over_c, over_v, mesh2 = scroll_frame(1.0, rest_v.x + 10, rest_v.y + 10)
    assert over_v.y == rest_v.y, "the viewport rect itself moved"
    assert mesh2 != mesh0, "wheel over the viewport did not move the content"
    assert over_c.y == rest_c.y, "beginScroll viewport moved with its content"

    # Settle at the clamp: travel is content_h - viewport_h = 300 - 60 = 240.
    # Wheel six times (6 x 48 = 288 > 240, so the target itself clamps),
    # then let the spring settle. The fixed viewport stays unchanged; clipping
    # changes which label vertices survive as the content moves.
    for i in range(6):
        scroll_frame(1.0, rest_v.x + 10, rest_v.y + 10)
    for i in range(120):
        scroll_frame(0.0, rest_v.x + 10, rest_v.y + 10)
    fin_c, fin_v, a = scroll_frame(0.0, rest_v.x + 10, rest_v.y + 10)
    b = scroll_frame(0.0, rest_v.x + 10, rest_v.y + 10)[2]
    assert a == b, "settled scroll is still moving"
    assert fin_v.y == rest_v.y, "viewport rect drifted during the scroll"
    assert fin_c.y == rest_c.y and fin_c.h == rest_c.h, \
        "beginScroll viewport changed after settling scroll"
    print("(c) wheel gated by hover; viewport stable; spring settled at clamp and mesh is byte-stable")

    ui.reconlUiDestroy(h)

    # ---- (d) creation is independent of the process cwd ----------------
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="reconlui-unrelated-cwd-")
    try:
        os.chdir(tmp)
        independent = ui.reconlUiCreate(W, H)
        assert independent, f"create from unrelated cwd failed: {ui.reconlUiLastError(None)!r}"
        ui.reconlUiDestroy(independent)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    print("(d) create -> destroy from unrelated cwd: succeeded")

    # ---- (e) two full lifecycles in one process -------------------------
    for cycle in range(2):
        h = ui.reconlUiCreate(W, H)
        assert h, f"cycle {cycle}: create failed"
        begin(ui, h)
        ui.reconlUiLabel(h, f"cycle {cycle}".encode(), 0, None)
        assert ui.reconlUiEndFrame(h) == 1
        out = ct.create_string_buffer(W * H * 4)
        assert ui.reconlUiRender(h, out, W * H * 4) == 0
        assert out.raw != bytes(BG) * (W * H), f"cycle {cycle}: nothing drawn"
        assert ui.reconlUiFrameIndex(h) == 1
        ui.reconlUiDestroy(h)
    print("(e) create -> render -> destroy, twice: independent and clean")

    print("reconl_ui.dll lifecycle: ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
