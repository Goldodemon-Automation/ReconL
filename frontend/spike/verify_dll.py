"""Foreign-process verification of zig-out/bin/reconl_ui.dll.

Loads the shared library exactly the way a Panama FFM or Kotlin consumer
will: resolve symbols by name, pass plain C structs, render through the real
reconl.dll behind it. Asserts the observable behavior, not just linkage:
  1. exports exist and report ABI version 1;
  2. struct layouts are what the header promises (28-byte input, 16-byte rect);
  3. a built frame presents pixels that differ from the clear colour;
  4. press-then-release flips the button and the toggle (state transitions
     through the DLL, not in-process);
  5. undersized buffers are refused with a message; destroy is clean.
"""
import ctypes as ct
import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "zig-out", "bin"))
os.environ["PATH"] = BASE + os.pathsep + os.environ.get("PATH", "")
try:
    os.add_dll_directory(BASE)
except (AttributeError, OSError):
    pass

DLL = os.path.join(BASE, "reconl_ui.dll")
assert os.path.isfile(DLL), f"missing {DLL} - run `zig build shared`"

W, H = 320, 180
BG = (0x0B, 0x0D, 0x0C)


class Input(ct.Structure):
    _fields_ = [("px", ct.c_float), ("py", ct.c_float),
                ("down", ct.c_uint32), ("pressed", ct.c_uint32),
                ("released", ct.c_uint32),
                ("wheel", ct.c_float), ("dt_ms", ct.c_float)]


class Rect(ct.Structure):
    _fields_ = [("x", ct.c_float), ("y", ct.c_float),
                ("w", ct.c_float), ("h", ct.c_float)]


def main():
    ui = ct.CDLL(DLL)

    # 1. exports + version
    ui.reconlUiAbiVersion.restype = ct.c_uint32
    assert ui.reconlUiAbiVersion() == 1, "ABI version is not 1"

    # 2. layouts
    assert ct.sizeof(Input) == 28, f"ReconLUiInput is {ct.sizeof(Input)} bytes"
    assert ct.sizeof(Rect) == 16, f"ReconLRect is {ct.sizeof(Rect)} bytes"

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
    ui.reconlUiButton.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.c_uint32]
    ui.reconlUiButton.restype = ct.c_uint32
    ui.reconlUiToggle.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.POINTER(ct.c_uint32)]
    ui.reconlUiToggle.restype = ct.c_uint32
    ui.reconlUiBeginPanel.argtypes = [ct.c_void_p, ct.POINTER(Rect),
                                      ct.c_float, ct.c_float, ct.c_uint32]
    ui.reconlUiEndFrame.restype = ct.c_uint32
    ui.reconlUiVertexCount.argtypes = [ct.c_void_p]
    ui.reconlUiVertexCount.restype = ct.c_uint32
    ui.reconlUiIndexCount.argtypes = [ct.c_void_p]
    ui.reconlUiIndexCount.restype = ct.c_uint32
    ui.reconlUiLastRect.argtypes = [ct.c_void_p]
    ui.reconlUiLastRect.restype = Rect
    ui.reconlUiFrameIndex.argtypes = [ct.c_void_p]
    ui.reconlUiFrameIndex.restype = ct.c_uint32
    ui.reconlUiResize.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint32]
    ui.reconlUiResize.restype = ct.c_int32

    handle = ui.reconlUiCreate(W, H)
    assert handle, f"reconlUiCreate failed: {ui.reconlUiLastError(None)!r}"

    def begin(px=-1e6, py=-1e6, down=False, pressed=False, released=False):
        inp = Input(px, py, int(down), int(pressed), int(released), 0.0, 1000.0 / 60.0)
        ui.reconlUiBegin(handle, ct.byref(inp))

    # 3. build a frame and present it
    begin()
    panel = Rect(8, 8, W - 16, 90)
    ui.reconlUiBeginPanel(handle, ct.byref(panel), 12.0, 8.0, 1)
    assert ui.reconlUiLabel(handle, b"Hello ctypes", 0, None) > 0
    assert ui.reconlUiButton(handle, b"go", b"Go", 1) == 0
    btn = ui.reconlUiLastRect(handle)
    assert btn.w > 0 and btn.h > 0, "button rect is empty"
    ui.reconlUiEnd(handle)
    assert ui.reconlUiEndFrame(handle) == 1, "no mesh produced"
    vcount, icount = ui.reconlUiVertexCount(handle), ui.reconlUiIndexCount(handle)
    assert vcount > 0 and icount > 0 and icount % 3 == 0, (vcount, icount)

    pixels = ct.create_string_buffer(W * H * 4)
    rc = ui.reconlUiRender(handle, pixels, W * H * 4)
    assert rc == 0, f"render failed: {ui.reconlUiLastError(handle)!r}"
    raw = pixels.raw
    drawn = any(raw[i] != BG[0] or raw[i + 1] != BG[1] or raw[i + 2] != BG[2]
                for i in range(0, len(raw), 4))
    assert drawn, "rendered frame equals the clear colour everywhere"
    assert ui.reconlUiFrameIndex(handle) == 1

    # 4. click the button through the DLL - same widget tree every frame,
    #    so the rect the capture frame recorded is where the button lives.
    def draw_button_tree():
        panel = Rect(8, 8, W - 16, 90)
        ui.reconlUiBeginPanel(handle, ct.byref(panel), 12.0, 8.0, 1)
        ui.reconlUiLabel(handle, b"Hello ctypes", 0, None)
        hit = ui.reconlUiButton(handle, b"go", b"Go", 1)
        ui.reconlUiEnd(handle)
        ui.reconlUiEndFrame(handle)
        return hit

    cx, cy = btn.x + btn.w / 2, btn.y + btn.h / 2
    begin(cx, cy, down=True, pressed=True)
    assert draw_button_tree() == 0, "clicked on press"
    begin(cx, cy, released=True)
    assert draw_button_tree() == 1, "no click on release"

    # Press the later row button and release there with the earlier button
    # first in call order. Release handling must remain with the captured ID.
    begin()
    assert ui.reconlUiButton(handle, b"order-a", b"First", 0) == 0
    order_b = ui.reconlUiButton(handle, b"order-b", b"Second", 0)
    order_rect = ui.reconlUiLastRect(handle)
    assert order_rect.y >= 34
    ui.reconlUiEndFrame(handle)
    ox, oy = order_rect.x + order_rect.w / 2, order_rect.y + order_rect.h / 2
    begin(ox, oy, down=True, pressed=True)
    assert ui.reconlUiButton(handle, b"order-a", b"First", 0) == 0
    assert ui.reconlUiButton(handle, b"order-b", b"Second", 0) == 0
    ui.reconlUiEndFrame(handle)
    begin(ox, oy, released=True)
    assert ui.reconlUiButton(handle, b"order-a", b"First", 0) == 0
    assert ui.reconlUiButton(handle, b"order-b", b"Second", 0) == 1
    ui.reconlUiEndFrame(handle)

    # If a host begins a release frame but abandons it before EndFrame, its
    # capture must be cleared before the next press can belong to a new widget.
    begin(20, 10, down=True, pressed=True)
    assert ui.reconlUiButton(handle, b"abandoned", b"Abandoned", 0) == 0
    ui.reconlUiEndFrame(handle)
    begin(20, 10, released=True)
    begin(20, 10, down=True, pressed=True)  # intentionally no EndFrame above
    assert ui.reconlUiButton(handle, b"recovered", b"Recovered", 0) == 0
    ui.reconlUiEndFrame(handle)
    begin(20, 10, released=True)
    assert ui.reconlUiButton(handle, b"recovered", b"Recovered", 0) == 1
    ui.reconlUiEndFrame(handle)

    # toggle: press/release pair flips 0 -> 1
    begin()
    on = ct.c_uint32(0)
    ui.reconlUiToggle(handle, b"t", b"Toggle", ct.byref(on))
    tr = ui.reconlUiLastRect(handle)
    ui.reconlUiEndFrame(handle)
    tx, ty = tr.x + tr.w / 2, tr.y + tr.h / 2
    begin(tx, ty, down=True, pressed=True)
    ui.reconlUiToggle(handle, b"t", b"Toggle", ct.byref(on))
    ui.reconlUiEndFrame(handle)
    assert on.value == 0, "toggle flipped on press"
    begin(tx, ty, released=True)
    changed = ui.reconlUiToggle(handle, b"t", b"Toggle", ct.byref(on))
    ui.reconlUiEndFrame(handle)
    assert changed == 1 and on.value == 1, f"toggle did not flip: changed={changed} on={on.value}"

    # 5. resize, undersized refusal, counter
    assert ui.reconlUiResize(handle, 640, 360) == 0
    begin()
    ui.reconlUiLabel(handle, b"resized", 2, None)
    ui.reconlUiEndFrame(handle)
    big = ct.create_string_buffer(640 * 360 * 4)
    assert ui.reconlUiRender(handle, big, 640 * 360 * 4) == 0
    small = ct.create_string_buffer(64)
    assert ui.reconlUiRender(handle, small, 64) < 0, "undersized buffer accepted"
    msg = ui.reconlUiLastError(handle).decode()
    assert "needs" in msg, f"unexpected error message: {msg!r}"
    assert ui.reconlUiFrameIndex(handle) == 2  # two good renders; refused one does not count

    ui.reconlUiDestroy(handle)
    print(f"DLL smoke: exports OK, vcount={vcount} icount={icount}, "
          f"clicks landed, resize+refusal OK, err={msg!r}")
    print("reconl_ui.dll foreign-consumer test: ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
