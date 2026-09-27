"""Benchmark release-mode UI and scene renders at common display sizes.

From the repository root or frontend/:
  python frontend/spike/benchmark_resolutions.py
  python spike/benchmark_resolutions.py --frames=60 --warmup=10

The UI path calls the published reconl_ui C ABI and renders a small dashboard
through its real renderer. The non-UI path uses reconl-bench's reference scene,
with shadows off to isolate resolution-dependent color rendering/readback.
"""
import argparse
import ctypes as ct
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
DLL_DIR = FRONTEND / "zig-out" / "bin"
UI_DLL = DLL_DIR / "reconl_ui.dll"
BENCH_EXE = ROOT / "target" / "release" / ("reconl-bench.exe" if os.name == "nt" else "reconl-bench")
RESOLUTIONS = (
    ("720p", 1280, 720),
    ("1080p", 1920, 1080),
    ("2K / 1440p", 2560, 1440),
    ("4K", 3840, 2160),
)


class Input(ct.Structure):
    _fields_ = [
        ("px", ct.c_float),
        ("py", ct.c_float),
        ("down", ct.c_uint32),
        ("pressed", ct.c_uint32),
        ("released", ct.c_uint32),
        ("wheel", ct.c_float),
        ("dt_ms", ct.c_float),
    ]


class Rect(ct.Structure):
    _fields_ = [("x", ct.c_float), ("y", ct.c_float), ("w", ct.c_float), ("h", ct.c_float)]


def run(command, *, cwd, env=None):
    print("$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def load_ui():
    if not UI_DLL.is_file():
        raise FileNotFoundError(f"Missing {UI_DLL}; `zig build shared` did not produce the UI DLL")
    os.environ["PATH"] = str(DLL_DIR) + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(str(DLL_DIR))
    except (AttributeError, OSError):
        pass

    ui = ct.CDLL(str(UI_DLL))
    ui.reconlUiCreate.argtypes = [ct.c_uint32, ct.c_uint32]
    ui.reconlUiCreate.restype = ct.c_void_p
    ui.reconlUiDestroy.argtypes = [ct.c_void_p]
    ui.reconlUiBegin.argtypes = [ct.c_void_p, ct.POINTER(Input)]
    ui.reconlUiBeginPanel.argtypes = [ct.c_void_p, ct.POINTER(Rect), ct.c_float, ct.c_float, ct.c_uint32]
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
    ui.reconlUiSlider.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.POINTER(ct.c_float), ct.c_float, ct.c_float]
    ui.reconlUiSlider.restype = ct.c_uint32
    ui.reconlUiProgress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_float, ct.c_float]
    ui.reconlUiSparkline.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_uint32, ct.c_float]
    return ui


def benchmark_ui(ui, width, height, warmup, frames):
    handle = ui.reconlUiCreate(width, height)
    if not handle:
        raise RuntimeError(f"reconlUiCreate({width}, {height}) failed")

    pixels = ct.create_string_buffer(width * height * 4)
    inp = Input(-1e6, -1e6, 0, 0, 0, 0.0, 1000.0 / 60.0)
    panel = Rect(16.0, 16.0, float(width - 32), float(height - 32))
    toggle_value = ct.c_uint32(1)
    slider_value = ct.c_float(0.72)
    samples = (ct.c_float * 32)(*(0.5 + 0.45 * ((i % 8) / 7.0) for i in range(32)))

    def frame(index, split=None):
        build_started = time.perf_counter_ns() if split is not None else 0
        ui.reconlUiBegin(handle, ct.byref(inp))
        ui.reconlUiBeginPanel(handle, ct.byref(panel), 20.0, 10.0, 0)
        if ui.reconlUiLabel(handle, b"ReconL Studio", 0, None) <= 0.0:
            raise RuntimeError("UI label produced no layout")
        ui.reconlUiLabel(handle, b"Resolution performance dashboard", 3, None)
        ui.reconlUiButton(handle, b"deploy", b"Deploy", 1)
        ui.reconlUiButton(handle, b"cancel", b"Cancel", 0)
        ui.reconlUiToggle(handle, b"live", b"Live updates", ct.byref(toggle_value))
        ui.reconlUiSlider(handle, b"quality", b"Quality", ct.byref(slider_value), 0.0, 1.0)
        ui.reconlUiProgress(handle, b"build", (index % 60) / 59.0, 8.0)
        ui.reconlUiSparkline(handle, samples, len(samples), 44.0)
        ui.reconlUiEnd(handle)
        if ui.reconlUiEndFrame(handle) != 1:
            raise RuntimeError("UI frame produced no mesh")
        if split is not None:
            split[0] += time.perf_counter_ns() - build_started
        render_started = time.perf_counter_ns() if split is not None else 0
        result = ui.reconlUiRender(handle, pixels, len(pixels))
        if result != 0:
            raise RuntimeError(f"reconlUiRender returned {result}")
        if split is not None:
            split[1] += time.perf_counter_ns() - render_started

    try:
        for i in range(warmup):
            frame(i)
        durations = []
        split = [0, 0]
        for i in range(frames):
            started = time.perf_counter_ns()
            frame(warmup + i, split)
            durations.append(time.perf_counter_ns() - started)
        mean_ms = statistics.fmean(durations) / 1_000_000
        build_ms = split[0] / frames / 1_000_000
        render_ms = split[1] / frames / 1_000_000
        return mean_ms, 1000.0 / mean_ms, build_ms, render_ms
    finally:
        ui.reconlUiDestroy(handle)


def benchmark_scene(backend, width, height, warmup, frames):
    command = [
        str(BENCH_EXE), f"--backend={backend}", f"--width={width}", f"--height={height}",
        f"--frames={frames}", f"--warmup={warmup}", "--shadows=off",
    ]
    env = os.environ.copy()
    if os.name == "nt":
        env["PATH"] = str(ROOT / "target" / "release") + os.pathsep + env.get("PATH", "")
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    wall = next((line for line in result.stdout.splitlines() if line.strip().startswith("wall ")), None)
    match = re.search(r"avg\s+([\d.]+)\s+ms.*\(([\d.]+)\s+fps at the mean\)", wall or "")
    if not match:
        raise RuntimeError(f"Could not parse reconl-bench wall measurement:\n{result.stdout}")
    return float(match.group(1)), float(match.group(2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=20, help="measured frames per configuration (default: 20)")
    parser.add_argument("--warmup", type=int, default=5, help="unmeasured frames per configuration (default: 5)")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup non-negative")

    run(["cargo", "build", "--release", "-p", "reconl-bench"], cwd=ROOT)
    run(["zig", "build", "shared", "-Doptimize=ReleaseFast", "-Dreconl-profile=release"], cwd=FRONTEND)
    if not BENCH_EXE.is_file():
        raise FileNotFoundError(f"Missing {BENCH_EXE}")
    ui = load_ui()

    print("\nRelease-mode mean frame time / average FPS; non-UI uses the reference scene with shadows off.")
    print(f"Measured frames={args.frames}, warmup={args.warmup}; UI C ABI uses automatic backend selection.\n")
    print(f"{'Workload':<20} {'Resolution':<14} {'Mean ms':>11} {'FPS':>10} {'30+ FPS':>10}")
    print("-" * 69)
    for name, width, height in RESOLUTIONS:
        ui_ms, ui_fps, ui_build_ms, ui_render_ms = benchmark_ui(ui, width, height, args.warmup, args.frames)
        print(f"{'UI / auto':<20} {name:<14} {ui_ms:>11.3f} {ui_fps:>10.1f} {'yes' if ui_fps >= 30 else 'NO':>10}")
        print(f"  UI stage mean: build {ui_build_ms:.3f} ms, renderer/present {ui_render_ms:.3f} ms")
        for backend in ("soft-cpu", "d3d11"):
            mean_ms, fps = benchmark_scene(backend, width, height, args.warmup, args.frames)
            print(f"{'scene / ' + backend:<20} {name:<14} {mean_ms:>11.3f} {fps:>10.1f} {'yes' if fps >= 30 else 'NO':>10}")


if __name__ == "__main__":
    main()
