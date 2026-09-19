# ReconL

A renderer with a C ABI: one portable core, several backends, and a **reference
tier** that defines correctness. The software rasteriser is not a fallback - it is
the thing every other tier is diffed against, so "the GPU and the CPU agree" is a
test result rather than a hope.

* **C ABI only at the boundary.** `include/reconl/reconl.h` is the contract:
  opaque ref-counted handles, versioned structs, no exceptions, no panics across
  the boundary, and a host allocator the library never bypasses.
* **Tiers, not flags.** A device resolves to one of T0..T4 from a probe and a
  budget, and can step down with a recorded reason. A weaker tier renders a
  cheaper frame; it does not silently render a different scene.
* **Bounded.** One per-allocation ceiling in `core::budget`, enforced before any
  allocator or driver is touched, so an absurd request is refused with a code
  instead of attempted.

## Layout

| Path | What lives there |
|---|---|
| `core/` | tiers, budget accounting, stats, logging, errors |
| `raster/` | the reference rasteriser: tile binning, fixed-point edges, shading, textures |
| `shadow/` | cascade fitting and snapping, the filter kernels, the bias table, the static-cascade cache key |
| `resource/` | mip chains, streaming, the spill arena |
| `backends/soft-cpu/` | the reference tier (T2), SIMD where it is exact |
| `backends/d3d11/` | the same passes on hardware |
| `backends/null/` | a deterministic no-op, for ABI and CI tests |
| `ffi/` | handles, validation, last-error, frame state - no rendering logic |
| `tools/host/` | host-side plumbing shared by the tools: allocator, device, scene, frame encoding, units |
| `tools/reconl-info/` | probe and tier report |
| `tools/reconl-bench/` | timed runs with a fingerprint, a trace and stage attribution |
| `tools/reconl-diff/` | golden-image compare, exact and tolerance modes |
| `docs/` | `architecture.md`, `determinism.md`, `bias.md`, `offload.md` |

Backend rule: **a backend implements the portable core, it does not define it.**
No backend-specific type appears in `include/`; a feature gap is declared in the
device's caps and resolved by the tier resolver or a fallback path.

## How a call flows

`reconlCreateDevice` → a probe result and a budget decide the tier → the backend
implements the `Backend` trait → `reconlBeginFrame` reserves the frame's targets →
`reconlCmd*` appends to a command list → `reconlSubmit` commits and renders →
`reconlPresent` produces pixels. The frame-state rules, including what a failed
call leaves behind, are documented next to those declarations in the header, and
the tool-side view of the same flow is in `docs/architecture.md`.

## The tools

```sh
cargo run -p reconl-info  -- --probe-only            # what the machine has, no device created
cargo run -p reconl-info  -- --backend=soft-cpu      # what a device grants, and its memory ledger
cargo run -p reconl-bench -- --backend=soft-cpu --frames=60 --trace=run.csv
cargo run -p reconl-bench -- --shadows=off --png=off.png
cargo run -p reconl-diff  -- render tests/golden/soft-cpu-shadow.png
cargo run -p reconl-diff  -- compare tests/golden/soft-cpu-shadow.png
```

`reconl-bench` prints the configuration it ran with **before** the first frame -
backend, tier, device, driver, scene, resolution, shadow mode, budgets, threads,
frame and warmup counts - then reports `min / avg / max`, the host's split at the
three ABI boundaries, the device's own stage split, shadow pass cost with its
cascade count and map resolution, and how many allocations the frame loop made.
`--trace=FILE` writes every frame plus a per-second summary; `--png=FILE` writes the
last frame, which is how a `--shadows=off` or `--spill=1` run is checked against the
golden instead of taken on trust.

`reconl-diff`'s exact mode is the default because that is what the reference tier
produces against itself. The 1%-budget tolerance mode is for the *cross-tier*
comparison, and its value is measured, not chosen (see below).

## How it was measured

Everything here was produced by the tools in this repository on one machine -
Windows, Intel UHD Graphics, 4 worker threads - and the commands are given so a
rerun is a comparison rather than a claim. **These numbers are provisional**: they
are single runs at 64x64 on the reference scene (2 draw calls, 3 triangles, one
shadow-casting directional light, 2 cascades at 512x512), which is small enough
that the shadow pass dominates and large enough to catch a regression.

| Measurement | soft-cpu (T2) | d3d11 (T1) |
|---|---|---|
| Frame wall time, 60 frames | min 14.2 / avg 30.6 / max 76.8 ms | min 2.5 / avg 3.4 / max 7.0 ms |
| Shadow pass | 11.4 ms | 294 us |
| Colour raster | 17.4 ms | 63 us |
| Present (readback) | 296 us | 56 us |
| Allocations per frame | 8 | 2 |
| Cross-tier divergence | 14 of 4096 pixels, worst channel delta 46, all within 4 px of the shadow's edge | |
| Documented cross-tier tolerance | `--tolerance=48` inside the 1% budget | |

```sh
cargo run -p reconl-bench -- --backend=soft-cpu --tier=2 --frames=60 --warmup=5
cargo run -p reconl-bench -- --backend=d3d11   --frames=60 --warmup=5
cargo run -p reconl-diff  -- compare tests/golden/soft-cpu-shadow.png
```

Scaling, same scene and command, mean wall time per frame:

| Resolution | soft-cpu (T2) | d3d11 (T1) | GPU advantage |
|---|---|---|---|
| 64x64 | 30.6 ms | 3.4 ms | 9.0x |
| 256x256 | 139.6 ms | 28.7 ms | 4.9x |
| 512x512 | 462.9 ms | 94.4 ms | 4.9x |

The hardware tier's own stage counters do not explain its frame. At 256x256 the
device reports 308 us of shadow pass and 65 us of colour raster inside a 28.7 ms
frame, and at 512x512 it reports 278 us and 76 us inside a 94 ms frame: under 1%
of the time is attributed, while `present` (the readback) measures well under
1 ms. Something between the two - most likely the synchronous drain the
`present_to_memory` readback forces, and the per-frame staging copies - is where
the frame goes. **This is the largest unexplained number in the project and the
cheapest performance work available**, and it is only visible because the tool
prints the device's split next to the host's.

The same load also shows what a GPU tier does when it cannot cope. At 512x512
with `--repeat=200`, `reconlSubmit` fails:

```
reconlSubmit: RECONL_ERR_DEVICE_LOST (-6) — device reported: map colour staging
  texture: The GPU device instance has been suspended. Use GetDeviceRemovedReason
  to determine the appropriate action. (0x887A0005)
```

That one is a genuine removal (`DXGI_ERROR_DEVICE_REMOVED`, which the classifier
maps to `DEVICE_LOST`). Contrast a frame the driver merely *refuses* - a
32768-wide target, past its 16384-texel limit - which surfaces as
`RECONL_ERR_INVALID_ARGUMENT (-1)` with the driver's `E_INVALIDARG` in the
message, and the device still usable.

The removal itself used to be a real
host-visible dead end - the frame failed, the device was unusable, and there was
no path off it. Now the device continues on the reference tier (the offload,
`docs/offload.md`): the same command finishes with `presented 1  dropped 0
failures 0`, a `final` line reading `backend soft-cpu  tier T2/cpu-ram  reason
device removed (DXGI_ERROR_DEVICE_REMOVED)`, and a frame whose `--png` output
`cmp`s identical to the reference tier's own render of it. (`safe-path events`
reads 0 in that run because the benchmark resets the device's counters after its
warmup and the GPU dies on the first warmup frame; the tier the run finished on
is the one thing that reset cannot hide.)

Two of those rows are findings rather than results. **Allocations per frame are
8 and 2, and the acceptance criterion is zero** - `docs/determinism.md` states the
rule and this is the measurement that says it is not met. And `reconl-info`
reports a **512 MiB per-allocation ceiling**; that bounds one allocation, not the
total, so an uncapped device still accepts a frame that fits in 512 MiB.

The golden is byte-identical when produced by either tool, at any worker count:
`reconl-bench --png` and `reconl-diff render` both read the scene from
`tools/host`, so a timing run and a golden cannot disagree about what they drew.

## What is provisional

* **The backend matrix is mostly unbuilt.** `d3d12`, `vulkan`, `gl`, `metal`,
  `webgpu` and `wasm-webgl2` are enum values with no implementation behind them;
  the honest reading of the matrix today is "d3d11, plus a reference tier".
* **No GPU timestamps.** The hardware path reports the stages it can measure on
  the host side of a submit. `reconl-bench` prints the device's split as reported
  and says so when a backend reports nothing, rather than printing a plausible
  zero as a number.
* **The tier caps over-promise.** `soft-cpu` advertises PCF 5x5 and PCSS-lite at
  every tier while the tier policy may clamp a given device to 3x3, so a host
  reading the caps can expect a filter it will not get.
* **The reference scene has a second copy** in `ffi/tests/tiers.rs`. The tools now
  share one owner (`tools/host`); the test's copy is kept in step by hand and is
  the next thing to collapse.
* **Driver-error classification is one mapping, with one known blind spot.** The
  d3d11 backend classifies every driver/OS `HRESULT` in one place
  (`classify_hresult` in `backends/d3d11/src/imp.rs`): the DXGI removal family is
  `RECONL_ERR_DEVICE_LOST`, `E_OUTOFMEMORY` is `RECONL_ERR_OUT_OF_MEMORY`,
  `E_INVALIDARG` is `RECONL_ERR_INVALID_ARGUMENT`, `E_NOTIMPL`/`E_NOINTERFACE`
  are `RECONL_ERR_NOT_SUPPORTED`, and anything unrecognised is
  `RECONL_ERR_BACKEND_UNAVAILABLE` - all existing codes, nothing invented. The
  per-class host guidance is documented next to the `ReconLResult` enum in
  `include/reconl/reconl.h`, and a driver-rejected 32768-wide frame is pinned
  end to end: the C probe and `ffi/tests/tiers.rs` both see `-1`, not `-6`. The
  blind spot: any `HRESULT` outside the mapped set lands on
  `BACKEND_UNAVAILABLE`, whose documented advice is "probe before destroy" - an
  unrecognised *removal-shaped* code would therefore not fail over
  automatically. The mapped set is the family D3D11 documents, so this is a
  known simplification rather than an open defect.
* **The offload's return trip needs a frame-time target.** The device comes back
  to the hardware after `downgrade_after_frames` frames inside `target_frame_ms`,
  once per resolution and shadow plan. A host that sets `target_frame_ms = 0` has
  no window to settle in, so an offloaded device stays on the reference tier until
  it is destroyed. The benchmark's own default is 0, which is why the `--repeat=200`
  run above ends (correctly) on the CPU.

## Build and test

```sh
cargo test --workspace          # unit, ABI layout, golden and cross-tier tests
cargo run  -p reconl-diff -- compare tests/golden/soft-cpu-shadow.png
```

The C ABI is verified from both sides: `ffi/tests/abi_layout.rs` generates
`_Static_assert`s from the Rust structs and asks a C compiler to compile them
against the shipped header, so a field added on one side and not the other stops
the build. `ffi/tests/abi.rs` drives every entry point through the C-shaped surface,
including the failure paths, and the runtime behaviour of the shipped DLL is
checked by C probes built against its import library.
