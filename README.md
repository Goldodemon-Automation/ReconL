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
* **A UI rendered by the renderer, not beside it.** `frontend/` is an
  immediate-mode UI core in Zig whose frames are drawn by ReconL through the same
  C ABI everything else uses. It is a second project in the same repository, and
  it is held to the same rule: no wall clock, so the same inputs twice produce
  the same frames.

## Layout

| Path | What lives there |
|---|---|
| `include/reconl/` | shipped headers: `reconl.h` (the contract), `reconl_version.h`, and `reconl_backends.h` (D3D11 adapter selection; SoftCPU/Null descriptors remain reserved) |
| `core/` | tiers, budget accounting, stats, logging, errors |
| `contract/` | `FrameInput` / `ShadowRequest`: what a backend is handed, with no dependency on any other backend |
| `scene/` | passes, resource uses and barriers, world revisions, static-vs-dynamic geometry |
| `raster/` | the reference rasteriser: tile binning, fixed-point edges, shading, textures, frame generation |
| `shadow/` | cascade fitting and snapping, the filter kernels, the bias table, the static-cascade cache key |
| `resource/` | mip chains, streaming, the spill arena |
| `backends/soft-cpu/` | the reference tier (T2), SIMD where it is exact |
| `backends/d3d11/` | the same passes on hardware |
| `backends/null/` | a deterministic no-op, for ABI and CI tests |
| `ffi/` | handles, validation, last-error, frame state, the offload ladder - no rendering logic |
| `frontend/` | the Zig UI core, its C ABI, and the showcase that drives them |
| `probes/` | 17 C programs that audit the *shipped* library; see `probes/README.md` |
| `tools/host/` | host-side plumbing shared by the tools: allocator, device, scene, frame encoding, units |
| `tools/reconl-info/` | probe and tier report |
| `tools/reconl-bench/` | timed runs with a fingerprint, a trace and stage attribution |
| `tools/reconl-diff/` | golden-image compare, exact and tolerance modes |
| `tools/png/` | the minimal deterministic PNG codec the goldens are written with |
| `tests/golden/` | the committed golden: the reference scene at 64x64, `sha256 7e8ecbab…` |
| `docs/` | `architecture.md`, `determinism.md`, `bias.md`, `offload.md` |
| `Context.md` | the map: which crate owns what, and how one frame travels through them |

Backend rule: **a backend implements the portable core, it does not define it.**
The portable header carries only the adapter record and selector enum; D3D11's
backend-specific creation descriptor remains in `reconl_backends.h`.

The D3D11 adapter table is queried through `reconlEnumerateAdapters` or shown by
`reconl-info`. Each entry reports its name, D3D11 11.0 usability, video/shared
memory, and LUID. DXCore reports integrated-vs-discrete accurately where present;
on older Windows versions adapter type is `unknown`. The C ABI defaults to
`auto` (discrete first, then any usable adapter), while applications can pin an
integrated GPU, discrete GPU, DXGI index, or exact LUID. `reconl-info` and
`reconl-bench` accept the same `--adapter=auto|integrated|discrete|INDEX|luid:HEX`
selector. This extends the D3D11 descriptor contract and bumps the library to
version 0.1.1 / ABI 101; SoftCPU and Null descriptors remain reserved.

## How a call flows

`reconlCreateDevice` → a probe result and a budget decide the tier → the backend
implements the `Backend` trait → `reconlBeginFrame` reserves the frame's targets →
`reconlCmd*` appends to a command list → `reconlSubmit` commits and renders →
`reconlPresent` produces pixels. The frame-state rules, including what a failed
call leaves behind, are documented next to those declarations in the header, and
the tool-side view of the same flow is in `docs/architecture.md`.

`reconlPresent` is where a frame's cost is *completed*, because the readback the
host waited for is part of the number the tier ladder judges. On the hardware
tier that readback is the largest single line in the frame - see the table below.
A host that wants more images than it renders asks for them instead
(`reconlPresentGenerated`, below), and a pass can confine itself to a viewport:
`ReconLRenderPassDesc.viewport_width`/`viewport_height` are in *frame* pixels, and
`raster::rendered_viewport` is the one rule that resolves them against whatever
target the tier actually renders into, so a tier at half resolution confines the
same fraction of the window. `0` in an axis means the whole target and an
oversized viewport is clamped rather than refused.

## The frontend

`frontend/` is a second, deliberately separate project in the same repository: a
**Zig immediate-mode UI core** whose frames are rendered *by* ReconL rather than
beside it. Roughly 5,400 lines of Zig, and the only thing it shares with the Rust
workspace is the C ABI it renders through.

The split is the point. `src/root.zig` is everything needed to *decide* a frame -
layout, text, animation, tessellation - and it never links `libreconl`, so its
tests run with no GPU, no window and no renderer. `src/backend.zig` is the
rendering half: it knows the frame contract from `include/reconl/reconl.h` and
nothing at all about widgets, and it performs begin / reset / open pass / push
transforms / draw / end pass / submit / present exactly the way
`tools/host/src/frame.rs` does, so a diff between a Zig-driven frame and a
Rust-driven one can only contain UI.

| module | owns |
|---|---|
| `geom`, `theme` | vectors, rects, colours, the type scale |
| `anim` | hover ramps, press dips, knob and scroll springs |
| `tess`, `polygon` | triangulation, and the rounded-rect and bevel emitters |
| `font`, `text` | a TrueType reader and a line breaker |
| `ui` | panels, rows, scroll, label, button, toggle, slider, progress, separator, sparkline |
| `png` | the showcase's frame output |
| `backend` | the ReconL half |
| `c_api` | the C ABI skin over all of the above |

`frontend/include/reconl_ui.h` is that ABI, and it is independent of `reconl.h`
on purpose: a consumer needs exactly that one file, and the renderer stays an
implementation detail behind `reconlUiRender`. A frame is `reconlUiBegin` ->
widgets -> `reconlUiEndFrame` -> `reconlUiRender`. Identity is imgui-style
(`"Save##a"` and `"Save##b"` are two buttons that both read Save), and every
animation advances by the host's `input.dt_ms` - never by a wall clock - which is
what makes a golden a golden. It is built for Java (Panama FFM), Kotlin, C and
Zig hosts, and a Java 21 consumer is checked in at
`frontend/spike/java/ReconLUiSmoke.java`.

```sh
cargo build                 # from the repo root, first: the rest need the DLL
cd frontend
zig build test              # 42 core unit tests - no ReconL, no GPU, no window
zig build abi-test          # 11 tests driving the UI through the C ABI and the real library
zig build shared            # zig-out/bin/reconl_ui.dll, plus reconl.dll and the import lib
zig build demo              # the animated showcase, 60 frames at 960x540
```

`zig build demo` writes `zig-out/showcase/frame_NNN.png`: a control-room dashboard
whose hover ramps, press dips, toggle spring, slider drag and scroll spring are
all functions of `dt_ms` and a scripted input, never of a clock. The script aims
at the rects the first frame records rather than at hard-coded coordinates, so
it stays correct if the layout moves.

When `libreconl` is missing the build says so and skips those steps instead of
failing - the same "skips with a printed reason" rule the C probes follow.
`frontend/spike/` holds the foreign-process checks that drive the built DLL the
way a real consumer would: `verify_dll.py`, `verify_lifecycle.py`,
`verify_showcase.py` and `verify_java.py`.

## The tools

```sh
cargo run -p reconl-info  -- --probe-only            # probe plus D3D11 adapter list
cargo run -p reconl-info  -- --backend=d3d11 --adapter=integrated
cargo run -p reconl-bench -- --backend=d3d11 --adapter=luid:0000000000000001 --frames=60
cargo run -p reconl-info  -- --backend=soft-cpu       # what a device grants, and its memory ledger
cargo run -p reconl-bench -- --backend=soft-cpu --frames=60 --trace=run.csv
cargo run -p reconl-bench -- --shadows=off --png=off.png
cargo run -p reconl-bench -- --backend=soft-cpu --resolution=512x512 --framegen=2
cargo run -p reconl-diff  -- render tests/golden/soft-cpu-shadow.png
cargo run -p reconl-diff  -- compare tests/golden/soft-cpu-shadow.png
```

`reconl-bench` prints the configuration it ran with **before** the first frame -
backend, tier, adapter preference, device, driver, scene, resolution, shadow mode, budgets, threads,
frame and warmup counts - then reports `min / avg / max`, the host's split at the
three ABI boundaries, the device's own stage split, shadow pass cost with its
cascade count and map resolution, and how many allocations the frame loop made.
`--trace=FILE` writes every frame plus a per-second summary; `--png=FILE` writes the
last frame, which is how a `--shadows=off` or `--spill=1` run is checked against the
golden instead of taken on trust.

`reconl-diff`'s exact mode is the default because that is what the reference tier
produces against itself. The 1%-budget tolerance mode is for the *cross-tier*
comparison, and its value is measured, not chosen (see below).

It compares at any resolution the tools can render: a compare takes its size from
the golden (a candidate of another size is an error, not a crop), `render --size=N`
writes a golden of that size, the 1% budget scales with the frame, and a difference
is reported per channel - the worst delta and the pixels differing on each of R, G,
B and A, the box those pixels occupy, and the pixels themselves. That is what makes
the 512x512 frame a benchmark writes diffable with the same tool and the same
tolerance.

## How it was measured

Everything here was produced by the tools in this repository on one machine -
Windows, Intel UHD Graphics, 4 worker threads - and **the command behind every row
is given with it**, so a rerun is a comparison rather than a claim. These are single
runs at 64x64 on the reference scene (2 draw calls, 3 triangles, one shadow-casting
directional light, 2 cascades at 512x512), which is small enough that the shadow
pass dominates and large enough to catch a regression.

| Measurement | soft-cpu (T2) | d3d11 (T1) |
|---|---|---|
| Frame wall time, 60 frames | min 1.18 / avg 1.35 / max 2.80 ms | min 0.65 / avg 0.84 / max 1.23 ms |
| Device-attributed total | 1.34 ms (99% of the frame) | 655 us (78%) |
| Shadow pass | 869 us | 137 us |
| Colour raster | 466 us | 28 us |
| Present (readback) | 6.5 us | 566 us |
| Allocations per frame | 0 | 0 |
| Cross-tier divergence | 14 of 4096 pixels, worst channel delta 46, all within 4 px of the shadow's edge | |
| Documented cross-tier tolerance | `--tolerance=48` inside the 1% budget | |

The same two runs, with `--png=sc.png` and `--png=d11.png` added, are what the two
pixel rows compare (`--png` writes the last frame the host read back):

```sh
cargo build --release
target/release/reconl-bench --backend=soft-cpu --tier=2 --frames=60 --warmup=5
target/release/reconl-bench --backend=d3d11   --frames=60 --warmup=5
target/release/reconl-diff  compare tests/golden/soft-cpu-shadow.png sc.png                  # identical (64x64, 4096 px)
target/release/reconl-diff  compare tests/golden/soft-cpu-shadow.png d11.png                 # 14 of 4096 px differ: exit 1
target/release/reconl-diff  compare tests/golden/soft-cpu-shadow.png d11.png --tolerance=48   # within the 1% budget: exit 0
```

That last comparison is the row above it, measured rather than asserted: it reports
`14 of 4096 px differ (0%), worst channel delta 46 (R 11, G 17, B 46, A 0), differing
per channel (R 14, G 5, B 5, A 0), box x 2..62 y 38..55`, and the diverging pixels
are the ones within 4 px of a pixel the shadow changes.

Scaling, same scene and the same commands with `--width=N --height=N`, mean wall
time per frame:

| Resolution | soft-cpu (T2) | d3d11 (T1) | GPU advantage |
|---|---|---|---|
| 64x64 | 1.35 ms | 0.84 ms | 1.6x |
| 256x256 | 4.64 ms | 1.09 ms | 4.3x |
| 512x512 | 15.3 ms | 2.30 ms | 6.7x |

The hardware tier's advantage grows with the frame because its per-frame cost is
dominated by a fixed readback rather than by pixels: 1.6x at 64x64, 6.7x at
512x512.

The hardware tier's frame is dominated by the readback the host waits for, and that
is now measured rather than inferred: at 256x256 `present` is 816 us of a 1.088 ms
frame and the device attributes 926 us of it; at 512x512 `present` is 1.885 ms of a
2.298 ms frame and the device attributes 2.013 ms. So 85-88% of a hardware frame is
attributed at those sizes, and within it the largest single line is the readback,
not the raster (55 us of colour raster at 512x512, against 1.885 ms of present).
The hardware backend has no GPU timestamp queries - its stage numbers are CPU
submit times, which the backend's own module comment says - and the null tier
reports the same kind of measurement, which is what keeps the tier ladder honest
about what it is comparing. The reference tier's frame has the opposite shape (at
512x512, 13.7 ms of raster inside 15.3 ms), and that is the cheapest performance
work left on the CPU side: the shadow pass and the raster are what it pays.

The same load also shows what a GPU tier does when it cannot cope. At 512x512 with
`--repeat=200`, this machine's driver once suspended the device mid-run and
`reconlSubmit` failed:

```
reconlSubmit: RECONL_ERR_DEVICE_LOST (-6) — device reported: map colour staging
  texture: The GPU device instance has been suspended. Use GetDeviceRemovedReason
  to determine the appropriate action. (0x887A0005)
```

That one is a genuine removal (`DXGI_ERROR_DEVICE_REMOVED`, which the classifier
maps to `DEVICE_LOST`). It is a state of the machine, not a property of the load:
a driver whose device instance is already suspended reports it on the first frame,
and a healthy driver renders the same 512x512 load indefinitely - re-measured, the
identical command completes 5 warmup + 60 measured frames with `failures 0`, 2.32 ms
a frame, and never leaves `d3d11`. Contrast a frame the driver merely *refuses* - a
32768-wide target, past its 16384-texel limit - which surfaces as
`RECONL_ERR_INVALID_ARGUMENT (-1)` with the driver's `E_INVALIDARG` in the
message, and the device still usable.

The removal itself used to be a real
host-visible dead end - the frame failed, the device was unusable, and there was
no path off it. Now the device continues on the reference tier and the frame that
faulted is re-rendered there (the offload, `docs/offload.md`), and that path is
gated from both sides: `ffi/tests/tiers.rs` drives the ladder's offload and return
legs through the real ABI, and the `offload` C probe (33 checks) drives the same
policy against the shipped DLL. What cannot be driven is the removal itself - there
is no fault hook, a failover needs a genuine driver verdict, and the `--repeat=200`
removal above was a suspended device instance on this machine rather than a
reproducible condition - so the fault's own verdict is the one step those pins do
not take.

One row that used to be a finding is now a result: **allocations per frame are 0
and 0**, which is `docs/determinism.md`'s acceptance criterion met at every size
measured (64x64, 256x256, 512x512, and the null tier). The exception is the disk
tier's streaming path, measured in the section below: a frame the RAM cap cannot
hold makes 3 allocations of 6144 bytes over 4 measured frames at 512x512 under a
2 MiB cap (0.75 per frame). And `reconl-info` reports a **512 MiB per-allocation
ceiling**; that bounds one allocation, not the total, so an uncapped device still
accepts a frame that fits in 512 MiB.

The golden is byte-identical when produced by either tool, at any worker count:
`reconl-bench --png` and `reconl-diff render` both read the scene from
`tools/host`, so a timing run and a golden cannot disagree about what they drew.
Measured here: the benchmark's own 64x64 frame (`--png=sc.png` above) compares
`identical (64x64, 4096 px)` against the committed file, which is still
`sha256 7e8ecbab…` - the disk-tier work below did not move the reference tier's
pixels.

## Frame generation

A host that can render 60 frames a second and display 120 asks the device for the
frames in between instead of rendering them. The device reprojects the newest
frame it was handed along the camera motion the host declared - no shaders, no
second render, nothing about how a frame is drawn changes - and the host asks
per frame, which is what makes it a toggle a game flips with its own quality
settings rather than a mode the device is put into.

`raster/src/framegen.rs` owns the warp and nothing else knows how to do it:
`reprojection` builds one matrix per generated frame
(`P_prev · V_prev · V_cur⁻¹ · P_cur⁻¹`) and `generate` applies one backward warp,
bilinear, clamped at the frame's edge. A still camera copies the frame it was
given, to the byte, which is why a menu, a paused game and the reference scene
are all still scenes. The retained pixels, the depth behind them and the camera
pair are the FFI's (`ffi/src/lib.rs::FrameGen`), because what a generated frame is
warped from is the image the host was handed.

The contract is in the header: `ReconLFrameGenDesc` is trailing on
`ReconLFrameDesc`, so an older caller's struct is read as if it were NULL;
`ahead` is in `(0, 1]`; a frame that did not ask is `RECONL_ERR_NO_FRAME`; a tier
with no depth to reproject is `RECONL_ERR_NOT_SUPPORTED`; and - the part that
keeps the ladder honest - a generated image never counts in `frames_presented`,
so a host cannot make a slow tier look fast by generating more images from its
frames.

Measured with `reconl-bench --resolution=R --frames=60 --warmup=5 --framegen=R`
on the reference scene, whose camera does not move. These are therefore cost
measurements, not quality ones, and single runs rather than averages of runs:

| backend | resolution | asked | achieved | cost per generated image |
|---|---|---|---|---|
| soft-cpu | 512x512 | 2.0 | 1.96 | 2.29% of a rendered frame |
| soft-cpu | 512x512 | 3.5 | 3.39 | 1.25% |
| soft-cpu | 1080p | 3.5 | 3.30 | 2.37% |
| d3d11 | 512x512 | 2.0 | 1.83 | 9.43% |
| d3d11 | 1080p | 3.5 | 2.57 | 14.43% |

The hardware tier's multiplier is the worse one because its frames are nearly
free and its present path is not: at 1080p a rendered frame is 19.1 ms and a
generated image 2.75 ms, most of that the host readback, while the reference tier
generates a 2.66 ms image inside a 112 ms frame. The ceiling on the ratio is the
game's own ratio of what a frame costs to what a present costs, which is exactly
why this is a toggle and not a default. `reconl-bench` prints the *achieved* rate
on the wall line, so a run that delivered twice the images cannot read as a run
that got slower.

The feature found one real defect, which is the argument for it existing: the
reference tier's depth buffer held *clip z* while the hardware tier's held the
documented reversed-Z NDC depth, so the same readback meant two different things
and the warp was wrong on the CPU tier. The rasteriser now stores the NDC depth
its own `Target` documents (one less divide per pixel in the hot loop), the
shadow passes are unaffected because an orthographic matrix has `w == 1`, and the
golden is unchanged by it.

## The disk tier

T4/out-of-core is implemented, and its own measurements are what say so. Two
mechanisms make it real: a static-geometry declaration the cascade cache is keyed
on, and banded streaming for a frame the RAM cap cannot hold.

**A draw declares whether its geometry is static, and that declaration is what the
cache is keyed on.** `RECONL_BUFFER_STATIC` / `RECONL_BUFFER_DYNAMIC` on a buffer
descriptor are read into the frame's draws (`ffi/src/handle.rs::is_static_geometry`),
so a host that says its geometry does not move gets the static cascade stored in the
arena and served back from it, while a host that says nothing keeps the dynamic path
and never touches the disk. The reference scene's vertex and index buffers are
declared static, and a benchmark run at T4 with the cache on reports it with no flag
beyond `--shadows=cached`:

```sh
target/release/reconl-bench --backend=soft-cpu --tier=t4 --width=512 --height=512 \
    --frames=4 --warmup=1 --ram-cap=2MB --disk-cap=64MB --spill=1 \
    --spill-dir=spill --shadows=cached --png=streamed.png
  shadow cache    hits 1  misses 1  read 256.0 KiB  hit 256.0 KiB  frozen cascades 3  fail-safe unshadowed 0
  memory          resident peak 1.5 MiB of 2.0 MiB  spill 5.3 MiB of 64.0 MiB  max allocation 2.0 MiB
  counters        presented 4  dropped 0  failures 0  safe-path events 0  audit divergences 0
```

The ABI reads a disk cap of 0 as *no disk use at all* (`ReconLMemoryBudget`), so a
tool that is asked to spill and not told how much disk it may use supplies a default
budget rather than passing the zero through: `--spill=1` alone means a 1 GiB cap, and
`--disk-cap=SIZE` is how to say otherwise. Nothing is written outside the cap a host
declares - a run that declares one too small to hold what it needs fails and names the
requirement instead of exceeding it.

256 KiB is one cascade map served back from the arena, and `spill/arena.rcls` is
26,477,080 bytes on disk afterwards. The same frame with a cap that holds it
(`--ram-cap=512MB`) reports `spill 256.0 KiB` - the cascade and nothing else - and a
262,200-byte arena, which is the difference between the cache alone and a frame that
streams.

**A frame larger than the cap streams through the arena instead of being refused.**
The frame is cut into horizontal bands, one band resident, each band written to and
read back from its own arena key. 512x512 at T4 under a 2 MiB cap renders:
`presented 4  failures 0`, 27.4 ms a frame against 12.9 ms for the same frame with a
cap that holds it - 2.1x, which is what streaming costs. A cap too tight for even one
band is still refused, measured: 1 MiB at 512x512 fails
`reconlCreateSwapchain: RECONL_ERR_BUDGET_EXCEEDED (-5)`.

**A streamed frame is not byte-identical to a resident one, and that is measured.**
Same tier, same shadow mode, 512x512: 2 of 262144 pixels differ, worst channel delta
46, at a shadow edge - `reconl-diff` names them at (372,304) and (197,387), and
`--tolerance=48` passes them (`2 of 262144 px differ (0%), … within the 1% budget`).
With `--shadows=off` the same pair is byte-identical, and both settings are
byte-identical run to run, so what a band changes is a shadow-edge pixel rather than
the frame: a band reaches the rasteriser's 1/256-px vertex snap through a
re-projected clip space, and the pinned pixels move with the band height (256-row
bands pin one of the two, 128-row bands both). It is the same class as the cross-tier
difference above and smaller than it - measured at 512x512, `soft-cpu` against
`d3d11` differs on 761 of 262144 pixels with the same worst channel delta of 46 -
and making it byte-exact would need the rasteriser to crop by an integer row origin
instead of re-projecting clip space.
`backends/soft-cpu/src/lib.rs::band_projection` says so where it describes the cut.

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
  it is destroyed. The benchmark's own default is 0, so a benchmark run that does
  offload stays on the reference tier for the rest of the run.
* **The frontend shapes no scripts.** `font.zig` is one codepoint in, one glyph
  out: no GSUB/GPOS, no kerning pairs, and no `gvar`, so the bundled variable
  fonts render at weight 400 and the hierarchy comes from size, colour and family
  instead. Advances come from `hmtx`, so Latin UI copy is metrically correct even
  where it is not kerned, and anything beyond Latin is a different project rather
  than a hidden part of this one.

## Build and test

```sh
cargo test --workspace                     # 290 passed / 0 failed across 35 binaries
cargo test --workspace --profile tested    # the same, at release optimisation
probes/run.sh                              # 24 C-probe rows against a freshly built DLL
cd frontend && zig build test              # 42 UI core tests; no ReconL, no GPU
cd frontend && zig build abi-test          # 11 tests through the C ABI and the real library
target/release/reconl-diff compare tests/golden/soft-cpu-shadow.png
```

**Do not run the suite with `--release`.** The release profile is `panic =
"abort"` - that is the crate's ABI contract, and the release profile is what
ships - and cargo forces `panic = "unwind"` for test targets, so the whole graph,
`reconl-ffi` included, is built twice under the same names in `deps/`. A test run
then fails every few tries with an unrelated tool reporting `can't find crate for
reconl_host`. `--profile tested` is release minus the abort, so a test run builds
one graph and cannot race itself.

The C ABI is verified from both sides: `ffi/tests/abi_layout.rs` generates
`_Static_assert`s from the Rust structs and from every constant and enum the
header exports - 1118 asserts in all - and asks a C compiler to compile them
against the shipped header, so a field or a value changed on one side and not the
other fails `cargo test` (it skips, with a printed reason, where the host has no C
compiler). `ffi/tests/abi.rs` drives every entry point through the C-shaped
surface, including the failure paths and the order its answers are decided in.

`probes/` is the third leg, and deliberately not a Rust test: a behavioural claim
is a claim about the **shipped library**, so the evidence is a C program driving
its exported functions. `probes/run.sh` builds the release library, copies the
built DLL/SO into `probes/.build/`, **md5-confirms that copy against the file it
was built from**, compiles every probe against that copy and
`include/reconl/reconl.h`, and prints one summary carrying the library's md5. Each
row's full output is left in `probes/.build/<row>.out`, and `probes/README.md`
says what each of the 17 probes is for and what its check count must be.
