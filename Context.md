# Context: how ReconL fits together

ReconL is a deterministic renderer behind a stable C ABI: a software tier that
defines correctness, a D3D11 tier that must agree with it, and a ladder that moves
a device between them when the hardware dies or falls behind. The ABI is the
product; everything below it exists to make one call cheap, measurable and
repeatable.

This file is the map: which crate owns what, how one frame travels through them,
and what each mechanism is measured by. The rules of the design live in
`docs/architecture.md` (ownership) and `docs/determinism.md` (what "the same image"
means); this file does not restate them, it orients you before you read them.
`README.md` carries the current performance table and the commands behind it.

## The pieces

```
host (C, C++, Rust, a tool)
  │  the documented surface, in include/reconl/reconl.h
  ▼
ffi/                     the only place the ABI exists
  │  validate · price · translate · own the frame state machine
  ├── core/              policy: tier resolution, budget, stats, error taxonomy
  ├── contract/          FrameInput / ShadowRequest: what a backend is handed
  └── backends/
        soft-cpu/        the reference tier — the definition of correct
        d3d11/           the graphics hardware tier — must match the reference
        gpu-compute/     the GPU tiers through a vendor compute API: CUDA on
                         NVIDIA, ROCm/HIP on AMD, both opened at run time.
                         Device layer built, raster not — docs/gpu-compute.md
        null/            no-op, so ABI tests run without a GPU
              └── raster/  tiled fixed-point rasteriser, SIMD fills, ShaderRef
              └── shadow/  cascade fit, texel snap, bias presets, filters
                     └── resource/  arena, spill, mip streaming
```

- **`contract/`** exists so a GPU backend does not import its input from the CPU
  backend. It depends on data types only and decides nothing; a new backend
  depends on it and on `core`, never on another backend.
- **`raster/`** owns the pixel rules both tiers must agree on: the subpixel grid,
  the top-left fill rule, reversed-Z, the viewport-to-pixels mapping
  (`rendered_viewport`), and the byte-exact `f32 → unorm8` conversion the
  reference tier presents through. `raster/src/framegen.rs` owns the frame
  generation warp (below).
- **`shadow/`** owns the cascade plan, the texel snap, the bias presets and the
  filters. The static-cascade cache's key is here, and the tier that has an arena
  is the tier that can honour it.
- **`resource/`** owns the spill arena: an append-with-eviction store with one
  entry per key, a checksum per record, and a byte cap. It is the only place
  anything in the renderer touches the disk, and `docs/architecture.md` has its
  contract.
- **`ffi/`** owns the *frame state machine*, the composed frame cost, and the
  offload policy, and it is split by concern rather than by layer:

  | module | owns |
  |---|---|
  | `handle.rs` | the object model: `Kind`, `HandleHeader`, the child handle types, `header_of` (the only place a kind word is stamped), retain/release and their destructor dispatch |
  | `entry.rs` | the boundary glue: `FrameOwner`, `guarded_entry`, the `entry!`/`device_mut!`/`child!` macros, `cstr` |
  | `offload.rs` | why this device is on the tier it is on: `PlanKey`, `Offload`, `apply_tier_policy`, the backend rebuilds, `downgrade_entries` |
  | `layout.rs` | how bytes move between a host's buffer and a frame or texture; `host_row_layout` is the one place a pitch is judged |
  | `order.rs` | the order each frame call considers its refusals in: one `Step` table per entry point, one function per question, one runner |
  | `version.rs` | version numbers, the name tables, the log controls |
  | `sizing.rs`, `abi.rs` | what a descriptor costs, and the Rust mirror of the header |
  | `lib.rs` | the device's state, the frame state machine, and the entry points over it |

  Backends answer one call at a time; they never hold frame state, and they never
  move themselves between tiers.
- **`tools/host`** is a shared host, not a library: allocator ledger, device
  wrapper, the reference scene, and the encode/submit/present order. Each tool
  (`reconl-diff`, `reconl-bench`, `reconl-info`) is a host that drives the same
  C ABI an application does, so a timing run, a golden and a cross-tier comparison
  cannot disagree about what they drew.

## One frame, end to end

1. `reconlBeginFrame` opens the frame: size, lights, shadow request, camera. The
   frame's targets are reserved here, from the device's budget.
2. `reconlCmd*` records a command list; `reconlSubmit` validates, prices and
   translates it into `FrameInput` — the one description of a frame — and hands
   it to whichever backend owns the device. A buffer's declared usage
   (`RECONL_BUFFER_STATIC` / `RECONL_BUFFER_DYNAMIC`) travels with each draw, and
   it is what the static cascade cache is keyed on.
3. The backend runs the shadow passes, then the colour pass, into its targets.
   On the reference tier at T4, a frame whose own colour and depth do not fit the
   RAM cap is cut into horizontal bands and moved through the disk arena rather
   than refused (below).
4. `reconlPresent` reads the frame out into the host's buffer, in the host's
   layout, and is where the frame's cost is completed: the readback the host
   waited for is part of the number the tier ladder judges. On the hardware tier
   that readback is the largest single line in the frame.
5. If the frame faulted or was measured over target, the ladder decides, in
   `ffi` alone: a genuine device removal or a measured overload offloads to the
   reference tier and the frame is re-rendered there out of the draw list the
   frame already holds; a rejected argument is an argument error and never a tier
   change.

## The golden

`tests/golden/soft-cpu-shadow.png` is the reference scene, rendered by the
reference tier, committed: 64x64, `sha256 7e8ecbab…`. Two independent paths check
it — `reconl-diff compare` against the file, and `reconl-bench --png` compared to
the same file in `tools/reconl-bench/tests/cli.rs` — and it is byte-identical
unless a change to the reference tier's pixels is intended. The disk-tier work did
not move it.

`reconl-diff` is not tied to that size: a compare takes its size from the golden, a
candidate of another size is an error rather than a crop, `render --size=N` writes
a golden of that size, and a difference is reported per channel (worst delta and
pixels differing on each of R, G, B and A, the box, and the pixels themselves).
That is what makes the 512x512 frames the benchmarks write diffable with the same
tool.

The cross-tier comparison is the other end of the same tool, and its tolerance is
measured rather than chosen. Against the golden, `d3d11` differs on 14 of 4096
pixels with a worst channel delta of 46, all within 4 px of a pixel the shadow
changes and none in the shadow's interior, which is why `--tolerance=48` inside the
1% budget is the documented setting. At 512x512 the same comparison is 761 of
262144 pixels with the same worst delta of 46, and `--tolerance=48` absorbs it
there too.

## The disk tier

T4/out-of-core exists to render a frame the machine cannot hold, and there are two
mechanisms in it: the static-cascade cache and banded streaming.

**The cache is keyed on what the host declared.** A buffer is created `STATIC` or
`DYNAMIC`, `ffi/src/handle.rs::is_static_geometry` reads the bits a draw was
submitted with, and the frame's draws carry the answer into the backend. A static
declaration is what puts the cascade plan in the arena and serves it back;
`RECONL_BUFFER_DYNAMIC` (and a host that declares nothing) keeps the dynamic path,
which never touches the disk. `ffi/tests/tiers.rs` pins the whole route through the
ABI (`the_static_declaration_is_what_the_cascade_cache_is_keyed_on`): one scene,
three legs, cold `hits 0 misses 1 frozen 1`, warm `hits 1 misses 0 frozen 1` off
262168 bytes of disk cache (map 256x256, tier 4), and the same scene declared
dynamic `hits 0 misses 0` with nothing on disk.

**A band is the frame's rows, moved through the arena.** `FrameStream`
(`backends/soft-cpu/src/lib.rs`) cuts the frame into bands of the tallest height the
RAM cap admits, keeps one band target resident, and gives each band its own arena
key derived from the frame's serial, so a band can never be read back as another
frame's rows. The band renders the frame's rows through `band_projection`, present
reassembles the bands in row order, and the colour checksum is chained band by band
so a streamed frame's fingerprint is the number a resident frame's would be.

Measured on the shipped bench, 512x512 at T4 with `--shadows=cached`, `--spill=1`
and a 2 MiB RAM cap: `shadow cache hits 1 misses 1 read 256.0 KiB hit 256.0 KiB
frozen cascades 3`, `spill 5.3 MiB of 64.0 MiB`, `presented 4 failures 0`, and
`arena.rcls` is 26,477,080 bytes on disk. The same frame with a cap that holds it
reports `spill 256.0 KiB` — the cascade and nothing else — and a 262,200-byte arena.
Streaming costs what it costs: 27.4 ms a frame against 12.9 ms for the resident
frame. A cap too tight for even one band is still refused — 1 MiB at 512x512 fails
`reconlCreateSwapchain: RECONL_ERR_BUDGET_EXCEEDED (-5)`.

**The cap a host declares is the cap the arena keeps.** `disk_cap_bytes` reaches the
arena as its hard cap (`ffi/src/lib.rs::config_from_desc` sets `SoftCpuConfig::arena_bytes`,
which becomes `SpillConfig::max_bytes`), the arena counts its own file header against
that cap, and a write that will not fit makes room by eviction and compaction or fails
with `BudgetExceeded` naming what it needed - so a host that declares 2 MiB gets at
most 2 MiB, and the tightest case measured, a 512x512 streamed frame under a 6 MiB cap,
reports `spill 5.0 MiB of 6.0 MiB`. A streamed frame reserves its whole disk footprint
before the first band is written, which is what keeps its own eviction off its own
bands. The ABI reads a cap of 0 as *no disk use at all*, so the tools name a budget
whenever they are asked to spill (`tools/host/src/device.rs::resolve_disk_budget`, 1 GiB
by default), and the probe's `disk-spill`/`out-of-core` caps follow the spill
directory's writability rather than a free-space guess (`ffi/src/lib.rs::spill_dir_usable`).

Two things about it are worth knowing before reading the code:

* **A streamed frame is not byte-identical to a resident one.** Same tier, same
  shadow mode, 512x512: 2 of 262144 pixels differ, worst channel delta 46, at a
  shadow edge — `reconl-diff` names them at (372,304) and (197,387), and
  `--tolerance=48` passes them. With `--shadows=off` the same pair is
  byte-identical and both settings are byte-identical run to run, so what a band
  changes is a shadow-edge pixel rather than the frame: a band reaches the
  rasteriser's 1/256-px vertex snap through a re-projected clip space, and the
  pinned pixels move with the band height. It is the same class as the cross-tier
  difference and smaller than it. Making it byte-exact would need the rasteriser to
  crop by an integer row origin instead of re-projecting; `band_projection` says so
  where it describes the cut.
* **It is the one path that allocates inside a frame.** A streamed frame reports 3
  allocations of 6144 bytes over 4 measured frames (0.75 per frame) where a resident
  frame reports 0, so `docs/determinism.md`'s zero-allocation rule is met
  everywhere but here.

## Frame generation

A host that can render 60 frames a second and display 120 asks the device for the
frames in between instead of rendering them. The device reprojects the newest frame
it was handed along the camera motion the host declared — no shaders, no second
render, nothing about how a frame is drawn changes — and the host decides *per
frame* whether it wants one, which is what makes it a toggle a game flips with its
own quality settings rather than a mode the device is put into.

- **`raster/src/framegen.rs`** — the warp, and nothing else knows how to do it:
  `Camera` (the view, plus the projection rebuilt from the same fov/near/far the ABI
  declares, so the two cannot drift), `reprojection` (one matrix per generated
  frame, `P_prev · V_prev · V_cur⁻¹ · P_cur⁻¹`, applied to `(ndc_x, ndc_y, depth,
  1)`), and `generate` (one backward warp, bilinear, clamped at the frame's edge).
  A still camera copies the frame it was given, to the byte, which is why a menu, a
  paused game and the reference scene are still scenes. `check_ahead` is the
  look-ahead interval, and both the ABI's argument gate and `generate` call it, so
  the interval a host is held to is one function rather than two copies.
- **`ffi/src/lib.rs::FrameGen`** — the retained history: the pixels the *host* was
  handed, the depth behind them, and the camera pair that is the motion. Owned by
  the device rather than a backend, because what a generated frame is warped from is
  the image the host saw, and that boundary is the FFI's. `reconlPresent` fills it
  only for a frame that asked; `reconlPresentGenerated` warps it out.
- **`include/reconl/reconl.h`** — `ReconLFrameGenDesc` on `ReconLFrameDesc`
  (trailing, so an older caller's struct is read as if it were NULL),
  `reconlPresentGenerated`, and the contract: the order its answers are decided in
  (tier, then whether there is anything to generate from, then the descriptor and
  `ahead` — see below), `ahead` in `(0, 1]`, `RECONL_ERR_NO_FRAME` when the last
  presented frame did not ask, `RECONL_ERR_NOT_SUPPORTED` on a tier with no depth to
  reproject, and the rule that a generated image never counts in `frames_presented`
  — the number the tier ladder judges — so a host cannot make a slow tier look fast
  by generating more images from its frames.
- **`backends/{soft-cpu,d3d11}`** — `depth_into`, the one read a present does not
  otherwise make, so a host that never asks pays neither the depth readback nor the
  three buffers.

What decides a generated call's answer is not the frame state machine (a generated
frame is not a frame: it answers identically in `open` and `submitted`) but whether
a frame is being kept: with none, every descriptor and `ahead` answer
`RECONL_ERR_NO_FRAME`; with one, the arguments are read; on a tier that keeps no
depth every cell is `RECONL_ERR_NOT_SUPPORTED`. `reconlPresent` has the same shape
with the frame state as its gate. The header states that order rather than leaving
it to be inferred, and `ffi/tests/abi.rs` pins it on both paths.

The depth convention is the one defect this feature found: the reference tier's
perspective depth buffer held *clip z* while the hardware tier's holds the
documented reversed-Z NDC depth, so the same readback meant two different things and
the warp was wrong on the CPU tier. The rasteriser now stores the NDC depth its own
`Target` documents (one less divide per pixel in the hot loop), the shadow passes are
unaffected because an orthographic matrix has `w == 1`, and the golden is unchanged
by it.

Measured with `reconl-bench --resolution=R --frames=60 --warmup=5 --framegen=R` on
the reference scene, whose camera does not move — so these are cost measurements,
not quality ones, and single runs rather than averages of runs:

| backend | resolution | asked | achieved | cost per generated image |
|---|---|---|---|---|
| soft-cpu | 512×512 | 2.0 | 1.96 | 2.29% of a rendered frame |
| soft-cpu | 512×512 | 3.5 | 3.39 | 1.25% |
| soft-cpu | 1080p | 3.5 | 3.30 | 2.37% |
| d3d11 | 512×512 | 2.0 | 1.83 | 9.43% |
| d3d11 | 1080p | 3.5 | 2.57 | 14.43% |

The hardware tier's multiplier is lower because its frames are nearly free and its
present path is not: at 1080p a rendered frame is 19.1 ms and a generated image
2.75 ms, most of that the host readback, while the reference tier generates a 2.66 ms
image inside a 112 ms frame. The ceiling on the ratio is the game's own
ratio of what a frame costs to what a present costs, which is why this is a toggle
and not a default. `reconl-bench` prints the *achieved* rate on the wall line
("per rendered frame: N rendered frames/s, M images/s presented in all") so a run
that delivered twice the images cannot read as a run that got slower.

Pinned by `ffi/tests/abi.rs`: the toggle per frame, the still-camera identity, the
motion and the prediction (the stripe lands where the camera's own motion says it
must, and the generated image is closer to a real render of the next camera than the
frame it was warped from, on both tiers), determinism and ladder isolation (a
generated frame counts only itself), every refusal plus that a refused generated
frame does not wedge the device, the null tier's `NOT_SUPPORTED`, and the
reservation a host that never asks does not pay (counted through the host's own
allocator). Through the shipped DLL, the `framegen` probe drives the toggle, the
refusals and the counters as a C host does.

## The tier ladder

The ladder had three writers once. It has one rule, one decider and one record:

* **One rule** — `core::tier::FrameLadder` holds the target, the threshold and the
  run. Both layers count the same run, once per presented frame, from the composed
  frame cost the device owns.
* **One decider** — `ffi/src/offload.rs::apply_tier_policy`. The relabel is decided
  first, from the run as it stood before this frame's cost, and applied to the
  backend that is live when the frame closes; the offload is decided next, from this
  frame's own observation, and is the only layer that changes the backend.
  `desc.downgrade_after_frames` has exactly one reader — the device's ladder.
* **One record** — `DeviceHandle::downgrades`, written only by `note_tier_change`.
  Offloads, returns, relabels and host tier requests land there in order, and
  `ReconLStats.downgrade_count` is the total ever, not just what fits in the ring.
* **Backends only apply** — `relabel(to, reason)` changes the tier and what the
  tier invalidates (shadow layout, the tier clock) and nothing else.

The schedule is part of the rule: a relabel lands on the frame *after* the cost that
armed it, the offload on the frame it just closed. Pinned in `core/src/tier.rs`,
`ffi/tests/tiers.rs` (the log holds every change however the backend moved, the
relabel is charged to the frame after the cost, a host that forbids tier changes
keeps its hardware) and
`backends/soft-cpu/tests/render.rs::the_reference_backend_does_not_decide_its_own_tier`.

**The offload's return trip needs a frame-time target.** The device comes back to
the hardware after `downgrade_after_frames` frames inside `target_frame_ms`, once
per resolution and shadow plan. A host that sets `target_frame_ms = 0` has no window
to settle in, so an offloaded device stays on the reference tier until it is
destroyed — which is the benchmark's own default.

The ladder's two halves are decided in one place but only one is reachable at a
time. *One return per plan* is pinned at 256×256, where the reference tier loses the
calibration and the device goes back to the hardware; *one calibration per plan*
needs the opposite branch and is pinned at 32×32, because at 64×64 and above the two
tiers' frame costs overlap in a way that leaves no whole millisecond between "the
reference tier meets it" and "a rebuilt hardware device is over it". Both legs share
one harness and one derivation — the target is a measurement, so a second copy of the
derivation would be a second opinion about the same number. The derivation uses the
reference tier's *cheapest* warm frame (what the settle window is armed against does
not drift over a millisecond) and raises the ABI's smallest whole millisecond only
when a fresh hardware device's first frame is over twice it; a host that cannot
produce the sequence fails loudly with both measurements instead of passing quietly.

## Render passes and the viewport

`ReconLRenderPassDesc.viewport_width`/`viewport_height` are in *frame* pixels, and
`raster::rendered_viewport` is the one rule that resolves them against the target a
tier actually renders into, so a tier at half resolution (T3/T4) confines the same
fraction of the window. `0` in an axis means the whole target, an oversized viewport
is clamped rather than refused, and the result is never zero in either axis. The
render area is part of the target the tile loop works on — triangle bounds clamp to
it and each tile's rect is intersected with it — so both the binning and the pixel
loop are confined and tiles still never overlap. The shadow passes ignore it and draw
their whole map.

Pinned by
`ffi/tests/abi.rs::a_pass_viewport_confines_the_render_to_the_rect_it_asked_for`
(four legs: the reference tier at T2 and at T3/T4, where the target is half the
frame, plus the hardware tier, asserting the exact lit-pixel count and bounding box
and the unchanged full-frame default), with each half revert-checked, and
`raster::rendered_viewport` has its own unit test for the rounding and clamping rule.

## Frame state, refusals and the answer order

`include/reconl/reconl.h` asserts that the order its answers are decided in is fixed,
so a host reads one code per call. That order has one home, `ffi/src/order.rs`: a
`Step` table per entry point (`BEGIN_FRAME`, `SUBMIT`, `PRESENT`,
`PRESENT_GENERATED`), a `Call` impl whose `ask` answers one named question, and
`walk` asking them in the table's order and stopping at the first refusal. Each entry
point is "build the call, walk its table, take the parts".

* **The precedence is a value.** `SUBMIT` lists `List` before `State`; `PRESENT`
  lists `State` before `Commit` before `Arguments`. Both asymmetries are documented
  in the header and both are visible side by side in the tables.
* **The commit point is a step.** `Step::Commit` is where a call takes the frame
  (`Present` and `Submit` have one, `PresentGenerated` and `BeginFrame` do not).
  Everything before it leaves the frame exactly as it was; everything after it ends
  the frame and counts in `frames_dropped`.
* **"Nothing dereferences a host pointer before its gate" is structural.** A
  question that does not involve the descriptor answers before the descriptor is
  read, so a NULL — or an unmapped — pointer in a state whose gate comes first
  returns a code. Swapping the generated path's `History`/`Arguments` steps makes the
  pin fault with `STATUS_ACCESS_VIOLATION` instead of failing an assertion, which is
  what makes that cell evidence rather than a restatement.

## The ABI

`include/reconl/reconl.h` is the contract: opaque ref-counted handles, versioned
structs, no exceptions, no panics across the boundary, and a host allocator the
library never bypasses.

**The layout claim is compiled, not asserted in prose.**
`ffi/tests/abi_layout.rs` walks one table per struct — the Rust mirror's fields and
types — and writes a C file of `_Static_assert`s: one `sizeof` per struct, an
`offsetof` and a field `sizeof` per field, and a `_Generic` check that each field's
category is the one the Rust side declares. It compiles that file with
`cc`/`gcc`/`clang` against the shipped header: 1118 asserts in all, left in
`CARGO_TARGET_TMPDIR` with its path in the failure message. It skips with a printed
reason where no C compiler exists, the way the d3d11 legs do.

**The numeric contract is gated the same way.** A second table asserts every macro
and enumerator the header exports against the Rust value the implementation uses —
result codes against both the `Code` enum and `abi::result`, struct types, backends,
caps (against `abi::caps` and `core::tier::caps`), downgrade flags, shading, tiers
and tier reasons, shadow filters and events, log levels, formats, usages, index
format and frame state, and the raster pipeline enums, which is where a renumbering
would change an image. `RECONL_COMMAND_BYTES` stays with its owner: `ffi/src/sizing.rs`
compares it to the header line in its own unit test.

**D3D11 adapter selection is part of ABI 101.** `reconlEnumerateAdapters`
returns DXGI adapters, D3D11 feature-level availability, DXCore integrated-vs-
discrete classification where supported, and an adapter LUID for exact selection.
The D3D11 descriptor retains its original prefix and appends a size-gated
preference/LUID selector. `auto` prefers a discrete GPU, then falls back to any
usable adapter; `integrated` prefers an iGPU; LUID selection does not silently
substitute a different GPU. DXCore is loaded dynamically so older Windows builds
can still enumerate D3D11 adapters with type `UNKNOWN`. SoftCPU and Null descriptor
pointers remain reserved and are accepted without being read. The C ABI tests pin
enumeration, descriptor validation, LUID selection and the reserved-pointer case.

The public host tools list adapter names/types/LUIDs in `reconl-info` and accept
`--adapter=auto|integrated|discrete|INDEX|luid:HEX` in both `reconl-info` and
`reconl-bench`. Adapter selection is copied into the device's config so a later
hardware rebuild after offload returns to the same physical adapter.

**Driver errors are classified in one place.** `classify_hresult` in
`backends/d3d11/src/imp.rs` maps the DXGI removal family to `DEVICE_LOST`,
`E_OUTOFMEMORY` to `OUT_OF_MEMORY`, `E_INVALIDARG` to `INVALID_ARGUMENT`,
`E_NOTIMPL`/`E_NOINTERFACE` to `NOT_SUPPORTED`, and anything unrecognised to
`BACKEND_UNAVAILABLE` — all existing codes, nothing invented. The per-class host
guidance is documented next to the `ReconLResult` enum, and the d3d11 backend has no
GPU timestamp queries: its stage numbers are CPU submit times, which the null tier's
numbers are too, so the ladder compares like with like.

## A steady-state frame allocates nothing

`docs/determinism.md` makes zero allocations inside a frame a rule, and the rule now
holds: 60 measured frames of the 64x64 reference scene report **0 allocator calls per
frame** at T2 and 0 on `d3d11`, and 0 at 256x256, 512x512 and on the null tier. The
one exception is the disk tier's streaming path, measured above.

Storage is reserved between frames and then cleared rather than reallocated, and
every list a frame writes into has one owner that outlives the frame:

* **The frame's draws** are `FrameRecord::items`, written in the command loop where
  the bound checks live, kept across frames by `FrameRecord::reset`, and read by a
  backend through `frame_input`. The entries point into the *host's* buffers — the
  host's reference count is what keeps those bytes alive while the frame is in use —
  and `FrameRecord::items_grown` counts the growth that is the one allocation a
  steady state does not have.
* **The reference tier's lists** are `SoftCpuDevice::cascade` and
  `SoftCpuDevice::colors`, filled by `fill_cascade` / `fill_colors`, both cleared at
  the start of the next fill. The cascade map view is a fixed `[ShadowMapRef;
  MAX_CASCADES]` on the stack, and the per-cascade caster filter is an `any()` over
  the frame's draws rather than two gathered lists.
* **The counters agree with the ledger**: `FrameNumbers::allocations_in_frame` counts
  only real growth, so a normal frame reports zero rather than a number nobody can
  check.

Pinned by `tools/reconl-bench/tests/cli.rs::a_steady_state_frame_allocates_nothing`
(both tiers, the d3d11 leg skipping with a printed reason where no device exists) and
`raster/tests/render.rs` for the rasteriser's own tables. Storage that is cleared and
refilled has one failure mode worth its own pin — a stale tail — so
`backends/soft-cpu/tests/render.rs` renders a floor-plus-caster frame and then a
floor-only one in the same device and requires the lean frame's checksum to equal a
fresh device's, and `ffi/tests/abi.rs` does the same one layer up (a two-quad scene
recorded as two draws, then as one, must present no white stripe) with the worlds
widened rather than copied. The fault path's re-render has its own pin in
`ffi/src/offload.rs::fault_tests`, which runs the fault's two steps in the order
`Submit`'s arm runs them against the real ABI. The honest limit: **the fault itself
cannot be injected** — there is no fault hook, a failover needs a genuine driver
verdict, and the `--repeat=200` removal the README records was a suspended device
instance on this machine, not a reproducible condition — so the fault's own verdict
is the one step those pins do not take.

## Probes

The C probes live in the repository (`probes/src/*.c`) with their runner
(`probes/run.sh`), because a behavioural claim that rests on a hand-copied library in
a temp directory is a claim with no provenance. The runner builds the release
library, copies the built DLL/SO into `probes/.build/`, **md5-confirms the copy
against the built file**, compiles every probe against `include/reconl/reconl.h` and
that copy, runs the plan (24 rows: probe × backend arguments), and prints one summary
with the library's md5. A row is red on a non-zero exit, a `FAIL`/`FIND` marker, a
check count that does not match the expected one, or a missing required line; each
row's output is kept in `probes/.build/<row>.out`.

Pinned expectations are the point of the table: `fghostile` 32, `framegen` 21,
`framestate` 43, `hostile` 11 / `hostile2` 13 / `hostile3` 11 / `hostile5` 16,
`ladder` 6, `narrowpitch` 11, `offload` 33, `onewriter` 8, `shadowconfig` 40, and a
required line for the four probes whose verdict is a field (`gpu_probe` `OK`,
`hostile4` `a full frame afterwards: 0 (recovered)`, `overtarget` `presented
12;failures 0`, `viewport` `lit pixels outside the requested 16x16 rect: 0`,
`fgstate` its five zero-markers). A probe that silently stops running its checks
cannot pass by printing nothing. `probes/README.md` says what each one is for.

The freed-handle lesson is worth keeping: `fghostile` once released a swapchain and
then passed it to `reconlPresentGenerated`, and that cell segfaulted about one run in
three, because asking for a defined answer about a pointer to freed memory asks the
library to read memory its host handed back. The cell now passes a live handle of the
wrong kind, which tests the property that is actually the ABI's — a handle is
validated by its kind word and never followed.

## The debug and fix pass

The tree went through one full verification pass before anything was pushed, on
the branch `reconl/debug-fix-phase`, as four commits:

* **`frontend: Zig immediate-mode UI core rendered through the ReconL C ABI`** —
  `frontend/` had never been committed; it went in with `.gitignore` rules for
  its caches and outputs (`.zig-cache`, `zig-out`, `*.exe`, `__pycache__`), and
  a compiled `spike/probe.exe` was kept out.
* **`ReconL: ABI 101 adapter selection, a hard-capped spill arena, and the
  band-streaming module`** — the working tree the sections above describe:
  adapter selection end to end, the arena's hard cap, `filter_caps`, the move of
  the out-of-core path into `backends/soft-cpu/src/stream.rs`, the 24-row probe
  plan, and the docs that go with them.
* **`probes: include reconl_backends.h in gpu_probe`** — the one *defect* the
  pass found. `gpu_probe` pins device creation to an enumerated adapter LUID
  through `ReconLD3D11Desc`, which lives in `reconl/reconl_backends.h`, but the
  probe only included `reconl/reconl.h`, so it did not compile against the
  shipped headers (`unknown type name`, and the `SETBASE` and selector fields
  failed with it). One include; all 24 rows compile and pass again.
* **`workspace: clear every compiler warning`** — 15 warnings to zero.
  `Command::SetVertexBuffer` no longer stores the stream index the entry gate
  already refuses to anything but 0; the descriptor test documents why its
  assignments are only read through the FFI pointer (`#[allow]` with the reason
  beside it, not a silent one); an unused `mut` in the sizing test, an unused
  `mut` in `SpillArena::open`, an unused test import in `shadow/` and
  unnecessary parens in `MipChain::bytes` go away; and `reconl-info`'s device
  report now *prints* the adapter preference, which is what the parameter
  threaded into `report_device` was for — `--adapter=` used to choose silently.
  The one piece of new code is `Display for AdapterSelection` in `tools/host`.

Every fix was verified by hash: the eight touched files were snapshotted,
reversed to isolate the carried-in tree for its own commit, re-applied, and
`sha256sum -c` confirmed them byte-identical to the state every test below ran
against.

### The build entry points

Four, and only four — each language's *native* workflow, not a wrapper over
another one:

| entry point | command | what it gates |
|---|---|---|
| Cargo workspace | `cargo build`; `cargo test --workspace`; the same with `--profile tested` | unit, ABI host, tier matrix, golden, CLI |
| Zig frontend | `cd frontend && zig build test` / `abi-test` / `shared` / `demo` | 42 core tests with no renderer linked; 11 through the C ABI against the debug DLL |
| C probes | `probes/run.sh` | the *shipped* release DLL, freshly built and md5-confirmed; 24 rows |
| Java spike | `cd frontend && python spike/verify_java.py` | the Panama FFM consumer against `reconl_ui.dll` (`javac --enable-preview --release 21`, because FFM is still a preview API on 21) |

The rest of the multi-language surface is deliberately *not* an entry point.
There is *no Gradle build and no Kotlin source* in the repository — the Java
spike is one file compiled by `javac` directly. `cmake` is installed but owns
nothing here: no `CMakeLists.txt` exists anywhere. And there are no C++
sources: the C probes compile through the runner's own `cc` line (`gcc`/`cc`
serves; `clang` is absent). `cargo check --workspace --all-targets` is the fast gate and currently answers
0 errors, 0 warnings.

### What comes next

* **The 60 FPS gate.** Automated benchmarking hooks that run the reference
  scene at 720p, 1080p, 1440p and 4K and record min/avg/max per configuration,
  so "60 FPS" becomes a number the harness reads rather than a claim. The
  baseline is the README's scaling table: d3d11 is at 2.30 ms for 512x512 and
  its per-frame cost is dominated by a fixed readback, while soft-cpu is at
  15.3 ms there and must be measured at each resolution before anything is
  promised. The hooks belong beside `reconl-bench`'s existing fingerprint and
  trace, and the ladder's `target_frame_ms` is where their verdict lands.
* **From this file's own honest limits:** the fault path still cannot be
  injected (no fault hook, so the offload's device-removal leg remains the one
  step no pin takes), and a streamed frame is not byte-identical to a resident
  one (an integer row origin instead of a re-projected clip space would make it
  so).
* **From the README's provisional list:** the backend matrix beyond
  `d3d11`/`soft-cpu`/`null`, no GPU timestamp queries, the second copy of the
  reference scene in `ffi/tests/tiers.rs`, the offload return trip's dependence
  on a non-zero `target_frame_ms` (the benchmark's default is 0), and the
  frontend's script shaping (no GSUB/GPOS, no kerning pairs).

## The 60 FPS gate

`reconl-bench --fps-gate` is the benchmarking hook that turns "does this machine
hold 60 FPS?" into a number, a table and an exit code. It sweeps the four
resolutions a host means by 720p, 1080p, 1440p and 4K - 1280x720, 1920x1080,
2560x1440, 3840x2160 - on one device, through the same warmup-then-measure
discipline every bench run uses (5 unmeasured frames, then 60 measured), and
judges each resolution's *mean* frame time against the budget.

* **The budget is an integer.** `--fps-target=N` (default 60) becomes
  `1_000_000_000 / N` nanoseconds - 16,666,666 ns at 60 - so the verdict is an
  integer comparison and the same target always means the same number. The
  *gated* statistic decides - the median of three repeated trials, each the
  median of its frames (`--fps-stat` / `--fps-trials`, see "The gate's
  measurement policy" in the 4K defect section below for why a mean could not).
  The trials and min/max are printed beside it because a number without its
  spread is a peak wearing a costume.
* **Failure is loud.** Any resolution over budget prints a FAIL row, the summary
  names every offender with its mean, its fps and the amount it was over by, and
  the process exits 3 (0 pass, 2 error, 3 budget exceeded) - a CI step cannot
  read a passing exit code off a failing machine.
* **The sweep is owned.** `--width`, `--height`, `--resolution`, `--framegen`,
  `--trace` and `--png` are refused alongside `--fps-gate` rather than silently
  ignored: an option that would quietly measure something else is a measurement
  mislabelled.
* **Headless by construction.** `--backend=soft-cpu` runs the whole gate with no
  GPU at all (the same reference path the offload ladder falls back to), so the
  gate exists on a machine with no display as well as on one with a GPU.

How to run it:

```bash
cargo build --release -p reconl-bench
target/release/reconl-bench --fps-gate --backend=soft-cpu   # headless: no GPU needed
target/release/reconl-bench --fps-gate --backend=d3d11      # the hardware tier
target/release/reconl-bench --fps-gate --fps-stat=best      # the ceiling, not the verdict
cargo test -p reconl-bench                                   # the gate's contract, on the null tier
```

Measured on this machine (Windows, Intel UHD Graphics, 4 worker threads, single
runs of the reference scene with shadows on, release build), 5 + 60 frames per
resolution:

| resolution | soft-cpu (T2) min / avg / max | fps | verdict | d3d11 (T1) min / avg / max | fps | verdict |
|---|---|---|---|---|---|---|
| 720p 1280x720 | 97.918 / 123.725 / 265.161 ms | 8.1 | FAIL | 4.695 / 9.912 / 20.628 ms | 100.9 | PASS |
| 1080p 1920x1080 | 223.987 / 255.781 / 317.991 ms | 3.9 | FAIL | 10.121 / 17.136 / 23.259 ms | 58.4 | FAIL (+469.6 us) |
| 1440p 2560x1440 | 400.376 / 452.623 / 780.300 ms | 2.2 | FAIL | 17.151 / 25.842 / 32.227 ms | 38.7 | FAIL (+9.176 ms) |
| 4K 3840x2160 | 870.654 / 975.962 / 1576 ms | 1.0 | FAIL | 38.803 / 45.565 / 54.899 ms | 21.9 | FAIL (+28.899 ms) |

Both runs exited 3. The honest reading:

* **60 FPS on this machine is a hardware-tier claim at 720p only.** The d3d11
  tier holds it at 1280x720 with half the budget to spare and misses 1080p by
  469.6 us - a margin small enough to flip between runs, reported as measured
  rather than rounded into a pass. 1440p and 4K are over by whole milliseconds.
  The reference tier cannot hold the budget at any of the four, which is what
  the design says: T2 defines correctness, not frame rate.
* **What the number is:** host wall time across the ABI's three boundaries
  (begin, submit, present) - the cost of *producing* a frame into memory. It
  does not include display scanout or vsync; this gate is headless, so there is
  no swap chain to wait on.
* **What cannot be measured here:** the hardware path has no GPU timestamp
  queries, so the gate cannot say *which stage* put a resolution over budget,
  only that the frame the host waited for was. And these are single runs on one
  machine: the min/avg/max spread in the table is the honest uncertainty, not a
  variance across runs.

Pinned by two unit tests in `tools/reconl-bench/src/main.rs` (the budget's
integer derivation, and the sweep being exactly 720p through 4K at the sizes
those names mean) and `tests/cli.rs::the_fps_gate_judges_every_resolution_and_fails_loudly`,
which drives both exit paths end to end on the null tier - a budget that fits
must exit 0, a one-nanosecond budget must exit 3 - so the gate's contract is
tested without making the suite's green depend on how fast this machine is.

## The present path, profiled - 1080p and 1440p inside the budget

The gate above said *where* the hardware tier stood; this pass asked *why*, on
a branch stacked on the gate's own (`reconl/d3d11-readback` atop
`reconl/fps-gate`) so the measuring tool is in the tree. The gate's own admission
framed the method: with no GPU timestamp queries, the stage-level question is
answered from the host side.

* **The host's split said where to look.** Across the frame's three ABI
  boundaries at 1080p, `present` was about 89% of the frame - submit 2.4 ms,
  begin 7.8 us. The miss was not per-draw CPU work, not allocation churn (a
  steady frame allocates nothing, above), not redundant state. It was
  `reconlPresent` - the readback the host waits on.
* **Inside the present, two legs** (temporary instrumentation in
  `copy_target_rows`, behind an env var, removed before the commit): one `Map`
  of the staging texture, then the row-copy loop out of it. At 1080p the
  map-wait ran 11-26 ms across frames - `Map` on a staging resource blocks until
  the GPU finishes the frame - and the loop 3-9 ms, streaming ~8.3 MB of
  uncached GPU memory at roughly 1.8 GB/s. Both legs have a constant floor and
  scale with bytes; at 4K the same instrumentation read 33-91 ms and 28-265 ms.

Two costs, two changes, both in `backends/d3d11/src/imp.rs`, neither touching
behaviour, the ABI, or a single pixel the reference tier defines:

1. **The copy is one span when the layout says it is.** `copy_target_rows`
   looped row by row even when there was no flip, the destination pitch was the
   row bytes, and the mapped `RowPitch` was too - the common case, since a width
   whose row is the driver's pitch alignment lands there. `copy_rows_contiguous`
   now takes that case as one `copy_nonoverlapping` of `rows * row_bytes`; a
   padded pitch or a flip still gets the per-row loop, unchanged. Measured
   alone, this change already passes: 1080p 9.834 ms, 1440p 14.977 ms.
2. **The frame's commands are flushed at the end of `render()`.** Without one,
   the driver decides when the queued commands reach the GPU, and the map inside
   present could be the first moment the frame's work really started - the tail
   the old path showed that way is visible below. The flush hands the GPU the
   whole frame while the CPU is still finishing submit; the host pays for the
   same GPU work either way, only the overlap changes.

Before and after, three runs each after the change (means, 5 + 60 frames per
resolution, `--fps-gate --backend=d3d11`):

| resolution | before (gate table above) | after: run 1 / run 2 / run 3 | verdict |
|---|---|---|---|
| 720p | 9.912 ms PASS | 7.026 / 6.168 / 7.635 ms | PASS x3 |
| 1080p | 17.136 ms FAIL (+469.6 us) | 10.078 / 10.415 / 10.082 ms | PASS x3 |
| 1440p | 25.842 ms FAIL (+9.176 ms) | 12.432 / 14.667 / 12.241 ms | PASS x3 |
| 4K | 45.565 ms FAIL (+28.899 ms) | 24.998 / 25.601 / 27.277 ms | FAIL x3 |

The honest reading, on a machine whose variance is wide enough that the old
binary measured 17.136 ms in the gate table and 18.359 ms in this session's own
baseline re-run:

* **1080p passes the 60 FPS gate, with room.** 10.08-10.42 ms against a
  16.667 ms budget - about a third spare, stable across runs, where the old path
  missed by 469.6 us. 1440p passes too, at 12.2-14.7 ms. The soft-cpu tier is
  untouched and remains what defines correctness: 277 workspace tests, the C
  probes (24/24 rows), and the byte-identical golden all stay green.
* **The tail is where the flush earns its place.** Interleaved runs of the old
  binary on this machine showed max frames of 58 ms at 1080p and 269 ms at 4K -
  whole frames spent inside the map-wait - while the runs above end at
  11.4/14.6/35.5 ms max, 13-30% over their own means. The mean improvement is
  the copy's; the flush is what stops an occasional frame from paying for the
  GPU's whole queue inside the present the host is timing.
* **4K stays over budget, and the data says it is the bytes.** The best 4K frame
  this build produced is 23.301 ms - 6.6 ms above budget even if everything else
  in the frame were free. The frame is a synchronous readback: 3840x2160x4 =
  33.2 MB of uncached GPU memory must cross to system memory inside the present
  the host waits on, and at the bandwidth the profiling loop showed (~1.8 GB/s)
  that transfer alone is most of the budget. No in-place copy optimisation
  closes a gap of that shape. Getting 4K under 60 means the readback stops being
  synchronous with the frame the host waits for - a staging ring whose copy
  overlaps the next frame's render, or a GPU-side downscale - which changes what
  present *means*, not how it copies. Not attempted here; the gate exits 3 with
  4K named, which is the honest verdict.
* **What the numbers are:** single machine, release build, one device, min/avg/max
  inside each run and the three runs side by side - the spread is the honest
  uncertainty, as above. The gate is unchanged: this pass moved the work, not
  the budget, the verdict rule or the exit codes.

Validation: `cargo test --workspace` 277 passed / 0 failed; `probes/run.sh` 24
rows, 24 pass, 0 fail, 0 skip; `reconl-diff compare tests/golden/soft-cpu-shadow.png`
identical (4096 px); no new compiler warnings - the five on this base are the
ones PR #2 removes, and none are in `d3d11`. The profiling instrumentation was
temporary and is gone from `backends/d3d11/src/imp.rs`.

## The 4K defect - what the frame is actually made of

The pass above left 4K over budget and said so. This one took the next defect off
the same list - put 4K inside the budget - and went looking for the stage that
put it there. What was found, and why it stops where it does, are both below.

* **A caution on cross-run numbers, which are the first thing to get wrong
  here.** This machine's run-to-run spread is roughly 15% at 4K *before* any
  change, and partway through the session the whole box degraded by 30-50% on an
  identical binary - 1440p went from 13.2 ms to 17-20 ms with no code change at
  all. So a number from this afternoon and a number from this morning are not
  comparable. Only *interleaved* A/B inside one run is evidence here, and the
  table below is the one that is labelled accordingly. Anything resting on a
  before/after comparison across hours on this machine is a comparison of
  machines, not of code.

**The stage split, from GPU timestamps.** `D3D11_QUERY_TIMESTAMP` around the
frame's four boundaries, read after the present's `Map` returned (so the stream
had drained), behind an env var, removed again before this was written. At 4K,
profiling build, healthy machine, with the LEVEL3 change below in the tree:

| stage | 4K |
|---|---|
| shadow pass | 0.045 ms |
| colour pass | 13.6 ms |
| - of which: clears | 0.096 ms |
| - of which: draw 0 | 12.9 ms |
| - of which: draw 1 | 0.55 ms |
| copy (driver DMA of the staging texture) | 4.6 ms |
| **GPU total** | **18.3 ms** |

The host side agreed: across the three ABI boundaries, `begin` ~3 us, `submit`
~0.9 ms, `present` ~26.9 ms - present is ~96% of the frame. Inside the present,
the `Map` wait covers the render tail plus the DMA (~13 + ~5.5 ms) and the host's
own copy out of the mapping runs a further 4-5.5 ms for 33.2 MB.

**The colour pass is the wall, and it is per-pixel.** Interleaved variants of the
shader, medians, 4K, draw 0:

| variant | draw 0 |
|---|---|
| as shipped | 12.9 ms |
| no shadow taps (PCF off) | 9.4 ms |
| loop body kept, loop work skipped | 8.0 ms |
| no lights at all | 2.5 ms |
| unlit (no light loop, no PCF) | 2.0 ms |

So of 12.9 ms: ~3.5 ms is the 9-tap PCF across two cascades, ~5.5-7.5 ms is the
*control flow* of the light loop - a dynamic `for (i = 0; i < 16; ++i) { if (i >=
count) break; }` that a compiler cannot unroll - and ~2 ms is everything else.
The loop's cost scales with pixels, not with lights: the reference scene has
**one** directional light (printed from the lights constant buffer to be sure),
and the loop still costs ~5.5 ms at 4K and ~3.3 ms at 1080p. The shader is
already at `ps_5_0`, and the scene is 3 triangles - this is pure fill rate on a
shared-memory Intel iGPU, 8.3 Mpixels of a full-screen-lit pixel shader.

### Two exact changes, kept

Both are semantics-preserving by construction - neither can change a pixel:

1. **Compile the shaders at optimisation level 3** (`backends/d3d11/src/imp.rs`,
   `compile()`). The default level left real time on the table: measured
   interleaved at 4K, the colour pass went 15.3 -> 13.6 ms. Optimisation cannot
   change what a program computes, and the frame stays byte-identical.
2. **Stop clamping the shadow uv in software** (`backends/d3d11/src/shaders.rs`,
   `tap()`). The comparison sampler's address mode is `D3D11_TEXTURE_ADDRESS_CLAMP`
   on U/V/W, so the hardware already resolves an out-of-range uv to the same edge
   texel the `clamp(uv, 0.0, 1.0)` was picking by hand. Same result, fewer
   instructions per tap - and there are 9 of them per lit pixel.

Both were then checked rather than argued. The d3d11 tier renders bit-
deterministically (same binary, two runs, byte-identical PNGs), so the honest
test is a differential one: render seven configurations - 64x64, 512x512 and
1280x720 with shadows on, plus 256x256 shadows=cached, shadows=off, framegen=2
and repeat=3 - against the committed binary and against this one, and compare.
All seven are byte-identical, and `reconl-diff` reports `PASS ... identical
(64x64, 4096 px)`.

The more interesting half is whether that comparison *could* have caught
anything. `lookup_at` already returns 1.0 the moment the cascade's uv leaves
[0, 1], so the software clamp could only ever fire on the *offset* taps in
`pcf`/`pcss` - a one-texel band at the cascade border - which is exactly the
kind of path a test quietly never reaches. Poisoning the clamp to
`clamp(uv, 0.0, 0.9)` and re-rendering is the falsification: it moves 173 of
4096 pixels at 64x64, worst channel delta 43, confined to the box
x 7..51 y 46..55. The clamp governs real pixels, so the A/B above is a real
test, and the hardware really is doing what the software was doing.

### What was tried and rejected, with the numbers

* **Banded staging** (the colour staging split into four horizontal bands, so one
   band's `Map` could return while the host copied another): the driver's first
   `Map` serialises on the *whole* immediate context. Band 0's map-wait was the
   entire GPU pipeline (16-20 ms) and bands 1-3 then mapped in ~100 us. No
   overlap to be had on this driver; reverted.
* **Manually unrolling the light loop** (16x `shade_light` + `pcf1`/`pcf2`):
   477 KB of bytecode, device init 60-150 s, and no frame-time gain at all
   (4K 27.2 ms). The driver is worse at a 477 KB shader, not better. Reverted.
* **Removing the `[loop]` attribute** so LEVEL3 could unroll it by itself - the
   natural version of the same idea, and the one thing not yet tried: it made
   things *worse*. 4K min 24.2 -> 28.9 ms, and the whole gate run took 88 s
   against ~20 s. Reverted.
* **MSAA**: `sample_desc()` is already `{Count: 1, Quality: 0}`. Not a knob.
* **A depth prepass**: the depth function is strict `GREATER` (reversed-Z), and a
  prepass would change what the strict comparison resolves for equal depths.
  Not exact, so not offered.
* **Fewer PCF taps**: this changes the image, and the d3d11 tier's whole job is
  to agree with the reference. Out of scope by the same rule that made change 2
  worth taking.

### Can the readback delete its copy? No - and the runtime says so

The 4.6 ms `CopyResource` from the colour target into a `D3D11_USAGE_STAGING`
texture is the obvious suspect: delete it and the GPU renders straight into
memory the host can map, the way D3D12 and Vulkan let a render target sit in a
host-visible heap. That is not available in D3D11, and the reason is a property
of the resource model rather than of this driver, so it is worth writing down
once. Measured on this machine (feature level 11_0) by walking every
description the readback would need and then every Map flag against the ones
that create:

| usage | bind | cpu access | creates? | mappable? | usable as RT? |
|---|---|---|---|---|---|
| `STAGING` | - | `READ` | yes | **yes** (READ) | **no** - `E_INVALIDARG` |
| `STAGING` | `RENDER_TARGET` | any | **no** `E_INVALIDARG` | - | - |
| `DEFAULT` | `RENDER_TARGET` | none | yes | no | yes |
| `DEFAULT` | `RENDER_TARGET` | `READ` | **yes** | **no** - `E_INVALIDARG` | yes |
| `DEFAULT` | `RENDER_TARGET` | `WRITE` | **yes** | **no** - `E_INVALIDARG` | yes |
| `DEFAULT` | none | `READ` | yes | no - `E_INVALIDARG` | - |
| `DYNAMIC` | any | any | **no** `E_INVALIDARG` | - | - |

* **The trap is the fourth row.** `D3D11_USAGE_DEFAULT` + `BIND_RENDER_TARGET` +
  `CPU_ACCESS_READ` *creates successfully* - the runtime accepts the description
  without complaint - and then rejects every `Map` on it with `E_INVALIDARG`:
  `READ`, `WRITE`, `WRITE_NO_OVERWRITE` and `WRITE_DISCARD` alike. A
  driver-side trap rather than a documented refusal, and exactly the kind of
  thing that costs an afternoon if nobody checked.
* **The only mappable texture in D3D11 is `STAGING` + `CPU_ACCESS_READ`, and
  staging cannot be a render target** (`E_INVALIDARG` on creation, the documented
  meaning of `D3D11_USAGE_STAGING`: copy source or destination only, never
  pipeline-bound). So the two halves the readback needs - GPU-writable and
  host-readable - have no description that is both. The copy is the bridge
  D3D11 provides and there is no route around it.
* **`DYNAMIC` does not open a door either.** It is rejected outright for a
  texture, and it is the usage the backend already uses for its mappable
  constant buffers (`imp.rs`), where it works - so the restriction is specific
  to textures, not to the flag.
* **There is no swapchain to fall back on.** `CreateDXGIFactory1` is used for
  adapter enumeration and nothing else: the backend has no
  `CreateSwapChainForHwnd`, no `IDXGISwapChain` and no `Present()`. It renders
  offscreen and copies into a host buffer, which is what makes the gate headless
  and keeps scanout out of a measurement that is about frame cost. A
  `DXGI_USAGE_STAGING` swapchain back buffer is the nearest D3D11 thing to a
  mapped render target, and reaching for it would mean inventing a window the
  readback path does not have, adding a present to a benchmark that deliberately
  has none, and changing what the gate measures.

So the copy stays, it is not an implementation accident to be tidied away, and
4K's floor is the colour pass plus that copy: 13.6 + 4.6 = 18.3 ms of GPU time
against a 16.667 ms budget, before the host has read a byte.

### The blocker, stated as a number

4K cannot pass on this device. The GPU alone is 18.3 ms - colour 13.6 + DMA 4.6
- against a 16.667 ms budget, *before* the host has copied a single byte out of
the mapping. The frame is over budget with the readback made free and perfectly
overlapped. Closing it needs the 13.6 ms colour pass, and the only things that
reach that far are giving up shading the image (unlit measures 2.0 ms) or not
rendering 4K (a GPU-side downscale). Both are changes to *what present means*,
not to how it copies, and neither is a performance fix.

Scaling is the other half of it: 4K is 4.0x 1080p's pixels, and the pass is
purely per-pixel, so 4K is ~4x 1080p's shading cost by construction. 1080p at
~3.4 ms of the same work fits the budget with room; 4K does not. This is a
hardware-tier ceiling, and the gate is correctly reporting it.

### The gate's measurement policy - making the hook reproducible

The caution at the top of this section was not decoration. The same binary, on the
same day, measured 4K at 24.8 ms and again at 34.7 ms, and 1440p came back PASS
on one run and FAIL on the next with no code in between. A verdict that changes
when nothing did is not a measurement, and until it stops, no 4K work can be
judged against it. So the fix was to the *hook*, in `tools/reconl-bench` only -
not one line of the render path was touched.

* **The gate now reduces twice, in two separate places.** Within a trial the
  statistic is the **median** of the measured frames, so a single stalled frame
  cannot move it: 59 frames at 10 ms and one at 500 ms average 18.167 ms (over
  budget) and median 10 ms (inside it). Across trials, `--fps-stat` picks how the
  trial medians collapse - `median` by default, which is what absorbs a machine
  that drifts part-way through a sweep. `--fps-stat=mean` reproduces the old
  policy and `--fps-stat=best` reports the honest ceiling, so the alternatives
  are reachable rather than argued about.
* **Repeated trials are also the warmup-outlier discard.** Each resolution is
  measured `--fps-trials` times (3 by default), each trial a full
  warmup-then-measure run on its own renderer. A caller who passes `--warmup=0`
  puts the cold frames in the *first* trial only, and the median across trials
  drops that trial rather than letting it set the verdict. Nothing is special-
  cased: the same reduction that ignores a stall ignores a cold start.
* **The table prints the statistic it gates on, and its evidence.** A `gated`
  column, the per-trial medians beside it, and `min`/`max` across every frame of
  every trial. The gap between `gated` and `max` is exactly what the old
  mean-based verdict was hostage to, and a row that shows its own spread is a
  row a reader can check instead of trust.
* **What it bought, measured.** Three back-to-back `--fps-gate --backend=d3d11`
  runs gave identical per-resolution verdicts: 720p PASS, 1080p PASS, 1440p PASS,
  4K FAIL, every time. 1440p gated 12.638 / 12.225 / 12.192 ms - 27% of headroom
  where the old mean sat on the budget at 16.4-17.5 ms and flipped.

An honest limit, since the number is easy to oversell: run back to back on a
*healthy* machine the old policy agrees with the new one (1440p 12.529 / 12.441 ms
old against 12.192-12.638 ms new), because on a healthy machine the outliers are
rare. What the change buys is reproducibility at the boundary and under
degradation, which is where the flipping lived. It does not and cannot make 4K
pass - and at 4K the trial-to-trial spread is itself the finding (24.8 ms, 29.8 ms
and 24.8 ms across three trials in one run), which the old single mean was
structurally unable to show.

Pinned by new unit tests in `tools/reconl-bench/src/main.rs`: the median ignores a
single catastrophic stall while the mean does not; the median is always a frame
that was really measured and is order-free; the three statistics collapse trials
as documented and read zero from an empty set; the trial column and the summary
say what they claim in milliseconds and in the right grammar.

**Verdict: not committed.** The two exact changes are real and worth keeping, but
the task was 4K inside the budget, 4K is not inside the budget, and committing a
partial fix while reporting it as the fix would be the dishonest outcome. The
tree is left with the two changes uncommitted and the gate exiting 3 with 4K
named. Two gate runs taken while the machine was *degraded* gave 720p 8.732 /
6.830 ms, 1080p 12.470 / 11.936 ms, 1440p 17.512 / 16.408 ms (FAIL / PASS - 1440p
straddled the budget), 4K 34.377 / 34.705 ms. A later run on a recovered machine
is the one the blocker above is stated from: 720p 6.635 / 8.616 ms, 1080p
9.243 / 9.845 ms, 1440p 11.301 / 11.955 ms - all three PASS with room - and 4K
at 23.634 / 24.843 ms, still 1.5x over. That spread (4K 24.8 ms healthy vs
34.7 ms degraded, same binary) is the machine, not the code, and is why the
interleaved splits above are the ones to trust.

Validation for this pass: `cargo check --workspace --all-targets` 0 errors,
0 warnings; `cargo test --workspace` 318 passed / 0 failed. All instrumentation
is gone from `backends/d3d11/src/imp.rs` and `shaders.rs`.

## Checks

```bash
probes/run.sh                                  # the C probes, against a freshly built and md5-confirmed DLL
cargo test --workspace                         # unit, ABI host, tier matrix, golden, CLI
cargo test --workspace --profile tested        # the same, at release optimisation
target/release/reconl-diff compare tests/golden/soft-cpu-shadow.png
target/release/reconl-bench --backend=d3d11 --resolution=512x512 --repeat=200
target/release/reconl-bench --backend=soft-cpu --resolution=512x512 --framegen=2
target/release/reconl-bench --backend=soft-cpu --tier=t4 --width=512 --height=512 \
    --frames=4 --warmup=1 --ram-cap=2MB --disk-cap=64MB --spill=1 \
    --spill-dir=spill --shadows=cached --png=streamed.png
target/release/reconl-bench --fps-gate --backend=soft-cpu   # the 60 FPS gate, headless
target/release/reconl-bench --fps-gate --backend=d3d11      # the same gate on the hardware tier
```

Last run on this machine (the debug and fix pass, `reconl/debug-fix-phase`):
`cargo check --workspace --all-targets` 0 errors, 0 warnings; `cargo test
--workspace` 302 passed / 0 failed, and the same 302 / 0 under `cargo test
--workspace --profile tested`; `probes/run.sh` 24 rows, 24 pass, 0 fail, 0 skip
against the freshly built, md5-confirmed DLL (`md5 dd172d3d…`); `zig build
test` 42/42 and `zig build abi-test` 11/11 against the current debug DLL, with
`zig build shared` and `zig build demo` succeeding; `python
spike/verify_java.py` all assertions passed; the golden byte-identical
(`reconl-diff compare` → identical, 4096 px).

**Do not run the suite with `--release`.** The release profile is `panic = "abort"`
(that is the crate's ABI contract), and `cargo test --release` forces `panic =
"unwind"` for test targets, so cargo builds the whole graph twice — an abort graph
for the bins, an unwind graph for the test binaries. Both emit `reconl.dll`,
`libreconl.a` and `libreconl.rlib` under the same names in `deps/`, which makes the
run fail every few tries with an unrelated tool reporting `can't find crate for
reconl_host`. `--profile tested` is release minus the abort, so a test run builds one
graph and cannot race itself.
