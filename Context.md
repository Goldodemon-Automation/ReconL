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
        d3d11/           the hardware tier — must match the reference
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
```

Last run on this machine: `cargo test --workspace` 279 passed / 0 failed across 35
binaries; `probes/run.sh` 24 rows, 24 pass, 0 fail, 0 skip against the freshly built
DLL (`md5 7fdb5fc0…`); the golden byte-identical (`sha256 7e8ecbab…`,
`reconl-diff compare` → identical, 4096 px).

**Do not run the suite with `--release`.** The release profile is `panic = "abort"`
(that is the crate's ABI contract), and `cargo test --release` forces `panic =
"unwind"` for test targets, so cargo builds the whole graph twice — an abort graph
for the bins, an unwind graph for the test binaries. Both emit `reconl.dll`,
`libreconl.a` and `libreconl.rlib` under the same names in `deps/`, which makes the
run fail every few tries with an unrelated tool reporting `can't find crate for
reconl_host`. `--profile tested` is release minus the abort, so a test run builds one
graph and cannot race itself.
