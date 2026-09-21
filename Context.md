# Context: how ReconL fits together, and where the last pass left it

ReconL is a deterministic renderer behind a stable C ABI: a software tier that
defines correctness, a D3D11 tier that must agree with it, and a ladder that
moves a device between them when the hardware dies or falls behind. The ABI is
the product; everything below it exists to make one call cheap, measurable and
repeatable.

This file is the map: which crate owns what, how one frame travels through
them, and what the most recent pass changed. The rules of the design live in
`docs/architecture.md` (ownership) and `docs/determinism.md` (what "the same
image" means); this file does not restate them, it orients you before you read
them.

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
  reference tier presents through.
- **`ffi/`** owns the *frame state machine*, the composed frame cost, and the
  offload policy, and it is split by concern rather than by layer: `handle`
  (the object model), `entry` (the boundary glue every call uses), `offload`
  (the tier policy and the backend rebuilds), `layout` (how rows move), `sizing`
  (what a descriptor costs) and `version` (names and log controls), with the
  device's state and the entry points over it in `lib.rs`. Backends answer one
  call at a time; they never hold frame state, and they never move themselves
  between tiers.
- **`tools/host`** is a shared host, not a library: allocator ledger, device
  wrapper, the reference scene, and the encode/submit/present order. Each tool
  (`reconl-diff`, `reconl-bench`, `reconl-info`) is a host that drives the same
  C ABI an application does.

## One frame, end to end

1. `reconlBeginFrame` opens the frame: size, lights, shadow request, camera.
2. `reconlCmd*` records a command list; `reconlSubmit` validates, prices and
   translates it into `FrameInput` — the one description of a frame — and hands
   it to whichever backend owns the device.
3. The backend runs the shadow passes, then the colour pass, into its targets.
4. `reconlPresent` reads the frame out into the host's buffer, in the host's
   layout, and is where the frame's cost is completed: the readback the host
   waited for is part of the number the tier ladder judges.
5. If the frame faulted or was measured over target, the ladder decides, in
   `ffi` alone: a genuine device removal or a measured overload offloads to the
   reference tier and the frame is re-rendered there; a rejected argument is an
   argument error and never a tier change.

## The golden

`tests/golden/soft-cpu-shadow.png` is the reference scene, rendered by the
reference tier, committed. Two independent paths check it: `reconl-diff compare`
against the file, and `reconl-bench --png` compared to the same file in
`tools/reconl-bench/tests/cli.rs`. It is byte-identical unless a change to the
reference tier's pixels is intended; every pass is expected to say which it is.

## What this pass did — the probes have a home in the repo

Every behavioural claim in this file rested on C probes and a hand-copied DLL
living in `%TEMP%\rlprobe`. That scratch directory had already produced one false
conclusion (a stale pre-pass DLL read as current) and was the standing cost of
every pass, so the evidence now lives with the code:

* `probes/src/*.c` — the 17 probe sources, tracked.
* `probes/run.sh` — one runner: builds the release library, copies the built
  DLL/SO into `probes/.build/`, **md5-confirms the copy against the built file**,
  compiles every probe against `include/reconl/reconl.h` and that copy, runs the
  plan (24 rows: probe × backend arguments), and prints one summary with the
  library's md5. A row is red on a non-zero exit, a `FAIL`/`FIND` marker, a check
  count that does not match the expected one, or a missing required line; each
  row's output is kept in `probes/.build/<row>.out`.
* `probes/README.md` — what each probe is for, and the notes below.
* `.gitignore` — `probes/.build/`.

Pinned expectations are the point of the table: `fghostile` 32, `framegen` 21,
`framestate` 43, `hostile` 11 / `hostile2` 13 / `hostile3` 11 / `hostile5` 16,
`ladder` 6, `narrowpitch` 11, `offload` 33, `onewriter` 8, `shadowconfig` 40, and
a required line for the four probes whose verdict is a field (`gpu_probe` `OK`,
`hostile4` `a full frame afterwards: 0 (recovered)`, `overtarget`
`presented 12;failures 0`, `viewport` `lit pixels outside the requested 16x16
rect: 0`, `fgstate` its five zero-markers). A probe that silently stops running
its checks cannot pass by printing nothing.

**The first thing the runner did was find a defect — in a probe.** `fghostile`'s
section 6 passed a swapchain to `reconlPresentGenerated` *after* releasing it and
asserted the code it got back. Run repeatedly, that cell segfaulted roughly one
run in three on d3d11 (`gdb`: `SIGSEGV` in `reconl::handle::check_handle`, called
from `reconlPresentGenerated` → `order::PresentGenerated::ask`), because the
library must read the handle's header and the host allocator is free to return
the page. The ABI's contract is ref-counted lifetime; asking for a defined answer
for a pointer to freed memory asks the library to read memory its host handed
back, and the recorded `fghostile 32/0` was a lucky run. The cell now passes a
live handle of the wrong kind (a command list where a swapchain belongs), which
tests the same property — the handle is validated by its kind word and never
followed — and is green 10/10 on both tiers, still at 32 checks. No library code
changed for it: the live wrong-kind handle was already refused with
`RECONL_ERR_INVALID_HANDLE`.

Nothing else moved: no exported function, no behaviour, no golden byte, no test.
`probes/.build/` is output and is ignored; the scratch directory is not part of
the project and nothing in the repo reads it.

## Previous pass — one owner for the answer order

`include/reconl/reconl.h` asserts that "the order its answers are decided in is
fixed, so a host reads one code per call". The code held that nowhere: four entry
points each inlined their own gate sequence, three of them in the same shape and
`reconlSubmit`'s inverted relative to `reconlPresent`'s - and the *order* was an
artifact of the sequence the checks happened to be written in, so a future edit
could reorder one call's answers without noticing.

The order now has one home, `ffi/src/order.rs`: a `Step` table per entry point
(`BEGIN_FRAME`, `SUBMIT`, `PRESENT`, `PRESENT_GENERATED`), a `Call` impl whose
`ask` answers one named question, and `walk` asking them in the table's order and
stopping at the first refusal. Each entry point is now "build the call, walk its
table, take the parts" - it asks nothing in an order of its own.

What the tables carry, and why they are not just a tidier spelling of the same
`if`s:

* **The precedence is a value.** `SUBMIT` lists `List` before `State`;
  `PRESENT` lists `State` before `Commit` before `Arguments`. Both asymmetries
  are documented in the header and both are now visible side by side instead of
  two screens apart. Nothing about the codes a host reads changed: every export
  returns exactly what it returned before, verified against the pre-pass entry
  points and the `fgstate` matrix.
* **The commit point is a step.** `Step::Commit` is where a call takes the frame
  (`Present` and `Submit` have one, `PresentGenerated` and `BeginFrame` do not,
  for the reasons their own tables state). Everything before it leaves the frame
  exactly as it was; everything after it ends the frame and counts in
  `frames_dropped`. That boundary used to be a comment above a statement.
* **"Nothing dereferences a host pointer before its gate" is structural.** A
  question that does not involve the descriptor answers before the descriptor is
  read, so a NULL - or an unmapped - pointer in a state whose gate comes first
  returns a code. Measured: with the generated path's `History`/`Arguments`
  steps swapped, the new pin faults with `STATUS_ACCESS_VIOLATION` instead of
  failing an assertion, which is what makes that cell evidence rather than a
  restatement. Restored, it passes.

One related rule moved into the same home: the look-ahead interval is now
`raster::framegen::check_ahead`, called by both the ABI's argument gate and
`generate` itself, so the interval a host is held to is one function rather than
two copies in two crates.

Validation: `cargo test --workspace --profile tested` is 268 passed / 0 failed;
`tests/golden/soft-cpu-shadow.png` stays byte-identical (`sha256 7e8ecbab…`,
`reconl-diff compare` -> identical, 4096 px); the probes against the rebuilt DLL
(`md5 ea5212dc…`, `cmp`-verified after the copy) are all at baseline - `fghostile`
32/0, `framegen` 21/0, `framestate` 43/0, `ladder` 6/0, `narrowpitch` 11/0,
`onewriter` 8/0, `shadowconfig` 40/0, `offload` 33/0, `hostile` 11/0, `hostile2`
13/0, `hostile3` 11/0, `hostile5` 16/0, `viewport` 0 px outside, `gpu_probe` OK,
`overtarget` 0 failures, `fgstate` documented order in every cell with 0 crashes.

## What the last pass did — one owner for the tier ladder

The ladder had three writers. The ffi's policy decided backend changes and
recorded them, and *each backend* kept a second, identical device-local ladder -
its own threshold, its own `frames_over_target`, its own ring - that decided
tier relabels and recorded those. `ReconLStats.downgrades` was a concatenation of
the device's log and whichever backend happened to be live. Measured through the
shipped DLL before the change (probe `onewriter.c`, arm B): the reference tier's
relabel during the calibration frame was read by the host at frame 1, gone from
the ring at frame 2, and the `RECOVERY` entry that replaced it named
`from T4/out-of-core` - a tier no surviving entry had ever produced. That audit
premise about the *d3d11* half being dead was wrong, and the measurement says so
(arm A: with `allow_downgrade = 0` the d3d11 relabel fires, `T1 -> T2`, and
`safe_path_events` stays 0, so the hardware half was armed and recording - into a
ring that dies with it).

Now there is one decider and one record:

* **One rule** - `core::tier::FrameLadder` holds the target, the threshold and
  the run. Both layers count the same run, once per presented frame, from the
  composed frame cost the device owns.
* **One decider** - `ffi/src/offload.rs::apply_tier_policy`. The relabel is
  decided first, from the run as it stood before this frame's cost, and is applied
  to the backend that is live when the frame closes; the offload is decided next,
  from this frame's own observation, and is the only layer that changes the
  backend. `desc.downgrade_after_frames` has exactly one reader - the device's
  ladder - instead of the ffi, the soft-cpu config and the d3d11 config.
* **One record** - `DeviceHandle::downgrades`, written only by `note_tier_change`.
  Offloads, returns, relabels and host tier requests all land there in order, and
  `ReconLStats.downgrade_count` is `DowngradeLog::total()` - the header's "total
  ever, not just in the ring", which is what it now means.
* **Backends only apply** - `relabel(to, reason)` changes the tier and what the
  tier invalidates (shadow layout, the tier clock) and nothing else. The
  duplicates are deleted rather than moved: two ladder bodies, two threshold
  config fields, `frame_time_override_ns`, two rings, two accessors, the ffi's
  backend-tier sync block, `null_downgrades`, `record_downgrade`, and
  `FrameInput::host_frame_ns` (which existed only to hand the composed cost to
  the backends' ladders).

The schedule is the one the backends' own ladders kept, restated in one place
and pinned: a relabel lands on the frame *after* the cost that armed it, the
offload on the frame it just closed (`docs/offload.md`, "Two layers, one
number"; `ffi/tests/tiers.rs::the_ladder_charges_a_relabel_to_the_frame_after_the_cost`).

Pins: `core/src/tier.rs::FrameLadder` tests, `ffi/tests/tiers.rs::the_tier_log_holds_every_change_however_the_backend_moved`
(revert-checked: fails on `frame 1 read an entry that is gone by the last frame`
against the concatenated rings),
`::the_ladder_charges_a_relabel_to_the_frame_after_the_cost` (the schedule, revert-checked:
fails on `the first frame stepped a tier the run had not armed yet`),
`::a_host_that_forbids_tier_changes_keeps_its_hardware` (the opt-out still gets
the relabel, and the host's tier equals the tier the ring logged), and
`backends/soft-cpu/tests/render.rs::the_reference_backend_does_not_decide_its_own_tier`.

## What this pass did — the schedule, restored, and verified against the artifact

The pass above changed one thing it did not mean to: the frame the relabel is
charged to. It answered from the frame's *own* observation, so a device whose
first frame missed the target stepped a tier on frame 0, where every earlier
build stepped on frame 1 - the frame after the cost that armed the run. This pass
established that by measurement rather than by reading, restored the order, and
found two more things on the way.

* **Restored.** `apply_tier_policy` now asks the relabel's question
  (`FrameLadder::acting`, the run as it stood *before* this frame's cost) before
  it observes the frame, and the offload keeps answering from the frame it just
  closed. The relabel is decided first and applied to the backend that is live
  then, so a frame that goes on to change the backend is recorded *after* the
  relabel the same frame earned - the order the pre-pass backends' own ladders
  kept. The run is the device's: `adopt_backend` no longer restarts it, because
  the frames a rebuilt backend presents were paid for by the device it replaced
  (`FrameLadder::reset` is gone with the last caller).
* **The measurement.** Through the shipped DLL, arm A (`onewriter.c`,
  `allow_downgrade = 0`, 1 ms target, 1-frame threshold): at 512x512 the pre-fix
  build stepped on frames 0, 1 and 2; the restored build steps on 1, 2 and 3, and
  its frame 0 reads `T1/gpu-shared` with an empty ring where the pre-fix build
  read `T2/cpu-ram` with one entry. At 64x64 the restored sequence is byte-for-byte
  the pre-pass one recorded earlier in this session (frame 0 `T1`/no entries,
  frame 1 `T2`/one entry), and arm B's return is logged `from T4` - the orphaned
  `from` the pre-pass left behind when the relabel's own ring died with the
  backend, which the single log now keeps.
* **This one had never run.** `ffi/tests/tiers.rs::a_plan_is_calibrated_once_however_often_it_offloads`
  lost its `#[test]` attribute when the offload proof was consolidated, so the
  "one calibration per plan" rule was pinned by a function nobody called (the
  compiler said so: "function is never used"). It runs now, and passes.
* **Probe hygiene, because the earlier claims outran their evidence.** The probe
  directory carried a DLL copied *before* that pass, and Windows loads a DLL from
  the executable's own directory first, so several probe runs labelled "against
  the rebuilt DLL" were not. Every run in this pass prints the md5 of the DLL in
  the directory and the script compares it with the fresh build before running
  anything; the stale copy is gone.

The lesson worth keeping: for this project, "measured through the shipped DLL"
means the md5 of the DLL was checked in the same breath as the run.

## Also this session — the bench reports what generation achieved

`reconl-bench --framegen=R` already printed the multiplier, the images-per-second
rate and the per-generated-image cost, but the *ordinary* wall line above them is
per **rendered** frame - so a run that delivered twice the images read as a run
that got slower (66 rendered fps for 130 images a second), and a host skimming
the output would turn the feature off. The wall line now says which frame it
times and carries the observed image rate ("per rendered frame: 54.9 rendered
frames/s, 108.2 images/s presented in all"), and the `presented` line names the
achieved ratio as well as the multiplier. Measured at 512x512: soft-cpu 54.9 ->
108.2 images/s (1.97x), d3d11 240.1 -> 441.3 images/s (1.84x against the same
run's rendered rate, 1.27x against a run without generation), each generated
image costing 1.5% of a rendered frame on the reference tier and 8.8% on the
hardware. Pinned by `tools/reconl-bench/tests/cli.rs::a_generation_run_reports_the_rate_it_achieved`,
which also holds the no-generation line to its old wording. Nothing about frame
generation, the tier policy, the frame-cost definition or the C ABI changed.

## Previous pass — the `ffi` crate's structure

`ffi/src/lib.rs` had grown to 4,025 lines owning everything a host can reach:
the handle table, the frame state machine, the tier policy, the row-layout rule,
the name tables and all ~60 exports. Nothing in it was wrong; the problem was
that a rule and its five reachable copies sat in one file, so a reader had to see
the whole surface to find the one place a decision was made.

It is now seven modules, each named for what it owns, and the rule that decides
where code goes is "the module named for the *rule*, not the entry point that
reaches it first" (the table in `docs/architecture.md` is the map):

| module | owns |
|---|---|
| `handle.rs` | the object model: `Kind`, `HandleHeader`, the child handle types, `header_of` (the only place a kind word is stamped), `reconlRetain`/`reconlRelease` and their destructor dispatch |
| `entry.rs` | the boundary glue: `FrameOwner`, `guarded_entry`, the `entry!`/`device_mut!`/`child!` macros, `cstr` |
| `offload.rs` | why this device is on the tier it is on: `PlanKey`, `Offload`, `apply_offload_policy`, the backend rebuilds, `downgrade_entries` |
| `layout.rs` | how bytes move between a host's buffer and a frame or texture; `host_row_layout` is the one place a pitch is judged |
| `order.rs` | the order each frame call considers its refusals in: one `Step` table per entry point, one function per question, one runner. Added by the pass below |
| `version.rs` | version numbers, the name tables, the log controls |
| `sizing.rs`, `abi.rs` | unchanged from earlier passes |
| `lib.rs` | the device's state and the entry points over it |

Two things were deleted rather than moved. `core/src/handle.rs` (271 lines) was
a complete, well-written, ref-counted handle table that **nothing called** - the
FFI has always had its own - so the project carried two representations of one
concern and the unused one was the more sophisticated. And the seven call sites
that stamped a `HandleHeader` by hand now call `handle::header_of::<T>(device)`,
which takes the kind word from `T`, so a kind and a type cannot drift apart.

The pass is shape only: no exported function changed, `tests/golden` is
byte-identical, and the workspace suite is 260 passed / 0 failed (four fewer than
before, all of them the deleted dead table's own tests). What it *did* change is
the cost of the next change: a rule about leaving the hardware now lands in
`offload.rs`, a rule about a pitch in `layout.rs`, and neither has a second copy
to keep in step.

One real defect surfaced while validating, and it was in the *test* rather than
the library: the 256x256 ladder leg derives its target window from both tiers'
measured costs, and on a machine loaded to ~35x normal those measurements cross -
the pilot's hardware first frame was 36.9 ms while the device the leg then built
presented its own first frame in 20.0 ms, under the derived 34 ms target, so the
offload the leg pins never fired. The rule is now a factor rather than a bare
comparison: the target is the ABI's smallest whole millisecond unless a fresh
hardware device's first frame is over *twice* the smallest millisecond that
covers the reference tier's typical frame. Three consecutive runs at 2 threads
(previously 1-in-3 failures) and two runs under deliberate double load now pass.

## Previous pass — built-in frame generation

A host that can render 60 frames a second and display 120 asks the device for the
frames in between instead of rendering them. The device reprojects the newest
frame it was handed along the camera motion the host declared — no shaders, no
second render, nothing about how a frame is drawn changes — and the host decides
*per frame* whether it wants one, which is what makes the feature a toggle a game
flips with its own quality settings rather than a mode the device is put into.

Who owns what:

- **`raster/src/framegen.rs`** — the warp, and nothing else knows how to do it:
  `Camera` (the view, plus the projection rebuilt from the same fov/near/far the
  ABI declares, so the two can never drift), `reprojection` (one matrix per
  generated frame, `P_prev · V_prev · V_cur⁻¹ · P_cur⁻¹`, applied to
  `(ndc_x, ndc_y, depth, 1)`), and `generate` (one backward warp, bilinear,
  clamped at the frame's edge). A still camera copies the frame it was given, to
  the byte — a menu, a paused game and the reference scene are still scenes.
- **`ffi/src/lib.rs::FrameGen`** — the retained history: the pixels the *host*
  was handed, the depth behind them, and the camera pair that is the motion.
  Owned by the device rather than a backend, because what a generated frame is
  warped from is the image the host saw, and that boundary is the FFI's.
  `reconlPresent` fills it only for a frame that asked;
  `reconlPresentGenerated` warps it out.
- **`include/reconl/reconl.h`** — `ReconLFrameGenDesc` on `ReconLFrameDesc`
  (trailing, so an older caller's struct is read as if it were NULL),
  `reconlPresentGenerated`, and the contract: the order its answers are decided
  in (tier, then whether there is anything to generate from, then the descriptor
  and `ahead` — see below), `ahead` in `(0, 1]`, `RECONL_ERR_NO_FRAME` when the
  last presented frame did not ask, `RECONL_ERR_NOT_SUPPORTED` on a tier with no
  depth to reproject, and the rule that a generated image never counts in
  `frames_presented` — the number the tier ladder judges — so a host cannot make
  a slow tier look fast by generating more images from its frames.

  That order was settled by measurement rather than argument, because two passes
  had explained the same probe finding differently. `%TEMP%\rlprobe\fgstate.c`
  drives both present paths with eight descriptors (a valid one, another call's
  struct type, null pixels, a zero-sized buffer, one row short, a pitch narrower
  than a row, `ahead > 1`, NaN `ahead`) across every reachable state — nothing
  presented, a frame open, a frame submitted, presented without asking,
  presented asking — on all three tiers, each cell on its own fresh rig: 90 cells
  per tier, 0 refusals that wrote into the host's buffer, 0 cells that left the
  device unusable. What decides a generated call's answer is **not the frame
  state machine** (a generated frame is not a frame: it answers identically in
  `open` and `submitted`) but whether a frame is being kept: with none, every
  descriptor and `ahead` answer `RECONL_ERR_NO_FRAME`; with one, the arguments are
  read (`-9`/`-1`); on a tier that keeps no depth every cell is
  `RECONL_ERR_NOT_SUPPORTED`. `reconlPresent` has the same shape with the frame
  state as its gate (`idle`/`open` → `-12` for every descriptor, `submitted` →
  the argument errors), which is what makes this one order rather than two
  accidents. So the library was right and the *header* was ambiguous: the
  headline sentence above now states the order instead of leaving it to be
  inferred. `ffi/tests/abi.rs::a_generated_frame_is_refused_for_its_arguments_only_once_there_is_one_to_generate`
  pins it on both paths, and fails on the rejected branch (validating arguments
  above the gate answers `-9` where the order says `-12`). The `fghostile` probe
  reported this as two findings for several passes; both were its own checks
  comparing a call the state does not accept with one that does, and neither was
  a library defect. It is 32 checks / 0 findings now, with the like-for-like
  comparison kept (`both present calls refuse the same narrow pitch in the state
  that accepts each`).
- **`backends/{soft-cpu,d3d11}`** — `depth_into`, the one read a present does
  not otherwise make, so a host that never asks pays neither the depth readback
  nor the three buffers.

The depth convention was a real defect this pass found. The reference tier's
perspective depth buffer held *clip z*; the hardware tier's holds the documented
reversed-Z NDC depth. The same readback therefore meant two different things, and
the warp was wrong on the CPU tier — a pixel 6 units away unprojected as if it
were 1 unit away. The rasteriser now stores the NDC depth its own `Target`
documents (one less divide per pixel in the hot loop), and the shadow passes are
unaffected because an orthographic matrix has `w == 1`. The committed golden is
unchanged by it, byte for byte.

Measured with `reconl-bench --framegen=R`, min-of-frame on a machine that was
also playing video, and with a scene camera that does not move — so these are
cost measurements, not quality ones:

| backend | resolution | asked | achieved | cost per generated image |
|---|---|---|---|---|
| soft-cpu | 512×512 | 2.0 | 1.98 | 0.83% of a rendered frame |
| soft-cpu | 512×512 | 3.5 | 3.40 | 1.20% |
| soft-cpu | 1080p | 3.5 | 3.31 | 2.34% |
| d3d11 | 512×512 | 2.0 | 1.87 | 6.83% |
| d3d11 | 1080p | 3.5 | 2.68 | 12.29% |

The hardware tier's multiplier is lower because its frames are nearly free and
its present path is not: at 1080p a rendered frame is ~8 ms and a generated image
~2.4 ms, most of that the host readback. The ceiling on the ratio is the game's
own ratio of what a frame costs to what a present costs — which is why this is a
toggle and not a default.

Pinned by `ffi/tests/abi.rs`: the toggle per frame
(`a_generated_frame_requires_a_frame_that_asked_for_one`), the still-camera
identity, the motion and the prediction
(`a_generated_frame_predicts_the_frame_after_it`: the stripe lands where the
camera's own motion says it must, *and* the generated image is closer to a real
render of the next camera than the frame it was warped from — on both tiers),
determinism and ladder isolation (`generation_counts_only_itself`), every
refusal plus that a refused generated frame does not wedge the device, the null
tier's `NOT_SUPPORTED`, the reservation a host that never asks does not pay
(`a_host_that_never_asks_reserves_nothing`, counted through the host's own
allocator), and a stats reset keeping the frame a host is looking at. Each was
revert-checked: reverting the depth convention, the reservation's allocator or
the reset each fails exactly its own test. Through the shipped DLL,
`framegen` (both tiers, 21 checks, 0 failures) drives the toggle, the refusals
and the counters as a C host does.

## Previous pass — render-pass viewport

`ReconLRenderPassDesc.viewport_width`/`viewport_height` were recorded by
`reconlCmdBeginRenderPass` and then destructured as `width: _, height: _` by the
replay: a host that asked for a sub-rect received a full-frame image and had no
way to tell. Through the shipped DLL (C probe, both tiers, 64×64 frame):

| requested viewport | before | after |
|---|---|---|
| 16×16 | 4096 lit pixels, all 3840 outside the rect | 256 lit, bounding box x[0..15] y[0..15] |
| 32×32 | 4096 lit | 1024 lit, x[0..31] y[0..31] |
| 0×0 (unset) | 4096 lit | 4096 lit (unchanged default) |

The change, in the fewest places that could carry it:

- `raster/src/lib.rs::rendered_viewport` — one rule, used by both tiers, that
  resolves a viewport in *frame* pixels against the target a tier actually
  renders into, so a tier at half resolution (T3/T4) confines the same fraction
  of the window. `0` in an axis means the whole target; an oversized viewport is
  clamped, never refused.
- `raster/src/tile.rs` — the render area is part of the target the tile loop
  works on: triangle bounds clamp to it and each tile's rect is intersected with
  it, so both the binning and the pixel loop are confined, and tiles still never
  overlap.
- `contract/src/lib.rs` — `FrameInput.viewport`, the per-frame seam the backends
  already take.
- `ffi/src/lib.rs` — the viewport is recorded from `BeginPass` into
  `FrameRecord` and carried into `FrameInput`; `FrameRecord::new` per frame
  resets it to `(0, 0)`.
- `backends/{soft-cpu,d3d11}` — resolve it and honour it; the shadow passes
  ignore it and draw their whole map.
- `include/reconl/reconl.h` — the fields' contract, in frame pixels, plus the
  present clause for a row pitch narrower than a row (refused with
  `RECONL_ERR_INVALID_ARGUMENT`, nothing written), which the previous pass
  introduced and had left undocumented.

Pinned by `ffi/tests/abi.rs::a_pass_viewport_confines_the_render_to_the_rect_it_asked_for`
(four legs - the reference tier at T2 and at T3/T4, where the render target is
half the frame, plus the hardware tier - asserting the exact lit-pixel count and
bounding box, and the unchanged full-frame default), with each half
revert-checked: ignoring the viewport in the replay, in the reference tier, or in
the hardware tier each fails the test.
`raster::rendered_viewport` has its own unit test for the rounding and clamping
rule, which is where a cross-tier difference would otherwise start.

## Checks

```bash
probes/run.sh                                  # the C probes, against a freshly built and md5-confirmed DLL
cargo test --workspace                         # unit, ABI host, tier matrix, golden, CLI
cargo test --workspace --profile tested        # the same, at release optimisation
cargo run -p reconl-diff -- compare tests/golden/soft-cpu-shadow.png
cargo run -p reconl-bench --release -- --backend=d3d11 --resolution=512x512 --repeat=200
cargo run -p reconl-bench --release -- --backend=soft-cpu --resolution=512x512 --framegen=2
```

**Do not run the suite with `--release`.** The release profile is `panic =
"abort"` (that is the crate's ABI contract), and `cargo test --release` forces
`panic = "unwind"` for test targets, so cargo builds the whole graph twice - an
abort graph for the bins and examples, an unwind graph for the test binaries.
Both emit `reconl.dll`, `libreconl.a` and `libreconl.rlib` under the same names
in `deps/`, which cargo warns about (issue 6313) and which makes the run fail
every few tries with an unrelated tool reporting `can't find crate for
reconl_host`. `--profile tested` is release minus the abort, so a test run builds
one graph and cannot race itself; it is what the numbers in past passes were
measured with.

Neither ladder pin selects its window from a measurement any more, which is both
what used to make them load-sensitive and what this pass removed.

`ffi/tests/abi.rs::the_device_ladder_judges_the_frame_the_host_reads` used to
place its target inside the gap between the device's own render time and the
frame the host read, derived from a pilot device's measured 1080p cost - so a
machine busy enough to stall the pilot could leave the target above the frames
the tested device then produced. It now drives 1080p frames at the ABI's smallest
target (1 ms, under every frame either tier renders) with the offload opted out,
and asserts arithmetic instead: the cost the ladder *recorded* in its first ring
entry must equal the `total_ns` the host read for that frame. To make that
possible the relabel's detail carries the number it acted on
(`FrameLadder::last_over_ns`, once, with the run it belongs to) - it used to name
only the run and the target, which is why the old form had to infer the number
from where the target sat. Revert-checked twice: judging `frame_numbers()` fails
on the step assertion (that number is under the target), and judging a floored
device number - so it steps - fails on the arithmetic (`the ladder must judge the
frame the host read (15781100 ns) ... the host read (5000000 ns)`). No
measurement is taken to find the window, so nothing about the pin moves with the
load on the box.

The two `ffi/tests/tiers.rs` offload legs still derive a window, because what
they pin (the plan memory, the settle window, the return) only happens at a
target a tier can miss or sit inside. That derivation now uses the reference
tier's **cheapest** warm frame rather than its median - the cheap end is what the
settle window needs, and it does not drift over a millisecond while the hardware
stays put, which is how the median-based form crossed its own boundary under
parallel execution - and the shared `ladder_leg` harness re-derives and re-drives
up to three times before it fails loudly. Both files were then run green
repeatedly under deliberate oversubscription; see the pass note below.

The C host probes (`gpu_probe`, `hostile*`, `framestate`, `shadowconfig`,
`offload`, `viewport`, `framegen`, `onewriter`) drive the shipped DLL and
currently live outside the repository, in the build machine's temp directory. All
of them are at zero on this machine - last run: `offload` 33/0, `framestate`
43/0, `shadowconfig` 40/0, `framegen` 21/0, `narrowpitch` 11/0, `ladder` 6/0
(3 backend changes, 2 offloads, 1 return), the three hostiles 11/13/11 with no
findings, `viewport` 0 pixels outside the rect, and `onewriter` 8/0 (which is
this pass's probe: its arm A is the d3d11 relabel with the offload opted out, and
its arm B reports 0 entries lost and 0 order breaks in the ring where the
concatenated logs lost one). `offload` included: its 64×64 arm used to judge a frame
of 0.8–2 ms against a hard-coded 1 ms target inside a fixed six-frame window, and
failed 3 of 16 runs on the last-frame `settled` check - a window that ended
mid-round-trip, not a violation (every one of those runs made exactly one
return, which is what the rule bounds). The target is now derived from the
reference tier's own cheapest measured frame, the arm prints its derivation and
its regime, and it drives the ladder until three frames change no backend. See
`docs/offload.md` for the measurements.

## The offload ladder's two halves

The tier ladder has one rule with two halves, and they are decided in one place
(`apply_tier_policy`) but only one of them is reachable at a time. *One return per plan* is pinned
at 256×256 (`an_overloaded_hardware_tier_offloads_and_the_measurement_decides`),
where the reference tier loses the calibration and the device goes back to the
hardware. *One calibration per plan* needs the opposite branch - the reference
tier wins, the device stays, the settle window buys it a return, and the frame
that misses after that must reuse the comparison - and it is pinned at 32×32
(`a_plan_is_calibrated_once_however_often_it_offloads`), because at 64×64 and
above the two tiers' frame costs overlap in a way that leaves no whole
millisecond between "the reference tier meets it" and "a rebuilt hardware device
is over it".

Both legs share one harness (`LadderWindow`/`ladder_window`, `LadderRun`/
`ladder_run`) and one derivation, deliberately: the target is a measurement, so a
second copy of the derivation is a second opinion about the same number. The
derivation takes both tiers' piloted costs. The hardware has to *miss* the
target, and where the reference tier is the faster tier its frames have to be
*inside* it, so the target is the ABI's smallest whole millisecond - which every
hardware frame measured here was over - raised to cover the reference tier's
typical warm frame **only when a fresh hardware device's first frame is over
twice the raised value**. The doubling is the point: a first frame is a device
setup frame whose cost varies by more than 2x between one cold device and the
next on a loaded machine, so a target only just under it stops being a target for
the device the leg then builds - which is exactly how this leg failed once under
load, with a derived 34 ms target and a run device presenting in 20 ms. Typical
and not cheapest, for the raised value: the settle window arms on an ordinary
frame. A host with no usable whole millisecond fails with both measurements
instead of passing.
The remembered leg runs its sequence twice, on a plan key whose shadow work is
real (the occluder scene at 32×32 with a 2 MiB map budget: enabled, two
cascades, PCF) and on one without, because the plan a comparison is filed under
is the resolution and the shadow configuration together.

Measured after the consolidation: 8/8 runs of the pin alone and 5/5 full-file
runs at `--test-threads=2`, which is the loaded configuration that exposed the
old 256×256 derivation's flaw - it capped the target with one pilot device's
first frame, and the run builds its own device, so a first frame a hair under the
cap meant no offload and a failed leg. When the reference tier is the slower one
the cap is no longer what picks the target; the leg still asserts the trigger, so
a host that cannot produce the sequence says so with the numbers.

Neither pin asserts a frame schedule: which frame a change lands on is a
measurement, so they assert the bound (≤3 backend changes, ≤1 return, three quiet
frames at the end) and read the ring for what the policy wrote. The pins' rule
and the C probe's arm differ in exactly one stated way - the arm needs only the
offload event, so it sets its target at the smallest whole millisecond the
reference tier's *cheapest* frame is inside, while a leg that needs the *stay*
branch has to cover an ordinary frame, which is why the pins use the typical one.

## The ABI layout claim, made real

`README.md` and `ffi/src/abi.rs` both cited `ffi/tests/abi_layout.rs` as the
thing that compiles the shipped header against the Rust mirror. No such file
existed: the layout agreement between `include/reconl/reconl.h` and
`ffi/src/abi.rs` was exercised only out of band, by the C probes. The file now
exists and does what the citations promised.

`ffi/tests/abi_layout.rs` walks one table per struct - the Rust mirror's fields
and types - and writes a C file of `_Static_assert`s: one `sizeof` per struct,
an `offsetof` and a field `sizeof` per field, and a `_Generic` check that the
field's category is the one the Rust side declares (integer, float, pointer,
integer array, float array). It compiles that file with `cc`/`gcc`/`clang`
against `include/reconl/reconl.h`: 33 structs, 899 asserts. It skips with a
printed reason where no C compiler exists, the way the d3d11 legs skip, and
leaves the generated file in `CARGO_TARGET_TMPDIR` with its path in the failure
message. The skip path was exercised by running the built test binary with no
compiler on `PATH`; the pin was revert-checked by swapping
`shadow_texel_budget_bytes` and `worker_threads_max` in `ReconLDeviceLimits`
(which compiles, unlike adding a field, whose initializers Rust rejects first),
after which the C compiler fails with
`"ReconLDeviceLimits::shadow_texel_budget_bytes: offset"`.

The same sweep settled every other citation of a file that does not exist.
`docs/abi.md` was cited three times and never written, so those comments now
point at what enforces the rule: the header's push-constant and logging notes and
its struct rules. `tests/thread_determinism.rs` and
`raster/tests/thread_determinism.rs` were cited as pinning worker-count
determinism; the test that actually pins it is
`raster/tests/render.rs::frame_is_bit_identical_for_1_2_4_and_8_workers`, and
`docs/determinism.md` and `raster/src/lib.rs` now say so. `tests/shared_edge.rs`
is the inline `top_left_rule_covers_a_shared_edge_exactly_once`. `ffi/src/handle.rs`
named `resources.rs`/`command.rs` as the modules using the child handle types;
they are `lib.rs` and `order.rs`. `docs/architecture.md` also claimed the C
probes live outside the repository - corrected to `probes/`. What remains
unresolved is the mission document's own roadmap: `PROMPT.md` lists deliverables
(`examples/c/triangle.c`, `examples/rust/triangle.rs`, `reconl.js`, `DESIGN.md`)
that do not exist yet, which is what a roadmap is.

Validation: `cargo test --workspace --profile tested` 269 passed / 0 failed (268
+ the new pin), 0 ignored; the golden is byte-identical
(`sha256 7e8ecbab…`, `reconl-diff compare` identical, 4096 px); `probes/run.sh`
reports 24 rows, 24 pass, 0 fail, 0 skip against the freshly built DLL
(`md5 2090bcf7…`).

## The numeric contract, gated

The layout pin covered structs; every number the shipped header exports was
still checked by eye. `ffi/tests/abi_layout.rs` now writes a second table - 219
`_Static_assert`s beside the 899 layout ones - one per macro and enumerator,
each against the Rust value the implementation uses:

- result codes against `Code` *and* `abi::result` (two Rust copies, so two
  asserts);
- struct types, backends, caps (against `abi::caps` *and* `core::tier::caps`),
  downgrade flags, shading;
- tiers and tier reasons, shadow filters and events, log levels;
- the raster pipeline enums - compare, cull, blend, light type, sampler
  filter and wrap - which is where a renumbering would change an image;
- formats, usages, index format and frame state, which the implementation never
  named: `abi::format` and `abi::index_format` are new and now load-bearing, so
  the texture gate, the swapchain gate and the indexed-draw gate test a
  descriptor against exactly those names instead of `1 | 3` and `!= 1`.

`abi.rs` also gained the two capacities its mirror was missing
(`RECONL_MAX_VERTEX_STREAMS`, `RECONL_MAX_CAPS`) and `struct_type::D3D11_DESC`.
`RECONL_COMMAND_BYTES` stays with its owner: `ffi/src/sizing.rs` compares it to
the header line in its own unit test, and that private constant is not reachable
from a test crate.

No value mismatch existed when the gate first ran: the header and the Rust side
agree on all 219 numbers. The gate was revert-checked both ways -
`result::NO_FRAME` set to -13 in Rust fails `"RECONL_ERR_NO_FRAME: value"`, and
`RECONL_MAX_CAPS` set to 31u in the header fails `"RECONL_MAX_CAPS: value"` -
and both were restored.

## The backend descriptors are reserved, and now say so

The three `ReconL*Desc` structs in `reconl_backends.h` read as if
`reconlCreateDevice` honoured them; it never has. The header now states the
status plainly - reserved in ABI 100: accepted and never read, a wild pointer as
harmless as NULL, the fields never repurposed, so the structs are the contract
for the revision that reads them - and `reconl.h`'s comment on the parameter
says the same. One dangling name went with it: the null desc's comment referred
to `RECONL_NULL_DESC_FAKE_TIER`, a macro that exists nowhere; it names the field
now.

A negative contract needs a pin, so `ffi/tests/abi.rs` gained
`a_reserved_backend_descriptor_is_accepted_and_never_read`: a device created
with `backend_desc` at address 8 must come back identical - backend, caps,
worker cap, tile size, allocation ceiling - to one created with none.
Revert-checked by making `reconlCreateDevice` read the field, at which point the
test dies with `STATUS_ACCESS_VIOLATION` at that address; restored to green.

Validation: `cargo test --workspace --profile tested` 270 passed / 0 failed (269
+ the descriptor pin), 0 ignored; the golden is byte-identical
(`sha256 7e8ecbab…`, `reconl-diff compare` identical, 4096 px); `probes/run.sh`
reports 24 rows, 24 pass, 0 fail, 0 skip against the freshly built DLL
(`md5 a30872de…`).

## A steady-state frame allocates nothing

`docs/determinism.md` §5 had been recording its own failure rather than hiding
it: 8 host-allocator calls per measured frame on the reference tier and 2 on
d3d11, against PROMPT §12's criterion of zero. The sites were countable and
there were exactly as many as the ledger said - the frame's draw list
(`reconlSubmit` allocated it per frame), `build_draws` (a translated copy of it),
and in the reference tier `static_draws`, `dynamic_draws`, one `cascade_draws`
per cascade, `map_refs` and the colour list.

Each is now owned by something that outlives the frame and cleared rather than
reallocated:

* **The frame's draws** are `FrameRecord::items`, written in the command loop
  where the bound checks live, kept across frames by `FrameRecord::reset` (a
  fresh record per frame is what dropped the capacity), and read by a backend
  through `frame_input`. `DrawRecord` - raw pointers that `build_draws`
  re-sliced - is gone with them: the slicing now happens where the check that
  justifies it is, which is also why the fault path re-renders a frame out of the
  list it already holds instead of building one.
* **The reference tier's lists** are `SoftCpuDevice::cascade` and
  `SoftCpuDevice::colors`, filled by `fill_cascade` / `fill_colors`. Both are
  `HostVec<DrawItem<'static>>` as *storage*: entries are copies of the frame's own
  draws, dropped by the `clear` at the start of the next fill, so the shortened
  lifetime one `transmute` per fill creates never outlives the borrow it was
  copied from. Both count their own growth in `FrameNumbers::allocations_in_frame`
  instead of counting an attempt.
* **The cascade map view** is a fixed `[ShadowMapRef; MAX_CASCADES]` on the
  stack: a cascade set is bounded, and a view of maps the device already owns
  needs no allocation at all.
* The per-cascade caster filter is an `any()` over the frame's draws rather than
  two gathered lists, so the shadow pass no longer copies the frame to split it.

Measured through `reconl-bench`'s own ledger (the host allocator, in the tool's
process): 0 per frame at 64x64 with shadows on and off, at 512x512, at 960x540 on
d3d11, and on the null tier - it was 8/2 before. Pinned by
`tools/reconl-bench/tests/cli.rs::a_steady_state_frame_allocates_nothing` (both
tiers, the d3d11 leg skipping with a printed reason where no device exists), which
was revert-checked twice: a fresh `FrameRecord::items` per submit reports "1 per
frame" on both tiers and a local colour list reports "1 per frame" at T2.

The library-side counters now agree with the ledger: `allocations_in_frame` counts
only real growth, so a normal frame reports zero rather than a number nobody could
check. The `RECONL_ALLOC_TRACE` backtrace hook used to attribute the sites was
temporary and is removed from `tools/host/src/alloc.rs`.

**Adversarial pass over that change.** Two claims it rested on were only in
comments, and one was false. `FrameRecord::items` and the soft-cpu fills claimed
the *device* owns the vertex buffers the entries point into; it does not - the
host's reference count does, the test rigs release a frame's buffers right after
submit, and the comments now say that. And the frame's own draw list did not count
its growth while `docs/determinism.md` said every list does: `FrameRecord` now
carries `items_grown`, counted at the two push sites where the list can take a
block and folded in once by `frame_cost()`, so the library's number covers the
frame's list too and a growth - the one allocation a steady state does not have -
is visible from inside the library, not only from a host's ledger.

Storage that is cleared and refilled has one failure mode worth its own pin: a
stale tail. `backends/soft-cpu/tests/render.rs` gained
`a_frame_that_draws_less_than_the_one_before_it_leaves_no_stale_entries` - a
floor-plus-caster frame followed, in the same device, by a floor-only frame, whose
checksum must equal a fresh device's floor-only frame. Revert-checked by deleting
`list.clear()` in `fill_cascade`: fails with "rendered storage the earlier frame
left behind". Restored. The pin covers the reference tier's two lists; the FFI's
`items` reuse is exercised by every multi-frame test but has no dedicated
shrink-from-the-ABI leg - the honest limit of this pass.

That limit is now closed. `ffi/tests/abi.rs` gained
`a_frame_that_records_less_than_the_one_before_it_leaves_no_stale_geometry`:
the `scene()` is two quads in one buffer - a dim plane at indices 0..6 and a white
stripe at 6..12 - so one frame records both as two draw commands and the next, on
the same live device, records the plane alone. The lean frame's presented pixels
must be byte-identical to the same lean frame on a freshly created device, and must
contain no white at all, on soft-cpu and on d3d11 (the hardware leg skipping with a
printed reason where no device exists). The existing world recorder was widened
rather than copied: `submit_world_frame` now takes `(index_count, first_index)`
segments and emits one draw command each, which is the only new machinery.

Revert-checked by removing the clear at both of the two sites that hold the
invariant - `FrameRecord::reset` and the command replay in `Submit` - and it fails
naming the leak: "soft-cpu: a frame that recorded one draw presented stale geometry
- the white stripe the frame before it recorded". Removing either site alone still
passes, because they are mutually sufficient; both are kept, since the replay's
clear is what makes the invariant true at the point the list is filled (including
the fault re-render path) and the reset's is what makes a begun frame empty.

**The fault path's re-render, and what cannot be driven.** The same hazard one
layer over - the offloaded frame is re-rendered out of the list the frame already
holds - now has a pin:
`ffi/src/offload.rs::fault_tests::the_frame_that_faults_is_rerendered_out_of_its_own_draws`.
It builds a hardware device and a reference-tier device through the exported ABI,
submits a frame whose geometry is the *other* half of the frame from the one
before it, then runs the fault's own two steps in the order `Submit`'s arm runs
them (`offload_on_fault` then `render_frame_on_soft`, the frame taken out of the
device and put back as the submitted frame), and asserts the pixels the host is
handed are byte-identical to the reference device's render of the same draws,
with the other half of the frame lit by neither. It runs wherever a D3D11 device
exists and prints the reason where none does.

The honest limit, recorded because the pass that asked for this assumed
otherwise: **the fault itself cannot be injected.** There is no fault hook in the
library or the ABI - a failover needs `Code::DeviceLost` *and* a genuine driver
verdict (`D3d11Device::device_removed`), and nothing offers a device as removed.
The `--repeat=200` removal the README recorded was a suspended device instance on
this machine, not a reproducible condition: re-measured at 512x512 it completes 5
warmup + 60 measured frames with `failures 0` and never leaves d3d11, so it was a
state of the driver, not of the load. The README now says so. The verdict is
therefore the one step the pin does not take: everything around it is the real
surface (device, frame, command replay, submit, present), and the two steps after
it are the arm's own.

Revert-checked twice. With both list clears removed (the accumulation failure),
it fails naming the contamination: "the frame that faulted presented the frame
before it as well". With the re-render reading a fresh record instead of the frame
it was handed, it fails "the frame that faulted lost its own geometry". Both
restored. One thing the pin's own first run taught: `items` is built by `Submit`'s
replay, so a frame that never submits has an empty list to re-render - which is
exactly the empty-record regression the pin now catches.
