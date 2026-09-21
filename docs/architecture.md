# Architecture: who owns what

The crate *layout* is `PROMPT.md` §5. This file records the **ownership**
decisions inside it — the ones a change has to respect, because they are what
keeps a behaviour in exactly one place. It describes the code as it stands, not
as it is planned.

## The flow of one call

```
host
 └─ ffi entry point            validate, price the request, translate
     ├─ ffi::sizing            what does this descriptor cost?      (pure)
     ├─ ffi::layout            does this host pitch make rows?      (rule)
     ├─ core::budget           is that affordable?                  (policy)
     ├─ ffi::offload           which tier renders the next frame?   (policy)
     └─ backend                allocate it, render, present         (mechanism)
         └─ raster / shadow    the reference implementation
```

## The `ffi` crate's own layout

The C surface is one crate, split so that each concern can be read alone. The
rule is that a *rule* lives in the module named for it, not in the entry point
that happens to reach it first:

| Module | Owns |
|---|---|
| `abi` | the `#[repr(C)]` mirror of the header |
| `sizing` | what a descriptor costs, from the descriptor alone (pure, unit-tested) |
| `handle` | the object model: kinds, the header every handle starts with, the child handle types, `header_of` (the only place a kind word is stamped), and the ref-counted lifetime |
| `entry` | the boundary glue every entry point uses: `FrameOwner` (a failed call ends its frame), `guarded_entry` (the panic boundary), the `entry!`/`device_mut!`/`child!` macros, `cstr` |
| `offload` | the whole answer to "why is this device on that tier": `PlanKey`, `Offload`, `apply_offload_policy`, the backend rebuilds, the `downgrade_entries` ring |
| `layout` | how bytes move between a host's buffer and a frame or texture; `host_row_layout` is the one place a pitch is judged |
| `version` | version numbers, the name tables, the log controls |
| `lib.rs` | the device's state (`BackendKind`, `FrameRecord`, `DeviceHandle`, `Origin`, `FrameGen`) and the entry points over it |

The `lib.rs` banner sections that are *not* yet files - `probe`, `device`
(create/limits), `resources`, `commands`, `frame`, `submit`, `present`, `stats` -
are the next boundaries to draw. They are recorded here rather than left to be
discovered, because a reader who finds a 3,000-line file should be told which
parts of it are already claimed by a module and which are waiting.

## Owners

| Decision | Owner | Everybody else |
|---|---|---|
| What a descriptor costs (texture chain, command slots, swapchain images) | `ffi/src/sizing.rs` | entry points *ask* it; nothing re-derives the arithmetic |
| What is affordable: the single-allocation ceiling, the RAM cap, the accounting | `core/src/budget.rs` | every allocation path goes through `admit_ram` / `reserve_ram` / `check_allocation` |
| Which tier runs, and what the shadow system may do | `core/src/tier.rs` (`resolve_tier`, `shadow_plan` → `ShadowPlan`) | backends allocate `plan.map_size` and reserve `plan.resident_bytes`; they never size a cascade set themselves |
| Whether a device continues on the reference tier, and what a tier change costs the host | `ffi/src/offload.rs` (`Origin` stays with the device; `Offload`, `apply_tier_policy`, `apply_tier`, `adopt_backend`) plus `core/src/tier.rs::FrameLadder` (the run a step is decided from), documented in `include/reconl/reconl.h` and `docs/offload.md` | backends never move a device between tiers or record one: a backend is told its tier (`relabel`) and applies its own part of the change. `apply_tier_policy` is the only decider, `DeviceHandle::downgrades` is the only log, and `desc.downgrade_after_frames` has one reader - the device's `FrameLadder` |
| Whether a driver failure means the device is gone | `backends/d3d11` (`device_removed`, the driver's own `GetDeviceRemovedReason`) | the ffi asks; it does not infer a removal from an error code |
| Cascade fitting, texel snapping, bias presets, cache keys | `shadow` | backends call it; they do not re-derive bias or fits |
| How a frame is reprojected into a generated one, and what its camera is | `raster/src/framegen.rs` (`Camera`, `reprojection`, `generate`) | the ffi retains the frame and calls it; no backend knows how to warp an image |
| What a generated frame is warped from: the pixels the *host* was handed, their depth, and the camera pair | `ffi/src/lib.rs::FrameGen` | backends expose `depth_into` and nothing else; they never keep a history of their own |
| Whether a frame asks for generation, and whether a host may present a generated one | the frame's own `ReconLFrameGenDesc`, checked by `reconlBeginFrame`/`reconlPresentGenerated`, documented in `include/reconl/reconl.h` | a tier that keeps no depth refuses it (`RECONL_ERR_NOT_SUPPORTED`); it is never a device mode |
| The order *every* frame call considers its refusals in, and where its commit point is | `ffi/src/order.rs` (one `Step` table per entry point, one function per question, one runner), documented in `include/reconl/reconl.h` beside each call | `reconlBeginFrame`, `reconlSubmit`, `reconlPresent` and `reconlPresentGenerated` each `walk` their own table; an entry point asks its questions and does not order them. The tables carry the two documented asymmetries side by side - `reconlSubmit` asks its argument before its state, `reconlPresent` the state before its descriptor - and are what make "nothing dereferences a host pointer before its gate" structural. `Step::Commit` is where a call takes the frame: everything before it leaves the frame as it was. Pinned by `ffi/tests/abi.rs::a_generated_frame_is_refused_for_its_arguments_only_once_there_is_one_to_generate` (all three tiers) and `::a_descriptor_is_never_read_before_the_gate_that_owns_it` (NULL, unreadable, wrong-type and prefix-only headers on both present paths); `%TEMP%\rlprobe\fgstate.c` is the measurement behind them (90 cells per tier, three tiers) |
| The frame state machine (idle → open → submitted) and its recovery contract | `ffi/src/lib.rs` `FrameState` + `ffi/src/entry.rs` `FrameOwner`, documented beside `reconlBeginFrame` in `include/reconl/reconl.h` | backends answer one call at a time and never hold frame state |
| Handle lifetime and refcounting | `ffi/src/handle.rs` (the header, `header_of`, `reconlRetain`/`reconlRelease` and the destructor dispatch) | no entry point frees or retains by hand. `core` used to carry a second, never-called handle table; two representations of one concern is how a reader learns the wrong one, so it was deleted rather than kept in step |
| What a host pitch means, and what a host gets when one is too narrow | `ffi/src/layout.rs` `host_row_layout` (the check) and `lay_out_rows` (the write) | every readback and upload path calls the check first; a path that lays rows out does not judge them |
| Render mechanism: targets, passes, present, GPU state | `backends/*` (the reference rasteriser is `raster`) | core and ffi know no platform or graphics type |
| The reference scene: geometry, light, camera, shadow spec, and the three shadow modes | `tools/host/src/scene.rs` | `reconl-diff` and `reconl-bench` both read it; neither carries a copy |
| The host side of a frame (create the resources, begin, encode, submit, present, time the boundaries) | `tools/host/src/frame.rs` | tools encode a frame; they do not re-implement the order |
| What a host allocator has been asked for, and what is still outstanding | `tools/host/src/alloc.rs` | tools read the ledger; the library is the only caller of it |
| The evidence for every claim about the *shipped* artifact: the C programs that drive it and the runner that builds and reports them | `probes/src/*.c` and `probes/run.sh`, documented in `probes/README.md` | the runner builds the release library, md5-confirms the copy the probes link against, and pins each probe's check count and required lines; a probe result always names the binary that produced it |

## Rules that keep it that way

- **One allocation, one gate, in this order:** price (`sizing`) → admit
  (`budget`) → allocate (allocator or driver). Nothing allocates and then checks.
- **Policy never lives in an entry point.** `ffi` validates, prices, translates
  and delegates; anything it decided would be policy that Rust tests cannot
  reach without a host.
- **A backend implements the portable core, it does not define it.** When a
  backend needs a number the core already decides (a map size, a tier rule), it
  reads it from the core instead of carrying a copy; two copies kept in step by
  a comment is the failure this rule exists to prevent.
- **`include/reconl/reconl.h` is the ABI source of truth**; `ffi/src/abi.rs`
  mirrors it field for field, and each struct's `MIN_SIZE` is what
  `check_header` enforces for a host compiled against an older header.
- **The reference tier defines correctness.** Hardware differences are defects
  or documented clamps, never a widened tolerance.
- **A tool is a host.** Tools drive the same C ABI an application does, through
  the same `reconl-ffi` entry points, and share their plumbing in `tools/host`
  rather than each growing a private copy of the allocator, the scene or the
  frame order. A measurement is only evidence if the thing measured is the thing
  that ships.

## The invariants, and what pins them

| Invariant | Pinned by |
|---|---|
| No request above the ceiling is ever attempted, and nothing is allocated before the refusal | `core/src/budget.rs` tests; `ffi/tests/abi.rs::an_absurd_request_is_refused_before_the_allocator_is_called` |
| One budget buys one map size, identically on both tiers | `core/src/tier.rs` tests; the `ffi/tests/tiers.rs` configuration matrix; the `shadowconfig` C probe |
| A tier change has one decider and one record: the ring is every change the device made, in the order it happened, and never loses one because the backend that made it was replaced | `core/src/tier.rs::FrameLadder` tests; `ffi/tests/tiers.rs::the_tier_log_holds_every_change_however_the_backend_moved` (revert-checked against the per-backend rings it replaced), `::a_host_that_forbids_tier_changes_keeps_its_hardware` (the opt-out still gets the relabel), and `backends/soft-cpu/tests/render.rs::the_reference_backend_does_not_decide_its_own_tier` |
| Which frame a tier change is visible on: the relabel on the frame after the cost that armed it, the backend change on the frame that missed | `ffi/tests/tiers.rs::the_ladder_charges_a_relabel_to_the_frame_after_the_cost` (revert-checked: fails when the relabel answers from the frame's own observation), measured through the shipped DLL by `%TEMP%\rlprobe\onewriter.c` |
| The reference scene renders byte-for-byte what is committed | `reconl-diff compare` against `tests/golden/soft-cpu-shadow.png`, and independently `reconl-bench --png` compared to the same file (`tools/reconl-bench/tests/cli.rs`) |
| No allocation inside a frame: every draw list a frame writes into has an owner that outlives it, and a steady state shows zero in the host's own ledger | `tools/reconl-bench/tests/cli.rs::a_steady_state_frame_allocates_nothing` (both tiers), documented in `docs/determinism.md` §5; the rasteriser's own tables are pinned by `raster/tests/render.rs` |
| A probe creates no device and allocates nothing | `reconl-info --probe-only`, whose allocation ledger is checked in `tools/reconl-info/tests/cli.rs` |
| A device gives every byte it took back at release | `reconl-info`'s host ledger, checked in the same test |
| A run reports the configuration that produced it, and the per-frame record | `tools/reconl-bench/tests/cli.rs` |
| A failed ABI call leaves the device able to make the call the state machine names next | `ffi/tests/abi.rs` frame-state tests; the `framestate` C probe |
| A generated image is not a rendered frame: it never moves `frames_presented`, the number the ladder judges | `ffi/tests/abi.rs::generation_counts_only_itself`; the `framegen` C probe's `frames_presented` check against the shipped DLL |
| A depth readback means the same thing on every tier | `ffi/tests/abi.rs::a_generated_frame_predicts_the_frame_after_it` (fails if the reference tier stores clip `z` instead of `z/w`); `docs/determinism.md` rule 1 |
| A device only leaves its hardware for a genuine removal or a measured overload, and a refused argument is neither | `ffi/tests/tiers.rs::a_refused_call_does_not_offload_the_device` (fails with a code-based trigger); the `offload` C probe |
| A driver failure's code is the HRESULT's own class, decided in one place | `backends/d3d11/src/imp.rs::classify_hresult` (the only HRESULT→code mapping); `ffi/tests/tiers.rs::a_driver_rejected_frame_reports_an_argument_error_not_a_loss` (fails under a blanket `DEVICE_LOST`); the `offload` C probe's arm 4 asserts `-1`, not `-6`, against the shipped DLL |
| An offloaded frame is the reference tier's frame | the calibration-frame comparison in `ffi/tests/tiers.rs`; `ffi/src/offload.rs::fault_tests::the_frame_that_faults_is_rerendered_out_of_its_own_draws` (a frame submitted on hardware, then the fault's own two steps, compared byte for byte with a reference-tier device rendering it - the driver's removal verdict is the one step no test can inject); `--png` from the `--repeat=200` fault run `cmp`-compared with a `--backend=soft-cpu` run |
| A calibration is decided by the measurement it took, and the host can read that measurement afterwards | `ffi/tests/tiers.rs::an_overloaded_hardware_tier_offloads_and_the_measurement_decides` (fails with the comparison inverted); `reconl-bench`'s `final` line; the `offload` C probe |
| The shipped header and the Rust mirror agree on every struct the ABI reads | `ffi/tests/abi_layout.rs`: it generates `_Static_assert`s from the Rust `size_of`/`offset_of` values - each struct's size, every field's offset and width, and its integer/float/pointer category - and asks a C compiler to compile them against `include/reconl/reconl.h` (33 structs, 899 asserts). A field added, removed, reordered or retyped on one side fails it; it skips, with a printed reason, where no C compiler exists |
| Every number the ABI exports equals the Rust value the implementation uses | the same `ffi/tests/abi_layout.rs`: one more `_Static_assert` per exported macro and enumerator - result codes (against both `Code` and `abi::result`), struct types, backends, tiers and tier reasons, caps (against both copies), downgrade/log/format/usage/index/pipeline/texture/light/shadow/frame-state enums (219 asserts). A renumbered code, flag or enumerator fails it naming the constant |
| The backend descriptors are reserved: accepted and never read | `ffi/tests/abi.rs::a_reserved_backend_descriptor_is_accepted_and_never_read` (a descriptor at a wild address must not fault, and the device must match one made with none); the status is stated in `include/reconl/reconl_backends.h` |

## Verification surfaces

- `cargo test --workspace` — unit tests, the in-process ABI host, the tier
  matrix, the golden compare, and the CLI tests that spawn `reconl-info` and
  `reconl-bench` as processes. Use `--profile tested` for the same suite at
  release optimisation; not `--release`, which builds the graph twice (abort for
  the binaries, unwind for the test targets) and intermittently fails to link a
  tool on the duplicate `deps/` output names. The abort/unwind split is why the
  profile exists; `Context.md` has the details.
- `reconl-diff render <png>` / `compare <png> [--tolerance=N]` — the real tool,
  through the same ABI a host uses.
- `reconl-info` / `reconl-bench` — the probe, the limit and ledger report, and
  the timed run with its fingerprint and trace. These run in their own process,
  which is what makes a claim about the process-wide allocation ledger
  measurable rather than racy.
- The C host probes (`probes/src/*.c`, run by `probes/run.sh`) drive the
  *shipped DLL* — the only place `reconlConfigureShadows` and the cross-ABI
  recovery paths are exercised as a C host would. The runner builds the library,
  md5-confirms the copy every probe links against, and prints one row per probe
  with its checks and findings; `probes/README.md` says what each one is for.
