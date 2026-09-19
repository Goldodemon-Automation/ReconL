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
     ├─ core::budget           is that affordable?                  (policy)
     └─ backend                allocate it, render, present         (mechanism)
         └─ raster / shadow    the reference implementation
```

## Owners

| Decision | Owner | Everybody else |
|---|---|---|
| What a descriptor costs (texture chain, command slots, swapchain images) | `ffi/src/sizing.rs` | entry points *ask* it; nothing re-derives the arithmetic |
| What is affordable: the single-allocation ceiling, the RAM cap, the accounting | `core/src/budget.rs` | every allocation path goes through `admit_ram` / `reserve_ram` / `check_allocation` |
| Which tier runs, and what the shadow system may do | `core/src/tier.rs` (`resolve_tier`, `shadow_plan` → `ShadowPlan`) | backends allocate `plan.map_size` and reserve `plan.resident_bytes`; they never size a cascade set themselves |
| Whether a device continues on the reference tier, and what a backend change costs the host | `ffi/src/lib.rs` (`Origin`, `Offload`, `apply_offload_policy`), documented in `include/reconl/reconl.h` and `docs/offload.md` | backends never move a device between tiers; `apply_offload_policy` is the only caller of `adopt_backend` |
| Whether a driver failure means the device is gone | `backends/d3d11` (`device_removed`, the driver's own `GetDeviceRemovedReason`) | the ffi asks; it does not infer a removal from an error code |
| Cascade fitting, texel snapping, bias presets, cache keys | `shadow` | backends call it; they do not re-derive bias or fits |
| The frame state machine (idle → open → submitted) and its recovery contract | `ffi/src/lib.rs` `FrameState`/`FrameOwner`, documented beside `reconlBeginFrame` in `include/reconl/reconl.h` | backends answer one call at a time and never hold frame state |
| Handle lifetime and refcounting | `core/src/handle.rs` + the `ffi` handle blocks | no entry point frees or retains by hand |
| Render mechanism: targets, passes, present, GPU state | `backends/*` (the reference rasteriser is `raster`) | core and ffi know no platform or graphics type |
| The reference scene: geometry, light, camera, shadow spec, and the three shadow modes | `tools/host/src/scene.rs` | `reconl-diff` and `reconl-bench` both read it; neither carries a copy |
| The host side of a frame (create the resources, begin, encode, submit, present, time the boundaries) | `tools/host/src/frame.rs` | tools encode a frame; they do not re-implement the order |
| What a host allocator has been asked for, and what is still outstanding | `tools/host/src/alloc.rs` | tools read the ledger; the library is the only caller of it |

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
| The reference scene renders byte-for-byte what is committed | `reconl-diff compare` against `tests/golden/soft-cpu-shadow.png`, and independently `reconl-bench --png` compared to the same file (`tools/reconl-bench/tests/cli.rs`) |
| A probe creates no device and allocates nothing | `reconl-info --probe-only`, whose allocation ledger is checked in `tools/reconl-info/tests/cli.rs` |
| A device gives every byte it took back at release | `reconl-info`'s host ledger, checked in the same test |
| A run reports the configuration that produced it, and the per-frame record | `tools/reconl-bench/tests/cli.rs` |
| A failed ABI call leaves the device able to make the call the state machine names next | `ffi/tests/abi.rs` frame-state tests; the `framestate` C probe |
| A device only leaves its hardware for a genuine removal or a measured overload, and a refused argument is neither | `ffi/tests/tiers.rs::a_refused_call_does_not_offload_the_device` (fails with a code-based trigger); the `offload` C probe |
| A driver failure's code is the HRESULT's own class, decided in one place | `backends/d3d11/src/imp.rs::classify_hresult` (the only HRESULT→code mapping); `ffi/tests/tiers.rs::a_driver_rejected_frame_reports_an_argument_error_not_a_loss` (fails under a blanket `DEVICE_LOST`); the `offload` C probe's arm 4 asserts `-1`, not `-6`, against the shipped DLL |
| An offloaded frame is the reference tier's frame | the calibration-frame comparison in `ffi/tests/tiers.rs`; `--png` from the `--repeat=200` fault run `cmp`-compared with a `--backend=soft-cpu` run |
| A calibration is decided by the measurement it took, and the host can read that measurement afterwards | `ffi/tests/tiers.rs::an_overloaded_hardware_tier_offloads_and_the_measurement_decides` (fails with the comparison inverted); `reconl-bench`'s `final` line; the `offload` C probe |

## Verification surfaces

- `cargo test --workspace` — unit tests, the in-process ABI host, the tier
  matrix, the golden compare, and the CLI tests that spawn `reconl-info` and
  `reconl-bench` as processes.
- `reconl-diff render <png>` / `compare <png> [--tolerance=N]` — the real tool,
  through the same ABI a host uses.
- `reconl-info` / `reconl-bench` — the probe, the limit and ledger report, and
  the timed run with its fingerprint and trace. These run in their own process,
  which is what makes a claim about the process-wide allocation ledger
  measurable rather than racy.
- The C host probes (`gpu_probe`, `hostile*`, `framestate`, `shadowconfig`,
  `offload`)
  drive the *shipped DLL* — the only place `reconlConfigureShadows` and the
  cross-ABI recovery paths are exercised as a C host would. They currently live
  outside the repository, in the build machine's temp directory, so they do not
  guard anything in CI yet.
