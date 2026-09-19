# GPU offload: continuing on the reference tier

**Status: built.** The contract is in `include/reconl/reconl.h` next to the frame
calls and `RECONL_ALLOW_DOWNGRADE_TIER`; the state machine is
`DeviceHandle::apply_offload_policy` in `ffi/src/lib.rs`; the trigger's verdict is
`D3d11Device::device_removed`. This file records the design and the measurements
it was decided from, and says which parts are proven by what.

## The two triggers, and why they are different

**1. Fault failover.** A GPU call fails in a way that means the device is gone.
The GPU is not slow, it is *dead*, so this offload needs no comparison: the
reference tier is faster than a device that cannot render. Before this, it was a
dead end - `reconlSubmit` returned `RECONL_ERR_DEVICE_LOST` and no ABI call
continued the session:

```
reconlSubmit: RECONL_ERR_DEVICE_LOST (-6) — device reported: map colour staging
  texture: The GPU device instance has been suspended. (0x887A0005)
```

The frame that hit it is **re-rendered on the reference tier and presented
normally**, so a fault costs the frame's GPU time and not the frame. The recorded
draws are host memory - `FrameRecord` keeps them for exactly this reason - so the
second render is the same frame, not a re-recording of it.

**2. Measured overload.** Frames miss `target_frame_ms` for
`downgrade_after_frames` consecutive frames. The ladder already stepped a tier
here, but stepping `T1 -> T2` on a D3D11 device only relabels it: the same GPU
keeps rendering, with a CPU tier's cascade caps. The offload is what makes the
step real.

This trigger must **not** be taken on the assumption that the CPU is faster. On
this machine, at 64x64 and up, it is not:

| Resolution | soft-cpu (T2) | d3d11 (T1) | GPU advantage |
|---|---|---|---|
| 64x64 | 30.6 ms | 3.4 ms | 9.0x |
| 256x256 | 139.6 ms | 28.7 ms | 4.9x |
| 512x512 | 462.9 ms | 94.4 ms | 4.9x |

So the policy is **measure, then decide**: the first frame after the decision is
rendered on the CPU as a *calibration*, and the device offloads only if the CPU
was actually faster. One slow frame is the price of not making every subsequent
frame slow, and the result is remembered per plan so the calibration is never paid
twice for the same regression. (On the *empty-pass* frame the C probe renders, the
very same rule sends the device the other way: the reference tier measures faster
there than the hardware's fixed per-frame cost, so it stays. Both directions of
the rule are exercised - see the test plan.)

## The state machine

```
Gpu ──(over target N frames)──► Calibrating ──(cpu < gpu)──► Offloaded
                                    └────(cpu >= gpu)──────► Gpu
                                                             │ remember cpu_ns
                                                             │ for this plan
Gpu ──(genuine device removal)─────────────────────────────► Offloaded
Offloaded ──(within target M frames)──► Recovering ──(ok)──► Gpu
                                            └─(misses)─────► Offloaded
```

* **Calibrating** renders exactly one frame on the CPU backend. `Calibrating` and
  `Offloaded` are the same backend state: the frame that decides is the first
  frame the reference tier renders, so the calibration costs no extra frame.
  Every backend change - the offload, the calibration's return - increments
  `ReconLStats.safe_path_events` and writes one `ReconLDowngrade`.
* **A result is filed under a plan**: resolution, cascade count, shadow filter
  and texel budget. A stored comparison expires when any of them changes, because
  all of them change what a tier costs. No invalidation logic, and nothing to
  forget to reset.
* **Coming back up.** Once offloaded, the device rebuilds the hardware backend
  after `M` consecutive frames inside `target_frame_ms` (_M_ is
  `downgrade_after_frames`, the same threshold the ladder uses, so the two cannot
  disagree), and tries it again. That is attempted **at most once per plan**: a
  workload that oscillates around the target must not thrash, and "the hardware
  was tried again and missed" is what "this frame does not fit the GPU" means.
  A return trip is logged with `RECONL_TIER_REASON_RECOVERY`, which is why that
  reason exists - `STARTUP_PROBE` cannot say "the device recovered".
* **Two faults and the return trip stops.** A rebuilt backend that dies again is
  not returned to; a broken driver must not start a rebuild loop.
* With `target_frame_ms = 0` there is no window to settle in, so a
  fault-offloaded device stays on the reference tier. The C probe's
  `--target-ms` and this project's own `reconl-bench` default are both 0, which is
  why the `--repeat=200` repro ends (correctly) on the CPU.

## The trigger has to be a genuine removal

Every driver failure the d3d11 backend raises passes through one classifier
(`classify_hresult` in `backends/d3d11/src/imp.rs`), so the code a host sees is
the HRESULT's own class: the DXGI removal family is `DEVICE_LOST`,
`E_OUTOFMEMORY` is `OUT_OF_MEMORY`, `E_INVALIDARG` is `INVALID_ARGUMENT`,
`E_NOTIMPL`/`E_NOINTERFACE` are `NOT_SUPPORTED`, anything unrecognised is
`BACKEND_UNAVAILABLE`. The host-facing advice per class is documented next to
the `ReconLResult` enum in `include/reconl/reconl.h`.

The offload trigger needs *both* readings before it moves a device: the
failure's classified code must be `DeviceLost` **and**
`ID3D11Device::GetDeviceRemovedReason` must return a removal reason - two
readings of the same truth, either of which alone could be wrong. A frame the
driver refuses (32768 wide, past the 16384-texel limit, answered
`E_INVALIDARG`) is pinned end to end: the host gets `-1` and the device stays on
the hardware (`ffi/tests/tiers.rs::a_driver_rejected_frame_reports_an_argument_error_not_a_loss`,
which fails under the old blanket mapping, and the `offload` C probe's arm 4
against the shipped DLL).

## What exists

1. **A second backend, built from the device's own config.** `Origin` keeps the
   `SoftCpuConfig` and the `D3d11Config` the device was created from, so a backend
   can be rebuilt at any later frame - by then the host's descriptor is long gone.
   A device created on the reference tier carries no hardware config and never
   offloads.
2. **Resource re-creation for the return trip** needs none: the ffi's draw records
   point into the host's own buffer bytes, and both backends render from them.
   The ffi keeps no registry of live handles, and the offload does not need one.
3. **Counted, logged, visible.** `ReconLStats.backend`/`tier`/`tier_reason`,
   `safe_path_events` per backend change, and one `ReconLDowngrade` per change with
   its reason and the measurement that caused it, in the detail string. The
   measurement lives in a *change*, so a device that comes back records both costs
   ("the reference tier measured X against the hardware's Y") and a device that
   stays records the cost it was compared against in the offload entry that
   started the calibration ("the hardware measured Y ns"). Neither direction
   leaves the host guessing, and neither needs a new ABI field: `reconlGetStats`
   writes `ReconLStats` whole, so an appended field would overwrite the memory of
   a host compiled against an older header.
4. **Opt-out.** A host that did not set `RECONL_ALLOW_DOWNGRADE_TIER` gets the
   behaviour it always had. No new flag: the bit already means "this device may
   move down the ladder".
5. **Not the only tier policy.** The hardware backend still relabels *itself* for
   frame time (`backends/d3d11/src/imp.rs`, "telemetry only") without consulting
   `allow_downgrade`, so a device under a 1 ms target logs a `T1 -> T2` relabel
   before the offload logs its own entry. That duplication is the audit's separate
   finding and is untouched here.

## How it is proven

* **`ffi/tests/tiers.rs`** (four tests, all through the public ABI):
  * `an_overloaded_hardware_tier_offloads_and_the_measurement_decides` - a
    hardware device with a 1 ms target and a 1-frame threshold hands the next
    frame to the reference tier, that frame is the calibration, and the decision
    matches the comparison the policy itself recorded (`cpu_ns >= gpu_ns` exactly
    when the device returns to the hardware, both numbers read from the entries it
    wrote rather than from two more devices this test measured and hoped the
    machine would schedule the same way). The recorded reference-tier cost is also
    held against that tier's own render of the same frame, so the number the
    policy acted on is a frame's cost and not an invention. The return is logged
    `RECOVERY` with "measured" in its detail, and four frames produce at most one
    offload and one return. **Revert-checked twice**: with `offload_on_overload`
    disabled the test fails on "a hardware frame over target did not offload", and
    with the calibration comparison inverted it fails on "the calibration decided
    against its own measurement". On this host the reference tier's own cost at
    256x256 (about 17 ms against the hardware's 6 ms) always sends the device
    back, so the *stay* direction is not reachable in this test here.
  * `a_generous_target_never_offloads` and
    `a_host_that_forbids_tier_changes_keeps_its_hardware` - the two ways the
    offload must *not* happen.
  * `a_refused_call_does_not_offload_the_device` - a frame whose targets the
    driver rejects (32768 wide, past D3D11's 16384 limit) and a present into a
    buffer too small for it both leave the device on the hardware with
    `safe_path_events == 0` and the frame dropped, as before. **This test fails
    with a code-based trigger** (verified by temporarily making the trigger
    `e.code == Code::DeviceLost`: the rejected frame then offloads, and the test
    fails on `assert_ne!(submit, RECONL_OK)`).
* **An offloaded frame is the reference tier's frame, byte for byte.** The
  calibration frame is compared against the same frame rendered directly on a
  reference-tier device (in the test), and the `--repeat=200` fault frame is
  `cmp`-identical to a `--backend=soft-cpu` run of the same scene.
* **`%TEMP%\rlprobe\offload.c`** (0 failures against the rebuilt DLL; 32 checks
  when the reference tier wins the calibration and 34 when it loses, because each
  outcome asserts its own branch): the opt-out, the measured offload, the no-miss
  arm, and the refused-frame arm that shows `submit -> -6` (`DEVICE_LOST`) with
  the device still on the hardware. Its arm 2 renders the empty-pass frame, where
  the two tiers measure within a few percent of each other (5.0 ms against
  4.4 ms on this machine), so successive runs of the same binary have produced
  **both** outcomes - the device came back in one run, and stayed on the
  reference tier in the next. That is the one place the *stay* direction is
  exercised through the real ABI.
* **The user-visible repro**:
  `reconl-bench --backend=d3d11 --resolution=512x512 --repeat=200 --frames=1`
  ends with `presented 1  dropped 0  failures 0` and `final backend soft-cpu  tier
  T2/cpu-ram  reason device removed (DXGI_ERROR_DEVICE_REMOVED)` instead of a dead
  session, and with `--png` that frame `cmp`s identical to a `--backend=soft-cpu`
  run of the same scene. `safe-path events` reads 0 there, and that is the
  benchmark's own doing rather than a missing count: it resets the device's
  counters after its warmup, and on this machine the GPU dies on the first warmup
  frame (frame 0), so the event lands before the reset the measured window starts
  from. The tier the run *finished* on cannot be reset away, which is why the
  report now prints it.
