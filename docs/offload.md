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
Gpu ──(over target, this plan already measured, cpu won)───► Offloaded
                                        no calibration: the comparison is filed
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
* **The ladder quits, and how many frames that takes is not part of the
  contract.** One calibration per plan, at most one return, and a re-offload that
  reuses the measurement instead of paying for a second one: that bounds the
  backend changes at three per plan, whatever the schedule. *When* they land is a
  property of the host - the return waits for a reference-tier frame inside the
  target, and the re-offload needs a rebuilt hardware device's first frame
  outside it - so this rule is proven by driving the device until frames stop
  changing and asserting the bound, not by sampling a fixed window and calling
  the sample the answer. A six-frame window, for instance, can land a run's
  return trip and the offload after it on its last two frames, which is one
  round trip - exactly what the rule promises - and not a violation of it.
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
5. **Two layers, one number.** A device has two responses to a frame that missed
   the target, and both stay:

   * the **device's own relabel** (`apply_tier` in `ffi/src/offload.rs`, applied
     by whichever backend is live: `T1 -> T2` on hardware, `T2 -> T3` on the
     reference tier), which keeps the same backend and changes only that
     device's quality tier, and
   * the **host-level offload** above, which changes the backend.

   The relabel is decided first, from the run as it stood *before* this frame's
   own cost, and it is applied to the backend that is live when the frame closes -
   the one being replaced, if the offload that follows changes it. The change is
   recorded *after* the relabel the same frame earned, and the incoming backend
   starts at the tier the device stands at rather than at the tier it started
   from. For a host that opted out of backend changes the relabel is the only
   response there is, and it is the response it always had: one step down, on the
   frame after the frame that missed.

   They used to answer "was this frame over target?" from two different numbers:
   the offload read the frame the host waited for, while the relabel read the
   device's own render pass, which cannot include the readback a
   memory-presenting host blocks on. At 1080p that copy is most of the frame
   (measured: a 0.9 ms pass inside a 9.7 ms frame on the hardware, a 6-10 ms pass
   inside a 16-19 ms frame on the reference tier), so the two could disagree
   about the same frame. Measured through the shipped DLL before the fix: at
   1920x1080 with a 14 ms target, every frame the host read cost 17.1-19.2 ms -
   over the target on all four - and the device sat still, because the number it
   judged was inside.

   Now the boundary that made the host wait composes the complete cost, and both
   layers judge that one number: the number a host compares
   (`ReconLFrameTiming.total_ns` against its own `target_frame_ms`) and the number
   the device acts on are the same one, and they are the same run of frames - one
   `FrameLadder` on the device, one threshold, one counter (`core/src/tier.rs`).
   Which layer acted is still visible in the ring: a relabel's detail says "this
   device's own tier", an offload's carries the backend comparison. Both carry the
   *cost they were decided from* - the relabel's reads `N frames over the M ms
   target the host read (X ns)` - so a host reading the ring can see the number the
   ladder acted on, not only that it acted. `FrameLadder::last_over_ns` is that
   number, held once with the run it belongs to.

   That is what makes the number assertion exact rather than inferred: the entry's
   cost is the frame the host read for the frame the entry names, character for
   character, on any machine under any load. A device-side ladder judging its own
   render pass writes a *smaller* number there (measured here: a 0.03-0.26 ms
   empty 1080p pass against a 4-16 ms host-visible frame), so the arithmetic fails
   instead of the pin having to place a target near the gap between the two - which
   is what made this pin a measurement of the machine's load.

   The relabel does not consult `allow_downgrade`, and does not need to: that
   bit refuses *backend* changes (`RECONL_ALLOW_DOWNGRADE_TIER`), and a relabel
   stays on the same backend. A host that opted out still never sees its backend
   move, which the probe's opt-out arm pins.

   Neither layer ever judges a device's own render pass. The relabel has nothing
   to act on until one over-target frame has been counted, so a device's first
   frame - and the first frame an incoming backend renders - steps nothing: a host
   sees the first tier change on the frame after the first miss. The run both
   layers answer from is the device's own, over the costs it composed in frame
   order, and a backend change does not restart it: the frames a rebuilt backend
   presents were paid for by the device it replaced.

   There is no `frames_over_target` counter in `ReconLStats` and no plan to add
   one: the run is per-device state, held once (`FrameLadder` on the device), and
   a copy in the host-visible counters would be a second definition. What a host
   reads is the consequence: `tier`, `tier_reason` and the downgrade ring.

   **Both layers are decided in one place, and one log records them.**
   `apply_tier_policy` (`ffi/src/offload.rs`) is the only thing that decides a
   tier change, and `DeviceHandle::downgrades` is the only thing that records one:
   the offloads and returns that change the backend, the relabels that change the
   tier a backend renders at, and the host's own tier requests all append there, in
   the order they happened. A backend used to keep its own ladder - its own
   threshold, its own counter, its own ring - in each backend, so the same event
   class had two writers and the host-visible ring was a concatenation of the
   device's entries and *whichever backend happened to be live*: an entry a host
   had already read disappeared the moment the frames moved to another backend
   (measured through the shipped DLL: the reference tier's relabel during a
   calibration frame, read at frame 1, absent from the ring at frame 2, and the
   `RECOVERY` entry that replaced it named a `from` tier no surviving entry had
   ever produced). A backend now applies the tier it is told and records nothing.
   `ReconLStats.downgrade_count` is every change this device has made - the
   header's "total ever, not just in the ring" - while `safe_path_events` remains
   the count of *backend* changes, as it always was.

   Which frame a tier change is visible on is part of what a host can depend on,
   so it is measured through the shipped DLL and pinned rather than left to
   whichever layer happens to run first: with the offload opted out and a 1 ms
   target, frame 0 reads `T1/gpu-shared` with an empty ring and frame 1 reads
   `T2/cpu-ram` with one entry whose `frame_index` is 1
   (`ffi/tests/tiers.rs::the_ladder_charges_a_relabel_to_the_frame_after_the_cost`).
   The relabel acts on the frame after the cost that armed it, one frame behind
   the offload, which acts on the frame it just closed - the order the backends'
   own ladders kept before both layers shared the device's run.

## How it is proven

* **`ffi/tests/tiers.rs`** (every test of it runs through the public ABI):
  * `the_ladder_charges_a_relabel_to_the_frame_after_the_cost` - the schedule
    above, pinned per frame: with the offload opted out and a 1 ms target, frame 0
    must read tier 1 with an empty ring and frame 1 must carry exactly one entry,
    a relabel whose `from`/`to`/`frame_index` are `(1, 2, 1)`. **Revert-checked**:
    with the relabel answering from the frame's own observation (the schedule the
    module split left behind, and what this pass restored), it fails on "the first
    frame stepped a tier the run had not armed yet (entries 1)".
  * `a_plan_is_calibrated_once_however_often_it_offloads` - the "one calibration
    per plan" rule, run on a shadowed plan key and on the single triangle. This
    one had lost its `#[test]` attribute and had never run; the schedule pin above
    is what surfaced it, and it passes now that it does.
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
* **The ladder judges the host's number, and only that number.**
  `ffi/tests/abi.rs::the_device_ladder_judges_the_frame_the_host_reads` drives
  1080p frames at a 1 ms target with the offload opted out (so the relabel is the
  only response and the hardware stays the renderer), then reads the cost its first
  ring entry recorded and requires it to equal the `ReconLFrameTiming.total_ns` the
  host read for that frame. Two revert-checks, both verified: with the policy
  judging `frame_numbers()` the pin fails on `a frame over target must step the
  device's own tier` (that number is under the target on this host), and with the
  policy judging a floored device number - so the ladder still steps - it fails on
  the arithmetic, `the ladder must judge the frame the host read (15781100 ns), not
  the device's own render time: ... the host read (5000000 ns)`.

  The pin used to place its target inside the measured gap between the two
  numbers, which made it a measurement of how loaded the machine was: the statistic
  it derived from drifted across a millisecond while the run's own frames did not,
  and it flaked under parallel test execution. It no longer measures anything to
  find its window - it reads the number the ladder recorded - so the same clause is
  pinned by arithmetic instead of by placement.
  `core/src/tier.rs`'s `FrameLadder` tests are the arithmetic underneath it: the
  frame the run reaches the threshold on, a frame exactly *at* the target counting
  as inside it, a ladder with no target or no threshold never acting, and the
  question the relabel answers - `acting()` - being false until the frame that
  arms the run has been counted and false again after a frame inside the target. The two tests that used to pin this in
  the reference backend - `the_tier_ladder_judges_the_frame_the_host_waited_for`
  and `the_tier_ladder_ignores_a_number_the_host_did_not_read` - are gone with the
  backend's own ladder, because a backend no longer judges anything: the number a
  pass cannot measure (the readback a host waited for) is no longer handed to one,
  and `backends/soft-cpu/tests/render.rs::the_reference_backend_does_not_decide_its_own_tier`
  pins that it stays where the device put it.
* **A declined offload does not lose the ladder.**
  `a_host_that_forbids_tier_changes_keeps_its_hardware` also asserts the other
  half of the opt-out: the device's frames miss the target, its backend does not
  move, and its *own* tier steps down by one with `FRAME_TIME_OVER_TARGET` in the
  ring and a detail naming it as the device's own tier - the response a host that
  refuses backend changes still gets.
* **The ring is a history.**
  `the_tier_log_holds_every_change_however_the_backend_moved` drives a leg that
  offloads, returns and offloads again, and asserts that every entry a host read
  at any frame is still readable in the last frame's ring, that the frame each
  entry carries never goes backwards, and that `downgrade_count` never decreases.
  **Revert-checked**: with the log split back into a device log concatenated with
  a backend ring that dies with the backend, it fails on `frame 1 read an entry
  that is gone by the last frame` - the reference tier's relabel during the
  calibration frame, which the host read and which then vanished.
* **An offloaded frame is the reference tier's frame, byte for byte.** The
  calibration frame is compared against the same frame rendered directly on a
  reference-tier device (in the test), and the `--repeat=200` fault frame is
  `cmp`-identical to a `--backend=soft-cpu` run of the same scene.* **`%TEMP%\rlprobe\offload.c`** (0 failures against the rebuilt DLL): the
  opt-out, the measured offload, the no-miss arm, and the refused-frame arm that
  shows `submit -> -6` (`DEVICE_LOST`) with the device still on the hardware.
  Its arm 2 renders the empty-pass frame, where the two tiers measure within a
  few percent of each other (5.0 ms against 4.4 ms on this machine), so
  successive runs of the same binary have produced **both** outcomes - the device
  came back in one run, and stayed on the reference tier in the next. That is the
  one place the *stay* direction is exercised through the real ABI. That arm now
  derives its target from the running machine instead of hard-coding a millisecond
  (the smallest whole millisecond the reference tier's own cheapest measured frame
  is inside, with the measurement printed) and drives the ladder until three
  frames change no backend, printing every change it saw. Its checks are the
  contract's, not a schedule's: at most one return, at most three backend
  changes, three consecutive quiet frames at the end, exactly one calibration in
  the ring, and any offload after the first recording that it reused the
  measurement. Measured: 20 consecutive runs at 0 failures and 0 skips, where the
  arm it replaces failed 3 of 16 on the same machine for a reason that was its
  own window (`the ladder settled: no backend change in the last two frames`) and
  not the rule - each of those runs had offloaded, returned, and re-offloaded,
  i.e. one return.
* **`ffi/tests/tiers.rs::a_plan_is_calibrated_once_however_often_it_offloads`**
  pins the half of that rule 256x256 cannot reach: at that size the reference
  tier always *loses* the calibration, so the device goes straight back to the
  hardware and never offloads twice at a plan. The sequence it pins needs a plan
  where the reference tier *wins* the calibration, the settle window buys the
  device a return trip, and the frame that misses after that return reuses the
  measurement.

  It runs that sequence on **two plan keys** - the occluder scene at 32x32 with a
  2 MiB shadow-map budget (enabled, two cascades, PCF, so the shadow work and the
  plan key are both real) and a single triangle with no shadows - because the
  plan a comparison is filed under is the resolution and the shadow configuration
  together, and a rule that held on one key and not the other would be a defect
  in exactly the dimension the memory is keyed on.

  Both ladder legs - this one and the 256x256 leg above - share one harness and
  one derivation (`LadderWindow`/`LadderRun` in `ffi/tests/tiers.rs`). The target
  is a measurement, so a second copy of the derivation is a second opinion about
  the same number, and the rules the legs assert (the bound, the ring, the quiet
  tail) exist once. The hardware has to *miss* the target, and where the
  reference tier is the faster tier its frames have to be *inside* it - so the
  target is the ABI's smallest whole millisecond (which every hardware frame
  measured here was over), raised to cover the reference tier's *typical* warm
  frame **only when a fresh hardware device's first frame is over twice the
  raised value**. The doubling is deliberate: that first frame is a device setup
  frame whose cost varies by more than 2x between one cold device and the next on
  a loaded machine, so a target only just under it is not a target for the device
  the leg then builds (measured failure under load: a derived 34 ms target with
  the run's device presenting its first frame in 20 ms). Typical and not cheapest,
  for the raised value: the settle window arms on an ordinary frame. A cheap
  scene at 32x32 keeps that window wide - the reference tier renders in tens of
  microseconds there while a hardware device's first frame is two to three
  milliseconds - and a host with no usable whole millisecond fails with both
  measurements instead of passing.

  The test asserts rather than skips: a run that does not reach the second
  offload fails with the ring, the counts and both measurements - so a regression
  reintroducing a second calibration cannot pass on an unexercised branch.
  Measured after the consolidation: **8/8** runs of the pin alone and **5/5**
  full-file runs at `--test-threads=2` (the loaded configuration that exposed the
  old 256x256 derivation, which capped the target with one pilot device's first
  frame while the run builds its own device). **Revert-checked**: with the
  already-measured branch disabled, the *shadowed* leg fails on `offload 2 of
  this plan paid for a second calibration`.
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
