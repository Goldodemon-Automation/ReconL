# `probes/` — the C probes

ReconL's claims about behaviour are claims about the **shipped library**, so the
evidence for them is a C program driving the exported functions of the built
DLL/SO — not a Rust test reaching the same code in-process. These are those
programs, and `run.sh` is how they are built and run.

```sh
probes/run.sh                 # build the library, build every probe, run, summarise
probes/run.sh fghostile       # only the named probes (the library is still built)
```

Exit status is 0 when every row passes, 1 when a row fails, 2 when the library or
a probe does not build.

## What the runner does

1. `cargo build --release -p reconl-ffi` — the artifact, not a test profile.
2. Copies the built `reconl.dll` / `libreconl.so` into `probes/.build/` and
   **md5-confirms the copy against the built file**, then prints that md5 with
   the summary. Every probe is linked against that copy, so a probe result is
   always traceable to the exact binary that produced it.
3. Compiles each probe from `probes/src/*.c` against `include/reconl/reconl.h`
   and the copied library — linked directly against the DLL/SO, so no import
   library or `.def` file has to be kept in step with it.
4. Runs the plan in `run.sh` (one row per probe invocation, with the backend
   arguments that matter) and prints one summary:

   ```
   probe      arguments                          checks  finds   verdict
   fghostile  --backend=soft-cpu                     32      0   PASS
   ...
   24 rows: 24 pass, 0 fail, 0 skip
   ```

   Each row's full output is left in `probes/.build/<row>.out`, so a red row is
   diagnosable without re-running anything.

A row is red when the probe exits non-zero, when its output contains a `FAIL` or
`FIND` marker, when the check count it prints does not match the expected count,
or when a required line is missing. The expected counts are pinned per row on
purpose: a probe that silently stops running its checks cannot pass by printing
nothing.

Rows that name `--backend=d3d11` (and `overtarget`, which is a hardware measurement)
are skipped when `gpu_probe` reports no usable D3D11 device on the host, the same
way the in-repo tests skip their d3d11 legs. Everything else is expected to hold on
any host.

## The probes

Each is a read-only audit of a surface, in the order they appear in the plan.
"Checks" is what the row prints today.

| probe | what it is for |
|---|---|
| `fghostile` (32) | frame generation under hostile conditions on both tiers: a malformed descriptor, a wrong-kind handle, a foreign device's swapchain, the history's allocation balance over 20 create/keep/release cycles |
| `framegen` (21) | the feature as a host uses it: a per-frame toggle, the same scene with it off byte-for-byte, a generated image warped along real camera motion, counters that count it only in `ReconLStats.framegen` |
| `framestate` (43) | the frame state machine and its recovery contract, including that an over-ceiling frame is refused before the driver and the next frame still renders |
| `gpu_probe` | the backend/tier probe: what each backend reports, a frame that is not just the clear colour, determinism across two renders, and the camera-extension `struct_size` paths |
| `hostile` (11) | absurd requests that must not take the process down: absurd sizes, orphaned swapchains, null buffers |
| `hostile2` (13) | zero-sized and zero-byte descriptors: refused, never coerced into something that allocates |
| `hostile3` (11) | the size ceiling and the refusal codes it produces, and that an uncapped device reports a concrete ceiling |
| `hostile4` | the recovery codes after a failed present and a failed begin, on both tiers |
| `hostile5` (16) | malformed push-constant and draw commands: the frame completes without taking the process down |
| `ladder` (6) | the tier ladder as the host sees it: every entry carries a reason and the measurement behind it, and the documented step order in `docs/offload.md` |
| `narrowpitch` (11) | the row-pitch contract on every readback route and all three tiers: a pitch narrower than a row is refused and nothing is written |
| `offload` (33) | the offload: fault failover, the calibration, the return to hardware, and that a refused frame takes no safe path |
| `onewriter` (8) | one decider and one record for tier changes: every change the device made is in the ring the host reads, in order |
| `overtarget` | the over-target path on real hardware: 12 frames at 1920x1080 with the smallest target, and the host-visible composed frame cost that drove each decision |
| `shadowconfig` (40) | `reconlConfigureShadows` and the per-frame override: filters, cascade counts and map sizes as a host sets them |
| `viewport` | `reconlCmdBeginPass`'s viewport: the render is confined to the rect asked for and no lit pixel lands outside it |
| `fgstate` | the answer order of both present paths, cell by cell: 90 cells per tier, each on a fresh rig, recording the code, whether the refusal wrote into the host's buffer, and whether the device was still usable |

## Notes

- **A freed handle is not a testable input.** An earlier revision of `fghostile`
  passed a swapchain to `reconlPresentGenerated` *after* releasing it and asserted
  a code. That asks the library to read memory its host has handed back, and it
  segfaulted about one run in three on d3d11. The ABI's contract is ref-counted
  lifetime, not "a stale pointer is refused"; the cell now uses a live handle of
  the wrong kind, which tests the same property — the handle is validated by its
  kind word and never followed — without undefined behaviour. The lesson is the
  reason this suite is in the repo: an unrun-once probe records a lucky run.
- The probes were previously kept in a scratch directory next to a hand-copied
  DLL, and a stale copy there once made a pass report on a library it had not
  built. That directory is not part of the project; this is. The one-off programs
  that stayed behind (`probe…probe6`, `dbg2`, `ringwrap`, `fgnull`) were each
  written to pin down a single defect while it was being fixed, and each finding
  now lives in `ffi/tests/` — what is tracked here is the set that stands on its
  own as a check.
- `probes/.build/` is build output and is ignored.
