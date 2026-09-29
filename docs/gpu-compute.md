# The `gpu-compute` backend

**Status: the device layer is built; the compute raster is not.** What that
sentence means precisely, and why it is written that way, is the whole of this
document.

## What it is

ReconL already has a hardware tier: `d3d11`. This is a second one, and the
difference is what the GPU is asked to be. D3D11 asks it to be a graphics
pipeline - fixed-function raster, a state object per draw, a swapchain, and a
driver that owns the frame. The compute backend asks it to be an array of
arithmetic units, because the reference tier's pixel rules are fixed-point edges
and a top-left fill test, and those are expressible as kernels that look like the
CPU ones rather than as a pipeline state object.

| runtime | vendor | opened from | 
|---|---|---|
| CUDA driver API | NVIDIA | `nvcuda.dll`, `libcuda.so.1` |
| ROCm / HIP runtime | AMD | `amdhip64.dll`, `libamdhip64.so` |

One ABI id covers both (`RECONL_BACKEND_GPU_COMPUTE = 10`) because the tier
question is the same one either way - T0 `gpu-discrete` or T1 `gpu-shared` - and
which vendor answered is reported in the device's name and driver strings rather
than in its identity. A host that wants a specific vendor says so in
`ReconLComputeDesc::vendor_preference`.

## Why neither stack is a build dependency

Neither driver is present on the machine this repository is developed on (Intel
UHD, no discrete GPU at all - see the measurements in `README.md`). Requiring the
CUDA toolkit or a ROCm install at build time would make the library fail to
*link* there, and would make `cargo test --workspace` unrunnable for anyone
without a GPU SDK.

So both are opened at run time, the way the Windows build already opens DXCore:
name the library, resolve the symbols, and let a missing library be an ordinary
`Err`. `backends/gpu-compute/src/loader.rs` is that mechanism and it is
deliberately small - `LoadLibraryA`/`GetProcAddress` on Windows,
`dlopen`/`dlsym` elsewhere - because it is the part that must not be clever.

A library that opens but does not export every entry point the backend uses is
dropped rather than half-loaded: `api::cuda::Api::load` requires all 20 symbols
it runs on, so a device is never built on a partial API and then fails inside a
frame, where the failure would be a lost frame instead of a refusal.

## What is built

* **Discovery.** `discovery.rs` opens CUDA, then ROCm, and the first runtime that
  loads *and reports at least one device* wins. A driver with nothing behind it
  does not shadow the other vendor's. The result is cached once per process
  (`OnceLock`), so the answer a probe put in `ReconLProbeInfo` and the answer a
  device creation acts on are the same enumeration rather than two looks at a
  driver that could have changed between them.
* **Enumeration.** `reconlEnumerateAdapters(RECONL_BACKEND_GPU_COMPUTE, ...)`
  answers on every machine, including one with no vendor driver, where the
  answer is an empty list rather than an error or an invented device. Each entry
  reports the driver's own device name, its total memory, the adapter type the
  driver declared (`UNKNOWN` when it declares none), and whether a compute
  context can be created on it.
* **A measured `usable`.** The adapter's `usable` is not inferred from
  enumeration. `Runtime::context_works` asks the driver to open a context on the
  device and hand it straight back, and a device whose context refuses is
  reported unusable with the driver's own words as the note. That is the same
  rule the D3D11 path follows, where `usable` means feature level 11.0 can be
  created rather than that an adapter was listed.
* **The tier rule.** `tier_for` in `lib.rs`: a driver that classified the device
  is believed over its size (a 96 GiB integrated part is still shared, a 1 GiB
  discrete part is still discrete); an unclassified device - ROCm on a release
  that does not answer the attribute, or an old driver - falls back to total
  memory, with the threshold deliberately low rather than a model of any product,
  because the ladder steps a wrong guess down within a few frames and a wrong
  guess the *other* way would strand a capable device on T1.
* **The ABI.** `RECONL_BACKEND_GPU_COMPUTE`, `ReconLComputeDesc`
  (`reconl_backends.h`), enumerable adapters, a probe row that names the device
  and carries the measured reason it is or is not usable, and
  `reconlBackendName(10) == "gpu-compute"`. The layout and every constant are
  pinned against the shipped header by `ffi/tests/abi_layout.rs` (1118 asserts
  before this work; the new struct and enumerators join them), and
  `ffi/tests/abi.rs` pins the three behaviours a host depends on: the id is
  named, enumeration answers with and without a driver, and a host that asks for
  the backend gets a classified refusal rather than software pixels.

## What is not built, and the one line it waits on

The compute raster: the kernels that would fill tiles on the device, and the
host-side dispatch that would hand them a frame.

Until they exist, this backend **enumerates but does not render**, and the ABI
says so rather than papering over it:

* `RENDER_PATH_BUILT` in `backends/gpu-compute/src/lib.rs` is `false`, and it is
  a named constant rather than an implicit state so that the one line which
  flips it is the one place every report follows it: the probe's `usable`, each
  adapter's `usable`, and `device_support`.
* `device_support()` is what `reconlCreateDevice` calls, and it is deliberately
  the same question the probe's row answers, asked once. A probe that said
  "usable" and a creation that then refused is exactly the disagreement the ABI
  exists to prevent.
* `reconlCreateDevice(backend_hint = GPU_COMPUTE)` returns
  `RECONL_ERR_BACKEND_UNAVAILABLE` when the machine has no vendor driver, and
  `RECONL_ERR_NOT_SUPPORTED` when it has one but this release cannot render on
  it. Both are existing codes; nothing was invented for this. What it never does
  is fall back to software, which is the rule the whole backend matrix already
  follows (`ffi/src/lib.rs`, "declared in the ABI but not built in this
  release").
* The automatic backend choice does not consider this backend a GPU until it can
  render: `reconlCreateDevice` with `backend_hint = NONE` picks it only once
  `device_support()` is `Ok`. A backend that enumerates a GPU and then refuses
  every device must not become the default on a machine that has one.

On this machine the probe row reads:

```
  gpu-compute no     T1/gpu-shared             —      1.0 GiB        3      12.0 MiB  no CUDA or ROCm device
              caps: none
              note: no CUDA driver (nvcuda.dll) and no ROCm runtime (amdhip64.dll) on this machine
```

On a machine with one, the same row names the device the driver reported and
says which of the two things is missing:

```
              note: the driver is present and a context opens, but this build has no compute raster path: a device would not render
```

## What the raster has to reproduce, and how it will be checked

The reference tier is the definition of correct (`README.md`), so a compute
backend is held to the same standard as D3D11 and its evidence has to be the same
kind:

* **The same passes.** Shadow passes first (cull front, reversed-Z, clear depth
  0.0), then the colour pass into the frame's targets, with the fits computed
  from the same `reconl-shadow` code the reference uses, so the cascade matrices
  stay bit-identical across tiers.
* **The same pixels, within the same measured tolerance.** `reconl-diff compare
  tests/golden/soft-cpu-shadow.png` at `--tolerance=48` inside the 1% budget is
  the documented cross-tier setting, and it is a measurement rather than a
  choice: `d3d11` differs from the golden on 14 of 4096 pixels at 64x64 and 761
  of 262144 at 512x512, worst channel delta 46, all within 4 px of a pixel the
  shadow changes. A compute backend joins that comparison or it does not ship.
* **Zero allocations per frame.** `docs/determinism.md` makes it a rule, and the
  measured state of the tree today is 0 per frame at every size on `soft-cpu`,
  `d3d11` and `null`. Device buffers and kernel parameter blocks are reserved
  between frames and cleared, not reallocated.
* **A probe that cannot over-promise.** The kernels have to compile at device
  creation, so a machine whose driver is too old for the kernel source is a
  classified refusal rather than a device that fails on its first frame.

The honest limit is stated rather than hidden: **no test in this repository can
exercise the kernel path**, because the machine it is developed on has no CUDA
and no ROCm device. That is why the raster is a separate, named piece of work
with its own evidence bar instead of a large unchecked blob in this release.
