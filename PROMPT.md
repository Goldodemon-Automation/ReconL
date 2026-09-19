# ReconL — Build Prompt

> A portable, embeddable 2D/3D renderer with one C ABI, many GPU backends, and a
> software path that degrades gracefully to CPU → RAM → DISK instead of failing.

**One-liner for the generator:** *Build ReconL — a renderer that never says no.
One C11 ABI, GPU backends (D3D11/D3D12, Vulkan, OpenGL, Metal, WebGPU) and a
tiled software rasterizer that spills out of VRAM into system RAM and then to
disk — with real shadows at every tier, down to a disk-cached cascade on a
graphics-less machine — so the same API, the same scene, and the same output
frame work from Rust, C, C++, Swift, Java, C#, JavaScript, and plain HTML.*

---

## 1. BRAND / VISUAL IDENTITY (REQUIRED)

- **Name:** ReconL (wordmark: `ReconL`, mixed case, no space before the `L`)
- **Mark:** a thin open arc sweeping over the wordmark, drawn to a single
  4-point sparkle/star at the lower left — "signal received, then reconciled
  into one frame". The mark must render at 16 px favicon size and at 512 px
  banner size without re-drawing.
- **Theme:** dark-first. Near Black background `#0B0D0C`, Deep Pine wordmark
  `#5F8A80`, muted Pine `#7FA79B` for secondary text and rules, Off-White
  `#F5F5F3` for body copy on dark.
- **Type:** headings **Outfit**, body/UI **Inter** (per the vault's default
  design system). For code, headers, docs and CLI art: a monospace stack
  (`ui-monospace, "Cascadia Mono", Consolas, monospace`).
- **Palette rule:** one palette per surface. The brand teal above is the
  exception for ReconL specifically (it is the logo's own colour); everything
  else stays monochrome — no purple, no orange-on-blue, no neon pairs.
- **Tone of voice:** measured, mechanism-first, evidence over adjectives. Never
  write a performance number without saying how it was read.

---

## 2. WHAT RECONL IS

ReconL is a **renderer**, not an engine and not a game framework. It owns:

- device/backend selection and capability negotiation,
- resource lifetime (buffers, textures, pipelines, render targets),
- **shadows** (see §7): shadow passes, cascade fitting, filters, shadow-map
  caching and their fallback policy,
- memory budgeting across VRAM → system RAM → disk,
- command submission and synchronisation,
- frame presentation to a surface *or* to a plain memory buffer.

It does **not** own: a scene graph, a physics world, an asset pipeline, a window
system, an input model, or a job scheduler above its own worker pool. Games,
tools, editors, CI harnesses and web pages bring those. It also does not decide
*which* lights exist — the host hands ReconL a light list and ReconL decides how
to shadow it, not whether a lamp is lit.

**The promise:** *the API never hard-fails because the machine is weak.* Every
missing capability resolves to a lower tier that still produces a correct frame,
and every downgrade is observable (logged, counted, queryable at runtime).

**Non-goals (v0.x):** ray tracing (**ray-traced shadows included** — shadow maps
are in scope, RT shadows are not), mesh shaders, video decode, VR runtimes,
mobile-first power tuning, and scripting layers.

---

## 3. THE CORE CONTRACT (THE ABI IS THE PRODUCT)

The C ABI is the compatibility surface. Every language binding is a *mechanical*
wrapper over it — if a binding needs hand-written logic to be useful, the ABI is
wrong and gets fixed first.

Rules, non-negotiable:

1. **C11, `extern "C"`, header-first.** One canonical header `reconl.h` plus
   `reconl_backends.h`. Everything is callable from C with no wrapper.
2. **Opaque, ref-counted handles** for every object
   (`ReconLDevice*`, `ReconLBuffer*`, `ReconLTexture*`, `ReconLPipeline*`),
   `reconlRetain` / `reconlRelease`, no global mutable singletons.
3. **No exceptions, no panics, no unwinding across the boundary.** Errors are
   returned as `ReconLResult` codes; the last error is retrievable with
   `reconlLastError()` and carries a message + source location.
4. **Versioned structs with `struct_size`** (Vulkan-style forward compatibility):
   every struct starts with
   `uint32_t struct_size; ReconLStructType type; const void* next;`
   so a newer library can be called from an older header and vice versa.
5. **Allocator injection.** `ReconLAllocator { void*(*alloc)(void*,size_t,size_t);
   void*(*realloc)(...); void(*free)(void*,void*); void* user; }` at device
   creation. ReconL never calls the C runtime allocator behind the host's back,
   and never allocates inside a frame loop.
6. **Explicit budget, never a guess.** `ReconLDeviceLimits` and
   `ReconLMemoryBudget` are queryable *before* the device is created
   (`reconlProbe()` enumerates backends with zero side effects). The host can
   cap VRAM/RAM/disk use and ReconL honours the cap or refuses to start with a
   clear code — it never silently exceeds it.
7. **Deterministic by rule, not by luck.** Given the same scene, same device
   tier and same seed, ReconL produces the same frame across backends and across
   runs. Rasterisation follows documented fill rules (top-left rule, fixed
   subpixel precision, defined tie-breaking), no wall-clock or frame-counter
   input into geometry, and the software tier is the *reference* implementation
   the GPU tiers are diffed against.
8. **Threading contract:** calls are either documented thread-safe or
   single-threaded-on-a-context. No "it's fine usually".

Sketch (shape, not final):

```c
typedef struct { uint32_t struct_size; ReconLStructType type; const void* next; } ReconLBase;

ReconLResult reconlProbe(ReconLProbeInfo* out_instances);            // no side effects
ReconLResult reconlCreateDevice(const ReconLDeviceDesc* desc, ReconLDevice** out);
ReconLResult reconlCreateSwapchain(ReconLDevice*, const ReconLSwapchainDesc*, ReconLSwapchain**);
ReconLResult reconlBeginFrame(ReconLDevice*, ReconLFrameDesc* /* in/out: chosen tier */);
ReconLResult reconlSubmit(ReconLDevice*, const ReconLCommandList*, ReconLFence*);
ReconLResult reconlPresent(ReconLDevice*, ReconLSwapchain*, ReconLPresentDesc*);
ReconLResult reconlGetStats(ReconLDevice*, ReconLStats*);            // tier, spill bytes, downgrades
```

**Language choice is swappable; the ABI is not.** This prompt assumes the core
implementation is **Rust** (`cdylib` + `staticlib`, `#[repr(C)]`, `unsafe`
confined to a thin `ffi` module, no TLS, panic strategy `abort` across FFI).
If you would rather have a C11 core (Vulkan-reference style), keep §3 exactly as
written and change only the implementation language — nothing else in this
prompt moves. Optionally use `A:\toolchain\zig` (`zig cc`) as the portable C
compiler for the C/C++ shims and the freestanding builds.

---

## 4. THE OFFLOAD LADDER (THE POINT OF THE PROJECT)

ReconL resolves the machine into **tiers**, picks the highest usable one at
startup, and can step *down* mid-session without losing the frame:

| Tier | Name | Resident where | Selected when | Frame path |
|---|---|---|---|---|
| **T0** | `gpu-discrete` | full VRAM | dedicated GPU, budget fits | hardware raster + compute |
| **T1** | `gpu-shared` | VRAM + system RAM spill | iGPU / shared memory / budget tight | hardware raster, streamed resources |
| **T2** | `cpu-ram` | RAM, multi-threaded | no usable GPU API, or budget below GPU floor | tiled software rasteriser, worker pool |
| **T3** | `cpu-thrifty` | RAM, capped | CPU can't hold target frame time | T2 with resolution scale, LOD, frozen caches |
| **T4** | `out-of-core` | RAM + **DISK** | working set exceeds RAM cap | T2/T3 with tile + mip streaming from a disk cache |

Requirements that make the ladder real:

- **Selection is explicit and logged.** `reconlGetStats()` reports the active
  tier, why it was chosen, and every downgrade with a reason string.
- **Downgrade triggers are telemetry-driven, not vibes:** over-budget allocation
  attempt, frame time over target for N consecutive frames, allocation failure
  / `DXGI_ERROR_DEVICE_REMOVED` / `VK_ERROR_DEVICE_LOST`, memory pressure from
  the OS, disk cache full.
- **The frame survives the step.** A device loss must not drop the frame: ReconL
  re-creates the tier below, re-uploads from the last-good resource snapshot, and
  continues. (If this cannot be done for a given case in v0.1, say so in the
  README instead of pretending — see §13.)
- **No hidden quality lie.** A downgrade changes quality; it must be reported and
  optional to allow (`allow_downgrade` bitmask), never silently substituted.
- **Disk is a cache, not a state store.** Deleting the spill directory at any
  time must cost performance and never correctness.

---

## 5. ARCHITECTURE / LAYOUT

```
reconl/
├── include/reconl/            # reconl.h, reconl_backends.h, reconl_version.h  (the ABI)
├── core/                      # tier resolver, budgets, handles, error model, stats, logging
├── scene/                     # render-graph-ish pass list, resource graph, barriers (backend-agnostic)
├── raster/                    # software rasteriser: tile binning, fixed-point edges, ref clip
├── shadow/                    # CSM fitting, shadow atlas, PCF/PCSS filters, static-cascade cache
├── resource/                  # upload staging, mip chain, streaming, spill/cache manager
├── backends/
│   ├── d3d11/  d3d12/  vulkan/  gl/  metal/  webgpu/  wasm-webgl2/
│   ├── soft-cpu/              # reference tier (T2) — SIMD (SSE2/AVX2/NEON/wasm-simd128)
│   └── null/                  # deterministic no-op for ABI/CI tests
├── ffi/                       # C ABI surface only: handles, validation, last-error, no logic
├── bindings/
│   ├── rust/  cpp/  swift/  java/  csharp/  node/  wasm-js/
│   └── c/                     # the generated/verified C example both ways
├── tools/                     # reconl-info (probe), reconl-bench (recorder), reconl-diff (golden images)
├── tests/                     # unit, golden-image, conformance, fuzz (cargo-fuzz / libFuzzer)
└── docs/                      # ABI notes, tier semantics, determinism rules, measurement protocol
```

Backend rule: **a backend implements the portable core, it does not define it.**
No backend-specific type may appear in `include/`. Feature gaps are declared in
`ReconLBackendCaps` and resolved by the tier resolver / fallback shaders.

---

## 6. BACKEND MATRIX

| Backend | Native API | Platforms | Tier | v0.1 |
|---|---|---|---|---|
| `d3d11` | D3D11 | Windows 7+ | T0/T1 | ✅ first |
| `d3d12` | D3D12 | Windows 10+ | T0/T1 | later |
| `vulkan` | Vulkan 1.1+ | Win/Linux/Android/macOS (MoltenVK) | T0/T1 | later |
| `gl` | OpenGL 3.3+ / GLES 3 | everything legacy | T1 | later |
| `metal` | Metal 2+ | macOS/iOS | T0 | later |
| `webgpu` | WebGPU | browser + native (wgpu-style) | T0/T1 | later |
| `wasm-webgl2` | WebGL2 | browser | T1 | later |
| `soft-cpu` | — | **everywhere incl. WASM** | T2/T3/T4 | ✅ first |
| `null` | — | every host | headless/CI | ✅ first |

Two backends ship in the first milestone on purpose: the **hardware one closest
to the dev machine** (D3D11) and the **reference one with no dependencies**
(`soft-cpu` + `null`). Everything else proves the abstraction is real; those two
prove the ladder is real.

---

## 7. SHADOWS (REQUIRED — AT EVERY TIER, FROM THE FIRST SLICE)

ReconL renders shadows. They are not a later phase, not a plugin, and not
conditional on having a GPU: a machine with no graphics hardware still gets
shadowed output, one cascade at a time, out of RAM and then off the disk.

**API shape** (same rules as §3 — versioned structs, no exceptions):

- `ReconLLight` — type (directional / spot / point), position/direction/cone/range,
  colour + intensity, `cast_shadow` flag, `shadow_priority`, `shadow_quality` hint.
- `ReconLShadowConfig` — cascade count (1–4), shadow texel budget **in bytes**,
  filter mode (`hard | pcf3x3 | pcf5x5 | pcss-lite`), max shadow distance with a
  documented blend to unshadowed beyond it, and `allow_disk_cache`.
- `reconlConfigureShadows(device, const ReconLShadowConfig*)` — callable at any
  time, applied at the next frame boundary; never mid-frame.

**Techniques, in implementation order:**

1. One directional light, one cascade, PCF 3×3. This is a milestone-1 feature,
   not a stretch goal.
2. Cascaded shadow maps (2–4 cascades, texel-snapped fits, blend band at the
   cascade boundary) for a single main directional light.
3. Spot lights as one perspective map; point lights as a cube map (or a
   documented per-face atlas cut when the texel budget is tight).
4. Static/dynamic geometry split: static geometry renders into a **cached**
   cascade that is only re-rendered when the light or the world revision changes.
5. `pcss-lite` contact-hardening approximation at T0/T1 only. It is an
   approximation and must never be described as "soft shadows" in docs or README.

**Non-negotiable correctness rules:**

- **No swimming, no popping.** Cascade fits snap to the shadow-map texel grid, so
  a moving camera never makes edges crawl. This is tested (§12), not eyeballed.
- **Bias is documented, not magical.** Normal-offset bias plus slope-scaled depth
  bias, with published per-tier defaults in the config and in `docs/`, so acne and
  peter-panning are tuned once and reproducibly rather than per debugging session.
- **Same rasteriser, same rules.** Depth-only shadow passes go through the same
  tile-binning core as the colour pass, so `soft-cpu` shadows obey §3's fill rules
  and stay the reference the GPU tiers are diffed against.
- **Determinism.** Fixed filter kernels, no stochastic sampling without a seed,
  and one depth encoding chosen once (reversed-Z everywhere, or documented per
  backend — decide in §14 and never mix).
- **Fail-safe, never crash.** A missing, stale or checksum-failed shadow map
  renders the scene *unshadowed*, counts the event, logs it once, and keeps the
  frame. Shadows are the one feature where "silently flatter" beats "loudly
  dead".

**Degradation ladder for shadows** — shadows exist at every tier; the tier decides
how good they are:

| Tier | Cascades | Map budget | Filter | Static caching | Fallback under pressure |
|---|---|---|---|---|---|
| T0 | 4 | full texel budget | `pcss-lite` | RAM | — |
| T1 | 3 | sized by the VRAM cap | `pcf5x5` | RAM | drop to 2 cascades |
| T2 | 2 | RAM-bounded, scaled with the scene | `pcf3x3` | RAM | 1 cascade, lower resolution |
| T3 | 1 | small (e.g. one 1024² map) | `pcf3x3` | RAM | freeze the cascade, refresh every N frames |
| T4 | 1 | RAM-capped; **maps spill to disk** | `pcf3x3` | **disk-backed** | shadowed-atlas region only, unshadowed outside |

**The disk angle — why shadows belong in this prompt:** a static cascade is a
build artifact, not a per-frame cost. Cache the rendered static cascade in the
§8 spill arena, keyed by (light hash, fitted cascade matrix, static geometry
revision, filter mode, resolution). A cold cache renders it, a warm cache `mmap`s
it, and a revision bump retires it in O(1) — the same world-revision
invalidation idea the sibling `GMS` repo already proved out. The payoff: on a
weak machine, correct shadows from the first presented frame, because they were
paid for last run.

---

## 8. DISK SPILL / OUT-OF-CORE DETAIL

- **What spills:** texture mips (least-detailed-demanded first), tile cache of
  rendered passes, uploaded vertex/index blocks, and the frame's intermediate
  targets when the pass list is larger than the RAM cap.
- **Where:** `spill_dir` from the device desc, else `RECONL_SPILL_DIR`, else
  `%LOCALAPPDATA%\ReconL\cache` / `$XDG_CACHE_HOME/reconl`. Never write outside
  it. Never write without the host opting in (`ReconL_ALLOW_DISK_SPILL`).
- **Format:** one append-only arena file + a compact index, `RCLS` magic, version
  header, per-entry `xxhash64` checksum, `mmap`'d reads, torn-entry recovery on
  open (a corrupt entry is dropped and re-fetched, never trusted).
- **Eviction:** LRU over tiles/mips, weighted by (cost to regenerate ÷ size),
  with a hard `max_disk_bytes` cap and a background compactor. Eviction must be
  interruptible and never run on the render thread.
- **Verification:** `reconl-bench --spill 0|1` must produce identical golden
  images with a cold cache, a warm cache, and a corrupt cache. That test *is* the
  feature.

---

## 9. BINDINGS — EXACT MECHANISM PER LANGUAGE

| Language | Mechanism | Notes |
|---|---|---|
| **C** | include the header | the reference consumer; keep `examples/c/triangle.c` compiling forever |
| **C++** | header-only RAII wrapper (`reconl.hpp`) | `reconl::Device`, no exceptions by default; opt-in `throw` layer |
| **Rust** | `reconl-sys` (`bindgen`-checked) + safe `reconl` crate | `unsafe` isolated; `Send`/`Sync` asserted only where the header promises it |
| **Swift** | SwiftPM `systemLibrary` module map + a C++ interop shim | hand-written `ReconL*.swift` wrappers with `deinit → reconlRelease` |
| **Java** | **Project Panama FFM (Java 21+)** — not JNI | `MemorySegment`, `Arena` for native lifetime; JNI kept only as a legacy fallback |
| **C#** | `[LibraryImport]` P/Invoke + `SafeHandle` | one NuGet package with `runtimes/{win-x64,linux-x64,osx-arm64}` natives |
| **JavaScript (Node)** | N-API addon (`node-addon-api`), prebuilt per-arch | no `node-gyp` at install time — ship prebuilds |
| **HTML / browser** | WASM build (`wasm32-unknown-unknown`) + WebGL2/WebGPU, else `soft-cpu` in WASM threads | the "GPU can't run it" case in a browser *is* `soft-cpu` + `wasm-simd128` |
| **JavaScript (browser)** | `reconl.js` ES module wrapping the WASM build | OffscreenCanvas + SharedArrayBuffer workers |

Binding acceptance rule: each binding must pass the same three tests —
create a device, render a golden image, and query stats — with no per-binding
special-casing in core.

---

## 10. TOOLCHAIN & BUILD (this machine)

The whole toolchain lives on `A:\toolchain` and is on `PATH`:

| Tool | Path | Use |
|---|---|---|
| Rust / cargo / rustup | `A:\toolchain\Rust` | core, `soft-cpu`, bindings, tools |
| MinGW-w64 GCC | `A:\toolchain\mingw64\bin` | C/C++ shims, Windows builds |
| Zig | `A:\toolchain\zig` | `zig cc` portable C compiler / cross builds |
| CMake + Make | `A:\toolchain\cmake\bin`, `A:\toolchain\bin` | C/C++ examples + packaging |
| JDK 21 + Gradle | `A:\toolchain\jdk`, `A:\toolchain\gradle` | Panama FFM binding tests |
| Node.js + npm | `A:\toolchain\Npm` | N-API addon, web build tooling |
| Python | `A:\toolchain\python` | golden-image diffing, CI scripts |
| Go, `gradle-home`, `go-path` | `A:\toolchain` | keep as-is, don't relocate |

Constraints:

- **No global installs, no vendored copies of the toolchain.** `CARGO_HOME`,
  `RUSTUP_HOME`, `JAVA_HOME`, `GOPATH`, `GRADLE_USER_HOME` already point at
  `A:\toolchain` — reuse them, never re-define them to somewhere else.
- **Project lives at `Desktop\Projects\ReconL`** (created alongside `Projects\GMS`).
  Nothing may be written outside the project dir except the *opt-in* spill cache.
- Provide `build.bat` and `bench.bat` at the root, in the style of the sibling
  `GMS` repo (`build.bat` → `target/release`, `bench.bat` → recorded run), so
  the workflow matches the rest of this vault.
- CI must be able to build and *render* with **no GPU at all** (that is the
  `soft-cpu` + `null` path) — if CI needs a GPU, the design has failed.
- **Commits, PRs and merges carry no agent attribution whatsoever, and a pull
  request waits for automated reviewers** — read §15 before running your first
  `git commit`.

---

## 11. MILESTONE 1 — VERTICAL SLICE (order matters)

1. `include/reconl/reconl.h` frozen for the slice: device, buffer, texture,
   pipeline, command list, swapchain/offscreen target, fence, stats, error model,
   **and `ReconLLight` / `ReconLShadowConfig`** (§7).
2. `core/` tier resolver + budget accounting + stats + logging (`--log-level`).
3. `backends/null/` — the no-op that makes ABI tests possible without a GPU.
4. `backends/soft-cpu/` — tiled reference rasteriser: one triangle → one
   textured, blended quad → depth buffer → mip sampling → **one directional light
   with a single texel-snapped cascade and PCF 3×3**. SIMD where measurable.
5. `backends/d3d11/` — the same passes on hardware, including the same single
   cascade, so the goldens are comparable line for line.
6. `shadow/` — cascade fit + snap, the filter kernels, and the static-cascade
   cache keyed for §8's spill arena (the cache can land in milestone 2, but the
   key and the invalidation hook are designed here).
7. `tools/reconl-info` (probe + tier report), `tools/reconl-bench` (timed run,
   min/avg/max + trace to file, `--shadows off|on|cached`), `tools/reconl-diff`
   (golden-image compare, exact and tolerance modes).
8. `tests/golden/` — committed reference PNGs (shadowed scene included) + the
   diff harness, plus the cascade-snap crawl test.
9. `examples/c/triangle.c`, `examples/rust/triangle.rs`, and a headless
   `examples/html/` page that renders the same shadowed scene in the browser.
10. `bindings/` — start with **C, C++, Rust**; add **Swift, Java, C#,
    Node, WASM/JS** once the ABI has survived one slice unchanged.
11. `README.md` written in the sibling `GMS` style: what it does, how it works,
    how it was measured, what is provisional.

**Definition of done for the slice:** the same **shadowed** scene renders to
visually identical golden images on `soft-cpu` and `d3d11`; `reconl-bench
--tier=soft-cpu --ram-cap=64MB --spill=1 --shadows=cached` completes a 60-frame
run under a 64 MB RAM cap using disk traffic, reuses the cached static cascade on
the second run, and produces a passing golden diff; and every binding listed as
shipped can do §9's three tests.

---

## 12. ACCEPTANCE CRITERIA (MACHINE-CHECKABLE)

- [ ] `reconlProbe()` returns a tier for every backend without creating a device.
- [ ] ABI version/`struct_size` mismatch is detected and reported, not crashed on.
- [ ] Host allocator is used for 100% of core allocations (verified in tests by a
      counting allocator); zero allocations after `reconlBeginFrame` in steady state.
- [ ] Deleting the spill dir mid-run: no crash, no wrong pixels, only slower.
- [ ] Aborting the process mid-spill leaves a recoverable cache on next open.
- [ ] `soft-cpu` output is bit-identical across 1, 2, 4, 8 worker threads.
- [ ] Forced downgrade chain T0 → T4 in one session keeps presenting frames.
- [ ] Every downgrade appears in the log with a reason and in `ReconLStats`.
- [ ] Header compiles as C11 and C++20, `-Wall -Wextra -Wpedantic` clean.
- [ ] `reconl_raster` is fuzz-clean for 1e6 iterations of random command streams.
- [ ] A shadowed golden scene exists, and `soft-cpu` vs `d3d11` diff within the
      documented tolerance in both exact and perceptual modes.
- [ ] Static cascade: cold-cache, warm-cache and corrupted-cache runs produce
      *identical* pixels (the cache changes speed, never output).
- [ ] Cascade snapping: a scripted camera dolly shows no edge crawl beyond the
      threshold in the test file — an automated comparison, not a look.
- [ ] Every tier in §7's shadow ladder renders shadows; no tier silently drops to
      unshadowed output, and any fail-safe fallback is counted in `ReconLStats`.
- [ ] `--shadows=off` changes only the shadowed pixels of a golden frame.
- [ ] Shadow-pass cost is reported separately (pass ms, cascades, map resolution,
      cache hit rate, cache bytes) and never exceeds its tier's texel budget.

---

## 13. MEASUREMENT & HONESTY PROTOCOL (bake this in from commit 1)

Borrowed directly from the sibling `GMS` repo, because the failure it prevents is
the same one: a number without the config that produced it.

- **Run fingerprint:** every `reconl-bench` run writes the full resolved config
  (backend, tier, budgets, worker count, spill on/off, resolution scale, driver
  version, device name) into the log before the first frame.
- **Recorded, not read:** the recorder writes a second-by-second trace plus
  `min / avg / max` to a file — never a peak quoted as the result.
- **Labelled claims:** provisional numbers are labelled provisional in the README,
  with the reason. No "up to N×" without the baseline and the read method.
- **Fail-safe over fast-fail:** on any internal disagreement (tier guess vs
  probe, budget accounting drift, checksum mismatch) ReconL falls back to the
  *safe* path and counts it, rather than trusting the fast path.
- **Audit mode:** `--audit` re-verifies the tier decision and budget totals every
  N frames and logs divergence. It is a diagnostic, off by default, documented as
  expensive.
- **Shadow cost is attributed, not guessed.** The recorder reports shadow-pass ms,
  cascade count, map resolution and cache hit rate as separate lines, so "shadows
  cost X" is a measurement with a config attached rather than an implication.

---

## 14. RISKS & OPEN QUESTIONS (answer before milestone 2)

1. **Determinism vs speed on GPU tiers.** Bit-identical GPU output across vendors
   is not free; the honest target may be *tolerance* diffing (`reconl-diff`
   supports both exact and perceptual modes). Decide and document.
2. **Spill latency budgeting.** Disk traffic inside a frame is a hitch generator.
   The rule must be: spill work happens on its own thread, on its own budget,
   with frame-scoped prefetch — or the tier is reported as slower, not smoother.
3. **ABI growth.** `struct_size`/`next` extension chains need a written policy
   for deprecation, or the header calcifies.
4. **`soft-cpu` scope creep.** A reference rasteriser that tries to be a full
   GPU is unbounded. Freeze the feature floor (blend modes, mip modes, wrap
   modes, MSAA yes/no) and make everything else a documented cap.
5. **Browser threading.** `SharedArrayBuffer` needs COOP/COEP headers; the WASM
   `soft-cpu` tier must have a single-threaded path that still passes goldens.
6. **Java binding choice.** Panama FFM is the right call on Java 21, but
   `jextract`-generated code must be regenerated in CI so it can't silently rot.
7. **Shadow depth precision across backends.** A software depth buffer and
   reversed-Z on D3D11 differ at cascade seams by more than a tolerance you can
   pick by taste. Choose the encoding once (§7), measure the real delta, and set
   `reconl-diff`'s tolerance from that measurement.
8. **CPU shadow cost is the biggest T2 cliff.** Four cascades at 2048² in
   software is not free — the cascade texel budget has to be a first-class budget
   line that scales with tier (which is why §7's ladder table exists), or T2 will
   be correct and unusable.
9. **Point-light cube shadows multiply work by six.** Cap the number of shadowed
   point lights explicitly, or six passes appear out of nowhere and dominate the
   frame; make the cap visible in `ReconLStats` rather than implicit.
10. **Bias defaults are per-tier.** A scene tuned at T0 can acne at T3. Either
    derive bias from cascade scale automatically or ship documented per-tier
    presets — do not leave it to whoever runs the demo.

---

## 15. REPO HYGIENE — COMMITS, PRS, MERGES (NO AGENT ATTRIBUTION; WAIT FOR AUTOMATED REVIEWERS)

**Hard rule, every agent, every session, no exceptions.**

- **No `Co-Authored-By` trailer for Codebuff, or for any agent, tool, model or
  assistant.** The only author or co-author on any commit is the human.
- **No "Generated with Codebuff 🤖" footer**, no 🤖 marker line, no
  `noreply@…` agent address, no `Co-Authored-By: …` of any variant — in commit
  messages, PR titles, PR bodies, merge commits, tags, release notes, or
  changelogs.
- **No agent or model names anywhere in the history**: not in the subject, not in
  the body, not as a trailer, not as `Signed-off-by`, not as "assisted by" /
  "co-written with" / "built with AI" credit.
- Commit messages describe **the change and why it was made** — nothing about how
  the text was produced. PR bodies are written in the maintainer's voice, with no
  generation banners or attribution blocks.
- If a tool's default behaviour injects such a trailer, strip it **before** the
  commit exists. After committing, verify with `git log -1 --format=%B`; amend
  only a commit that is yours and not yet pushed.
- This outranks any tool default, any workflow habit, and any agent's own
  convention. When in doubt, ship fewer lines and zero attribution.
- Documentation and READMEs likewise carry no agent credit unless the human
  explicitly asks for it.

**Automated reviewers exist — expect them, and give them a window.**

- A pull request may be reviewed by **automated reviewers** (bots, CI review
  services, agent reviewers) as well as humans, and their comments can land
  anywhere from seconds to a few minutes after the PR is opened or updated. They
  may also re-review after a push.
- **After opening a PR — or pushing to one — wait 1–3 minutes before treating it
  as clean, merging it, or walking away.** Poll the PR during that window rather
  than idling once and declaring silence; a review that arrives at minute two is
  normal, not late.
- **Do not merge while a review is still pending or unaddressed,** and never
  merge on the assumption that silence means approval. If the window closes with
  nothing raised, say so explicitly before merging.
- If you push fixes in response to a review, the window restarts: wait again,
  because an automated reviewer will usually take another look.
- Treat automated findings exactly like human findings — read them, answer them,
  or fix them; don't dismiss them for being automated, and don't obey them
  blindly either. If a comment is wrong, say why in the thread.
- If the wait is genuinely pointless (a docs-only typo PR, no reviewers
  configured), state that in one line instead of silently skipping the window.

---

💡 **Tip:** there is no `DESIGN.md` in this vault. For consistent visual output
across ReconL's docs, banner, and any site page, create one with the `design-md`
skill and drop it at `Projects/ReconL/DESIGN.md` (the brand block in §1 is the
starting point).

## Next Prompt

Copy from here to reuse this prompt elsewhere:

> Build **ReconL**: a portable, embeddable renderer at `Desktop\Projects\ReconL`
> with one frozen C11 ABI as the compatibility surface and a graceful offload
> ladder — discrete GPU → shared GPU → multi-threaded CPU → thrifty CPU →
> disk-backed out-of-core software rasteriser — so the same scene renders
> correctly from Rust, C, C++, Swift, Java (Panama FFM), C#, Node, and plain
> HTML/WASM, and never hard-fails on weak hardware. **Shadows are required at
> every tier** (cascaded directional maps down to a single disk-cached cascade on
> a machine with no GPU at all), with texel-snapped cascades, documented bias
> defaults, and a fail-safe unshadowed fallback that is counted rather than
> silent. Ship milestone 1 as a vertical slice: the ABI, a `null` backend, a
> tiled SIMD `soft-cpu` reference rasteriser with one shadow cascade, a D3D11
> backend rendering the same cascade, golden-image tests proving both produce the
> same frame, and probe/bench/diff tools — built entirely with the toolchain at
> `A:\toolchain` and measured with run fingerprints and recorded traces, not
> peaks. Follow the brand block and the measurement protocol in this file, and
> **never add agent attribution** (no `Co-Authored-By`, no "generated with"
> footer, no model names) to commits, PRs, merges, tags or release notes, and
> **after opening or pushing to a pull request, wait 1–3 minutes** for automated
> reviewers and CI before treating it as clean or merging it.
