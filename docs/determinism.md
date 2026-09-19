# Determinism

The point of this document is that a golden image is only evidence if the same
scene produces the same bytes. Everything below is a rule that exists to make that
true, with the code that implements it and the test that would catch it being
undone. When a rule and a test disagree, the test is right and this file is stale.

The **reference tier** (`soft-cpu`, T2) defines correctness. A hardware backend is
correct when it agrees with the reference inside the documented tolerance - not
the other way round.

## The rules

**1. Reversed-Z depth, one encoding, everywhere.**
Near maps to `1.0`, far to `0.0`, depth clears to `0.0`, and the comparison is
`GREATER`. This is the only depth encoding in the engine
(`raster/src/math.rs`), and it is what a hardware backend must reproduce: D3D11 is
created with `COMPARE_GREATER` and a `[0,1]` float depth buffer, not with a
converted projection.
*Pinned by:* `raster/src/math.rs` unit tests; the reference transform's
near→1/far→0 check in `tools/host/src/scene.rs`.

**2. Top-left fill rule and 8-bit subpixel vertex precision.**
A pixel is covered when its centre is inside the edge, or exactly on a top or
left edge, so two triangles that share an edge cover every pixel between them
exactly once and neither double-shades nor leaves a seam.
*Pinned by:* `raster/src/fixed.rs` and its tests.

**3. Tiles are independent, so worker count cannot change a pixel.**
A tile is written by exactly one worker, and triangles are visited in the frame's
draw order within a tile - which is why binning is a counting sort rather than a
race. The frame is bit-identical for 1, 2, 4 or 8 workers.
*Pinned by:* `raster/tests/thread_determinism.rs`, and end to end by
`tools/reconl-bench/tests/cli.rs::the_reference_frame_it_times_is_the_committed_golden`,
which renders the golden frame with the tool's own (auto-detected) thread count and
compares it byte for byte with the committed PNG.

**4. No wall-clock input into geometry.**
Time may pick a *tier* (the frame-time ladder) and may be *measured*, but no
transform, bias or filter may read it. A scene rendered at 3 ms and at 30 ms is the
same scene.

**5. No allocation inside a frame.**
Storage is reserved between frames - `Rasterizer::prepare`, `BeginFrame`'s target
reservation - so that a frame is a fixed sequence of writes over fixed buffers.
**This one is currently not met.** `reconl-bench` measures the real rate from the
host's own allocator ledger and reports it per frame: on this machine, 60 measured
frames of the 64x64 reference scene at T2 give **8 allocator calls per frame**
(including reallocs), and `d3d11` gives 2. The acceptance criterion in PROMPT §12
is zero. Attributing and closing that gap is a pass of its own; the measurement is
what makes it visible rather than assumed.
*Measured by:* `reconl-bench`'s `allocations` line.

**6. SIMD only where it is exact.**
Fill, copy, clear and checksum are vectorised; the shaded path is scalar in every
tier, on purpose. A vectorised path that could differ from the scalar one in the
last bit would make the goldens build-dependent, which is worse than being slow.
The capabilities the library reports are the **build's**, not the machine's.
*Pinned by:* `raster/src/simd.rs` and the `RECONL_CAP_SIMD_*` bits coming only
from `simd::detect`.

**7. A frozen feature floor for textures.**
RGBA8 sources, nearest and bilinear filtering, point or linear mip selection,
repeat/clamp/mirror wrap. Anything a backend does beyond that is a declared
capability, not a surprise: a host can see it in `ReconLDeviceCaps` before it
depends on it.
*Pinned by:* `raster/src/texture.rs`, whose wrap and texel selection use integer
arithmetic on the wrapped coordinate so a tap at `u = 1.0` lands on the same texel
in every run and every tier.

**8. Shadow bias is visible to the host, not chosen per run.**
The tier's default comes from one table; a scene may pin all three values through
`ReconLShadowConfig`. The reference scene pins them, which is what makes the
golden the scene's image rather than the tier's. See `docs/bias.md`.
*Pinned by:* `shadow/src/lib.rs::bias_preset` and the cross-tier comparison in
`ffi/tests/tiers.rs`.

**9. Shadow culling follows the renderer's winding, not the light's convenience.**
The shadow pass culls the far side on the light's view, using the same front-face
convention as the colour pass. A caster wound the other way is culled and silently
stops casting, which is why the reference scene's caster is wound like its ground
and why a test asserts they turn the same way.
*Pinned by:* `tools/host/src/scene.rs::tests::the_reference_scene_is_the_shape_it_documents`,
`ffi/tests/tiers.rs`.

## How determinism is checked

Three layers, cheapest first:

| Layer | What it proves | Where |
|---|---|---|
| Unit tests | one rule, in isolation (fixed-point, math, bias table) | `cargo test` |
| Golden images | the whole reference scene, byte for byte, from two independent producers | `reconl-diff compare`, `reconl-bench --png` |
| Cross-tier comparison | a hardware backend agrees with the reference within a *measured* tolerance, with the divergence confined to the shadow's edge | `ffi/tests/tiers.rs`, `reconl-diff compare --tolerance=48` |

The tolerance is not chosen. For the shadowed reference scene, `soft-cpu` and
`d3d11` differ on 14 of 4096 pixels with a worst channel delta of 46, all within
4 px of a pixel the shadow changes, and none in the shadow's interior. The
documented setting for the cross-tier comparison is therefore `--tolerance=48`
inside the 1% budget. Widening the tolerance to absorb a divergence is not a fix,
it is a deletion of the evidence.
