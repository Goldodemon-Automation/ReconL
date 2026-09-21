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

The *stored* value is the normalised depth `z/w`, not the clip-space `z`: what a
depth buffer holds has to mean the same thing on every tier, because more than the
depth test reads it. A depth readback is what frame generation unprojects pixels
with (`reconlPresentGenerated`), and hardware stores `z/w`, so the reference
tier's rasteriser does too. An orthographic matrix has `w == 1`, which is why the
shadow passes are bit-identical under either choice and why this rule is about
the perspective case.
*Pinned by:* `raster/src/math.rs` unit tests; the reference transform's
near→1/far→0 check in `tools/host/src/scene.rs`;
`ffi/tests/abi.rs::a_generated_frame_predicts_the_frame_after_it`, which fails by
a mile if the CPU tier stores the other one.

**2. Top-left fill rule and 8-bit subpixel vertex precision.**
A pixel is covered when its centre is inside the edge, or exactly on a top or
left edge, so two triangles that share an edge cover every pixel between them
exactly once and neither double-shades nor leaves a seam.
*Pinned by:* `raster/src/fixed.rs` and its tests.

**3. Tiles are independent, so worker count cannot change a pixel.**
A tile is written by exactly one worker, and triangles are visited in the frame's
draw order within a tile - which is why binning is a counting sort rather than a
race. The frame is bit-identical for 1, 2, 4 or 8 workers.
*Pinned by:* `raster/tests/render.rs::frame_is_bit_identical_for_1_2_4_and_8_workers`,
and end to end by
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
Every draw list a frame writes into has one owner, and that owner outlives the
frame: the frame's draws (`FrameRecord::items`, built by the FFI, whose entries
point into the *host's* buffers - the host's reference count is what keeps those
bytes alive while the frame is in use), the reference tier's colour entries
(`SoftCpuDevice::colors`) and its cascade entries (`SoftCpuDevice::cascade`).
Each is cleared and refilled rather than reallocated, and each counts its own
growth in `FrameNumbers::allocations_in_frame` - so a steady state reports zero
rather than a number nobody can check. A frame that needs *more* storage than any
before it still takes a block, which is a new size rather than a hitch in the
steady state.
*Measured by:* `reconl-bench`'s `allocations` line, which counts the host's own
allocator ledger: 60 measured frames of the 64x64 reference scene report **0
allocator calls per frame** at T2 and 0 on `d3d11` on this machine (8 and 2
before the lists above had owners).
*Pinned by:* `tools/reconl-bench/tests/cli.rs::a_steady_state_frame_allocates_nothing`,
and for the rasteriser's own tables `raster/tests/render.rs`.

**6. SIMD only where it is exact.**
Fill, copy, clear, checksum and the `f32`->unorm8 conversion a readback performs
are vectorised; the shaded path is scalar in every tier, on purpose. A vectorised
path that could differ from the scalar one in the last bit would make the goldens
build-dependent, which is worse than being slow. The conversion qualifies because
it is not shading: every input maps to one defined byte, and the vector path is
checked against the scalar definition for NaN, signed zero, subnormals,
out-of-range and rounding-boundary inputs.
*Pinned by:* `raster/src/simd.rs` and the `RECONL_CAP_SIMD_*` bits coming only
from `simd::detect`.

**7. One pass from the device's pixels to the host's buffer.**
A frame reaches a present-to-memory host in a single pass: the backend converts
and lays the frame straight into the rows the host handed over, at the pitch and
flip the present descriptor asked for. There is no tightly packed intermediate of
the frame on either tier, and the layout rule has one owner - the code that knows
the pixel format. `out_row_pitch` and `flip` are covered through the ABI on both
tiers, because every other caller in the tree presents tight and top-down.
*Pinned by:* `the_present_layout_honours_the_row_pitch_and_the_flip` and
`an_audited_frame_presents_the_same_bytes_as_an_unaudited_one` in `ffi/tests/abi.rs`.

**8. A frozen feature floor for textures.**
RGBA8 sources, nearest and bilinear filtering, point or linear mip selection,
repeat/clamp/mirror wrap. Anything a backend does beyond that is a declared
capability, not a surprise: a host can see it in `ReconLDeviceCaps` before it
depends on it.
*Pinned by:* `raster/src/texture.rs`, whose wrap and texel selection use integer
arithmetic on the wrapped coordinate so a tap at `u = 1.0` lands on the same texel
in every run and every tier.

**9. Shadow bias is visible to the host, not chosen per run.**
The tier's default comes from one table; a scene may pin all three values through
`ReconLShadowConfig`. The reference scene pins them, which is what makes the
golden the scene's image rather than the tier's. See `docs/bias.md`.
*Pinned by:* `shadow/src/lib.rs::bias_preset` and the cross-tier comparison in
`ffi/tests/tiers.rs`.

**10. Shadow culling follows the renderer's winding, not the light's convenience.**
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
