# Shadow bias

A shadow lookup compares a fragment's depth against the depth stored in the
shadow map. Two surfaces that are the same distance from the light do not store
the same number - the map holds one sample per texel, the fragment may be
sub-texel, and the stored depth is whatever the rasteriser wrote for the triangle
that covered that texel. Without a bias the comparison flickers: a lit surface
shadows *itself* (acne), and every shadow acquires a lace edge.

The bias is the smallest nudge that removes that without detaching the shadow
from its caster (peter-panning). There is exactly one policy, it is written down
here, and it is implemented in one place: `shadow/src/lib.rs::bias_preset` and
`raster/src/shade.rs`.

## The formula

```
bias = depth_bias + slope_bias * texel_world * tan(theta) / depth_span
```

* `depth_bias` - a constant in reversed-Z depth units, already scaled for the map
  size. It covers the flat-surface case, where a fragment and the texel it samples
  differ only by sub-texel position.
* `slope_bias` - how many texel footprints of slack a sloped surface gets.
* `texel_world` - one texel's world-space footprint in that cascade, computed
  from the light's frustum and the map resolution.
* `tan(theta)` - the tangent of the angle between the surface normal and the
  light direction: a surface seen edge-on by the light spans more depth per texel,
  so it needs more slack.
* `depth_span` - the cascade's world-space depth range, which converts that
  world-space error into the cascade's depth units.

Keeping the slope term in **texel footprints** rather than in depth units is the
whole point: it is what lets one table work at 512, 1024 and 4096 texels. The
conversion to depth units happens per fragment, where the cascade's span is known.

Bias is added, not subtracted. Under reversed-Z a larger depth is nearer the
light, and a fragment is lit when it is at least as near as the stored occluder -
so the reference depth has to move *toward* the light. Subtracting would make every
surface shadow itself by exactly the bias, and the acne would grow with the bias
instead of shrinking.

## The table

One table, tier-aware (`bias_preset(tier, map_size, filter)` in
`shadow/src/lib.rs`):

| Tier | `normal_bias` (texels) | `depth_bias` (depth units at 1024) | `slope_bias` |
|---|---|---|---|
| T0 gpu-discrete | 1.00 | 1.5e-4 | 1.5 |
| T1 gpu-shared | 1.25 | 2.5e-4 | 1.75 |
| T2 cpu-ram | 1.75 | 4.0e-4 | 2.0 |
| T3 cpu-thrifty | 2.25 | 6.0e-4 | 2.5 |
| T4 out-of-core | 2.50 | 8.0e-4 | 3.0 |

Two adjustments are applied to `depth_bias`:

* **Filter width.** More taps average more of the neighbourhood, which lifts the
  depth the comparison sees, so the depth bias scales with the filter: 0.5x for a
  single tap, 1.0x for PCF 3x3, 1.5x for PCF 5x5, 2.0x for the PCSS-lite blocker
  search.
* **Map size.** Smaller maps have larger texels, so depth error grows as texels do:
  `clamp(1024 / map_size, 0.5, 8.0)`.

The whole table is scaled by tier because the tiers differ in *why* they can afford
precision: a GPU can spend a texel and a depth bit on accuracy that a CPU tier
spends milliseconds on. That is a deliberate difference in image, not a bug in one
tier - which is exactly why a *golden* scene pins all three values through the ABI
(below) and why comparing two tiers with their own presets is a different
comparison from comparing two rasterisers.

## Which number is the documented one

`ReconLShadowConfig` carries `normal_bias`, `depth_bias` and `slope_bias`
directly. Their meaning on the wire:

* `normal_bias` is in **shadow-map texels** - how far to offset the lookup along
  the surface normal.
* `depth_bias` is in **reversed-Z depth units**, already scaled for the map size
  the value will be used at. A host that intends to sit at an unfamiliar map size
  should scale it as the table does rather than guess.
* `slope_bias` is in **texel footprints** and is size-independent by construction.

**All three zero means "use yours"**, not "use no bias": that is the default, and
it is what a host compiled before these fields existed sends. **Any non-zero value
means the host owns all three**, and they are used exactly as given - the library
does not rescale, re-scale by map size, or fall back on the table for the others.
A host that means to pin the policy must therefore pin all three, which is what the
reference scene does.

The consequence worth knowing: there is no way to ask for *literally no* bias,
because zero is the sentinel. A diagnostic that wants to see raw acne has to send a
denormal-small non-zero value. That is a wart in the encoding rather than a bug in
the policy, and it is named here so the next pass changes it deliberately if it
changes it at all - a sentinel cannot be retired without a new field and an ABI
bump, since a host compiled against this header may be sending zero to mean
"default" forever.

## Why the golden scene pins them

The reference scene sets `normal_bias = 1.25`, `depth_bias = 5.0e-4` and
`slope_bias = 1.75` (the T1 preset, which both tiers round to 512x512 maps at that
budget). With the presets in force instead, the same scene through `d3d11` differs
from the reference on 220 of 4096 pixels; pinned, on 14. Both numbers are real, and
the smaller one is the one worth a golden: it measures the rasterisers agreeing
rather than the bias tables differing.

That is the rule to carry forward: **a golden pins its bias; a tier comparison
that means to test bias does not.**
