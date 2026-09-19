//! The HLSL shaders, and the constant-buffer layouts they compile against.
//!
//! Every expression is a port of the reference implementation in the same
//! order with the same parentheses (`reconl-raster` `shade.rs`). Two choices
//! exist because the reference is what it is, not because they are pretty:
//!
//! - **Lighting happens at the raw vertex position.** The rasteriser passes
//!   vertex attributes through untouched (`tile.rs` `transform_vertex` only
//!   transforms the clip position), so `SurfaceShader::shade` lights at
//!   whatever space the vertices are in. The GPU path does the same: the
//!   vertex shader forwards the raw position and normal, and the transform it
//!   applies to the clip position is the *pre-composed* view·proj·model the
//!   CPU already built with `math::mul` — the same matrix, in the same
//!   multiply order, the reference rasterises with.
//! - **Blend modes are pre-multiplied in the shader.** The reference blends
//!   `dst + src·a` variants by hand; fixed function can express each of them
//!   exactly if the shader emits `src·a` for the modes that need it (see
//!   `BLEND_*` in the device). Opaque also forces alpha to 1, as the
//!   reference's blend does.
//!
//! Shadow lookup specifics ported from `shade.rs`:
//! - cascade scan by camera-space distance `-(row2 · p)`, crossfade band,
//!   distance fade to unshadowed — in the reference's order
//! - bias `depth_bias + slope_bias · texel_world · tan(theta) / depth_span`
//!   with `tan(theta)` capped at `MAX_SLOPE_BIAS` (8.0)
//! - reversed-Z comparison: lit when `reference + 1.0e-7 >= stored`; the
//!   epsilon lives only in the comparison taps, the PCSS blocker search
//!   compares raw depth with a strict `>`
//! - `pcss-lite`: raw-depth 3×3 blocker search, then a 5×5 comparison tap
//!   grid scaled by the penumbra width
//! - the crossfade re-runs the *full* filter dispatch on the next cascade,
//!   exactly as `factor()` calls `factor_for(next, ..)`
//!
//! `#pragma pack_matrix` is NOT overridden: `Mat4` is column-major "like
//! HLSL/GLSL constant buffers", so D3D11's default `column_major` packing
//! plus `mul(M, v)` is exactly the reference's `x·m[0]+y·m[4]+z·m[8]+w`.

/// Constant buffer 0: per-draw. 96 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DrawCb {
    /// Pre-composed view·proj·model (colour pass) or light·model (shadow pass).
    pub transform: [f32; 16],
    /// x = textured, y = lit, z = receives_shadow, w = blend mode (0..3).
    pub flags: [f32; 4],
    /// x = two_sided (flip the normal on back faces), rest unused.
    pub flags1: [f32; 4],
}

/// Constant buffer 1: the lights, in `reconl-raster`'s element order. 656 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LightsCb {
    /// xyz = world position (directional: unused), w = kind (0 dir, 1 spot, 2 point).
    pub light_a: [[f32; 4]; 16],
    /// xyz = direction, w = intensity.
    pub light_b: [[f32; 4]; 16],
    /// xyz = colour, w = range (<= 0 = no attenuation).
    pub light_c: [[f32; 4]; 16],
    /// x = cos(inner), y = cos(outer), z = cast_shadow, w = unused.
    pub light_d: [[f32; 4]; 16],
    /// xyz = ambient, w = light count.
    pub ambient_count: [f32; 4],
}

/// Constant buffer 2: the whole shadow lookup. 16·4 + 4·(64+16) = 368 bytes.
///
/// All four cascades live in one buffer so the pixel shader's cascade scan is
/// a dynamic index (SM 5.0 `ldc`), exactly like the reference's linear scan.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShadowCb {
    /// Row 2 (the third row) of the camera view matrix: camera-space depth.
    pub view_z_row: [f32; 4],
    /// x = filter (0 hard, 1 pcf3x3, 2 pcf5x5, 3 pcss-lite), y = cascade count,
    /// z = normal_bias (texels), w = depth_bias (cascade depth units).
    pub params_a: [f32; 4],
    /// x = slope_bias, y = max_distance, z = blend_band, w = map_size (texels).
    pub params_b: [f32; 4],
    /// Light view·projection per cascade, column-major as stored.
    pub cascade_mvp: [[f32; 16]; 4],
    /// x = split_distance (camera-space), y = texel_world, z = depth_span,
    /// w = map_index (array slice).
    pub cascade_params: [[f32; 4]; 4],
}

pub const VS_HLSL: &str = r#"
// column_major is the default and is what Mat4 is stored as.

struct VSIn {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0;
};

struct VSOut {
    float4 clip     : SV_POSITION;
    float3 world    : WORLDPOS;   // the raw vertex position, as the reference shades at
    float3 normal   : NORMAL;     // the raw vertex normal
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0;
};

cbuffer DrawCb : register(b0) { float4x4 transform; float4 flags; float4 flags1; }

VSOut main(VSIn input) {
    VSOut o;
    o.clip = mul(transform, float4(input.position, 1.0));
    o.world = input.position;
    o.normal = input.normal;
    o.uv = input.uv;
    o.color = input.color;
    return o;
}
"#;

pub const PS_HLSL: &str = r#"
// column_major is the default and is what Mat4 is stored as.

Texture2DArray    base_tex   : register(t0);
SamplerState      samp       : register(s0);
Texture2DArray         shadow_map : register(t1);  // comparison reads
SamplerComparisonState shadow_cmp : register(s1);
Texture2DArray    shadow_raw : register(t2);  // raw-depth reads (PCSS blocker search)

struct PSIn {
    float4 clip     : SV_POSITION;
    float3 world    : WORLDPOS;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0;
    bool  is_front  : SV_IsFrontFace;   // rasteriser-generated, matches area2 > 0
};

cbuffer DrawCb : register(b0) { float4x4 transform; float4 flags; float4 flags1; }
cbuffer LightsCb : register(b1) {
    float4 light_a[16];       // xyz position, w kind (0 dir, 1 spot, 2 point)
    float4 light_b[16];       // xyz direction, w intensity
    float4 light_c[16];       // xyz colour, w range
    float4 light_d[16];       // x cos_inner, y cos_outer, z cast_shadow, w unused
    float4 ambient_count;
};
cbuffer ShadowCb : register(b2) {
    float4 view_z_row;
    float4 params_a;          // filter, cascade_count, normal_bias, depth_bias
    float4 params_b;          // slope_bias, max_distance, blend_band, map_size
    float4x4 cascade_mvp[4];
    float4 cascade_params[4]; // split_distance, texel_world, depth_span, map_index
};

// The reference `slope_bias_term`: tan(theta) capped at 8.0.
float slope_term(float3 n, float3 l) {
    float ndl = dot(n, l);
    if (ndl <= 1.0e-4) { return 8.0; }
    return min(sqrt(max(0.0, 1.0 - ndl * ndl)) / ndl, 8.0);
}

// One comparison tap: lit when the (epsilon-shifted) reference depth is at
// least the stored occluder. Reversed-Z: GREATER_EQUAL, near = 1.
float tap(float2 uv, float slice, float reference) {
    return shadow_map.SampleCmp(shadow_cmp,
        float3(clamp(uv, 0.0, 1.0), slice), reference + 1.0e-7);
}

float pcf(float2 uv, float slice, float reference, int radius) {
    float2 texel = 1.0 / params_b.w;
    float sum = 0.0;
    [loop]
    for (int dy = -radius; dy <= radius; ++dy) {
        [loop]
        for (int dx = -radius; dx <= radius; ++dx) {
            sum += tap(uv + float2(dx, dy) * texel, slice, reference);
        }
    }
    float n = float((2 * radius + 1) * (2 * radius + 1));
    return sum / n;
}

// pcss-lite, ported: a raw-depth 3x3 blocker search over the reference's
// own comparison (`stored > reference`), then a 5x5 comparison tap grid at
// `dx * width * 0.5` texels. PCSS_LITE_K = 16, max radius 4.
float pcss_lite(float2 uv, float slice, float reference) {
    float2 texel = 1.0 / params_b.w;
    float blocker_sum = 0.0;
    float blockers = 0.0;
    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            float d = shadow_raw.SampleLevel(samp,
                float3(clamp(uv + float2(dx, dy) * texel, 0.0, 1.0), slice), 0.0).x;
            if (d > reference) {
                blocker_sum += d;
                blockers += 1.0;
            }
        }
    }
    if (blockers == 0.0) { return 1.0; }
    float avg = blocker_sum / blockers;
    float width = clamp(16.0 * max(avg - reference, 0.0) / max(avg, 1.0e-4), 0.0, 4.0);
    float sum = 0.0;
    [unroll]
    for (int j = -2; j <= 2; ++j) {
        [unroll]
        for (int i = -2; i <= 2; ++i) {
            sum += tap(uv + float2(i, j) * (width * 0.5) * texel, slice, reference);
        }
    }
    return sum / 25.0;
}

// The reference `factor_for` for one cascade index. Every fail-safe returns
// 1.0 (fully lit), exactly as the reference's does.
float lookup_at(int ci, float3 world, float3 n, float3 l) {
    float4 cp = cascade_params[ci];
    float4x4 mvp = cascade_mvp[ci];

    float texel_world = max(cp.y, 1.0e-6);
    float3 wp = world + n * (params_a.z * texel_world);
    float4 lp = mul(mvp, float4(wp, 1.0));
    float w = lp.w;
    if (abs(w) < 1.0e-9) { return 1.0; }
    float3 ndc = lp.xyz / w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }

    float slope = slope_term(n, l);
    float depth_span = max(abs(cp.z), 1.0e-6);
    float bias = params_a.w + params_b.x * texel_world * slope / depth_span;
    float reference = ndc.z + bias;

    float filter = params_a.x;
    if (filter < 0.5) {
        return tap(uv, cp.w, reference);
    } else if (filter < 1.5) {
        return pcf(uv, cp.w, reference, 1);
    } else if (filter < 2.5) {
        return pcf(uv, cp.w, reference, 2);
    }
    return pcss_lite(uv, cp.w, reference);
}

// The reference `factor`: cascade scan, primary lookup, crossfade toward the
// next cascade inside the blend band (re-running the full filter on it), then
// the distance fade to unshadowed.
float shadow_factor(float3 world, float3 n, float3 l) {
    int count = (int)params_a.y;
    float max_distance = params_b.y;
    float blend_band = params_b.z;

    float distance = -(view_z_row.x * world.x + view_z_row.y * world.y + view_z_row.z * world.z + view_z_row.w);
    if (distance > max_distance || count <= 0) { return 1.0; }

    int ci = count - 1;
    [unroll]
    for (int k = 0; k < 3; ++k) {
        if (k < count && distance <= cascade_params[k].x) { ci = k; break; }
    }
    float lit = lookup_at(ci, world, n, l);

    if (blend_band > 0.0 && distance > cascade_params[ci].x - blend_band) {
        float t = saturate((distance - (cascade_params[ci].x - blend_band)) / blend_band);
        if (ci + 1 < count) {
            float f = lookup_at(ci + 1, world, n, l);
            lit = lit + (f - lit) * t;
        } else {
            lit = lit + (1.0 - lit) * t;
        }
    }
    if (blend_band > 0.0 && distance > max_distance - blend_band) {
        float t = saturate((distance - (max_distance - blend_band)) / blend_band);
        lit = lit + (1.0 - lit) * t;
    }
    return lit;
}

float4 main(PSIn input) : SV_TARGET {
    bool textured = flags.x > 0.5;
    bool lit = flags.y > 0.5;
    bool receives_shadow = flags.z > 0.5;
    int blend_mode = (int)(flags.w + 0.5);

    float4 texel = float4(1.0, 1.0, 1.0, 1.0);
    if (textured) { texel = base_tex.Sample(samp, float3(input.uv, 0.0)); }

    // two_sided: the reference flips the normal on back faces before the
    // normalise (`TriJob::flip_normal = !front && pipeline.two_sided`, applied
    // inside shade() before the normalise). SV_IsFrontFace is true exactly when
    // the rasteriser's area2 > 0, which is the reference's `front`.
    float3 nrm = input.normal;
    if (flags1.x > 0.5 && !input.is_front) { nrm = -nrm; }
    float3 n = normalize(nrm);
    float3 albedo = input.color.rgb * texel.rgb;
    float a = saturate(input.color.a * texel.a);

    float3 rgb;
    if (!lit) {
        rgb = saturate(albedo);
    } else {
        float3 lit_acc = ambient_count.xyz;
        int count = (int)ambient_count.w;
        [loop]
        for (int i = 0; i < 16; ++i) {
            if (i >= count) { break; }
            float3 to_light;
            float attenuation;
            if (light_a[i].w < 0.5) {                      // directional
                to_light = normalize(-light_b[i].xyz);
                attenuation = 1.0;
            } else if (light_a[i].w > 1.5) {               // point
                float3 delta = light_a[i].xyz - input.world;
                float d = length(delta);
                to_light = (d > 1.0e-6) ? delta / d : float3(0.0, 1.0, 0.0);
                float fa = (light_c[i].w <= 0.0) ? 1.0 : saturate(1.0 - d / light_c[i].w);
                attenuation = fa * fa;
            } else {                                        // spot
                float3 delta = light_a[i].xyz - input.world;
                float d = length(delta);
                to_light = (d > 1.0e-6) ? delta / d : float3(0.0, 1.0, 0.0);
                float cos_theta = -dot(normalize(light_b[i].xyz), to_light);
                float inner = light_d[i].x;
                float outer = light_d[i].y;
                float spot;
                if (inner - outer <= 1.0e-6) {
                    spot = (cos_theta >= outer) ? 1.0 : 0.0;
                } else {
                    spot = saturate((cos_theta - outer) / (inner - outer));
                }
                float fa = (light_c[i].w <= 0.0) ? 1.0 : saturate(1.0 - d / light_c[i].w);
                attenuation = fa * fa * spot;
            }
            float ndl = max(dot(n, to_light), 0.0);
            if (ndl <= 0.0) { continue; }
            float shadow = 1.0;
            if (receives_shadow && light_d[i].z > 0.5) {
                shadow = shadow_factor(input.world, n, to_light);
            }
            float k = light_b[i].w * ndl * attenuation * shadow;
            lit_acc = lit_acc + light_c[i].xyz * k;
        }
        rgb = saturate(albedo * lit_acc);
    }

    // Blend pre-multiplication: the fixed-function states in the device were
    // chosen so that emitting `src` as below reproduces the reference's
    // blend() exactly. Opaque forces alpha to 1, as the reference does.
    if (blend_mode == 0) { return float4(rgb, 1.0); }
    if (blend_mode == 2) { rgb = rgb * a; }             // additive: dst + src*a
    if (blend_mode == 3) { rgb = rgb * a; }             // multiply: dst*(1-a) + dst*rgb*a
    return float4(rgb, a);
}
"#;

pub const VS_DEPTH_HLSL: &str = r#"
struct VSIn {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0;
};

cbuffer DrawCb : register(b0) { float4x4 transform; float4 flags; float4 flags1; }

float4 main(VSIn input) : SV_POSITION {
    return mul(transform, float4(input.position, 1.0));
}
"#;

/// Depth-only pixel shader: D3D11 has no PS-less path that also writes depth
/// cleanly, so a trivial return is cheapest.
pub const PS_DEPTH_HLSL: &str = r#"
float4 main() : SV_TARGET { return 0; }
"#;
