//! The windows D3D11 implementation.
//!
//! Frame path, and why it matches the reference:
//!
//! 1. **Shadow pass.** Per cascade the same `reconl-shadow` fit the reference
//!    rasteriser uses is computed on the CPU (identical inputs, identical code),
//!    then every `casts_shadow` draw is rendered into one `R32_TYPELESS` array
//!    slice with the light·view·projection pre-composed by `math::mul`, front
//!    faces culled, reversed-Z depth, and *no* fixed-function bias: the
//!    reference applies bias at lookup time, not at write time.
//! 2. **Colour pass.** Every draw is rendered with the ported pixel shader in
//!    `shaders.rs`: the raw vertex position is what gets lit, the cascade scan,
//!    crossfade and filter dispatch run in the reference's order, and the blend
//!    states below reproduce `shade::blend` exactly.
//! 3. **Readback.** The colour target is copied to a staging texture, mapped,
//!    and laid into the buffer the host handed to the present, in the row layout
//!    it asked for - the same bytes the reference's `to_rgba8_rows` produces.
//!    The frame's colour checksum - the fingerprint the audit compares two
//!    renders of one frame by - is taken from those bytes, and only on the frames
//!    that ask for one, because it is a full pass over the frame that no other
//!    caller reads.
//!
//! The one deliberate fidelity note: D3D11's `R8G8B8A8_UNORM` write truncates
//! where the reference's `to_u8` rounds. That is a property of the hardware
//! format, not of this backend's arithmetic - cascades, bias, filter taps and
//! lighting all use the same inputs and the same order on both tiers, and the
//! golden comparison is a within-tolerance diff for exactly this reason.
//!
//! Every failure is an error return, never a silent fallback, and its code is
//! classified in one place (`classify_hresult`, applied by `err`): a genuine
//! device removal is `DeviceLost`, out-of-memory is `OutOfMemory`, a rejected
//! descriptor is `InvalidArgument`/`NotSupported`, anything unclassifiable is
//! `BackendUnavailable` - never a blanket loss over a healthy device.

use reconl_contract::{FrameInput, ShadowRequest};
use reconl_core::alloc::HostAlloc;
use reconl_core::budget::{Budget, Reservation};
use reconl_core::error::{Code, Error, Result};
use reconl_core::stats::{Counters, FrameNumbers, ShadowCounters};
use reconl_core::tier::{caps, rules, shadow_plan, Backend, ShadowFilter, ShadowPlan, Tier, TierReason};
use reconl_raster::math::{self, Mat4};
use reconl_raster::shade::LightSet;
use reconl_raster::{
    checksum_bytes, rendered_viewport, DrawItem, ShaderRef, Vertex, CULL_BACK, CULL_FRONT, COMPARE_GREATER,
};
use reconl_shadow as shadow;
use std::sync::Arc;
use std::time::Instant;

use windows::core::PCSTR;
use windows::Win32::Foundation::{BOOL, HMODULE};
use windows::Win32::Graphics::Direct3D::{
    ID3DBlob, D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_0,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_SRV_DIMENSION_TEXTURE2DARRAY,
};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11BlendState, ID3D11Buffer, ID3D11ClassLinkage, ID3D11DepthStencilState,
    ID3D11DepthStencilView, ID3D11Device, ID3D11DeviceContext, ID3D11InputLayout,
    ID3D11PixelShader, ID3D11RasterizerState, ID3D11RenderTargetView, ID3D11SamplerState,
    ID3D11ShaderResourceView, ID3D11Texture2D, ID3D11VertexShader, D3D11_BIND_CONSTANT_BUFFER,
    D3D11_BIND_DEPTH_STENCIL, D3D11_BIND_INDEX_BUFFER, D3D11_BIND_RENDER_TARGET,
    D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_VERTEX_BUFFER, D3D11_BLEND, D3D11_BLEND_DESC,
    D3D11_BLEND_DEST_COLOR, D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ONE, D3D11_BLEND_OP_ADD,
    D3D11_BLEND_SRC_ALPHA, D3D11_BLEND_ZERO, D3D11_BUFFER_DESC, D3D11_CLEAR_DEPTH,
    D3D11_CLEAR_STENCIL, D3D11_COLOR_WRITE_ENABLE_ALL, D3D11_COMPARISON_FUNC,
    D3D11_COMPARISON_GREATER, D3D11_CPU_ACCESS_READ, D3D11_CPU_ACCESS_WRITE,
    D3D11_CREATE_DEVICE_FLAG, D3D11_CULL_BACK, D3D11_CULL_FRONT, D3D11_CULL_MODE, D3D11_CULL_NONE,
    D3D11_DEPTH_STENCIL_DESC, D3D11_DEPTH_STENCIL_VIEW_DESC, D3D11_DEPTH_STENCILOP_DESC,
    D3D11_DEPTH_WRITE_MASK_ALL, D3D11_DEPTH_WRITE_MASK_ZERO, D3D11_DSV_DIMENSION_TEXTURE2D,
    D3D11_DSV_DIMENSION_TEXTURE2DARRAY, D3D11_FILTER, D3D11_FILTER_COMPARISON_MIN_MAG_MIP_POINT,
    D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_FILL_SOLID, D3D11_INPUT_ELEMENT_DESC,
    D3D11_INPUT_PER_VERTEX_DATA,
    D3D11_MAP_READ, D3D11_MAP_WRITE_DISCARD, D3D11_MAPPED_SUBRESOURCE, D3D11_RASTERIZER_DESC,
    D3D11_RENDER_TARGET_BLEND_DESC, D3D11_RENDER_TARGET_VIEW_DESC, D3D11_RTV_DIMENSION_TEXTURE2D,
    D3D11_SAMPLER_DESC, D3D11_SDK_VERSION, D3D11_SHADER_RESOURCE_VIEW_DESC,
    D3D11_STENCIL_OP_KEEP, D3D11_SUBRESOURCE_DATA, D3D11_TEX2D_ARRAY_DSV,
    D3D11_TEX2D_ARRAY_SRV, D3D11_TEX2D_DSV, D3D11_TEX2D_RTV, D3D11_TEXTURE2D_DESC,
    D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_USAGE_DEFAULT, D3D11_USAGE_DYNAMIC, D3D11_USAGE_STAGING,
    D3D11_VIEWPORT, D3D11CreateDevice,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT,
    DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32_FLOAT,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1};

use crate::shaders::{DrawCb, LightsCb, ShadowCb, PS_DEPTH_HLSL, PS_HLSL, VS_DEPTH_HLSL, VS_HLSL};

/// Growth step for the per-draw staging buffers, so an oscillating draw size
/// does not re-create them every frame.
const UPLOAD_SLACK: u32 = 4 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct D3d11Config {
    pub tier: Tier,
    /// Index into `probe_adapters`' list. The default adapter when out of range.
    pub adapter_index: usize,
    pub resolution_scale: f32,
    pub target_frame_ms: u32,
    /// Cascade split lambda. Shared with the reference backend's config so both
    /// tiers fit the same cascades: `0` = uniform, `1` = logarithmic.
    pub split_lambda: f32,
    pub shadow: ShadowRequest,
}

impl Default for D3d11Config {
    fn default() -> Self {
        Self {
            tier: Tier::GpuShared,
            adapter_index: 0,
            resolution_scale: 1.0,
            target_frame_ms: 16,
            split_lambda: 0.75,
            shadow: ShadowRequest::default(),
        }
    }
}

/// One GPU adapter, as `reconlProbe` reports it.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub description: String,
    pub dedicated_video_memory: u64,
    pub vendor_id: u32,
    pub feature_level_11: bool,
}

#[derive(Clone, Copy, Default)]
pub struct D3d11Snapshot {
    pub counters: Counters,
    pub shadows: ShadowCounters,
    pub frame: FrameNumbers,
    pub color_checksum: u64,
}

const VERTEX_STRIDE: u32 = std::mem::size_of::<Vertex>() as u32;

/// The vertex layout the shaders consume: position, normal, uv and colour,
/// packed exactly as `reconl_raster::Vertex` is (all `f32`, so the offsets are
/// the same whether or not the struct is `repr(C)`).
fn input_descs() -> [D3D11_INPUT_ELEMENT_DESC; 4] {
    [
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(b"POSITION\0".as_ptr()),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(b"NORMAL\0".as_ptr()),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 12,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(b"TEXCOORD\0".as_ptr()),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 24,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(b"COLOR\0".as_ptr()),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 32,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ]
}

fn sample_desc() -> DXGI_SAMPLE_DESC {
    DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }
}

/// The one place a driver or OS failure becomes a ReconL error.
///
/// A `windows::core::Error` carries an `HRESULT`; the HRESULT *is* the evidence,
/// and [`windows_error`] is where it is read. The mapping is the header's
/// taxonomy (`RECONL_ERR_*`, nothing invented), and the rule is that the code
/// answers the host's one real question - "is my device gone?" - honestly:
///
///   * the DXGI removal family (`DEVICE_REMOVED` / `_RESET` / `_HUNG` /
///     `_DRIVER`), plus `E_FAIL` from a driver that gave up without saying why,
///     is a genuine removal: the only route to `Code::DeviceLost`;
///   * `E_OUTOFMEMORY` is `Code::OutOfMemory`; the device is healthy and a
///     later call may succeed;
///   * `E_INVALIDARG` is `Code::InvalidArgument` and `E_NOTIMPL`/`E_NOINTERFACE`
///     are `Code::NotSupported`: a healthy device rejected the caller's
///     descriptor;
///   * anything unrecognised is `Code::BackendUnavailable` - unknown, not lost.
///
/// The host-facing contract for each class lives next to the `ReconLResult`
/// enum in `include/reconl/reconl.h`.
fn classify_hresult(hresult: i32) -> Code {
    match u32::from_ne_bytes(hresult.to_ne_bytes()) {
        // DXGI device-removal family: REMOVED (0x887A0005), RESET (0x887A0006),
        // HUNG (0x887A0020), DRIVER_INTERNAL_ERROR (0x887A0027), and the
        // standalone DEVICE_REMOVED GetDeviceRemovedReason returns (0x887A0030).
        0x887A0005 | 0x887A0006 | 0x887A0020 | 0x887A0027 | 0x887A0030 => Code::DeviceLost,
        // Classic COM failure codes.
        0x8007000E => Code::OutOfMemory,               // E_OUTOFMEMORY
        0x80070057 => Code::InvalidArgument,           // E_INVALIDARG
        0x80004001 | 0x80004002 => Code::NotSupported, // E_NOTIMPL / E_NOINTERFACE
        // A driver that fails with E_FAIL has given up without a reason; treat
        // that as the removal it usually is.
        0x80004005 => Code::DeviceLost,
        // Anything else: unclassifiable, so the device is not declared dead over it.
        _ => Code::BackendUnavailable,
    }
}

/// The single mapping entry point for the backend's call sites: [`classify_hresult`]
/// reads the driver's HRESULT, the verdict becomes the code, and the failing
/// call's label is attached to the driver's own text. Every
/// `.map_err(|e| err(label, e))` in this backend goes through here, which is
/// what keeps the mapping out of the call sites.
fn err(label: &str, e: windows::core::Error) -> Error {
    let code = classify_hresult(e.code().0);
    Error::fmt_at(
        code,
        file!(),
        line!(),
        "reconl-backend-d3d11",
        format_args!("{label}: {} ({:#010x})", e, e.code().0 as u32),
    )
}

fn null_error(context: &str) -> Error {
    Error::fmt_at(
        Code::BackendUnavailable,
        file!(),
        line!(),
        "reconl-backend-d3d11",
        format_args!("{context}: the driver returned a null object"),
    )
}

fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()) }
}

fn blob_text(blob: &ID3DBlob) -> String {
    String::from_utf8_lossy(blob_bytes(blob)).into_owned()
}

/// Compiles one entry point. A failure carries the driver's own message, which
/// is the only thing that makes a bad shader debuggable.
fn compile(source: &str, entry: PCSTR, target: PCSTR, label: &str) -> Result<ID3DBlob> {
    let mut code = None;
    let mut errors = None;
    let result = unsafe {
        D3DCompile(
            source.as_ptr() as *const core::ffi::c_void,
            source.len(),
            PCSTR(b"reconl\0".as_ptr()),
            None,
            None::<&windows::Win32::Graphics::Direct3D::ID3DInclude>,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    match result {
        Ok(()) => code.ok_or_else(|| null_error(label)),
        Err(e) => {
            let text = errors.map(|blob| blob_text(&blob)).unwrap_or_default();
            Err(Error::fmt_at(
                Code::BackendUnavailable,
                file!(),
                line!(),
                "reconl-backend-d3d11",
                format_args!("{label} failed to compile: {e}\n{text}"),
            ))
        }
    }
}

fn create_buffer(device: &ID3D11Device, desc: &D3D11_BUFFER_DESC, label: &str) -> Result<ID3D11Buffer> {
    let mut out: Option<ID3D11Buffer> = None;
    unsafe { device.CreateBuffer(desc, None, Some(&mut out)) }.map_err(|e| err(label, e))?;
    out.ok_or_else(|| null_error(label))
}

fn constant_buffer(device: &ID3D11Device, bytes: u32, label: &str) -> Result<ID3D11Buffer> {
    create_buffer(
        device,
        &D3D11_BUFFER_DESC {
            ByteWidth: bytes,
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
            MiscFlags: 0,
            StructureByteStride: 0,
        },
        label,
    )
}

fn dynamic_buffer(device: &ID3D11Device, bytes: u32, bind: u32, label: &str) -> Result<ID3D11Buffer> {
    create_buffer(
        device,
        &D3D11_BUFFER_DESC {
            ByteWidth: bytes.max(4),
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: bind,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
            MiscFlags: 0,
            StructureByteStride: 0,
        },
        label,
    )
}

fn create_texture(device: &ID3D11Device, desc: &D3D11_TEXTURE2D_DESC, label: &str) -> Result<ID3D11Texture2D> {
    let mut out: Option<ID3D11Texture2D> = None;
    unsafe { device.CreateTexture2D(desc, None, Some(&mut out)) }.map_err(|e| err(label, e))?;
    out.ok_or_else(|| null_error(label))
}

fn create_rtv(
    device: &ID3D11Device,
    texture: &ID3D11Texture2D,
    desc: &D3D11_RENDER_TARGET_VIEW_DESC,
    label: &str,
) -> Result<ID3D11RenderTargetView> {
    let mut out: Option<ID3D11RenderTargetView> = None;
    unsafe { device.CreateRenderTargetView(texture, Some(desc), Some(&mut out)) }
        .map_err(|e| err(label, e))?;
    out.ok_or_else(|| null_error(label))
}

fn create_dsv(
    device: &ID3D11Device,
    texture: &ID3D11Texture2D,
    desc: &D3D11_DEPTH_STENCIL_VIEW_DESC,
    label: &str,
) -> Result<ID3D11DepthStencilView> {
    let mut out: Option<ID3D11DepthStencilView> = None;
    unsafe { device.CreateDepthStencilView(texture, Some(desc), Some(&mut out)) }
        .map_err(|e| err(label, e))?;
    out.ok_or_else(|| null_error(label))
}

fn create_srv(
    device: &ID3D11Device,
    texture: &ID3D11Texture2D,
    desc: &D3D11_SHADER_RESOURCE_VIEW_DESC,
    label: &str,
) -> Result<ID3D11ShaderResourceView> {
    let mut out: Option<ID3D11ShaderResourceView> = None;
    unsafe { device.CreateShaderResourceView(texture, Some(desc), Some(&mut out)) }
        .map_err(|e| err(label, e))?;
    out.ok_or_else(|| null_error(label))
}

fn raster_state(device: &ID3D11Device, cull: D3D11_CULL_MODE, label: &str) -> Result<ID3D11RasterizerState> {
    let mut out: Option<ID3D11RasterizerState> = None;
    unsafe {
        device.CreateRasterizerState(
            &D3D11_RASTERIZER_DESC {
                FillMode: D3D11_FILL_SOLID,
                CullMode: cull,
                // Which winding the reference calls front. Its rule is
                // `Setup::area2 > 0` over a y-down screen, and D3D11 defines front
                // faces by their screen-space winding too - but the two disagree
                // on the sign, which is not a detail: it inverted every `CULL_*`
                // constant, so a caller asking to cull the far side got the near
                // side culled instead. Measured through the ABI on a scene where
                // the shadow pass culls the far side: with `BOOL(0)` the caster
                // vanished and the hardware tier darkened 0 pixels where the
                // reference darkened 628; with `BOOL(1)` the two agree.
                FrontCounterClockwise: BOOL(1),
                // Bias lives in the pixel shader, applied at lookup, so the
                // depth the shadow pass writes stays unbiased.
                DepthBias: 0,
                DepthBiasClamp: 0.0,
                SlopeScaledDepthBias: 0.0,
                DepthClipEnable: BOOL(1),
                ScissorEnable: BOOL(0),
                MultisampleEnable: BOOL(0),
                AntialiasedLineEnable: BOOL(0),
            },
            Some(&mut out),
        )
    }
    .map_err(|e| err(label, e))?;
    out.ok_or_else(|| null_error(label))
}

/// `test`/`write` select which of the three states the device keeps.
fn depth_state(
    device: &ID3D11Device,
    test: bool,
    write: bool,
    label: &str,
) -> Result<ID3D11DepthStencilState> {
    let keep = D3D11_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D11_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D11_STENCIL_OP_KEEP,
        StencilPassOp: D3D11_STENCIL_OP_KEEP,
        StencilFunc: D3D11_COMPARISON_GREATER,
    };
    let mut out: Option<ID3D11DepthStencilState> = None;
    unsafe {
        device.CreateDepthStencilState(
            &D3D11_DEPTH_STENCIL_DESC {
                DepthEnable: BOOL(test as i32),
                DepthWriteMask: if write {
                    D3D11_DEPTH_WRITE_MASK_ALL
                } else {
                    D3D11_DEPTH_WRITE_MASK_ZERO
                },
                // Reversed-Z with a strict comparison, matching the
                // rasteriser's `keep = depth > stored`.
                DepthFunc: D3D11_COMPARISON_GREATER,
                StencilEnable: BOOL(0),
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
            },
            Some(&mut out),
        )
    }
    .map_err(|e| err(label, e))?;
    out.ok_or_else(|| null_error(label))
}

/// One `shade::blend` mode as fixed-function state.
///
/// The shader pre-multiplies rgb by alpha for the additive and multiply modes
/// (see `shaders.rs`), which is what lets each of the reference's hand-written
/// formulas be expressed exactly:
///
/// | mode | reference | state |
/// |------|-----------|-------|
/// | opaque | `dst = src`, alpha 1 | blending off |
/// | alpha | `src·a + dst·(1-a)` | `SRC_ALPHA / INV_SRC_ALPHA` |
/// | additive | `dst + src·a` | `ONE / ONE`, shader emits `src·a` |
/// | multiply | `dst·(1-a) + dst·src·a` | `DEST_COLOR / INV_SRC_ALPHA`, shader emits `src·a` |
fn blend_state(device: &ID3D11Device, mode: u32, label: &str) -> Result<ID3D11BlendState> {
    let (enable, src, dst, src_a, dst_a): (bool, D3D11_BLEND, D3D11_BLEND, D3D11_BLEND, D3D11_BLEND) =
        match mode {
            1 => (true, D3D11_BLEND_SRC_ALPHA, D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ONE, D3D11_BLEND_INV_SRC_ALPHA),
            2 => (true, D3D11_BLEND_ONE, D3D11_BLEND_ONE, D3D11_BLEND_ONE, D3D11_BLEND_ONE),
            // `blend()` leaves the destination alpha alone in multiply mode.
            3 => (true, D3D11_BLEND_DEST_COLOR, D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ZERO, D3D11_BLEND_ONE),
            _ => (false, D3D11_BLEND_ONE, D3D11_BLEND_ZERO, D3D11_BLEND_ONE, D3D11_BLEND_ZERO),
        };
    let target = D3D11_RENDER_TARGET_BLEND_DESC {
        BlendEnable: BOOL(enable as i32),
        SrcBlend: src,
        DestBlend: dst,
        BlendOp: D3D11_BLEND_OP_ADD,
        SrcBlendAlpha: src_a,
        DestBlendAlpha: dst_a,
        BlendOpAlpha: D3D11_BLEND_OP_ADD,
        RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };
    let desc = D3D11_BLEND_DESC {
        AlphaToCoverageEnable: BOOL(0),
        IndependentBlendEnable: BOOL(0),
        // Only slot 0 is ever bound, and the descriptor is the same for every
        // slot, so the state cannot depend on which bound it.
        RenderTarget: [target; 8],
    };
    let mut out: Option<ID3D11BlendState> = None;
    unsafe { device.CreateBlendState(&desc, Some(&mut out)) }.map_err(|e| err(label, e))?;
    out.ok_or_else(|| null_error(label))
}

fn sampler_state(
    device: &ID3D11Device,
    filter: D3D11_FILTER,
    func: D3D11_COMPARISON_FUNC,
    label: &str,
) -> Result<ID3D11SamplerState> {
    let mut out: Option<ID3D11SamplerState> = None;
    unsafe {
        device.CreateSamplerState(
            &D3D11_SAMPLER_DESC {
                Filter: filter,
                AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: func,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: f32::MAX,
            },
            Some(&mut out),
        )
    }
    .map_err(|e| err(label, e))?;
    out.ok_or_else(|| null_error(label))
}

/// The colour target, its depth buffer, and the staging copies readback uses.
struct ColorTarget {
    width: u32,
    height: u32,
    color: ID3D11Texture2D,
    rtv: ID3D11RenderTargetView,
    /// Held for the depth/stencil view that binds it, like `_white` below: the
    /// view is what the passes use, and the texture has to outlive the resize
    /// that would otherwise drop it.
    _depth: ID3D11Texture2D,
    dsv: ID3D11DepthStencilView,
    color_staging: ID3D11Texture2D,
    /// A second staging copy, of the *depth* buffer, for the one caller that
    /// reads it: frame generation reprojects pixels by their depth. Only
    /// allocated alongside the target the way the colour copy is, and only used
    /// when a host asked for generated frames.
    depth_staging: ID3D11Texture2D,
    reservation: Option<Reservation>,
}

/// The cascades as one `R32_TYPELESS` array: one depth slice and one DSV per
/// cascade, and a single `R32_FLOAT` array SRV that serves both the comparison
/// taps and the PCSS blocker search.
struct ShadowMaps {
    views: Vec<ID3D11DepthStencilView>,
    srv: ID3D11ShaderResourceView,
    size: u32,
    count: u32,
    reservation: Option<Reservation>,
}

pub struct D3d11Device {
    alloc: HostAlloc,
    budget: Arc<Budget>,
    config: D3d11Config,
    tier: Tier,
    tier_reason: TierReason,

    device: ID3D11Device,
    context: ID3D11DeviceContext,
    adapter_name: String,
    vendor_id: u32,
    dedicated_video_memory: u64,

    vs: ID3D11VertexShader,
    ps: ID3D11PixelShader,
    vs_depth: ID3D11VertexShader,
    ps_depth: ID3D11PixelShader,
    layout: ID3D11InputLayout,

    draw_cb: ID3D11Buffer,
    lights_cb: ID3D11Buffer,
    shadow_cb: ID3D11Buffer,
    vertex_upload: ID3D11Buffer,
    vertex_capacity: u32,
    index_upload: ID3D11Buffer,
    index_capacity: u32,

    /// Indexed by `CULL_NONE`/`CULL_BACK`/`CULL_FRONT`.
    raster: [ID3D11RasterizerState; 3],
    /// Indexed by `BLEND_*`.
    blend: [ID3D11BlendState; 4],
    depth_test_write: ID3D11DepthStencilState,
    depth_test_only: ID3D11DepthStencilState,
    depth_off: ID3D11DepthStencilState,
    point_sampler: ID3D11SamplerState,
    cmp_sampler: ID3D11SamplerState,
    /// The 1x1 white array bound where a base texture would go: the ABI's draw
    /// path carries no texture payload, so a "textured" draw reads white here
    /// exactly as it does through the reference rasteriser.
    white_srv: ID3D11ShaderResourceView,
    _white: ID3D11Texture2D,

    color: Option<ColorTarget>,
    maps: Option<ShadowMaps>,
    /// Tightly packed RGBA8 of a frame, filled by `capture`. Both `pixels` and
    /// the colour checksum are taken from these bytes, so the two can never
    /// describe different frames.
    rgba8: Vec<u8>,
    /// The frame's depth, tightly packed, filled by `depth_into` for frame
    /// generation. Kept beside `rgba8` for the same reason: one buffer sized by
    /// the target rather than one allocation per read.
    depth_f32: Vec<f32>,
    /// Which frame `rgba8` holds, so the frame is copied out of the driver once:
    /// an audit capture and a present of the same frame share one map.
    captured: Option<u64>,

    counters: Counters,
    shadows: ShadowCounters,
    frame: FrameNumbers,
    color_checksum: u64,
}

impl D3d11Device {
    pub fn new(alloc: HostAlloc, budget: Arc<Budget>, config: D3d11Config) -> Result<Self> {
        alloc.self_check()?;
        let adapters = enumerate_adapters()?;
        let chosen = if config.adapter_index < adapters.len() {
            &adapters[config.adapter_index]
        } else {
            adapters
                .first()
                .ok_or_else(|| Error::new(Code::BackendUnavailable, "no D3D11 adapter was found"))?
        };
        let desc = unsafe { chosen.GetDesc1() }.map_err(|e| err("adapter desc", e))?;
        let name_len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        let adapter_name = String::from_utf16_lossy(&desc.Description[..name_len]);

        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                chosen,
                D3D_DRIVER_TYPE_UNKNOWN, // an adapter was named, so the type must be UNKNOWN
                HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(0),
                Some(&[D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        }
        .map_err(|e| err("d3d11 create device", e))?;
        let device = device.ok_or_else(|| null_error("d3d11 create device"))?;
        let context = context.ok_or_else(|| null_error("d3d11 create device"))?;

        let vs_blob = compile(VS_HLSL, PCSTR(b"main\0".as_ptr()), PCSTR(b"vs_5_0\0".as_ptr()), "vertex shader")?;
        let ps_blob = compile(PS_HLSL, PCSTR(b"main\0".as_ptr()), PCSTR(b"ps_5_0\0".as_ptr()), "pixel shader")?;
        let vs_depth_blob = compile(VS_DEPTH_HLSL, PCSTR(b"main\0".as_ptr()), PCSTR(b"vs_5_0\0".as_ptr()), "shadow vertex shader")?;
        let ps_depth_blob = compile(PS_DEPTH_HLSL, PCSTR(b"main\0".as_ptr()), PCSTR(b"ps_5_0\0".as_ptr()), "shadow pixel shader")?;

        let mut out: Option<ID3D11VertexShader> = None;
        unsafe {
            device.CreateVertexShader(blob_bytes(&vs_blob), None::<&ID3D11ClassLinkage>, Some(&mut out))
        }
        .map_err(|e| err("create vertex shader", e))?;
        let vs = out.ok_or_else(|| null_error("create vertex shader"))?;

        let mut out: Option<ID3D11PixelShader> = None;
        unsafe {
            device.CreatePixelShader(blob_bytes(&ps_blob), None::<&ID3D11ClassLinkage>, Some(&mut out))
        }
        .map_err(|e| err("create pixel shader", e))?;
        let ps = out.ok_or_else(|| null_error("create pixel shader"))?;

        let mut out: Option<ID3D11VertexShader> = None;
        unsafe {
            device.CreateVertexShader(blob_bytes(&vs_depth_blob), None::<&ID3D11ClassLinkage>, Some(&mut out))
        }
        .map_err(|e| err("create shadow vertex shader", e))?;
        let vs_depth = out.ok_or_else(|| null_error("create shadow vertex shader"))?;

        let mut out: Option<ID3D11PixelShader> = None;
        unsafe {
            device.CreatePixelShader(blob_bytes(&ps_depth_blob), None::<&ID3D11ClassLinkage>, Some(&mut out))
        }
        .map_err(|e| err("create shadow pixel shader", e))?;
        let ps_depth = out.ok_or_else(|| null_error("create shadow pixel shader"))?;

        let mut out: Option<ID3D11InputLayout> = None;
        unsafe { device.CreateInputLayout(&input_descs(), blob_bytes(&vs_blob), Some(&mut out)) }
            .map_err(|e| err("create input layout", e))?;
        let layout = out.ok_or_else(|| null_error("create input layout"))?;

        let draw_cb = constant_buffer(&device, std::mem::size_of::<DrawCb>() as u32, "draw constant buffer")?;
        let lights_cb = constant_buffer(&device, std::mem::size_of::<LightsCb>() as u32, "lights constant buffer")?;
        let shadow_cb = constant_buffer(&device, std::mem::size_of::<ShadowCb>() as u32, "shadow constant buffer")?;
        let vertex_capacity = 1 << 16;
        let index_capacity = 1 << 17;
        let vertex_upload = dynamic_buffer(
            &device,
            vertex_capacity * VERTEX_STRIDE,
            D3D11_BIND_VERTEX_BUFFER.0 as u32,
            "vertex upload buffer",
        )?;
        let index_upload = dynamic_buffer(
            &device,
            index_capacity * 4,
            D3D11_BIND_INDEX_BUFFER.0 as u32,
            "index upload buffer",
        )?;

        let raster = [
            raster_state(&device, D3D11_CULL_NONE, "rasteriser state (no cull)")?,
            raster_state(&device, D3D11_CULL_BACK, "rasteriser state (cull back)")?,
            raster_state(&device, D3D11_CULL_FRONT, "rasteriser state (cull front)")?,
        ];
        let blend = [
            blend_state(&device, 0, "blend state (opaque)")?,
            blend_state(&device, 1, "blend state (alpha)")?,
            blend_state(&device, 2, "blend state (additive)")?,
            blend_state(&device, 3, "blend state (multiply)")?,
        ];
        let depth_test_write = depth_state(&device, true, true, "depth state (test+write)")?;
        let depth_test_only = depth_state(&device, true, false, "depth state (test only)")?;
        let depth_off = depth_state(&device, false, false, "depth state (off)")?;
        // Point sampling everywhere: the reference reads discrete texels for
        // both its PCF taps and its PCSS blocker search, so no implicit bilinear
        // filtering may be layered on top of its own tap grid.
        let point_sampler = sampler_state(
            &device,
            D3D11_FILTER_MIN_MAG_MIP_POINT,
            D3D11_COMPARISON_GREATER,
            "point sampler",
        )?;
        let cmp_sampler = sampler_state(
            &device,
            D3D11_FILTER_COMPARISON_MIN_MAG_MIP_POINT,
            D3D11_COMPARISON_GREATER,
            "comparison sampler",
        )?;

        let white_pixels: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
        let init = D3D11_SUBRESOURCE_DATA {
            pSysMem: white_pixels.as_ptr() as *const core::ffi::c_void,
            SysMemPitch: 4,
            SysMemSlicePitch: 4,
        };
        let mut white_tex: Option<ID3D11Texture2D> = None;
        unsafe {
            device.CreateTexture2D(
                &D3D11_TEXTURE2D_DESC {
                    Width: 1,
                    Height: 1,
                    MipLevels: 1,
                    ArraySize: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: sample_desc(),
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                    CPUAccessFlags: 0,
                    MiscFlags: 0,
                },
                Some(&init),
                Some(&mut white_tex),
            )
        }
        .map_err(|e| err("white texture", e))?;
        let white = white_tex.ok_or_else(|| null_error("white texture"))?;
        let white_srv = create_srv(
            &device,
            &white,
            &D3D11_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D_SRV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: windows::Win32::Graphics::Direct3D11::D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D11_TEX2D_ARRAY_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        FirstArraySlice: 0,
                        ArraySize: 1,
                    },
                },
            },
            "white srv",
        )?;

        Ok(Self {
            alloc,
            budget,
            tier: config.tier,
            tier_reason: TierReason::HostRequest,
            config,
            device,
            context,
            adapter_name,
            vendor_id: desc.VendorId,
            dedicated_video_memory: desc.DedicatedVideoMemory as u64,
            vs,
            ps,
            vs_depth,
            ps_depth,
            layout,
            draw_cb,
            lights_cb,
            shadow_cb,
            vertex_upload,
            vertex_capacity,
            index_upload,
            index_capacity,
            raster,
            blend,
            depth_test_write,
            depth_test_only,
            depth_off,
            point_sampler,
            cmp_sampler,
            white_srv,
            _white: white,
            color: None,
            maps: None,
            rgba8: Vec::new(),
            depth_f32: Vec::new(),
            captured: None,
            counters: Counters::default(),
            shadows: ShadowCounters::default(),
            frame: FrameNumbers::default(),
            color_checksum: 0,
        })
    }

    pub fn backend(&self) -> Backend {
        Backend::D3d11
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    pub fn tier_reason(&self) -> TierReason {
        self.tier_reason
    }

    pub fn caps(&self) -> u32 {
        caps_for(self.tier)
    }

    pub fn device_name(&self) -> &'static str {
        "d3d11 (hardware rasteriser)"
    }

    pub fn driver(&self) -> &'static str {
        "d3d11"
    }

    /// The enumerator's own name for the adapter, so a host can tell which GPU
    /// the tier actually landed on.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn vendor_id(&self) -> u32 {
        self.vendor_id
    }

    pub fn dedicated_video_memory(&self) -> u64 {
        self.dedicated_video_memory
    }

    pub fn host_alloc(&self) -> HostAlloc {
        self.alloc
    }

    pub fn resolution_scale(&self) -> f32 {
        self.config.resolution_scale.min(rules(self.tier).resolution_scale)
    }

    /// Whether the driver reports this device as removed or reset.
    ///
    /// A second, independent verdict from the classification: a failure whose
    /// code maps to `DeviceLost` happens only where this returns true, and the
    /// failover trigger in the ffi checks this before moving a device, so a
    /// misclassification in either direction cannot abandon a healthy GPU or
    /// keep rendering on a dead one.
    pub fn device_removed(&self) -> bool {
        // SAFETY: `device` is a live ID3D11Device; the call has no arguments and
        // is safe to make from any state, including after a failed draw.
        unsafe { self.device.GetDeviceRemovedReason() }.is_err()
    }

    pub fn counters(&self) -> Counters {
        self.counters
    }

    pub fn shadows(&self) -> ShadowCounters {
        self.shadows
    }

    pub fn snapshot(&self) -> D3d11Snapshot {
        D3d11Snapshot {
            counters: self.counters,
            shadows: self.shadows,
            frame: self.frame,
            color_checksum: self.color_checksum,
        }
    }

    pub fn color_checksum(&self) -> u64 {
        self.color_checksum
    }

    pub fn frame_size(&self) -> (u32, u32) {
        self.color.as_ref().map(|c| (c.width, c.height)).unwrap_or((1, 1))
    }

    /// Resident bytes this device is accountable for on the CPU side: the
    /// colour/depth targets with their staging copies, and the shadow array.
    pub fn resident_bytes(&self) -> u64 {
        self.color.as_ref().and_then(|c| c.reservation.as_ref()).map(|r| r.bytes()).unwrap_or(0)
            + self.maps.as_ref().and_then(|m| m.reservation.as_ref()).map(|r| r.bytes()).unwrap_or(0)
    }

    /// Applies a tier the *device* decided on: the same backend at a lower
    /// quality tier, with everything a tier change invalidates dropped.
    ///
    /// The backend does not decide this and does not record it: a device's tier
    /// has one owner (the frame-time ladder in `ffi/src/offload.rs`, `docs/offload.md`)
    /// and one log, and a backend's own ring would die with the backend while the
    /// tier it changed does not.
    pub fn relabel(&mut self, to: Tier, reason: TierReason) {
        self.tier = to;
        self.tier_reason = reason;
        self.counters.frames_since_tier_change = 0;
        // A tier change invalidates the shadow layout (the new tier has a
        // different cascade cap, filter cap and cache location), so the maps are
        // dropped and rebuilt by the next frame.
        self.maps = None;
    }

    /// Flushes the immediate context. Called at frame boundaries only.
    pub fn on_frame_end(&mut self) -> Result<()> {
        unsafe { self.context.Flush() };
        Ok(())
    }

    /// Reserves everything the next frame needs, between frames: the colour and
    /// depth targets, their staging copies, and the shadow array.
    pub fn prepare_frame(&mut self, width: u32, height: u32) -> Result<()> {
        self.ensure_targets(width, height)?;
        if self.config.shadow.enabled && (self.caps() & caps::SHADOWS) != 0 {
            let plan = shadow_plan(
                self.tier,
                self.config.shadow.cascades,
                self.config.shadow.texel_budget_bytes,
                self.config.shadow.filter,
                self.caps(),
            );
            self.ensure_maps(&plan)?;
        }
        Ok(())
    }

    fn ensure_targets(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(Error::new(Code::InvalidArgument, "zero-sized frame"));
        }
        let scale = self.resolution_scale();
        let target_w = (((width as f32) * scale).round() as u32).clamp(1, width);
        let target_h = (((height as f32) * scale).round() as u32).clamp(1, height);
        if let Some(c) = &self.color {
            if c.width == target_w && c.height == target_h {
                return Ok(());
            }
        }
        // Colour, depth and their two staging copies - including the depth
        // staging, which only frame generation reads - counted against the
        // budget the same way the reference counts its CPU targets.
        let bytes = u64::from(target_w) * u64::from(target_h) * (4 + 4) * 2;
        let reservation = self.budget.reserve_ram(bytes).map_err(|e| {
            self.counters.safe_path_events += 1;
            e
        })?;

        let color_desc = D3D11_TEXTURE2D_DESC {
            Width: target_w,
            Height: target_h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: sample_desc(),
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE).0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let color = create_texture(&self.device, &color_desc, "colour texture")?;
        let rtv = create_rtv(
            &self.device,
            &color,
            &D3D11_RENDER_TARGET_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D11_RTV_DIMENSION_TEXTURE2D,
                Anonymous: windows::Win32::Graphics::Direct3D11::D3D11_RENDER_TARGET_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_RTV { MipSlice: 0 },
                },
            },
            "colour render target view",
        )?;

        // The depth buffer needs its own bind flags: a colour texture's
        // render-target/shader-resource flags are not legal on a typeless depth
        // format.
        let depth = create_texture(
            &self.device,
            &D3D11_TEXTURE2D_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                BindFlags: D3D11_BIND_DEPTH_STENCIL.0 as u32,
                ..color_desc
            },
            "depth texture",
        )?;
        let dsv = create_dsv(
            &self.device,
            &depth,
            &D3D11_DEPTH_STENCIL_VIEW_DESC {
                Format: DXGI_FORMAT_D32_FLOAT,
                ViewDimension: D3D11_DSV_DIMENSION_TEXTURE2D,
                Flags: 0,
                Anonymous: windows::Win32::Graphics::Direct3D11::D3D11_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_DSV { MipSlice: 0 },
                },
            },
            "depth stencil view",
        )?;

        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            ..color_desc
        };
        let color_staging = create_texture(&self.device, &staging_desc, "colour staging texture")?;
        // Same shape as the depth texture it copies from, so `CopyResource` is
        // legal: both are `R32_TYPELESS` and neither has a view applied.
        let depth_staging = create_texture(
            &self.device,
            &D3D11_TEXTURE2D_DESC { Format: DXGI_FORMAT_R32_TYPELESS, ..staging_desc },
            "depth staging texture",
        )?;

        self.rgba8.clear();
        self.depth_f32.clear();
        self.rgba8.resize((target_w as usize) * (target_h as usize) * 4, 0);
        self.color = Some(ColorTarget {
            width: target_w,
            height: target_h,
            color,
            rtv,
            _depth: depth,
            dsv,
            color_staging,
            depth_staging,
            reservation: Some(reservation),
        });
        Ok(())
    }

    /// Allocates the cascade set the plan decided on, reserving what the plan
    /// says it costs. The plan owns the size, so the hardware tier and the
    /// reference tier allocate the same maps for the same budget without a
    /// comment keeping two copies of the arithmetic in step.
    fn ensure_maps(&mut self, plan: &ShadowPlan) -> Result<()> {
        let (cascades, size) = (plan.cascades, plan.map_size);
        if let Some(m) = &self.maps {
            if m.count == cascades && m.size == size {
                return Ok(());
            }
        }
        let bytes = plan.resident_bytes;
        let reservation = self.budget.reserve_ram(bytes).map_err(|e| {
            self.counters.safe_path_events += 1;
            e
        })?;
        let texture = create_texture(
            &self.device,
            &D3D11_TEXTURE2D_DESC {
                Width: size,
                Height: size,
                MipLevels: 1,
                ArraySize: cascades,
                Format: DXGI_FORMAT_R32_TYPELESS,
                SampleDesc: sample_desc(),
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: (D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE).0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            },
            "shadow map array",
        )?;
        let mut views = Vec::with_capacity(cascades as usize);
        for slice in 0..cascades {
            views.push(create_dsv(
                &self.device,
                &texture,
                &D3D11_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    ViewDimension: D3D11_DSV_DIMENSION_TEXTURE2DARRAY,
                    Flags: 0,
                    Anonymous: windows::Win32::Graphics::Direct3D11::D3D11_DEPTH_STENCIL_VIEW_DESC_0 {
                        Texture2DArray: D3D11_TEX2D_ARRAY_DSV {
                            MipSlice: 0,
                            FirstArraySlice: slice,
                            ArraySize: 1,
                        },
                    },
                },
                "shadow cascade view",
            )?);
        }
        let srv = create_srv(
            &self.device,
            &texture,
            &D3D11_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_FLOAT,
                ViewDimension: D3D_SRV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: windows::Win32::Graphics::Direct3D11::D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D11_TEX2D_ARRAY_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        FirstArraySlice: 0,
                        ArraySize: cascades,
                    },
                },
            },
            "shadow array srv",
        )?;
        self.maps = Some(ShadowMaps {
            views,
            srv,
            size,
            count: cascades,
            reservation: Some(reservation),
        });
        Ok(())
    }

    fn write_constant<T: Copy>(&self, buffer: &ID3D11Buffer, value: &T, label: &str) -> Result<()> {
        write_mapped(&self.context, buffer, value, label)
    }

    fn draw_depth(&mut self, transform: Mat4, draw: &DrawItem<'_>) -> Result<()> {
        let cb = DrawCb { transform, flags: [0.0; 4], flags1: [0.0; 4] };
        write_mapped(&self.context, &self.draw_cb, &cb, "draw constants (shadow pass)")?;
        self.bind_geometry(draw)?;
        unsafe {
            match draw.indices {
                Some(indices) => self.context.DrawIndexed(indices.len() as u32, 0, 0),
                None => self.context.Draw(draw.vertices.len() as u32, 0),
            }
        }
        Ok(())
    }

    fn draw_surface(
        &mut self,
        draw: &DrawItem<'_>,
        maps: Option<&ID3D11ShaderResourceView>,
    ) -> Result<()> {
        let surface = match draw.shader {
            ShaderRef::Surface(surface) => Some(surface),
            ShaderRef::DepthOnly => None,
        };
        let blend = (draw.pipeline.blend & 3) as usize;
        let cb = DrawCb {
            transform: draw.transform,
            flags: [
                surface.map(|s| s.textured as u32 as f32).unwrap_or(0.0),
                surface.map(|s| s.lit as u32 as f32).unwrap_or(0.0),
                surface.map(|s| s.receives_shadow as u32 as f32).unwrap_or(0.0),
                blend as f32,
            ],
            flags1: [draw.pipeline.two_sided as u32 as f32, 0.0, 0.0, 0.0],
        };
        write_mapped(&self.context, &self.draw_cb, &cb, "draw constants")?;
        self.bind_geometry(draw)?;

        let depth = if !draw.pipeline.depth_test {
            &self.depth_off
        } else if draw.pipeline.depth_write && draw.pipeline.depth_compare == COMPARE_GREATER {
            &self.depth_test_write
        } else {
            &self.depth_test_only
        };
        let raster = &self.raster[cull_index(draw.pipeline.cull)];
        let white = self.white_srv.clone();
        let shadow = maps.cloned();
        unsafe {
            self.context.RSSetState(raster);
            self.context.OMSetDepthStencilState(depth, 0);
            self.context.OMSetBlendState(&self.blend[blend], None, 0xffff_ffff);
            self.context
                .PSSetShaderResources(0, Some(&[Some(white), shadow.clone(), shadow]));
            match draw.indices {
                Some(indices) => self.context.DrawIndexed(indices.len() as u32, 0, 0),
                None => self.context.Draw(draw.vertices.len() as u32, 0),
            }
        }
        Ok(())
    }

    /// Uploads the draw's vertices (and indices) and binds them.
    fn bind_geometry(&mut self, draw: &DrawItem<'_>) -> Result<()> {
        upload_into(
            &self.device,
            &self.context,
            &mut self.vertex_upload,
            &mut self.vertex_capacity,
            D3D11_BIND_VERTEX_BUFFER.0 as u32,
            draw.vertices,
            "vertex upload buffer",
        )?;
        if let Some(indices) = draw.indices {
            upload_into(
                &self.device,
                &self.context,
                &mut self.index_upload,
                &mut self.index_capacity,
                D3D11_BIND_INDEX_BUFFER.0 as u32,
                indices,
                "index upload buffer",
            )?;
        }
        let vertex_buffer = [Some(self.vertex_upload.clone())];
        let strides = [VERTEX_STRIDE];
        let offsets = [0u32];
        unsafe {
            self.context.IASetVertexBuffers(
                0,
                1,
                Some(vertex_buffer.as_ptr()),
                Some(strides.as_ptr()),
                Some(offsets.as_ptr()),
            );
            if draw.indices.is_some() {
                self.context
                    .IASetIndexBuffer(&self.index_upload, DXGI_FORMAT_R32_UINT, 0);
            }
        }
        Ok(())
    }    /// Copies the colour target to staging, maps it, and lays the rows into
    /// `out` - honouring `pitch` and the flip the present asked for. The only
    /// place a frame leaves the driver.
    fn copy_target_rows(
        context: &ID3D11DeviceContext,
        target: &ColorTarget,
        out: &mut [u8],
        pitch: usize,
        flip: bool,
    ) -> Result<()> {
        let rows = target.height as usize;
        let row_bytes = target.width as usize * 4;

        unsafe { context.CopyResource(&target.color_staging, &target.color) };
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(&target.color_staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) }
            .map_err(|e| err("map colour staging texture", e))?;
        for row in 0..rows {
            let source = if flip { rows - 1 - row } else { row };
            let src = unsafe {
                (mapped.pData as *const u8).add(source * mapped.RowPitch as usize)
            };
            let bytes = unsafe { std::slice::from_raw_parts(src, row_bytes) };
            let at = row * pitch;
            out[at..at + row_bytes].copy_from_slice(bytes);
        }
        unsafe { context.Unmap(&target.color_staging, 0) };
        Ok(())
    }

    /// (Re)lays tightly packed rows into a destination with a row pitch and
    /// possibly a flip.
    fn lay_out_rows(
        out: &mut [u8],
        src: &[u8],
        rows: usize,
        row_bytes: usize,
        pitch: usize,
        flip: bool,
    ) {
        for row in 0..rows {
            let source = if flip { rows - 1 - row } else { row };
            let at = row * pitch;
            let from = source * row_bytes;
            out[at..at + row_bytes].copy_from_slice(&src[from..from + row_bytes]);
        }
    }

    /// Copies the frame's depth out of the driver into `out`, tightly packed,
    /// `width * height` values.
    ///
    /// This is the one thing frame generation needs that a present does not: the
    /// depth buffer, which tells the generator where each pixel's content sits in
    /// space. It is a driver readback like the colour one, so it costs a copy - 
    /// which is why nothing calls it unless a host asked for generated frames.
    pub fn depth_into(&mut self, out: &mut [f32]) -> Result<()> {
        let target = self
            .color
            .as_ref()
            .ok_or_else(|| Error::new(Code::NotReady, "no frame has been prepared"))?;
        let rows = target.height as usize;
        let row_f32 = target.width as usize;
        if out.len() < rows * row_f32 {
            return Err(Error::new(Code::InvalidArgument, "the depth buffer is too small for the frame"));
        }
        unsafe {
            self.context.CopyResource(&target.depth_staging, &target._depth);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context
                .Map(&target.depth_staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|e| err("map depth staging texture", e))?;
            for row in 0..rows {
                // SAFETY: the mapped region spans `RowPitch * height` bytes and
                // holds one `f32` per pixel of the row.
                let src = (mapped.pData as *const u8).add(row * mapped.RowPitch as usize) as *const f32;
                let dst = &mut out[row * row_f32..row * row_f32 + row_f32];
                dst.copy_from_slice(core::slice::from_raw_parts(src, row_f32));
            }
            self.context.Unmap(&target.depth_staging, 0);
        }
        Ok(())
    }

    /// Reads the frame out of the driver into the tightly packed `rgba8`, the
    /// form the checksum hashes. `frame_index` records which frame it holds.
    fn capture(&mut self, frame_index: u64) -> Result<()> {
        let Some(target) = self.color.as_ref() else {
            return Ok(());
        };
        let row_bytes = target.width as usize * 4;
        self.rgba8.resize(row_bytes * target.height as usize, 0);
        Self::copy_target_rows(&self.context, target, &mut self.rgba8, row_bytes, false)?;
        self.captured = Some(frame_index);
        Ok(())
    }

    /// Writes the last rendered frame into the host's presentation buffer, in
    /// the layout the present asked for: `pitch` is the destination's row length
    /// in bytes, `flip` reverses the row order, and a zero pitch means tightly
    /// packed.
    ///
    /// The frame is read out of the driver here, on the frame a host asks for
    /// it, rather than on every submit - a host that renders to a swapchain never
    /// pays for a readback it does not read - and it is laid straight into the
    /// host's own buffer, so there is no tightly packed copy of the frame
    /// between the driver's staging texture and the host.
    pub fn read_frame_into(&mut self, out: &mut [u8], pitch: u32, flip: u32) -> Result<()> {
        let Some(target) = self.color.as_ref() else {
            return Err(Error::new(Code::NotReady, "no frame has been rendered"));
        };
        let rows = target.height as usize;
        let row_bytes = target.width as usize * 4;
        let pitch = if pitch == 0 { row_bytes } else { pitch as usize };
        let needed = rows
            .saturating_sub(1)
            .saturating_mul(pitch)
            .saturating_add(row_bytes);
        if pitch < row_bytes || out.len() < needed {
            return Err(Error::new(Code::InvalidArgument, "readback buffer is too small"));
        }
        if self.captured == Some(self.frame.frame_index) {
            // An audit already read this frame out of the driver: those bytes are
            // this frame's pixels, so laying them out is cheaper than asking the
            // driver again for the same frame. `rgba8` is tightly packed, so its
            // row stride is a row.
            Self::lay_out_rows(out, &self.rgba8, rows, row_bytes, pitch, flip != 0);
            return Ok(());
        }
        Self::copy_target_rows(&self.context, target, out, pitch, flip != 0)
    }

    /// Renders one frame. `prepare_frame` has already reserved the targets, so
    /// the only growth left inside the frame is the per-draw upload buffers.
    pub fn render(&mut self, input: &FrameInput<'_>) -> Result<FrameNumbers> {
        let frame_start = Instant::now();
        self.ensure_targets(input.width, input.height)?;
        let (width, height) = self.frame_size();

        let plan = if input.shadow.enabled && (self.caps() & caps::SHADOWS) != 0 {
            Some(shadow_plan(
                self.tier,
                input.shadow.cascades,
                input.shadow.texel_budget_bytes,
                input.shadow.filter,
                self.caps(),
            ))
        } else {
            None
        };

        // ---- shadow pass -----------------------------------------------------
        let shadow_start = Instant::now();
        let mut fit_ns = 0u64;
        let mut cascades_rendered = 0u32;
        let mut shadowed_triangles = 0u64;
        let fits = if let Some(p) = plan {
            self.ensure_maps(&p)?;
            let size = p.map_size;
            let fit_start = Instant::now();
            let fits = shadow::fit_cascades(&shadow::FitInput {
                camera_view: input.camera_view,
                fov_y_deg: input.fov_y_deg,
                aspect: input.aspect,
                near: input.near,
                light_dir: input.light_dir,
                cascade_count: p.cascades,
                max_distance: input.shadow.max_distance,
                split_lambda: self.config.split_lambda,
                map_size: size,
                snap: true,
            });
            fit_ns += fit_start.elapsed().as_nanos() as u64;

            self.shadows.map_width = size;
            self.shadows.map_height = size;
            self.shadows.map_bytes = p.resident_bytes;
            self.shadows.cascades_active = p.cascades;
            self.shadows.filter_active = p.filter;
            self.shadows.filter_requested = p.filter_requested;
            self.shadows.filter_taps = p.filter.taps();
            self.shadows.shadowed_lights = count_shadowed_lights(&input.lights);

            let (views, map_srv) = {
                let maps = self.maps.as_ref().ok_or_else(|| {
                    Error::new(Code::NotReady, "the shadow maps were not prepared")
                })?;
                (maps.views.clone(), maps.srv.clone())
            };
            let shadow_cull = &self.raster[cull_index(CULL_BACK)];
            let shadow_depth = &self.depth_test_write;
            let draw_cb = self.draw_cb.clone();
            unsafe {
                self.context.IASetInputLayout(&self.layout);
                self.context
                    .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                self.context.VSSetShader(&self.vs_depth, None);
                self.context.PSSetShader(&self.ps_depth, None);
                self.context.VSSetConstantBuffers(0, Some(&[Some(draw_cb)]));
                self.context.RSSetState(shadow_cull);
                self.context.OMSetBlendState(&self.blend[0], None, 0xffff_ffff);
                self.context.OMSetDepthStencilState(shadow_depth, 0);
                self.context.RSSetViewports(Some(&[D3D11_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: size as f32,
                    Height: size as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]));
            }
            for (index, fit) in fits.iter().enumerate() {
                let view = &views[index];
                unsafe {
                    self.context.OMSetRenderTargets(None, view);
                    self.context.ClearDepthStencilView(
                        view,
                        (D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL).0 as u32,
                        0.0,
                        0,
                    );
                }
                for draw in input.draws.iter() {
                    if !draw.casts_shadow {
                        continue;
                    }
                    shadowed_triangles += draw_triangles(draw);
                    self.draw_depth(math::mul(&fit.view_proj, &draw.model), draw)?;
                }
                cascades_rendered += 1;
            }
            let _ = map_srv;
            fits
        } else {
            self.shadows = ShadowCounters::default();
            shadow::CascadeSet::empty()
        };
        let shadow_ns = shadow_start.elapsed().as_nanos() as u64;

        // ---- colour pass -----------------------------------------------------
        let raster_start = Instant::now();
        let (rtv, dsv) = {
            let target = self
                .color
                .as_ref()
                .ok_or_else(|| Error::new(Code::NotReady, "no colour target"))?;
            (target.rtv.clone(), target.dsv.clone())
        };
        unsafe {
            if input.clear_color_enabled {
                self.context.ClearRenderTargetView(&rtv, &input.clear_color);
            }
            if input.clear_depth_enabled {
                self.context.ClearDepthStencilView(
                    &dsv,
                    (D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL).0 as u32,
                    input.clear_depth,
                    0,
                );
            }
            self.context.OMSetRenderTargets(Some(&[Some(rtv)]), &dsv);
            // The host's viewport, resolved against the target this tier
            // actually renders into: a tier that renders at a fraction of the
            // frame confines the same fraction, and `(0, 0)` - the documented
            // default - is the whole target. D3D11 clips rasterisation to this
            // rect, which is the same confinement the reference tier does by
            // clamping its tiles, so the two tiers agree pixel for pixel.
            let (view_w, view_h) =
                rendered_viewport(input.viewport, (input.width, input.height), (width, height));
            self.context.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: view_w as f32,
                Height: view_h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            self.context.IASetInputLayout(&self.layout);
            self.context
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            self.context.VSSetShader(&self.vs, None);
            self.context.PSSetShader(&self.ps, None);
            self.context.VSSetConstantBuffers(0, Some(&[Some(self.draw_cb.clone())]));
            self.context.PSSetConstantBuffers(0, Some(&[Some(self.draw_cb.clone())]));
            self.context.PSSetConstantBuffers(1, Some(&[Some(self.lights_cb.clone())]));
            self.context.PSSetConstantBuffers(2, Some(&[Some(self.shadow_cb.clone())]));
            self.context.PSSetSamplers(
                0,
                Some(&[Some(self.point_sampler.clone()), Some(self.cmp_sampler.clone())]),
            );
        }

        let lights = lights_cb(&input.lights);
        self.write_constant(&self.lights_cb, &lights, "lights constants")?;
        let filter = plan.map(|p| p.filter);
        let shadow_uniforms = self.shadow_cb_data(&fits, input, filter);
        self.write_constant(&self.shadow_cb, &shadow_uniforms, "shadow constants")?;

        let map_srv = self.maps.as_ref().map(|m| m.srv.clone());
        let mut triangles_in = 0u64;
        for draw in input.draws.iter() {
            triangles_in += draw_triangles(draw);
            self.draw_surface(draw, map_srv.as_ref())?;
        }
        let raster_ns = raster_start.elapsed().as_nanos() as u64;

        // ---- the frame's fingerprint, on the frames that asked for one -------
        if input.checksum {
            self.capture(input.frame_index)?;
            self.color_checksum = checksum_bytes(&self.rgba8);
        } else {
            self.color_checksum = 0;
        }

        self.shadows.cascades_rendered = cascades_rendered;
        self.shadows.shadow_pass_ns = shadow_ns;
        self.shadows.fit_ns = fit_ns;

        let frame = FrameNumbers {
            frame_index: input.frame_index,
            total_ns: frame_start.elapsed().as_nanos() as u64,
            shadow_ns,
            raster_ns,
            bin_ns: 0,
            upload_ns: 0,
            spill_wait_ns: 0,
            // The GPU schedules its own work. The binning, tile and shading
            // counters are properties of the tiled reference rasteriser and stay
            // honestly zero here rather than being invented.
            tiles_total: 0,
            tiles_rendered: 0,
            triangles_in: triangles_in.min(u32::MAX as u64) as u32,
            triangles_binned: (triangles_in + shadowed_triangles).min(u32::MAX as u64) as u32,
            triangles_culled: 0,
            pixels_shaded: 0,
            worker_threads: 0,
            resolution_scale: self.resolution_scale(),
            spill_io_bytes: 0,
            allocations_in_frame: 0,
        };
        self.frame = frame;
        self.counters.frames_since_tier_change += 1;
        Ok(frame)
    }

    fn shadow_cb_data(
        &self,
        fits: &shadow::CascadeSet,
        input: &FrameInput<'_>,
        filter: Option<ShadowFilter>,
    ) -> ShadowCb {
        let map_size = self.maps.as_ref().map(|m| m.size).unwrap_or(1);
        let cascades = fits.as_lookup_cascades();
        let bias = input.shadow.bias.unwrap_or_else(|| {
            shadow::bias_preset(
                self.tier,
                map_size.max(1),
                filter.unwrap_or(ShadowFilter::Hard),
            )
        });
        let mut cb = ShadowCb {
            // Row 2 of the camera view: the reference's camera-space depth row.
            view_z_row: [
                input.camera_view[2],
                input.camera_view[6],
                input.camera_view[10],
                input.camera_view[14],
            ],
            params_a: [
                filter.map(|f| f as u32 as f32).unwrap_or(0.0),
                fits.count as f32,
                bias.normal_bias,
                bias.depth_bias,
            ],
            params_b: [
                bias.slope_bias,
                input.shadow.max_distance,
                input.shadow.blend_band,
                map_size as f32,
            ],
            cascade_mvp: [[0.0; 16]; shadow::MAX_CASCADES],
            cascade_params: [[0.0; 4]; shadow::MAX_CASCADES],
        };
        for (index, cascade) in cascades.iter().enumerate() {
            cb.cascade_mvp[index] = cascade.view_proj;
            cb.cascade_params[index] = [
                cascade.split_distance,
                cascade.texel_world,
                cascade.depth_span,
                cascade.map_index as f32,
            ];
        }
        cb
    }
}

/// Maps a dynamic buffer, writes one value into it, and unmaps.
fn write_mapped<T: Copy>(
    context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    value: &T,
    label: &str,
) -> Result<()> {
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    unsafe { context.Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped)) }
        .map_err(|e| err(label, e))?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            value as *const T as *const u8,
            mapped.pData as *mut u8,
            std::mem::size_of::<T>(),
        );
        context.Unmap(buffer, 0);
    }
    Ok(())
}

/// Uploads `data` into a dynamic buffer, replacing it once if this draw needs
/// more room than the buffer was created with.
fn upload_into<T: Copy>(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    buffer: &mut ID3D11Buffer,
    capacity: &mut u32,
    bind: u32,
    data: &[T],
    label: &str,
) -> Result<()> {
    let needed = data.len() as u64 * std::mem::size_of::<T>() as u64;
    if needed > u64::from(*capacity) {
        let want = (needed as u32).saturating_add(UPLOAD_SLACK);
        *buffer = dynamic_buffer(device, want, bind, label)?;
        *capacity = want;
    }
    if data.is_empty() {
        return Ok(());
    }
    // `Map`/`Unmap` want a shared reference; `buffer` is `&mut` here only so it
    // can be replaced above.
    let target: &ID3D11Buffer = buffer;
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    unsafe { context.Map(target, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped)) }
        .map_err(|e| err(label, e))?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            mapped.pData as *mut u8,
            needed as usize,
        );
        context.Unmap(target, 0);
    }
    Ok(())
}

fn cull_index(cull: u32) -> usize {
    match cull {
        CULL_BACK => 1,
        CULL_FRONT => 2,
        _ => 0,
    }
}

fn draw_triangles(draw: &DrawItem<'_>) -> u64 {
    let count = match draw.indices {
        Some(indices) => indices.len(),
        None => draw.vertices.len(),
    };
    (count / 3) as u64
}

fn count_shadowed_lights(lights: &LightSet) -> u32 {
    lights
        .lights
        .iter()
        .take(lights.count as usize)
        .flatten()
        .filter(|l| l.cast_shadow)
        .count() as u32
}

/// The light array in `reconl-raster`'s element order.
fn lights_cb(lights: &LightSet) -> LightsCb {
    let mut cb = LightsCb {
        light_a: [[0.0; 4]; 16],
        light_b: [[0.0; 4]; 16],
        light_c: [[0.0; 4]; 16],
        light_d: [[0.0; 4]; 16],
        ambient_count: [
            lights.ambient[0],
            lights.ambient[1],
            lights.ambient[2],
            lights.count as f32,
        ],
    };
    for (slot, light) in lights.lights.iter().take(lights.count as usize).enumerate() {
        let Some(light) = light else { continue };
        cb.light_a[slot] = [
            light.position[0],
            light.position[1],
            light.position[2],
            light.kind as f32,
        ];
        cb.light_b[slot] = [
            light.direction[0],
            light.direction[1],
            light.direction[2],
            light.intensity,
        ];
        cb.light_c[slot] = [light.color[0], light.color[1], light.color[2], light.range];
        cb.light_d[slot] = [
            light.cos_inner,
            light.cos_outer,
            if light.cast_shadow { 1.0 } else { 0.0 },
            0.0,
        ];
    }
    cb
}

/// Capabilities of the hardware tiers.
///
/// Compared with the reference's `caps_for`: a GPU keeps the texture, mip,
/// shadow, filter, compute and present capabilities and does not claim the
/// SIMD, disk-spill or out-of-core ones, which describe CPU tiers rather than
/// this API.
pub fn caps_for(tier: Tier) -> u32 {
    let mut caps = caps::TEXTURES
        | caps::MIPMAPS
        | caps::SHADOWS
        | caps::PCF_5X5
        | caps::COMPUTE
        | caps::PRESENT_TO_MEMORY;
    if tier <= Tier::GpuShared {
        // pcss-lite is a T0/T1 feature in the tier table: a hardware tier has
        // the tap budget for it.
        caps |= caps::PCSS_LITE;
    }
    caps
}

fn enumerate_adapters() -> Result<Vec<IDXGIAdapter1>> {
    let factory: IDXGIFactory1 =
        unsafe { CreateDXGIFactory1() }.map_err(|e| err("create dxgi factory", e))?;
    let mut out = Vec::new();
    for index in 0..16u32 {
        match unsafe { factory.EnumAdapters1(index) } {
            Ok(adapter) => out.push(adapter),
            Err(_) => break,
        }
    }
    Ok(out)
}

/// One adapter as `reconlProbe` should report it.
pub fn probe_adapters() -> Result<Vec<AdapterInfo>> {
    let mut out = Vec::new();
    for adapter in enumerate_adapters()? {
        let desc = unsafe { adapter.GetDesc1() }.map_err(|e| err("adapter desc", e))?;
        let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        out.push(AdapterInfo {
            description: String::from_utf16_lossy(&desc.Description[..len]),
            dedicated_video_memory: desc.DedicatedVideoMemory as u64,
            vendor_id: desc.VendorId,
            feature_level_11: probe_feature_level(&adapter),
        });
    }
    Ok(out)
}

/// Whether the adapter will actually grant a device at the 11.0 feature level.
/// Reported by the probe rather than asserted from the adapter's name.
fn probe_feature_level(adapter: &IDXGIAdapter1) -> bool {
    let mut device: Option<ID3D11Device> = None;
    let mut context: Option<ID3D11DeviceContext> = None;
    let mut level = windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL(0);
    let result = unsafe {
        D3D11CreateDevice(
            adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_FLAG(0),
            Some(&[D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            Some(&mut level),
            Some(&mut context),
        )
    };
    result.is_ok() && level.0 >= D3D_FEATURE_LEVEL_11_0.0
}

/// Whether a D3D11 hardware device can be created at all, without keeping one.
/// This is what the startup probe asks before it offers a GPU tier.
pub fn hardware_available() -> bool {
    enumerate_adapters()
        .map(|adapters| adapters.first().map(probe_feature_level).unwrap_or(false))
        .unwrap_or(false)
}
