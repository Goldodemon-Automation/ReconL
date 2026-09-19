//! The D3D11 backend: the GPU tiers (T0 `gpu-discrete` / T1 `gpu-shared`).
//!
//! Division of labour, identical to the soft-cpu backend: the host builds the
//! draw list and says *what to shade*; this device supplies *the lighting
//! state* - the cascade fits, the maps, the bias preset, the filter the tier
//! permits. The fits are computed here with the same `reconl-shadow` code the
//! reference uses, from the same inputs, so the cascade matrices are
//! bit-identical across tiers.
//!
//! Reference semantics this backend must reproduce (`reconl-raster`):
//! - reversed-Z, GREATER comparison, clear depth 0.0
//! - front face = clockwise on screen (D3D11's default `FrontCounterClockwise:
//!   FALSE` is exactly the rasteriser's `area2 > 0`)
//! - shadow pass culls front faces (`CULL_FRONT`) so bias stays honest
//! - the bias policy of `reconl-shadow::bias_preset`, in the pixel shader
//! - blend modes: pre-multiplied in the shader + fixed function, per `BLEND_*`
//! - lighting at the raw vertex position: the rasteriser never transforms
//!   attributes, so neither does the vertex shader
//!
//! What is deliberately not here yet: GPU-side timestamp queries (frame times
//! below are CPU submit times - the ladder stays honest because the null tier
//! reports the same kind of measurement), and texture uploads (the ABI draw
//! path does not carry texture payloads to any backend yet - `DrawRecord` has
//! no texture slot - so a "textured" draw renders with a white texel here
//! exactly as it does through the reference rasteriser).
//!
//! Every D3D11 failure is an error return, never a silent fallback: a host
//! that asked for the GPU backend gets a classified `RECONL_ERR_*` code -
//! `DEVICE_LOST` only for a genuine removal, `OUT_OF_MEMORY`, argument codes
//! for a rejected descriptor, `BACKEND_UNAVAILABLE` when the cause is unknown -
//! never software pixels it did not ask for.

#[cfg(windows)]
mod shaders;
#[cfg(windows)]
pub mod imp;

#[cfg(windows)]
pub use imp::{
    caps_for, hardware_available, probe_adapters, AdapterInfo, D3d11Config, D3d11Device,
    D3d11Snapshot,
};

#[cfg(not(windows))]
mod stub;

#[cfg(not(windows))]
pub use stub::{
    caps_for, hardware_available, probe_adapters, AdapterInfo, D3d11Config, D3d11Device,
    D3d11Snapshot,
};
