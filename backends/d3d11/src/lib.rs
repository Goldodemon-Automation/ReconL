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

use reconl_core::error::{Code, Error, Result};

/// DXCore's classification of a physical adapter. Older Windows versions may
/// not expose this metadata, in which case the type remains `Unknown`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdapterType {
    #[default]
    Unknown,
    Integrated,
    Discrete,
}

/// One physical GPU as reported by DXGI, with DXCore classification where
/// available. `adapter_luid` is stable for this adapter until reboot or driver
/// restart and is the preferred way for a host to select an exact device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdapterInfo {
    pub description: String,
    pub adapter_type: AdapterType,
    pub adapter_luid: u64,
    pub dedicated_video_memory: u64,
    pub shared_system_memory: u64,
    pub vendor_id: u32,
    pub device_id: u32,
    pub feature_level_11: bool,
}

/// How D3D11 chooses an adapter. `Auto` intentionally prefers a discrete GPU,
/// matching the engine's highest-performance default, and falls back to any
/// usable D3D11 adapter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdapterSelection {
    #[default]
    Auto,
    Integrated,
    Discrete,
    Index(usize),
    Luid(u64),
}

/// Returns a usable adapter matching the requested selection, or a classified
/// error when an exact selection does not exist or cannot create D3D11 11.0.
pub fn select_adapter_index(adapters: &[AdapterInfo], selection: AdapterSelection) -> Result<usize> {
    let exact = match selection {
        AdapterSelection::Index(index) => {
            let Some(adapter) = adapters.get(index) else {
                return Err(Error::new(Code::InvalidArgument, "D3D11 adapter index is outside the enumerated adapter list"));
            };
            Some((index, adapter))
        }
        AdapterSelection::Luid(luid) => {
            let Some((index, adapter)) = adapters.iter().enumerate().find(|(_, a)| a.adapter_luid == luid) else {
                return Err(Error::new(Code::BackendUnavailable, "the requested D3D11 adapter LUID is not present"));
            };
            Some((index, adapter))
        }
        _ => None,
    };

    if let Some((index, adapter)) = exact {
        return if adapter.feature_level_11 {
            Ok(index)
        } else {
            Err(Error::new(Code::BackendUnavailable, "the requested D3D11 adapter cannot create feature level 11.0"))
        };
    }

    let preferred_type = match selection {
        AdapterSelection::Integrated => AdapterType::Integrated,
        AdapterSelection::Auto | AdapterSelection::Discrete => AdapterType::Discrete,
        AdapterSelection::Index(_) | AdapterSelection::Luid(_) => unreachable!(),
    };
    adapters
        .iter()
        .position(|adapter| adapter.feature_level_11 && adapter.adapter_type == preferred_type)
        .or_else(|| adapters.iter().position(|adapter| adapter.feature_level_11))
        .ok_or_else(|| Error::new(Code::BackendUnavailable, "no D3D11 adapter can create feature level 11.0"))
}

#[cfg(windows)]
mod shaders;
#[cfg(windows)]
pub mod imp;

#[cfg(windows)]
pub use imp::{caps_for, hardware_available, probe_adapters, D3d11Config, D3d11Device, D3d11Snapshot};

#[cfg(not(windows))]
mod stub;

#[cfg(not(windows))]
pub use stub::{caps_for, hardware_available, probe_adapters, D3d11Config, D3d11Device, D3d11Snapshot};

#[cfg(test)]
mod selection_tests {
    use super::*;

    fn adapters() -> Vec<AdapterInfo> {
        vec![
            AdapterInfo {
                description: "integrated".into(),
                adapter_type: AdapterType::Integrated,
                adapter_luid: 10,
                feature_level_11: true,
                ..AdapterInfo::default()
            },
            AdapterInfo {
                description: "discrete".into(),
                adapter_type: AdapterType::Discrete,
                adapter_luid: 20,
                feature_level_11: true,
                ..AdapterInfo::default()
            },
            AdapterInfo {
                description: "unusable discrete".into(),
                adapter_type: AdapterType::Discrete,
                adapter_luid: 30,
                feature_level_11: false,
                ..AdapterInfo::default()
            },
        ]
    }

    #[test]
    fn auto_prefers_discrete_and_falls_back_when_needed() {
        let adapters = adapters();
        assert_eq!(select_adapter_index(&adapters, AdapterSelection::Auto).unwrap(), 1);
        assert_eq!(select_adapter_index(&adapters[..1], AdapterSelection::Auto).unwrap(), 0);
        assert_eq!(select_adapter_index(&[], AdapterSelection::Auto).unwrap_err().code, Code::BackendUnavailable);
    }

    #[test]
    fn explicit_type_index_and_luid_selection_are_checked() {
        let adapters = adapters();
        assert_eq!(select_adapter_index(&adapters, AdapterSelection::Integrated).unwrap(), 0);
        assert_eq!(select_adapter_index(&adapters, AdapterSelection::Index(1)).unwrap(), 1);
        assert_eq!(select_adapter_index(&adapters, AdapterSelection::Luid(10)).unwrap(), 0);
        assert_eq!(select_adapter_index(&adapters, AdapterSelection::Luid(999)).unwrap_err().code, Code::BackendUnavailable);
        assert_eq!(select_adapter_index(&adapters, AdapterSelection::Index(99)).unwrap_err().code, Code::InvalidArgument);
        assert_eq!(select_adapter_index(&adapters, AdapterSelection::Index(2)).unwrap_err().code, Code::BackendUnavailable);
    }

    #[test]
    fn unknown_adapter_types_keep_dxgi_order_as_the_fallback() {
        let adapters = vec![
            AdapterInfo { adapter_luid: 1, feature_level_11: true, ..AdapterInfo::default() },
            AdapterInfo {
                adapter_luid: 2,
                adapter_type: AdapterType::Discrete,
                feature_level_11: true,
                ..AdapterInfo::default()
            },
        ];
        assert_eq!(select_adapter_index(&adapters, AdapterSelection::Auto).unwrap(), 1);
        assert_eq!(select_adapter_index(&adapters, AdapterSelection::Integrated).unwrap(), 0);
    }
}
