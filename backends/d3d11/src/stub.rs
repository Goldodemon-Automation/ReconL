//! Non-windows fallback: the backend is unavailable, loudly.

use reconl_core::error::{Code, Error, Result};
use reconl_backend_softcpu::ShadowRequest;
use reconl_core::budget::Budget;
use reconl_core::stats::{Counters, FrameNumbers, ShadowCounters};
use reconl_core::tier::{Backend, Tier, TierReason};
use std::sync::Arc;

use reconl_raster::math::Mat4;
use reconl_raster::shade::LightSet;

pub use reconl_backend_softcpu::FrameInput;

/// The probe types the non-windows build still has to name, so the ABI layer
/// compiles and reports the backend as unavailable rather than not existing.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub description: String,
    pub dedicated_video_memory: u64,
    pub vendor_id: u32,
    pub feature_level_11: bool,
}

/// No D3D11 on this target, so no adapter can be created.
pub fn hardware_available() -> bool {
    false
}

/// No hardware tiers on this target.
pub fn caps_for(_tier: Tier) -> u32 {
    0
}

/// An empty adapter list is the honest answer off Windows.
pub fn probe_adapters() -> Result<Vec<AdapterInfo>> {
    Ok(Vec::new())
}

#[derive(Clone, Debug)]
pub struct D3d11Config {
    pub tier: Tier,
    pub adapter_index: usize,
    pub resolution_scale: f32,
    pub target_frame_ms: u32,
    pub over_target_frames_to_downgrade: u32,
    pub shadow: ShadowRequest,
}

impl Default for D3d11Config {
    fn default() -> Self {
        Self {
            tier: Tier::GpuShared,
            adapter_index: 0,
            resolution_scale: 1.0,
            target_frame_ms: 16,
            over_target_frames_to_downgrade: 0,
            shadow: ShadowRequest::default(),
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct D3d11Snapshot {
    pub counters: Counters,
    pub shadows: ShadowCounters,
    pub frame: FrameNumbers,
    pub color_checksum: u64,
    pub depth_checksum: u64,
}

pub struct D3d11Device;

impl D3d11Device {
    pub fn new(_alloc: reconl_core::alloc::HostAlloc, _budget: Arc<Budget>, _config: D3d11Config) -> Result<Self> {
        Err(Error::new(Code::BackendUnavailable, "the d3d11 backend requires windows"))
    }

    pub fn backend(&self) -> Backend {
        Backend::D3d11
    }

    pub fn tier(&self) -> Tier {
        Tier::GpuShared
    }

    pub fn tier_reason(&self) -> TierReason {
        TierReason::HostRequest
    }

    pub fn caps(&self) -> u32 {
        0
    }

    pub fn device_name(&self) -> &'static str {
        "unavailable"
    }

    pub fn driver(&self) -> &'static str {
        "unavailable"
    }

    pub fn resolution_scale(&self) -> f32 {
        1.0
    }

    pub fn render(&mut self, _input: &FrameInput<'_>) -> Result<FrameNumbers> {
        Err(Error::new(Code::BackendUnavailable, "the d3d11 backend requires windows"))
    }

    pub fn readback(&self, _out: &mut [u8]) -> Result<u64> {
        Err(Error::new(Code::BackendUnavailable, "the d3d11 backend requires windows"))
    }

    pub fn color_checksum(&self) -> u64 {
        0
    }

    pub fn depth_checksum(&self) -> u64 {
        0
    }

    pub fn frame_size(&self) -> (u32, u32) {
        (1, 1)
    }

    pub fn snapshot(&self) -> D3d11Snapshot {
        D3d11Snapshot::default()
    }

    pub fn on_frame_end(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn prepare_frame(&mut self, _width: u32, _height: u32) -> Result<()> {
        Err(Error::new(Code::BackendUnavailable, "the d3d11 backend requires windows"))
    }

    /// No device exists to have been removed: the ABI's device-removal contract
    /// is the same on both builds, and nothing is ever lost here.
    pub fn device_removed(&self) -> bool {
        false
    }
}
