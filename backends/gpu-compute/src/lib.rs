//! The `gpu-compute` backend: the GPU tiers (T0 `gpu-discrete` / T1 `gpu-shared`)
//! reached through a *vendor compute API* rather than a graphics one.
//!
//! ReconL already has a hardware tier - `d3d11` - and it is not this one. The
//! difference is what the GPU is asked to be. D3D11 asks it to be a graphics
//! pipeline: fixed-function raster, a state object per draw, a swapchain. This
//! backend asks it to be a big array of arithmetic units, which is what a
//! rasteriser built on ReconL's own rules actually needs - the reference tier's
//! pixel rules are fixed-point edges and a top-left fill test, and those are
//! expressible as kernels that look exactly like the CPU ones.
//!
//! Two runtimes are spoken, and neither is a build dependency:
//!
//! | runtime | vendor | opened from | used for |
//! |---|---|---|---|
//! | CUDA driver API | NVIDIA | `nvcuda.dll` / `libcuda.so.1` | enumeration, kernels |
//! | ROCm / HIP runtime | AMD | `amdhip64.dll` / `libamdhip64.so` | enumeration, kernels |
//!
//! They are opened at run time (see [`loader`]) because requiring the CUDA
//! toolkit or a ROCm install at build time would make this library fail to link
//! on the machine most of this repository is developed on, which has neither.
//! Which runtime answered is part of the report: `device_name` and `driver` name
//! it, so a host is never left guessing which vendor's stack it is on.
//!
//! ## What this backend does not do
//!
//! It does not report a device it cannot render on. [`discovery`] answers only
//! "is a driver here, and what does it say it has"; the device creation path
//! opens a context and compiles this backend's kernels before it admits a device
//! exists, and refuses with a classified `RECONL_ERR_*` when either step fails.
//! A machine with no vendor driver is `RECONL_ERR_BACKEND_UNAVAILABLE`, which is
//! the same answer the ABI gives for the declared-but-unbuilt backends - the
//! difference being that this one is a measurement rather than a placeholder.

pub mod api;
pub mod discovery;
pub mod loader;

pub use discovery::{best, caps_for, discover, discover_uncached, hardware_available, probe, AdapterInfo};

use reconl_core::error::{Code, Error, Result};
use reconl_core::tier::Tier;

/// Whether this *build* contains a render path.
///
/// False, and named rather than implicit, because the difference it settles is
/// the one a host can act on: enumeration still answers (a C host can list the
/// machine's CUDA and ROCm devices through `reconlEnumerateAdapters`), while
/// device creation refuses with `RECONL_ERR_NOT_SUPPORTED` instead of handing
/// back a device whose every frame would fail. When the compute raster lands,
/// this is the one line that flips and every report follows it - the probe's
/// `usable`, the adapter's `usable`, and `device_support` below.
pub const RENDER_PATH_BUILT: bool = false;

/// The reason a device cannot be created here and now, or `Ok(())` if it can.
///
/// This is the function `reconlCreateDevice` calls, and it is deliberately the
/// same question the probe's `usable` field answers, asked once: a probe that
/// said "usable" and a creation that then refused would be the kind of
/// disagreement the ABI exists to make impossible.
///
/// The two failures it can report are distinct and a host should treat them
/// differently:
///
/// * `RECONL_ERR_BACKEND_UNAVAILABLE` - no CUDA driver and no ROCm runtime on
///   this machine. Nothing a host can do but use another backend.
/// * `RECONL_ERR_NOT_SUPPORTED` - a driver is here, but this release cannot
///   render on it. Nothing a host can do either, but it names the difference.
pub fn device_support() -> Result<()> {
    let adapters = discover();
    if adapters.is_empty() {
        return Err(Error::new(
            Code::BackendUnavailable,
            if cfg!(windows) {
                "no CUDA driver (nvcuda.dll) and no ROCm runtime (amdhip64.dll) on this machine"
            } else {
                "no CUDA driver (libcuda.so.1) and no ROCm runtime (libamdhip64.so) on this machine"
            },
        ));
    }
    if !RENDER_PATH_BUILT {
        return Err(Error::new(
            Code::NotSupported,
            "a compute driver is present but this release has no compute raster path; kernels are the next step, and until they exist a device would not render",
        ));
    }
    Ok(())
}

/// The tier a device on this adapter starts at.
///
/// The rule is the ABI's own definitions rather than a benchmark: T0 is the
/// discrete GPU, T1 is a GPU whose memory the host shares. Three answers are
/// possible and all three are used:
///
/// * `Some(false)` - the driver said discrete - is T0, whatever its size.
/// * `Some(true)` - the driver said integrated - is T1, whatever its size. An
///   APU with 16 GiB of addressable memory is still a device whose bandwidth and
///   memory the CPU is using.
/// * `None` - the driver did not classify it - falls back to total memory, and
///   the threshold is deliberately low rather than a model of any particular
///   product: under 2 GiB is a device that shares memory in practice (an old
///   part, or an APU whose driver does not answer), and everything above it is
///   given the benefit of the doubt. The tier ladder steps down from a wrong
///   guess within a few frames, which is exactly what it is for; a wrong guess
///   in the *other* direction would strand a capable device on T1 forever.
pub fn tier_for(vram_bytes: u64, integrated: Option<bool>) -> Tier {
    match integrated {
        Some(true) => Tier::GpuShared,
        Some(false) => Tier::GpuDiscrete,
        None => {
            if vram_bytes >= (2 << 30) {
                Tier::GpuDiscrete
            } else {
                Tier::GpuShared
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_driver_that_classified_the_device_is_believed_over_its_size() {
        // A 96 GiB integrated part is still shared; a 1 GiB discrete part is
        // still discrete.
        assert_eq!(tier_for(96 << 30, Some(true)), Tier::GpuShared);
        assert_eq!(tier_for(1 << 30, Some(false)), Tier::GpuDiscrete);
    }

    #[test]
    fn an_unclassified_device_falls_back_to_memory() {
        assert_eq!(tier_for(8 << 30, None), Tier::GpuDiscrete);
        assert_eq!(tier_for(512 << 20, None), Tier::GpuShared);
        assert_eq!(tier_for(0, None), Tier::GpuShared);
    }
}
