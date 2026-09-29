//! Finding a usable compute runtime, and saying so in terms the ABI can report.
//!
//! Discovery is the whole of this module's job, and it is deliberately the one
//! place a *choice between vendors* happens: CUDA is tried first, then ROCm, and
//! the first runtime that loads **and reports at least one device** wins. A
//! library that loads but reports no devices is not a runtime - it is a driver
//! with nothing behind it - so it does not shadow the other vendor's.
//!
//! Nothing here trusts a name. The report a host reads is built from what the
//! driver answered: the device name string, the total memory the driver
//! reported, the attribute it returned. Where a driver does not answer - ROCm's
//! integrated-vs-discrete attribute is one, and an old driver's compute
//! capability another - the field is `None` and stays `None` all the way to the
//! probe's note, rather than being filled in with a plausible default.

use std::sync::OnceLock;

use reconl_core::error::Result;
use reconl_core::tier::{caps, Tier};

use crate::api::{Runtime, DeviceInfo};
use crate::loader::Library;

/// The sonames each runtime is known by, most specific first.
///
/// A versioned soname is tried before the unversioned one: on a machine with two
/// ROCm stacks installed, the report should name the one whose API was actually
/// resolved rather than whichever the linker would have picked.
const CUDA_NAMES: &[&str] = if cfg!(windows) {
    &["nvcuda.dll"]
} else {
    &["libcuda.so.1", "libcuda.so"]
};

const ROCM_NAMES: &[&str] = if cfg!(windows) {
    &["amdhip64.dll", "amdhip64_6.dll", "amdhip64_5.dll"]
} else {
    &["libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so"]
};

/// One GPU as the probe and the adapter enumeration report it.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    /// `"nvidia"` or `"amd"`: the runtime that answered.
    pub vendor: &'static str,
    /// `"cuda"` or `"rocm"`.
    pub runtime: &'static str,
    pub name: String,
    pub vram_bytes: u64,
    /// Shared system memory the device may address, when the driver says it is
    /// integrated. `0` when unknown rather than when absent.
    pub shared_system_memory: u64,
    /// `Some(true)` integrated, `Some(false)` discrete, `None` unclassified.
    pub integrated: Option<bool>,
    pub compute_capability: Option<(u32, u32)>,
    pub driver: String,
    pub index: i32,
    /// The tier a device created on this adapter would start at.
    pub best_tier: Tier,
    /// Whether a ReconL device can be created here. This is the field the probe
    /// and `reconlCreateDevice` must agree on, so both read it from here.
    pub usable: bool,
    /// Why it is not usable, in the driver's own words where the driver refused.
    pub note: String,
}

impl Default for AdapterInfo {
    fn default() -> Self {
        Self {
            vendor: "",
            runtime: "",
            name: String::new(),
            vram_bytes: 0,
            shared_system_memory: 0,
            integrated: None,
            compute_capability: None,
            driver: String::new(),
            index: 0,
            // Not a GPU tier: the only way to reach this value is a caller that
            // built the struct field by field and did not say, and "no GPU" is
            // the honest thing to say when nothing has been read from a driver.
            best_tier: Tier::CpuRam,
            usable: false,
            note: String::new(),
        }
    }
}

/// The probe's answer, computed once and cached for the process.
///
/// `reconlProbe` is documented as side-effect free and cheap enough to call
/// between frames, and `cuInit`/`hipInit` plus a device enumeration is neither
/// free nor free of driver state. One discovery per process is also the honest
/// reading of the ABI: a GPU does not appear while the process runs. A caller
/// that wants a fresh look calls [`discover_uncached`] - which is what a device
/// creation does, so a driver installed or a device lost between a probe and a
/// create is seen by the second call rather than hidden by the first.
static CACHE: OnceLock<Vec<AdapterInfo>> = OnceLock::new();

/// The machine's compute adapters, discovered once.
pub fn discover() -> Vec<AdapterInfo> {
    CACHE.get_or_init(discover_uncached).clone()
}

/// Runs discovery without consulting the cache.
pub fn discover_uncached() -> Vec<AdapterInfo> {
    let mut out = Vec::new();
    for runtime in open_runtimes() {
        match runtime.probe_devices() {
            Ok(devices) if !devices.is_empty() => {
                let driver = describe_driver(runtime);
                for device in devices {
                    out.push(adapter_from(runtime, device, &driver));
                }
                // The first runtime that answers is the one this backend uses.
                // A machine with both installed reports the CUDA devices; the
                // report says which runtime it used, so a host is never left to
                // guess.
                break;
            }
            Ok(_) => continue,
            Err(_) => continue,
        }
    }
    out
}

fn describe_driver(runtime: &Runtime) -> String {
    match runtime.driver_version() {
        Some(v) => format!("{} driver {} ({})", runtime.backend_name(), v, runtime.library_name()),
        None => format!("{} ({})", runtime.backend_name(), runtime.library_name()),
    }
}

fn adapter_from(runtime: &'static Runtime, device: DeviceInfo, driver: &str) -> AdapterInfo {
    let integrated = device.integrated;
    let best_tier = crate::tier_for(device.vram_bytes, integrated);
    // Shared memory is derived rather than guessed: the driver reports total
    // memory, and an integrated device's video memory *is* carved out of system
    // memory, so what an integrated device reports as its own is what the host
    // may also count on. A discrete device reports none of it.
    let shared = if integrated == Some(true) { device.vram_bytes } else { 0 };
    // `usable` is measured, not assumed from enumeration: the driver reported
    // this device, and this asks it to open a context and give it straight
    // back. A device that is present but cannot be opened says so in the note.
    let (usable, note) = match runtime.context_works(device.index) {
        // The hardware answers and the build can render on it.
        Ok(()) if crate::RENDER_PATH_BUILT => (true, String::new()),
        // The hardware answers, but this build has nothing to run on it - which
        // is a property of the release, said plainly, rather than a device a
        // host would create and then find refuses every frame.
        Ok(()) => (
            false,
            "the driver is present and a context opens, but this build has no compute raster path: a device would not render"
                .to_string(),
        ),
        Err(reason) => (false, reason),
    };
    AdapterInfo {
        vendor: runtime.vendor(),
        runtime: runtime.backend_name(),
        name: if device.name.is_empty() { format!("{} device {}", runtime.backend_name(), device.index) } else { device.name },
        vram_bytes: device.vram_bytes,
        shared_system_memory: shared,
        integrated,
        compute_capability: device.compute_capability,
        driver: driver.to_string(),
        index: device.index,
        best_tier,
        usable,
        note,
    }
}

/// Opens every runtime this machine has, in preference order.
///
/// A library that loads but whose API is incomplete is dropped here: see
/// `api::cuda::Api::load`, which requires every entry point the backend uses
/// rather than a subset.
///
/// The runtimes are leaked rather than owned, because a device holds a context
/// made from one and a `Vec` that went out of scope would take the API table
/// with it. One `Runtime` per vendor per process is also what the drivers
/// expect (their own state is process-wide).
fn open_runtimes() -> Vec<&'static Runtime> {
    let mut out = Vec::new();
    if let Some(lib) = open_static(CUDA_NAMES) {
        if let Some(api) = crate::api::cuda::Api::load(lib) {
            out.push(&*Box::leak(Box::new(Runtime::Cuda { lib, api })));
        }
    }
    if let Some(lib) = open_static(ROCM_NAMES) {
        if let Some(api) = crate::api::hip::Api::load(lib) {
            out.push(&*Box::leak(Box::new(Runtime::Rocm { lib, api })));
        }
    }
    out
}

/// Opens a library and keeps it for the process's lifetime.
///
/// The handle is leaked rather than dropped because a driver's objects outlive
/// this value: a device holds a context created through this handle, and
/// unloading a driver while one of its contexts is live is a crash in the
/// driver, not in ReconL. One `Library` per soname per process is also exactly
/// what `dlopen`/`LoadLibrary` hand back anyway - they refcount internally.
fn open_static(names: &[&'static str]) -> Option<&'static Library> {
    Library::open_first(names).map(|lib| &*Box::leak(Box::new(lib)))
}

/// Whether any runtime reported at least one device.
///
/// Cached, so the answer a probe put in `reconlProbeInfo` and the answer a
/// device creation acts on are the same enumeration rather than two looks at a
/// driver that could have changed between them.
pub fn hardware_available() -> bool {
    !discover().is_empty()
}

/// The adapter a device creation with no explicit index would choose: the
/// discrete GPU with the most memory, then any discrete GPU, then any device.
///
/// This is the same preference order the D3D11 backend's `Auto` makes, and it
/// is why a two-GPU machine does not silently render on the integrated part.
pub fn best(adapters: &[AdapterInfo]) -> Option<&AdapterInfo> {
    adapters
        .iter()
        .filter(|a| a.usable && a.integrated == Some(false))
        .max_by_key(|a| a.vram_bytes)
        .or_else(|| adapters.iter().find(|a| a.usable))
}

/// The capabilities a compute tier grants.
///
/// The GPU tiers report the same set the D3D11 backend reports for T0/T1: this
/// backend implements the same passes and the subset of shadow filters is
/// resolved by the tier policy, not by this table. What is deliberately *not*
/// claimed is `DISK_SPILL`/`OUT_OF_CORE` - a device with VRAM has the GPU tiers'
/// own answer to a frame that does not fit (step down), and the arena is the
/// reference tier's mechanism.
pub fn caps_for(tier: Tier) -> u32 {
    let base = caps::TEXTURES
        | caps::MIPMAPS
        | caps::SHADOWS
        | caps::PCF_5X5
        | caps::PCSS_LITE
        | caps::MULTITHREAD
        | caps::COMPUTE
        | caps::PRESENT_TO_MEMORY;
    match tier {
        Tier::GpuDiscrete => base | caps::CACHED_CASCADE,
        Tier::GpuShared => base,
        // A device that has stepped to a CPU tier is no longer this backend's;
        // reporting the GPU set here would be a lie in the caps a host reads.
        _ => 0,
    }
}

/// Reads the adapters, or the reason there are none.
pub fn probe() -> Result<Vec<AdapterInfo>> {
    Ok(discover())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(integrated: Option<bool>, vram: u64) -> AdapterInfo {
        AdapterInfo {
            vendor: "nvidia",
            runtime: "cuda",
            name: "test".into(),
            vram_bytes: vram,
            integrated,
            best_tier: crate::tier_for(vram, integrated),
            usable: true,
            ..AdapterInfo::default()
        }
    }

    #[test]
    fn auto_prefers_the_largest_discrete_device() {
        let adapters = vec![
            adapter(Some(true), 8 << 30),
            adapter(Some(false), 6 << 30),
            adapter(Some(false), 24 << 30),
        ];
        assert_eq!(best(&adapters).unwrap().vram_bytes, 24 << 30);
    }

    #[test]
    fn auto_falls_back_to_an_integrated_device_when_it_is_all_there_is() {
        let adapters = vec![adapter(Some(true), 4 << 30)];
        assert_eq!(best(&adapters).unwrap().vram_bytes, 4 << 30);
    }

    #[test]
    fn an_unusable_adapter_is_never_chosen() {
        let mut dead = adapter(Some(false), 48 << 30);
        dead.usable = false;
        let live = adapter(Some(false), 4 << 30);
        assert_eq!(best(&[dead, live]).unwrap().vram_bytes, 4 << 30);
    }

    #[test]
    fn no_adapter_at_all_is_none_rather_than_a_default() {
        assert!(best(&[]).is_none());
    }

    #[test]
    fn the_gpu_tiers_get_the_gpu_caps_and_a_cpu_tier_gets_none() {
        assert_ne!(caps_for(Tier::GpuDiscrete) & caps::SHADOWS, 0);
        assert_ne!(caps_for(Tier::GpuDiscrete) & caps::CACHED_CASCADE, 0);
        assert_eq!(caps_for(Tier::GpuShared) & caps::CACHED_CASCADE, 0);
        // The disk tier belongs to the reference backend; a hardware device that
        // stepped to T2 has changed backend, not grown an arena.
        assert_eq!(caps_for(Tier::CpuRam), 0);
        assert_eq!(caps_for(Tier::OutOfCore), 0);
    }

    /// The one claim that has to hold on a machine with no vendor driver, which
    /// is the machine this crate is developed on. It is written so it passes
    /// both ways: with a GPU it asserts the report is self-consistent, and
    /// without one it asserts the answer is empty rather than invented.
    #[test]
    fn discovery_reports_only_what_the_driver_answered() {
        for adapter in discover() {
            assert!(adapter.usable);
            assert!(!adapter.name.is_empty(), "an adapter with no name is not reported");
            assert!(
                adapter.runtime == "cuda" || adapter.runtime == "rocm",
                "a runtime name has to be one of the two this backend speaks"
            );
            if adapter.integrated == Some(true) {
                assert_eq!(
                    adapter.shared_system_memory, adapter.vram_bytes,
                    "an integrated device's memory is shared, and the report says so"
                );
            } else {
                assert_eq!(adapter.shared_system_memory, 0);
            }
        }
    }
}
