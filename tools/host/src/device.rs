//! Creating a device and reading back what it says about itself.
//!
//! Two entry points matter here, and they are deliberately separate:
//!
//! * [`probe`] calls `reconlProbe` and creates **nothing**. A host uses it to
//!   decide what to ask for, and the tools use it to report what the machine can
//!   do before any device exists - which is only a claim worth making if the
//!   probe really does not allocate or initialise a backend (the allocator
//!   counters in [`crate::alloc`] are how `reconl-info` proves it does not).
//! * [`Device::create`] asks for a backend and tier and returns the device the
//!   library actually granted, which may be a step down the ladder.
//!
//! Everything a device reports back - limits, memory, stats, shadow stats - is
//! reached through a named method rather than by reading descriptors at each
//! call site, so a tool that wants to print "the map resolution" asks the
//! device instead of rebuilding the shadow configuration it thinks it set.

use crate::{failed, hdr, names};
use reconl::abi;
use std::ffi::{c_void, CString};

/// What to ask for when creating a device.
///
/// The defaults are a host's defaults, not a test's: no backend or tier hint
/// (the library picks), every downgrade allowed, no frame-time ladder (`0`
/// disables it, so nothing steps down behind the tool's back), and no memory
/// caps, which leaves the device bounded only by the library's own per
/// allocation ceiling.
#[derive(Clone, Debug)]
pub struct Config {
    pub backend: u32,
    pub tier: u32,
    pub allow_downgrade: u32,
    pub threads: u32,
    pub target_frame_ms: u32,
    pub downgrade_after_frames: u32,
    pub seed: u32,
    pub ram_cap: u64,
    pub vram_cap: u64,
    pub disk_cap: u64,
    pub allow_disk_spill: bool,
    pub spill_dir: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: abi::backend::NONE,
            tier: 0,
            allow_downgrade: abi::allow_downgrade::ALL,
            threads: 0,
            target_frame_ms: 0,
            downgrade_after_frames: 16,
            seed: 7,
            ram_cap: 0,
            vram_cap: 0,
            disk_cap: 0,
            allow_disk_spill: false,
            spill_dir: None,
        }
    }
}

impl Config {
    /// A backend by the ABI's name, or `None` for "auto".
    pub fn backend_from_name(name: &str) -> Option<u32> {
        Some(match name {
            "auto" | "none" => abi::backend::NONE,
            "soft-cpu" | "softcpu" | "cpu" => abi::backend::SOFT_CPU,
            "null" => abi::backend::NULL,
            "d3d11" => abi::backend::D3D11,
            "d3d12" => abi::backend::D3D12,
            "vulkan" => abi::backend::VULKAN,
            "gl" => abi::backend::GL,
            "metal" => abi::backend::METAL,
            "webgpu" => abi::backend::WEBGPU,
            "wasm-webgl2" => abi::backend::WASM_WEBGL2,
            _ => return None,
        })
    }

    /// A tier by number (`0`..`4`) or by the library's own name.
    pub fn tier_from_name(name: &str) -> Option<u32> {
        if let Ok(v) = name.parse::<u32>() {
            return Some(v.min(4));
        }
        Some(match name {
            "auto" | "none" => 0,
            "t0" | "gpu-discrete" => 0,
            "t1" | "gpu-shared" => 1,
            "t2" | "cpu-ram" => 2,
            "t3" | "cpu-thrifty" => 3,
            "t4" | "out-of-core" => 4,
            _ => return None,
        })
    }
}

/// A live device. Released on drop, so a tool cannot forget to.
pub struct Device {
    ptr: *mut reconl::DeviceHandle,
    /// Kept alive because the descriptor points into it for the device's life.
    _spill_dir: Option<CString>,
    /// Kept alive for the same reason.
    _budget: Option<Box<abi::ReconLMemoryBudget>>,
}

impl Device {
    /// Creates a device, or reports what refused it.
    pub fn create(config: &Config) -> Result<Device, String> {
        let spill_dir = match &config.spill_dir {
            Some(dir) => Some(
                CString::new(dir.as_str()).map_err(|_| format!("spill dir `{dir}` contains a NUL"))?,
            ),
            None => None,
        };
        let wants_budget = config.ram_cap != 0
            || config.vram_cap != 0
            || config.disk_cap != 0
            || config.allow_disk_spill
            || spill_dir.is_some();
        let budget = wants_budget.then(|| {
            Box::new(abi::ReconLMemoryBudget {
                base: hdr::<abi::ReconLMemoryBudget>(),
                vram_cap_bytes: config.vram_cap,
                ram_cap_bytes: config.ram_cap,
                disk_cap_bytes: config.disk_cap,
                allow_disk_spill: u32::from(config.allow_disk_spill),
                reserved: 0,
                spill_dir: spill_dir.as_ref().map_or(core::ptr::null(), |c| c.as_ptr()),
            })
        });

        // SAFETY: every pointer below is either null or points at a descriptor
        // that outlives this call, and the descriptor's header is filled from the
        // Rust type so its size cannot disagree with the header's.
        unsafe {
            let desc = abi::ReconLDeviceDesc {
                base: hdr::<abi::ReconLDeviceDesc>(),
                backend_hint: config.backend,
                tier_hint: config.tier,
                allow_downgrade: config.allow_downgrade,
                worker_threads: config.threads,
                target_frame_ms: config.target_frame_ms,
                downgrade_after_frames: config.downgrade_after_frames,
                seed: config.seed,
                flags: 0,
                budget: budget.as_ref().map_or(core::ptr::null(), |b| &**b as *const _),
                allocator: crate::alloc::allocator(),
                backend_desc: core::ptr::null(),
            };
            let mut ptr: *mut reconl::DeviceHandle = core::ptr::null_mut();
            let r = reconl::reconlCreateDevice(&desc, &mut ptr);
            if r != abi::result::OK || ptr.is_null() {
                return Err(failed("reconlCreateDevice", r));
            }
            Ok(Device { ptr, _spill_dir: spill_dir, _budget: budget })
        }
    }

    /// The raw handle, for the other modules in this crate.
    pub fn handle(&self) -> *mut reconl::DeviceHandle {
        self.ptr
    }

    /// What the device will admit, including the bounded-allocation ceiling.
    pub fn limits(&self) -> Result<abi::ReconLDeviceLimits, String> {
        // SAFETY: the handle is live and `out` is a local the call fully writes.
        unsafe {
            let mut out: abi::ReconLDeviceLimits = core::mem::zeroed();
            let r = reconl::reconlGetDeviceLimits(self.ptr, &mut out);
            if r != abi::result::OK {
                return Err(failed("reconlGetDeviceLimits", r));
            }
            Ok(out)
        }
    }

    /// Resident memory, spill traffic and the host allocator's own ledger.
    pub fn memory(&self) -> Result<abi::ReconLMemoryStats, String> {
        // SAFETY: as above.
        unsafe {
            let mut out: abi::ReconLMemoryStats = core::mem::zeroed();
            let r = reconl::reconlGetMemoryStats(self.ptr, &mut out);
            if r != abi::result::OK {
                return Err(failed("reconlGetMemoryStats", r));
            }
            Ok(out)
        }
    }

    /// Everything the device has counted since creation or the last reset:
    /// frames, tier, downgrades, shadow pass attribution, frame timing.
    pub fn stats(&self) -> Result<abi::ReconLStats, String> {
        // SAFETY: the handle is live and the library writes the whole struct.
        unsafe {
            let mut out: abi::ReconLStats = core::mem::zeroed();
            out.base = hdr::<abi::ReconLStats>();
            let r = reconl::reconlGetStats(self.ptr, &mut out);
            if r != abi::result::OK {
                return Err(failed("reconlGetStats", r));
            }
            Ok(out)
        }
    }

    /// Zeroes the counters, so a measurement window starts clean.
    pub fn reset_stats(&self) -> Result<(), String> {
        // SAFETY: live handle.
        let r = unsafe { reconl::reconlResetStats(self.ptr) };
        if r != abi::result::OK {
            return Err(failed("reconlResetStats", r));
        }
        Ok(())
    }

    /// Asks the library to re-verify its tier decision and budget totals every
    /// `every_n_frames` frames, and returns how many divergences it has found.
    /// Diagnostic and expensive - PROMPT §13's audit mode.
    pub fn audit(&self, every_n_frames: u32) -> Result<u32, String> {
        // SAFETY: live handle, and `out` is written on success.
        unsafe {
            let mut divergences = 0u32;
            let r = reconl::reconlAudit(self.ptr, every_n_frames, &mut divergences);
            if r != abi::result::OK {
                return Err(failed("reconlAudit", r));
            }
            Ok(divergences)
        }
    }

    /// Replaces the device's shadow plan. Applies at the next frame boundary.
    pub fn configure_shadows(&self, config: &abi::ReconLShadowConfig) -> Result<(), String> {
        // SAFETY: live handle; the descriptor outlives the call.
        let r = unsafe { reconl::reconlConfigureShadows(self.ptr, config) };
        if r != abi::result::OK {
            return Err(failed("reconlConfigureShadows", r));
        }
        Ok(())
    }

    /// The backend the library actually chose, named.
    pub fn backend_name(&self) -> String {
        self.limits().map(|l| names::backend(l.backend)).unwrap_or_else(|_| "?".into())
    }

    /// The device's own name, as the backend reported it.
    pub fn device_name(&self) -> String {
        self.limits().map(|l| names::field(&l.device_name)).unwrap_or_default()
    }

    /// The driver version string, empty on a backend with no driver to name.
    pub fn driver(&self) -> String {
        self.limits().map(|l| names::field(&l.driver)).unwrap_or_default()
    }

    /// The last error the library recorded, formatted with its code - the reason
    /// a host that saw a failure should log.
    pub fn last_error(&self) -> Option<String> {
        // SAFETY: live handle; the library writes the whole struct on success.
        unsafe {
            let mut info: abi::ReconLErrorInfo = core::mem::zeroed();
            if reconl::reconlGetLastError(self.ptr, &mut info) != abi::result::OK {
                return None;
            }
            let message = names::field(&info.message);
            // The library reports OK with a "nothing has failed" message, which
            // is not an error and must not print as one.
            if info.result == abi::result::OK || message.is_empty() {
                return None;
            }
            Some(format!("{} ({})", message, crate::result_name(info.result)))
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: the handle is live and released exactly once.
            unsafe { reconl::reconlRelease(self.ptr as *mut c_void) };
            self.ptr = core::ptr::null_mut();
        }
    }
}

/// The library's report on this machine, gathered without creating a device.
pub struct Probed {
    pub info: abi::ReconLProbeInfo,
}

impl Probed {
    /// Only the entries the library filled, in its own order.
    pub fn entries(&self) -> &[abi::ReconLBackendProbe] {
        let n = (self.info.entry_count as usize).min(abi::RECONL_MAX_BACKENDS);
        &self.info.entries[..n]
    }

    /// The recommended backend, named.
    pub fn recommended_backend(&self) -> String {
        names::backend(self.info.recommended_backend)
    }

    /// The recommended tier, named.
    pub fn recommended_tier(&self) -> String {
        names::tier(self.info.recommended_tier)
    }
}

/// Probes every backend the library knows. Creates no device.
pub fn probe(spill_dir: Option<&str>) -> Result<Probed, String> {
    let dir = match spill_dir {
        Some(d) => Some(CString::new(d).map_err(|_| format!("spill dir `{d}` contains a NUL"))?),
        None => None,
    };
    // SAFETY: `dir` outlives the call; `out` is written on success.
    unsafe {
        let desc = abi::ReconLProbeDesc {
            base: hdr::<abi::ReconLProbeDesc>(),
            flags: 0,
            spill_dir: dir.as_ref().map_or(core::ptr::null(), |c| c.as_ptr()),
        };
        let mut info: abi::ReconLProbeInfo = core::mem::zeroed();
        info.base = hdr::<abi::ReconLProbeInfo>();
        let r = reconl::reconlProbe(&desc, &mut info);
        if r != abi::result::OK {
            return Err(failed("reconlProbe", r));
        }
        Ok(Probed { info })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_and_tier_names_resolve_the_way_the_cli_spells_them() {
        assert_eq!(Config::backend_from_name("auto"), Some(abi::backend::NONE));
        assert_eq!(Config::backend_from_name("soft-cpu"), Some(abi::backend::SOFT_CPU));
        assert_eq!(Config::backend_from_name("d3d11"), Some(abi::backend::D3D11));
        assert_eq!(Config::backend_from_name("d3d9"), None);
        assert_eq!(Config::tier_from_name("t2"), Some(2));
        assert_eq!(Config::tier_from_name("cpu-ram"), Some(2));
        assert_eq!(Config::tier_from_name("9"), Some(4), "a tier above the ladder clamps");
        assert_eq!(Config::tier_from_name("nonsense"), None);
    }

    /// A probe is a report about the machine, not a device: it names every
    /// backend it knows and recommends one, and it is usable before anything has
    /// been created.
    ///
    /// What it does *not* do - allocate through the host allocator - is a claim
    /// about the process-wide ledger, so it is asserted where the process has one
    /// thread: `reconl-info --probe-only` reports the counters and
    /// `tests/cli.rs` checks them there. Asserting it here would be measuring
    /// whichever other test in this crate happened to run alongside.
    #[test]
    fn probing_names_every_backend_without_creating_a_device() {
        let probed = probe(None).expect("probe");
        assert!(probed.entries().len() >= 3, "soft-cpu, null and d3d11 are always probed");
        assert!(!probed.recommended_backend().is_empty());
        assert!(!probed.recommended_tier().is_empty());
        assert!(probed.entries().iter().any(|e| e.backend == abi::backend::SOFT_CPU && e.usable != 0));
    }

    /// The reference tier is always creatable, so a tool always has something to
    /// measure, and the device it hands back is the one it was asked for.
    #[test]
    fn the_reference_tier_is_creatable_and_reports_its_own_limits() {
        let device = Device::create(&Config {
            backend: abi::backend::SOFT_CPU,
            tier: 2,
            allow_downgrade: abi::allow_downgrade::NONE,
            threads: 1,
            ..Config::default()
        })
        .expect("soft-cpu device");
        let limits = device.limits().expect("limits");
        assert_eq!(limits.backend, abi::backend::SOFT_CPU);
        assert!(limits.max_allocation_bytes > 0);
        assert_ne!(names::caps(limits.caps), "none");
        assert!(device.stats().is_ok());
        assert!(device.memory().is_ok());
        // The device names its own backend and driver; an empty driver is
        // legitimate on a tier with no driver to name, so only the backend is
        // asserted to be named.
        assert_eq!(device.backend_name(), "soft-cpu");
    }
}
