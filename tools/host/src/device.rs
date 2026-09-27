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

/// How a host asks D3D11 to choose its physical adapter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdapterSelection {
    #[default]
    Auto,
    Integrated,
    Discrete,
    Index(u32),
    Luid(u64),
}

impl AdapterSelection {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "auto" => Self::Auto,
            "integrated" | "igpu" => Self::Integrated,
            "discrete" | "dgpu" => Self::Discrete,
            value if value.starts_with("luid:") || value.starts_with("LUID:") => {
                let value = &value[5..];
                let value = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")).unwrap_or(value);
                Self::Luid(u64::from_str_radix(value, 16).ok()?)
            }
            value => {
                let index = value.parse::<u32>().ok()?;
                i32::try_from(index).ok()?;
                Self::Index(index)
            }
        })
    }

    pub fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }
}

impl core::fmt::Display for AdapterSelection {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto (discrete first)"),
            Self::Integrated => f.write_str("integrated"),
            Self::Discrete => f.write_str("discrete"),
            Self::Index(index) => write!(f, "index {index}"),
            Self::Luid(luid) => write!(f, "LUID {luid:016x}"),
        }
    }
}

/// One call to the C ABI's two-call adapter enumeration contract.
pub fn enumerate_adapters(backend: u32) -> Result<Vec<abi::ReconLAdapterInfo>, String> {
    let mut count = 0u32;
    let r = unsafe { reconl::reconlEnumerateAdapters(backend, core::ptr::null_mut(), 0, &mut count) };
    if r != abi::result::OK {
        return Err(failed("reconlEnumerateAdapters", r));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut adapters = vec![abi::ReconLAdapterInfo::default(); count as usize];
    let mut written_count = 0u32;
    let r = unsafe {
        reconl::reconlEnumerateAdapters(backend, adapters.as_mut_ptr(), count, &mut written_count)
    };
    if r != abi::result::OK {
        return Err(failed("reconlEnumerateAdapters", r));
    }
    if written_count > count {
        return Err("reconlEnumerateAdapters returned more adapters than the supplied capacity".into());
    }
    adapters.truncate(written_count as usize);
    Ok(adapters)
}

/// Resolves the D3D11 selection policy against an enumerated list, for reporting
/// the exact adapter a created device should be using.
pub fn selected_adapter(
    adapters: &[abi::ReconLAdapterInfo],
    selection: AdapterSelection,
) -> Option<&abi::ReconLAdapterInfo> {
    match selection {
        AdapterSelection::Index(index) => adapters.get(index as usize).filter(|adapter| adapter.usable != 0),
        AdapterSelection::Luid(luid) => adapters
            .iter()
            .find(|adapter| adapter.adapter_luid == luid && adapter.usable != 0),
        AdapterSelection::Auto | AdapterSelection::Discrete | AdapterSelection::Integrated => {
            let preferred_type = match selection {
                AdapterSelection::Integrated => abi::adapter_type::INTEGRATED,
                AdapterSelection::Auto | AdapterSelection::Discrete => abi::adapter_type::DISCRETE,
                AdapterSelection::Index(_) | AdapterSelection::Luid(_) => unreachable!(),
            };
            adapters
                .iter()
                .find(|adapter| adapter.usable != 0 && adapter.adapter_type == preferred_type)
                .or_else(|| adapters.iter().find(|adapter| adapter.usable != 0))
        }
    }
}

fn effective_backend_hint(config: &Config) -> u32 {
    if config.backend == abi::backend::NONE && !config.adapter.is_auto() {
        abi::backend::D3D11
    } else {
        config.backend
    }
}

fn validate_adapter_backend(config: &Config) -> Result<(), String> {
    if !config.adapter.is_auto() && !matches!(config.backend, abi::backend::NONE | abi::backend::D3D11) {
        return Err("D3D11 adapter selection can only be used with the D3D11 or auto backend".into());
    }
    Ok(())
}

#[cfg(test)]
mod adapter_selection_tests {
    use super::*;

    fn adapter(luid: u64, adapter_type: u32, usable: u32) -> abi::ReconLAdapterInfo {
        abi::ReconLAdapterInfo {
            adapter_type,
            usable,
            adapter_luid: luid,
            ..abi::ReconLAdapterInfo::default()
        }
    }

    #[test]
    fn reported_adapter_matches_the_creation_selection_policy() {
        let adapters = [
            adapter(1, abi::adapter_type::INTEGRATED, 1),
            adapter(2, abi::adapter_type::DISCRETE, 1),
            adapter(3, abi::adapter_type::DISCRETE, 0),
            adapter(4, abi::adapter_type::UNKNOWN, 1),
        ];
        assert_eq!(selected_adapter(&adapters, AdapterSelection::Auto).unwrap().adapter_luid, 2);
        assert_eq!(selected_adapter(&adapters, AdapterSelection::Discrete).unwrap().adapter_luid, 2);
        assert_eq!(selected_adapter(&adapters, AdapterSelection::Integrated).unwrap().adapter_luid, 1);
        assert_eq!(selected_adapter(&adapters, AdapterSelection::Index(3)).unwrap().adapter_luid, 4);
        assert_eq!(selected_adapter(&adapters, AdapterSelection::Luid(4)).unwrap().adapter_luid, 4);
        assert!(selected_adapter(&adapters, AdapterSelection::Index(2)).is_none());
        assert!(selected_adapter(&adapters, AdapterSelection::Luid(99)).is_none());
        assert_eq!(selected_adapter(&adapters[3..], AdapterSelection::Auto).unwrap().adapter_luid, 4);
    }

    #[test]
    fn an_explicit_adapter_turns_auto_backend_selection_into_d3d11() {
        let mut config = Config::default();
        assert_eq!(effective_backend_hint(&config), abi::backend::NONE);

        config.adapter = AdapterSelection::Integrated;
        assert_eq!(effective_backend_hint(&config), abi::backend::D3D11);
        assert!(validate_adapter_backend(&config).is_ok());

        config.backend = abi::backend::SOFT_CPU;
        assert!(validate_adapter_backend(&config).is_err());
    }
}

// The user-facing host configuration is resolved to the D3D11 descriptor only
// for D3D11. Non-D3D11 backend descriptors retain their reserved-pointer contract.
// Keep the synchronous ABI conversion below scoped to the host API.

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
    pub adapter: AdapterSelection,
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
            adapter: AdapterSelection::Auto,
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

    /// The disk budget a spill gets when the host did not name one.
    ///
    /// The ABI is explicit that `disk_cap_bytes` of 0 means *no disk use at all*
    /// (see `ReconLMemoryBudget`), so "allow the spill, name no budget" cannot
    /// reach the arena as it stands. A tool that passes the flag through as a
    /// zero would honour the letter of the header and quietly do nothing: the run
    /// reports `spill 0 B of none` and `spill off`, and the host has no way to see
    /// that the arena it asked for never opened. Asking to spill therefore means
    /// accepting this much disk unless the host says otherwise - a budget the
    /// host *does* name is never touched.
    pub const DEFAULT_DISK_CAP: u64 = 1 << 30;

    /// Fills in the disk budget a bare `--spill=1` implies.
    pub fn resolve_disk_budget(&mut self) {
        if self.allow_disk_spill && self.disk_cap == 0 {
            self.disk_cap = Self::DEFAULT_DISK_CAP;
        }
    }

    /// A tier by number (`0`..`4`) or by the library's own name.
    pub fn adapter_from_name(name: &str) -> Option<AdapterSelection> {
        AdapterSelection::from_name(name)
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
        validate_adapter_backend(config)?;
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

        // The backend descriptor is consumed synchronously inside
        // reconlCreateDevice and all selected values are copied into D3d11Config.
        // Software backends keep their reserved-pointer contract.
        let backend_desc = if matches!(config.backend, abi::backend::NONE | abi::backend::D3D11)
            && !config.adapter.is_auto()
        {
            let (adapter_preference, adapter_index, adapter_luid) = match config.adapter {
                AdapterSelection::Auto => (abi::adapter_preference::AUTO, 0, 0),
                AdapterSelection::Integrated => (abi::adapter_preference::INTEGRATED, 0, 0),
                AdapterSelection::Discrete => (abi::adapter_preference::DISCRETE, 0, 0),
                AdapterSelection::Index(index) => {
                    let index = i32::try_from(index).map_err(|_| "adapter index exceeds i32".to_string())?;
                    (abi::adapter_preference::INDEX, index, 0)
                }
                AdapterSelection::Luid(luid) => (abi::adapter_preference::LUID, 0, luid),
            };
            Some(abi::ReconLD3D11Desc {
                base: hdr::<abi::ReconLD3D11Desc>(),
                adapter_index,
                feature_level_min: 0,
                debug_layer: 0,
                allow_warp: 0,
                prefer_flip_model: 0,
                reserved: 0,
                requested_vram_cap: 0,
                adapter_preference,
                reserved2: 0,
                adapter_luid,
            })
        } else {
            None
        };

        // SAFETY: every pointer below is either null or points at a descriptor
        // that outlives this call, and the descriptor's header is filled from the
        // Rust type so its size cannot disagree with the header's.
        unsafe {
            let desc = abi::ReconLDeviceDesc {
                base: hdr::<abi::ReconLDeviceDesc>(),
                backend_hint: effective_backend_hint(config),
                tier_hint: config.tier,
                allow_downgrade: config.allow_downgrade,
                worker_threads: config.threads,
                target_frame_ms: config.target_frame_ms,
                downgrade_after_frames: config.downgrade_after_frames,
                seed: config.seed,
                flags: 0,
                budget: budget.as_ref().map_or(core::ptr::null(), |b| &**b as *const _),
                allocator: crate::alloc::allocator(),
                backend_desc: backend_desc.as_ref().map_or(core::ptr::null(), |d| d as *const _ as *const c_void),
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
        assert_eq!(Config::adapter_from_name("integrated"), Some(AdapterSelection::Integrated));
        assert_eq!(Config::adapter_from_name("2"), Some(AdapterSelection::Index(2)));
        assert_eq!(Config::adapter_from_name("-1"), None);
        assert_eq!(Config::adapter_from_name("2147483648"), None);
        assert_eq!(Config::adapter_from_name("luid:0x1a"), Some(AdapterSelection::Luid(0x1a)));
        assert_eq!(Config::adapter_from_name("LUID:ABcd"), Some(AdapterSelection::Luid(0xabcd)));
        assert_eq!(Config::adapter_from_name("luid:nope"), None);
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

    /// A spill with no budget named still has somewhere to spill.
    ///
    /// The ABI is explicit that `disk_cap_bytes` of 0 means *no disk use at all*, so
    /// a tool that passed the flag through as a zero would open no arena while its
    /// fingerprint said "spill on" - the flag a no-op with nothing to see. A budget
    /// the host *does* name is never touched, and asking for no spill infers nothing.
    #[test]
    fn a_spill_without_a_named_budget_gets_the_documented_default() {
        let mut inferred = Config {
            allow_disk_spill: true,
            ..Config::default()
        };
        inferred.resolve_disk_budget();
        assert_eq!(inferred.disk_cap, Config::DEFAULT_DISK_CAP);

        let mut named = Config {
            allow_disk_spill: true,
            disk_cap: 8 << 20,
            ..Config::default()
        };
        named.resolve_disk_budget();
        assert_eq!(
            named.disk_cap,
            8 << 20,
            "a budget the host named is the budget it gets"
        );

        let mut none = Config::default();
        none.resolve_disk_budget();
        assert_eq!(none.disk_cap, 0, "no spill means no budget to infer");
    }
}
