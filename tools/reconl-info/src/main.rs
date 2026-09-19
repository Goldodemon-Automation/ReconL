//! `reconl-info` — what this machine can do, before anything is drawn.
//!
//! Three questions, answered in order, each with the evidence attached:
//!
//! * **What does the machine have?** `reconlProbe` reports every backend the
//!   library knows, whether each is usable, its capabilities, its tier and the
//!   memory it sees - and it creates no device. The tool measures that claim
//!   rather than repeating it: the host allocator's ledger is printed across the
//!   probe, and it is zero.
//! * **What would the library choose?** The recommendation, which is the tier a
//!   device with no hint would resolve to.
//! * **What does the device actually grant?** Limits, caps, the shadow texel
//!   budget, the memory ledger, and the host allocator's own view of what the
//!   device took and gave back - including that it gave all of it back after
//!   release.
//!
//! Exit codes: 0 reported, 2 error.

use reconl::abi;
use reconl_host::device::{self, Config, Device};
use reconl_host::units::{self, bytes};
use reconl_host::names;
use std::process::ExitCode;

const USAGE: &str = "\
reconl-info — report what ReconL can do on this machine

USAGE:
  reconl-info [options]

OPTIONS:
  --probe-only          report the probe and stop. Creates no device, and prints
                        the host allocations the probe made (the claim is zero).
  --all-devices         create a device for every usable backend, not just the
                        recommended one, and report each.
  --backend=NAME        auto | soft-cpu | null | d3d11 | d3d12 | vulkan | gl |
                        metal | webgpu | wasm-webgl2
  --tier=N|NAME         auto | t0..t4 | gpu-discrete | gpu-shared | cpu-ram |
                        cpu-thrifty | out-of-core
  --threads=N           worker threads; 0 lets the backend choose
  --ram-cap=SIZE        device RAM cap, e.g. 64MB
  --vram-cap=SIZE       device video memory cap
  --disk-cap=SIZE       spill arena size cap
  --spill=0|1           allow the disk spill arena
  --spill-dir=PATH      directory the arena may use
  --help                this text

Sizes accept B, K, M, G and T, all binary: 64MB is 64 MiB.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::from(0),
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(2)
        }
    }
}

const FLAGS: [&str; 12] = [
    "--probe-only",
    "--all-devices",
    "--backend",
    "--tier",
    "--threads",
    "--ram-cap",
    "--vram-cap",
    "--disk-cap",
    "--spill",
    "--spill-dir",
    "--help",
    "--device",
];

fn run(args: &[String]) -> Result<(), String> {
    if units::has_flag(args, "--help") || args.iter().any(|a| a == "-h") {
        print!("{USAGE}");
        return Ok(());
    }
    units::reject_unknown(args, &FLAGS)?;
    let probe_only = units::has_flag(args, "--probe-only");

    version();
    let config = config_from(args)?;
    let spill_dir = config.spill_dir.clone();

    // --- what the machine has, and what the probe costs -----------------------
    println!("\nprobe (no device created)");
    reconl_host::alloc::reset();
    let probed = device::probe(spill_dir.as_deref())?;
    let after_probe = reconl_host::alloc::counters();
    print_probe(&probed);
    println!(
        "  host allocations during the probe: {} ({} bytes requested, {} outstanding)",
        after_probe.alloc_calls, after_probe.alloc_bytes, after_probe.live_blocks()
    );
    if probe_only {
        return Ok(());
    }

    // --- what a device grants -------------------------------------------------
    let wanted: Vec<Option<u32>> = if units::has_flag(args, "--all-devices") {
        probed
            .entries()
            .iter()
            .filter(|e| e.usable != 0)
            .map(|e| Some(e.backend))
            .collect()
    } else if config.backend != abi::backend::NONE {
        vec![Some(config.backend)]
    } else {
        vec![None]
    };

    for wanted_backend in wanted {
        let mut one = config.clone();
        if let Some(backend) = wanted_backend {
            one.backend = backend;
        }
        let requested = if one.backend == abi::backend::NONE {
            "auto".to_string()
        } else {
            names::backend(one.backend)
        };
        match Device::create(&one) {
            Ok(device) => {
                report_device(&device, &requested)?;
                drop(device);
                // Read *after* the release, which is the only way "gave every
                // block back" is a measurement rather than an intention.
                let ledger = reconl_host::alloc::counters();
                println!(
                    "  host ledger after release: {} allocations, {} peak, {} outstanding",
                    ledger.alloc_calls,
                    bytes(ledger.peak_bytes),
                    ledger.live_blocks()
                );
            }
            Err(e) => {
                // A backend that is probed as usable but cannot be created is a
                // finding about this machine, not a tool failure: the other
                // backends still get reported.
                println!("\ndevice: {requested}");
                println!("  refused: {e}");
            }
        }
    }
    Ok(())
}

fn version() {
    let (mut major, mut minor, mut patch) = (0u32, 0u32, 0u32);
    reconl::reconlVersion(&mut major, &mut minor, &mut patch);
    // SAFETY: reconlVersionString returns a static NUL-terminated string.
    let long = unsafe { reconl_host::cstr_of(reconl::reconlVersionString()) };
    println!(
        "reconl-info — ReconL {major}.{minor}.{patch} (ABI {}), built against this header\n{long}",
        abi::ABI_VERSION
    );
}

fn config_from(args: &[String]) -> Result<Config, String> {
    let mut config = Config::default();
    if let Some(name) = units::value_of(args, "--backend") {
        config.backend = Config::backend_from_name(name)
            .ok_or_else(|| format!("`{name}` is not a backend (see --help)"))?;
    }
    if let Some(name) = units::value_of(args, "--tier") {
        config.tier = Config::tier_from_name(name)
            .ok_or_else(|| format!("`{name}` is not a tier (see --help)"))?;
    }
    if let Some(v) = units::value_of(args, "--threads") {
        config.threads = units::parse_u32(v, "thread count")?;
    }
    if let Some(v) = units::value_of(args, "--ram-cap") {
        config.ram_cap = units::parse_size(v)?;
    }
    if let Some(v) = units::value_of(args, "--vram-cap") {
        config.vram_cap = units::parse_size(v)?;
    }
    if let Some(v) = units::value_of(args, "--disk-cap") {
        config.disk_cap = units::parse_size(v)?;
    }
    if let Some(v) = units::value_of(args, "--spill") {
        config.allow_disk_spill = v == "1" || v == "on" || v == "true";
    }
    if let Some(dir) = units::value_of(args, "--spill-dir") {
        config.spill_dir = Some(dir.to_string());
    }
    Ok(config)
}

fn print_probe(probed: &device::Probed) {
    println!(
        "  {:<11} {:<6} {:<16} {:>10} {:>12} {:>8} {:>13}  {}",
        "backend", "usable", "best tier", "vram", "ram", "cascades", "shadow budget", "device"
    );
    for e in probed.entries() {
        let usable = if e.usable != 0 { "yes" } else { "no" };
        let vram = if e.vram_bytes == 0 { "—".to_string() } else { bytes(e.vram_bytes) };
        println!(
            "  {:<11} {:<6} {:<16} {:>10} {:>12} {:>8} {:>13}  {}",
            names::field(&e.name),
            usable,
            names::tier(e.best_tier),
            vram,
            bytes(e.ram_bytes),
            e.max_cascades,
            bytes(e.shadow_texel_budget),
            names::field(&e.device_name)
        );
        println!("              caps: {}", names::caps(e.caps));
        let note = names::field(&e.note);
        if !note.is_empty() {
            println!("              note: {note}");
        }
    }
    println!(
        "\n  recommended: {} on {}",
        probed.recommended_backend(),
        probed.recommended_tier()
    );
}

fn report_device(device: &Device, requested: &str) -> Result<(), String> {
    let limits = device.limits()?;
    let memory = device.memory()?;
    let stats = device.stats()?;

    println!("\ndevice: {} (requested {requested})", names::backend(limits.backend));
    println!(
        "  resolved tier: {} — {}",
        names::tier(stats.tier),
        names::field(&stats.tier_reason_text)
    );
    if !device.device_name().is_empty() {
        println!("  device: {}  driver: {}", device.device_name(), device.driver());
    }
    println!(
        "  limits: vram {}  ram {}  disk {}  max allocation {}",
        bytes(limits.vram_bytes),
        bytes(limits.ram_bytes),
        bytes(limits.disk_bytes),
        bytes(limits.max_allocation_bytes)
    );
    println!(
        "  limits: worker threads {}  tile size {}  cascades {}  lights {}  shadow budget {}",
        limits.worker_threads_max,
        limits.tile_size_min,
        limits.max_cascades,
        limits.max_lights,
        bytes(limits.shadow_texel_budget_bytes)
    );
    println!("  caps: {}", names::caps(limits.caps));
    println!(
        "  memory: resident {} of {} (peak {})  spill {} on disk of {} ({} entries, {} evictions, {} errors)",
        bytes(memory.ram_resident_bytes),
        bytes(memory.ram_budget_bytes),
        bytes(memory.ram_peak_bytes),
        bytes(memory.spill_disk_bytes),
        bytes(memory.spill_disk_cap_bytes),
        memory.spill_entries,
        memory.spill_evictions,
        memory.spill_errors
    );
    println!(
        "  device ledger for the host allocator: {} allocations, {} bytes, {} frees",
        memory.host_alloc_calls, memory.host_alloc_bytes, memory.host_free_calls
    );
    println!(
        "  counters: frames presented {}  dropped {}  failures {}  safe-path events {}  downgrades {}",
        stats.frames_presented,
        stats.frames_dropped,
        stats.failures,
        stats.safe_path_events,
        stats.downgrade_count
    );
    for i in 0..(stats.downgrade_count as usize).min(abi::RECONL_MAX_DOWNGRADES) {
        let d = &stats.downgrades[i];
        println!(
            "    downgrade {} -> {} at frame {}: {} ({})",
            names::tier(d.from),
            names::tier(d.to),
            d.frame_index,
            names::field(&d.detail),
            d.reason
        );
    }

    // A device that recorded an error while reporting healthy counters is worth
    // knowing about, because the next call may not be as lucky.
    if let Some(last) = device.last_error() {
        println!("  last error: {last}");
    }
    Ok(())
}
