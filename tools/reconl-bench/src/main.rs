//! `reconl-bench` — a timed run with its configuration attached.
//!
//! The failure this tool exists to prevent is a number without the config that
//! produced it (PROMPT §13). So every run prints a **fingerprint** - backend,
//! tier, device, driver, scene, resolution, shadow mode, budgets, thread count,
//! frame and warmup counts - *before* the first frame, and every figure below it
//! is recorded rather than read:
//!
//! * **min / avg / max**, never a peak quoted as the result;
//! * the **host's** split of a frame at the three ABI boundaries (begin, submit,
//!   present), which every backend has;
//! * the **device's** split (shadow, raster, bin, upload), labelled as the
//!   device's own report and reported as zero where a backend does not
//!   instrument it - the hardware path has no GPU timestamps, and saying so is
//!   the honest answer rather than printing a plausible one;
//! * **shadow cost attributed**: pass nanoseconds, cascades, map resolution,
//!   filter active against requested, and the cache counters;
//! * **allocations during the measured frames**, from the host allocator's own
//!   ledger - the steady-state claim, measured.
//!
//! `--trace=FILE` writes the per-frame record and a per-second summary, so a
//! result can be re-read rather than re-run; `--png=FILE` writes the last frame,
//! which is how `--shadows=off` and `--spill=1` runs are checked against a
//! golden rather than taken on trust.
//!
//! Exit codes: 0 completed, 2 error.

use reconl::abi;
use reconl_host::device::{Config, Device};
use reconl_host::frame::{FrameCost, Options as RenderOptions, Renderer};
use reconl_host::scene::{Scene, ShadowMode};
use reconl_host::units::{self, bytes, fps, mpix_per_sec, ns};
use reconl_host::names;
use std::process::ExitCode;

const USAGE: &str = "\
reconl-bench — time the reference scene through the ReconL C ABI

USAGE:
  reconl-bench [options]

OPTIONS:
  --backend=NAME        auto (default) | soft-cpu | null | d3d11 | ...
  --tier=N|NAME         auto (default) | t0..t4 | gpu-shared | cpu-ram | ...
  --frames=N            measured frames (default 60)
  --warmup=N            unmeasured frames first (default 5)
  --width=N             frame width (default 64, the golden scene's width)
  --height=N            frame height (default 64)
  --resolution=WxH      shorthand for both
  --repeat=N            draw the scene N times per frame (default 1)
  --shadows=off|on|cached   shadow mode (default on)
  --threads=N           worker threads; 0 lets the backend choose
  --ram-cap=SIZE        device RAM cap, e.g. 64MB
  --vram-cap=SIZE       device video memory cap
  --disk-cap=SIZE       spill arena size cap
  --spill=0|1           allow the disk spill arena
  --spill-dir=PATH      directory the arena may use
  --target-ms=N         frame-time target for the tier ladder; 0 disables it
  --audit=N             re-verify tier and budget every N frames (expensive)
  --trace=PATH          write the per-frame trace and per-second summary
  --png=PATH            write the last frame as an RGBA PNG
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

const FLAGS: [&str; 20] = [
    "--backend",
    "--tier",
    "--frames",
    "--warmup",
    "--width",
    "--height",
    "--resolution",
    "--repeat",
    "--shadows",
    "--threads",
    "--ram-cap",
    "--vram-cap",
    "--disk-cap",
    "--spill",
    "--spill-dir",
    "--target-ms",
    "--audit",
    "--trace",
    "--png",
    "--help",
];

/// What the run was asked for, before a device exists.
struct Options {
    config: Config,
    frames: u32,
    warmup: u32,
    width: u32,
    height: u32,
    repeat: u32,
    shadows: ShadowMode,
    audit_every: Option<u32>,
    trace: Option<String>,
    png: Option<String>,
}

fn options_from(args: &[String]) -> Result<Options, String> {
    let mut config = Config::default();
    let mut options = Options {
        config: Config::default(),
        frames: 60,
        warmup: 5,
        width: 64,
        height: 64,
        repeat: 1,
        shadows: ShadowMode::On,
        audit_every: None,
        trace: None,
        png: None,
    };
    if let Some(name) = units::value_of(args, "--backend") {
        config.backend = Config::backend_from_name(name)
            .ok_or_else(|| format!("`{name}` is not a backend (see --help)"))?;
    }
    if let Some(name) = units::value_of(args, "--tier") {
        config.tier = Config::tier_from_name(name)
            .ok_or_else(|| format!("`{name}` is not a tier (see --help)"))?;
    }
    if let Some(v) = units::value_of(args, "--frames") {
        options.frames = units::parse_u32(v, "frame count")?.max(1);
    }
    if let Some(v) = units::value_of(args, "--warmup") {
        options.warmup = units::parse_u32(v, "warmup count")?;
    }
    if let Some(v) = units::value_of(args, "--width") {
        options.width = units::parse_u32(v, "frame width")?.max(1);
    }
    if let Some(v) = units::value_of(args, "--height") {
        options.height = units::parse_u32(v, "frame height")?.max(1);
    }
    if let Some(v) = units::value_of(args, "--resolution") {
        let (w, h) = v
            .split_once(['x', 'X'])
            .ok_or_else(|| format!("`{v}` is not a resolution: expected WxH"))?;
        options.width = units::parse_u32(w, "frame width")?.max(1);
        options.height = units::parse_u32(h, "frame height")?.max(1);
    }
    if let Some(v) = units::value_of(args, "--repeat") {
        options.repeat = units::parse_u32(v, "repeat count")?.max(1);
    }
    if let Some(v) = units::value_of(args, "--shadows") {
        options.shadows =
            ShadowMode::from_name(v).ok_or_else(|| format!("`{v}` is not a shadow mode: off, on or cached"))?;
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
    if let Some(v) = units::value_of(args, "--target-ms") {
        config.target_frame_ms = units::parse_u32(v, "target frame time")?;
    }
    if let Some(v) = units::value_of(args, "--audit") {
        options.audit_every = Some(units::parse_u32(v, "audit interval")?.max(1));
    }
    options.trace = units::value_of(args, "--trace").map(str::to_string);
    options.png = units::value_of(args, "--png").map(str::to_string);
    options.config = config;
    Ok(options)
}

fn run(args: &[String]) -> Result<(), String> {
    if units::has_flag(args, "--help") || args.iter().any(|a| a == "-h") {
        print!("{USAGE}");
        return Ok(());
    }
    units::reject_unknown(args, &FLAGS)?;
    let options = options_from(args)?;
    let scene = Scene::with_shadow_mode(options.shadows);

    let (mut major, mut minor, mut patch) = (0u32, 0u32, 0u32);
    reconl::reconlVersion(&mut major, &mut minor, &mut patch);
    println!("reconl-bench — ReconL {major}.{minor}.{patch} (ABI {})\n", abi::ABI_VERSION);

    // The fingerprint is printed from the *config*, before a device exists, and
    // completed from the device once it does. A number that cannot be traced to
    // one of these lines is not a result.
    let requested_backend = if options.config.backend == abi::backend::NONE {
        "auto".to_string()
    } else {
        names::backend(options.config.backend)
    };
    let requested_tier = if options.config.tier == 0 {
        "auto".to_string()
    } else {
        names::tier(options.config.tier)
    };
    println!("fingerprint (resolved before the first frame)");
    println!("  requested       backend {requested_backend}, tier {requested_tier}");
    println!(
        "  scene           reference: {} chunks, {} triangles, repeat {}",
        scene.chunks.len(),
        scene.triangle_count(),
        options.repeat
    );
    println!("  frame           {} x {}", options.width, options.height);
    println!(
        "  shadows         {} — {} cascades requested, {} texel budget, filter {}",
        options.shadows.name(),
        scene.shadows.cascade_count,
        bytes(scene.shadows.texel_budget_bytes),
        names::filter(scene.shadows.filter)
    );
    println!(
        "  threads         {}",
        if options.config.threads == 0 {
            "0 (backend chooses)".to_string()
        } else {
            options.config.threads.to_string()
        }
    );
    println!(
        "  budgets         ram {}, vram {}, disk {}, spill {}",
        cap(options.config.ram_cap),
        cap(options.config.vram_cap),
        cap(options.config.disk_cap),
        if options.config.allow_disk_spill { "on" } else { "off" }
    );
    if let Some(dir) = &options.config.spill_dir {
        println!("  spill dir       {dir}");
    }
    println!("  warmup/measured {} + {} frames", options.warmup, options.frames);
    if let Some(n) = options.audit_every {
        println!("  audit           every {n} frames (expensive, diagnostics only)");
    }

    let device = Device::create(&options.config)?;
    let limits = device.limits()?;
    println!(
        "  resolved        backend {}, tier {}",
        names::backend(limits.backend),
        names::tier(device.stats()?.tier)
    );
    if !device.device_name().is_empty() {
        println!("  device          {}", device.device_name());
    }
    if !device.driver().is_empty() {
        println!("  driver          {}", device.driver());
    }
    println!("  caps            {}", names::caps(limits.caps));

    let renderer = Renderer::new(
        &device,
        &scene,
        RenderOptions {
            width: options.width,
            height: options.height,
            present_to_memory: true,
            repeat: options.repeat,
            max_draws: 0,
        },
    )?;
    let mut pixels = vec![0u8; renderer.frame_bytes()];

    // A frame that fails is reported with the device's own last error, which is
    // where the backend's real cause lives: the result code alone says *that* a
    // call failed, and the last-error text is the only place the detail (an
    // HRESULT, a limit, a rejected descriptor) survives.
    let mut render = |seed: u32| -> Result<FrameCost, String> {
        renderer.frame(&scene, seed, &mut pixels).map_err(|e| match device.last_error() {
            Some(detail) => format!("{e} — device reported: {detail}"),
            None => e,
        })
    };

    // Warmup is unmeasured on purpose: the first frames of any backend pay for
    // allocation, shader compilation and cache misses that a steady state does
    // not, and quoting them as the result is the mistake §13 is about.
    for i in 0..options.warmup {
        render(1 + i)?;
    }
    device.reset_stats()?;
    if let Some(n) = options.audit_every {
        device.audit(n)?;
    }
    reconl_host::alloc::reset();

    let mut run = Run::default();
    for i in 0..options.frames {
        let cost = render(1 + options.warmup + i)?;
        let stats = device.stats()?;
        run.record(cost, &stats);
    }
    let ledger = reconl_host::alloc::counters();
    let stats = device.stats()?;

    run.report(&stats, &options, &ledger, &limits);
    if let Some(path) = &options.trace {
        std::fs::write(path, run.trace(&stats, &options)).map_err(|e| format!("{path}: {e}"))?;
        println!("\ntrace written to {path} ({} frames)", run.frames());
    }
    if let Some(path) = &options.png {
        let png = reconl_png::write_rgba8(options.width, options.height, &pixels)?;
        std::fs::write(path, png).map_err(|e| format!("{path}: {e}"))?;
        println!("last frame written to {path}");
    }
    if stats.audit_divergences > 0 {
        println!("\nAUDIT: {} divergences reported", stats.audit_divergences);
    }
    Ok(())
}

fn cap(value: u64) -> String {
    if value == 0 {
        "none".to_string()
    } else {
        bytes(value)
    }
}

/// One measured frame: the host's three boundaries and the device's own split.
#[derive(Clone, Copy, Default)]
struct Sample {
    total_ns: u64,
    begin_ns: u64,
    submit_ns: u64,
    present_ns: u64,
    device_total_ns: u64,
    shadow_ns: u64,
    raster_ns: u64,
    bin_ns: u64,
    upload_ns: u64,
    spill_wait_ns: u64,
    pixels: u32,
    triangles: u32,
    tiles: u32,
}

#[derive(Default)]
struct Second {
    frames: u32,
    total_ns: u64,
    worst_ns: u64,
}

/// Every measured frame, kept whole.
///
/// Kept as samples rather than running totals because the trace prints each
/// frame's own numbers, and a running total printed per frame is the kind of
/// near-enough instrumentation that makes a trace useless for finding the one
/// frame that stalled.
#[derive(Default)]
struct Run {
    samples: Vec<Sample>,
    per_second: Vec<Second>,
    elapsed_ns: u64,
}

impl Run {
    fn frames(&self) -> u32 {
        self.samples.len() as u32
    }

    fn record(&mut self, cost: FrameCost, stats: &abi::ReconLStats) {
        let frame = &stats.frame;
        self.samples.push(Sample {
            total_ns: cost.total_ns(),
            begin_ns: cost.begin_ns,
            submit_ns: cost.submit_ns,
            present_ns: cost.present_ns,
            device_total_ns: frame.total_ns,
            shadow_ns: frame.shadow_ns,
            raster_ns: frame.raster_ns,
            bin_ns: frame.bin_ns,
            upload_ns: frame.upload_ns,
            spill_wait_ns: frame.spill_wait_ns,
            pixels: frame.pixels_shaded,
            triangles: frame.triangles_binned,
            tiles: frame.tiles_rendered,
        });
        // A "second" here is a wall-clock second of the run, not a frame count,
        // so a slow configuration shows the stall in the bucket it happened in.
        self.elapsed_ns += cost.total_ns();
        let bucket = (self.elapsed_ns / 1_000_000_000) as usize;
        while self.per_second.len() <= bucket {
            self.per_second.push(Second::default());
        }
        let second = &mut self.per_second[bucket];
        second.frames += 1;
        second.total_ns += cost.total_ns();
        second.worst_ns = second.worst_ns.max(cost.total_ns());
    }

    /// The mean of one field over every measured frame.
    fn mean(&self, field: impl Fn(&Sample) -> u64) -> u64 {
        if self.samples.is_empty() {
            0
        } else {
            self.samples.iter().map(field).sum::<u64>() / self.samples.len() as u64
        }
    }

    fn total(&self, field: impl Fn(&Sample) -> u64) -> u64 {
        self.samples.iter().map(field).sum()
    }

    fn bounds(&self) -> (u64, u64, u64) {
        let min = self.samples.iter().map(|s| s.total_ns).min().unwrap_or(0);
        let max = self.samples.iter().map(|s| s.total_ns).max().unwrap_or(0);
        (min, self.mean(|s| s.total_ns), max)
    }

    fn report(&self, stats: &abi::ReconLStats, options: &Options, ledger: &reconl_host::alloc::Counters, limits: &abi::ReconLDeviceLimits) {
        let (min, avg, max) = self.bounds();
        println!("\nsteady state, {} measured frames", self.frames());
        println!(
            "  wall            min {}  avg {}  max {}   ({} fps at the mean)",
            ns(min),
            ns(avg),
            ns(max),
            fps(avg)
        );
        println!(
            "  host split      begin {}  submit {}  present {}   (mean)",
            ns(self.mean(|s| s.begin_ns)),
            ns(self.mean(|s| s.submit_ns)),
            ns(self.mean(|s| s.present_ns))
        );
        let instrumented = self.total(|s| s.device_total_ns) > 0;
        if instrumented {
            println!(
                "  device split    total {}  shadow {}  raster {}  bin {}  upload {}  spill-wait {}   (mean, as the device reports it)",
                ns(self.mean(|s| s.device_total_ns)),
                ns(self.mean(|s| s.shadow_ns)),
                ns(self.mean(|s| s.raster_ns)),
                ns(self.mean(|s| s.bin_ns)),
                ns(self.mean(|s| s.upload_ns)),
                ns(self.mean(|s| s.spill_wait_ns))
            );
        } else {
            println!(
                "  device split    not instrumented on this backend (no stage timings reported); the host split above is the boundary cost"
            );
        }
        let s = &stats.shadows;
        println!(
            "  shadow          cascades {}  map {} x {} ({})  filter {} (requested {})  pass {}",
            s.cascades_active,
            s.map_width,
            s.map_height,
            bytes(u64::from(s.map_bytes)),
            names::filter(s.filter_active),
            names::filter(s.filter_requested),
            if instrumented { ns(self.mean(|f| f.shadow_ns)) } else { "—".into() }
        );
        println!(
            "  shadow cache    hits {}  misses {}  read {}  hit {}  frozen cascades {}  fail-safe unshadowed {}",
            s.cache_hits,
            s.cache_misses,
            bytes(s.cache_bytes_read),
            bytes(s.cache_bytes_hit),
            s.frozen_cascades,
            s.fail_safe_unshadowed
        );
        println!(
            "  work            pixels shaded {}/frame ({}), triangles binned {}/frame, tiles {}/frame",
            self.mean(|s| u64::from(s.pixels)),
            mpix_per_sec(self.total(|s| u64::from(s.pixels)), self.total(|s| s.total_ns)),
            self.mean(|s| u64::from(s.triangles)),
            self.mean(|s| u64::from(s.tiles))
        );
        println!(
            "  memory          resident peak {} of {}  spill {} of {}  max allocation {}",
            bytes(stats.memory.ram_peak_bytes),
            bytes(limits.ram_bytes),
            bytes(stats.memory.spill_disk_bytes),
            cap(stats.memory.spill_disk_cap_bytes),
            bytes(limits.max_allocation_bytes)
        );
        // The steady-state target is zero allocations per frame (PROMPT §12), so
        // the rate is the number to read: a frame that allocates is a frame that
        // will hitch when the allocator is busy. Reported as measured either way.
        println!(
            "  allocations     {} during the measured frames, {} bytes ({} per frame; the steady-state target is 0)",
            ledger.alloc_calls,
            ledger.alloc_bytes,
            if self.frames() == 0 { 0.0 } else { ledger.alloc_calls as f64 / f64::from(self.frames()) }
        );
        println!(
            "  counters        presented {}  dropped {}  failures {}  safe-path events {}  audit divergences {}",
            stats.frames_presented,
            stats.frames_dropped,
            stats.failures,
            stats.safe_path_events,
            stats.audit_divergences
        );
        // What the device ended on. The fingerprint's "resolved" line is what was
        // *asked for*, and a fallback moves it: a run that lost its GPU has to
        // show the tier it finished on, or the fallback is invisible in the one
        // output a host reads.
        println!(
            "  final           backend {}  tier {}  reason {}",
            names::backend(stats.backend),
            names::tier(stats.tier),
            names::field(&stats.tier_reason_text)
        );
        let _ = options;
    }

    /// The trace: the fingerprint, every frame, then the per-second buckets.
    fn trace(&self, stats: &abi::ReconLStats, options: &Options) -> String {
        let mut out = String::new();
        out.push_str("reconl-bench trace v1\n");
        out.push_str(&format!("# frames {} warmup {}\n", self.frames(), options.warmup));
        out.push_str(&format!("# frame {}x{} repeat {}\n", options.width, options.height, options.repeat));
        out.push_str(&format!("# shadows {}\n", options.shadows.name()));
        out.push_str(&format!("# tier {}\n", names::tier(stats.tier)));
        out.push_str(&format!("# backend {}\n", names::backend(stats.backend)));
        out.push_str(&format!(
            "# shadow cascades {} map {}x{} filter {} (requested {})\n",
            stats.shadows.cascades_active,
            stats.shadows.map_width,
            stats.shadows.map_height,
            names::filter(stats.shadows.filter_active),
            names::filter(stats.shadows.filter_requested)
        ));
        out.push_str(&format!("# device {}\n", names::field(&stats.device_name)));
        out.push_str("frame,total_ms,begin_us,submit_us,present_us,shadow_us,pixels\n");
        for (i, s) in self.samples.iter().enumerate() {
            out.push_str(&format!(
                "{i},{:.3},{},{},{},{},{}\n",
                s.total_ns as f64 / 1e6,
                s.begin_ns / 1_000,
                s.submit_ns / 1_000,
                s.present_ns / 1_000,
                s.shadow_ns / 1_000,
                s.pixels
            ));
        }
        out.push_str("second,frames,mean_ms,max_ms\n");
        for (i, second) in self.per_second.iter().enumerate() {
            out.push_str(&format!(
                "{i},{},{:.3},{:.3}\n",
                second.frames,
                (second.total_ns as f64 / f64::from(second.frames.max(1))) / 1e6,
                second.worst_ns as f64 / 1e6
            ));
        }
        out
    }
}
