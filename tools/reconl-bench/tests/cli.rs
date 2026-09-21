//! `reconl-bench` driven as a host drives it: the shipped binary, real
//! arguments, real PNG output compared against the committed golden.
//!
//! These run the tool in a *process* rather than calling into it, for two
//! reasons. The allocator's ledger is process-wide, so a claim about it is only
//! meaningful where there is one thread - the tool's own. And the PNG is the
//! artifact a reviewer reads, so comparing bytes is the end-to-end statement:
//! the same scene, through the same ABI, from a tool that exists to time it.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bench(args: &[&str]) -> (String, String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_reconl-bench"))
        .args(args)
        .output()
        .expect("spawn reconl-bench");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("reconl-bench-tests");
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir.join(format!("{name}-{}", std::process::id()))
}

fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/soft-cpu-shadow.png")
}

/// The reference frame the golden was made from, produced by the timing tool:
/// byte-identical, with the tool's own default thread count. That is the
/// determinism claim (PROMPT §12: identical output across worker counts) checked
/// through a binary that has nothing to do with goldens.
#[test]
fn the_reference_frame_it_times_is_the_committed_golden() {
    let png = tmp("golden.png");
    let arg = format!("--png={}", png.display());
    let (stdout, stderr, code) = bench(&["--backend=soft-cpu", "--frames=1", "--warmup=0", &arg]);
    assert_eq!(code, 0, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("fingerprint (resolved before the first frame)"), "{stdout}");
    assert_eq!(
        std::fs::read(&png).expect("bench PNG"),
        std::fs::read(golden()).expect("committed golden"),
        "the timing tool and the golden tool must render the same scene"
    );
}

/// The same run twice is the same bytes, and turning shadows off changes pixels
/// - the two halves of "the golden shows a shadow" from the tool's side.
#[test]
fn a_run_is_reproducible_and_shadows_change_the_frame() {
    let mut runs = Vec::new();
    for name in ["repro-a.png", "repro-b.png"] {
        let path = tmp(name);
        let arg = format!("--png={}", path.display());
        let (_, err, code) = bench(&["--backend=soft-cpu", "--frames=2", "--warmup=1", &arg]);
        assert_eq!(code, 0, "{err}");
        runs.push(path);
    }
    let unshadowed = tmp("repro-off.png");
    let arg = format!("--png={}", unshadowed.display());
    let (_, err, code) = bench(&[
        "--backend=soft-cpu",
        "--frames=2",
        "--warmup=1",
        "--shadows=off",
        &arg,
    ]);
    assert_eq!(code, 0, "{err}");

    let first = std::fs::read(&runs[0]).expect("first run");
    assert_eq!(first, std::fs::read(&runs[1]).expect("second run"), "two identical runs must agree");
    assert_ne!(
        first,
        std::fs::read(&unshadowed).expect("unshadowed run"),
        "--shadows=off must change the frame"
    );
}

/// A capped device with the spill arena enabled renders the same pixels as an
/// uncapped one with it off: the cache changes speed, never output (PROMPT §12).
#[test]
fn a_ram_cap_and_the_spill_arena_do_not_change_a_pixel() {
    let spill_dir = tmp("spill-dir");
    std::fs::create_dir_all(&spill_dir).expect("spill dir");
    let capped = tmp("capped.png");
    let spill = format!("--spill-dir={}", spill_dir.display());
    let png = format!("--png={}", capped.display());
    let (stdout, stderr, code) = bench(&[
        "--backend=soft-cpu",
        "--tier=2",
        "--frames=3",
        "--warmup=1",
        "--ram-cap=64MB",
        "--spill=1",
        &spill,
        &png,
    ]);
    assert_eq!(code, 0, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("ram 64.0 MiB"), "{stdout}");
    assert_eq!(
        std::fs::read(&capped).expect("capped run"),
        std::fs::read(golden()).expect("committed golden"),
        "a 64 MiB cap with spill on must not change the frame"
    );
}

/// The fingerprint is the point of the tool: a run prints what it was before it
/// prints anything else, and reports where the time went.
#[test]
fn every_run_carries_its_configuration_and_its_split() {
    let trace = tmp("trace.csv");
    let arg = format!("--trace={}", trace.display());
    let (stdout, stderr, code) = bench(&[
        "--backend=soft-cpu",
        "--tier=2",
        "--frames=5",
        "--warmup=2",
        "--shadows=cached",
        "--threads=2",
        "--repeat=2",
        &arg,
    ]);
    assert_eq!(code, 0, "stdout:\n{stdout}\nstderr:\n{stderr}");
    for expected in [
        "requested       backend soft-cpu, tier T2/cpu-ram",
        "scene           reference: 2 chunks, 3 triangles, repeat 2",
        "shadows         cached",
        "threads         2",
        "warmup/measured 2 + 5 frames",
        "host split      begin",
        "shadow          cascades",
        "allocations     ",
    ] {
        assert!(stdout.contains(expected), "missing `{expected}` in:\n{stdout}");
    }
    let trace = std::fs::read_to_string(&trace).expect("trace file");
    assert!(trace.starts_with("reconl-bench trace v1\n"), "{trace}");
    assert!(trace.contains("# shadows cached\n"), "{trace}");
    // One frame row per measured frame (6 commas), and at least one per-second
    // bucket row (3 commas) - the two sections of the recorded trace.
    let frame_rows = trace
        .lines()
        .filter(|l| l.matches(',').count() == 6 && l.starts_with(|c: char| c.is_ascii_digit()))
        .count();
    assert_eq!(frame_rows, 5, "five measured frames in:\n{trace}");
    assert!(trace.contains("second,frames,mean_ms,max_ms"), "{trace}");
    assert!(
        trace
            .lines()
            .any(|l| l.matches(',').count() == 3 && l.starts_with(|c: char| c.is_ascii_digit())),
        "a per-second bucket in:\n{trace}"
    );
}

/// A generation run reports the rate the feature exists to raise, and says which
/// frame its wall line times.
///
/// The failure this pins is a report that reads as a regression: the wall line is
/// per *rendered* frame, so a run delivering twice the images printed a lower
/// frame rate than the same run without generation, and the multiplier was only
/// derivable by hand. A host quoting that number would turn the feature off.
#[test]
fn a_generation_run_reports_the_rate_it_achieved() {
    let (stdout, stderr, code) = bench(&[
        "--backend=soft-cpu",
        "--resolution=64x64",
        "--frames=4",
        "--warmup=1",
        "--framegen=2",
    ]);
    assert_eq!(code, 0, "stdout:\n{stdout}\nstderr:\n{stderr}");
    for expected in [
        "frame generation",
        "2.00 images per rendered frame",
        "images/s against",
        "presented per rendered",
        "generated cost  min",
        "per rendered frame:",
    ] {
        assert!(stdout.contains(expected), "missing `{expected}` in:\n{stdout}");
    }

    // Without generation, one frame is one image: the wall line is the run's rate
    // and the report claims nothing about images.
    let (plain, _, code) = bench(&["--backend=soft-cpu", "--resolution=64x64", "--frames=4", "--warmup=1"]);
    assert_eq!(code, 0);
    assert!(plain.contains("fps at the mean"), "{plain}");
    assert!(!plain.contains("per rendered frame:"), "{plain}");
    assert!(!plain.contains("frame generation"), "{plain}");
}

/// A mistyped option is an error, not a silently ignored default - and an option
/// whose value was written as a separate argument says so.
#[test]
fn a_mistyped_option_is_refused() {
    let (_, stderr, code) = bench(&["--frams=10"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("unknown option"), "{stderr}");

    let (_, stderr, code) = bench(&["--png", "out.png"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("take their value with `=`"), "{stderr}");
}

/// A steady-state frame takes nothing from the host allocator.
///
/// This is PROMPT §12's criterion and `docs/determinism.md` §5's rule: storage
/// is reserved before a frame, never during it. The number is the host's own
/// ledger - the counters of the allocator this tool hands the device, in this
/// tool's process - so it counts what the device really took, however it took
/// it. Every list a frame needs (the frame's draws, the reference tier's colour
/// entries and cascade entries) is owned before the frame starts, which is why
/// the line reads zero: a list built per frame puts a number above zero here at
/// once, and that is the regression this pin exists to catch.
#[test]
fn a_steady_state_frame_allocates_nothing() {
    for backend in ["soft-cpu", "d3d11"] {
        let arg = format!("--backend={backend}");
        let (stdout, stderr, code) = bench(&[&arg, "--resolution=64x64", "--frames=40", "--warmup=5"]);
        if code != 0 {
            // No usable device for this backend on this host: the d3d11 legs in
            // `ffi/tests/tiers.rs` and `probes/run.sh` skip the same way. The
            // reference tier is always runnable, so it may not skip.
            assert_eq!(
                backend, "d3d11",
                "the soft-cpu leg must run:\n{stdout}\n{stderr}"
            );
            eprintln!("skipping the {backend} leg: {}", stderr.trim());
            continue;
        }
        let line = stdout
            .lines()
            .find(|l| l.trim_start().starts_with("allocations "))
            .unwrap_or_else(|| panic!("no allocations line in the {backend} run:\n{stdout}"));
        assert!(
            line.contains("(0 per frame;"),
            "{backend}: a steady-state frame must allocate nothing, but the ledger says `{}`\n{stdout}",
            line.trim()
        );
    }
}
