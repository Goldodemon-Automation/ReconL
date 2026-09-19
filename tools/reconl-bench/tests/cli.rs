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
