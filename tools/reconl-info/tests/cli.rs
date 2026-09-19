//! `reconl-info` driven as a host drives it.
//!
//! The probe's central claim is that it inspects the machine *without* creating a
//! device, which is a claim about allocations. The allocator's ledger is
//! process-wide, so it is only meaningful where the process has one thread: in
//! the tool's own process, which is exactly what these tests spawn.

use std::process::Command;

fn info(args: &[&str]) -> (String, String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_reconl-info"))
        .args(args)
        .output()
        .expect("spawn reconl-info");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

/// The reference tier is always on the machine, so the report always has at least
/// one usable backend, a recommendation, and a measured probe cost of zero.
#[test]
fn the_probe_reports_the_machine_and_allocates_nothing() {
    let (stdout, stderr, code) = info(&["--probe-only"]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("probe (no device created)"), "{stdout}");
    assert!(stdout.contains("soft-cpu"), "{stdout}");
    assert!(stdout.contains("recommended: "), "{stdout}");
    assert!(
        stdout.contains("host allocations during the probe: 0"),
        "the probe must not allocate:\n{stdout}"
    );
}

/// A device report names the tier it resolved, the limits it will honour, and
/// gives every byte back at release.
#[test]
fn a_device_reports_its_limits_and_gives_the_memory_back() {
    let (stdout, stderr, code) = info(&["--backend=soft-cpu", "--tier=2"]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("resolved tier: T2/cpu-ram"), "{stdout}");
    assert!(stdout.contains("max allocation"), "{stdout}");
    assert!(stdout.contains("device ledger for the host allocator:"), "{stdout}");
    let ledger = stdout
        .lines()
        .find(|l| l.contains("host ledger after release"))
        .unwrap_or_else(|| panic!("no ledger line in:\n{stdout}"));
    assert!(
        ledger.trim_end().ends_with("0 outstanding"),
        "every block must be released: {ledger}"
    );
}

/// A device the library cannot honour is reported as a refusal with its code,
/// not as a crash and not as a silent success.
#[test]
fn a_refused_device_is_reported_with_its_code() {
    let (stdout, _, code) = info(&["--backend=vulkan"]);
    assert_eq!(code, 0, "an unusable backend is a finding, not a tool failure");
    assert!(stdout.contains("device: vulkan"), "{stdout}");
    assert!(stdout.contains("refused: reconlCreateDevice:"), "{stdout}");
}

/// Options are checked, so a typo cannot quietly measure something else.
#[test]
fn a_mistyped_option_is_refused() {
    let (_, stderr, code) = info(&["--backend=softcpu7"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("is not a backend"), "{stderr}");
    let (_, stderr, code) = info(&["--probe"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("unknown option"), "{stderr}");
}
