//! The golden-image harness, run by `cargo test --workspace`.
//!
//! Renders the reference scene through the real C ABI and exercises the
//! `reconl-diff` binary's whole contract against the committed golden in
//! `tests/golden/`: exact mode is exact, tolerance absorbs deltas but only
//! within the 1% pixel budget, and corrupt input is a hard error — each with
//! the exit code CI keys on.

use std::path::PathBuf;
use std::process::Command;

const GOLDEN: &str = "tests/golden/soft-cpu-shadow.png";

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_reconl-diff")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn golden_path() -> PathBuf {
    repo_root().join(GOLDEN)
}

/// Runs `reconl-diff <args>` and returns (exit_code, stdout+stderr).
fn run(args: &[&str]) -> (i32, String) {
    let out = Command::new(bin())
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("spawn reconl-diff");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), text)
}

/// Writes a copy of the golden with `n` red channels flipped, using the same
/// codec the tool itself reads and writes.
fn perturbed_copy(n: usize, name: &str) -> PathBuf {
    let img = reconl_png::read_rgba8(&std::fs::read(golden_path()).unwrap()).unwrap();
    let mut px = img.pixels.clone();
    assert!(px.len() >= 4 * (n * 64 + 64), "golden too small for {n} flips");
    for k in 0..n {
        let i = ((32 + k / 64) * img.width as usize + (40 + k % 64)) * 4;
        px[i] ^= 0xFF;
    }
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, reconl_png::write_rgba8(img.width, img.height, &px).unwrap()).unwrap();
    path
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

#[test]
fn ci_path_rerender_is_identical_to_golden() {
    // Determinism claim, end to end: the ABI re-render must be byte-identical
    // to the committed golden.
    let (code, msg) = run(&["compare", GOLDEN]);
    assert_eq!(code, 0, "CI compare failed: {msg}");
    assert!(msg.contains("identical"), "expected identical: {msg}");
}

#[test]
fn exact_mode_fails_on_a_single_pixel() {
    let cand = perturbed_copy(1, "reconl_test_flip1.png");
    let cand = cand.to_str().unwrap();
    let (code, msg) = run(&["compare", GOLDEN, cand]);
    assert_eq!(code, 1, "a flipped pixel must be a finding, not a crash: {msg}");
    assert!(msg.contains("DIFFER"), "{msg}");
}

#[test]
fn tolerance_absorbs_one_pixel_but_not_the_budget() {
    let one = perturbed_copy(1, "reconl_test_tol1.png");
    let fifty = perturbed_copy(50, "reconl_test_tol50.png");
    let one = one.to_str().unwrap();
    let fifty = fifty.to_str().unwrap();

    // 1 flipped px: worst delta 255-worth of flip absorbed, within 1% budget.
    let (code, msg) = run(&["compare", GOLDEN, one, "--tolerance=255"]);
    assert_eq!(code, 0, "1 px within budget must pass: {msg}");

    // 50 flipped px: budget is 1% of 4096 = 40, so this must fail even at
    // tolerance 255 — the budget is over *count*, not just magnitude.
    let (code, msg) = run(&["compare", GOLDEN, fifty, "--tolerance=255"]);
    assert_eq!(code, 1, "50 px over a 40 px budget must fail: {msg}");
    assert!(msg.contains("DIFFER"), "{msg}");
}

#[test]
fn corrupt_input_is_a_hard_error_not_a_diff() {
    let bad = temp_path("reconl_test_corrupt.png");
    std::fs::write(&bad, b"\x89PNG\r\n\x1a\nnot really a png").unwrap();
    let (code, msg) = run(&["compare", bad.to_str().unwrap()]);
    assert_eq!(code, 2, "corrupt input is a tool error: {msg}");
}

/// The golden has to *contain a shadow*, which is the thing it is named for and
/// the thing that can silently stop being true: this scene rendered without a
/// shadow for its whole life, through a fit given a view-projection where a view
/// belonged, and the golden was regenerated from that broken output and
/// committed. Byte-comparison cannot catch that - the bytes agree, with
/// themselves.
///
/// The scene's flat materials - the clear, the lit ground and the caster - are
/// unshaded metal: with no shadow reaching the pixel shader it renders exactly
/// three colours. A shadow adds a ramp between the lit and the ambient level
/// (PCF's partial coverage, and the cascade crossfade), and it is that ramp the
/// count measures. Measured: 3 colours shadowless, 93 in the committed golden.
///
/// The bound is 20 rather than 4, because 4 is what a single stray partial pixel
/// would clear: the point is that the shadow's *edge* is present across the
/// frame, not that something somewhere was darkened.
#[test]
fn the_golden_contains_a_shadow() {
    let img = reconl_png::read_rgba8(&std::fs::read(golden_path()).unwrap()).unwrap();
    let mut colours = std::collections::BTreeSet::new();
    for pixel in img.pixels.chunks_exact(4) {
        colours.insert([pixel[0], pixel[1], pixel[2], pixel[3]]);
    }
    assert!(
        colours.len() > 20,
        "the golden has only {} distinct colours, near the 3 this scene renders with no shadow at \
         all: regenerate it with the shadow actually reaching the pixel shader, over a substantial \
         part of the frame",
        colours.len()
    );
}

#[test]
fn render_writes_a_valid_golden_sized_png() {
    let out = temp_path("reconl_test_render.png");
    let (code, msg) = run(&["render", out.to_str().unwrap()]);
    assert_eq!(code, 0, "render failed: {msg}");
    let img = reconl_png::read_rgba8(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!((img.width, img.height), (64, 64));
    // Not uniformly the clear colour: some pixel must differ from pixel 0.
    assert!(
        img.pixels.chunks_exact(4).any(|p| p != &img.pixels[0..4]),
        "render produced a flat frame"
    );
}
