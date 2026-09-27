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

/// Writes `img` with the pixel at (x, y) flipped on red, using the tool's codec.
fn flip(img: &reconl_png::Image, x: u32, y: u32, name: &str) -> PathBuf {
    let mut px = img.pixels.clone();
    let i = (y as usize * img.width as usize + x as usize) * 4;
    px[i] ^= 0xFF;
    let path = temp_path(name);
    std::fs::write(&path, reconl_png::write_rgba8(img.width, img.height, &px).unwrap()).unwrap();
    path
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

/// The compare tool is not a 64x64 tool: the project benchmarks at 256x256 and
/// 512x512, and before this test existed the only way to look at those frames
/// was a throwaway script, because `compare` refused any size but the golden's.
///
/// Every path of the contract is exercised at 512x512 here — the size is taken
/// from the golden, the re-render happens at that size, the budget scales with
/// it (1% of 262144 is 2621, not 40), and a find is reported per channel.
#[test]
fn compare_works_at_the_size_the_benchmarks_render() {
    for name in ["reconl_test_s512.png", "reconl_test_s512_twin.png"] {
        let (code, msg) = run(&["render", temp_path(name).to_str().unwrap(), "--size=512"]);
        assert_eq!(code, 0, "render --size=512 failed: {msg}");
    }
    let a = std::fs::read(temp_path("reconl_test_s512.png")).unwrap();
    let img = reconl_png::read_rgba8(&a).unwrap();
    assert_eq!((img.width, img.height), (512, 512), "render --size=512 wrote another size");

    // The CI path at another resolution: no candidate, so the reference is
    // re-rendered at the golden's own size and must land on it exactly.
    let a = temp_path("reconl_test_s512.png");
    let b = temp_path("reconl_test_s512_twin.png");
    let (code, msg) = run(&["compare", a.to_str().unwrap()]);
    assert_eq!(code, 0, "512x512 re-render must be identical: {msg}");
    assert!(msg.contains("identical (512x512"), "{msg}");

    // A single flipped pixel is a finding at this size too, and the report says
    // which channel moved, how far, and where.
    let bad = flip(&img, 40, 32, "reconl_test_s512_flip.png");
    let (code, msg) = run(&["compare", a.to_str().unwrap(), bad.to_str().unwrap()]);
    assert_eq!(code, 1, "one flipped 512x512 pixel must be a finding: {msg}");
    assert!(msg.contains("DIFFER"), "{msg}");
    assert!(msg.contains("1 of 262144 px differ"), "{msg}");
    assert!(
        msg.contains("worst channel delta ") && !msg.contains("worst channel delta 0 ("),
        "the per-channel worst delta has to be reported and nonzero: {msg}"
    );
    assert!(msg.contains("differing per channel (R 1, G 0, B 0, A 0)"), "{msg}");
    assert!(msg.contains("box x 40..40 y 32..32"), "the find's location is missing: {msg}");

    // The same flip is absorbed once the tolerance admits it, and the budget is
    // 1% of the frame's pixels rather than of the golden's.
    let (code, msg) = run(&["compare", a.to_str().unwrap(), bad.to_str().unwrap(), "--tolerance=255"]);
    assert_eq!(code, 0, "1 px within a 2621 px budget must pass: {msg}");
    assert!(msg.contains("within the 1% budget"), "{msg}");

    // Two sizes are not compared by cropping one to the other.
    let (code, msg) = run(&["compare", GOLDEN, b.to_str().unwrap()]);
    assert_eq!(code, 2, "a size mismatch is a tool error: {msg}");
    assert!(msg.contains("size mismatch"), "{msg}");
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
