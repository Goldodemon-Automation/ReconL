//! `reconl-diff` — golden-image compare (PROMPT §12, item 8).
//!
//! Two modes:
//!
//! * `render <out.png> [--size=N | --width=N --height=N]` — render the reference
//!   frame through the real C ABI (device → swapchain → command list → submit →
//!   present → readback) and write it as an RGBA PNG. This is how goldens are
//!   produced and refreshed, and it exercises the exact surface a host uses.
//! * `compare golden.png [candidate.png] [--tolerance=N]` — compare two PNGs.
//!   With no candidate, the golden is re-rendered through the ABI and compared
//!   against itself on disk. `--tolerance=N` allows per-channel deltas up to
//!   N (with a 1% differing-pixel budget); exact mode is the default.
//!
//! Neither mode is tied to a resolution. The committed golden is 64x64 and a
//! bare `render` still produces that, but a compare takes its size from the
//! golden it was given (a candidate of another size is a hard error rather than
//! a silent crop), the 1% budget scales with that size, and the summary is per
//! channel. That is what makes the tool usable on the 256x256 and 512x512
//! frames the benchmarks produce — `reconl-bench --png=FILE` writes the frame
//! the host read back, at whatever size the run used, and this tool is what
//! compares those.
//!
//! The default is exact because that is what the reference tier produces against
//! itself, byte for byte. The tolerance is for the *cross-tier* comparison, and
//! it is measured rather than chosen (§12 risk 7): `soft-cpu` against `d3d11` on
//! the shadowed golden scene differs on 14 of 4096 pixels, with a worst channel
//! delta of 46. All 14 sit within 4 px of a pixel the shadow changes, and the
//! rest of the frame - the shadow's interior and everything the shadow does not
//! reach - is bit-identical, so the documented setting for that comparison is
//! `--tolerance=48` and it passes inside the 1% budget.
//!
//! That is 0.34% of the frame rather than the 5.37% the same scene measures
//! when the tiers pick their own shadow bias. The presets differ by design (a
//! weaker tier gets more slack, which is why the tier is an input to them at
//! all), and what that buys on screen is a pixel of shadow-edge coverage: 220
//! pixels differing, with the shadow's extent 554 against 559. Both numbers are
//! real; the scene pins the bias through the ABI's own bias fields so that the
//! image is the scene's rather than the tier's, which is what makes it a golden.
//!
//! The scene itself lives in `reconl-host`, which is also what `reconl-bench`
//! renders, so the golden, the timing run and the cross-tier comparison are all
//! the same scene rather than three copies of it.
//!
//! Exit codes (CI-friendly): 0 pass, 1 images differ, 2 hard error.

use reconl::abi;
use reconl_host::device::{Config, Device};
use reconl_host::frame::{Options, Renderer};
use reconl_host::scene::Scene;
use std::process::ExitCode;

/// The reference resolution a bare `render` produces and the committed golden is
/// stored at. Everything else takes its size from what it is asked for or from
/// the golden under comparison.
const GOLDEN_W: u32 = 64;
const GOLDEN_H: u32 = 64;

/// The host configuration the golden is produced with.
///
/// Pinned to the reference tier with downgrades disabled and one worker thread,
/// so the golden depends on the scene and nothing about this machine.
fn reference_config() -> Config {
    Config {
        backend: abi::backend::SOFT_CPU,
        tier: 2,
        allow_downgrade: abi::allow_downgrade::NONE,
        threads: 1,
        target_frame_ms: 1000,
        ..Config::default()
    }
}

/// Renders the reference frame through the ABI and returns RGBA8 pixels.
fn render_reference(width: u32, height: u32) -> Result<Vec<u8>, String> {
    let device = Device::create(&reference_config())?;
    let scene = Scene::reference();
    let renderer = Renderer::new(&device, &scene, Options { width, height, ..Options::default() })?;
    let mut pixels = vec![0u8; renderer.frame_bytes()];
    renderer.frame(&scene, 1, &mut pixels)?;
    Ok(pixels)
}

// -------------------------------------------------------------------- compare

/// What a comparison found. Everything the report needs is measured here, so
/// the report is a rendering of this and nothing recomputed from the images.
#[derive(Default)]
struct Verdict {
    /// Pixels with any nonzero channel delta.
    changed: usize,
    /// Pixels whose worst channel delta exceeds the tolerance.
    exceeded: usize,
    /// Worst delta on any channel, over the whole frame.
    worst: u8,
    /// Worst delta on each channel, in RGBA order.
    worst_channel: [u8; 4],
    /// Pixels differing on each channel, in RGBA order.
    changed_channel: [usize; 4],
    /// The box the differing pixels occupy, as (x0, y0, x1, y1), inclusive.
    bounds: Option<(u32, u32, u32, u32)>,
    /// Where the differing pixels are, up to [`SITES`] of them. A count says how
    /// much moved; this says where, which is what separates a systematic find
    /// from a scattered one at the same count.
    sites: [(u32, u32); SITES],
    sites_len: usize,
}

/// How many differing pixels a report names before it stops counting them.
const SITES: usize = 8;

impl Verdict {
    /// The per-channel divergence: how far each channel moved and how many
    /// pixels moved on it, plus where the differences are. A whole-frame
    /// delta can be one channel at one pixel or four channels at thousands,
    /// and the box is the first thing that says which — a difference confined
    /// to the row a band cut lands on is not the same finding as one spread
    /// across the frame, even at the same count and the same worst delta.
    fn summary(&self) -> String {
        let (x0, y0, x1, y1) = self.bounds.expect("summary of an identical frame");
        let named: Vec<String> = self.sites[..self.sites_len]
            .iter()
            .map(|(x, y)| format!("({x},{y})"))
            .collect();
        let rest = self.changed - self.sites_len;
        let where_ = if rest > 0 {
            format!("{} and {rest} more", named.join(" "))
        } else {
            named.join(" ")
        };
        format!(
            "worst channel delta {} (R {}, G {}, B {}, A {}), \
             differing per channel (R {}, G {}, B {}, A {}), \
             box x {x0}..{x1} y {y0}..{y1}, at {where_}",
            self.worst,
            self.worst_channel[0],
            self.worst_channel[1],
            self.worst_channel[2],
            self.worst_channel[3],
            self.changed_channel[0],
            self.changed_channel[1],
            self.changed_channel[2],
            self.changed_channel[3],
        )
    }

    /// Why a frame that differs was not absorbed by the tolerance.
    fn rejection(&self, tolerance: u8, budget: usize) -> String {
        if self.exceeded > 0 {
            format!(", {} px over tolerance {tolerance}", self.exceeded)
        } else {
            format!(", over the 1% pixel budget of {budget} px")
        }
    }
}

fn compare(a: &reconl_png::Image, b: &reconl_png::Image, tolerance: u8) -> Result<Verdict, String> {
    if a.width != b.width || a.height != b.height {
        return Err(format!(
            "size mismatch: {}x{} vs {}x{} — comparing across sizes is a hard error, not a crop",
            a.width, a.height, b.width, b.height
        ));
    }
    let mut v = Verdict::default();
    for (i, (pa, pb)) in a.pixels.chunks_exact(4).zip(b.pixels.chunks_exact(4)).enumerate() {
        let delta = [
            pa[0].abs_diff(pb[0]),
            pa[1].abs_diff(pb[1]),
            pa[2].abs_diff(pb[2]),
            pa[3].abs_diff(pb[3]),
        ];
        let max = *delta.iter().max().unwrap();
        v.worst = v.worst.max(max);
        if max > 0 {
            v.changed += 1;
            let (x, y) = (i as u32 % a.width, i as u32 / a.width);
            v.bounds = Some(match v.bounds {
                None => (x, y, x, y),
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            });
            if v.sites_len < SITES {
                v.sites[v.sites_len] = (x, y);
                v.sites_len += 1;
            }
        }
        if max > tolerance {
            v.exceeded += 1;
        }
        for (c, d) in delta.iter().enumerate() {
            v.worst_channel[c] = v.worst_channel[c].max(*d);
            if *d > 0 {
                v.changed_channel[c] += 1;
            }
        }
    }
    Ok(v)
}

fn write_png(path: &str, width: u32, height: u32, pixels: &[u8]) -> Result<(), String> {
    let bytes = reconl_png::write_rgba8(width, height, pixels)?;
    std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))
}

fn load_png(path: &str) -> Result<reconl_png::Image, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    reconl_png::read_rgba8(&bytes).map_err(|e| format!("{path}: {e}"))
}

const USAGE: &str = "\
reconl-diff — golden-image compare for ReconL

USAGE:
  reconl-diff render <out.png> [--size=N | --width=N --height=N]
      Render the reference scene through the ReconL C ABI and write it
      as an RGBA PNG. This is how goldens are produced. 64x64 unless a
      size is asked for.

  reconl-diff compare <golden.png> [candidate.png] [--tolerance=N]
      Compare a candidate against the committed golden, at whatever size
      the golden is. With no candidate the reference scene is re-rendered
      through the ABI at the golden's size — the CI path. Exact by
      default: any differing pixel fails. --tolerance=N allows per-channel
      deltas up to N, with a 1% differing-pixel budget.

  A candidate of a different size than the golden is a hard error: this
  compares images, it does not crop one to the other.

  A comparison that is not identical reports the divergence per channel —
  the worst delta on each of R, G, B and A, how many pixels differ on
  each, the box those pixels occupy and, when there are few, where they
  are. A handful of pixels confined to one row is a different finding
  from the same count spread over the frame.

  The scene is the one `reconl-bench` renders: both tools read it from
  `reconl-host`, so a golden and a timed run cannot disagree about it.

Exit codes: 0 pass, 1 images differ, 2 error.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(|s| s.as_str()) {
        Some("render") => do_render(&args[1..]),
        Some("compare") if args.len() >= 2 => parse_compare(&args[1..]),
        _ => Err(USAGE.into()),
    };
    match result {
        Ok(msg) => {
            println!("{msg}");
            ExitCode::from(0)
        }
        Err(e) => {
            eprintln!("{e}");
            // "images differ" is a *finding*, not a crash: exit 1, not 2.
            if e.starts_with("DIFFER") {
                ExitCode::from(1)
            } else {
                ExitCode::from(2)
            }
        }
    }
}

fn parse_dimension(text: &str) -> Result<u32, String> {
    match text.parse::<u32>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(format!("bad dimension `{text}`: want a positive integer")),
    }
}

fn do_render(args: &[String]) -> Result<String, String> {
    let mut out: Option<String> = None;
    let (mut width, mut height) = (GOLDEN_W, GOLDEN_H);
    let mut bad: Option<String> = None;
    for a in args {
        if let Some(v) = a.strip_prefix("--size=") {
            match parse_dimension(v) {
                Ok(n) => {
                    width = n;
                    height = n;
                }
                Err(e) => bad = Some(e),
            }
        } else if let Some(v) = a.strip_prefix("--width=") {
            match parse_dimension(v) {
                Ok(n) => width = n,
                Err(e) => bad = Some(e),
            }
        } else if let Some(v) = a.strip_prefix("--height=") {
            match parse_dimension(v) {
                Ok(n) => height = n,
                Err(e) => bad = Some(e),
            }
        } else if a.starts_with("--") {
            bad = Some(format!("unknown option `{a}`"));
        } else if out.is_some() {
            bad = Some(format!("render writes one file; `{a}` is a second path"));
        } else {
            out = Some(a.clone());
        }
    }
    if let Some(e) = bad {
        return Err(e);
    }
    let out = out.ok_or_else(|| USAGE.to_string())?;
    let pixels = render_reference(width, height)?;
    write_png(&out, width, height, &pixels)?;
    Ok(format!("rendered {width}x{height} to {out}"))
}

fn parse_compare(args: &[String]) -> Result<String, String> {
    let mut tolerance = 0u8;
    let mut paths = Vec::new();
    let mut bad = None;
    for a in args {
        if let Some(t) = a.strip_prefix("--tolerance=") {
            match t.parse::<u8>() {
                Ok(v) => tolerance = v,
                Err(_) => bad = Some(format!("bad tolerance {t}")),
            }
        } else if a == "--tolerance" {
            bad = Some("--tolerance needs a value: use --tolerance=N".into());
        } else if a.starts_with("--") {
            bad = Some(format!("unknown option `{a}`"));
        } else {
            paths.push(a.clone());
        }
    }
    if let Some(e) = bad {
        Err(e)
    } else {
        do_compare(&paths, tolerance)
    }
}

fn do_compare(paths: &[String], tolerance: u8) -> Result<String, String> {
    let golden_path = paths.first().ok_or_else(|| USAGE.to_string())?;
    if paths.len() > 2 {
        return Err(format!(
            "compare takes a golden and at most one candidate, got {} paths",
            paths.len()
        ));
    }
    // The golden is the reference, so its size is the size being compared. With
    // no candidate it is also the size re-rendered, which is what keeps the CI
    // path working at any resolution the scene can be drawn at.
    let golden = load_png(golden_path)?;
    let candidate = match paths.get(1) {
        Some(p) => load_png(p)?,
        None => reconl_png::Image {
            width: golden.width,
            height: golden.height,
            pixels: render_reference(golden.width, golden.height)?,
        },
    };
    let budget = golden.pixels.len() / 4 / 100; // 1% of pixels may differ
    let v = compare(&golden, &candidate, tolerance)?;
    let (w, h) = (golden.width, golden.height);
    if v.changed == 0 {
        if tolerance == 0 {
            Ok(format!("PASS {golden_path} vs candidate: identical ({w}x{h}, {} px)", w * h))
        } else {
            Ok(format!(
                "PASS {golden_path} vs candidate (tolerance {tolerance}): identical ({w}x{h}, {} px)",
                w * h
            ))
        }
    } else if tolerance > 0 && v.exceeded == 0 && v.changed <= budget {
        // Tolerance mode: deltas up to N are absorbed, and at most 1% of
        // pixels may differ at all. Exact mode (the default) has no budget:
        // a golden diff is exact or it is a finding.
        Ok(format!(
            "PASS {golden_path} vs candidate (tolerance {tolerance}): {} of {} px differ ({}%), {}, within the 1% budget",
            v.changed,
            w * h,
            v.changed * 100 / (w * h) as usize,
            v.summary()
        ))
    } else {
        let mode = if tolerance == 0 { "exact" } else { "tolerance" };
        Err(format!(
            "DIFFER {golden_path} vs candidate ({mode}): {} of {} px differ ({}%), {}{}",
            v.changed,
            w * h,
            v.changed * 100 / (w * h) as usize,
            v.summary(),
            v.rejection(tolerance, budget)
        ))
    }
}
