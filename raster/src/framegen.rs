//! Frame generation: the frames *between* real ones, reprojected rather than
//! re-rendered.
//!
//! A frame generator only earns its cost if a generated frame is much cheaper
//! than a rendered one. ReconL can afford the opposite trade - it renders the
//! reference scene deterministically on demand - so this is deliberately not a
//! second renderer: it is one reprojection of the frame the host was last
//! handed, with no shaders, no shadow pass and no geometry.
//!
//! ## What it does, exactly
//!
//! For every pixel of the newest real frame the generator knows two things the
//! hardware would tell a motion-vector buffer: the pixel's depth, and the camera
//! the frame was drawn with. Depth plus the source camera unprojects the pixel
//! into view space; the previous frame's camera projects it back to the screen;
//! the difference is where that pixel's *content* was one frame ago. Sampling
//! the source image offset by `ahead` times that difference produces the image
//! `ahead` frame-intervals past the newest real frame.
//!
//! Both projections are fused into one matrix per generated frame
//! ([`reprojection`]), so the per-pixel work is one 4x4 multiply and one divide -
//! not two unproject/reproject chains.
//!
//! ## What it is not
//!
//! * **Not a renderer.** Nothing is re-shaded: a pixel's colour is the source
//!   frame's colour, moved. Geometry that appears from behind the camera
//!   (disocclusion) is stretched, not revealed, and anything that moved
//!   *within* the frame is treated as if it moved with the camera.
//! * **Not interpolated.** Interpolation would mean holding a rendered frame
//!   back until the next one exists, which adds a frame of latency to every
//!   host that enables it. This extrapolates past the newest frame from the
//!   camera's own motion, so a host's input latency is unchanged.
//! * **Not free of error growth.** The approximation is first order in `ahead`:
//!   sampling is a backward warp that assumes the screen-space velocity at the
//!   output pixel applies across the whole offset. Small `ahead` is close to
//!   exact; a large one is a smear. That is the trade the host's schedule makes
//!   (see the ABI's `reconlPresentGenerated`), and it is why `ahead` is capped
//!   at one frame interval.

use crate::math::{self, Mat4};
use reconl_core::err;
use reconl_core::error::{Code, Error, Result};

/// The camera a frame was drawn with: a rigid view and the reversed-Z
/// perspective projection its pixel grid was mapped through.
///
/// Held as parameters *and* matrices because a reprojection needs both the
/// forward map (for the pixel it starts at) and the inverse (for the view-space
/// point it recovers), and building either from anything but these parameters is
/// how the two drift apart.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// World to camera space, as `ReconLCamera.view` declares it.
    pub view: Mat4,
    pub proj: Mat4,
    /// The inverse of `proj`, built from the same parameters.
    pub unproj: Mat4,
    /// The frame's aspect ratio, which `proj` was built with.
    pub aspect: f32,
}

impl Camera {
    /// The camera an ABI frame declares: a rigid view plus the vertical field of
    /// view and frustum distances the host's projection used.
    pub fn new(view: Mat4, fov_y_deg: f32, aspect: f32, near: f32, far: f32) -> Camera {
        Camera {
            view,
            proj: math::perspective_rh_reversed_z(fov_y_deg, aspect, near, far),
            unproj: math::perspective_rh_reversed_z_inverse(fov_y_deg, aspect, near, far),
            aspect,
        }
    }

    /// Whether two cameras draw the same image of a still scene.
    pub fn same_pose(&self, other: &Camera) -> bool {
        self.view == other.view && self.proj == other.proj
    }
}

/// One real frame, as the layer that presented it retained it.
///
/// The colour is the image the host was handed - RGBA8, tightly packed, at the
/// size the tier presented - so a generated frame is warped from exactly the
/// pixels a host saw, not from a second copy that could drift from them.
pub struct History<'a> {
    pub color: &'a [u8],
    pub depth: &'a [f32],
    pub camera: Camera,
    pub width: u32,
    pub height: u32,
}

/// The matrix that takes a pixel of `cur` to where its content was in `prev`.
///
/// `P_prev * V_prev * V_cur^-1 * P_cur^-1`, applied to the homogeneous clip point
/// `(ndc_x, ndc_y, depth, 1)`. The composition is legal because every factor is
/// projective and the intervening view matrices are affine: the two perspective
/// divides happen once, at the end, instead of once per chain.
pub fn reprojection(cur: &Camera, prev: &Camera) -> Mat4 {
    let view = math::mul(&prev.view, &math::invert_rigid(&cur.view));
    math::mul(&math::mul(&prev.proj, &view), &cur.unproj)
}

/// The look-ahead's own rule, in one place: it is a fraction of one frame
/// interval, so it is finite and in `(0, 1]`.
///
/// Both the ABI's argument gate and [`generate`] ask this, so the interval a
/// host is held to is the same one wherever it is refused.
pub fn check_ahead(ahead: f32) -> Result<()> {
    if !ahead.is_finite() || ahead <= 0.0 || ahead > 1.0 {
        return err!(
            Code::InvalidArgument,
            "the look-ahead is {ahead}; it must be in (0, 1] frame intervals"
        );
    }
    Ok(())
}

/// Warps `source` (the newest real frame) forward by `ahead` frame intervals and
/// writes `source.width * source.height * 4` bytes of RGBA8 into `out`.
///
/// `previous` is the camera of the frame *before* `source`; the pair is what
/// gives the motion. `ahead` is in `(0, 1]`: one frame interval is the distance
/// between the two cameras the host declared.
pub fn generate(source: &History<'_>, previous: &Camera, ahead: f32, out: &mut [u8]) -> Result<()> {
    let (w, h) = (source.width, source.height);
    if w == 0 || h == 0 {
        return Err(Error::new(Code::InvalidArgument, "no frame to generate from"));
    }
    let pixels = (w as usize) * (h as usize);
    if source.color.len() < pixels * 4 {
        return Err(Error::new(
            Code::InvalidArgument,
            "the retained frame is smaller than its own size",
        ));
    }
    if source.depth.len() < pixels {
        return Err(Error::new(
            Code::NotSupported,
            "this frame has no depth to reproject; frame generation needs a depth buffer",
        ));
    }
    if out.len() < pixels * 4 {
        return Err(Error::new(
            Code::InvalidArgument,
            "the present buffer is too small for the generated frame",
        ));
    }
    check_ahead(ahead)?;

    // A still camera means every pixel's content is already where it belongs:
    // the frame itself, to the byte. Taking the copy here rather than hoping the
    // resample lands on the pixel centre is what makes "a generated frame of a
    // still scene is the frame" an invariant a test can pin - and a menu, a
    // paused game and the reference scene are all still scenes.
    if source.camera.same_pose(previous) {
        out[..pixels * 4].copy_from_slice(&source.color[..pixels * 4]);
        return Ok(());
    }

    let m = reprojection(&source.camera, previous);
    let color = source.color;
    let depth = source.depth;
    let (fw, fh) = (w as f32, h as f32);
    let mut pixel = [0u8; 4];

    for y in 0..h {
        let yc = y as f32 + 0.5;
        let ny = 1.0 - (yc * 2.0 / fh);
        for x in 0..w {
            let at = ((y as usize) * (w as usize) + x as usize) * 4;
            let d = depth[(y as usize) * (w as usize) + x as usize];
            // Nothing was drawn here (the depth cleared to 0 and reversed-Z
            // puts the clear at the far plane): background does not move, so the
            // pixel is the frame's own.
            if !(d > 0.0) || !d.is_finite() {
                out[at..at + 4].copy_from_slice(&color[at..at + 4]);
                continue;
            }
            let xc = x as f32 + 0.5;
            let nx = xc * 2.0 / fw - 1.0;
            let clip = math::mul_point(&m, [nx, ny, d]);
            let cw = clip[3];
            // `cw <= 0` puts the point behind the previous camera: there is no
            // screen position to sample, and the frame's own pixel is the least
            // wrong answer.
            if !(cw > 1.0e-9) {
                out[at..at + 4].copy_from_slice(&color[at..at + 4]);
                continue;
            }
            let px = (clip[0] / cw * 0.5 + 0.5) * fw;
            let py = (1.0 - (clip[1] / cw * 0.5 + 0.5)) * fh;
            // One backward step: where this output pixel's content sits in the
            // source frame, given the motion the same pixel showed over the
            // interval between the two real frames.
            //
            // `(px, py)` is where the content now at this pixel was one frame
            // ago, so `(xc - px, yc - py)` is its one-interval screen velocity -
            // it travelled that far to get here, and by the look-ahead time it
            // has travelled `ahead` times further. The source it came from is
            // therefore *behind* this pixel along that velocity, which is the
            // minus: content flowing right is sampled from the left.
            let sx = xc - ahead * (xc - px);
            let sy = yc - ahead * (yc - py);
            sample_bilinear(color, w, h, sx, sy, &mut pixel);
            out[at..at + 4].copy_from_slice(&pixel);
        }
    }
    Ok(())
}

/// Bilinear sample at a pixel-centre coordinate, clamped to the image: a
/// reprojection that reaches past the frame samples its edge rather than
/// inventing black or wrapping around to the other side.
#[inline]
fn sample_bilinear(color: &[u8], w: u32, h: u32, x: f32, y: f32, out: &mut [u8; 4]) {
    let fx = (x - 0.5).clamp(0.0, (w - 1) as f32);
    let fy = (y - 0.5).clamp(0.0, (h - 1) as f32);
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let x0 = x0 as usize;
    let y0 = y0 as usize;
    let x1 = (x0 + 1).min(w as usize - 1);
    let y1 = (y0 + 1).min(h as usize - 1);
    let row = w as usize * 4;
    let c00 = &color[y0 * row + x0 * 4..][..4];
    let c10 = &color[y0 * row + x1 * 4..][..4];
    let c01 = &color[y1 * row + x0 * 4..][..4];
    let c11 = &color[y1 * row + x1 * 4..][..4];
    for c in 0..4 {
        let top = c00[c] as f32 + (c10[c] as f32 - c00[c] as f32) * tx;
        let bottom = c01[c] as f32 + (c11[c] as f32 - c01[c] as f32) * tx;
        let value = top + (bottom - top) * ty;
        out[c] = value.round().clamp(0.0, 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera looking down `-z` from the origin, with the project's own
    /// 60-degree frustum.
    fn camera(view: Mat4) -> Camera {
        Camera::new(view, 60.0, 1.0, 0.1, 100.0)
    }

    /// A `w`x`h` frame on a plane at view depth `z`, as depth and colour: the
    /// colour is a one-pixel-wide white column at `column`, which is what makes
    /// the warp's displacement measurable rather than a matter of opinion.
    fn frame(w: u32, h: u32, z: f32, column: u32, view: Mat4) -> (Vec<u8>, Vec<f32>) {
        let cam = camera(view);
        let mut color = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                if x == column {
                    let at = ((y * w + x) * 4) as usize;
                    color[at..at + 4].copy_from_slice(&[255, 255, 255, 255]);
                }
            }
        }
        // The depth of a point `z` in front of the camera, unprojected through
        // the frame's own projection - so the test's depth and the generator's
        // unprojection are the same map, which is the point of using the
        // projection rather than a literal.
        let _ = cam;
        let clip = math::mul_point(&camera(view).proj, [0.0, 0.0, z]);
        let d = clip[2] / clip[3];
        (color, vec![d; (w * h) as usize])
    }

    /// With the camera still, a generated frame is the frame - to the byte. A
    /// menu, a paused game and the reference scene are all still scenes, and a
    /// generator that resamples them is a generator that softens them.
    #[test]
    fn a_still_camera_generates_the_frame_itself() {
        let view = math::look_at([0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let (color, depth) = frame(16, 16, -4.0, 7, view);
        let cam = camera(view);
        let history = History { color: &color, depth: &depth, camera: cam, width: 16, height: 16 };
        let mut out = vec![0u8; color.len()];
        generate(&history, &cam, 1.0, &mut out).unwrap();
        assert_eq!(out, color, "a still camera must generate the frame itself");
    }

    /// The motion model, measured: a camera that slides sideways by `shift`
    /// moves a plane at depth `z` by an amount the projection predicts, and the
    /// generated image must show that shift - in the right direction, by the
    /// right number of pixels, or the whole feature is a blur.
    #[test]
    fn a_sideways_camera_moves_the_image_by_the_projection_prediction() {
        let w = 64u32;
        let h = 64u32;
        let z = -4.0f32;
        let shift = 1.0f32;
        let prev_view = math::look_at([0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // The camera moves +x, so screen content moves -x.
        let cur_view = math::look_at([shift, 0.0, 4.0], [shift, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let (color, depth) = frame(w, h, z, 40, cur_view);
        let cur = camera(cur_view);
        let prev = camera(prev_view);
        let history = History { color: &color, depth: &depth, camera: cur, width: w, height: h };

        // Where the plane's centre lands in each camera, in pixels.
        let project = |cam: &Camera, p: [f32; 3]| -> f32 {
            let view = math::mul_point(&cam.view, p);
            let clip = math::mul_point(&cam.proj, [view[0], view[1], view[2]]);
            (clip[0] / clip[3] * 0.5 + 0.5) * w as f32
        };
        // The world point 4 units in front of the *current* camera, at x = 0.
        let world = [0.0, 0.0, 0.0];
        let cur_x = project(&cur, world);
        let prev_x = project(&prev, world);
        let expected_shift = (cur_x - prev_x) * 1.0; // ahead = 1

        let mut out = vec![0u8; color.len()];
        generate(&history, &prev, 1.0, &mut out).unwrap();

        let column_of_brightest = |img: &[u8]| -> f32 {
            let mut best = (0.0f32, 0.0f32);
            for x in 0..w {
                let mut sum = 0.0f32;
                for y in 0..h {
                    sum += img[((y * w + x) * 4) as usize] as f32;
                }
                if sum > best.1 {
                    best = (x as f32 + 0.5, sum);
                }
            }
            best.0
        };
        let moved = column_of_brightest(&out) - column_of_brightest(&color);
        assert!(
            (moved - expected_shift).abs() < 1.5,
            "a {shift}-unit camera move should shift the image by {expected_shift:.2} px, it moved {moved:.2}"
        );
        // The direction: the camera went right, the content went left.
        assert!(moved < 0.0, "content moved {moved} px for a rightward camera");
    }

    /// The same inputs generate the same bytes, every time: a deterministic
    /// renderer cannot have a nondeterministic generator on its present path.
    #[test]
    fn generation_is_deterministic_and_bounded() {
        let w = 32u32;
        let h = 32u32;
        let prev_view = math::look_at([0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let cur_view = math::look_at([0.4, 0.1, 3.9], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let (color, depth) = frame(w, h, -4.0, 20, cur_view);
        let cur = camera(cur_view);
        let prev = camera(prev_view);
        let history = History { color: &color, depth: &depth, camera: cur, width: w, height: h };

        let mut a = vec![0u8; color.len()];
        let mut b = vec![0u8; color.len()];
        generate(&history, &prev, 0.5, &mut a).unwrap();
        generate(&history, &prev, 0.5, &mut b).unwrap();
        assert_eq!(a, b, "two generated frames of one pair differ");

        // Refusals, not panics: a look-ahead outside the documented interval, a
        // buffer that cannot hold the frame, and a depth buffer that is not one.
        let mut out = vec![0u8; color.len()];
        for ahead in [0.0f32, -1.0, 1.5, f32::NAN, f32::INFINITY] {
            assert_eq!(
                generate(&history, &prev, ahead, &mut out).unwrap_err().code,
                Code::InvalidArgument,
                "ahead {ahead} was accepted"
            );
        }
        let mut small = vec![0u8; color.len() - 4];
        assert_eq!(
            generate(&history, &prev, 0.5, &mut small).unwrap_err().code,
            Code::InvalidArgument
        );
        let short = History { color: &color, depth: &depth[..depth.len() - 1], camera: cur, width: w, height: h };
        assert_eq!(
            generate(&short, &prev, 0.5, &mut out).unwrap_err().code,
            Code::NotSupported
        );
    }
}
