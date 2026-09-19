//! The matrices a host has to hand the ABI, written in the engine's convention.
//!
//! The ABI carries transforms, not cameras: `PUSH_CONSTANT_VIEW_PROJ` slot 0 is
//! the vertex transform and `ReconLCamera.view` is the same view on its own,
//! because cascade fitting needs an un-multiplied view. A host that gets the
//! convention wrong renders a plausible-looking wrong frame, so the two
//! conventions that matter are named here and pinned by the reference scene's
//! golden.
//!
//! * **Right-handed, reversed depth.** Near maps to 1.0 and far to 0.0, and
//!   depth compares `GREATER`. This is what lets a host clear to 0.0 and use a
//!   float depth buffer with its precision where the geometry is.
//! * **Column-major, `m * v`.** The same layout a constant buffer expects, and
//!   the same one `reconl-raster` uses, so the reference tier is the definition
//!   of what a hardware backend must reproduce.
//!
//! The arithmetic below is deliberately the literal series of operations the
//! reference scene has always used, not an algebraically equal rearrangement:
//! `-near * nf` and `near / (far - near)` are the same number on paper and can
//! differ by an ulp in `f32`, and an ulp of projection is a pixel of a golden.

/// The 4x4 identity, for a draw with no model transform.
pub const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// `a * b` for column-major 4x4 matrices.
pub fn mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    out
}

/// A right-handed, reversed-Z perspective projection, column-major.
///
/// `aspect` is width/height. The frame is usually square, in which case this is
/// exactly the reference scene's projection.
pub fn perspective_rh_reversed_z(fov_y_deg: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y_deg * 0.5 * core::f32::consts::PI / 180.0).tan();
    let nf = 1.0 / (near - far);
    [
        f / aspect, 0.0, 0.0, 0.0, //
        0.0, f, 0.0, 0.0, //
        0.0, 0.0, -near * nf, -1.0, //
        0.0, 0.0, -near * far * nf, 0.0,
    ]
}

/// A right-handed look-at view matrix, column-major.
///
/// The reference scene writes its own view out in closed form instead (see
/// `scene::CameraSpec::view`), because its camera is pitched with no yaw and the
/// closed form is worth reading. This is for hosts with a camera that turns.
pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if l == 0.0 {
            [0.0, 0.0, 0.0]
        } else {
            [v[0] / l, v[1] / l, v[2] / l]
        }
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let back = norm(sub(eye, target));
    let right = norm(cross(up, back));
    let up_axis = cross(back, right);
    [
        right[0],
        up_axis[0],
        back[0],
        0.0,
        right[1],
        up_axis[1],
        back[1],
        0.0,
        right[2],
        up_axis[2],
        back[2],
        0.0,
        -dot(right, eye),
        -dot(up_axis, eye),
        -dot(back, eye),
        1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reversed depth: the near plane lands on 1.0 and the far plane on 0.0, and
    /// `w` is the view distance - the property every depth compare relies on.
    #[test]
    fn near_is_one_and_far_is_zero() {
        let (near, far) = (0.5f32, 100.0f32);
        let m = perspective_rh_reversed_z(45.0, 1.0, near, far);
        let project = |z_view: f32| {
            let clip_z = m[10] * z_view + m[14];
            let w = m[11] * z_view;
            clip_z / w
        };
        assert!((project(-near) - 1.0).abs() < 1e-6, "near -> {}", project(-near));
        assert!(project(-far).abs() < 1e-6, "far -> {}", project(-far));
        // Monotonic: nearer geometry gets the larger depth, which is what
        // COMPARE_GREATER means.
        assert!(project(-1.0) > project(-10.0));
    }

    /// The closed-form view the reference scene uses and this general one are
    /// the same matrix for that scene's camera, to within the last couple of
    /// ulps.
    ///
    /// Not bit-identical, and that is the point of keeping the closed form in
    /// `scene`: `169/L + 100/L` and `sqrt(269)` agree on paper and differ by an
    /// ulp in `f32`, and an ulp of view translation moves a shadow's edge. The
    /// tolerance here is 1e-5 because that is the measured gap; a general
    /// look-at is for cameras that turn, not for the golden.
    #[test]
    fn look_at_agrees_with_the_reference_closed_form() {
        let (height, distance) = (13.0f32, 10.0f32);
        let len = (height * height + distance * distance).sqrt();
        let closed = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, distance / len, height / len, 0.0, //
            0.0, -height / len, distance / len, 0.0, //
            0.0, 0.0, -len, 1.0,
        ];
        let general = look_at([0.0, height, distance], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        for i in 0..16 {
            assert!(
                (closed[i] - general[i]).abs() <= 1e-5,
                "element {i}: closed {} vs look_at {}",
                closed[i],
                general[i]
            );
        }
    }

    #[test]
    fn aspect_scales_only_the_horizontal_term() {
        let wide = perspective_rh_reversed_z(60.0, 2.0, 1.0, 50.0);
        let square = perspective_rh_reversed_z(60.0, 1.0, 1.0, 50.0);
        assert!((wide[0] - square[0] / 2.0).abs() < 1e-9);
        assert_eq!(wide[5], square[5]);
        assert_eq!(mul(&IDENTITY, &wide), wide);
    }
}
