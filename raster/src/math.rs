//! Small, explicit matrix/vector math.
//!
//! Deliberately not `mul_add`, not `simd`, not a dependency: the projection and
//! the view matrix are part of the determinism contract, so they are written out
//! where they can be read and diffed against the HLSL port.

pub type Mat4 = [f32; 16]; // column-major, like HLSL/GLSL constant buffers
pub type Vec3 = [f32; 3];
pub type Vec4 = [f32; 4];

pub const IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

#[inline]
pub fn col(m: &Mat4, index: usize) -> Vec4 {
    [m[index * 4], m[index * 4 + 1], m[index * 4 + 2], m[index * 4 + 3]]
}

#[inline]
pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0f32; 16];
    for c in 0..4 {
        let bc = col(b, c);
        for r in 0..4 {
            out[c * 4 + r] = a[r] * bc[0] + a[4 + r] * bc[1] + a[8 + r] * bc[2] + a[12 + r] * bc[3];
        }
    }
    out
}

/// Transforms a point: `w` is 1.
#[inline]
pub fn mul_point(m: &Mat4, p: Vec3) -> Vec4 {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
        m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15],
    ]
}

/// Transforms a direction: `w` is 0.
#[inline]
pub fn mul_dir(m: &Mat4, d: Vec3) -> Vec3 {
    [
        m[0] * d[0] + m[4] * d[1] + m[8] * d[2],
        m[1] * d[0] + m[5] * d[1] + m[9] * d[2],
        m[2] * d[0] + m[6] * d[1] + m[10] * d[2],
    ]
}

#[inline]
pub fn normalize(v: Vec3) -> Vec3 {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 1.0e-20 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[inline]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn scale(v: Vec3, s: f32) -> Vec3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn length(v: Vec3) -> f32 {
    dot(v, v).sqrt()
}

/// Right-handed look-at: the view direction is `-z` in view space.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = normalize(sub(target, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        s[0], u[0], -f[0], 0.0,
        s[1], u[1], -f[1], 0.0,
        s[2], u[2], -f[2], 0.0,
        -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0,
    ]
}

/// Inverts a view matrix built by [`look_at`] (rotation + translation, no scale).
///
/// Used to walk the camera frustum's corners back out to world space for shadow
/// cascade fitting, which is the only inversion ReconL needs.
pub fn invert_rigid(m: &Mat4) -> Mat4 {
    // Rotation part is orthonormal: its inverse is its transpose.
    let r = [
        m[0], m[4], m[8],
        m[1], m[5], m[9],
        m[2], m[6], m[10],
    ];
    let t = [m[12], m[13], m[14]];
    // The inverse's rows are the original's columns, so its column-major array
    // is the original's rows, laid out the same way.
    [
        r[0], r[1], r[2], 0.0,
        r[3], r[4], r[5], 0.0,
        r[6], r[7], r[8], 0.0,
        -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
        -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
        -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
        1.0,
    ]
}

/// Right-handed perspective projection with a **reversed-Z, 0..1** depth range.
///
/// This is the one depth encoding in ReconL (docs/determinism.md): near maps to
/// 1.0, far maps to 0.0, the depth test is GREATER, and depth clears to 0.0 on
/// every backend. Nothing else in the project is allowed to pick a different one.
pub fn perspective_rh_reversed_z(fov_y_degrees: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y_degrees * 0.5 * core::f32::consts::PI / 180.0).tan();
    // z_ndc = -A - B/z_view with A = n/(f-n), B = n*f/(f-n): 1.0 at the near
    // plane, 0.0 at the far plane, and monotonic in between.
    let nf = 1.0 / (near - far);
    let a = -near * nf;
    let b = -near * far * nf;
    [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, a, -1.0,
        0.0, 0.0, b, 0.0,
    ]
}

/// Right-handed orthographic projection, reversed-Z, `[0,1]` depth.
pub fn ortho_rh_reversed_z(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    let rl = 1.0 / (right - left);
    let tb = 1.0 / (top - bottom);
    // Same reversed-Z convention as the perspective matrix: -near -> 1, -far -> 0.
    let nf = 1.0 / (near - far);
    let a = -nf; // 1/(far-near)
    let b = -far * nf; // far/(far-near)
    [
        2.0 * rl, 0.0, 0.0, 0.0,
        0.0, 2.0 * tb, 0.0, 0.0,
        0.0, 0.0, a, 0.0,
        -(right + left) * rl, -(top + bottom) * tb, b, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversed_z_maps_near_to_one_and_far_to_zero() {
        let p = perspective_rh_reversed_z(60.0, 16.0 / 9.0, 0.1, 100.0);
        let near = mul_point(&p, [0.0, 0.0, -0.1]);
        let far = mul_point(&p, [0.0, 0.0, -100.0]);
        let z_near = near[2] / near[3];
        let z_far = far[2] / far[3];
        assert!((z_near - 1.0).abs() < 1.0e-5, "near -> {z_near}");
        assert!(z_far.abs() < 1.0e-5, "far -> {z_far}");
        // monotonic: closer means larger depth
        let mid = mul_point(&p, [0.0, 0.0, -10.0]);
        let z_mid = mid[2] / mid[3];
        assert!(z_mid > z_far && z_mid < z_near, "mid {z_mid} not between");
    }

    #[test]
    fn ortho_reversed_z_spans_the_range() {
        let p = ortho_rh_reversed_z(-1.0, 1.0, -1.0, 1.0, 0.1, 50.0);
        let a = mul_point(&p, [0.0, 0.0, -0.1]);
        let b = mul_point(&p, [0.0, 0.0, -50.0]);
        assert!((a[2] - 1.0).abs() < 1.0e-6);
        assert!(b[2].abs() < 1.0e-6);
        // corners map to the [-1,1] box in x/y
        let c = mul_point(&p, [1.0, 1.0, -1.0]);
        assert!((c[0] - 1.0).abs() < 1.0e-6 && (c[1] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn rigid_inverse_round_trips_view_space() {
        let v = look_at([3.0, 4.0, 5.0], [1.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
        let inv = invert_rigid(&v);
        let world = [2.0, 1.0, -3.0];
        let view = mul_point(&v, world);
        let back = mul_point(&inv, [view[0], view[1], view[2]]);
        for i in 0..3 {
            assert!((back[i] - world[i]).abs() < 1.0e-4, "component {i}: {} vs {}", back[i], world[i]);
        }
    }

    #[test]
    fn look_at_puts_the_target_down_negative_z() {
        let v = look_at([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let p = mul_point(&v, [0.0, 0.0, 0.0]);
        assert!((p[2] + 5.0).abs() < 1.0e-5, "z = {}", p[2]);
        assert!(p[0].abs() < 1.0e-6 && p[1].abs() < 1.0e-6);
    }

    #[test]
    fn composed_matrix_equals_view_then_projection() {
        let view = look_at([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = perspective_rh_reversed_z(45.0, 1.5, 0.5, 20.0);
        let view_proj = mul(&proj, &view);
        let point = [4.0, 5.0, 6.0];
        let staged = mul_point(&view, point);
        let staged = mul_point(&proj, [staged[0], staged[1], staged[2]]);
        let once = mul_point(&view_proj, point);
        assert_eq!(once[3], staged[3], "w must be identical");
        for i in 0..4 {
            assert!((once[i] - staged[i]).abs() < 1.0e-4, "component {i}: {} vs {}", once[i], staged[i]);
        }
    }
}
