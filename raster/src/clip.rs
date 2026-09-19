//! Homogeneous clipping.
//!
//! Every triangle is clipped against the six frustum planes before it reaches an
//! edge function, because a GPU does the same thing and the software tier is the
//! reference the GPU tier is diffed against. Clipping in clip space also keeps
//! `w` strictly positive, which is what makes the perspective divide defined.

use crate::math::Mat4;
use crate::ATTR_COUNT;

/// A vertex in clip space, carrying its interpolatable attributes.
#[derive(Clone, Copy, Debug)]
pub struct ClipVertex {
    pub clip: [f32; 4],
    pub attr: [f32; ATTR_COUNT],
}

impl Default for ClipVertex {
    fn default() -> Self {
        Self { clip: [0.0; 4], attr: [0.0; ATTR_COUNT] }
    }
}

/// The six planes, as `a*x + b*y + c*z + d*w >= 0` is inside.
/// Reversed-Z, 0..1 depth range: `z >= 0` is the near plane, `z <= w` the far one.
const PLANES: [[f32; 4]; 6] = [
    [1.0, 0.0, 0.0, 1.0],  // x >= -w
    [-1.0, 0.0, 0.0, 1.0], // x <= w
    [0.0, 1.0, 0.0, 1.0],  // y >= -w
    [0.0, -1.0, 0.0, 1.0], // y <= w
    [0.0, 0.0, 1.0, 0.0],  // z >= 0   (near, reversed-Z)
    [0.0, 0.0, -1.0, 1.0], // z <= w   (far)
];

/// Smallest `w` a vertex may carry and still be divided by.
pub const W_MIN: f32 = 1.0e-6;

fn distance(plane: &[f32; 4], v: &ClipVertex) -> f32 {
    plane[0] * v.clip[0] + plane[1] * v.clip[1] + plane[2] * v.clip[2] + plane[3] * v.clip[3]
}

fn lerp(a: &ClipVertex, b: &ClipVertex, t: f32) -> ClipVertex {
    let mut out = ClipVertex { clip: [0.0; 4], attr: [0.0; ATTR_COUNT] };
    for i in 0..4 {
        out.clip[i] = a.clip[i] + (b.clip[i] - a.clip[i]) * t;
    }
    for i in 0..ATTR_COUNT {
        out.attr[i] = a.attr[i] + (b.attr[i] - a.attr[i]) * t;
    }
    out
}

/// Does this triangle need clipping at all?
///
/// The common case (every vertex comfortably inside) skips the clipper entirely,
/// which is why the fast path is worth a branch.
pub fn needs_clip(v: &[ClipVertex; 3]) -> bool {
    for vert in v.iter() {
        if vert.clip[3] <= W_MIN {
            return true;
        }
        for plane in PLANES.iter() {
            // A hair of slack: vertices exactly on a plane must not force a clip
            // whose floating-point result could differ from the direct path.
            if distance(plane, vert) < -1.0e-5 {
                return true;
            }
        }
    }
    false
}

/// Clips one triangle, appending output triangles to `out`.
///
/// Returns how many triangles were written. `out` must hold at least
/// [`MAX_OUTPUT_TRIANGLES`]; anything else is a caller bug, not a render path.
pub fn clip_triangle(input: &[ClipVertex; 3], out: &mut [[ClipVertex; 3]; MAX_OUTPUT_TRIANGLES]) -> usize {
    let mut poly: [ClipVertex; MAX_POLY] = [input[0]; MAX_POLY];
    let mut len = 3usize;
    poly[0] = input[0];
    poly[1] = input[1];
    poly[2] = input[2];

    let mut scratch: [ClipVertex; MAX_POLY] = [input[0]; MAX_POLY];

    for plane in PLANES.iter() {
        if len < 3 {
            return 0;
        }
        let mut out_len = 0usize;
        let mut prev = poly[len - 1];
        let mut prev_d = distance(plane, &prev);
        for i in 0..len {
            let cur = poly[i];
            let cur_d = distance(plane, &cur);
            let prev_in = prev_d >= 0.0;
            let cur_in = cur_d >= 0.0;
            if prev_in != cur_in {
                let denom = prev_d - cur_d;
                // denom is non-zero because the signs differ.
                let t = prev_d / denom;
                if out_len < MAX_POLY {
                    scratch[out_len] = lerp(&prev, &cur, t);
                    out_len += 1;
                }
            }
            if cur_in && out_len < MAX_POLY {
                scratch[out_len] = cur;
                out_len += 1;
            }
            prev = cur;
            prev_d = cur_d;
        }
        // Copy the scratch polygon back for the next plane.
        let n = out_len.min(MAX_POLY);
        poly[..n].copy_from_slice(&scratch[..n]);
        len = n;
    }

    if len < 3 {
        return 0;
    }
    let mut written = 0usize;
    for i in 1..(len - 1) {
        if written >= MAX_OUTPUT_TRIANGLES {
            break;
        }
        out[written] = [poly[0], poly[i], poly[i + 1]];
        written += 1;
    }
    written
}

/// A triangle clipped against 6 planes has at most 9 vertices; 12 is slack.
pub const MAX_POLY: usize = 12;
/// 9 vertices fan-triangulate into 7 triangles; 8 is slack.
pub const MAX_OUTPUT_TRIANGLES: usize = 8;

/// Transforms one vertex from object space into clip space plus attributes.
#[inline]
pub fn transform_vertex(m: &Mat4, position: [f32; 3], attr: [f32; ATTR_COUNT]) -> ClipVertex {
    let clip = [
        m[0] * position[0] + m[4] * position[1] + m[8] * position[2] + m[12],
        m[1] * position[0] + m[5] * position[1] + m[9] * position[2] + m[13],
        m[2] * position[0] + m[6] * position[1] + m[10] * position[2] + m[14],
        m[3] * position[0] + m[7] * position[1] + m[11] * position[2] + m[15],
    ];
    ClipVertex { clip, attr }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32, w: f32) -> ClipVertex {
        ClipVertex { clip: [x, y, z, w], attr: [x; ATTR_COUNT] }
    }

    #[test]
    fn inside_triangle_does_not_ask_for_clipping() {
        let t = [v(0.0, 0.0, 0.5, 1.0), v(1.0, 0.0, 0.5, 1.0), v(0.0, 1.0, 0.5, 1.0)];
        assert!(!needs_clip(&t));
    }

    #[test]
    fn behind_the_near_plane_needs_clipping() {
        let t = [v(0.0, 0.0, -0.1, 1.0), v(1.0, 0.0, 0.5, 1.0), v(0.0, 1.0, 0.5, 1.0)];
        assert!(needs_clip(&t));
        let mut out = [[v(0.0, 0.0, 0.0, 1.0); 3]; MAX_OUTPUT_TRIANGLES];
        let n = clip_triangle(&t, &mut out);
        assert!(n >= 1, "near-plane clip produced nothing");
        for tri in out.iter().take(n) {
            for vert in tri.iter() {
                assert!(vert.clip[2] >= -1.0e-4, "z = {}", vert.clip[2]);
                assert!(vert.clip[3] > 0.0);
            }
        }
    }

    #[test]
    fn fully_behind_the_camera_produces_nothing() {
        let t = [v(0.0, 0.0, -1.0, -1.0), v(1.0, 0.0, -1.0, -1.0), v(0.0, 1.0, -1.0, -1.0)];
        let mut out = [[v(0.0, 0.0, 0.0, 1.0); 3]; MAX_OUTPUT_TRIANGLES];
        assert_eq!(clip_triangle(&t, &mut out), 0);
    }

    #[test]
    fn straddling_the_near_plane_yields_a_quad_inside_the_frustum() {
        // One vertex behind the near plane, two comfortably inside.
        let t = [
            v(0.0, 0.0, -1.0, 1.0),
            v(0.9, 0.9, 0.5, 1.0),
            v(-0.9, 0.9, 0.5, 1.0),
        ];
        let mut out = [[v(0.0, 0.0, 0.0, 1.0); 3]; MAX_OUTPUT_TRIANGLES];
        let n = clip_triangle(&t, &mut out);
        assert_eq!(n, 2, "a triangle cut once is a quad: two triangles");
        for tri in out.iter().take(n) {
            for vert in tri.iter() {
                for plane in PLANES.iter() {
                    assert!(distance(plane, vert) > -1.0e-4, "vertex escaped a plane: {vert:?}");
                }
            }
        }
    }

    #[test]
    fn whatever_survives_clipping_is_inside_every_plane() {
        // A triangle whose projection misses the frustum entirely may clip down
        // to nothing; if anything survives, it must be inside all six planes.
        let t = [
            v(-1000.0, -1000.0, -1000.0, 1.0),
            v(1000.0, -900.0, 0.5, 1.0),
            v(-900.0, 1000.0, 0.5, 1.0),
        ];
        let mut out = [[v(0.0, 0.0, 0.0, 1.0); 3]; MAX_OUTPUT_TRIANGLES];
        let n = clip_triangle(&t, &mut out);
        for tri in out.iter().take(n) {
            for vert in tri.iter() {
                assert!(vert.clip[3] > 0.0, "non-positive w escaped the clipper");
                for plane in PLANES.iter() {
                    assert!(distance(plane, vert) > -1.0e-3, "vertex escaped a plane: {vert:?}");
                }
            }
        }
    }

    #[test]
    fn attributes_are_interpolated_exactly_at_the_near_plane_cut() {
        // z goes -1 -> +1 with attr 0 -> 2, so the cut at z = 0 has t = 0.5 and
        // the interpolated attribute must be exactly 1.0 (the other two output
        // vertices keep attr = 2.0).
        let mut a = v(0.0, 0.0, -1.0, 1.0);
        a.attr = [0.0; ATTR_COUNT];
        let mut b = v(0.0, 1.0, 1.0, 1.0);
        b.attr = [2.0; ATTR_COUNT];
        let mut c = v(1.0, 0.0, 1.0, 1.0);
        c.attr = [2.0; ATTR_COUNT];
        let mut out = [[v(0.0, 0.0, 0.0, 1.0); 3]; MAX_OUTPUT_TRIANGLES];
        let n = clip_triangle(&[a, b, c], &mut out);
        assert_eq!(n, 2, "one plane cut turns the triangle into a quad");
        let mut min_attr = f32::MAX;
        for vert in out[0].iter() {
            min_attr = min_attr.min(vert.attr[0]);
        }
        assert!((min_attr - 1.0).abs() < 1.0e-5, "clipped attribute is {min_attr}");
    }
}
