//! Fixed-point screen-space setup and the fill rule.
//!
//! Two of the documented determinism rules live here:
//!
//! * **8-bit subpixel precision.** Positions are snapped to 1/256 of a pixel
//!   before any edge function is evaluated, so vertex position noise below one
//!   subpixel produces the same coverage on every backend and every run.
//! * **Top-left fill rule.** A pixel centre exactly on a shared edge belongs to
//!   exactly one of the two triangles that share it: never doubled, never
//!   dropped. Tested in `tests` below
//!   (`top_left_rule_covers_a_shared_edge_exactly_once`).

/// Subpixel bits: 8, i.e. a fixed-point unit of 1/256 px.
pub const SUBPIXEL_BITS: u32 = 8;
pub const SUBPIXEL_UNITS: i32 = 1 << SUBPIXEL_BITS;

/// Snaps a screen-space coordinate to the subpixel grid.
///
/// `round` is half-away-from-zero, which is deterministic and, unlike
/// `as i32` truncation, symmetric about the origin.
#[inline]
pub fn to_fixed(v: f32) -> i32 {
    let scaled = v * (SUBPIXEL_UNITS as f32);
    if !scaled.is_finite() {
        return if scaled > 0.0 { i32::MAX } else { i32::MIN };
    }
    scaled.round() as i32
}

/// One edge of a triangle in the form `E(x, y) = a*x + b*y + c`.
///
/// For edge `v0 -> v1` with `dx = x1 - x0`, `dy = y1 - y0`:
/// `E(x,y) = dy*(x - x0) - dx*(y - y0)`, so `a = dy`, `b = -dx`, `c = dx*y0 - dy*x0`.
/// With the screen convention ReconL uses (y down, front faces wound so that
/// `v0 -> v1 -> v2` turns clockwise on screen) the interior is `E > 0`.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub a: i64,
    pub b: i64,
    pub c: i64,
    /// 1 when the edge is a top or left edge, else 0. Added to `E` so that the
    /// inclusive test is a single `> 0` comparison.
    pub bias: i64,
}

#[inline]
pub fn edge_from_fixed(x0: i32, y0: i32, x1: i32, y1: i32) -> Edge {
    let dx = (x1 as i64) - (x0 as i64);
    let dy = (y1 as i64) - (y0 as i64);
    Edge {
        a: dy,
        b: -dx,
        c: dx * (y0 as i64) - dy * (x0 as i64),
        bias: top_left_bias(dx, dy),
    }
}

/// Top or left edge (D3D/GU convention, y down): horizontal edges that run
/// toward -x, and edges that run toward +y.
#[inline]
pub fn top_left_is_inclusive(dx: i64, dy: i64) -> bool {
    (dy == 0 && dx < 0) || dy > 0
}

#[inline]
fn top_left_bias(dx: i64, dy: i64) -> i64 {
    if top_left_is_inclusive(dx, dy) {
        1
    } else {
        0
    }
}

#[inline]
pub fn eval(edge: &Edge, x: i32, y: i32) -> i64 {
    edge.a * (x as i64) + edge.b * (y as i64) + edge.c + edge.bias
}

/// A triangle ready to rasterise: snapped vertices plus the three edge functions.
#[derive(Clone, Copy, Debug)]
pub struct Setup {
    pub x: [i32; 3],
    pub y: [i32; 3],
    pub edges: [Edge; 3],
    /// Twice the signed area in subpixel units; positive for front faces.
    pub area2: i64,
}

impl Setup {
    /// Builds the setup for `v0 -> v1 -> v2`. Coordinates are in pixels, with
    /// (0,0) at the centre of the top-left pixel.
    pub fn new(v: [[f32; 2]; 3]) -> Option<Setup> {
        let x = [to_fixed(v[0][0]), to_fixed(v[1][0]), to_fixed(v[2][0])];
        let y = [to_fixed(v[0][1]), to_fixed(v[1][1]), to_fixed(v[2][1])];
        let edges = [
            edge_from_fixed(x[0], y[0], x[1], y[1]),
            edge_from_fixed(x[1], y[1], x[2], y[2]),
            edge_from_fixed(x[2], y[2], x[0], y[0]),
        ];
        // area2 = E0(v2) with the bias removed, so it is a true area.
        let area2 = edges[0].a * (x[2] as i64) + edges[0].b * (y[2] as i64) + edges[0].c;
        if area2 == 0 {
            return None; // degenerate: no pixels, and no divide by zero later
        }
        Some(Setup { x, y, edges, area2 })
    }

    #[inline]
    pub fn front_facing(&self) -> bool {
        self.area2 > 0
    }

    /// Axis-aligned bounds in whole pixels, clamped to the target.
    pub fn bounds(&self, width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
        let min_x = self.x.iter().copied().min()? >> SUBPIXEL_BITS;
        let max_x = (self.x.iter().copied().max()? + SUBPIXEL_UNITS - 1) >> SUBPIXEL_BITS;
        let min_y = self.y.iter().copied().min()? >> SUBPIXEL_BITS;
        let max_y = (self.y.iter().copied().max()? + SUBPIXEL_UNITS - 1) >> SUBPIXEL_BITS;
        let x0 = min_x.max(0);
        let y0 = min_y.max(0);
        let x1 = max_x.min(width as i32);
        let y1 = max_y.min(height as i32);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        Some((x0 as u32, y0 as u32, x1 as u32, y1 as u32))
    }

    /// Pixel-centre offsets in subpixel units, for the incremental evaluators.
    #[inline]
    pub fn subpixel_deltas(&self) -> ([i64; 3], [i64; 3]) {
        // E at pixel centre (px + 0.5) is E(px*256 + 128).
        let mut dxs = [0i64; 3];
        let mut dys = [0i64; 3];
        for i in 0..3 {
            dxs[i] = self.edges[i].a * (SUBPIXEL_UNITS as i64);
            dys[i] = self.edges[i].b * (SUBPIXEL_UNITS as i64);
        }
        (dxs, dys)
    }

    #[inline]
    pub fn eval_at_pixel(&self, px: u32, py: u32) -> [i64; 3] {
        let sx = (px as i32) * SUBPIXEL_UNITS + SUBPIXEL_UNITS / 2;
        let sy = (py as i32) * SUBPIXEL_UNITS + SUBPIXEL_UNITS / 2;
        [
            eval(&self.edges[0], sx, sy),
            eval(&self.edges[1], sx, sy),
            eval(&self.edges[2], sx, sy),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subpixel_snap_is_symmetric() {
        assert_eq!(to_fixed(1.0), 256);
        assert_eq!(to_fixed(-1.0), -256);
        assert_eq!(to_fixed(0.5), 128);
        assert_eq!(to_fixed(-0.5), -128);
        assert_eq!(to_fixed(1.0 / 1024.0), 0, "half a subpixel rounds to zero");
    }

    #[test]
    fn interior_is_positive_for_a_screen_clockwise_triangle() {
        // v0 top-left, v1 bottom-left, v2 top-right: clockwise on a y-down screen.
        let s = Setup::new([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0]]).unwrap();
        assert!(s.front_facing());
        let e = s.eval_at_pixel(1, 1);
        assert!(e.iter().all(|v| *v > 0), "interior sample failed: {e:?}");
        // Pixel centre (8.5, 8.5) is past the hypotenuse x + y = 10.
        let outside = s.eval_at_pixel(8, 8);
        assert!(outside.iter().any(|v| *v < 0), "outside sample passed every edge: {outside:?}");
    }

    #[test]
    fn winding_flips_the_area_sign() {
        let a = Setup::new([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0]]).unwrap();
        let b = Setup::new([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]).unwrap();
        assert!(a.area2 > 0 && b.area2 < 0);
        assert!(a.area2 == -b.area2);
    }

    #[test]
    fn degenerate_triangles_are_rejected() {
        assert!(Setup::new([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]).is_none());
        assert!(Setup::new([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]).is_none());
    }

    #[test]
    fn bounds_clamp_to_the_target() {
        let s = Setup::new([[-5.0, -5.0], [-5.0, 40.0], [40.0, -5.0]]).unwrap();
        let (x0, y0, x1, y1) = s.bounds(16, 16).unwrap();
        assert_eq!((x0, y0, x1, y1), (0, 0, 16, 16));
        let s = Setup::new([[100.0, 100.0], [100.0, 120.0], [120.0, 100.0]]).unwrap();
        assert!(s.bounds(16, 16).is_none());
    }

    #[test]
    fn top_left_rule_covers_a_shared_edge_exactly_once() {
        // Two triangles filling the square [0,4)x[0,4), sharing the diagonal
        // (0,0)-(4,4). Every pixel centre must be covered exactly once.
        let a = Setup::new([[0.0, 0.0], [0.0, 4.0], [4.0, 4.0]]).unwrap();
        let b = Setup::new([[0.0, 0.0], [4.0, 4.0], [4.0, 0.0]]).unwrap();
        let mut coverage = [[0u32; 4]; 4];
        for t in [&a, &b] {
            for py in 0..4 {
                for px in 0..4 {
                    let e = t.eval_at_pixel(px, py);
                    if e.iter().all(|v| *v > 0) {
                        coverage[py as usize][px as usize] += 1;
                    }
                }
            }
        }
        for row in coverage {
            assert_eq!(row, [1, 1, 1, 1], "coverage: {coverage:?}");
        }
    }
}
