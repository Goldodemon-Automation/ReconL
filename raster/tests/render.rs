//! End-to-end rasteriser tests on real pixels.
//!
//! These are the tests that turn "the code compiles" into "the code draws": a
//! triangle is rendered, its coverage is compared against the analytic area, the
//! depth buffer is inspected, and the whole frame is hashed at 1, 2, 4 and 8
//! worker threads to prove the determinism claim in the docs.

use reconl_core::alloc::HostAlloc;
use reconl_raster::math::{ortho_rh_reversed_z, IDENTITY};
use reconl_raster::shade::SurfaceShader;
use reconl_raster::tile::{RasterConfig, Rasterizer};
use reconl_raster::{checksum_f32, DrawItem, PipelineState, ShaderRef, Target, Vertex};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

fn alloc() -> HostAlloc {
    HostAlloc::system()
}

/// World-space quad corners of the test triangle, in the [-1,1] ortho box.
const TRI: [[f32; 2]; 3] = [[-0.75, -0.5], [0.75, -0.5], [-0.75, 0.6]];

fn triangle_vertices() -> [Vertex; 3] {
    // z = -1 is in front of the camera; reversed-Z turns that into a depth
    // between 0 and 1.
    [
        Vertex::plain([TRI[0][0], TRI[0][1], -1.0], [1.0, 0.0, 0.0, 1.0]),
        Vertex::plain([TRI[1][0], TRI[1][1], -1.0], [1.0, 0.0, 0.0, 1.0]),
        Vertex::plain([TRI[2][0], TRI[2][1], -1.0], [1.0, 0.0, 0.0, 1.0]),
    ]
}

fn projection() -> [f32; 16] {
    ortho_rh_reversed_z(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0)
}

struct Render {
    target: Target,
    stats: reconl_raster::RasterStats,
}

fn render(threads: u32, tile_size: u32, samples: &[[f32; 2]; 3]) -> Render {
    let a = alloc();
    let mut raster = Rasterizer::new(a, RasterConfig { tile_size, worker_threads: threads, ..Default::default() });
    raster.prepare(WIDTH, HEIGHT, 6, 6).unwrap();
    let mut target = Target::new_color(a, WIDTH, HEIGHT).unwrap().with_depth().unwrap();
    target.clear_color([0.0, 0.0, 0.0, 1.0]);
    target.clear_depth(0.0);

    let verts = [
        Vertex::plain([samples[0][0], samples[0][1], -1.0], [1.0, 0.0, 0.0, 1.0]),
        Vertex::plain([samples[1][0], samples[1][1], -1.0], [1.0, 0.0, 0.0, 1.0]),
        Vertex::plain([samples[2][0], samples[2][1], -1.0], [1.0, 0.0, 0.0, 1.0]),
    ];
    let draws = [DrawItem {
        vertices: &verts,
        indices: None,
        transform: projection(),
        model: IDENTITY,
        pipeline: PipelineState::default(),
        shader: ShaderRef::Surface(SurfaceShader::unlit()),
        dynamic: false,
        casts_shadow: true,
    }];
    let stats = raster.rasterize(&mut target, &draws, (0, 0)).unwrap();
    Render { target, stats }
}

#[test]
fn triangle_covers_its_interior_and_nothing_else() {
    let r = render(1, 64, &TRI);
    let color = r.target.color_slice().unwrap();
    let depth = r.target.depth_slice().unwrap();

    let covered = |x: usize, y: usize| color[(y * WIDTH as usize + x) * 4] > 0.5;
    // Centroid of the triangle in screen space.
    let centroid_world = [
        (TRI[0][0] + TRI[1][0] + TRI[2][0]) / 3.0,
        (TRI[0][1] + TRI[1][1] + TRI[2][1]) / 3.0,
    ];
    let cx = ((centroid_world[0] * 0.5 + 0.5) * WIDTH as f32) as usize;
    let cy = ((1.0 - (centroid_world[1] * 0.5 + 0.5)) * HEIGHT as f32) as usize;
    assert!(covered(cx, cy), "centroid pixel ({cx},{cy}) was not drawn");

    // A corner far outside the triangle stays at the clear colour.
    assert!(!covered(62, 1) && !covered(1, 62));

    // Reversed-Z: drawn pixels are strictly in front of the cleared depth.
    let idx = cy * WIDTH as usize + cx;
    assert!(depth[idx] > 0.0 && depth[idx] < 1.0, "depth {}", depth[idx]);
    assert_eq!(depth[0], 0.0, "untouched pixels keep the clear value");

    // Coverage must match the analytic area within the rasterisation bound:
    // one pixel of perimeter error per edge, plus the top-left rule's share.
    let analytic = triangle_area();
    let drawn = (0..WIDTH as usize * HEIGHT as usize).filter(|p| covered(p % WIDTH as usize, p / WIDTH as usize)).count() as f32;
    let perimeter = triangle_perimeter();
    let bound = perimeter * 0.5 + 3.0;
    assert!(
        (drawn - analytic).abs() <= bound,
        "drawn {drawn} px, analytic {analytic} px, bound {bound}"
    );
}

/// The pixel counters are published once per tile, not once per pixel. That is
/// only a performance change if the totals are *identical* - so this pins them
/// against a count that owes nothing to the batching: a triangle that covers the
/// whole viewport must report exactly one shaded pixel per pixel of the target,
/// and one tested pixel wherever the depth test ran, at any worker count and any
/// tile size.
///
/// Every pixel centre is strictly inside the triangle's edges (the diagonal
/// leaves the [0,1] box through the corners), so the expected count is exact and
/// does not depend on the fill rule or on where tile boundaries fall.
#[test]
fn the_pixel_counters_total_exactly_the_pixels_covered() {
    let full_screen = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];
    let pixels = WIDTH as u64 * HEIGHT as u64;
    for (threads, tile) in [(1, 64), (8, 64), (4, 16), (8, 32)] {
        let r = render(threads, tile, &full_screen);
        assert_eq!(
            r.stats.pixels_shaded, pixels,
            "{threads} workers, {tile}px tiles: shaded {} of {pixels} covered pixels",
            r.stats.pixels_shaded
        );
        assert_eq!(
            r.stats.pixels_tested, pixels,
            "{threads} workers, {tile}px tiles: tested {} of {pixels} covered pixels",
            r.stats.pixels_tested
        );
        assert_eq!(
            r.stats.tiles_rendered, r.stats.tiles_total,
            "{threads} workers, {tile}px tiles: every tile must publish its work"
        );
    }
}

fn triangle_area() -> f32 {
    // World units to pixels: the ortho box is 2 units wide over WIDTH pixels.
    let scale_x = WIDTH as f32 / 2.0;
    let scale_y = HEIGHT as f32 / 2.0;
    let a = [TRI[0][0] * scale_x, TRI[0][1] * scale_y];
    let b = [TRI[1][0] * scale_x, TRI[1][1] * scale_y];
    let c = [TRI[2][0] * scale_x, TRI[2][1] * scale_y];
    ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs() * 0.5
}

fn triangle_perimeter() -> f32 {
    let scale_x = WIDTH as f32 / 2.0;
    let scale_y = HEIGHT as f32 / 2.0;
    let p = |i: usize| [TRI[i][0] * scale_x, TRI[i][1] * scale_y];
    let d = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    d(p(0), p(1)) + d(p(1), p(2)) + d(p(2), p(0))
}

#[test]
fn frame_is_bit_identical_for_1_2_4_and_8_workers() {
    let reference_color = render(1, 64, &TRI).target.color_slice().unwrap().to_vec();
    let reference_depth = render(1, 64, &TRI).target.depth_slice().unwrap().to_vec();
    let reference_hash = checksum_f32(&reference_color);

    for threads in [2u32, 4, 8] {
        let r = render(threads, 64, &TRI);
        assert_eq!(
            checksum_f32(r.target.color_slice().unwrap()),
            reference_hash,
            "colour differs with {threads} workers"
        );
        assert_eq!(r.target.color_slice().unwrap(), reference_color.as_slice(), "colour buffer differs with {threads} workers");
        assert_eq!(r.target.depth_slice().unwrap(), reference_depth.as_slice(), "depth buffer differs with {threads} workers");
    }
}

#[test]
fn tile_size_does_not_change_the_frame() {
    // A tile boundary must not be visible: the same scene at three tile sizes
    // produces the same picture. (The bin order is documented to be tile-major,
    // so this can only hold because tiles never overlap.)
    let reference = render(1, 64, &TRI).target.color_slice().unwrap().to_vec();
    for tile in [16u32, 32, 128] {
        let got = render(1, tile, &TRI).target.color_slice().unwrap().to_vec();
        assert_eq!(got, reference, "tile size {tile} changed the frame");
    }
}

#[test]
fn degenerate_and_offscreen_triangles_draw_nothing() {
    let degenerate = render(1, 64, &[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
    assert_eq!(degenerate.stats.triangles_degenerate, 1);
    assert!(degenerate.target.color_slice().unwrap().iter().all(|c| *c == 0.0 || *c == 1.0));
    assert_eq!(degenerate.stats.pixels_shaded, 0);

    let offscreen = render(1, 64, &[[-3.0, -3.0], [-2.5, -3.0], [-3.0, -2.5]]);
    assert_eq!(offscreen.stats.pixels_shaded, 0);
}

#[test]
fn back_faces_are_culled_and_two_sided_draws_them_flipped() {
    let clockwise = [[-0.75, -0.5], [0.75, -0.5], [-0.75, 0.6]];
    let flipped = [clockwise[0], clockwise[2], clockwise[1]];
    let front = render(1, 64, &clockwise);
    let back = render(1, 64, &flipped);
    assert!(front.stats.pixels_shaded > 0, "front face drew nothing");
    assert_eq!(back.stats.pixels_shaded, 0, "back face should be culled");
    assert_eq!(back.stats.triangles_culled, 1);

    // With culling off, the same triangle draws the same pixels.
    let a = alloc();
    let mut raster = Rasterizer::new(a, RasterConfig { tile_size: 64, worker_threads: 1, ..Default::default() });
    raster.prepare(WIDTH, HEIGHT, 6, 6).unwrap();
    let mut target = Target::new_color(a, WIDTH, HEIGHT).unwrap().with_depth().unwrap();
    target.clear_color([0.0, 0.0, 0.0, 1.0]);
    let verts = [
        Vertex::plain([flipped[0][0], flipped[0][1], -1.0], [1.0, 0.0, 0.0, 1.0]),
        Vertex::plain([flipped[1][0], flipped[1][1], -1.0], [1.0, 0.0, 0.0, 1.0]),
        Vertex::plain([flipped[2][0], flipped[2][1], -1.0], [1.0, 0.0, 0.0, 1.0]),
    ];
    let draws = [DrawItem {
        vertices: &verts,
        indices: None,
        transform: projection(),
        model: IDENTITY,
        pipeline: PipelineState { cull: reconl_raster::CULL_NONE, ..Default::default() },
        shader: ShaderRef::Surface(SurfaceShader::unlit()),
        dynamic: false,
        casts_shadow: true,
    }];
    let stats = raster.rasterize(&mut target, &draws, (0, 0)).unwrap();
    assert_eq!(stats.pixels_shaded, front.stats.pixels_shaded, "two-sided coverage differs");
    assert_eq!(target.color_slice().unwrap(), front.target.color_slice().unwrap());
}

#[test]
fn depth_prefers_the_nearer_triangle_in_either_draw_order() {
    // Two overlapping quads at different depths: the nearer one must win, and
    // the order they are submitted in must not matter.
    // Two quads at different view depths (z = -0.2 is nearer than z = -1.2).
    let dirs: [(f32, [f32; 4]); 2] = [(0.2, [0.0, 1.0, 0.0, 1.0]), (1.2, [0.0, 0.0, 1.0, 1.0])];
    for order in [[0usize, 1], [1, 0]] {
        let a = alloc();
        let mut raster = Rasterizer::new(a, RasterConfig { tile_size: 64, worker_threads: 1, ..Default::default() });
        raster.prepare(WIDTH, HEIGHT, 12, 12).unwrap();
        let mut target = Target::new_color(a, WIDTH, HEIGHT).unwrap().with_depth().unwrap();
        target.clear_color([0.0, 0.0, 0.0, 1.0]);
        target.clear_depth(0.0);
        let mut caches = Vec::new();
        for (z, color) in dirs {
            caches.push((
                [
                    Vertex::plain([-0.5, -0.5, -z], color),
                    Vertex::plain([0.5, -0.5, -z], color),
                    Vertex::plain([0.0, 0.5, -z], color),
                ],
                color,
            ));
        }
        let draws = [
            DrawItem {
                vertices: &caches[order[0]].0,
                indices: None,
                transform: projection(),
                model: IDENTITY,
                pipeline: PipelineState::default(),
                shader: ShaderRef::Surface(SurfaceShader::unlit()),
                dynamic: false,
                casts_shadow: true,
            },
            DrawItem {
                vertices: &caches[order[1]].0,
                indices: None,
                transform: projection(),
                model: IDENTITY,
                pipeline: PipelineState::default(),
                shader: ShaderRef::Surface(SurfaceShader::unlit()),
                dynamic: false,
                casts_shadow: true,
            },
        ];
        raster.rasterize(&mut target, &draws, (0, 0)).unwrap();
        let color = target.color_slice().unwrap();
        let mid = (32 * WIDTH as usize + 32) * 4;
        assert_eq!(&color[mid..mid + 3], &[0.0, 1.0, 0.0], "nearer quad lost with order {order:?}");
    }
}

#[test]
fn indexed_draws_match_the_unindexed_ones() {
    let verts = triangle_vertices();
    let indices = [0u32, 1, 2];
    let a = alloc();
    let mut raster = Rasterizer::new(a, RasterConfig::default());
    raster.prepare(WIDTH, HEIGHT, 6, 6).unwrap();
    let mut target = Target::new_color(a, WIDTH, HEIGHT).unwrap().with_depth().unwrap();
    target.clear_color([0.0, 0.0, 0.0, 1.0]);
    let draws = [DrawItem {
        vertices: &verts,
        indices: Some(&indices),
        transform: projection(),
        model: IDENTITY,
        pipeline: PipelineState::default(),
        shader: ShaderRef::Surface(SurfaceShader::unlit()),
        dynamic: false,
        casts_shadow: true,
    }];
    let stats = raster.rasterize(&mut target, &draws, (0, 0)).unwrap();
    let direct = render(1, 64, &TRI);
    assert_eq!(stats.pixels_shaded, direct.stats.pixels_shaded);
    assert_eq!(target.color_slice().unwrap(), direct.target.color_slice().unwrap());
}

#[test]
fn prepare_then_rasterize_allocates_nothing_in_the_frame() {
    let a = alloc();
    let mut raster = Rasterizer::new(a, RasterConfig { tile_size: 32, worker_threads: 2, ..Default::default() });
    let verts = triangle_vertices();
    raster.prepare(WIDTH, HEIGHT, 3, 3).unwrap();
    let mut target = Target::new_color(a, WIDTH, HEIGHT).unwrap().with_depth().unwrap();
    let draws = [DrawItem {
        vertices: &verts,
        indices: None,
        transform: projection(),
        model: IDENTITY,
        pipeline: PipelineState::default(),
        shader: ShaderRef::Surface(SurfaceShader::unlit()),
        dynamic: false,
        casts_shadow: true,
    }];
    for frame in 0..4 {
        target.clear_color([0.0, 0.0, 0.0, 1.0]);
        let stats = raster.rasterize(&mut target, &draws, (0, 0)).unwrap();
        assert_eq!(stats.allocations_in_frame, 0, "frame {frame} allocated");
    }
}

#[test]
fn identity_transform_puts_geometry_where_the_maths_says() {
    // Sanity check on the pipeline: with an identity transform and no culling,
    // a triangle covering NDC (-1..1) fills the whole 2x2 target.
    let a = alloc();
    let mut raster = Rasterizer::new(a, RasterConfig { tile_size: 8, worker_threads: 1, ..Default::default() });
    raster.prepare(2, 2, 3, 3).unwrap();
    let mut target = Target::new_color(a, 2, 2).unwrap().with_depth().unwrap();
    target.clear_color([0.0, 0.0, 0.0, 1.0]);
    let verts = [
        Vertex::plain([-1.0, -1.0, 0.5], [1.0, 1.0, 1.0, 1.0]),
        Vertex::plain([3.0, -1.0, 0.5], [1.0, 1.0, 1.0, 1.0]),
        Vertex::plain([-1.0, 3.0, 0.5], [1.0, 1.0, 1.0, 1.0]),
    ];
    let draws = [DrawItem {
        vertices: &verts,
        indices: None,
        transform: IDENTITY,
        model: IDENTITY,
        pipeline: PipelineState { cull: reconl_raster::CULL_NONE, ..Default::default() },
        shader: ShaderRef::Surface(SurfaceShader::unlit()),
        dynamic: false,
        casts_shadow: true,
    }];
    let stats = raster.rasterize(&mut target, &draws, (0, 0)).unwrap();
    assert_eq!(stats.pixels_shaded, 4, "the 2x2 target should be fully covered");
    for px in target.color_slice().unwrap().chunks_exact(4) {
        assert_eq!(px, [1.0, 1.0, 1.0, 1.0]);
    }
}
