//! The tiled rasteriser: bin, then raster, tile by tile.
//!
//! Binning is a two-pass counting sort over the frame's triangles, exactly like
//! a hardware tiler: count per tile, prefix-sum, place. Placement appends to a
//! tile's range in triangle order, so the sequence a tile sees is a subsequence
//! of the frame's draw order - which is what makes the output independent of the
//! worker count.
//!
//! Tile independence is the whole determinism argument:
//!
//! * each tile is claimed by exactly one worker (an atomic counter),
//! * no tile writes a pixel that another tile writes,
//! * within a tile the triangle order is fixed,
//!
//! so 1, 2, 4 and 8 workers produce byte-identical frames.

use crate::clip::{self, ClipVertex, W_MIN};
use crate::fixed::Setup;
use crate::shade;
use crate::texture::lod_from_derivatives;
use crate::{DrawItem, PipelineState, ShaderRef, Target, ATTR_COUNT, CULL_BACK, CULL_FRONT, COMPARE_GREATER};
use reconl_core::alloc::{HostAlloc, HostVec};
use reconl_core::error::{Code, Error, Result};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[derive(Clone, Copy, Debug)]
pub struct RasterConfig {
    pub tile_size: u32,
    pub worker_threads: u32,
    /// Keep the bin order fixed (the default). Off is an experiment, not a mode.
    pub deterministic_bins: bool,
    /// Force the scalar path even when the build has vector units available.
    pub scalar: bool,
}

impl Default for RasterConfig {
    fn default() -> Self {
        Self { tile_size: 64, worker_threads: 1, deterministic_bins: true, scalar: false }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct RasterStats {
    pub tiles_total: u32,
    pub tiles_rendered: u32,
    pub draws: u32,
    pub triangles_in: u32,
    pub triangles_binned: u32,
    pub triangles_culled: u32,
    pub triangles_clipped: u32,
    pub triangles_degenerate: u32,
    pub bin_entries: u32,
    pub pixels_tested: u64,
    pub pixels_shaded: u64,
    /// Times pre-reserved storage had to grow mid-frame. Must be 0 in steady
    /// state; counted rather than hidden.
    pub allocations_in_frame: u32,
    pub storage_bytes: u64,
    /// Highest number of entries placed into any single tile, for sizing the
    /// next frame's reservation.
    pub peak_tile_entries: u32,
}

#[derive(Clone, Copy, Default, Debug)]
struct Entry {
    draw: u32,
    /// Offset into the flattened index buffer of this triangle's first index.
    first_index: u32,
}

/// Shared target memory for the tile workers.
///
/// # Safety
/// Sound because the tile grid partitions the framebuffer: no two workers receive
/// the same tile, and a tile writes only its own pixels. The pointers stay valid
/// for the duration of [`Rasterizer::rasterize`], which is the only place they
/// are handed to another thread.
struct SharedTarget {
    color: *mut f32,
    depth: *mut f32,
    width: u32,
    height: u32,
    /// The pass's viewport, resolved to target pixels by
    /// [`crate::rendered_viewport`]. Clip space maps onto this rect and no pixel
    /// of the target outside it is touched, so the render is confined exactly
    /// where a sub-rect was asked for. It is `(width, height)` - the whole
    /// target - whenever the host left the viewport unset.
    render: (u32, u32),
}

unsafe impl Send for SharedTarget {}
unsafe impl Sync for SharedTarget {}

#[derive(Default)]
struct Counters {
    tiles_rendered: AtomicU32,
    pixels_tested: AtomicU64,
    pixels_shaded: AtomicU64,
}

/// One tile's pixel work, accumulated on the stack and folded into [`Counters`]
/// once per tile.
///
/// The counts used to be `fetch_add`ed from inside the pixel loop, one atomic
/// read-modify-write per tested and per shaded pixel. That is instrumentation
/// costing 44% of the raster pass (76.7 ms -> 42.9 ms at 1080p on the reference
/// tier, measured), and every worker contending on the same two cache lines.
/// Counting into a local and publishing once per tile counts the same pixels and
/// changes no pixel.
#[derive(Default, Clone, Copy)]
struct TileWork {
    tested: u64,
    shaded: u64,
}

impl Counters {
    fn add(&self, work: TileWork) {
        self.pixels_tested.fetch_add(work.tested, Ordering::Relaxed);
        self.pixels_shaded.fetch_add(work.shaded, Ordering::Relaxed);
    }
}

/// One triangle, as the binning pass sees it.
struct TriInfo {
    needs_clip: bool,
    culled: bool,
    degenerate: bool,
    bounds: Option<(u32, u32, u32, u32)>,
}

/// Tile indices touched by the pixel rect `[x0,x1) x [y0,y1)`, row-major.
fn tiles_of(x0: u32, y0: u32, x1: u32, y1: u32, tiles_x: u32, tile_size: u32) -> impl Iterator<Item = u32> {
    let tx0 = x0 / tile_size;
    let tx1 = (x1 + tile_size - 1) / tile_size;
    let ty0 = y0 / tile_size;
    let ty1 = (y1 + tile_size - 1) / tile_size;
    (ty0..ty1).flat_map(move |ty| (tx0..tx1).map(move |tx| ty * tiles_x + tx))
}

pub struct Rasterizer {
    config: RasterConfig,
    xform: HostVec<ClipVertex>,
    indices: HostVec<u32>,
    counts: HostVec<u32>,
    starts: HostVec<u32>,
    cursors: HostVec<u32>,
    entries: HostVec<Entry>,
    tiles_x: u32,
    tiles_y: u32,
    stats: RasterStats,
}

impl Rasterizer {
    pub fn new(alloc: HostAlloc, config: RasterConfig) -> Self {
        let tile_size = config.tile_size.clamp(8, 1024).next_power_of_two();
        Self {
            config: RasterConfig { tile_size, ..config },
            xform: HostVec::new(alloc),
            indices: HostVec::new(alloc),
            counts: HostVec::new(alloc),
            starts: HostVec::new(alloc),
            cursors: HostVec::new(alloc),
            entries: HostVec::new(alloc),
            tiles_x: 0,
            tiles_y: 0,
            stats: RasterStats::default(),
        }
    }

    pub fn config(&self) -> RasterConfig {
        self.config
    }

    pub fn worker_threads(&self) -> u32 {
        self.config.worker_threads.max(1)
    }

    pub fn tile_size(&self) -> u32 {
        self.config.tile_size
    }

    pub fn stats(&self) -> RasterStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = RasterStats::default();
    }

    /// Reserves every buffer this frame needs.
    ///
    /// **This is the only place the rasteriser allocates**, and it is called at a
    /// frame boundary, never between `BeginFrame` and `Present`.
    pub fn prepare(&mut self, width: u32, height: u32, vertex_capacity: usize, index_capacity: usize) -> Result<()> {
        let (tiles_x, tiles_y) = tile_grid(width, height, self.config.tile_size);
        self.tiles_x = tiles_x;
        self.tiles_y = tiles_y;
        let tiles = (self.tiles_x as usize).max(1) * (self.tiles_y as usize).max(1);

        self.xform.clear();
        self.indices.clear();
        self.entries.clear();
        self.xform.try_reserve(vertex_capacity.max(3))?;
        self.indices.try_reserve(index_capacity.max(3))?;
        // One entry per triangle, plus slack for triangles that bin into several
        // tiles (clipped ones bin into every tile).
        let entry_capacity = (index_capacity / 3).max(64) + tiles;
        self.entries.try_reserve(entry_capacity)?;
        self.counts.try_reserve(tiles)?;
        self.starts.try_reserve(tiles)?;
        self.cursors.try_reserve(tiles)?;
        Ok(())
    }

    pub fn storage_bytes(&self) -> u64 {
        let entries = (self.entries.capacity() * std::mem::size_of::<Entry>()) as u64;
        let clip = (self.xform.capacity() * std::mem::size_of::<ClipVertex>()) as u64;
        let indices = (self.indices.capacity() * std::mem::size_of::<u32>()) as u64;
        let tiles = (self.counts.capacity() * 3 * std::mem::size_of::<u32>()) as u64;
        entries + clip + indices + tiles
    }

    /// Renders `draws` into `target`, confined to `viewport`.
    ///
    /// Colour is written when the target has a colour buffer and the draw is not
    /// depth-only; depth follows the pipeline state. The viewport is in frame
    /// pixels and is resolved against the target here by
    /// [`crate::rendered_viewport`], so a caller that rasterises into a scaled
    /// target passes the host's request unchanged and gets the same fraction of
    /// the frame. `(0, 0)` renders the whole target.
    pub fn rasterize(
        &mut self,
        target: &mut Target,
        draws: &[DrawItem<'_>],
        viewport: (u32, u32),
    ) -> Result<RasterStats> {
        let width = target.width;
        let height = target.height;
        if width == 0 || height == 0 {
            return Err(Error::new(Code::InvalidArgument, "zero-sized target"));
        }
        // Frame and target coincide here: a caller holding a scaled target
        // resolves the viewport itself and passes the target's own size, which
        // makes this a no-op.
        let render = crate::rendered_viewport(viewport, (width, height), (width, height));
        let tile_size = self.config.tile_size.max(8);
        let mut frame = RasterStats {
            tiles_total: (self.tiles_x * self.tiles_y) as u32,
            draws: draws.len() as u32,
            ..Default::default()
        };
        // The grid belongs to the target, not to the rasteriser. A device that
        // rasterises into two sizes - a colour target and a shadow map, say -
        // must rebin for each, and `prepare` is where the tables for a grid get
        // reserved. Keeping a grid from an earlier, smaller target is how a
        // shadow pass came to bin tile indices that only exist in a bigger map.
        // The capacities are retained, so after the first frame this is a
        // comparison and a `resize_with` that never grows.
        let (tiles_x, tiles_y) = tile_grid(width, height, self.config.tile_size);
        if self.tiles_x != tiles_x || self.tiles_y != tiles_y {
            self.prepare(width, height, draws.iter().map(|d| d.vertices.len()).sum(), draws.iter().map(|d| index_len(d)).sum())?;
            frame.tiles_total = (self.tiles_x * self.tiles_y) as u32;
        }
        let tiles = (self.tiles_x as usize) * (self.tiles_y as usize);

        // ---- transform every draw's vertices and flatten the index buffers.
        // These are per-frame scratch: clearing keeps the capacity `prepare`
        // reserved, so the frame reuses the same memory instead of regrowing.
        self.xform.clear();
        self.indices.clear();
        let mut vertex_base = 0usize;
        for draw in draws.iter() {
            let grew = append_transformed(&mut self.xform, draw)?;
            if grew {
                frame.allocations_in_frame += 1;
            }
            let grew = append_indices(&mut self.indices, draw, vertex_base)?;
            if grew {
                frame.allocations_in_frame += 1;
            }
            vertex_base += draw.vertices.len();
            frame.triangles_in += (index_len(draw) / 3) as u32;
        }

        // ---- pass A: how many triangles does each tile see?
        // Zeroed, not merely sized: `resize_with` only grows, so a count left
        // behind by the previous frame would be added to and would place that
        // frame's triangles again, on top of this frame's. The bin would then
        // hold every frame so far and the cost would climb with the frame
        // count instead of staying flat.
        self.counts.resize_with(tiles, || 0)?;
        self.counts.as_mut_slice().fill(0);
        self.starts.resize_with(tiles, || 0)?;
        self.cursors.resize_with(tiles, || 0)?;
        // Read the grid into locals before destructuring `self`, so both passes
        // provably use the same values.
        let grid_x = self.tiles_x.max(1);
        {
            let Self { xform, indices, counts, .. } = self;
            let xform_slice = xform.as_slice();
            let index_slice = indices.as_slice();
            let count_slice = counts.as_mut_slice();
            let mut culled = 0u32;
            let mut clipped = 0u32;
            let mut degenerate = 0u32;
            visit_triangles(draws, xform_slice, index_slice, render, &mut |info, _draw, _first, _v| {
                if info.degenerate {
                    degenerate += 1;
                    return;
                }
                if info.culled {
                    culled += 1;
                    return;
                }
                if info.needs_clip {
                    clipped += 1;
                    for c in count_slice.iter_mut() {
                        *c = c.saturating_add(1);
                    }
                } else if let Some((x0, y0, x1, y1)) = info.bounds {
                    for tile in tiles_of(x0, y0, x1, y1, grid_x, tile_size) {
                        count_slice[tile as usize] = count_slice[tile as usize].saturating_add(1);
                    }
                }
            });
            frame.triangles_culled = culled;
            frame.triangles_clipped = clipped;
            frame.triangles_degenerate = degenerate;
        }

        // ---- prefix sum.
        let mut running = 0u32;
        for t in 0..tiles {
            self.starts[t] = running;
            self.cursors[t] = running;
            running = running.saturating_add(self.counts[t]);
        }
        let total_entries = running as usize;
        if self.entries.capacity() < total_entries {
            frame.allocations_in_frame += 1;
            self.entries.try_reserve(total_entries - self.entries.capacity())?;
        }
        self.entries.resize_with(total_entries, Entry::default)?;

        // ---- pass B: place entries. Same iteration order as pass A, so a tile's
        // range is a stable subsequence of the frame's triangle order.
        {
            let Self { xform, indices, cursors, entries, .. } = self;
            let xform_slice = xform.as_slice();
            let index_slice = indices.as_slice();
            let cursor_slice = cursors.as_mut_slice();
            let entry_slice = entries.as_mut_slice();
            visit_triangles(draws, xform_slice, index_slice, render, &mut |info, draw_index, first_index, _v| {
                if info.culled || info.degenerate {
                    return;
                }
                let tile_count = cursor_slice.len() as u32;
                let mut place = |tile: u32| {
                    let slot = cursor_slice[tile as usize] as usize;
                    if slot < entry_slice.len() {
                        entry_slice[slot] = Entry { draw: draw_index as u32, first_index: first_index as u32 };
                        cursor_slice[tile as usize] += 1;
                    }
                };
                if info.needs_clip {
                    for tile in 0..tile_count {
                        place(tile);
                    }
                } else if let Some((x0, y0, x1, y1)) = info.bounds {
                    for tile in tiles_of(x0, y0, x1, y1, grid_x, tile_size) {
                        place(tile);
                    }
                }
            });
        }

        // ---- raster.
        let shared = SharedTarget {
            color: target.color_slice_mut().map(|s| s.as_mut_ptr()).unwrap_or(std::ptr::null_mut()),
            depth: target.depth_slice_mut().map(|s| s.as_mut_ptr()).unwrap_or(std::ptr::null_mut()),
            width,
            height,
            render,
        };
        let counters = Counters::default();
        {
            let Self { xform, indices, starts, counts, entries, config, .. } = self;
            raster_tiles(
                config.worker_threads.max(1),
                tile_size,
                self.tiles_x,
                self.tiles_y,
                draws,
                xform.as_slice(),
                indices.as_slice(),
                starts.as_slice(),
                counts.as_slice(),
                entries.as_slice(),
                &shared,
                &counters,
            );
        }

        frame.tiles_rendered = counters.tiles_rendered.load(Ordering::Relaxed);
        frame.pixels_tested = counters.pixels_tested.load(Ordering::Relaxed);
        frame.pixels_shaded = counters.pixels_shaded.load(Ordering::Relaxed);
        frame.bin_entries = total_entries as u32;
        frame.triangles_binned = total_entries as u32;
        frame.peak_tile_entries = self.counts.iter().map(|c| *c).max().unwrap_or(0);
        frame.storage_bytes = self.storage_bytes();
        self.stats = frame;
        Ok(frame)
    }
}

fn index_len(draw: &DrawItem<'_>) -> usize {
    draw.indices.map(|i| i.len()).unwrap_or(draw.vertices.len())
}

fn append_transformed(dst: &mut HostVec<ClipVertex>, draw: &DrawItem<'_>) -> Result<bool> {
    let mut grew = false;
    if dst.capacity() < dst.len() + draw.vertices.len() {
        grew = true;
        dst.try_reserve(draw.vertices.len())?;
    }
    for v in draw.vertices.iter() {
        dst.push(clip::transform_vertex(&draw.transform, v.position, v.attrs()))?;
    }
    Ok(grew)
}

fn append_indices(dst: &mut HostVec<u32>, draw: &DrawItem<'_>, vertex_base: usize) -> Result<bool> {
    let mut grew = false;
    let count = index_len(draw);
    if dst.capacity() < dst.len() + count {
        grew = true;
        dst.try_reserve(count)?;
    }
    match draw.indices {
        Some(idx) => {
            for i in idx.iter() {
                dst.push(*i + vertex_base as u32)?;
            }
        }
        None => {
            for i in 0..draw.vertices.len() {
                dst.push((i + vertex_base) as u32)?;
            }
        }
    }
    Ok(grew)
}

/// Iterates the frame's triangles in draw order, applying culling and setup.
///
/// `render` is the pass's viewport in target pixels: it is what clip space maps
/// onto, and therefore what bounds are clamped to.
fn visit_triangles<F>(
    draws: &[DrawItem<'_>],
    xform: &[ClipVertex],
    indices: &[u32],
    render: (u32, u32),
    visit: &mut F,
) where
    F: FnMut(&TriInfo, usize, usize, [ClipVertex; 3]),
{
    let mut first_index = 0usize;
    for (draw_index, draw) in draws.iter().enumerate() {
        let count = index_len(draw) / 3;
        for t in 0..count {
            let base = first_index + t * 3;
            if base + 2 >= indices.len() {
                break;
            }
            let i0 = indices[base] as usize;
            let i1 = indices[base + 1] as usize;
            let i2 = indices[base + 2] as usize;
            if i0 >= xform.len() || i1 >= xform.len() || i2 >= xform.len() {
                continue;
            }
            let verts = [xform[i0], xform[i1], xform[i2]];
            let info = classify(&verts, draw.pipeline.cull, render);
            visit(&info, draw_index, base, verts);
        }
        first_index += index_len(draw);
    }
}

/// Clipping decision, screen setup, culling decision and bounds for one triangle.
fn classify(v: &[ClipVertex; 3], cull: u32, render: (u32, u32)) -> TriInfo {
    if clip::needs_clip(v) {
        return TriInfo { needs_clip: true, culled: false, degenerate: false, bounds: None };
    }
    let mut screen = [[0.0f32; 2]; 3];
    for i in 0..3 {
        let w = v[i].clip[3];
        if w <= W_MIN {
            return TriInfo { needs_clip: true, culled: false, degenerate: false, bounds: None };
        }
        let inv = 1.0 / w;
        // Clip space is y-up, the framebuffer is y-down (D3D convention), so a
        // front face winds clockwise on screen and `Setup::area2 > 0`.
        screen[i] = [
            (v[i].clip[0] * inv * 0.5 + 0.5) * render.0 as f32,
            (1.0 - (v[i].clip[1] * inv * 0.5 + 0.5)) * render.1 as f32,
        ];
    }
    let setup = match Setup::new(screen) {
        Some(s) => s,
        None => return TriInfo { needs_clip: false, culled: true, degenerate: true, bounds: None },
    };
    let front = setup.front_facing();
    let culled = match cull {
        CULL_BACK => !front,
        CULL_FRONT => front,
        _ => false,
    };
    // Clamped to the render area, not the target: a triangle that lies outside
    // the viewport is binned nowhere, which is what confines a sub-rect pass.
    let bounds = if culled { None } else { setup.bounds(render.0, render.1) };
    TriInfo { needs_clip: false, culled, degenerate: false, bounds }
}

/// Per-triangle state used by the pixel loop.
struct TriJob<'a> {
    setup: Setup,
    aw: [[f32; ATTR_COUNT]; 3],
    iw: [f32; 3],
    zow: [f32; 3],
    inv_area: f32,
    biases: [i64; 3],
    dl_dx: [f32; 3],
    dl_dy: [f32; 3],
    pipeline: PipelineState,
    shader: ShaderRef<'a>,
    flip_normal: bool,
    rect: (u32, u32, u32, u32),
    texels: [f32; 2],
}

impl<'a> TriJob<'a> {
    fn build(t: [ClipVertex; 3], draw: &DrawItem<'a>, rect: (u32, u32, u32, u32), render: (u32, u32)) -> Option<TriJob<'a>> {
        let mut screen = [[0.0f32; 2]; 3];
        let mut iw = [0.0f32; 3];
        for i in 0..3 {
            let w = t[i].clip[3];
            if w <= W_MIN {
                return None;
            }
            let inv = 1.0 / w;
            iw[i] = inv;
            screen[i] = [
                (t[i].clip[0] * inv * 0.5 + 0.5) * render.0 as f32,
                (1.0 - (t[i].clip[1] * inv * 0.5 + 0.5)) * render.1 as f32,
            ];
        }

        let setup = Setup::new(screen)?;
        let front = setup.front_facing();
        let culled = match draw.pipeline.cull {
            CULL_BACK => !front,
            CULL_FRONT => front,
            _ => false,
        };
        if culled {
            return None;
        }
        // Normalise the winding so the interior is always `E > 0`.
        let flip = !front;
        let (t, screen, iw) = if flip {
            ([t[0], t[2], t[1]], [screen[0], screen[2], screen[1]], [iw[0], iw[2], iw[1]])
        } else {
            (t, screen, iw)
        };
        let setup = Setup::new(screen)?;

        let (bx0, by0, bx1, by1) = setup.bounds(render.0, render.1)?;
        let x0 = bx0.max(rect.0);
        let y0 = by0.max(rect.1);
        let x1 = bx1.min(rect.2);
        let y1 = by1.min(rect.3);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }

        let mut aw = [[0.0f32; ATTR_COUNT]; 3];
        let mut zow = [0.0f32; 3];
        for i in 0..3 {
            for k in 0..ATTR_COUNT {
                aw[i][k] = t[i].attr[k] * iw[i];
            }
            zow[i] = t[i].clip[2] * iw[i];
        }

        let inv_area = 1.0 / (setup.area2 as f32);
        // lambda_0 = E1/area, lambda_1 = E2/area, lambda_2 = E0/area
        let dl_dx = [
            setup.edges[1].a as f32 * inv_area,
            setup.edges[2].a as f32 * inv_area,
            setup.edges[0].a as f32 * inv_area,
        ];
        let dl_dy = [
            setup.edges[1].b as f32 * inv_area,
            setup.edges[2].b as f32 * inv_area,
            setup.edges[0].b as f32 * inv_area,
        ];
        let biases = [setup.edges[0].bias, setup.edges[1].bias, setup.edges[2].bias];

        let texels = match draw.shader {
            ShaderRef::Surface(s) => match s.texture {
                Some(tex) if !tex.levels.is_empty() => [tex.levels[0].width as f32, tex.levels[0].height as f32],
                _ => [1.0, 1.0],
            },
            ShaderRef::DepthOnly => [1.0, 1.0],
        };

        Some(TriJob {
            setup,
            aw,
            iw,
            zow,
            inv_area,
            biases,
            dl_dx,
            dl_dy,
            pipeline: draw.pipeline,
            shader: draw.shader,
            flip_normal: flip && draw.pipeline.two_sided,
            rect: (x0, y0, x1, y1),
            texels,
        })
    }

    /// Level of detail from the analytic screen-space uv derivatives at a pixel.
    #[inline]
    fn lod(&self, attr: &[f32; ATTR_COUNT], l: [f32; 3], iw: f32) -> f32 {
        let n_u = l[0] * self.aw[0][crate::ATTR_UV] + l[1] * self.aw[1][crate::ATTR_UV] + l[2] * self.aw[2][crate::ATTR_UV];
        let n_v = l[0] * self.aw[0][crate::ATTR_UV + 1] + l[1] * self.aw[1][crate::ATTR_UV + 1] + l[2] * self.aw[2][crate::ATTR_UV + 1];
        let dw_dx = self.dl_dx[0] * self.iw[0] + self.dl_dx[1] * self.iw[1] + self.dl_dx[2] * self.iw[2];
        let dw_dy = self.dl_dy[0] * self.iw[0] + self.dl_dy[1] * self.iw[1] + self.dl_dy[2] * self.iw[2];
        let dn_u_dx = self.dl_dx[0] * self.aw[0][crate::ATTR_UV]
            + self.dl_dx[1] * self.aw[1][crate::ATTR_UV]
            + self.dl_dx[2] * self.aw[2][crate::ATTR_UV];
        let dn_u_dy = self.dl_dy[0] * self.aw[0][crate::ATTR_UV]
            + self.dl_dy[1] * self.aw[1][crate::ATTR_UV]
            + self.dl_dy[2] * self.aw[2][crate::ATTR_UV];
        let dn_v_dx = self.dl_dx[0] * self.aw[0][crate::ATTR_UV + 1]
            + self.dl_dx[1] * self.aw[1][crate::ATTR_UV + 1]
            + self.dl_dx[2] * self.aw[2][crate::ATTR_UV + 1];
        let dn_v_dy = self.dl_dy[0] * self.aw[0][crate::ATTR_UV + 1]
            + self.dl_dy[1] * self.aw[1][crate::ATTR_UV + 1]
            + self.dl_dy[2] * self.aw[2][crate::ATTR_UV + 1];
        let rcp2 = 1.0 / (iw * iw);
        let du_dx = (dn_u_dx * iw - n_u * dw_dx) * rcp2;
        let du_dy = (dn_u_dy * iw - n_u * dw_dy) * rcp2;
        let dv_dx = (dn_v_dx * iw - n_v * dw_dx) * rcp2;
        let dv_dy = (dn_v_dy * iw - n_v * dw_dy) * rcp2;
        // Attribute values are in [0,1] texture space; the texel scale is applied
        // by the lod rule itself.
        let _ = attr;
        lod_from_derivatives(du_dx, dv_dx, du_dy, dv_dy, self.texels[0], self.texels[1])
    }

    fn raster(&self, shared: &SharedTarget, work: &mut TileWork) {
        let (x0, y0, x1, y1) = self.rect;
        let (dxs, dys) = self.setup.subpixel_deltas();
        let width = shared.width as usize;
        let surface = match self.shader {
            ShaderRef::Surface(s) => {
                let mut s = s;
                s.flip_normal = self.flip_normal;
                Some(s)
            }
            ShaderRef::DepthOnly => None,
        };
        let write_color = surface.is_some() && !shared.color.is_null();
        let textured = surface.as_ref().map(|s| s.textured).unwrap_or(false);
        let mut row = self.setup.eval_at_pixel(x0, y0);

        for py in y0..y1 {
            let mut e = row;
            for px in x0..x1 {
                if e[0] > 0 && e[1] > 0 && e[2] > 0 {
                    let l = [
                        ((e[1] - self.biases[1]) as f32) * self.inv_area,
                        ((e[2] - self.biases[2]) as f32) * self.inv_area,
                        ((e[0] - self.biases[0]) as f32) * self.inv_area,
                    ];
                    let iw = l[0] * self.iw[0] + l[1] * self.iw[1] + l[2] * self.iw[2];
                    if iw > 0.0 {
                        let pixel = py as usize * width + px as usize;
                        // NDC depth: `zow` holds `z/w` per vertex and the edge
                        // functions are affine barycentrics, so their sum is the
                        // NDC depth at this pixel - which is what the depth
                        // buffer is documented to hold (`Target`), what hardware
                        // depth buffers hold, and what the frame generator
                        // unprojects with. Dividing by `iw` as well would
                        // interpolate *clip* z instead, a different scale that
                        // the depth test cannot tell apart but a depth readback
                        // can. For an orthographic matrix `w` is exactly 1, so
                        // the shadow passes are unaffected.
                        let depth = l[0] * self.zow[0] + l[1] * self.zow[1] + l[2] * self.zow[2];
                        let mut keep = true;
                        if self.pipeline.depth_test && !shared.depth.is_null() {
                            // SAFETY: `pixel` is inside the target and this tile
                            // owns it, so no other worker touches this element.
                            let stored = unsafe { *shared.depth.add(pixel) };
                            keep = if self.pipeline.depth_compare == COMPARE_GREATER {
                                depth > stored
                            } else {
                                depth < stored
                            };
                            work.tested += 1;
                        }
                        if keep {
                            if let (true, Some(surface)) = (write_color, surface.as_ref()) {
                                let rcp = 1.0 / iw;
                                let mut attr = [0.0f32; ATTR_COUNT];
                                for k in 0..ATTR_COUNT {
                                    attr[k] = (l[0] * self.aw[0][k] + l[1] * self.aw[1][k] + l[2] * self.aw[2][k]) * rcp;
                                }
                                let lod = if textured { self.lod(&attr, l, iw) } else { 0.0 };
                                let mut out = [0.0f32; 4];
                                surface.shade(&attr, lod, &mut out);
                                // SAFETY: as above, this pixel belongs to this tile.
                                let dst = unsafe { std::slice::from_raw_parts_mut(shared.color.add(pixel * 4), 4) };
                                shade::blend(self.pipeline.blend, out, dst);
                                work.shaded += 1;
                            }
                            if self.pipeline.depth_write && !shared.depth.is_null() {
                                // SAFETY: as above.
                                unsafe { *shared.depth.add(pixel) = depth };
                            }
                        }
                    }
                }
                e[0] += dxs[0];
                e[1] += dxs[1];
                e[2] += dxs[2];
            }
            row[0] += dys[0];
            row[1] += dys[1];
            row[2] += dys[2];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn raster_tiles(
    threads: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    draws: &[DrawItem<'_>],
    xform: &[ClipVertex],
    indices: &[u32],
    starts: &[u32],
    counts: &[u32],
    entries: &[Entry],
    shared: &SharedTarget,
    counters: &Counters,
) {
    let tiles = tiles_x * tiles_y;
    let width = shared.width;
    let height = shared.height;
    let (render_w, render_h) = shared.render;
    let body = |tile: u32| {
        let tx = tile % tiles_x;
        let ty = tile / tiles_x;
        let tx0 = tx * tile_size;
        let ty0 = ty * tile_size;
        // The tile is clipped to both the target and the viewport, so a tile
        // that straddles the viewport's edge stops exactly at it and a tile
        // entirely outside it does no work. Tiles still do not overlap: this
        // only shrinks each tile's own rect.
        let tx1 = (tx0 + tile_size).min(width).min(render_w);
        let ty1 = (ty0 + tile_size).min(height).min(render_h);
        if tx0 >= tx1 || ty0 >= ty1 {
            return;
        }
        let rect = (tx0, ty0, tx1, ty1);
        if tile as usize >= starts.len() || tile as usize >= counts.len() {
            return;
        }
        let start = starts[tile as usize] as usize;
        let count = counts[tile as usize] as usize;
        let mut work = TileWork::default();
        for k in start..start.saturating_add(count) {
            let entry = match entries.get(k) {
                Some(e) => *e,
                None => break,
            };
            let draw = match draws.get(entry.draw as usize) {
                Some(d) => d,
                None => continue,
            };
            let base = entry.first_index as usize;
            if base + 2 >= indices.len() {
                continue;
            }
            let verts = [
                xform[indices[base] as usize],
                xform[indices[base + 1] as usize],
                xform[indices[base + 2] as usize],
            ];
            if clip::needs_clip(&verts) {
                let mut out = [[ClipVertex::default(); 3]; clip::MAX_OUTPUT_TRIANGLES];
                let n = clip::clip_triangle(&verts, &mut out);
                for tri in out.iter().take(n) {
                    if let Some(job) = TriJob::build(*tri, draw, rect, (render_w, render_h)) {
                        job.raster(shared, &mut work);
                    }
                }
            } else if let Some(job) = TriJob::build(verts, draw, rect, (render_w, render_h)) {
                job.raster(shared, &mut work);
            }
        }
        counters.add(work);
        counters.tiles_rendered.fetch_add(1, Ordering::Relaxed);
    };

    if threads <= 1 {
        for tile in 0..tiles {
            body(tile);
        }
        return;
    }
    let next = AtomicU32::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let tile = next.fetch_add(1, Ordering::Relaxed);
                if tile >= tiles {
                    break;
                }
                body(tile);
            });
        }
    });
}

/// The tile grid a target of this size produces at this tile size.
///
/// `prepare` and `rasterize` both need it, and a disagreement between them is
/// not a small bug: the grid is what sizes the per-tile tables, so a grid from
/// an earlier, smaller target is what indexes past the end of them.
pub fn tile_grid(width: u32, height: u32, tile_size: u32) -> (u32, u32) {
    let tile = tile_size.max(8);
    ((width + tile - 1) / tile, (height + tile - 1) / tile)
}

pub fn tile_count(width: u32, height: u32, tile_size: u32) -> u32 {
    let (x, y) = tile_grid(width, height, tile_size);
    x * y
}
