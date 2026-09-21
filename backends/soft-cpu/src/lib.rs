//! The `soft-cpu` backend: ReconL's reference implementation.
//!
//! Everything here exists to be *believable*. It is the tier the GPU tiers are
//! diffed against, so it may not approximate, guess, or take a shortcut that
//! changes a pixel. It has no platform code, no GPU API, no threads of its own
//! beyond the rasteriser's worker pool, and it runs the same on a laptop, in CI,
//! and inside a WebAssembly sandbox.
//!
//! What it actually implements, tier by tier:
//!
//! * **T2 `cpu-ram`** - the tiled reference rasteriser, full cascades, RAM-resident
//!   static cascade cache, PCF/PCSS shadow filtering.
//! * **T3 `cpu-thrifty`** - the same frame path with the shadow maps and static
//!   cascades *frozen* between refreshes: the map from frame `N` is reused on
//!   frames `N+1 .. N+k` and every reuse is counted as a `FrozenCascade` event,
//!   never silently. Resolution scale and LOD caps follow the tier rules.
//! * **T4 `out-of-core`** - the static cascade cache moves into the `RCLS` disk
//!   arena. A cascade that hits the arena is uploaded instead of re-rasterised;
//!   the bytes moved are reported in `FrameNumbers::spill_io_bytes` and the hit
//!   rate in `ShadowCounters`, so "out-of-core" is a measurement rather than a
//!   claim.
//!
//! The division of labour with the host is deliberate: the host builds the draw
//! list and says *what to shade* (textured or not, lit or not, shadow-receiving
//! or not) and this backend supplies *the lighting state* - the cascade fits,
//! the maps, the bias preset, the filter the tier actually permits. A host
//! cannot accidentally hand the rasteriser a shadow map it did not render, and
//! cannot accidentally omit one it did.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod policy;

pub use policy::{Emptiness, FrameClassifier, FramePolicy};

use reconl_contract::{FrameInput, ShadowRequest};
use reconl_core::alloc::{HostAlloc, HostVec};
use reconl_core::budget::{Budget, Reservation};
use reconl_core::error::{Code, Error, Result};
use reconl_core::stats::{Counters, FrameNumbers, ShadowCounters};
use reconl_core::tier::{caps, rules, shadow_plan, Backend, ShadowFilter, ShadowPlan, Tier, TierReason};
use reconl_raster::math;
use reconl_raster::shade::{LightSet, ShadowLookup, ShadowMapRef, SurfaceShader};
use reconl_raster::tile::RasterConfig;
use reconl_raster::{
    checksum_f32, rendered_viewport, DrawItem, PipelineState, RasterStats, Rasterizer, ShaderRef, Target,
    COMPARE_GREATER, CULL_BACK,
};
use reconl_resource::spill::{Hit, SpillArena, SpillConfig};
use reconl_shadow as shadow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct SoftCpuConfig {
    pub tier: Tier,
    pub worker_threads: u32,
    pub tile_size: u32,
    /// Force the non-SIMD pixel loop. Both paths must agree bit for bit.
    pub scalar: bool,
    pub resolution_scale: f32,
    pub target_frame_ms: u32,
    pub seed: u32,
    pub spill_dir: Option<PathBuf>,
    pub frame_policy: FramePolicy,
    /// Cascade split lambda: 0 = uniform, 1 = logarithmic.
    pub split_lambda: f32,
    /// Cap on the arena. `0` = the budget's disk cap only.
    pub arena_bytes: u64,
    pub shadow: ShadowRequest,
}

impl Default for SoftCpuConfig {
    fn default() -> Self {
        Self {
            tier: Tier::CpuRam,
            worker_threads: 0, // 0 = one per hardware thread
            tile_size: 32,
            scalar: false,
            resolution_scale: 1.0,
            target_frame_ms: 16,
            seed: 0x5ECD_1000,
            spill_dir: None,
            frame_policy: FramePolicy::default(),
            split_lambda: 0.75,
            arena_bytes: 0,
            shadow: ShadowRequest::default(),
        }
    }
}

/// Everything the device reports to the host in one struct, so the FFI layer
/// copies numbers rather than reaching into the backend.
#[derive(Clone, Copy, Default)]
pub struct SoftCpuSnapshot {
    pub counters: Counters,
    pub shadows: ShadowCounters,
    pub frame: FrameNumbers,
    pub classifier: policy::FrameClassifier,
    pub color_checksum: u64,
    pub resident_bytes: u64,
    pub arena_entries: u64,
    pub arena_bytes: u64,
    pub spill_io_bytes: u64,
}

pub struct SoftCpuDevice {
    alloc: HostAlloc,
    budget: Arc<Budget>,
    config: SoftCpuConfig,
    tier: Tier,
    tier_reason: TierReason,
    raster: Rasterizer,
    color: Target,
    maps: HostVec<Target>,
    map_size: u32,
    map_count: u32,
    /// Which cascade's map holds valid content from the same light.
    maps_valid: [bool; shadow::MAX_CASCADES],
    map_light_hash: u64,
    /// The cascade pass's draw list, owned by the device.
    ///
    /// The shadow pass writes this frame's casters into it, one cascade at a
    /// time, so the storage is a frame's to reuse rather than a frame's to
    /// allocate. `DrawItem<'static>` is the *type* of the storage, not a claim
    /// about what is in it: an entry is a copy of the frame's own draw, whose
    /// vertex slices point into the host's buffer storage - kept alive for the
    /// frame's use by the host's own references (the ABI's handle rules) - and
    /// whose shading state is nothing at all (a depth pass shades no colour).
    /// `fill_cascade` clears the list before it writes, so no entry is read
    /// after the frame that put it there.
    cascade: HostVec<DrawItem<'static>>,
    /// The colour pass's draw list, owned by the device for the same reason and
    /// under the same rule as `cascade`: an entry is this frame's draw with this
    /// backend's lighting state stamped into it, cleared by the next frame's
    /// fill, so nothing here outlives the frame whose data it copies.
    colors: HostVec<DrawItem<'static>>,
    plan: Option<ShadowPlan>,
    arena: Option<SpillArena>,
    arena_reservation: Option<Reservation>,
    color_reservation: Option<Reservation>,
    map_reservation: Option<Reservation>,
    classifier: FrameClassifier,
    counters: Counters,
    shadows: ShadowCounters,
    frame: FrameNumbers,
    color_checksum: u64,
    spill_io_bytes: u64,
    arena_error: Option<Error>,
}

impl SoftCpuDevice {
    pub fn new(alloc: HostAlloc, budget: Arc<Budget>, config: SoftCpuConfig) -> Result<Self> {
        alloc.self_check()?;
        if config.tile_size != 0 && config.tile_size < 8 {
            return Err(Error::new(Code::InvalidArgument, "tile_size must be 0 or at least 8"));
        }
        let threads = if config.worker_threads == 0 {
            std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1)
        } else {
            config.worker_threads
        };
        let raster = Rasterizer::new(
            alloc,
            RasterConfig {
                tile_size: config.tile_size.max(8),
                worker_threads: threads,
                deterministic_bins: true,
                scalar: config.scalar,
            },
        );
        let color = Target::new_color(alloc, 1, 1).map(|t| t.with_depth())??;
        let frame_policy = config.frame_policy;
        Ok(Self {
            alloc,
            budget,
            tier: config.tier,
            config,
            tier_reason: TierReason::HostRequest,
            raster,
            color,
            maps: HostVec::new(alloc),
            cascade: HostVec::new(alloc),
            colors: HostVec::new(alloc),
            map_size: 0,
            map_count: 0,
            maps_valid: [false; shadow::MAX_CASCADES],
            map_light_hash: 0,
            plan: None,
            arena: None,
            arena_reservation: None,
            color_reservation: None,
            map_reservation: None,
            classifier: FrameClassifier::new(frame_policy),
            counters: Counters::default(),
            shadows: ShadowCounters::default(),
            frame: FrameNumbers::default(),
            color_checksum: 0,
            spill_io_bytes: 0,
            arena_error: None,
        })
    }

    pub fn config(&self) -> &SoftCpuConfig {
        &self.config
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    pub fn tier_reason(&self) -> TierReason {
        self.tier_reason
    }

    pub fn backend(&self) -> Backend {
        Backend::SoftCpu
    }

    pub fn worker_threads(&self) -> u32 {
        self.raster.worker_threads()
    }

    pub fn tile_size(&self) -> u32 {
        self.raster.tile_size()
    }

    pub fn device_name(&self) -> &'static str {
        "soft-cpu (reference tiled rasteriser)"
    }

    pub fn driver(&self) -> &'static str {
        "reconl reference"
    }

    pub fn caps(&self) -> u32 {
        caps_for(self.tier)
    }

    pub fn set_frame_policy(&mut self, policy: FramePolicy) {
        self.classifier.policy = policy;
    }

    pub fn classifier(&self) -> &FrameClassifier {
        &self.classifier
    }

    pub fn plan(&self) -> Option<ShadowPlan> {
        self.plan
    }

    pub fn map_size(&self) -> u32 {
        self.map_size
    }

    pub fn budget(&self) -> &Arc<Budget> {
        &self.budget
    }

    /// The last frame's colour, `f32` RGBA. Exposed for tests and for the
    /// golden-image comparison, which must see the same numbers a GPU sees.
    pub fn color_slice(&self) -> &[f32] {
        self.color.color_slice().unwrap_or(&[])
    }

    pub fn depth_slice(&self) -> &[f32] {
        self.color.depth_slice().unwrap_or(&[])
    }

    /// One cascade map's raw reversed-Z depth.
    pub fn cascade_depth(&self, index: usize) -> Option<&[f32]> {
        self.maps.get(index).and_then(|m| m.depth_slice())
    }

    /// Resident bytes this device is accountable for.
    pub fn resident_bytes(&self) -> u64 {
        self.color_reservation.as_ref().map(|r| r.bytes()).unwrap_or(0)
            + self.map_reservation.as_ref().map(|r| r.bytes()).unwrap_or(0)
    }

    /// Applies a tier the *device* decided on: the same backend at a lower
    /// quality tier, with everything a tier change invalidates dropped.
    ///
    /// The backend does not decide this and does not record it. A device's tier
    /// has one owner - the frame-time ladder the device runs
    /// (`ffi/src/offload.rs`, `docs/offload.md`) - and one log, because a
    /// backend's own ring dies with the backend while the tier it changed does
    /// not.
    pub fn relabel(&mut self, to: Tier, reason: TierReason) {
        self.tier = to;
        self.tier_reason = reason;
        self.counters.frames_since_tier_change = 0;
        // A tier change invalidates the shadow layout: the new tier has a
        // different cascade cap, filter cap and cache location.
        for slot in self.maps_valid.iter_mut() {
            *slot = false;
        }
    }

    fn ensure_targets(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(Error::new(Code::InvalidArgument, "zero-sized frame"));
        }
        if self.color.width == width && self.color.height == height && self.color.depth.is_some() {
            return Ok(());
        }
        let bytes = (width as u64) * (height as u64) * 4 * 2;
        let reservation = self.budget.reserve_ram(bytes).map_err(|e| {
            self.counters.safe_path_events += 1;
            e
        })?;
        let target = Target::new_color(self.alloc, width, height)?.with_depth()?;
        self.color = target;
        self.color_reservation = Some(reservation);
        Ok(())
    }

    /// Allocates the cascade set the plan decided on, reserving what the plan
    /// says it costs - `plan.resident_bytes`, not a second opinion about the
    /// arithmetic. No size is decided here: both tiers allocate the set
    /// [`ShadowPlan`] describes, so one budget buys the same maps on either.
    fn ensure_maps(&mut self, plan: &ShadowPlan) -> Result<()> {
        let (cascades, map_size) = (plan.cascades, plan.map_size);
        if self.map_count == cascades && self.map_size == map_size && self.maps.len() == cascades as usize {
            return Ok(());
        }
        let bytes = plan.resident_bytes;
        let reservation = self.budget.reserve_ram(bytes).map_err(|e| {
            self.counters.safe_path_events += 1;
            e
        })?;
        let mut maps = HostVec::with_capacity(self.alloc, cascades as usize)?;
        for _ in 0..cascades {
            maps.push(Target::new_depth(self.alloc, map_size, map_size)?)?;
        }
        self.maps = maps;
        self.map_size = map_size;
        self.map_count = cascades;
        self.maps_valid = [false; shadow::MAX_CASCADES];
        self.map_reservation = Some(reservation);
        Ok(())
    }

    /// Fills the cascade list with this frame's casters, for one cascade.
    ///
    /// Owned storage, written per pass: clearing keeps the capacity, so a frame
    /// whose casters fit in the storage an earlier frame grew allocates nothing.
    /// Growth is counted in `frame.allocations_in_frame`, the way the rasteriser
    /// counts its own tables - so a steady state reports zero rather than a
    /// number nobody can check.
    /// Takes the list rather than `&mut self` on purpose: the colour pass fills
    /// its list while a view of the shadow maps - this device's `maps` - is still
    /// live, and a `&mut self` here would end that borrow.
    fn fill_cascade<'s>(
        list: &mut HostVec<DrawItem<'static>>,
        casters: impl Iterator<Item = &'s DrawItem<'s>>,
        view_proj: &math::Mat4,
        pipeline: PipelineState,
        frame: &mut FrameNumbers,
    ) -> Result<()> {
        list.clear();
        for draw in casters {
            let mut shadow_draw = *draw;
            shadow_draw.transform = math::mul(view_proj, &draw.model);
            shadow_draw.shader = ShaderRef::DepthOnly;
            shadow_draw.pipeline = pipeline;
            if list.len() == list.capacity() {
                frame.allocations_in_frame += 1;
            }
            // SAFETY: the two `DrawItem`s differ only in the lifetimes their
            // slices carry, so no byte of the entry changes. What is stored was
            // copied from this frame's draws and is dropped - on the `clear`
            // above - before the next frame begins, so the shortened lifetime
            // never outlives the borrow it was copied from.
            let entry = unsafe { core::mem::transmute::<DrawItem<'s>, DrawItem<'static>>(shadow_draw) };
            list.push(entry)?;
        }
        Ok(())
    }

    /// Fills the colour pass's list from the frame's draws, stamping in the
    /// lighting state this backend owns.
    ///
    /// Owned storage, written once per frame: clearing keeps the capacity, so a
    /// frame no larger than an earlier one allocates nothing. Growth is counted
    /// the same way `fill_cascade` counts it.
    fn fill_colors<'s>(
        list: &mut HostVec<DrawItem<'static>>,
        items: impl Iterator<Item = &'s DrawItem<'s>>,
        lights: &'s LightSet,
        lookup: Option<&'s ShadowLookup<'s>>,
        frame: &mut FrameNumbers,
    ) -> Result<()> {
        list.clear();
        for draw in items {
            let mut item = *draw;
            if let ShaderRef::Surface(surface) = item.shader {
                let mut surface = surface;
                surface.lights = if surface.lit { Some(lights) } else { None };
                surface.shadows = if surface.receives_shadow { lookup } else { None };
                item.shader = ShaderRef::Surface(surface);
            }
            if list.len() == list.capacity() {
                frame.allocations_in_frame += 1;
            }
            // SAFETY: as in `fill_cascade` - the entry is a copy of this frame's
            // draw with this frame's lighting state in it, and the `clear` above
            // drops it before the next frame is filled.
            let entry = unsafe { core::mem::transmute::<DrawItem<'s>, DrawItem<'static>>(item) };
            list.push(entry)?;
        }
        Ok(())
    }

    fn ensure_arena(&mut self) -> Option<&mut SpillArena> {
        if self.arena.is_none() {
            if !self.budget.caps().allow_disk_spill {
                return None;
            }
            let dir = self.config.spill_dir.clone().unwrap_or_else(reconl_resource::spill::default_spill_dir);
            let mut config = SpillConfig::new(dir);
            config.max_bytes = self.config.arena_bytes;
            config.seed = 0x5243_4C53_0000_0001 ^ (u64::from(self.config.seed) << 1);
            match SpillArena::open(config) {
                Ok(arena) => {
                    // Account for the arena up front; a refused reservation means
                    // the host capped disk use below the arena, so the RAM-only
                    // path is used and the refusal is counted by the budget.
                    if self.config.arena_bytes > 0 {
                        match self.budget.reserve_disk(self.config.arena_bytes) {
                            Ok(r) => self.arena_reservation = Some(r),
                            Err(e) => {
                                self.arena_error = Some(e);
                                self.counters.safe_path_events += 1;
                                return None;
                            }
                        }
                    }
                    self.arena = Some(arena);
                }
                Err(e) => {
                    self.arena_error = Some(e);
                    self.counters.safe_path_events += 1;
                    return None;
                }
            }
        }
        self.arena.as_mut()
    }

    pub fn arena_error(&self) -> Option<&Error> {
        self.arena_error.as_ref()
    }

    pub fn snapshot(&self) -> SoftCpuSnapshot {
        SoftCpuSnapshot {
            counters: self.counters,
            shadows: self.shadows,
            frame: self.frame,
            classifier: self.classifier,
            color_checksum: self.color_checksum,
            resident_bytes: self.resident_bytes(),
            arena_entries: self.arena.as_ref().map(|a| a.stats().entries).unwrap_or(0),
            arena_bytes: self.arena.as_ref().map(|a| a.stats().bytes).unwrap_or(0),
            spill_io_bytes: self.spill_io_bytes,
        }
    }

    pub fn arena_stats(&self) -> Option<reconl_resource::spill::ArenaStats> {
        self.arena.as_ref().map(|a| a.stats())
    }

    /// Renders one frame into the internal target.
    ///
    /// Everything the frame needs was reserved by `ensure_*` first, and every
    /// draw list a pass writes into is owned before the frame starts: the
    /// cascade list and the colour list are the device's ([`SoftCpuDevice`]'s
    /// `cascade` and `colors`). A steady-state frame therefore takes nothing from
    /// the host allocator - what can still grow, a frame with more casters than
    /// any before it, is counted in `frame.allocations_in_frame`.
    pub fn render(&mut self, input: &FrameInput<'_>) -> Result<FrameNumbers> {
        let frame_start = Instant::now();
        let mut frame = FrameNumbers {
            frame_index: input.frame_index,
            resolution_scale: self.config.resolution_scale.min(rules(self.tier).resolution_scale),
            worker_threads: self.raster.worker_threads(),
            ..Default::default()
        };

        self.ensure_targets(input.width, input.height)?;

        let request = input.shadow;
        let shadow_wanted = request.enabled && self.plan_capable();
        let plan = if shadow_wanted {
            Some(shadow_plan(
                self.tier,
                request.cascades,
                request.texel_budget_bytes,
                request.filter,
                self.caps(),
            ))
        } else {
            None
        };
        self.plan = plan;

        // ---- shadow pass -----------------------------------------------------
        let mut shadow_stats = RasterStats::default();
        let mut shadow_ns = 0u64;
        let mut fit_ns = 0u64;
        if let Some(plan) = plan {
            self.ensure_maps(&plan)?;
            let map_size = plan.map_size;
            self.shadows.map_width = map_size;
            self.shadows.map_height = map_size;
            self.shadows.map_bytes = plan.resident_bytes;
            self.shadows.cascades_active = plan.cascades;
            self.shadows.filter_active = plan.filter;
            self.shadows.filter_requested = plan.filter_requested;
            self.shadows.filter_taps = plan.filter.taps();
            // A clamp is not a fail-safe: nothing was skipped, the tier simply
            // gets smaller maps and a cheaper filter, and the host sees that as
            // `cascades_active < requested` and `filter_active != requested`.
            // `safe_path_events` is reserved for the cases where the fast path
            // was abandoned (a refused allocation, a failed arena write).
            let _ = plan.clamp_event;

            let fit_start = Instant::now();
            let cascades = shadow::fit_cascades(&shadow::FitInput {
                camera_view: input.camera_view,
                fov_y_deg: input.fov_y_deg,
                aspect: input.aspect,
                near: input.near,
                light_dir: input.light_dir,
                cascade_count: plan.cascades,
                max_distance: request.max_distance,
                split_lambda: self.config.split_lambda,
                map_size,
                snap: true,
            });
            fit_ns += fit_start.elapsed().as_nanos() as u64;

            let refresh = request.refresh_interval_frames.max(1);
            let refresh_frame = input.frame_index % u64::from(refresh) == 0;
            if self.map_light_hash != input.light_hash {
                self.maps_valid = [false; shadow::MAX_CASCADES];
                self.map_light_hash = input.light_hash;
            }

            let cache_enabled = plan.disk_backed && request.allow_disk_cache;

            // Which casters this frame has, asked rather than gathered: the two
            // lists this pass used to build were a copy of the frame's draws,
            // and the cascade sub-passes read the frame's own list instead.
            let has_static = input.draws.iter().any(|d| d.casts_shadow && !d.dynamic);
            let has_dynamic = input.draws.iter().any(|d| d.casts_shadow && d.dynamic);

            let shadow_pipeline = PipelineState {
                // Back-face culling on the light's view: the surface facing the
                // light is the one the shadow ray would hit, and culling the far
                // side is what keeps the depth bias small enough to be honest.
                //
                // This said `CULL_FRONT` while the sentence above says the far
                // side is what gets culled, and the two do not agree: culling
                // front faces on the light's view culls the surface the ray
                // hits, which for a single-sided caster is the caster itself.
                // Every scene in this project casts from a single-sided quad or
                // triangle facing the light, so the shadow pass rendered an
                // empty map and no light in the project could cast a shadow.
                cull: CULL_BACK,
                depth_compare: COMPARE_GREATER,
                depth_test: true,
                depth_write: true,
                ..PipelineState::default()
            };

            for index in 0..plan.cascades as usize {
                let fit = match cascades.get(index) {
                    Some(f) => *f,
                    None => break,
                };
                if plan.freeze_static && !refresh_frame && self.maps_valid[index] {
                    // T3/T4: the map from the last refresh still describes the
                    // same static geometry under the same light. Reusing it is a
                    // counted event, not a silent shortcut.
                    self.shadows.frozen_cascades += 1;
                    continue;
                }

                let key = shadow::cache_key(&shadow::CacheKeyInput {
                    light_hash: input.light_hash,
                    cascade_index: index as u32,
                    view_proj: fit.view_proj,
                    world_revision: input.world_revision,
                    filter: plan.filter as u32,
                    map_size,
                    reversed_z: true,
                    static_geometry_revision: input.static_geometry_revision,
                });

                // One arena round trip: the borrow is confined to this block so
                // the counters it feeds stay outside it.
                let mut loaded: Option<Vec<u8>> = None;
                let mut load_hit = Hit::Miss;
                if cache_enabled && has_static {
                    if let Some(arena) = self.ensure_arena() {
                        let mut bytes = Vec::new();
                        load_hit = arena.get(key, &mut bytes);
                        if load_hit == Hit::Fresh {
                            loaded = Some(bytes);
                        }
                    }
                }
                let mut rendered_statics = false;
                let mut from_cache = false;
                match load_hit {
                    Hit::Fresh => {
                        let bytes = loaded.take().unwrap_or_default();
                        let expected = (map_size as usize) * (map_size as usize) * 4;
                        if bytes.len() == expected {
                            write_depth_bytes(self.maps.get_mut(index), &bytes);
                            self.shadows.cache_hits += 1;
                            self.shadows.cache_bytes_hit += bytes.len() as u64;
                            self.shadows.cache_bytes_read += bytes.len() as u64;
                            self.spill_io_bytes += bytes.len() as u64;
                            frame.spill_io_bytes += bytes.len() as u64;
                            from_cache = true;
                        } else {
                            // Right key, wrong payload size: treat it as
                            // corruption rather than trusting it.
                            self.shadows.cache_corrupt += 1;
                        }
                    }
                    Hit::Corrupt => {
                        self.shadows.cache_corrupt += 1;
                    }
                    Hit::Miss => {
                        if cache_enabled && has_static {
                            self.shadows.cache_misses += 1;
                        }
                    }
                }

                let pass_start = Instant::now();
                if !from_cache {
                    if let Some(map) = self.maps.get_mut(index) {
                        map.clear_depth(0.0);
                    }
                    Self::fill_cascade(
                        &mut self.cascade,
                        input.draws.iter().filter(|d| d.casts_shadow && !d.dynamic),
                        &fit.view_proj,
                        shadow_pipeline,
                        &mut frame,
                    )?;
                    rendered_statics = !self.cascade.is_empty();
                    if let Some(map) = self.maps.get_mut(index) {
                        // The shadow pass draws the whole map: the viewport the
                        // host set applies to the colour pass, not to a cascade.
                        let stats = self.raster.rasterize(map, self.cascade.as_slice(), (0, 0))?;
                        accumulate_raster(&mut shadow_stats, &stats);
                    }
                }

                if rendered_statics && cache_enabled {
                    let bytes = depth_bytes(self.maps.get(index));
                    let mut written = 0u64;
                    let mut failed = false;
                    if !bytes.is_empty() {
                        if let Some(arena) = self.ensure_arena() {
                            match arena.put(key, &bytes) {
                                Ok(n) => written = n,
                                Err(_) => failed = true,
                            }
                        }
                    }
                    self.spill_io_bytes += written;
                    frame.spill_io_bytes += written;
                    if failed {
                        self.counters.safe_path_events += 1;
                    }
                }

                if has_dynamic {
                    Self::fill_cascade(
                        &mut self.cascade,
                        input.draws.iter().filter(|d| d.casts_shadow && d.dynamic),
                        &fit.view_proj,
                        shadow_pipeline,
                        &mut frame,
                    )?;
                    if let Some(map) = self.maps.get_mut(index) {
                        let stats = self.raster.rasterize(map, self.cascade.as_slice(), (0, 0))?;
                        accumulate_raster(&mut shadow_stats, &stats);
                    }
                }
                shadow_ns += pass_start.elapsed().as_nanos() as u64;
                self.maps_valid[index] = true;
                self.shadows.cascades_rendered += 1;
            }
        } else {
            self.shadows = ShadowCounters::default();
        }

        // ---- colour pass -----------------------------------------------------
        // The host's draws carry the *shading intent*; the lighting state (fits,
        // maps, bias) is supplied here, so a host cannot pass a shadow map the
        // backend did not produce.
        let raster_start = Instant::now();
        if input.clear_color_enabled {
            self.color.clear_color(input.clear_color);
        }
        if input.clear_depth_enabled {
            self.color.clear_depth(input.clear_depth);
        }

        let emptiness = self.classifier.classify(input.draws.len(), count_triangles(input.draws));
        let clears_anything = input.clear_color_enabled || input.clear_depth_enabled;
        let mut color_stats = RasterStats::default();

        if !self.classifier.may_skip_pass(emptiness, clears_anything) {
            // The fits are recomputed here from the same inputs as the shadow
            // pass, so sampling and rendering agree by construction (the fit is
            // snapped to texels, which is what makes it reproducible rather than
            // merely close).
            let fits = if let Some(p) = plan {
                shadow::fit_cascades(&shadow::FitInput {
                    camera_view: input.camera_view,
                    fov_y_deg: input.fov_y_deg,
                    aspect: input.aspect,
                    near: input.near,
                    light_dir: input.light_dir,
                    cascade_count: p.cascades,
                    max_distance: request.max_distance,
                    split_lambda: self.config.split_lambda,
                    map_size: self.map_size,
                    snap: true,
                })
            } else {
                shadow::CascadeSet::empty()
            };
            let cascade_array = fits.as_lookup_cascades();

            // A cascade set is bounded by `MAX_CASCADES`, so what the shading
            // step sees is a fixed array on the stack: this is a *view* of maps
            // the device already owns, and a view needs no allocation.
            let mut map_refs: [ShadowMapRef<'_>; shadow::MAX_CASCADES] =
                [ShadowMapRef { width: 0, height: 0, depth: &[] }; shadow::MAX_CASCADES];
            let mut map_count = 0usize;
            for index in 0..self.map_count.min(shadow::MAX_CASCADES as u32) as usize {
                if let Some(map) = self.maps.get(index) {
                    map_refs[index] = ShadowMapRef {
                        width: map.width,
                        height: map.height,
                        depth: map.depth_slice().unwrap_or(&[]),
                    };
                    map_count = index + 1;
                }
            }
            let map_refs = &map_refs[..map_count];

            let filter = plan.map(|p| p.filter).unwrap_or(ShadowFilter::Hard);
            let bias = request
                .bias
                .unwrap_or_else(|| shadow::bias_preset(self.tier, self.map_size.max(1), filter));
            let lookup = shadow::lookup(
                &cascade_array,
                map_refs,
                &input.camera_view,
                filter,
                bias,
                request.max_distance,
                request.blend_band,
            );
            let lookup_opt = if plan.is_some() { Some(lookup) } else { None };

            // The host's draws carry the shading *intent*; the lighting state
            // (fits, maps, bias) is supplied here, so a host cannot pass a shadow
            // map the backend did not produce. It goes into this device's own
            // list rather than into a fresh one per frame.
            Self::fill_colors(
                &mut self.colors,
                input.draws.iter(),
                &input.lights,
                lookup_opt.as_ref(),
                &mut frame,
            )?;

            let rasterize_start = Instant::now();
            // The host's viewport is in frame pixels; this tier may be rendering
            // a scaled target, so resolve it against that target. `(0, 0)` - the
            // documented default - resolves to the whole target.
            let render = rendered_viewport(
                input.viewport,
                (input.width, input.height),
                (self.color.width, self.color.height),
            );
            let stats = self.raster.rasterize(&mut self.color, self.colors.as_slice(), render)?;
            accumulate_raster(&mut color_stats, &stats);
            let _ = rasterize_start;

            self.shadows.shadowed_lights = count_shadowed_lights(&input.lights);
        }

        let raster_ns = raster_start.elapsed().as_nanos() as u64;

        self.shadows.shadow_pass_ns = shadow_ns;
        self.shadows.fit_ns = fit_ns;
        frame.shadow_ns = shadow_ns;
        frame.raster_ns = raster_ns;
        frame.tiles_total = color_stats.tiles_total + shadow_stats.tiles_total;
        frame.tiles_rendered = color_stats.tiles_rendered + shadow_stats.tiles_rendered;
        frame.triangles_in = color_stats.triangles_in;
        frame.triangles_binned = color_stats.triangles_binned + shadow_stats.triangles_binned;
        frame.triangles_culled = color_stats.triangles_culled + shadow_stats.triangles_culled;
        frame.pixels_shaded = (color_stats.pixels_shaded + shadow_stats.pixels_shaded).min(u32::MAX as u64) as u32;
        frame.total_ns = frame_start.elapsed().as_nanos() as u64;

        // The frame fingerprint the audit compares, computed only when it is
        // asked for: it is a full pass over the colour target and nothing else
        // reads it.
        self.color_checksum = if input.checksum {
            checksum_f32(self.color.color_slice().unwrap_or(&[]))
        } else {
            0
        };

        // Classification and the frame-time ladder. Both are telemetry: neither
        // changes a pixel, they only decide what gets counted and what steps
        // down.
        self.classifier.record(emptiness, input.frame_index)?;
        self.counters.frames_since_tier_change += 1;
        self.frame = frame;
        self.shadows.cascades_active = plan.map(|p| p.cascades).unwrap_or(0);
        Ok(frame)
    }

    fn plan_capable(&self) -> bool {
        (self.caps() & caps::SHADOWS) != 0
    }

    /// Reserves everything the next frame needs, between frames.
    ///
    /// This is the frame loop's allocation boundary: the colour and depth
    /// targets, the rasteriser's tile storage and the cascade maps are all taken
    /// here, so `Submit` reuses them instead of growing. A refused reservation
    /// is reported (and counted) rather than taken out mid-frame.
    pub fn prepare_frame(&mut self, width: u32, height: u32) -> Result<()> {
        self.ensure_targets(width, height)?;
        let vertices = 1usize << 16;
        let indices = 1usize << 16;
        self.raster.prepare(width, height, vertices, indices)?;
        if self.config.shadow.enabled {
            let plan = shadow_plan(
                self.tier,
                self.config.shadow.cascades,
                self.config.shadow.texel_budget_bytes,
                self.config.shadow.filter,
                self.caps(),
            );
            self.ensure_maps(&plan)?;
        }
        Ok(())
    }

    /// Records a fail-safe event the host asked for, so the counters a host
    /// reads back include the ones it caused.
    pub fn note_shadow_event(&mut self, event: reconl_core::tier::ShadowEvent) {
        use reconl_core::tier::ShadowEvent as E;
        match event {
            E::CacheHit => self.shadows.cache_hits += 1,
            E::CacheMiss => self.shadows.cache_misses += 1,
            E::CacheCorrupt => self.shadows.cache_corrupt += 1,
            E::FrozenCascade => self.shadows.frozen_cascades += 1,
            E::FallbackUnshadowed => self.shadows.fail_safe_unshadowed += 1,
            // The rest describe the *plan*, which the host already sees as
            // `cascades_active` and `filter_active` in the shadow stats.
            E::None | E::CascadeDropped | E::MapBudgetClamped | E::FilterDowngraded => {}
        }
    }

    /// Copies the frame's depth out of the target into `out`, tightly packed,
    /// `width * height` values, for frame generation.
    ///
    /// The reference tier's depth is a slice it already owns, so this costs a
    /// copy and no driver round trip - the same buffer the rasteriser wrote.
    pub fn depth_into(&mut self, out: &mut [f32]) -> Result<()> {
        let depth = self.color.depth_slice().ok_or_else(|| {
            Error::new(Code::NotSupported, "the frame target has no depth buffer to reproject")
        })?;
        if out.len() < depth.len() {
            return Err(Error::new(Code::InvalidArgument, "the depth buffer is too small for the frame"));
        }
        out[..depth.len()].copy_from_slice(depth);
        Ok(())
    }

    /// Writes the last rendered frame into the host's presentation buffer, in
    /// the layout the present asked for: `pitch` is the destination's row length
    /// in bytes and `flip` reverses the row order.
    ///
    /// The conversion happens straight into those rows. There is no tightly
    /// packed intermediate: a frame crossed the target->host boundary once, and
    /// the buffer the host handed over is the only one involved.
    pub fn read_frame_into(&mut self, out: &mut [u8], pitch: u32, flip: u32) -> Result<()> {
        let row_bytes = self.color.width as usize * 4;
        let pitch = if pitch == 0 { row_bytes } else { pitch as usize };
        self.color.to_rgba8_rows(out, pitch, flip != 0)
    }

    pub fn color_checksum(&self) -> u64 {
        self.color_checksum
    }

    pub fn frame_size(&self) -> (u32, u32) {
        (self.color.width, self.color.height)
    }

    pub fn counters(&self) -> Counters {
        self.counters
    }

    pub fn shadows(&self) -> ShadowCounters {
        self.shadows
    }

    /// Flushes the arena. Called at frame boundaries, never mid-frame.
    pub fn on_frame_end(&mut self) -> Result<()> {
        if let Some(arena) = self.arena.as_mut() {
            arena.flush()?;
        }
        Ok(())
    }
}

fn count_triangles(draws: &[DrawItem<'_>]) -> u64 {
    let mut total = 0u64;
    for draw in draws {
        let n = match draw.indices {
            Some(idx) => idx.len(),
            None => draw.vertices.len(),
        };
        total += (n / 3) as u64;
    }
    total
}

fn count_shadowed_lights(lights: &LightSet) -> u32 {
    lights
        .lights
        .iter()
        .take(lights.count as usize)
        .flatten()
        .filter(|l| l.cast_shadow)
        .count() as u32
}

fn accumulate_raster(total: &mut RasterStats, stats: &RasterStats) {
    total.tiles_total += stats.tiles_total;
    total.tiles_rendered += stats.tiles_rendered;
    total.draws += stats.draws;
    total.triangles_in += stats.triangles_in;
    total.triangles_binned += stats.triangles_binned;
    total.triangles_culled += stats.triangles_culled;
    total.triangles_clipped += stats.triangles_clipped;
    total.triangles_degenerate += stats.triangles_degenerate;
    total.bin_entries += stats.bin_entries;
    total.pixels_tested += stats.pixels_tested;
    total.pixels_shaded += stats.pixels_shaded;
    total.allocations_in_frame += stats.allocations_in_frame;
    total.storage_bytes = total.storage_bytes.max(stats.storage_bytes);
    total.peak_tile_entries = total.peak_tile_entries.max(stats.peak_tile_entries);
}

fn depth_bytes(map: Option<&Target>) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(map) = map {
        if let Some(depth) = map.depth_slice() {
            out.reserve(depth.len() * 4);
            for v in depth {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
    }
    out
}

fn write_depth_bytes(map: Option<&mut Target>, bytes: &[u8]) {
    if let Some(map) = map {
        if let Some(depth) = map.depth_slice_mut() {
            for (i, slot) in depth.iter_mut().enumerate() {
                let at = i * 4;
                if at + 4 > bytes.len() {
                    break;
                }
                let mut raw = [0u8; 4];
                raw.copy_from_slice(&bytes[at..at + 4]);
                *slot = f32::from_bits(u32::from_le_bytes(raw));
            }
        }
    }
}

/// Capabilities of a software tier. There is no GPU, so nothing here claims to
/// be hardware: what it claims is that the reference path can do the work.
pub fn caps_for(tier: Tier) -> u32 {
    let mut caps = caps::TEXTURES
        | caps::MIPMAPS
        | caps::SHADOWS
        | caps::PCF_5X5
        | caps::PCSS_LITE
        | caps::MULTITHREAD
        | caps::CACHED_CASCADE
        | caps::COMPUTE
        | caps::PRESENT_TO_MEMORY;
    if tier >= Tier::CpuThrifty {
        caps |= caps::DISK_SPILL;
    }
    if tier == Tier::OutOfCore {
        caps |= caps::OUT_OF_CORE;
    }
    caps
}

/// What the soft-cpu backend would do at a tier, without creating a device.
/// This is what `reconlProbe` reports.
pub fn probe(tier: Tier) -> (u32, ShadowPlan) {
    let request = ShadowRequest::default();
    let plan = shadow_plan(tier, request.cascades, request.texel_budget_bytes, request.filter, caps_for(tier));
    (caps_for(tier), plan)
}

/// Where the arena would live for this configuration.
pub fn spill_dir(config: &SoftCpuConfig) -> PathBuf {
    config
        .spill_dir
        .clone()
        .unwrap_or_else(reconl_resource::spill::default_spill_dir)
}

/// The surface shader this backend gives a draw when the host asked for a plain
/// lit surface. Exported so the FFI layer and the tools do not have to guess.
pub fn default_surface<'a>(textured: bool, lit: bool, receives_shadow: bool) -> SurfaceShader<'a> {
    SurfaceShader {
        textured,
        lit,
        receives_shadow,
        ..SurfaceShader::unlit()
    }
}

