//! The out-of-core frame path, in one place.
//!
//! A frame whose own colour and depth targets do not fit the RAM cap is *cut into
//! horizontal bands*: one band target stays resident, every band is written to its
//! own entry in the disk arena, and present reads the bands back in row order. What
//! the host sees is the whole frame - the same pixels, the same checksum, the same
//! readback conversion - and the bytes it cost are counted rather than claimed.
//!
//! The module owns four things and nothing else owns any of them:
//!
//! * the band geometry - how the frame's rows divide into bands, and the clip-space
//!   remap that puts a band's rows where the frame's rows are (`band_projection`);
//! * the arena keying - one key per band, carrying the frame's serial, so a band
//!   left behind by an earlier frame can never be presented as this one;
//! * the band's own storage - packed colour then depth, written once and read back
//!   by both readback paths;
//! * the decision that a frame streams at all - the RAM cap refused the frame's
//!   targets, the tier is out-of-core, and the host allowed a disk spill.
//!
//! It is a child module of the backend rather than a crate of its own because a
//! band is drawn by the *same* rasteriser, from the *same* draw list, through the
//! *same* pipeline as a resident frame: only where the pixels land changes.

use reconl_contract::FrameInput;
use reconl_core::alloc::HostVec;
use reconl_core::budget::Reservation;
use reconl_core::error::{Code, Error, Result};
use reconl_core::tier::caps;
use reconl_raster::math;
use reconl_raster::simd;
use reconl_raster::{
    checksum_bytes_with, rendered_viewport, DrawItem, RasterStats, Rasterizer, Target, CHECKSUM_SEED,
};
use reconl_resource::spill::{Hit, SpillArena, RECORD_HEADER_BYTES};

use super::{accumulate_raster, SoftCpuDevice};

impl SoftCpuDevice {
    /// Whether this device streams a frame's own targets through the arena when
    /// the RAM cap cannot hold them: the out-of-core tier, on a device whose host
    /// allowed a disk spill.
    pub(super) fn streams_targets(&self) -> bool {
        (self.caps() & caps::OUT_OF_CORE) != 0 && self.budget.caps().allow_disk_spill
    }

    /// Reserves the frame's own colour and depth targets, if the RAM cap can hold
    /// them. `Ok(false)` is a refused reservation rather than an error: the
    /// caller decides whether the frame streams through the arena instead.
    fn ensure_targets(&mut self, width: u32, height: u32) -> Result<bool> {
        if width == 0 || height == 0 {
            return Err(Error::new(Code::InvalidArgument, "zero-sized frame"));
        }
        if self.stream.is_none()
            && self.color.width == width
            && self.color.height == height
            && self.color.depth.is_some()
        {
            return Ok(true);
        }
        let bytes = target_bytes(width, height);
        let reservation = match self.budget.reserve_ram(bytes) {
            Ok(r) => r,
            Err(e) => {
                if self.streams_targets() {
                    // A refusal here is not a fail-safe: a frame the RAM cap
                    // cannot hold is the case the disk tier exists for, and the
                    // caller streams it. A frame that cannot *stream* either is
                    // the failure, and [`Self::start_stream`] counts that one.
                    return Ok(false);
                }
                self.counters.safe_path_events += 1;
                return Err(e);
            }
        };
        let target = Target::new_color(self.alloc, width, height)?.with_depth()?;
        self.color = target;
        self.color_reservation = Some(reservation);
        Ok(true)
    }

    /// Makes the frame's own targets the right size for `width x height`:
    /// resident when the RAM cap can hold them, and streamed through the disk
    /// arena when it cannot and this tier has one.
    ///
    /// This is the whole decision, in one place, so `prepare_frame` and `render`
    /// cannot disagree about which of the two a frame is using - the frame that
    /// reserves is the frame that renders.
    pub(super) fn ensure_frame_targets(&mut self, width: u32, height: u32) -> Result<()> {
        if self.stream.as_ref().map(|s| (s.width, s.height)) == Some((width, height)) {
            return Ok(());
        }
        // A frame that fits again drops the bands of the frame before it: they
        // hold the frame that streamed, and the frame that just arrived is not
        // in them.
        self.stream = None;
        if self.ensure_targets(width, height)? {
            return Ok(());
        }
        self.start_stream(width, height)
    }

    /// Starts streaming this frame's colour and depth through the arena.
    ///
    /// The frame is cut into horizontal bands and one band target is kept: what
    /// is resident is a band, and what is not is on disk, so a frame larger than
    /// the RAM cap renders instead of being refused. That is T4's whole promise,
    /// and it is a *measurement* - every band written and read back is counted in
    /// `FrameNumbers::spill_io_bytes`.
    pub(super) fn start_stream(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(Error::new(Code::InvalidArgument, "zero-sized frame"));
        }
        // The arena first, and no borrow of it is held across the allocation
        // below: without one, the refusal the caller already saw stands, with the
        // reason it could not be streamed attached to it.
        if self.ensure_arena().is_none() {
            let reason = match self.arena_error.as_ref() {
                Some(e) => e.to_string(),
                None => "the host did not allow a disk spill".to_string(),
            };
            self.counters.safe_path_events += 1;
            return Err(Error::new(
                Code::BudgetExceeded,
                "the frame does not fit the RAM cap and cannot be streamed through the arena",
            )
            .with_context(&reason));
        }

        // The tallest band the RAM cap can hold, found by halving the frame: a
        // cap too tight for the frame costs more bands and more disk traffic, not
        // a refused frame.
        let mut band_rows = height;
        let band_reservation = loop {
            match self.budget.reserve_ram(target_bytes(width, band_rows)) {
                Ok(r) => break r,
                Err(e) => {
                    if band_rows <= 1 {
                        self.counters.safe_path_events += 1;
                        return Err(e);
                    }
                    band_rows = (band_rows / 2).max(1);
                }
            }
        };

        // One band more than the frame's rows divide into: a pass whose viewport
        // cuts a band in two - the frame is cut at the viewport's edge, and that
        // edge is only known when the frame is rendered - needs exactly one more,
        // and every key is allocated once, here.
        let bands = (u64::from(height) + u64::from(band_rows) - 1) / u64::from(band_rows) + 1;

        // Every band of the frame is live at once: present reads them back in row
        // order after the last one is written, so a frame that does not fit the
        // host's disk cap cannot be streamed by writing part of it. Make the room
        // now - before a byte is written - and refuse with the requirement if the
        // cap cannot hold it. Writing past the cap is the one answer that is not
        // on this list.
        let frame_disk_bytes = target_bytes(width, height) + bands * RECORD_HEADER_BYTES;
        let room = match self.arena.as_mut() {
            Some(arena) => arena.make_room(frame_disk_bytes),
            None => Ok(()),
        };
        if let Err(e) = room {
            self.counters.safe_path_events += 1;
            return Err(e.with_context(&format!(
                "a {width}x{height} frame streams as {bands} bands of {band_rows} rows and needs \
                 {frame_disk_bytes} bytes on disk"
            )));
        }

        let band_bytes = (width as usize) * (band_rows as usize) * PIXEL_BYTES as usize;
        let band = Target::new_color(self.alloc, width, band_rows)?.with_depth()?;
        self.stream_serial += 1;
        let mut keys = HostVec::with_capacity(self.alloc, bands as usize)?;
        for index in 0..bands as u32 {
            keys.push(band_key(self.stream_serial, index))?;
        }
        let mut payload = HostVec::with_capacity(self.alloc, band_bytes)?;
        payload.resize_with(band_bytes, || 0u8)?;
        let draws = HostVec::with_capacity(self.alloc, self.colors.capacity())?;

        // The frame's own colour target goes with the frame: what is resident is
        // the band, and keeping a full-size target would hold in RAM the frame
        // that did not fit.
        self.color = Target::new_color(self.alloc, 0, 0)?;
        self.color_reservation = None;
        self.stream = Some(FrameStream {
            width,
            height,
            band_rows,
            // The whole frame until a pass confines it.
            viewport_rows: height,
            band,
            band_reservation,
            keys,
            serial: self.stream_serial,
            payload,
            draws,
            scratch: Vec::new(),
        });
        Ok(())
    }

    pub fn frame_size(&self) -> (u32, u32) {
        match self.stream.as_ref() {
            Some(stream) => (stream.width, stream.height),
            None => (self.color.width, self.color.height),
        }
    }

    /// Renders the colour pass band by band into the arena, and returns the bytes
    /// moved through it and the frame's colour checksum as it was written out.
    ///
    /// The band target, the arena and the colour list are three different fields
    /// of this device, so one method owns the borrowing rather than each caller
    /// doing it: what the band loop needs is passed to `stream_bands` as the
    /// pieces it is.
    pub(super) fn stream_bands(&mut self, input: &FrameInput<'_>, draw: bool, stats: &mut RasterStats) -> Result<(u64, u64)> {
        match (self.stream.as_mut(), self.arena.as_mut()) {
            (Some(stream), Some(arena)) => {
                stream_bands(stream, arena, &mut self.raster, &self.colors, input, draw, stats)
            }
            _ => Err(Error::new(
                Code::NotReady,
                "a streamed frame needs the arena it was opened with",
            )),
        }
    }
}

// ---------------------------------------------------------------- streaming
// The T4 path for a frame the RAM cap cannot hold. A frame is cut into rows of
// bands; each band is drawn into the one resident target the cap can hold and
// written to its own arena entry, and present reads the bands back in row order.
// What the host sees is the whole frame - and the bytes it cost are counted.

/// Bytes per pixel of the frame's own targets: colour (four `f32` channels) and
/// depth (one).
///
/// One number for the frame's charge against the RAM cap, a band's charge, and
/// what an arena entry holds per pixel, so the accounting and the storage cannot
/// disagree about what a frame costs.
const PIXEL_BYTES: u64 = 4 * 4 + 4;

/// What the frame's own colour and depth targets cost against the RAM cap.
///
/// The same function charges a band, because a band is the frame's targets cut
/// down to `rows` of the frame's rows: reserving a band the way the frame is
/// reserved is what makes a cap too tight for the frame cost more bands rather
/// than a refused frame.
pub(super) fn target_bytes(width: u32, height: u32) -> u64 {
    u64::from(width) * u64::from(height) * PIXEL_BYTES
}

/// A frame whose own colour and depth do not fit the RAM cap, streamed through
/// the disk arena.
///
/// This is what T4 is for: a frame larger than RAM renders by being cut into
/// horizontal bands. What is resident is one band target; what is not is in the
/// arena, and `present` reads it back in row order. Every buffer the frame needs
/// is allocated once, here, so a streamed frame allocates nothing while it runs.
pub(super) struct FrameStream {
    width: u32,
    height: u32,
    /// The frame's rows per band: the tallest band the RAM cap can hold.
    band_rows: u32,
    /// The rows the last pass that wrote this frame was confined to - its
    /// viewport - which is where the band cuts fall. Recorded rather than
    /// recomputed, because the reader of a band has no viewport to recompute it
    /// from, and a band row range that differed between the write and the read
    /// would be a row range presented from the wrong place.
    viewport_rows: u32,
    /// The one resident target every band is drawn into.
    band: Target,
    /// What that target costs against the RAM cap, held for the frame's life.
    band_reservation: Reservation,
    /// One arena key per band, each carrying the serial of the frame that wrote
    /// it, so a band left by an earlier frame cannot be presented as this one.
    keys: HostVec<u64>,
    /// The frame's serial: the arena keys of the frame before it are derived from
    /// it, which is how the rows a frame does not clear are carried forward.
    serial: u64,
    /// The band in flight - colour bytes then depth bytes - read and written
    /// through this, so a band costs no allocation per frame.
    payload: HostVec<u8>,
    /// The frame's colour list with one band's projection applied.
    draws: HostVec<DrawItem<'static>>,
    /// Scratch for the arena's reads, which take a `Vec` their caller owns.
    scratch: Vec<u8>,
}

impl FrameStream {
    /// The rows of the band that starts at `top`.
    ///
    /// The frame is cut at the viewport's edge, so no band straddles it: a band is
    /// wholly inside the pass or wholly outside it, and the one outside is written
    /// cleared - which is what the host asked for by asking for a sub-rect.
    /// What this stream holds resident against the RAM cap: the one band target
    /// every band of the frame is drawn into. The rest of the frame is the
    /// arena's, counted by the budget that opened it.
    pub(super) fn resident_bytes(&self) -> u64 {
        self.band_reservation.bytes()
    }

    fn rows_at(&self, top: u32) -> u32 {
        let limit = if top < self.viewport_rows {
            self.viewport_rows.min(self.height)
        } else {
            self.height
        };
        self.band_rows.min(limit - top).max(1)
    }
}

/// The error for a frame whose band walk ran past the keys it allocated. The walk
/// is bounded by the frame's rows and every key is allocated for the rows they
/// divide into plus the one a viewport cut can add, so this is a bug in the two
/// agreeing - and a wrong row range presented from a key that is not this frame's
/// is exactly what must not happen quietly.
fn short_bands(stream: &FrameStream) -> Error {
    Error::new(
        Code::NotSupported,
        "a frame taller than the band keys it allocated: refusing to present the wrong rows",
    )
    .with_context(&format!(
        "frame {}x{}, band {} rows, {} keys, viewport {} rows",
        stream.width,
        stream.height,
        stream.band_rows,
        stream.keys.len(),
        stream.viewport_rows
    ))
}

/// A band's key in the arena.
///
/// The frame's serial is part of it, and has to be: a band is a row range of one
/// frame, so a stale entry under the same key would be a wrong row range
/// presented as this frame's - the one mix-up the arena's checksum cannot catch,
/// because a band's bytes are whole and valid whichever frame wrote them.
fn band_key(serial: u64, index: u32) -> u64 {
    let hash = checksum_bytes_with(CHECKSUM_SEED, b"reconl-band");
    let hash = checksum_bytes_with(hash, &serial.to_le_bytes());
    checksum_bytes_with(hash, &index.to_le_bytes())
}

/// The projection a band renders the frame through.
///
/// The band's clip space is remapped so the frame's rows `top .. top + rows` land
/// on the band's own `0 .. rows`: the band rasterises exactly the pixels the frame
/// would have put there, which is what lets a frame be cut into bands without a
/// seam. It is applied to clip space - `y' = (y - c*w)/s` - so it composes with
/// whatever projection the host supplied rather than replacing it, and `w` is left
/// alone so the clip and the perspective divide still agree.
///
/// `render_height` is the height the rasteriser maps clip space onto, which is the
/// pass's *viewport*, not the frame: a host that asked for a sub-rect gets the same
/// patch from a band as from a whole-frame target, and the default (whole-frame)
/// viewport is the case where the two are the same number.
///
/// The remap is exact in real arithmetic, so a band has no seam - but a band's
/// vertices reach the rasteriser's 1/256-px snap through one more multiply and one
/// more divide than a whole-frame pass puts them through, and where that rounding
/// lands a vertex on the neighbouring subpixel unit the coverage of the pixels
/// within one unit of that edge flips. It is only *visible* where the surfaces
/// meeting at that edge differ in colour, which is why a streamed frame reproduces
/// a resident one exactly with no shadow term and differs on a couple of pixels at
/// a shadow edge - and why the difference moves with the band height rather than
/// with the frame. Measured at 512x512 through `reconl-bench --png` and
/// `reconl-diff` (see `a_frame_the_ram_cap_cannot_hold_streams_through_the_arena`
/// for the numbers). An exact crop would have to be expressed as an integer row
/// origin applied after the perspective divide, which the rasteriser's clip-to-
/// screen transform has no way to carry today; this projection is what a band can
/// express with the interface that exists.
fn band_projection(render_height: u32, rows: u32, top: u32) -> math::Mat4 {
    let (h, r, t) = (render_height as f32, rows as f32, top as f32);
    // Clip y for the first row of this band, and the band's scale: the band's row
    // 0 is row `top` of the pass, and one band row is `rows/render_height` of one
    // of its rows.
    let (c, s) = (1.0 - (2.0 * t + r) / h, r / h);
    let mut m = math::IDENTITY;
    m[5] = 1.0 / s;
    m[13] = -c / s;
    m
}

/// Renders the colour pass one band at a time into the arena.
///
/// `draw` says whether the pass draws at all: a frame the classifier skipped
/// still has to be *something* in the arena, and what it holds is the cleared
/// bands a resident target would have been left holding.
///
/// Returns the bytes moved through the arena and the checksum of the frame's
/// colour as it was written out - the same number `checksum_f32` gives for a
/// resident frame, because it is the same fold over the same bytes in the same
/// row order.
///
/// Takes its pieces rather than `&mut self`, as `fill_cascade` does: the band, the
/// arena, the rasteriser and the colour list are four fields of one device, and
/// the caller owns the borrowing.
#[allow(clippy::too_many_arguments)]
fn stream_bands(
    stream: &mut FrameStream,
    arena: &mut SpillArena,
    raster: &mut Rasterizer,
    colors: &HostVec<DrawItem<'static>>,
    input: &FrameInput<'_>,
    draw: bool,
    stats: &mut RasterStats,
) -> Result<(u64, u64)> {
    // Room for the whole frame before the first band goes in: a `put` evicts when
    // the arena is at its cap, and an eviction in the middle of a frame would
    // throw away a band this frame has not read back yet. The host's disk cap is
    // the arena's, so a frame that cannot fit is refused here, naming what it
    // needed, rather than written past the cap.
    let frame_bytes = target_bytes(stream.width, stream.height) + stream.keys.len() as u64 * RECORD_HEADER_BYTES;
    arena.make_room(frame_bytes)?;

    // The host's viewport, in the band's pixels: a band is the frame's full width,
    // so only the rows are at stake. The frame is cut at the viewport's edge
    // below, so no band straddles it - a band is wholly inside the pass or wholly
    // outside it, and one outside it is written cleared, which is what the host
    // asked for by asking for a sub-rect.
    let view = rendered_viewport(
        input.viewport,
        (input.width, input.height),
        (stream.width, stream.height),
    );

    // Where the bands are cut is a property of the frame, not of the pass that
    // happens to read it back: recorded here, so present and a reprojection cut the
    // same rows this pass wrote.
    stream.viewport_rows = view.1.min(stream.height);

    let mut io = 0u64;
    let mut hash = CHECKSUM_SEED;
    let mut top = 0u32;
    let mut index = 0usize;
    while top < stream.height {
        let in_view = top < view.1;
        let rows = stream.rows_at(top);
        let key = match stream.keys.get(index) {
            Some(key) => *key,
            None => return Err(short_bands(stream)),
        };
        seed_band(stream, arena, input, index, rows, &mut io)?;

        if draw && in_view {
            // The band's own projection, applied to every draw: the frame's colour
            // list is the same list, warped, so a band shades exactly what the
            // frame would have shaded in those rows. The height it is built for is
            // the viewport's - the rect this pass maps clip space onto - because
            // that is what puts the band's rows where the frame's rows are.
            let projection = band_projection(view.1, rows, top);
            stream.draws.clear();
            for item in colors.as_slice() {
                let mut copy = *item;
                copy.transform = math::mul(&projection, &item.transform);
                stream.draws.push(copy)?;
            }
            let rendered = raster.rasterize(&mut stream.band, stream.draws.as_slice(), (view.0, rows))?;
            accumulate_raster(stats, &rendered);
        }

        // Pack the band - colour bytes then depth, the layout `decode_band` reads
        // back - and put it where present will find it.
        let pixels = (stream.width as usize) * (rows as usize);
        let color_bytes = pixels * 16;
        {
            let color = stream.band.color_slice().unwrap_or(&[]);
            let depth = stream.band.depth_slice().unwrap_or(&[]);
            let payload = stream.payload.as_mut_slice();
            for i in 0..pixels * 4 {
                payload[i * 4..i * 4 + 4].copy_from_slice(&color[i].to_bits().to_le_bytes());
            }
            for i in 0..pixels {
                let at = color_bytes + i * 4;
                payload[at..at + 4].copy_from_slice(&depth[i].to_bits().to_le_bytes());
            }
        }
        let record = &stream.payload.as_slice()[..color_bytes + pixels * 4];
        hash = checksum_bytes_with(hash, &record[..color_bytes]);
        io += arena.put(key, record)?;
        top += rows;
        index += 1;
    }
    Ok((io, hash))
}

/// Puts a band into the state the frame's own target would have been in.
///
/// A resident target keeps the last frame's pixels where the frame asked for no
/// clear; a band lives on disk, so it reads its own previous entry back - under
/// the serial before this frame's, which is where the last streamed frame left
/// these rows. A band with nothing to read starts zeroed, which is what a fresh
/// target holds.
fn seed_band(
    stream: &mut FrameStream,
    arena: &mut SpillArena,
    input: &FrameInput<'_>,
    index: usize,
    rows: u32,
    io: &mut u64,
) -> Result<()> {
    if input.clear_color_enabled {
        stream.band.clear_color(input.clear_color);
    }
    if input.clear_depth_enabled {
        stream.band.clear_depth(input.clear_depth);
    }
    if input.clear_color_enabled && input.clear_depth_enabled {
        return Ok(());
    }
    let expected = (stream.width as usize) * (rows as usize) * PIXEL_BYTES as usize;
    let previous = band_key(stream.serial.wrapping_sub(1), index as u32);
    let mut bytes = core::mem::take(&mut stream.scratch);
    let hit = arena.get(previous, &mut bytes);
    if hit == Hit::Fresh && bytes.len() == expected {
        *io += bytes.len() as u64;
        decode_band(stream, &bytes, rows, !input.clear_color_enabled, !input.clear_depth_enabled);
    } else if !input.clear_color_enabled || !input.clear_depth_enabled {
        if !input.clear_color_enabled {
            stream.band.clear_color([0.0, 0.0, 0.0, 0.0]);
        }
        if !input.clear_depth_enabled {
            stream.band.clear_depth(0.0);
        }
    }
    stream.scratch = bytes;
    Ok(())
}

/// Decodes one arena entry back into the band target: colour first, then depth, in
/// the layout `stream_bands` wrote. `want_color` and `want_depth` select the halves
/// the caller is asking for - a present needs the colour, a reprojection the depth.
fn decode_band(stream: &mut FrameStream, payload: &[u8], rows: u32, want_color: bool, want_depth: bool) {
    let pixels = (stream.width as usize) * (rows as usize);
    let color_bytes = pixels * 16;
    let word = |at: usize| -> f32 {
        let mut raw = [0u8; 4];
        raw.copy_from_slice(&payload[at..at + 4]);
        f32::from_bits(u32::from_le_bytes(raw))
    };
    if want_color {
        if let Some(color) = stream.band.color_slice_mut() {
            for i in 0..(pixels * 4).min(color.len()) {
                color[i] = word(i * 4);
            }
        }
    }
    if want_depth {
        if let Some(depth) = stream.band.depth_slice_mut() {
            for i in 0..pixels.min(depth.len()) {
                depth[i] = word(color_bytes + i * 4);
            }
        }
    }
}

/// Reads one band of the frame back out of the arena into the band target.
///
/// A band that is no longer there is not presented as if it were: the frame can be
/// rendered again, but a row range the arena cannot produce is an error rather
/// than a wrong pixel.
fn load_band(stream: &mut FrameStream, arena: &mut SpillArena, index: usize, rows: u32) -> Result<()> {
    let expected = (stream.width as usize) * (rows as usize) * PIXEL_BYTES as usize;
    let key = match stream.keys.get(index) {
        Some(key) => *key,
        None => return Err(short_bands(stream)),
    };
    let mut bytes = core::mem::take(&mut stream.scratch);
    let hit = arena.get(key, &mut bytes);
    let held = hit == Hit::Fresh && bytes.len() == expected;
    if held {
        decode_band(stream, &bytes, rows, true, true);
    }
    stream.scratch = bytes;
    if held {
        Ok(())
    } else {
        Err(Error::new(
            Code::NotReady,
            "a band of this frame is not in the arena any more (evicted or damaged): render it again to present it",
        ))
    }
}

/// Presents a streamed frame: every band is read back from the arena, in row
/// order, and converted into the host's rows by the same exact conversion a
/// resident target goes through.
pub(super) fn read_bands_into(
    stream: &mut FrameStream,
    arena: &mut SpillArena,
    out: &mut [u8],
    pitch: u32,
    flip: bool,
) -> Result<()> {
    let row_bytes = stream.width as usize * 4;
    let pitch = if pitch == 0 { row_bytes } else { pitch as usize };
    let needed = (stream.height as usize - 1).saturating_mul(pitch) + row_bytes;
    if pitch < row_bytes || out.len() < needed {
        return Err(Error::new(Code::InvalidArgument, "readback buffer is too small"));
    }
    let mut top = 0u32;
    let mut index = 0usize;
    while top < stream.height {
        let rows = stream.rows_at(top);
        load_band(stream, arena, index, rows)?;
        let color = stream.band.color_slice().unwrap_or(&[]);
        for row in 0..rows as usize {
            let frame_row = if flip {
                (stream.height as usize - 1) - (top as usize + row)
            } else {
                top as usize + row
            };
            let at = frame_row * pitch;
            simd::rgba_f32_to_unorm8(&color[row * row_bytes..(row + 1) * row_bytes], &mut out[at..at + row_bytes]);
        }
        top += rows;
        index += 1;
    }
    Ok(())
}

/// Copies a streamed frame's depth out of the arena into `out`, band by band and
/// in row order: the same values a resident target hands back.
pub(super) fn read_bands_depth(stream: &mut FrameStream, arena: &mut SpillArena, out: &mut [f32]) -> Result<()> {
    let width = stream.width as usize;
    if out.len() < width * stream.height as usize {
        return Err(Error::new(Code::InvalidArgument, "the depth buffer is too small for the frame"));
    }
    let mut top = 0u32;
    let mut index = 0usize;
    while top < stream.height {
        let rows = stream.rows_at(top);
        load_band(stream, arena, index, rows)?;
        let depth = stream.band.depth_slice().unwrap_or(&[]);
        let count = width * rows as usize;
        let at = top as usize * width;
        out[at..at + count].copy_from_slice(&depth[..count]);
        top += rows;
        index += 1;
    }
    Ok(())
}
