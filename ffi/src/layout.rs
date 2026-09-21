//! How bytes move between a host's buffer and a frame or a texture.
//!
//! Every readback and every upload moves `width * 4`-byte rows, `pitch` bytes
//! apart, and a pitch narrower than a row overlaps them: the bytes a host named
//! as row `r` are partly the bytes it named as row `r + 1`, so what it reads back
//! - or hands in - is not the picture it asked for. That is the one thing this
//! module exists to prevent, and it is prevented in one place rather than at each
//! of the eight call sites (see [`host_row_layout`]).
//!
//! The two functions are deliberately separate and deliberately unlike each
//! other in strictness: `host_row_layout` *judges* a host's numbers before a byte
//! is written, and `lay_out_rows` *lays out* rows it is told are already valid.
//! The second does arithmetic; the first owns the rule.

use reconl_core::error::{Code, Result};
use reconl_core::err;

/// The one place a host's row layout is judged.
///
/// Every readback and every upload moves `width * 4`-byte rows between host
/// memory and a frame or a texture, `pitch` bytes apart. A pitch narrower than
/// a row overlaps the rows, so the bytes a host named as row `r` are partly the
/// bytes it named as row `r + 1`: what it reads back, or hands in, is not the
/// picture it asked for. That is refused here - with the code the header
/// documents for it - rather than laid out wrongly, and a buffer too small to
/// hold the rows asked for is refused the same way.
///
/// One rule, one place: the presents, the generated present, the texture read
/// and the texture write all take a host pitch, and all of them come through
/// here, so a host cannot be refused on one path and quietly given overlapping
/// rows on another. `0` means tightly packed, which is this header's idiom for
/// "the obvious default".
///
/// Returns the pitch, in bytes, to lay the rows out with.
pub(crate) fn host_row_layout(width: u32, rows: u32, buffer: u64, pitch: u32) -> Result<u32> {
    let row_bytes = width as u64 * 4;
    let pitch = if pitch == 0 { row_bytes } else { u64::from(pitch) };
    if pitch < row_bytes {
        return err!(
            Code::InvalidArgument,
            "a row pitch of {} bytes is narrower than the {} bytes one {}x4 row needs",
            pitch,
            row_bytes,
            width
        );
    }
    let needed = u64::from(rows.saturating_sub(1))
        .saturating_mul(pitch)
        .saturating_add(row_bytes);
    if buffer < needed {
        return err!(
            Code::InvalidArgument,
            "the buffer is too small for {} rows {} bytes apart: it needs {}",
            rows,
            pitch,
            needed
        );
    }
    Ok(pitch as u32)
}

/// Laid tightly packed rows into a destination with a row pitch and optionally a
/// flip: the layout rule a present documents, applied to bytes the library has
/// already converted - the backend that owns the pixel format produced them.
///
/// `pitch` must be one `host_row_layout` accepted for this width, which is what
/// keeps the rows from overlapping: the check belongs to that one function, not
/// to this copy.
pub(crate) fn lay_out_rows(out: &mut [u8], src: &[u8], rows: usize, row_bytes: usize, pitch: usize, flip: bool) {
    for row in 0..rows {
        let source = if flip { rows - 1 - row } else { row };
        let at = row * pitch;
        let from = source * row_bytes;
        out[at..at + row_bytes].copy_from_slice(&src[from..from + row_bytes]);
    }
}
