//! A minimal, deterministic PNG codec: just enough to write and read the
//! 8-bit RGBA golden images `reconl-diff` compares.
//!
//! `stored` deflate blocks only. A 64×64 RGBA frame compresses poorly and
//! diffs correctly all the same, and stored blocks are byte-exact by
//! construction — which is the property a golden test needs. No dependencies,
//! so the tool chain builds where the toolchain is.

/// CRC-32 (IEEE), table-driven.
fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *entry = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Adler-32, as zlib requires.
fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let mut crc_input = Vec::with_capacity(4 + body.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(body);
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// Writes an 8-bit RGBA PNG. Returns the encoded bytes.
///
/// The colour profile is left unspecified on purpose: goldens are compared
/// byte-for-byte against frames produced by this same library, so adding a
/// profile would be a second source of divergence, not a fix for one.
pub fn write_rgba8(width: u32, height: u32, pixels: &[u8]) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("a golden image needs non-zero dimensions".into());
    }
    let expected = width as usize * height as usize * 4;
    if pixels.len() != expected {
        return Err(format!("need {expected} bytes for {width}x{height} RGBA8, got {}", pixels.len()));
    }
    let mut out = Vec::with_capacity(expected + 1024);
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // colour type: RGBA
    ihdr.push(0); // compression: deflate
    ihdr.push(0); // filter: adaptive (we use none per row)
    ihdr.push(0); // interlace: none
    chunk(&mut out, b"IHDR", &ihdr);

    // Raw stream: one filter byte (0 = None) per scanline, then stored-deflate.
    let stride = width as usize * 4;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for y in 0..height as usize {
        raw.push(0);
        raw.extend_from_slice(&pixels[y * stride..(y + 1) * stride]);
    }

    let mut z = vec![0x78, 0x01]; // zlib header, no compression, fastest
    let max_block = 65_535;
    let mut at = 0;
    while at < raw.len() {
        let take = (raw.len() - at).min(max_block);
        let last = at + take == raw.len();
        z.push(if last { 1 } else { 0 });
        z.extend_from_slice(&(take as u16).to_le_bytes());
        z.extend_from_slice(&(!take as u16).to_le_bytes());
        z.extend_from_slice(&raw[at..at + take]);
        at += take;
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

/// One decoded RGBA8 image.
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// RGBA8, tightly packed, top-down.
    pub pixels: Vec<u8>,
}

/// Reads an 8-bit RGBA PNG. Filter type 0 only — the kind this crate writes.
pub fn read_rgba8(data: &[u8]) -> Result<Image, String> {
    const SIG: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];
    if data.len() < 8 || data[..8] != SIG {
        return Err("not a PNG (bad signature)".into());
    }
    let mut width = 0u32;
    let mut height = 0u32;
    let mut idat = Vec::new();
    let mut at = 8;
    while at + 8 <= data.len() {
        let len = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        let kind = &data[at + 4..at + 8];
        let body_end = at + 8 + len;
        if body_end + 4 > data.len() {
            return Err("truncated chunk".into());
        }
        let body = &data[at + 8..body_end];
        let stored_crc = u32::from_be_bytes(data[body_end..body_end + 4].try_into().unwrap());
        let mut crc_input = Vec::with_capacity(4 + body.len());
        crc_input.extend_from_slice(kind);
        crc_input.extend_from_slice(body);
        if crc32(&crc_input) != stored_crc {
            return Err(format!("chunk {} has a bad CRC", String::from_utf8_lossy(kind)));
        }
        match kind {
            b"IHDR" => {
                if body.len() != 13 {
                    return Err("IHDR must be 13 bytes".into());
                }
                width = u32::from_be_bytes(body[0..4].try_into().unwrap());
                height = u32::from_be_bytes(body[4..8].try_into().unwrap());
                if body[8] != 8 || body[9] != 6 {
                    return Err(format!("only 8-bit RGBA goldens are supported (depth {}, colour {})", body[8], body[9]));
                }
                if body[12] != 0 {
                    return Err("interlaced goldens are not supported".into());
                }
            }
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {} // ancillary chunks are legal and ignored
        }
        at = body_end + 4;
    }
    if width == 0 || height == 0 {
        return Err("empty image".into());
    }
    let raw = inflate_stored(&idat)?;
    let stride = width as usize * 4;
    let expected = (stride + 1) * height as usize;
    if raw.len() != expected {
        return Err(format!("raw stream is {} bytes, expected {expected}", raw.len()));
    }
    let mut pixels = Vec::with_capacity(stride * height as usize);
    for y in 0..height as usize {
        let row = &raw[y * (stride + 1)..(y + 1) * (stride + 1)];
        if row[0] != 0 {
            return Err(format!("filter type {} on row {y}; goldens are written with None", row[0]));
        }
        pixels.extend_from_slice(&row[1..1 + stride]);
    }
    Ok(Image { width, height, pixels })
}

/// Inflates a zlib stream made only of stored blocks — what [`write_rgba8`]
/// emits. Anything else is refused rather than half-decoded.
fn inflate_stored(z: &[u8]) -> Result<Vec<u8>, String> {
    if z.len() < 6 {
        return Err("zlib stream too short".into());
    }
    let cmf = z[0];
    if cmf & 0x0F != 8 {
        return Err("not a deflate stream".into());
    }
    if (u16::from(z[0]) << 8 | u16::from(z[1])) % 31 != 0 {
        return Err("zlib header check failed".into());
    }
    let mut at = 2;
    let mut out = Vec::new();
    loop {
        if at + 5 > z.len() {
            return Err("truncated stored block".into());
        }
        let header = z[at];
        if header & 0x06 != 0 {
            return Err("non-stored deflate block; goldens are written uncompressed".into());
        }
        // Compare in u16: complementing a widened integer checks against the
        // wrong width (a 64-bit !nlen never equals a 16-bit len).
        let len = u16::from_le_bytes(z[at + 1..at + 3].try_into().unwrap());
        let nlen = u16::from_le_bytes(z[at + 3..at + 5].try_into().unwrap());
        if len != !nlen {
            return Err("stored block length check failed".into());
        }
        let len = len as usize;
        at += 5;
        if at + len > z.len() {
            return Err("stored block overruns the stream".into());
        }
        out.extend_from_slice(&z[at..at + len]);
        at += len;
        if header & 1 != 0 {
            break;
        }
    }
    // Trailer: Adler-32 of the raw bytes. Verify when present; tolerate a
    // stream that ends exactly after the final block (not ours, but harmless).
    if at + 4 <= z.len() {
        let want = u32::from_be_bytes(z[at..at + 4].try_into().unwrap());
        let got = adler32(&out);
        if want != got {
            return Err("adler32 mismatch".into());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_gradient() {
        let mut px = Vec::new();
        for y in 0u32..16 {
            for x in 0u32..16 {
                px.extend_from_slice(&[x as u8 * 16, y as u8 * 16, 0x40, 0xFF]);
            }
        }
        let png = write_rgba8(16, 16, &px).unwrap();
        let img = read_rgba8(&png).unwrap();
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
        assert_eq!(img.pixels, px);
    }

    #[test]
    fn round_trips_across_block_boundaries() {
        // >64 KiB of pixel data forces multiple stored blocks.
        let px = vec![0xABu8; 5_000 * 4 * 4];
        let png = write_rgba8(5_000, 4, &px).unwrap();
        let img = read_rgba8(&png).unwrap();
        assert_eq!(img.pixels, px);
    }

    #[test]
    fn rejects_wrong_size_and_corruption() {
        assert!(write_rgba8(4, 4, &[0; 15]).is_err());
        let png = write_rgba8(2, 2, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).unwrap();
        let mut bad = png.clone();
        // Corrupt a byte inside the IDAT payload (after sig + IHDR chunk + IDAT header).
        let idat_data = 8 + 25 + 8;
        bad[idat_data] ^= 0xFF;
        // Corrupting pixel data or a chunk CRC must never decode silently.
        assert!(read_rgba8(&bad).is_err(), "corruption must not decode silently");
        let mut bad_crc = png.clone();
        let n = bad_crc.len();
        bad_crc[n - 1] ^= 0xFF; // IEND CRC
        assert!(read_rgba8(&bad_crc).is_err(), "chunk CRC must be verified");
        assert!(read_rgba8(b"not a png at all").is_err());
    }
}
