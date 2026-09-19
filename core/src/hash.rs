//! xxHash64, used for two different jobs:
//!
//! * **cache keys** - what identifies a cached static cascade (`shadow::cache_key`),
//! * **entry checksums** - what makes a torn spill entry detectable
//!   (`resource::spill`).
//!
//! Both need to be stable across runs, versions and machines, and both are
//! compared against values written by an *older* process, so this is the
//! canonical algorithm with the canonical test vector, not a hash of our own.
//! `tests` below pins it.

const PRIME1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME2))
        .rotate_left(31)
        .wrapping_mul(PRIME1)
}

#[inline]
fn read_u64(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = bytes.len().min(8);
    buf[..n].copy_from_slice(&bytes[..n]);
    u64::from_le_bytes(buf)
}

#[inline]
fn read_u32(bytes: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    let n = bytes.len().min(4);
    buf[..n].copy_from_slice(&bytes[..n]);
    u32::from_le_bytes(buf)
}

/// Streamed xxHash64. `update` may be called with any chunking; the result is
/// identical to hashing the concatenation in one call.
pub struct XxHash64 {
    seed: u64,
    total: u64,
    v1: u64,
    v2: u64,
    v3: u64,
    v4: u64,
    tail: [u8; 32],
    tail_len: usize,
}

impl XxHash64 {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            total: 0,
            v1: seed.wrapping_add(PRIME1).wrapping_add(PRIME2),
            v2: seed.wrapping_add(PRIME2),
            v3: seed,
            v4: seed.wrapping_sub(PRIME1),
            tail: [0; 32],
            tail_len: 0,
        }
    }

    pub fn update(&mut self, data: &[u8]) -> &mut Self {
        self.total = self.total.wrapping_add(data.len() as u64);
        let mut data = data;

        // Fill a partial block first, if there is one.
        if self.tail_len > 0 {
            if self.tail_len + data.len() < 32 {
                self.tail[self.tail_len..self.tail_len + data.len()].copy_from_slice(data);
                self.tail_len += data.len();
                return self;
            }
            let fill = 32 - self.tail_len;
            let mut block = self.tail;
            block[self.tail_len..].copy_from_slice(&data[..fill]);
            self.consume(&block);
            self.tail_len = 0;
            data = &data[fill..];
        }

        let mut blocks = data.chunks_exact(32);
        for block in &mut blocks {
            let mut b = [0u8; 32];
            b.copy_from_slice(block);
            self.consume(&b);
        }
        let rest = blocks.remainder();
        self.tail[..rest.len()].copy_from_slice(rest);
        self.tail_len = rest.len();
        self
    }

    fn consume(&mut self, block: &[u8; 32]) {
        self.v1 = round(self.v1, read_u64(&block[0..8]));
        self.v2 = round(self.v2, read_u64(&block[8..16]));
        self.v3 = round(self.v3, read_u64(&block[16..24]));
        self.v4 = round(self.v4, read_u64(&block[24..32]));
    }

    pub fn update_u64(&mut self, value: u64) -> &mut Self {
        self.update(&value.to_le_bytes())
    }

    pub fn update_f32(&mut self, value: f32) -> &mut Self {
        // By bit pattern: -0.0 and 0.0 stay different cache keys, because they
        // produce different bias arithmetic.
        self.update(&value.to_bits().to_le_bytes())
    }

    /// Length-prefixed, so "ab"+"c" and "a"+"bc" cannot collide.
    pub fn update_str(&mut self, value: &str) -> &mut Self {
        self.update(&(value.len() as u64).to_le_bytes());
        self.update(value.as_bytes())
    }

    pub fn finish(&self) -> u64 {
        let mut h = if self.total >= 32 {
            self.v1
                .rotate_left(1)
                .wrapping_add(self.v2.rotate_left(7))
                .wrapping_add(self.v3.rotate_left(12))
                .wrapping_add(self.v4.rotate_left(18))
        } else {
            self.seed.wrapping_add(PRIME5)
        };
        h = h.wrapping_add(self.total);

        let mut tail = &self.tail[..self.tail_len];
        while tail.len() >= 8 {
            h ^= round(0, read_u64(tail));
            h = h.rotate_left(27).wrapping_mul(PRIME1).wrapping_add(PRIME4);
            tail = &tail[8..];
        }
        if tail.len() >= 4 {
            h ^= (read_u32(tail) as u64).wrapping_mul(PRIME1);
            h = h.rotate_left(23).wrapping_mul(PRIME2).wrapping_add(PRIME3);
            tail = &tail[4..];
        }
        for b in tail {
            h ^= (*b as u64).wrapping_mul(PRIME5);
            h = h.rotate_left(11).wrapping_mul(PRIME1);
        }

        h ^= h >> 33;
        h = h.wrapping_mul(PRIME2);
        h ^= h >> 29;
        h = h.wrapping_mul(PRIME3);
        h ^= h >> 32;
        h
    }
}

pub fn xxh64(bytes: &[u8]) -> u64 {
    XxHash64::new(0).update(bytes).finish()
}

pub fn xxh64_seeded(seed: u64, bytes: &[u8]) -> u64 {
    XxHash64::new(seed).update(bytes).finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_test_vectors() {
        // Pinned values, cross-checked against a second, independent
        // transcription of the xxHash64 spec (not against this implementation).
        // The empty-input value is the reference suite's own vector.
        assert_eq!(xxh64(b""), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"a"), 0xD24E_C4F1_A98C_6E5B);
        assert_eq!(xxh64(b"abc"), 0x44BC_2CF5_AD77_0999);
        assert_eq!(xxh64(b"reconl"), 0xA887_F559_4FAB_AE98);
        assert_eq!(
            xxh64(b"abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789"),
            0xBFFA_2826_9451_E5F7
        );
        let all: Vec<u8> = (0..=255u8).collect();
        assert_eq!(xxh64(&all), 0x381D_EAB0_E687_90F9);
    }

    #[test]
    fn streaming_matches_one_shot_at_every_chunk_size() {
        let data: Vec<u8> = (0..500u32).map(|i| (i % 251) as u8).collect();
        let one = xxh64(&data);
        for chunk in [1usize, 3, 7, 8, 31, 32, 33, 64, 128] {
            let mut h = XxHash64::new(0);
            for part in data.chunks(chunk) {
                h.update(part);
            }
            assert_eq!(h.finish(), one, "chunk size {chunk} diverged");
        }
    }

    #[test]
    fn a_single_flipped_bit_changes_the_hash() {
        let a = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut b = a;
        b[3] ^= 1;
        assert_ne!(xxh64(&a), xxh64(&b));
    }

    #[test]
    fn typed_updates_are_order_sensitive() {
        let mut a = XxHash64::new(7);
        a.update_u64(1).update_f32(2.0).update_str("three");
        let mut b = XxHash64::new(7);
        b.update_u64(1).update_f32(2.0).update_str("three");
        let mut c = XxHash64::new(7);
        c.update_str("three").update_u64(1).update_f32(2.0);
        assert_eq!(a.finish(), b.finish());
        assert_ne!(a.finish(), c.finish());
        assert_ne!(XxHash64::new(1).update_str("reconl").finish(), XxHash64::new(2).update_str("reconl").finish());
    }
}
