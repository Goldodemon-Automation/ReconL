//! Names for the ABI's enums and the fixed-size strings it writes back.
//!
//! Backend, tier and result names come from the ABI itself (`reconlBackendName`
//! and friends) rather than a table here, because the library is the thing that
//! decides what a code means and a tool that invents its own vocabulary is a
//! second source of truth. Only the bitfields, which the ABI exposes as bits
//! with no accessor, are spelled out - and their spelling matches the header's.

use crate::cstr_of;
use reconl::abi;

/// The capability bits, in the header's order, with the header's names.
pub const CAPS: [(u32, &str); 15] = [
    (abi::caps::TEXTURES, "textures"),
    (abi::caps::MIPMAPS, "mipmaps"),
    (abi::caps::SHADOWS, "shadows"),
    (abi::caps::PCF_5X5, "pcf5x5"),
    (abi::caps::PCSS_LITE, "pcss-lite"),
    (abi::caps::DISK_SPILL, "disk-spill"),
    (abi::caps::MULTITHREAD, "multithread"),
    (abi::caps::OUT_OF_CORE, "out-of-core"),
    (abi::caps::CACHED_CASCADE, "cached-cascade"),
    (abi::caps::SIMD_SSE2, "simd-sse2"),
    (abi::caps::SIMD_AVX2, "simd-avx2"),
    (abi::caps::SIMD_NEON, "simd-neon"),
    (abi::caps::SIMD_WASM128, "simd-wasm128"),
    (abi::caps::COMPUTE, "compute"),
    (abi::caps::PRESENT_TO_MEMORY, "present-to-memory"),
];

/// The names of the set bits, in header order.
pub fn caps_list(bits: u32) -> Vec<&'static str> {
    CAPS.iter().filter(|(bit, _)| bits & bit != 0).map(|(_, name)| *name).collect()
}

/// The set bits as one comma-separated line, plus any bit this tool does not
/// know - an unknown bit is reported rather than dropped, because a capability
/// a host cannot see is one it cannot use.
pub fn caps(bits: u32) -> String {
    let mut out = String::new();
    for (bit, name) in CAPS {
        if bits & bit != 0 {
            if !out.is_empty() {
                out.push_str(", ");
            }
            out.push_str(name);
        }
    }
    let unknown = bits & !CAPS.iter().fold(0u32, |acc, (bit, _)| acc | bit);
    if unknown != 0 {
        if !out.is_empty() {
            out.push_str(", ");
        }
        out.push_str(&format!("unknown:{unknown:#x}"));
    }
    if out.is_empty() {
        "none".into()
    } else {
        out
    }
}

/// A `ReconLShadowFilter` value.
pub fn filter(v: u32) -> &'static str {
    match v {
        0 => "hard",
        1 => "pcf3x3",
        2 => "pcf5x5",
        _ => "pcss-lite",
    }
}

/// A `ReconLShading` value.
pub fn shading(v: u32) -> &'static str {
    match v {
        0 => "unlit",
        1 => "lambert",
        2 => "textured",
        3 => "textured-lambert",
        _ => "unknown",
    }
}

/// A `ReconLTier` value, named by the library.
pub fn tier(v: u32) -> String {
    // SAFETY: reconlTierName returns a static NUL-terminated string for any input.
    unsafe { cstr_of(reconl::reconlTierName(v)) }
}

/// A `ReconLBackendId` value, named by the library.
pub fn backend(v: u32) -> String {
    // SAFETY: reconlBackendName returns a static NUL-terminated string for any input.
    unsafe { cstr_of(reconl::reconlBackendName(v)) }
}

/// A fixed-size NUL-terminated field the library wrote, trimmed at the NUL.
pub fn field(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// The tier ladder, top to bottom - what `reconl-info` reports and what a
/// downgrade walks.
pub const LADDER: [(u32, &str); 5] = [(0, "T0"), (1, "T1"), (2, "T2"), (3, "T3"), (4, "T4")];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_capability_bit_is_named_and_none_is_dropped() {
        let all = CAPS.iter().fold(0u32, |acc, (bit, _)| acc | bit);
        assert_eq!(caps_list(all).len(), CAPS.len());
        assert_eq!(caps(0), "none");
        assert_eq!(caps(abi::caps::SHADOWS | abi::caps::MULTITHREAD), "shadows, multithread");
        // A bit the ABI gains before this tool knows about it still shows up.
        assert!(caps(all | (1 << 31)).contains("unknown:0x80000000"));
    }

    #[test]
    fn fixed_size_fields_stop_at_the_nul() {
        let mut buf = [0u8; 8];
        buf[..3].copy_from_slice(b"gpu");
        assert_eq!(field(&buf), "gpu");
        assert_eq!(field(b"no terminator"), "no terminator");
        assert_eq!(field(&[]), "");
    }

    #[test]
    fn filter_and_shading_names_match_the_header() {
        assert_eq!(filter(0), "hard");
        assert_eq!(filter(1), "pcf3x3");
        assert_eq!(filter(2), "pcf5x5");
        assert_eq!(filter(3), "pcss-lite");
        assert_eq!(filter(9), "pcss-lite", "out of range clamps like the ABI's from_u32");
        assert_eq!(shading(1), "lambert");
        assert_eq!(shading(3), "textured-lambert");
    }
}
