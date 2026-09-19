//! Resource lifetime: upload staging, mip chains, and the disk arena everything
//! else spills into.
//!
//! Three jobs, in the order a frame needs them:
//!
//! 1. [`staging`] - host-allocator-backed upload buffers, so a vertex or texture
//!    upload never touches the Rust global allocator behind the host's back.
//! 2. [`mips`] - deterministic mip chain generation. A box filter with a fixed
//!    sample order, because a mip chain that differs between tiers is a golden
//!    image that cannot be diffed.
//! 3. [`spill`] - the `RCLS` append-only arena: texture mips, static cascades and
//!    tile caches live there when RAM runs out.

pub mod mips;
pub mod spill;
pub mod staging;

pub use mips::{generate_mip_chain, MipChain};
pub use spill::{default_spill_dir, ArenaStats, Hit, SpillArena, SpillConfig};
pub use staging::{StagingBuffer, StagingPool};

/// Where a resource is resident right now. Reported in `ReconLMemoryStats`, and
/// the reason a "the same scene, the same output frame" claim can be checked
/// against what actually happened.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Residency {
    Vram,
    Ram,
    Disk,
}
