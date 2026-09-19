//! What a descriptor costs, before anything is allocated.
//!
//! Every ABI struct that implies memory is priced here, from the descriptor
//! alone: a create call asks "how many bytes does this imply", checks that
//! number against the budget, and only then allocates. Whether a size is
//! *affordable* is `reconl_core::budget`'s rule; how many bytes a
//! `ReconLTextureDesc` or a swapchain descriptor implies is knowledge of the
//! C surface, so it lives beside `abi.rs` rather than being re-derived inside
//! each entry point.
//!
//! Everything here is pure arithmetic on the descriptor, which is why it is
//! unit-tested without a device: the chain a texture descriptor implies, the
//! slots a command list gets, the images a swapchain names. A buffer needs no
//! arithmetic at all - its price is its `size_bytes`, checked where it is used.

use crate::abi::ReconLSwapchainDesc;

/// One command's encoded size in a command list.
///
/// The header publishes the same number as `RECONL_COMMAND_BYTES` so a host can
/// size a list from the commands it intends to record; the two are kept equal by
/// `the_per_command_cost_the_header_publishes_is_the_one_priced_here` below.
pub const COMMAND_BYTES: u32 = 64;

/// The tightly packed RGBA8 mip chain a texture descriptor implies.
pub struct TextureChain {
    /// `(width, height)` per level, from the base down. Never empty: every
    /// texture has at least one level. Level `i` costs `w * h * 4` bytes, so
    /// `bytes` below is exactly what allocating this chain will ask for.
    pub sizes: Vec<(u32, u32)>,
    /// Every level's bytes summed: the price of the chain.
    pub bytes: u64,
}

impl TextureChain {
    /// The first `levels` levels of a `width` x `height` RGBA8 chain.
    ///
    /// `levels` is a count the caller has already validated against
    /// [`mip_levels`]; this never grows a chain past the mips that exist.
    pub fn of(width: u32, height: u32, levels: u32) -> TextureChain {
        let (mut w, mut h) = (width.max(1), height.max(1));
        let count = levels.max(1);
        let mut sizes = Vec::with_capacity(count as usize);
        let mut bytes = 0u64;
        for _ in 0..count {
            sizes.push((w, h));
            bytes = bytes.saturating_add(u64::from(w) * u64::from(h) * 4);
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        TextureChain { sizes, bytes }
    }

    pub fn levels(&self) -> u32 {
        self.sizes.len() as u32
    }
}

/// How many levels a full chain of a `width` x `height` texture has.
pub fn mip_levels(width: u32, height: u32) -> u32 {
    let (mut w, mut h) = (width.max(1), height.max(1));
    let mut count = 1u32;
    while w > 1 || h > 1 {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        count += 1;
    }
    count
}

/// The slots a command list of `capacity_bytes` gets, and what those slots
/// cost - the number to check before allocating them. At least one slot, so a
/// hostile zero still yields a usable (tiny) list rather than an empty one.
pub fn command_list_slots(capacity_bytes: u32) -> (u32, u64) {
    let slots = (capacity_bytes / COMMAND_BYTES).max(1);
    (slots, u64::from(slots) * u64::from(COMMAND_BYTES))
}

/// The image bytes a swapchain descriptor names.
///
/// The library allocates none of this - the host owns the present buffer - but
/// the descriptor is still a claim about how much memory is about to exist, and
/// it is the only number a host can be refused on before it builds the buffer.
pub fn swapchain_bytes(desc: &ReconLSwapchainDesc) -> u64 {
    u64::from(desc.width) * u64::from(desc.height) * 4 * u64::from(desc.image_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_chain_is_its_levels_summed() {
        let chain = TextureChain::of(1024, 1024, mip_levels(1024, 1024));
        assert_eq!(chain.levels(), 11);
        assert_eq!(chain.sizes[0], (1024, 1024));
        assert_eq!(chain.sizes[10], (1, 1));
        let summed: u64 = chain.sizes.iter().map(|&(w, h)| u64::from(w) * u64::from(h) * 4).sum();
        assert_eq!(chain.bytes, summed, "the price is the chain, not an estimate of it");
        // 4 MiB plus its thirds: 5 MiB and change, all of it really allocated.
        assert!(chain.bytes > 5 << 20 && chain.bytes < 6 << 20, "{}", chain.bytes);
    }

    #[test]
    fn a_truncated_chain_costs_less_and_stops_early() {
        let full = TextureChain::of(256, 256, mip_levels(256, 256));
        let two = TextureChain::of(256, 256, 2);
        assert_eq!(two.sizes, vec![(256, 256), (128, 128)]);
        assert!(two.bytes < full.bytes);
    }

    #[test]
    fn odd_and_non_square_halve_to_one_not_to_zero() {
        let chain = TextureChain::of(5, 1, 3);
        assert_eq!(chain.sizes, vec![(5, 1), (2, 1), (1, 1)]);
        assert_eq!(chain.bytes, (5 + 2 + 1) * 4);
        // A degenerate zero edge still prices one level rather than dividing by
        // zero or handing back an empty chain; the caller refuses it earlier.
        assert_eq!(TextureChain::of(0, 0, 1).sizes, vec![(1, 1)]);
    }

    #[test]
    fn an_absurd_descriptor_is_priced_without_allocating() {
        // The largest texture the ABI can describe, priced in arithmetic: this
        // is what makes the refusal cheap instead of a machine that swaps.
        let chain = TextureChain::of(65535, 65535, mip_levels(65535, 65535));
        assert!(chain.bytes > 16 << 30, "a 65535^2 chain is over 16 GiB, got {}", chain.bytes);
        assert_eq!(chain.levels(), 16);
    }

    #[test]
    fn mip_levels_counts_until_both_edges_are_one() {
        assert_eq!(mip_levels(1, 1), 1);
        assert_eq!(mip_levels(2, 1), 2);
        assert_eq!(mip_levels(4096, 4096), 13);
    }

    /// A host sizes a command list with `commands * RECONL_COMMAND_BYTES`, so
    /// the number the header publishes and the number priced here have to be the
    /// same one. Reading the header is what makes that a check rather than a
    /// promise.
    #[test]
    fn the_per_command_cost_the_header_publishes_is_the_one_priced_here() {
        let header = include_str!("../../include/reconl/reconl.h");
        let line = header
            .lines()
            .find(|l| l.contains("define RECONL_COMMAND_BYTES"))
            .expect("the header publishes the per-command cost");
        assert_eq!(
            line.split_whitespace().last().unwrap(),
            COMMAND_BYTES.to_string(),
            "the header and sizing disagree about what a command costs"
        );
        // And the default capacity the create call falls back to is a round
        // number of commands, so a host that passes 0 can still count slots.
        assert_eq!(COMMAND_BYTES * 1024, 64 * 1024);
    }

    #[test]
    fn a_command_list_always_gets_at_least_one_slot() {
        assert_eq!(command_list_slots(0), (1, u64::from(COMMAND_BYTES)));
        assert_eq!(command_list_slots(4096), (64, 4096));
        assert_eq!(command_list_slots(65), (1, 64));
    }

    #[test]
    fn a_swapchain_is_priced_by_the_images_it_names() {
        let mut desc: ReconLSwapchainDesc = unsafe { core::mem::zeroed() };
        desc.width = 1920;
        desc.height = 1080;
        desc.image_count = 2;
        assert_eq!(swapchain_bytes(&desc), 1920 * 1080 * 4 * 2);
        desc.width = 65535;
        desc.height = 65535;
        assert_eq!(swapchain_bytes(&desc), 65535 * 65535 * 4 * 2, "two images of 4 bytes a pixel");
        assert!(swapchain_bytes(&desc) > 17 << 30, "which is over 17 GiB");
    }
}
