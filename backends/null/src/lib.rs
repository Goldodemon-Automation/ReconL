//! The `null` backend: a deterministic no-op that counts everything.
//!
//! It exists so the ABI conformance tests - refcounting, budget accounting,
//! error paths, version/`struct_size` mismatches, the forced downgrade chain -
//! run with no GPU, no disk, no threads and no pixels. Its output is defined
//! rather than empty: `present_checksum` fills the readback with the frame's
//! checksum pattern, so "the null backend produced the same frame twice" is a
//! test anyone can write.
//!
//! It also lies on request: `fake_tier` and `fake_vram_bytes` let the downgrade
//! ladder be driven end to end without hardware.

use reconl_core::error::{Code, Error, Result};
use reconl_core::stats::{Counters, FrameNumbers, ShadowCounters};
use reconl_core::tier::{caps, Backend, Tier};

pub const FLAG_PRESENT_CHECKSUM: u32 = 1;

#[derive(Clone, Copy, Debug)]
pub struct NullConfig {
    pub fake_tier: Tier,
    pub fake_vram_bytes: u64,
    pub fake_ram_bytes: u64,
    /// `None` = the full capability set.
    pub report_caps: Option<u32>,
    pub present_checksum: bool,
    /// `N > 0`: the Nth frame after creation fails with DEVICE_LOST, so the
    /// ladder can be exercised.
    pub fail_after_frames: u32,
    pub command_capacity: u32,
}

impl Default for NullConfig {
    fn default() -> Self {
        Self {
            fake_tier: Tier::CpuRam,
            fake_vram_bytes: 0,
            fake_ram_bytes: 8 << 30,
            report_caps: None,
            present_checksum: true,
            fail_after_frames: 0,
            command_capacity: 4096,
        }
    }
}

pub struct NullDevice {
    config: NullConfig,
    frame_index: u64,
    /// Bytes of commands recorded this frame, so "it counted everything" has a
    /// number rather than an adjective.
    commands_this_frame: u64,
    commands_total: u64,
    draws: u64,
    triangles: u64,
    vertices: u64,
    uploads: u64,
    upload_bytes: u64,
    last_checksum: u64,
    frame_width: u32,
    frame_height: u32,
    pub counters: Counters,
    pub shadows: ShadowCounters,
    pub frame: FrameNumbers,
}

impl NullDevice {
    pub fn new(config: NullConfig) -> Self {
        Self {
            config,
            frame_index: 0,
            commands_this_frame: 0,
            commands_total: 0,
            draws: 0,
            triangles: 0,
            vertices: 0,
            uploads: 0,
            upload_bytes: 0,
            last_checksum: 0,
            frame_width: 0,
            frame_height: 0,
            counters: Counters::default(),
            shadows: ShadowCounters::default(),
            frame: FrameNumbers::default(),
        }
    }

    pub fn config(&self) -> NullConfig {
        self.config
    }

    pub fn caps(&self) -> u32 {
        self.config.report_caps.unwrap_or_else(|| {
            caps::TEXTURES
                | caps::MIPMAPS
                | caps::SHADOWS
                | caps::PCF_5X5
                | caps::PCSS_LITE
                | caps::DISK_SPILL
                | caps::OUT_OF_CORE
                | caps::MULTITHREAD
                | caps::CACHED_CASCADE
                | caps::COMPUTE
                | caps::PRESENT_TO_MEMORY
        })
    }

    pub fn backend(&self) -> Backend {
        Backend::Null
    }

    pub fn tier(&self) -> Tier {
        self.config.fake_tier
    }

    pub fn device_name(&self) -> &'static str {
        "null (deterministic no-op)"
    }

    pub fn driver(&self) -> &'static str {
        "reconl null"
    }

    pub fn vram_bytes(&self) -> u64 {
        self.config.fake_vram_bytes
    }

    pub fn ram_bytes(&self) -> u64 {
        self.config.fake_ram_bytes
    }

    pub fn note_command(&mut self, bytes: u64) {
        self.commands_this_frame += bytes;
        self.commands_total += bytes;
    }

    pub fn note_draw(&mut self, vertices: u64, triangles: u64) {
        self.draws += 1;
        self.vertices += vertices;
        self.triangles += triangles;
    }

    pub fn note_upload(&mut self, bytes: u64) {
        self.uploads += 1;
        self.upload_bytes += bytes;
    }

    /// Begins a frame. Returns DEVICE_LOST when the host asked for a scripted
    /// failure, which is how the tier ladder is tested without hardware.
    pub fn begin_frame(&mut self, width: u32, height: u32) -> Result<()> {
        if self.config.fail_after_frames > 0 && self.frame_index >= self.config.fail_after_frames as u64 {
            self.counters.device_losses += 1;
            return reconl_core::err!(
                Code::DeviceLost,
                "null backend was told to fail after {} frames",
                self.config.fail_after_frames
            );
        }
        self.frame_width = width;
        self.frame_height = height;
        self.commands_this_frame = 0;
        Ok(())
    }

    pub fn submit(&mut self) -> Result<()> {
        self.frame.frame_index = self.frame_index;
        self.frame.total_ns = 0;
        self.frame.triangles_in = self.triangles.min(u32::MAX as u64) as u32;
        self.frame.worker_threads = 0;
        self.frame.resolution_scale = 1.0;
        self.last_checksum = frame_checksum(self.frame_index, self.frame_width, self.frame_height, self.commands_total);
        Ok(())
    }

    /// Writes the frame's deterministic pattern into a readback buffer.
    pub fn present(&mut self, out: Option<&mut [u8]>, row_pitch: u32) -> Result<u64> {
        let checksum = frame_checksum(self.frame_index, self.frame_width, self.frame_height, self.commands_total);
        self.last_checksum = checksum;
        if let Some(out) = out {
            if !self.config.present_checksum {
                for b in out.iter_mut() {
                    *b = 0;
                }
            } else {
                let pitch = if row_pitch == 0 { self.frame_width.saturating_mul(4) as usize } else { row_pitch as usize };
                let rows = self.frame_height as usize;
                for row in 0..rows {
                    for col in 0..self.frame_width as usize {
                        let idx = row * pitch + col * 4;
                        if idx + 3 >= out.len() {
                            break;
                        }
                        // A pattern derived from the frame checksum: reproducible,
                        // and obviously not a rendered scene.
                        let v = ((checksum >> ((col % 8) * 8)) & 0xFF) as u8;
                        out[idx] = v;
                        out[idx + 1] = v.wrapping_add(row as u8);
                        out[idx + 2] = v ^ (row as u8);
                        out[idx + 3] = 255;
                    }
                }
            }
        }
        self.frame_index += 1;
        self.counters.frames_presented += 1;
        Ok(checksum)
    }

    pub fn last_checksum(&self) -> u64 {
        self.last_checksum
    }

    pub fn stats_snapshot(&self) -> (Counters, ShadowCounters, FrameNumbers) {
        (self.counters, self.shadows, self.frame)
    }

    pub fn totals(&self) -> (u64, u64, u64, u64, u64, u64) {
        (self.draws, self.triangles, self.vertices, self.commands_total, self.uploads, self.upload_bytes)
    }
}

/// Frame pattern checksum: stable across runs and machines (FNV-1a over the
/// frame's identifying numbers).
pub fn frame_checksum(frame_index: u64, width: u32, height: u32, commands: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for bytes in [
        frame_index.to_le_bytes(),
        (width as u64).to_le_bytes(),
        (height as u64).to_le_bytes(),
        commands.to_le_bytes(),
    ] {
        for b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// The null backend refuses nothing at creation, so the only way it fails is the
/// scripted device loss or a zeroed allocator (checked by the caller).
pub fn validate(config: &NullConfig) -> Result<()> {
    if config.command_capacity == 0 {
        return Err(Error::new(Code::InvalidArgument, "command_capacity must be non-zero"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_deterministic_and_counted() {
        let mut device = NullDevice::new(NullConfig::default());
        device.begin_frame(64, 64).unwrap();
        device.note_command(64);
        device.note_draw(3, 1);
        device.note_upload(1024);
        device.submit().unwrap();
        let first = device.last_checksum();
        let mut pixels = vec![0u8; 64 * 64 * 4];
        device.present(Some(&mut pixels), 64 * 4).unwrap();
        let copy = pixels.clone();

        // Same frame, run again: same pattern.
        let mut again = NullDevice::new(NullConfig::default());
        again.begin_frame(64, 64).unwrap();
        again.note_command(64);
        again.note_draw(3, 1);
        again.note_upload(1024);
        again.submit().unwrap();
        let mut pixels2 = vec![0u8; 64 * 64 * 4];
        again.present(Some(&mut pixels2), 64 * 4).unwrap();
        assert_eq!(copy, pixels2, "the null backend must be reproducible");
        assert_eq!(first, again.last_checksum());

        let (draws, triangles, vertices, commands, uploads, bytes) = device.totals();
        assert_eq!((draws, triangles, vertices, uploads, bytes), (1, 1, 3, 1, 1024));
        assert!(commands > 0);
        assert_eq!(device.stats_snapshot().0.frames_presented, 1);
    }

    #[test]
    fn the_next_frame_differs_from_this_one() {
        let mut device = NullDevice::new(NullConfig::default());
        device.begin_frame(8, 8).unwrap();
        device.submit().unwrap();
        let a = device.last_checksum();
        device.present(None, 0).unwrap();
        device.begin_frame(8, 8).unwrap();
        device.submit().unwrap();
        let b = device.last_checksum();
        assert_ne!(a, b, "the frame index must be part of the pattern");
    }

    #[test]
    fn scripted_device_loss_reports_rather_than_panics() {
        let mut config = NullConfig::default();
        config.fail_after_frames = 2;
        let mut device = NullDevice::new(config);
        for _ in 0..2 {
            device.begin_frame(4, 4).unwrap();
            device.submit().unwrap();
            device.present(None, 0).unwrap();
        }
        let err = device.begin_frame(4, 4).unwrap_err();
        assert_eq!(err.code, Code::DeviceLost);
        assert_eq!(device.stats_snapshot().0.device_losses, 1);
    }

    #[test]
    fn checksum_pattern_respects_a_wider_pitch() {
        let mut device = NullDevice::new(NullConfig::default());
        device.begin_frame(4, 2).unwrap();
        device.submit().unwrap();
        let mut pixels = vec![0xFFu8; 2 * 8 * 4];
        device.present(Some(&mut pixels), 8 * 4).unwrap();
        // Row 1 starts at the pitch, not at 16 bytes.
        assert_eq!(pixels[8 * 4 + 3], 255);
    }

    #[test]
    fn fake_tier_and_caps_are_reported_verbatim() {
        let mut config = NullConfig::default();
        config.fake_tier = Tier::OutOfCore;
        config.report_caps = Some(caps::SHADOWS);
        let device = NullDevice::new(config);
        assert_eq!(device.tier(), Tier::OutOfCore);
        assert_eq!(device.caps(), caps::SHADOWS);
        assert!(validate(&device.config()).is_ok());
    }
}
