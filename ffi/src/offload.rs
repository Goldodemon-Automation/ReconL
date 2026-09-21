//! The tier ladder: the one place a tier change is decided, applied and written
//! down.
//!
//! This module is the *whole* answer to "why is this device on that tier": the
//! state a comparison is filed under ([`PlanKey`]), the state the policy carries
//! between frames ([`Offload`]), the one function that decides
//! (`apply_tier_policy`), the backend rebuilds every decision needs, and the one
//! log a host reads to see what happened (`DeviceHandle::downgrades`).
//! `docs/offload.md` is the specification; nothing here may disagree with it.
//!
//! Two layers act on a tier (`docs/offload.md`, "Two layers, one number"):
//!
//! * the **relabel**, which keeps the backend and lowers its quality tier, and
//! * the **offload**, which changes the backend.
//!
//! Both are decided here, from one number (the frame cost the device composed,
//! readback and all) counted once ([`FrameLadder`]). The relabel answers from
//! the run as it stood before the frame's own cost and the offload from the
//! frame's observation, so the two are one frame apart in when they act - the
//! order a backend's own ladder kept before both layers shared the device's run. Neither is decided by a
//! backend any more: a backend is told its tier and applies it, and it records
//! nothing, because a backend's own ring died with the backend while the tier it
//! changed did not. One decision point, one record - which is what makes the ring
//! a history (`ReconLStats.downgrades`) rather than whatever the live backend
//! happened to remember.
//!
//! Every method below is on `DeviceHandle` rather than on a struct of its own
//! because the decision needs the device's frame cost, its frame, its budget and
//! its host bits at once; carving that up would trade one owner for four.

use crate::abi::{allow_downgrade, backend};
use crate::layout::host_row_layout;
use crate::{frame_input, BackendKind, DeviceHandle, FrameRecord};
use reconl_backend_d3d11::D3d11Device;
use reconl_backend_softcpu::SoftCpuDevice;
use reconl_contract::ShadowRequest;
use reconl_core::error::{Code, Error, Result};
use reconl_core::log_warn;
use reconl_core::stats::Downgrade;
use reconl_core::tier::{Tier, TierReason};
use std::sync::Arc;
use std::time::Instant;

/// What a measured tier comparison is filed under.
///
/// A comparison is only valid for the frame it was measured on: resolution and
/// the shadow plan (which is per frame, from `ReconLShadowConfig`) both change
/// what a tier costs. Filing the result under both is what makes it expire when
/// either changes, with no invalidation logic to get wrong.
type PlanKey = (u32, u32, u32, u32, u64);

fn plan_key(width: u32, height: u32, shadow: &ShadowRequest) -> PlanKey {
    (width, height, shadow.cascades, shadow.filter as u32, shadow.texel_budget_bytes)
}

/// The offload state (docs/offload.md): measured, then remembered.
///
/// Nothing here is a guess. The comparison that decides whether an overload is
/// worth offloading is the reference tier's own measured frame cost against the
/// hardware's, at one plan, taken from the first frame the reference tier
/// rendered - which is why the calibration costs no extra frame.
pub(crate) struct Offload {
    /// Where the reference tier's cost was measured, and whether it lost there.
    /// `measured_key` is `None` until a calibration has run, so an unmeasured
    /// device never trades its hardware for the CPU. The cost itself is in the
    /// downgrade entry's detail, which is where a host reads it.
    measured_key: Option<PlanKey>,
    /// The reference tier measured slower than the hardware at `measured_key`.
    /// The question is settled there and the offload is not attempted again.
    cpu_lost: bool,
    /// A return trip to the hardware was made at this key. One per key: a
    /// workload that oscillates must not thrash between tiers, and "the hardware
    /// was tried again and missed" is what "this frame does not fit the GPU"
    /// means. It also means the CPU's cost here is known, so a second offload at
    /// this key needs no calibration.
    returned_key: Option<PlanKey>,
    /// Consecutive frames inside the target while offloaded. The over-target run
    /// is the device's [`FrameLadder`], shared by both layers.
    within_target: u32,
    /// The first offloaded frame is also the calibration, so its cost is
    /// compared before the decision to stay stands.
    calibrating: bool,
    /// The hardware cost the calibration is compared against.
    gpu_ns: u64,
    /// Hardware faults this device has seen. A rebuilt backend that faults too
    /// is not returned to, so a broken driver cannot start a rebuild loop.
    faults: u32,
}

impl Offload {
    pub(crate) const fn new() -> Self {
        Self {
            measured_key: None,
            cpu_lost: false,
            returned_key: None,
            within_target: 0,
            calibrating: false,
            gpu_ns: 0,
            faults: 0,
        }
    }
}

impl DeviceHandle {
    /// Whether this device may continue its frames on the reference tier.
    ///
    /// Three things have to hold: the host left `RECONL_DOWNGRADE_TIER` set, the
    /// device was built with a hardware backend, and the hardware config to come
    /// back to exists.
    pub(crate) fn may_offload(&self) -> bool {
        (self.allow_downgrade & allow_downgrade::TIER) != 0
            && self.origin.gpu.is_some()
            && self.backend_id() == backend::D3D11
    }

    /// Whether this device is currently rendering on the reference tier because
    /// it offloaded there, as opposed to having been created there.
    fn offloaded(&self) -> bool {
        self.origin.gpu.is_some() && self.backend_id() == backend::SOFT_CPU
    }

    /// Builds the reference-tier backend an offload renders on.
    fn build_cpu_backend(&self) -> Result<SoftCpuDevice> {
        let mut config = self.origin.cpu.clone();
        // The ladder's own step: a hardware tier lands on T2, because a GPU that
        // cannot serve a frame cannot serve a higher CPU tier either.
        config.tier = if self.tier >= Tier::CpuRam { self.tier } else { Tier::CpuRam };
        SoftCpuDevice::new(self.alloc, Arc::clone(&self.budget), config)
    }

    /// Rebuilds the hardware backend a return trip goes back to.
    fn build_gpu_backend(&self) -> Result<D3d11Device> {
        let config = self.origin.gpu.clone().ok_or_else(|| {
            Error::new(Code::BackendUnavailable, "this device has no hardware backend to return to")
        })?;
        D3d11Device::new(self.alloc, Arc::clone(&self.budget), config)
    }

    /// The one place a tier change is written down.
    ///
    /// Every entry a host reads in `ReconLStats.downgrades` comes through here,
    /// in the order it happened, whether the change kept the backend or replaced
    /// it. One log, because the record outlives the backend that made the change:
    /// a ring per backend lost every entry the backend had recorded the moment
    /// its frames moved elsewhere.
    fn note_tier_change(&mut self, from: Tier, to: Tier, reason: TierReason, frame_index: u64, detail: &str) {
        self.downgrades.record(Downgrade::new(from, to, reason, frame_index, 0, detail));
    }

    /// Applies a tier change that keeps the backend, and writes it down.
    ///
    /// The backend applies its own part of a tier change - what a lower tier
    /// invalidates - and nothing else: it does not decide the change and holds no
    /// record of it. The null backend has no quality tiers of its own (it reports
    /// the tier the host asked it to claim), so a change to one is recorded
    /// rather than applied; that is the only backend for which the two differ.
    pub(crate) fn apply_tier(&mut self, to: Tier, reason: TierReason, frame_index: u64, detail: &str) {
        let from = self.tier;
        match &mut self.backend {
            BackendKind::SoftCpu(d) => d.relabel(to, reason),
            BackendKind::D3d11(d) => d.relabel(to, reason),
            BackendKind::Null(_) => {}
        }
        self.tier = to;
        self.tier_reason = reason;
        self.stats.tier = to;
        self.stats.tier_reason = reason;
        self.stats.tier_reason_text.set(reason.text());
        self.stats.counters.frames_since_tier_change = 0;
        self.note_tier_change(from, to, reason, frame_index, detail);
    }

    /// The device's own answer to an over-target frame: one step down the quality
    /// ladder, same backend.
    ///
    /// The detail names the layer, so a host reading the ring can tell this
    /// relabel from the host-level offload, which reports the same reason for the
    /// same measurement.
    fn relabel_on_overload(&mut self, frame_index: u64) {
        let to = self.tier.step_down();
        if to == self.tier {
            // T4 is the floor: a device there has nothing left to step to.
            return;
        }
        let detail = format!(
            "{} frames over the {} ms target the host read ({} ns): this device's own tier",
            self.ladder.over_target(),
            self.ladder.target_ms(),
            self.ladder.last_over_ns()
        );
        self.apply_tier(to, TierReason::FrameTimeOverTarget, frame_index, &detail);
    }

    /// Adopts `backend` as the renderer, with the bookkeeping a host reads: the
    /// tier, the caps, the names, the counter that says the safe path was taken,
    /// and one downgrade entry carrying the reason.
    fn adopt_backend(&mut self, backend: BackendKind, reason: TierReason, frame_index: u64, detail: &str) {
        let from = self.tier;
        let (to, caps, name, driver, id) = match &backend {
            BackendKind::SoftCpu(d) => (d.tier(), d.caps(), d.device_name(), d.driver(), backend::SOFT_CPU),
            BackendKind::D3d11(d) => (d.tier(), d.caps(), d.device_name(), d.driver(), backend::D3D11),
            BackendKind::Null(d) => (d.tier(), d.caps(), d.device_name(), d.driver(), backend::NULL),
        };
        self.backend = backend;
        self.tier = to;
        self.tier_reason = reason;
        self.caps = caps;
        self.device_name.set(name);
        self.driver.set(driver);
        self.stats.backend = id;
        self.stats.caps = caps;
        self.stats.tier = to;
        self.stats.tier_reason = reason;
        self.stats.tier_reason_text.set(reason.text());
        self.stats.device_name.set(name);
        self.stats.counters.safe_path_events += 1;
        self.stats.counters.frames_since_tier_change = 0;
        // A return trip buys a settle window: the frames inside the target that
        // the incoming backend must show are its own. The *run* the ladder acts
        // on is the device's and is not restarted here - the frames a rebuilt
        // backend presents were paid for by the device that handed them over,
        // and a relabel that the run had already earned still lands
        // (`core/src/tier.rs`, `FrameLadder`).
        self.offload.within_target = 0;
        self.note_tier_change(from, to, reason, frame_index, detail);
        log_warn!(
            "now rendering on {} at tier {} ({}): {}",
            reconl_core::tier::Backend::from_u32(id).name(),
            to.name(),
            reason.text(),
            detail
        );
    }

    /// Hands the frames to the reference tier after a hardware fault.
    ///
    /// No comparison here: a device that has been removed is not a slow device,
    /// so the reference tier is faster by definition and the calibration would
    /// be measuring nothing.
    pub(crate) fn offload_on_fault(&mut self, frame_index: u64, detail: &str) -> Result<()> {
        let cpu = self.build_cpu_backend()?;
        self.offload.faults += 1;
        self.offload.calibrating = false;
        self.offload.within_target = 0;
        self.adopt_backend(
            BackendKind::SoftCpu(Box::new(cpu)),
            TierReason::DeviceRemoved,
            frame_index,
            detail,
        );
        Ok(())
    }

    /// Offloads because the hardware missed its target, and - unless the
    /// comparison is already known - marks the next frame as the calibration that
    /// decides whether the offload stands.
    fn offload_on_overload(&mut self, frame_index: u64, detail: &str, calibrate: bool) -> Result<()> {
        let cpu = self.build_cpu_backend()?;
        self.offload.calibrating = calibrate;
        self.adopt_backend(
            BackendKind::SoftCpu(Box::new(cpu)),
            TierReason::FrameTimeOverTarget,
            frame_index,
            detail,
        );
        Ok(())
    }

    /// Rebuilds the hardware backend after a settle window inside the target.
    fn return_to_gpu(&mut self, frame_index: u64, detail: &str) -> Result<()> {
        let gpu = self.build_gpu_backend()?;
        self.adopt_backend(
            BackendKind::D3d11(Box::new(gpu)),
            // Up the ladder, which is what the reason says: a host reading the
            // log can tell a recovery from how the device started.
            TierReason::Recovery,
            frame_index,
            detail,
        );
        Ok(())
    }

    /// The measured ladder, run after every frame that rendered: decides what the
    /// *next* frame renders on, and at what tier.
    ///
    /// One function decides both layers, from one measurement: the device counts
    /// the frame's complete cost into the one [`FrameLadder`], so the two cannot
    /// disagree about whether a frame was over target. They answer it one frame
    /// apart, because that is when each acts:
    ///
    /// * the **relabel** is decided first, from the run as it stood *before* this
    ///   frame's cost - so it lands on the frame after the over-target one, which
    ///   is the frame the backend it names then renders;
    /// * the **offload** is decided next, from this frame's own observation, which
    ///   is what makes a frame that misses the target hand the *next* frame to the
    ///   reference tier.
    ///
    /// Only the offload changes the backend. A frame that changes it is recorded
    /// as that change *after* the relabel the same frame earned: the relabel is
    /// applied to the backend that is live when the frame closes, which is the
    /// one being replaced, and the change then names the tier the device stands
    /// at. Every rule here is a rule from docs/offload.md.
    pub(crate) fn apply_tier_policy(&mut self, frame_index: u64, width: u32, height: u32) -> Result<()> {
        let target_ms = self.ladder.target_ms();
        let target_ns = self.ladder.target_ns();
        let threshold = self.ladder.threshold();
        let total_ns = self.frame_cost().total_ns;
        let key = plan_key(width, height, &self.frame.shadow);

        // Layer 1, the cheaper response, and the only one a host that opted out
        // of backend changes ever sees: the device's own relabel, answered from
        // the run as it stood before this frame's cost. The null backend is left
        // alone - it has no quality tiers of its own and reports the tier the
        // host asked it to claim.
        if self.ladder.acting()
            && matches!(self.backend, BackendKind::SoftCpu(_) | BackendKind::D3d11(_))
        {
            self.relabel_on_overload(frame_index);
        }

        // The frame's one observation, and layer 2's answer from it.
        let over = self.ladder.observe(total_ns);

        if self.backend_id() == backend::D3D11 {
            // Layer 2: the backend change. Measured overload - the target being
            // zero means the ladder is off, which is a host's way of saying "do
            // not decide for me" - and a comparison this plan has already lost,
            // which settles the question here for good.
            let measured_slower = self.offload.measured_key == Some(key) && self.offload.cpu_lost;
            if over && self.offload.faults == 0 && self.may_offload() && !measured_slower {
                if self.offload.measured_key == Some(key) {
                    // Measured here and the CPU won. The hardware was tried
                    // again after a return trip and missed, so this frame
                    // genuinely does not fit it - and the calibration's answer is
                    // already known.
                    let detail = format!(
                        "{} frames over the {} ms target at {}x{}, where the reference tier already measured faster",
                        self.ladder.over_target(), target_ms, width, height
                    );
                    self.offload_on_overload(frame_index, &detail, false)?;
                } else {
                    // Not measured here: one frame on the CPU is the price of not
                    // making every subsequent frame slow. The measurement the
                    // calibration is compared against goes into the entry the
                    // miss caused: a host (and the test that pins this) can then
                    // read why the device came back, rather than guessing.
                    self.offload.gpu_ns = total_ns;
                    let detail = format!(
                        "{} frames over the {} ms target at {}x{}; the hardware measured {} ns; calibrating the reference tier",
                        self.ladder.over_target(), target_ms, width, height, total_ns
                    );
                    self.offload_on_overload(frame_index, &detail, true)?;
                }
                return Ok(());
            }
            // The backend stays: this frame's answer was the relabel above, and
            // the offload is not permitted or not warranted.
            return Ok(());
        }

        if !self.offloaded() {
            // Not a device that offloaded: one created on the reference tier,
            // whose tier is still the ladder's to step (the relabel above), or
            // the null backend, which the ladder does not relabel at all.
            return Ok(());
        }

        if self.offload.calibrating {
            // The first offloaded frame is the calibration: the reference tier's
            // own cost, measured, against the hardware's - no extra frame spent.
            self.offload.calibrating = false;
            self.offload.measured_key = Some(key);
            if target_ns > 0 && self.offload.gpu_ns > 0 && total_ns >= self.offload.gpu_ns {
                self.offload.cpu_lost = true;
                let detail = format!(
                    "the reference tier measured {} against the hardware's {} at {}x{}, so the offload was not a win",
                    total_ns, self.offload.gpu_ns, width, height
                );
                self.return_to_gpu(frame_index, &detail)?;
                return Ok(());
            }
            // The offload stands, and the frame that decided it is over target
            // like any other: it costs this device a tier step, not a backend.
            // That step is the relabel above, on the next frame.
            return Ok(());
        }

        // The return trip. It needs a target to be inside and a settle window of
        // frames inside it, and it is attempted at most once per plan: a workload
        // that oscillates around the target must not thrash between tiers. A
        // device that offloaded because of a fault rather than a miss therefore
        // stays offloaded on a host that set no frame-time target, which is the
        // only host that cannot say whether the hardware has recovered.
        let may_return = target_ns > 0
            && threshold > 0
            && self.offload.faults <= 1
            && self.offload.returned_key != Some(key);
        if may_return {
            self.offload.within_target = if total_ns <= target_ns {
                self.offload.within_target.saturating_add(1)
            } else {
                0
            };
            if self.offload.within_target >= threshold {
                self.offload.returned_key = Some(key);
                let detail = format!(
                    "{} frames inside the {} ms target: rebuilding the hardware backend",
                    self.offload.within_target, target_ms
                );
                self.return_to_gpu(frame_index, &detail)?;
                return Ok(());
            }
        }
        // Still offloaded: this frame's answer is a tier, not a backend, and the
        // tier was the relabel above.
        Ok(())
    }

    /// Renders `frame` on the reference backend the device is now running.
    ///
    /// The frame's draw list points into the host's own buffer blocks - that is
    /// why `FrameRecord` keeps it past Submit - so the reference tier can
    /// reproduce a frame the hardware did not finish, exactly, out of the list
    /// the frame already holds. That is what makes an offloaded frame
    /// byte-identical to the same frame rendered on the reference tier directly
    /// - and it is why this path allocates nothing to render it.
    pub(crate) fn render_frame_on_soft(&mut self, frame: &mut FrameRecord) -> Result<()> {
        // No fingerprint: this is the frame the hardware could not finish,
        // re-rendered here, and nothing on this path compares one.
        let input = frame_input(frame, false);
        let shadows = {
            let soft = self.softcpu_mut().ok_or_else(|| {
                Error::new(Code::BackendUnavailable, "the reference backend is not available")
            })?;
            soft.prepare_frame(frame.width, frame.height)?;
            soft.render(&input)?;
            soft.snapshot().shadows
        };
        self.stats.shadows = shadows;
        Ok(())
    }

    /// A hardware fault during a frame: continue on the reference tier and
    /// render the frame that faulted there, out of the frame the device still
    /// holds.
    ///
    /// Used by Present, where the frame was already rendered on the GPU and only
    /// its readback failed; Submit renders its own frame, which it is still
    /// holding, so it calls the two steps directly.
    pub(crate) fn recover_frame(&mut self, fault: &Error) -> Result<()> {
        let mut frame = std::mem::replace(&mut self.frame, FrameRecord::new(self.alloc));
        let detail = format!("frame {}: {}", frame.index, fault.message.as_str());
        let result = match self.offload_on_fault(frame.index, &detail) {
            Ok(()) => self.render_frame_on_soft(&mut frame),
            Err(e) => Err(e),
        };
        self.frame = frame;
        result
    }

    /// The present path of a device that just lost its hardware: the frame is
    /// re-rendered on the reference tier and the host gets the pixels it came
    /// for, in the layout it asked for.
    ///
    /// Returns how long that readback took: it is this frame's readback half,
    /// which the caller adds to the frame rather than this function keeping a
    /// second set of books.
    pub(crate) fn present_after_fault(
        &mut self,
        fault: &Error,
        pixels: Option<&mut [u8]>,
        out_size: u64,
        out_pitch: u32,
        flip: u32,
    ) -> Result<u64> {
        self.recover_frame(fault)?;
        let (width, height) = self
            .softcpu()
            .map(|soft| soft.frame_size())
            .ok_or_else(|| Error::new(Code::BackendUnavailable, "the reference backend is not available"))?;
        let mut readback_ns = 0u64;
        if let Some(pixels) = pixels {
            let pitch = host_row_layout(width, height, out_size, out_pitch)?;
            let soft = self.softcpu_mut().ok_or_else(|| {
                Error::new(Code::BackendUnavailable, "the reference backend is not available")
            })?;
            let started = Instant::now();
            soft.read_frame_into(pixels, pitch, flip)?;
            readback_ns = started.elapsed().as_nanos() as u64;
        }
        if let Some(soft) = self.softcpu_mut() {
            soft.on_frame_end()?;
        }
        Ok(readback_ns)
    }

}

/// The fault path's re-render, driven as far as a test can drive it.
///
/// There is no fault-injection hook in this library and none in the ABI: a
/// failover needs both `Code::DeviceLost` and a genuine driver verdict
/// (`D3d11Device::device_removed`), and no machine removes a healthy device on
/// request. The `--repeat=200` removal `README.md` records was a suspended
/// device instance on one machine, not a reproducible condition - re-measured at
/// 512x512 it now runs 5 warmup + 60 measured frames with `failures 0`, still on
/// d3d11. The verdict is therefore the one step this test cannot take, and it
/// says so rather than pretending: it drives every step around it through the
/// exported ABI (device, frame, command list, submit, present) and then the
/// fault's own two steps, which is exactly what `Submit`'s fault arm runs once
/// the verdict is in.
///
/// What it pins is the property the zero-allocation pass put at risk: the frame
/// that faulted is re-rendered out of *its own* draws - the list this frame
/// recorded, not the frame before it and not an empty one - and the host is
/// handed that frame, byte for byte, as a reference-tier device renders it.
#[cfg(test)]
mod fault_tests {
    use crate::abi;
    use crate::{DeviceHandle, FrameRecord, FrameState};
    use core::ffi::c_void;
    use reconl_core::{ABIStruct, StructHeader};

    const W: u32 = 32;
    const H: u32 = 32;
    /// `(index_count, first_index)` of the large triangle the frame before the
    /// fault draws, and of the small one the faulting frame draws. Different
    /// halves of the frame on purpose: an accumulated or stale re-render cannot
    /// land on the same pixels as the frame that faulted.
    const LARGE: (u32, u32) = (0, 3);
    const SMALL: (u32, u32) = (3, 3);

    /// The smallest conforming host allocator: the allocation's base pointer in
    /// a header slot, so `free` can hand the block back without an alignment.
    extern "C" fn a_alloc(_user: *mut c_void, size: usize, alignment: usize) -> *mut c_void {
        let align = alignment.clamp(16, 4096);
        let total = match size.max(1).checked_add(align).and_then(|v| v.checked_add(16)) {
            Some(t) => t,
            None => return core::ptr::null_mut(),
        };
        let layout = match std::alloc::Layout::from_size_align(total, 16) {
            Ok(l) => l,
            Err(_) => return core::ptr::null_mut(),
        };
        // SAFETY: the layout has a non-zero size.
        let raw = unsafe { std::alloc::alloc(layout) };
        if raw.is_null() {
            return core::ptr::null_mut();
        }
        let aligned = (raw as usize + 16 + align - 1) & !(align - 1);
        // SAFETY: `aligned - 8` sits in the 16 bytes of slack the layout kept.
        unsafe { ((aligned - 8) as *mut usize).write_unaligned(raw as usize) };
        aligned as *mut c_void
    }

    extern "C" fn a_free(_user: *mut c_void, ptr: *mut c_void, _size: usize) {
        if ptr.is_null() {
            return;
        }
        // SAFETY: `a_alloc` wrote the allocation base into the 8 bytes before
        // the pointer, and every block came from one `std::alloc::alloc` with
        // that padded layout.
        unsafe {
            let base = ((ptr as usize - 8) as *const usize).read_unaligned() as *mut u8;
            std::alloc::dealloc(base, std::alloc::Layout::from_size_align_unchecked(1, 16));
        }
    }

    extern "C" fn a_realloc(user: *mut c_void, ptr: *mut c_void, old_size: usize, new_size: usize, alignment: usize) -> *mut c_void {
        let fresh = a_alloc(user, new_size, alignment);
        if fresh.is_null() {
            return core::ptr::null_mut();
        }
        if !ptr.is_null() {
            // SAFETY: both blocks are live for their full old/new sizes and do
            // not overlap.
            unsafe {
                core::ptr::copy_nonoverlapping(ptr as *const u8, fresh as *mut u8, old_size.min(new_size));
            }
            a_free(user, ptr, old_size);
        }
        fresh
    }

    fn header<H: ABIStruct>() -> StructHeader {
        StructHeader::new(core::mem::size_of::<H>() as u32, H::STRUCT_TYPE)
    }

    /// Whether the probe says a D3D11 device can be created here - the ABI's own
    /// answer, the fact the integration tests gate their hardware legs on.
    fn d3d11_usable() -> bool {
        let mut info: abi::ReconLProbeInfo = unsafe { core::mem::zeroed() };
        info.base = header::<abi::ReconLProbeInfo>();
        if unsafe { crate::reconlProbe(core::ptr::null(), &mut info) } != abi::result::OK {
            return false;
        }
        (0..info.entry_count as usize)
            .any(|i| info.entries[i].backend == abi::backend::D3D11 && info.entries[i].usable != 0)
    }

    /// Six clip-space vertices: a large triangle in the right half of the frame
    /// (0..3) and a small one in the lower left (3..6), both lit white by the
    /// frame's headlight. The frame's clear is blue, so "lit" is the red
    /// channel, which the clear cannot reach.
    fn vertices() -> [abi::ReconLVertex; 6] {
        let mut v = [abi::ReconLVertex::default(); 6];
        fn put(slot: &mut abi::ReconLVertex, x: f32, y: f32) {
            slot.position = [x, y, 0.5];
            slot.normal = [0.0, 0.0, 1.0];
            slot.color = [1.0, 1.0, 1.0, 1.0];
        }
        put(&mut v[0], 0.95, 0.8);
        put(&mut v[1], 0.25, -0.8);
        put(&mut v[2], 0.95, -0.8);
        put(&mut v[3], -0.95, -0.15);
        put(&mut v[4], -0.35, -0.9);
        put(&mut v[5], -0.95, -0.9);
        v
    }

    /// Lit pixels in one half of the frame.
    fn lit(pixels: &[u8], left: bool) -> usize {
        let mut count = 0;
        for y in 0..H as usize {
            for x in 0..W as usize {
                let inside = if left { x < W as usize / 2 } else { x >= W as usize / 2 };
                if inside && pixels[(y * W as usize + x) * 4] > 200 {
                    count += 1;
                }
            }
        }
        count
    }

    /// The host boilerplate a fault needs: a device, a swapchain that presents to
    /// memory, one pipeline, one command list, and the two buffers the draws read.
    struct Rig {
        device: *mut DeviceHandle,
        swapchain: *mut crate::SwapchainHandle,
        pipeline: *mut crate::PipelineHandle,
        commands: *mut crate::CommandListHandle,
        vb: *mut crate::BufferHandle,
        ib: *mut crate::BufferHandle,
    }

    impl Rig {
        unsafe fn new(backend_hint: u32, tier_hint: u32) -> Rig {
            let allocator = abi::ReconLAllocator {
                alloc: Some(a_alloc),
                realloc: Some(a_realloc),
                free: Some(a_free),
                user: core::ptr::null_mut(),
            };
            let dd = abi::ReconLDeviceDesc {
                base: header::<abi::ReconLDeviceDesc>(),
                backend_hint,
                tier_hint,
                // A host that has opted into backend changes: a fault offloads.
                allow_downgrade: abi::allow_downgrade::TIER,
                worker_threads: if backend_hint == abi::backend::D3D11 { 0 } else { 1 },
                target_frame_ms: 1000,
                downgrade_after_frames: 16,
                seed: 7,
                flags: 0,
                budget: core::ptr::null(),
                allocator,
                backend_desc: core::ptr::null(),
            };
            let mut device: *mut DeviceHandle = core::ptr::null_mut();
            assert_eq!(crate::reconlCreateDevice(&dd, &mut device), abi::result::OK, "device creation failed");
            let sd = abi::ReconLSwapchainDesc {
                base: header::<abi::ReconLSwapchainDesc>(),
                width: W,
                height: H,
                format: 1, // RGBA8
                image_count: 2,
                present_to_memory: 1,
                depth_format: 1,
                flags: 0,
                reserved: 0,
            };
            let mut swapchain: *mut crate::SwapchainHandle = core::ptr::null_mut();
            assert_eq!(crate::reconlCreateSwapchain(device, &sd, &mut swapchain), abi::result::OK);
            let pd = abi::ReconLPipelineDesc {
                base: header::<abi::ReconLPipelineDesc>(),
                shading: abi::shading::LAMBERT,
                blend: 0,
                cull: 0,
                depth_compare: 1, // GREATER, reversed-Z
                depth_write: 1,
                texture_slots: 0,
                texture_formats: [0; abi::RECONL_MAX_TEXTURE_SLOTS],
                receives_shadow: 0,
                casts_shadow: 0,
                flags: 0,
                reserved: 0,
                debug_name: core::ptr::null(),
            };
            let mut pipeline: *mut crate::PipelineHandle = core::ptr::null_mut();
            assert_eq!(crate::reconlCreatePipeline(device, &pd, &mut pipeline), abi::result::OK);
            let cd = abi::ReconLCommandListDesc {
                base: header::<abi::ReconLCommandListDesc>(),
                capacity_bytes: 4096,
                reserved: 0,
                debug_name: core::ptr::null(),
            };
            let mut commands: *mut crate::CommandListHandle = core::ptr::null_mut();
            assert_eq!(crate::reconlCreateCommandList(device, &cd, &mut commands), abi::result::OK);

            let verts = vertices();
            let vbd = abi::ReconLBufferDesc {
                base: header::<abi::ReconLBufferDesc>(),
                size_bytes: (verts.len() * core::mem::size_of::<abi::ReconLVertex>()) as u64,
                usage: 1, // VERTEX
                reserved: 0,
                data: verts.as_ptr() as *const c_void,
                data_size: (verts.len() * core::mem::size_of::<abi::ReconLVertex>()) as u64,
                debug_name: core::ptr::null(),
            };
            let mut vb: *mut crate::BufferHandle = core::ptr::null_mut();
            assert_eq!(crate::reconlCreateBuffer(device, &vbd, &mut vb), abi::result::OK);
            let indices: [u32; 6] = [0, 1, 2, 3, 4, 5];
            let ibd = abi::ReconLBufferDesc {
                base: header::<abi::ReconLBufferDesc>(),
                size_bytes: (indices.len() * 4) as u64,
                usage: 2, // INDEX
                reserved: 0,
                data: indices.as_ptr() as *const c_void,
                data_size: (indices.len() * 4) as u64,
                debug_name: core::ptr::null(),
            };
            let mut ib: *mut crate::BufferHandle = core::ptr::null_mut();
            assert_eq!(crate::reconlCreateBuffer(device, &ibd, &mut ib), abi::result::OK);

            Rig { device, swapchain, pipeline, commands, vb, ib }
        }

        /// Begins the rig's standard frame and records one indexed draw of
        /// `count` indices starting at `first` - the same command sequence a
        /// host records, because the list it builds is what the fault re-renders
        /// from. Returns the draw's result; the frame is left open.
        unsafe fn record(&mut self, first: u32, count: u32) -> i32 {
            // A headlight, so the Lambert pipeline is lit rather than black.
            let mut light: abi::ReconLLight = core::mem::zeroed();
            light.base = header::<abi::ReconLLight>();
            light.r#type = 0; // directional
            light.direction = [0.0, 0.0, -1.0];
            light.color = [1.0, 1.0, 1.0];
            light.intensity = 1.0;
            let ll = abi::ReconLLightList {
                base: header::<abi::ReconLLightList>(),
                count: 1,
                reserved: 0,
                lights: &light,
            };
            let fd = abi::ReconLFrameDesc {
                base: header::<abi::ReconLFrameDesc>(),
                width: W,
                height: H,
                seed: 1,
                reserved: 0,
                lights: &ll,
                shadows: core::ptr::null(),
                camera: core::ptr::null(),
                framegen: core::ptr::null(),
            };
            assert_eq!(crate::reconlBeginFrame(self.device, &fd as *const _ as *mut _), abi::result::OK);

            crate::reconlCmdReset(self.commands);
            let rp = abi::ReconLRenderPassDesc {
                base: header::<abi::ReconLRenderPassDesc>(),
                color_count: 0,
                reserved: 0,
                color: [unsafe { core::mem::zeroed() }; abi::RECONL_MAX_ATTACHMENTS],
                depth: core::ptr::null_mut(),
                viewport_width: 0,
                viewport_height: 0,
                load_color: 1,
                load_depth: 1,
                clear_color: [0.0, 0.0, 1.0, 1.0],
                clear_depth: 0.0,
                stencil_clear: 0,
                reserved2: 0,
            };
            crate::reconlCmdBeginRenderPass(self.commands, &rp);
            crate::reconlCmdSetPipeline(self.commands, self.pipeline);
            let identity: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
            crate::reconlCmdPushConstants(self.commands, 0, identity.as_ptr() as *const c_void, 64);
            crate::reconlCmdPushConstants(self.commands, 1, identity.as_ptr() as *const c_void, 64);
            crate::reconlCmdSetVertexBuffer(self.commands, 0, self.vb, 0);
            crate::reconlCmdSetIndexBuffer(self.commands, self.ib, 0, 1); // UINT32
            let draw_r = crate::reconlCmdDrawIndexed(self.commands, count, first, 0);
            crate::reconlCmdEndRenderPass(self.commands);
            draw_r
        }

        unsafe fn submit(&mut self) -> i32 {
            crate::reconlSubmit(self.device, self.commands, core::ptr::null_mut())
        }

        unsafe fn present(&mut self, out: &mut [u8]) -> i32 {
            let mut prd = abi::ReconLPresentDesc {
                base: header::<abi::ReconLPresentDesc>(),
                out_pixels: out.as_mut_ptr() as *mut c_void,
                out_pixels_size: out.len() as u64,
                out_row_pitch: W * 4,
                out_format: 1,
                flip: 0,
            };
            crate::reconlPresent(self.device, self.swapchain, &mut prd)
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            // SAFETY: handles created by this rig, released once.
            unsafe {
                crate::reconlRelease(self.ib as *mut c_void);
                crate::reconlRelease(self.vb as *mut c_void);
                crate::reconlRelease(self.commands as *mut c_void);
                crate::reconlRelease(self.pipeline as *mut c_void);
                crate::reconlRelease(self.swapchain as *mut c_void);
                crate::reconlRelease(self.device as *mut c_void);
            }
        }
    }

    /// The fault's own two steps, in the order `Submit`'s arm runs them: the
    /// frame is taken out of the device, the reference tier takes over, and the
    /// frame is re-rendered out of the list it already holds - then put back as
    /// the submitted frame, which is what makes it presentable.
    ///
    /// This is called *after* the frame's own submit, so the list it re-renders
    /// from is the one the replay built - the same list the arm sees when the
    /// driver's verdict arrives instead of a completed render.
    unsafe fn fault_and_rerender(rig: &mut Rig) {
        let mut frame = core::mem::replace(&mut (*rig.device).frame, FrameRecord::new((*rig.device).alloc));
        let index = frame.index;
        (*rig.device)
            .offload_on_fault(index, "the test drives the fault's own steps")
            .expect("the offload after a fault");
        (*rig.device)
            .render_frame_on_soft(&mut frame)
            .expect("the frame that faulted re-renders on the reference tier");
        (*rig.device).frame = frame;
        (*rig.device).frame_state = FrameState::Submitted;
    }

    /// A frame that faults is re-rendered out of its own draws and handed to the
    /// host as the reference tier renders it - not the frame before it, not an
    /// empty one.
    #[test]
    fn the_frame_that_faults_is_rerendered_out_of_its_own_draws() {
        if !d3d11_usable() {
            eprintln!(
                "no usable D3D11 device on this machine; the fault path is a hardware story, \
                 so this leg is skipped rather than approximated"
            );
            return;
        }
        let mut prior = vec![0u8; (W * H * 4) as usize];
        let mut faulted = vec![0u8; (W * H * 4) as usize];
        let mut clean = vec![0u8; (W * H * 4) as usize];
        let mut prior_reference = vec![0u8; (W * H * 4) as usize];
        unsafe {
            // A hardware device with a frame behind it when the next one faults.
            let mut hardware = Rig::new(abi::backend::D3D11, 1);
            assert_eq!(hardware.record(LARGE.0, LARGE.1), abi::result::OK);
            assert_eq!(hardware.submit(), abi::result::OK);
            assert_eq!(hardware.present(&mut prior), abi::result::OK);

            // The faulting frame: its own draws, recorded and submitted - the
            // replay that builds the frame's draw list is inside Submit, so a
            // frame that never submits has an empty list to re-render. The GPU
            // renders it here; only its *failure* is what no test can produce.
            assert_eq!(hardware.record(SMALL.0, SMALL.1), abi::result::OK);
            assert_eq!(hardware.submit(), abi::result::OK);
            fault_and_rerender(&mut hardware);
            assert_eq!(
                (*hardware.device).backend_id(),
                abi::backend::SOFT_CPU,
                "the device that faulted did not move to the reference tier"
            );
            assert_eq!(hardware.present(&mut faulted), abi::result::OK);

            // The same frame, rendered cleanly on a reference-tier device with
            // the same frame history.
            let mut reference = Rig::new(abi::backend::SOFT_CPU, 2);
            assert_eq!(reference.record(LARGE.0, LARGE.1), abi::result::OK);
            assert_eq!(reference.submit(), abi::result::OK);
            assert_eq!(reference.present(&mut prior_reference), abi::result::OK);
            assert_eq!(reference.record(SMALL.0, SMALL.1), abi::result::OK);
            assert_eq!(reference.submit(), abi::result::OK);
            assert_eq!(reference.present(&mut clean), abi::result::OK);
        }

        // Non-vacuity: the frame before the fault drew the *other* triangle, so
        // a stale or accumulated re-render cannot produce the same pixels.
        assert!(lit(&prior, false) > 0, "the frame before the fault drew nothing on the right");
        assert_eq!(lit(&prior, true), 0, "the frame before the fault had already drawn the left half");
        assert_eq!(lit(&prior_reference, false), lit(&prior, false), "the two tiers disagree about the frame before the fault");
        assert!(lit(&faulted, true) > 0, "the frame that faulted lost its own geometry");
        assert_eq!(
            lit(&faulted, false),
            0,
            "the frame that faulted presented the frame before it as well"
        );
        assert_eq!(
            faulted, clean,
            "the frame the hardware could not finish is not the frame the reference tier renders \
             for the same draws"
        );
    }
}
