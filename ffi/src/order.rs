//! The order each frame call considers its refusals in - one table per entry
//! point, one function per question, and one runner that asks them in the
//! table's order and stops at the first refusal.
//!
//! `include/reconl/reconl.h` states this order to a host ("the order its answers
//! are decided in is fixed, so a host reads one code per call"), and this is
//! where that order is decided. A call's precedence is a value here rather than
//! the sequence its checks happen to be written in, so a reorder is a change to
//! one array and the code a host reads can be read off the table.
//!
//! Three things the tables are load-bearing for:
//!
//! * **Which question comes first, per call.** `reconlSubmit` asks its argument
//!   question before its state question and `reconlPresent` asks the state first;
//!   both are deliberate, both are documented, and both are visible side by side
//!   here instead of in two entry points a screen apart.
//! * **Nothing dereferences a host pointer before the question that owns it.** A
//!   call that has nothing to do with a descriptor answers before it reads one,
//!   which is why a null `desc` is safe in every state and on every tier: the
//!   null test is a pointer comparison and the header is only read by the
//!   argument question.
//! * **Where the commit point is.** [`Step::Commit`] is where a call takes the
//!   frame it is working on. Everything the table asks before it leaves the
//!   frame exactly as it was; a refusal after it ends the frame and counts in
//!   `ReconLStats.frames_dropped`. That boundary is the same rule as "a call
//!   refused for its arguments changes nothing", and it is a step, not a comment.

use crate::entry::{child, FrameOwner};
use crate::handle::Kind;
use crate::layout::host_row_layout;
use crate::{abi, CommandListHandle, DeviceHandle, FenceHandle, FrameState, SwapchainHandle};
use reconl_core::check_header;
use reconl_core::err;
use reconl_core::error::{Code, Result};

/// One step of a call's order.
pub(crate) enum Step<G> {
    /// A question the call asks. Where it sits in the table is the whole point.
    Ask(G),
    /// The point where the call takes the frame it is working on.
    Commit,
}

/// A call whose refusals are ordered by a table.
pub(crate) trait Call {
    /// This call's questions, named so each table is exhaustive over its own
    /// vocabulary - a question in the table that the call does not answer does
    /// not compile.
    type Gate: Copy;

    /// Answers one question. `Ok` means "this one does not refuse".
    fn ask(&mut self, gate: Self::Gate) -> Result<()>;

    /// Takes the frame. Calls whose table has no [`Step::Commit`] take nothing -
    /// a generated frame is not a frame, so it commits nothing - and leave this
    /// as it is.
    fn commit(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Asks `order`'s questions in order and stops at the first refusal. **Nothing
/// after a refusal runs**, which is the whole reason the order is written down.
pub(crate) fn walk<C: Call>(order: &[Step<C::Gate>], call: &mut C) -> Result<()> {
    for step in order {
        match step {
            Step::Ask(gate) => call.ask(*gate)?,
            Step::Commit => call.commit()?,
        }
    }
    Ok(())
}

/// The success side of a commit: the frame is one to keep, so dropping the owner
/// does not drop the frame. A no-op when the table never committed.
pub(crate) fn keep_the_frame(frame: &mut Option<FrameOwner>) {
    if let Some(frame) = frame.as_mut() {
        frame.keep = true;
    }
}

// ----------------------------------------------------------------- begin frame

/// The questions `reconlBeginFrame` asks, and the order it asks them in: the
/// descriptor pointer, then the state.
///
/// The descriptor is looked at *before* the state - the opposite of
/// `reconlPresent` - so a host that begins a frame while one is open after
/// passing a null descriptor is told about the null descriptor. Nothing here
/// dereferences anything: the first question compares a pointer.
///
/// No commit step, and not because this call does not commit: it opens the frame
/// itself, and every refusal the descriptor can earn - its header, its
/// dimensions, its lights, its camera, the reservation for its targets - happens
/// before the statement that marks the frame open, which is the last one in the
/// entry point. A frame that cannot be built therefore leaves the device able to
/// begin another, and there is no window in which a frame exists in pieces.
#[derive(Clone, Copy)]
pub(crate) enum BeginGate {
    Descriptor,
    State,
}

pub(crate) const BEGIN_FRAME: &[Step<BeginGate>] = &[
    Step::Ask(BeginGate::Descriptor),
    Step::Ask(BeginGate::State),
];

pub(crate) struct BeginFrame<'a> {
    device: &'a mut DeviceHandle,
    desc: *mut abi::ReconLFrameDesc,
}

impl<'a> BeginFrame<'a> {
    pub(crate) fn new(device: &'a mut DeviceHandle, desc: *mut abi::ReconLFrameDesc) -> Self {
        Self { device, desc }
    }

    /// What the order left: the device, for the frame this call is about to
    /// build.
    pub(crate) fn into_parts(self) -> &'a mut DeviceHandle {
        self.device
    }
}

impl Call for BeginFrame<'_> {
    type Gate = BeginGate;

    fn ask(&mut self, gate: BeginGate) -> Result<()> {
        match gate {
            BeginGate::Descriptor => {
                if self.desc.is_null() {
                    err!(Code::InvalidArgument, "null frame descriptor")
                } else {
                    Ok(())
                }
            }
            BeginGate::State => {
                if self.device.frame_state == FrameState::Idle {
                    Ok(())
                } else {
                    err!(Code::FrameInProgress, "a frame is already open")
                }
            }
        }
    }

}

// ---------------------------------------------------------------------- submit

/// The questions `reconlSubmit` asks, in order: the command list pointer, the
/// frame state, then the handles it was given.
///
/// The list comes before the state on purpose, and the state before the handles:
/// the handles are the only checks that touch another object, so they run once
/// there is a frame to submit into and a list to read. Both handle checks happen
/// before the commit, so a handle from another device costs the host nothing.
#[derive(Clone, Copy)]
pub(crate) enum SubmitGate {
    List,
    State,
    Handles,
}

pub(crate) const SUBMIT: &[Step<SubmitGate>] = &[
    Step::Ask(SubmitGate::List),
    Step::Ask(SubmitGate::State),
    Step::Ask(SubmitGate::Handles),
    Step::Commit,
];

pub(crate) struct Submit<'a> {
    device: &'a mut DeviceHandle,
    list: *const CommandListHandle,
    fence: *mut FenceHandle,
    frame: Option<FrameOwner>,
}

impl<'a> Submit<'a> {
    pub(crate) fn new(
        device: &'a mut DeviceHandle,
        list: *const CommandListHandle,
        fence: *mut FenceHandle,
    ) -> Self {
        Self { device, list, fence, frame: None }
    }

    /// What the order left: the device, and the frame the commit took.
    pub(crate) fn into_parts(self) -> (&'a mut DeviceHandle, Option<FrameOwner>) {
        (self.device, self.frame)
    }
}

impl Call for Submit<'_> {
    type Gate = SubmitGate;

    fn ask(&mut self, gate: SubmitGate) -> Result<()> {
        match gate {
            SubmitGate::List => {
                if self.list.is_null() {
                    err!(Code::InvalidArgument, "null command list")
                } else {
                    Ok(())
                }
            }
            SubmitGate::State => {
                if self.device.frame_state == FrameState::Open {
                    Ok(())
                } else {
                    err!(Code::NoFrame, "no frame is open; call reconlBeginFrame first")
                }
            }
            SubmitGate::Handles => {
                child!(
                    self.device,
                    self.list as *const CommandListHandle as *mut CommandListHandle,
                    Kind::CommandList,
                    "command list"
                );
                if !self.fence.is_null() {
                    // Both handles are checked before the frame is committed: a
                    // handle from another device is a caller bug that must not
                    // cost the host the frame it is about to render, so the frame
                    // stays open and the call can be retried with a valid one.
                    child!(self.device, self.fence, Kind::Fence, "fence");
                }
                Ok(())
            }
        }
    }

    fn commit(&mut self) -> Result<()> {
        self.frame = Some(FrameOwner::new(self.device as *mut DeviceHandle));
        Ok(())
    }
}

// --------------------------------------------------------------------- present

/// The questions `reconlPresent` asks, in order: the swapchain handle, the frame
/// state, the commit, then the host's arguments.
///
/// The state comes first here because a present with nothing submitted has
/// nothing to present whatever its descriptor says - and the descriptor is not
/// read until after the commit, which is the documented rule for this call: a
/// present that cannot be delivered consumes the frame it was given rather than
/// leaving the device in `Submitted` until the host happens to present
/// successfully. Hence the table's order: everything before the commit leaves
/// the frame, everything after it ends the frame.
#[derive(Clone, Copy)]
pub(crate) enum PresentGate {
    Handles,
    State,
    Arguments,
}

pub(crate) const PRESENT: &[Step<PresentGate>] = &[
    Step::Ask(PresentGate::Handles),
    Step::Ask(PresentGate::State),
    Step::Commit,
    Step::Ask(PresentGate::Arguments),
];

/// What the argument question read out of the host's present descriptor.
///
/// A pointer and a size rather than a slice: the slice is made where the bytes
/// are written, so the borrow of the host's buffer is no longer than the read.
#[derive(Clone, Copy, Default)]
pub(crate) struct PresentArguments {
    pub(crate) out_pixels: *mut core::ffi::c_void,
    pub(crate) out_size: u64,
    pub(crate) out_pitch: u32,
    pub(crate) flip: u32,
}

pub(crate) struct Present<'a> {
    device: &'a mut DeviceHandle,
    swapchain: *mut SwapchainHandle,
    desc: *mut abi::ReconLPresentDesc,
    frame: Option<FrameOwner>,
    pulled: PresentArguments,
}

impl<'a> Present<'a> {
    pub(crate) fn new(
        device: &'a mut DeviceHandle,
        swapchain: *mut SwapchainHandle,
        desc: *mut abi::ReconLPresentDesc,
    ) -> Self {
        Self { device, swapchain, desc, frame: None, pulled: PresentArguments::default() }
    }

    /// What the order left: the device, what the argument question read, and the
    /// frame this call took at its commit point.
    pub(crate) fn into_parts(
        self,
    ) -> (&'a mut DeviceHandle, PresentArguments, Option<FrameOwner>) {
        (self.device, self.pulled, self.frame)
    }
}

impl Call for Present<'_> {
    type Gate = PresentGate;

    fn ask(&mut self, gate: PresentGate) -> Result<()> {
        match gate {
            PresentGate::Handles => {
                if self.swapchain.is_null() {
                    return err!(Code::InvalidArgument, "null swapchain");
                }
                child!(self.device, self.swapchain, Kind::Swapchain, "swapchain");
                Ok(())
            }
            PresentGate::State => {
                if self.device.frame_state == FrameState::Submitted {
                    Ok(())
                } else {
                    err!(Code::NoFrame, "there is no submitted frame to present")
                }
            }
            PresentGate::Arguments => {
                // A present with no descriptor is a present with no readback:
                // legal, and the frame is still delivered. The header is only
                // read when there is one to read.
                let (out_pixels, out_size, out_pitch, flip) = if self.desc.is_null() {
                    (core::ptr::null_mut(), 0u64, 0u32, 0u32)
                } else {
                    let d = unsafe {
                        check_header::<abi::ReconLPresentDesc>(
                            self.desc as *const reconl_core::StructHeader,
                            abi::struct_type::PRESENT_DESC,
                            core::mem::size_of::<abi::ReconLPresentDesc>() as u32,
                            "ReconLPresentDesc",
                        )?
                    };
                    (d.out_pixels, d.out_pixels_size, d.out_row_pitch, d.flip)
                };
                self.pulled = PresentArguments { out_pixels, out_size, out_pitch, flip };
                Ok(())
            }
        }
    }

    fn commit(&mut self) -> Result<()> {
        // Past this point the frame is consumed: this call either presents it or
        // ends it. A present that fails for any later reason - an unreadable
        // descriptor, a buffer too small for the frame - therefore leaves the
        // device able to open the next frame rather than stuck in `Submitted`
        // until the host happens to present successfully. Same contract as a
        // failed submit, and the same counter.
        self.frame = Some(FrameOwner::new(self.device as *mut DeviceHandle));
        Ok(())
    }
}

// ----------------------------------------------------------- present generated

/// The questions `reconlPresentGenerated` asks, in order: the handles, the tier,
/// whether there is anything to generate from, then the descriptor and `ahead`.
///
/// Deliberately not gated on the frame state machine - a generated frame is not
/// a frame: it renders nothing, consumes nothing and can fail without leaving
/// the device anywhere it cannot render from - and so deliberately with no commit
/// step: it takes no frame. What it does gate on is the *history*: until a
/// presented frame asked to be kept there is nothing to generate from, and every
/// descriptor and every `ahead` answers `RECONL_ERR_NO_FRAME` rather than being
/// read. That is the same shape `reconlPresent` has with no frame submitted, and
/// the reason the header can promise one code per call.
#[derive(Clone, Copy)]
pub(crate) enum GeneratedGate {
    Handles,
    Tier,
    History,
    Arguments,
}

pub(crate) const PRESENT_GENERATED: &[Step<GeneratedGate>] = &[
    Step::Ask(GeneratedGate::Handles),
    Step::Ask(GeneratedGate::Tier),
    Step::Ask(GeneratedGate::History),
    Step::Ask(GeneratedGate::Arguments),
];

/// What the argument question read: where the image goes, under which layout,
/// and how big the frame it is warped from is.
#[derive(Clone, Copy, Default)]
pub(crate) struct GeneratedArguments {
    pub(crate) out_pixels: *mut core::ffi::c_void,
    pub(crate) out_size: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pitch: u32,
    pub(crate) flip: u32,
}

pub(crate) struct PresentGenerated<'a> {
    device: &'a mut DeviceHandle,
    swapchain: *mut SwapchainHandle,
    desc: *mut abi::ReconLPresentDesc,
    ahead: f32,
    pulled: GeneratedArguments,
}

impl<'a> PresentGenerated<'a> {
    pub(crate) fn new(
        device: &'a mut DeviceHandle,
        swapchain: *mut SwapchainHandle,
        desc: *mut abi::ReconLPresentDesc,
        ahead: f32,
    ) -> Self {
        Self { device, swapchain, desc, ahead, pulled: GeneratedArguments::default() }
    }

    /// What the order left: the device, and what the argument question read.
    pub(crate) fn into_parts(self) -> (&'a mut DeviceHandle, GeneratedArguments) {
        (self.device, self.pulled)
    }
}

impl Call for PresentGenerated<'_> {
    type Gate = GeneratedGate;

    fn ask(&mut self, gate: GeneratedGate) -> Result<()> {
        match gate {
            GeneratedGate::Handles => {
                if self.swapchain.is_null() {
                    return err!(Code::InvalidArgument, "null swapchain");
                }
                child!(self.device, self.swapchain, Kind::Swapchain, "swapchain");
                Ok(())
            }
            GeneratedGate::Tier => {
                if self.device.backend.can_generate() {
                    Ok(())
                } else {
                    err!(
                        Code::NotSupported,
                        "this tier keeps no depth to generate frames from; a generated frame needs a rendered one"
                    )
                }
            }
            GeneratedGate::History => {
                if self.device.framegen.ready {
                    Ok(())
                } else {
                    err!(
                        Code::NoFrame,
                        "no frame is available to generate from: present one that asks for generation (ReconLFrameGenDesc.enabled)"
                    )
                }
            }
            GeneratedGate::Arguments => {
                if self.desc.is_null() {
                    return err!(Code::InvalidArgument, "null present descriptor");
                }
                // SAFETY: the caller passed a present descriptor for this call.
                let d = unsafe {
                    check_header::<abi::ReconLPresentDesc>(
                        self.desc as *const reconl_core::StructHeader,
                        abi::struct_type::PRESENT_DESC,
                        core::mem::size_of::<abi::ReconLPresentDesc>() as u32,
                        "ReconLPresentDesc",
                    )?
                };
                if d.out_pixels.is_null() {
                    return err!(Code::InvalidArgument, "a generated frame needs out_pixels to be written to");
                }
                // The layout the host asked for, laid over the frame being
                // warped, and then the look-ahead: the buffer rules first, so a
                // host that got both wrong is told about the one it can fix
                // without thinking about the image.
                let (width, height) = (self.device.framegen.width, self.device.framegen.height);
                let pitch = host_row_layout(width, height, d.out_pixels_size, d.out_row_pitch)?;
                reconl_raster::framegen::check_ahead(self.ahead)?;
                self.pulled = GeneratedArguments {
                    out_pixels: d.out_pixels,
                    out_size: d.out_pixels_size,
                    width,
                    height,
                    pitch,
                    flip: d.flip,
                };
                Ok(())
            }
        }
    }
}
