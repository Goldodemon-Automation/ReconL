//! Frame policy: what counts as a frame.
//!
//! ReconL's whole claim is about what *actually* got drawn, so "we presented a
//! frame" is never assumed - it is classified. An empty frame (no geometry in,
//! nothing shaded) is legal, common in a scene editor with nothing selected, and
//! must never be counted as output: a benchmark that presents empty frames would
//! report a beautiful frame time for doing nothing.
//!
//! The policy is deliberately boring and testable:
//!
//! * an empty frame is counted in `empty_frames`, never in `presented_frames`;
//! * `require_geometry` turns an empty frame into `RECONL_EMPTY_FRAME`, which is
//!   how CI catches "the renderer silently stopped drawing";
//! * `idle_after` consecutive empty frames raise `idle`, which is what a host
//!   uses to stop calling `reconlSubmit`, and which is what makes the T3/T4
//!   "freeze and skip" paths observable from the outside.
//!
//! The classifier never looks at pixels. Pixels are the rasteriser's business;
//! this module only answers "was there work, and did it happen".

use reconl_core::error::{Code, Error, Result};

/// How a frame was classified.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Emptiness {
    /// No draws, or draws that contributed no primitives.
    Empty,
    NonEmpty,
}

#[derive(Clone, Copy, Debug)]
pub struct FramePolicy {
    /// An empty frame is an error rather than a no-op.
    pub require_geometry: bool,
    /// Consecutive empty frames that mean "idle".
    pub idle_after: u32,
    /// Skip the colour pass entirely when nothing can be visible.
    pub skip_empty_pass: bool,
    /// Do the shadow pass on empty frames (so a host can precompute statics).
    pub shadow_on_empty_frames: bool,
}

impl Default for FramePolicy {
    fn default() -> Self {
        Self {
            require_geometry: false,
            idle_after: 30,
            skip_empty_pass: true,
            shadow_on_empty_frames: false,
        }
    }
}

/// Classifier state, owned by the device and reported in the stats blob.
#[derive(Clone, Copy, Default, Debug)]
pub struct FrameClassifier {
    pub policy: FramePolicy,
    pub consecutive_empty: u32,
    pub empty_frames: u64,
    pub non_empty_frames: u64,
    pub idle_events: u64,
    /// Frame index of the last frame that had work, or `None` before any.
    pub last_work_frame: Option<u64>,
    pub idle: bool,
}

impl FrameClassifier {
    pub fn new(policy: FramePolicy) -> Self {
        Self { policy, ..Self::default() }
    }

    /// Classifies a frame from what the host actually asked for.
    pub fn classify(&self, draws: usize, triangles: u64) -> Emptiness {
        if draws == 0 || triangles == 0 {
            Emptiness::Empty
        } else {
            Emptiness::NonEmpty
        }
    }

    /// Records the outcome of presenting. Returns the error when the policy
    /// says an empty frame is not acceptable.
    pub fn record(&mut self, emptiness: Emptiness, frame_index: u64) -> Result<()> {
        if emptiness == Emptiness::Empty {
            self.empty_frames += 1;
            self.consecutive_empty = self.consecutive_empty.saturating_add(1);
            let threshold = self.policy.idle_after.max(1);
            if self.consecutive_empty >= threshold && !self.idle {
                self.idle = true;
                self.idle_events += 1;
            }
            if self.policy.require_geometry {
                return Err(Error::new(
                    Code::EmptyFrame,
                    "the frame contained no geometry and the frame policy requires some",
                ));
            }
            Ok(())
        } else {
            self.non_empty_frames += 1;
            self.consecutive_empty = 0;
            self.idle = false;
            self.last_work_frame = Some(frame_index);
            Ok(())
        }
    }

    /// True when the colour pass can be skipped without changing the output:
    /// nothing to draw, and the caller is not clearing to a visible colour that
    /// must still be produced.
    pub fn may_skip_pass(&self, emptiness: Emptiness, clears_anything: bool) -> bool {
        self.policy.skip_empty_pass && emptiness == Emptiness::Empty && !clears_anything
    }

    pub fn may_skip_shadow_pass(&self, emptiness: Emptiness) -> bool {
        emptiness == Emptiness::Empty && !self.policy.shadow_on_empty_frames
    }

    pub fn reset(&mut self) {
        let policy = self.policy;
        let idle = self.idle;
        *self = Self { policy, ..Self::default() };
        self.idle = idle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_frames_are_not_presented_but_are_counted() {
        let mut c = FrameClassifier::new(FramePolicy::default());
        assert_eq!(c.classify(0, 0), Emptiness::Empty);
        c.record(Emptiness::Empty, 0).unwrap();
        c.record(Emptiness::Empty, 1).unwrap();
        assert_eq!(c.empty_frames, 2);
        assert_eq!(c.non_empty_frames, 0);
        assert_eq!(c.last_work_frame, None);
    }

    #[test]
    fn a_draw_with_no_triangles_is_still_empty() {
        let c = FrameClassifier::new(FramePolicy::default());
        assert_eq!(c.classify(1, 0), Emptiness::Empty);
        assert_eq!(c.classify(0, 3), Emptiness::Empty);
        assert_eq!(c.classify(1, 1), Emptiness::NonEmpty);
    }

    #[test]
    fn idle_after_counts_consecutive_empty_frames_only() {
        let mut c = FrameClassifier::new(FramePolicy { idle_after: 3, ..Default::default() });
        for i in 0..2 {
            c.record(Emptiness::Empty, i).unwrap();
        }
        assert!(!c.idle, "two empty frames is not idle when the threshold is three");
        c.record(Emptiness::Empty, 2).unwrap();
        c.record(Emptiness::Empty, 3).unwrap();
        assert!(c.idle);
        assert_eq!(c.idle_events, 1);

        c.record(Emptiness::NonEmpty, 4).unwrap();
        assert!(!c.idle);
        assert_eq!(c.last_work_frame, Some(4));
        assert_eq!(c.consecutive_empty, 0);

        // Idling again is a second event, not a duplicate of the first.
        for i in 5..8 {
            c.record(Emptiness::Empty, i).unwrap();
        }
        assert_eq!(c.idle_events, 2);
    }

    #[test]
    fn require_geometry_rejects_the_empty_frame() {
        let mut c = FrameClassifier::new(FramePolicy { require_geometry: true, ..Default::default() });
        let err = c.record(Emptiness::Empty, 0).unwrap_err();
        assert_eq!(err.code, Code::EmptyFrame);
        // The frame was still counted as empty: refusing it is not the same as
        // pretending it did not happen.
        assert_eq!(c.empty_frames, 1);
    }

    #[test]
    fn empty_pass_skipping_respects_clears() {
        let c = FrameClassifier::new(FramePolicy::default());
        assert!(c.may_skip_pass(Emptiness::Empty, false));
        assert!(!c.may_skip_pass(Emptiness::Empty, true));
        assert!(!c.may_skip_pass(Emptiness::NonEmpty, false));
        let strict = FrameClassifier::new(FramePolicy { skip_empty_pass: false, ..Default::default() });
        assert!(!strict.may_skip_pass(Emptiness::Empty, false));
    }

    #[test]
    fn shadow_on_empty_frames_is_opt_in() {
        let default = FrameClassifier::new(FramePolicy::default());
        assert!(default.may_skip_shadow_pass(Emptiness::Empty));
        let eager = FrameClassifier::new(FramePolicy { shadow_on_empty_frames: true, ..Default::default() });
        assert!(!eager.may_skip_shadow_pass(Emptiness::Empty));
    }
}
