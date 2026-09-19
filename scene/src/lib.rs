//! Backend-agnostic frame structure: passes, the resource graph, barriers, and
//! world revisions.
//!
//! ReconL is not an engine, so this is deliberately thin. It answers three
//! questions a renderer must answer before it can be a *renderer* rather than a
//! pile of draw calls:
//!
//! 1. **What passes exist, in what order, and what do they touch?**
//!    [`PassList`] and [`ResourceUse`], which is also where barriers come from.
//! 2. **What has changed since last frame?** [`WorldRevision`]: bumping it
//!    retires cached static shadow cascades and cached static uploads in O(1)
//!    instead of re-uploading the world.
//! 3. **What is static?** [`GeometryClass`], which decides whether a draw is
//!    eligible for the cached static cascade at all.

use reconl_core::error::{Code, Result};
use reconl_core::log_debug;

/// Resource kinds a pass can read or write. Kept as plain numbers so the ABI can
/// name them without the header having to include this crate.
pub const RES_TARGET_COLOR: u32 = 1;
pub const RES_TARGET_DEPTH: u32 = 2;
pub const RES_SHADOW_MAP: u32 = 3;
pub const RES_TEXTURE: u32 = 4;
pub const RES_VERTEX_BUFFER: u32 = 5;
pub const RES_INDEX_BUFFER: u32 = 6;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug)]
pub struct ResourceUse {
    pub kind: u32,
    pub index: u32,
    pub access: Access,
}

impl ResourceUse {
    pub const fn read(kind: u32, index: u32) -> Self {
        Self { kind, index, access: Access::Read }
    }
    pub const fn write(kind: u32, index: u32) -> Self {
        Self { kind, index, access: Access::Write }
    }
}

/// A dependency that has to be resolved before a pass runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Barrier {
    pub kind: u32,
    pub index: u32,
    /// Pass index that produced the last write, or `u32::MAX` when the resource
    /// starts the frame in an undefined state.
    pub after_pass: u32,
    pub to_write: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum PassKind {
    /// Depth-only shadow cascade render.
    Shadow { cascade: u32 },
    /// Colour pass with shading and shadow lookup.
    Scene,
    /// Copies the colour target to host memory (headless present).
    Readback,
}

#[derive(Clone, Copy, Debug)]
pub struct Pass {
    pub kind: PassKind,
    pub uses: [Option<ResourceUse>; 8],
    pub use_count: u32,
}

impl Pass {
    pub fn new(kind: PassKind) -> Self {
        Self { kind, uses: [None; 8], use_count: 0 }
    }

    pub fn using(mut self, use_: ResourceUse) -> Self {
        if (self.use_count as usize) < self.uses.len() {
            self.uses[self.use_count as usize] = Some(use_);
            self.use_count += 1;
        }
        self
    }

    pub fn touches(&self, other: &Pass) -> Option<ResourceUse> {
        for a in self.uses.iter().take(self.use_count as usize).flatten() {
            for b in other.uses.iter().take(other.use_count as usize).flatten() {
                if a.kind == b.kind && a.index == b.index {
                    if a.access != Access::Read || b.access != Access::Read {
                        return Some(*b);
                    }
                }
            }
        }
        None
    }
}

#[derive(Default)]
pub struct PassList {
    passes: Vec<Pass>,
    barriers: Vec<Barrier>,
    /// Last use of each (kind, index): the pass, and whether that use wrote it.
    ///
    /// The *access*, not just the writer, has to be remembered, because the
    /// hazards are not symmetric: two reads in a row need nothing, a read after
    /// a write and a write after a read both need a barrier.
    last_use: Vec<(u32, u32, u32, bool)>,
}

impl PassList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.passes.clear();
        self.barriers.clear();
        self.last_use.clear();
    }

    pub fn push(&mut self, pass: Pass) {
        let pass_index = self.passes.len() as u32;
        for use_ in pass.uses.iter().take(pass.use_count as usize).flatten() {
            let writes = use_.access != Access::Read;
            match self
                .last_use
                .iter_mut()
                .find(|(k, i, _, _)| *k == use_.kind && *i == use_.index)
            {
                Some(slot) => {
                    let (_, _, previous_pass, previous_wrote) = *slot;
                    if previous_pass != pass_index && (previous_wrote || writes) {
                        self.barriers.push(Barrier {
                            kind: use_.kind,
                            index: use_.index,
                            after_pass: previous_pass,
                            to_write: writes,
                        });
                    }
                    slot.2 = pass_index;
                    slot.3 = writes;
                }
                None => self.last_use.push((use_.kind, use_.index, pass_index, writes)),
            }
        }
        self.passes.push(pass);
    }

    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }

    pub fn barriers(&self) -> &[Barrier] {
        &self.barriers
    }

    pub fn len(&self) -> usize {
        self.passes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// A pass order that must hold for correctness, checked rather than assumed:
    /// every shadow pass for a cascade has to come before the pass that samples
    /// it.
    pub fn validate(&self) -> Result<()> {
        let mut shadow_written: Vec<u32> = Vec::new();
        for (index, pass) in self.passes.iter().enumerate() {
            match pass.kind {
                PassKind::Shadow { cascade } => shadow_written.push(cascade),
                PassKind::Scene => {
                    let samples_shadows = pass.uses.iter().take(pass.use_count as usize).flatten().any(|u| {
                        u.kind == RES_SHADOW_MAP && u.access == Access::Read
                    });
                    if samples_shadows && shadow_written.is_empty() {
                        return reconl_core::err!(
                            Code::NotReady,
                            "scene pass {} samples shadow maps but no shadow pass precedes it",
                            index
                        );
                    }
                }
                PassKind::Readback => {}
            }
        }
        Ok(())
    }
}

/// World revision and geometry class: the invalidation hook the static-cascade
/// cache is built on.
#[derive(Clone, Copy, Default)]
pub struct WorldRevision {
    revision: u64,
    shadow_relevant: u64,
    static_geometry: u64,
    pub bumps: u64,
}

impl WorldRevision {
    pub fn new() -> Self {
        Self { revision: 1, shadow_relevant: 1, static_geometry: 1, bumps: 0 }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Revision used in shadow cache keys: only changes when shadow-relevant
    /// geometry moved, so adding a UI quad does not throw away the cascade.
    pub fn shadow_revision(&self) -> u64 {
        self.shadow_relevant
    }

    pub fn static_geometry_revision(&self) -> u64 {
        self.static_geometry
    }

    /// Records a change. `flags == 0` is "something changed that no shadow cares
    /// about" - a UI quad, a material, a camera - and must not retire a cascade,
    /// because retiring one costs a full re-render of the static set. A light
    /// change does retire them: it moves the whole cascade fit.
    pub fn bump(&mut self, flags: u32) {
        self.revision = self.revision.wrapping_add(1);
        if flags & (BUMP_SHADOW_RELEVANT | BUMP_LIGHTS) != 0 {
            self.shadow_relevant = self.shadow_relevant.wrapping_add(1);
        }
        if flags & BUMP_STATIC_GEOMETRY != 0 {
            self.static_geometry = self.static_geometry.wrapping_add(1);
        }
        self.bumps += 1;
        log_debug!(
            "world revision -> {} (shadow {}, static {}, flags {})",
            self.revision,
            self.shadow_relevant,
            self.static_geometry,
            flags
        );
    }
}

pub const BUMP_SHADOW_RELEVANT: u32 = 1;
pub const BUMP_STATIC_GEOMETRY: u32 = 2;
pub const BUMP_LIGHTS: u32 = 4;

/// Whether a draw is eligible for the cached static cascade.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GeometryClass {
    /// Never moves. Renders into the cached cascade, which survives across frames.
    Static,
    /// Moves or animates. Rendered every shadow pass.
    Dynamic,
}

impl GeometryClass {
    pub fn from_flags(flags: u32) -> Self {
        if flags & 1 != 0 {
            GeometryClass::Dynamic
        } else {
            GeometryClass::Static
        }
    }

    pub fn is_static(self) -> bool {
        matches!(self, GeometryClass::Static)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barriers_appear_when_a_pass_writes_what_another_touched() {
        let mut list = PassList::new();
        list.push(Pass::new(PassKind::Shadow { cascade: 0 }).using(ResourceUse::write(RES_SHADOW_MAP, 0)));
        assert!(list.barriers().is_empty(), "first use of a resource needs no barrier");
        list.push(Pass::new(PassKind::Scene).using(ResourceUse::read(RES_SHADOW_MAP, 0)));
        // A write followed by a read from a different pass is a dependency.
        assert_eq!(list.barriers().len(), 1);
        assert_eq!(list.barriers()[0].kind, RES_SHADOW_MAP);
        assert_eq!(list.barriers()[0].after_pass, 0);
    }

    #[test]
    fn reading_the_same_resource_twice_makes_no_barrier() {
        let mut list = PassList::new();
        list.push(Pass::new(PassKind::Scene).using(ResourceUse::read(RES_TEXTURE, 3)));
        list.push(Pass::new(PassKind::Scene).using(ResourceUse::read(RES_TEXTURE, 3)));
        assert!(list.barriers().is_empty());
    }

    #[test]
    fn a_scene_pass_sampling_shadows_before_any_shadow_pass_is_rejected() {
        let mut list = PassList::new();
        list.push(Pass::new(PassKind::Scene).using(ResourceUse::read(RES_SHADOW_MAP, 0)));
        let err = list.validate().unwrap_err();
        assert_eq!(err.code, Code::NotReady);

        let mut ok = PassList::new();
        ok.push(Pass::new(PassKind::Shadow { cascade: 0 }).using(ResourceUse::write(RES_SHADOW_MAP, 0)));
        ok.push(Pass::new(PassKind::Scene).using(ResourceUse::read(RES_SHADOW_MAP, 0)));
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn bumping_a_world_revision_retires_caches_in_o_one_step() {
        let mut rev = WorldRevision::new();
        let shadow = rev.shadow_revision();
        let static_geo = rev.static_geometry_revision();

        // A UI quad changing is not shadow relevant.
        rev.bump(0);
        assert_ne!(rev.revision(), 1);
        assert_eq!(rev.shadow_revision(), shadow, "a plain bump must not retire cascades");
        assert_eq!(rev.static_geometry_revision(), static_geo);

        // A shadow-casting object moving is.
        rev.bump(BUMP_SHADOW_RELEVANT);
        assert_ne!(rev.shadow_revision(), shadow);

        // Static geometry changing retires the cached cascade differently.
        let after = rev.shadow_revision();
        rev.bump(BUMP_STATIC_GEOMETRY);
        assert_ne!(rev.static_geometry_revision(), static_geo);
        assert_eq!(rev.shadow_revision(), after, "a static edit is not a light change");
        assert_eq!(rev.bumps, 3);
    }

    #[test]
    fn geometry_class_reads_the_flag() {
        assert_eq!(GeometryClass::from_flags(0), GeometryClass::Static);
        assert_eq!(GeometryClass::from_flags(1), GeometryClass::Dynamic);
        assert!(GeometryClass::Static.is_static());
    }

    #[test]
    fn clear_resets_the_dependency_tracking() {
        let mut list = PassList::new();
        list.push(Pass::new(PassKind::Scene).using(ResourceUse::write(RES_TARGET_COLOR, 0)));
        list.clear();
        assert!(list.is_empty());
        list.push(Pass::new(PassKind::Scene).using(ResourceUse::write(RES_TARGET_COLOR, 0)));
        assert!(list.barriers().is_empty(), "the first write after a clear is not a dependency");
    }
}
