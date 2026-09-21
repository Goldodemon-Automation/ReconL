//! Encoding and submitting frames: the host side of the frame contract.
//!
//! A host does the same six things every frame - begin, reset the command list,
//! open a pass, push the transforms, draw, end the pass, submit, present - and
//! the order is not free: `BeginFrame` reserves the frame's targets, `Submit`
//! commits and renders it, `Present` produces the pixels. This module is that
//! order in one place, with the scene as input and pixels as output, so a tool
//! measures a frame rather than re-implementing one.
//!
//! It also reports where the host's own wall clock went. The device reports its
//! internal split (shadow / raster / bin / upload) through `ReconLStats.frame`,
//! which is the number to quote for stage attribution; the timings here are the
//! *boundaries* a host can see, and a backend whose inner stages are not
//! instrumented (the hardware path has no GPU timestamps) still has these.

use crate::device::Device;
use crate::scene::Scene;
use crate::{failed, hdr};
use reconl::abi;
use std::time::Instant;

/// Host-side wall time at the three ABI boundaries of a frame.
///
/// `begin` covers target reservation and the shadow system's planning, `submit`
/// the commit and whatever rendering happens inside it, and `present` the
/// readback into the host's buffer - which is the only place a presenting-to-
/// memory backend has to pay for pixels crossing the bus.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameCost {
    pub begin_ns: u64,
    pub submit_ns: u64,
    pub present_ns: u64,
}

impl FrameCost {
    pub fn total_ns(&self) -> u64 {
        self.begin_ns + self.submit_ns + self.present_ns
    }
}

/// What a renderer is built for: the frame size, where it presents, and how many
/// times it re-issues the scene's draws per frame.
///
/// `repeat` is here rather than passed per call because it is what the command
/// list is sized for, and a command list never grows inside a frame - recording
/// past its capacity is refused, not reallocated. `max_draws` sizes it for a
/// dynamic scene instead: a host that draws one instance per game object
/// records more draws than the scene has chunks, and says so up front.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub width: u32,
    pub height: u32,
    /// Selects the offscreen path a headless tool uses; a host with a real
    /// swapchain sets it to false.
    pub present_to_memory: bool,
    /// Draw the scene this many times per frame. Multiplies draw calls and
    /// shaded fragments without changing what the scene is.
    pub repeat: u32,
    /// The most draws one frame will record; 0 sizes for `repeat` whole scenes.
    pub max_draws: u32,
    /// Ask the device to keep each frame so `present_generated` can warp one
    /// forward. This is the toggle a game flips with its own quality settings:
    /// off, a frame keeps nothing and pays a depth readback it does not make;
    /// on, it keeps its pixels, its depth and its camera.
    pub framegen: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { width: 64, height: 64, present_to_memory: true, repeat: 1, max_draws: 0, framegen: false }
    }
}

/// One command's slot in a command list, as the header publishes it
/// (`RECONL_COMMAND_BYTES`). A list is created with exactly the slots a frame
/// needs, which is how a tool can scale its workload without tripping the
/// capacity it forgot to size.
const COMMAND_BYTES: u32 = 64;

/// Commands one frame records: the pass's begin and end, the view transform,
/// and per draw the pipeline, vertex buffer, index buffer, model and draw.
fn commands_per_frame(scene: &Scene, options: &Options) -> u32 {
    let draws = if options.max_draws > 0 {
        options.max_draws
    } else {
        options.repeat.max(1) * scene.chunks.len() as u32
    };
    draws * 5 + 3
}

/// The scene's device-side resources: the buffers, pipeline, command list and
/// swapchain that a frame reuses.
///
/// Owned handles are released on drop. The scene is *not* a resource: it is
/// data, so the same renderer draws a scene with shadows off and one with them
/// on without being rebuilt.
pub struct Renderer {
    device: *mut reconl::DeviceHandle,
    width: u32,
    height: u32,
    repeat: u32,
    framegen: bool,
    swapchain: *mut reconl::SwapchainHandle,
    pipeline: *mut reconl::PipelineHandle,
    commands: *mut reconl::CommandListHandle,
    buffers: Vec<(*mut reconl::BufferHandle, *mut reconl::BufferHandle)>,
}

impl Renderer {
    /// Creates the frame's resources for `scene`.
    pub fn new(device: &Device, scene: &Scene, options: Options) -> Result<Renderer, String> {
        let device_ptr = device.handle();
        let (width, height) = (options.width.max(1), options.height.max(1));
        let present_to_memory = options.present_to_memory;
        let framegen = options.framegen;
        // SAFETY: the device is live for this call; every descriptor is filled
        // from the Rust type and outlives the call that reads it.
        unsafe {
            let sd = abi::ReconLSwapchainDesc {
                base: hdr::<abi::ReconLSwapchainDesc>(),
                width,
                height,
                format: 1,
                image_count: 2,
                present_to_memory: u32::from(present_to_memory),
                depth_format: 1,
                flags: 0,
                reserved: 0,
            };
            let mut swapchain: *mut reconl::SwapchainHandle = core::ptr::null_mut();
            let r = reconl::reconlCreateSwapchain(device_ptr, &sd, &mut swapchain);
            if r != abi::result::OK {
                return Err(failed("reconlCreateSwapchain", r));
            }

            let pd = abi::ReconLPipelineDesc {
                base: hdr::<abi::ReconLPipelineDesc>(),
                shading: abi::shading::LAMBERT,
                blend: 0,
                cull: 0,
                depth_compare: 1,
                depth_write: 1,
                texture_slots: 0,
                texture_formats: [0; abi::RECONL_MAX_TEXTURE_SLOTS],
                receives_shadow: 1,
                casts_shadow: 1,
                flags: 0,
                reserved: 0,
                debug_name: core::ptr::null(),
            };
            let mut pipeline: *mut reconl::PipelineHandle = core::ptr::null_mut();
            let r = reconl::reconlCreatePipeline(device_ptr, &pd, &mut pipeline);
            if r != abi::result::OK {
                reconl::reconlRelease(swapchain as *mut core::ffi::c_void);
                return Err(failed("reconlCreatePipeline", r));
            }

            let cd = abi::ReconLCommandListDesc {
                base: hdr::<abi::ReconLCommandListDesc>(),
                capacity_bytes: commands_per_frame(scene, &options) * COMMAND_BYTES,
                reserved: 0,
                debug_name: core::ptr::null(),
            };
            let mut commands: *mut reconl::CommandListHandle = core::ptr::null_mut();
            let r = reconl::reconlCreateCommandList(device_ptr, &cd, &mut commands);
            if r != abi::result::OK {
                reconl::reconlRelease(swapchain as *mut core::ffi::c_void);
                reconl::reconlRelease(pipeline as *mut core::ffi::c_void);
                return Err(failed("reconlCreateCommandList", r));
            }

            let mut buffers = Vec::with_capacity(scene.chunks.len());
            for chunk in &scene.chunks {
                let verts = chunk.verts.as_slice();
                let vbd = abi::ReconLBufferDesc {
                    base: hdr::<abi::ReconLBufferDesc>(),
                    size_bytes: core::mem::size_of_val(verts) as u64,
                    usage: 1,
                    reserved: 0,
                    data: verts.as_ptr() as *const core::ffi::c_void,
                    data_size: core::mem::size_of_val(verts) as u64,
                    debug_name: core::ptr::null(),
                };
                let mut vb: *mut reconl::BufferHandle = core::ptr::null_mut();
                let r = reconl::reconlCreateBuffer(device_ptr, &vbd, &mut vb);
                if r != abi::result::OK {
                    release_all(swapchain, pipeline, commands, &[]);
                    return Err(failed("reconlCreateBuffer (vertices)", r));
                }
                let indices = chunk.indices.as_slice();
                let ibd = abi::ReconLBufferDesc {
                    base: hdr::<abi::ReconLBufferDesc>(),
                    size_bytes: core::mem::size_of_val(indices) as u64,
                    usage: 2,
                    reserved: 0,
                    data: indices.as_ptr() as *const core::ffi::c_void,
                    data_size: core::mem::size_of_val(indices) as u64,
                    debug_name: core::ptr::null(),
                };
                let mut ib: *mut reconl::BufferHandle = core::ptr::null_mut();
                let r = reconl::reconlCreateBuffer(device_ptr, &ibd, &mut ib);
                if r != abi::result::OK {
                    reconl::reconlRelease(vb as *mut core::ffi::c_void);
                    release_all(swapchain, pipeline, commands, &buffers);
                    return Err(failed("reconlCreateBuffer (indices)", r));
                }
                buffers.push((vb, ib));
            }

            Ok(Renderer {
                device: device_ptr,
                width,
                height,
                repeat: options.repeat.max(1),
                framegen,
                swapchain,
                pipeline,
                commands,
                buffers,
            })
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Bytes a present buffer must hold for this renderer.
    pub fn frame_bytes(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 4
    }

    /// How many times a frame re-issues the scene's draws.
    pub fn repeat(&self) -> u32 {
        self.repeat
    }

    /// Renders one frame of `scene` and presents it into `pixels`.
    pub fn frame(&self, scene: &Scene, seed: u32, pixels: &mut [u8]) -> Result<FrameCost, String> {
        let plan: Vec<(usize, [f32; 16])> =
            (0..scene.chunks.len()).map(|i| (i, crate::math::IDENTITY)).collect();
        self.frame_instances(scene, seed, pixels, &plan)
    }

    /// Renders a frame of explicitly listed instances and presents it.
    ///
    /// One `(chunk, model)` pair per draw, in order, each with its own slot-1
    /// model matrix - the dynamic-scene path. A game moves its objects between
    /// frames by pure translation (the one transform both tiers guarantee for
    /// a moving, shadow-casting draw) without uploading a byte of geometry
    /// again; `frame()` is this with every chunk once at identity.
    ///
    /// The command list must have been sized for the longest plan through
    /// `Options::max_draws`; recording past the capacity is refused.
    pub fn frame_instances(
        &self,
        scene: &Scene,
        seed: u32,
        pixels: &mut [u8],
        instances: &[(usize, [f32; 16])],
    ) -> Result<FrameCost, String> {
        if pixels.len() < self.frame_bytes() {
            return Err(format!(
                "present buffer is {} bytes, a {}x{} frame needs {}",
                pixels.len(),
                self.width,
                self.height,
                self.frame_bytes()
            ));
        }
        // SAFETY: the device and every handle are live; descriptors are filled
        // from the Rust types and outlive each call.
        unsafe {
            let light = scene.light.abi();
            let lights = abi::ReconLLightList {
                base: hdr::<abi::ReconLLightList>(),
                count: 1,
                reserved: 0,
                lights: &light,
            };
            let shadows = scene.shadows.abi();
            let camera = scene.camera.abi();
            let framegen = abi::ReconLFrameGenDesc {
                base: hdr::<abi::ReconLFrameGenDesc>(),
                enabled: u32::from(self.framegen),
                reserved: 0,
            };
            let fd = abi::ReconLFrameDesc {
                base: hdr::<abi::ReconLFrameDesc>(),
                width: self.width,
                height: self.height,
                seed,
                reserved: 0,
                lights: &lights,
                shadows: &shadows,
                camera: &camera,
                // A host's choice, per frame: the request is always declared,
                // with the toggle off unless the host asked for generated
                // frames, so switching it off costs a frame exactly what it
                // cost before the feature existed.
                framegen: &framegen,
            };

            let mut cost = FrameCost::default();

            let t0 = Instant::now();
            let r = reconl::reconlBeginFrame(self.device, &fd as *const _ as *mut _);
            cost.begin_ns = t0.elapsed().as_nanos() as u64;
            if r != abi::result::OK {
                return Err(failed("reconlBeginFrame", r));
            }

            let t1 = Instant::now();
            let rp = abi::ReconLRenderPassDesc {
                base: hdr::<abi::ReconLRenderPassDesc>(),
                color_count: 0,
                reserved: 0,
                color: [core::mem::zeroed(); abi::RECONL_MAX_ATTACHMENTS],
                depth: core::ptr::null_mut(),
                viewport_width: self.width,
                viewport_height: self.height,
                load_color: 1,
                load_depth: 1,
                clear_color: [0.06, 0.07, 0.10, 1.0],
                clear_depth: 0.0,
                stencil_clear: 0,
                reserved2: 0,
            };
            reconl::reconlCmdReset(self.commands);
            check(reconl::reconlCmdBeginRenderPass(self.commands, &rp), "reconlCmdBeginRenderPass")?;
            let aspect = self.width as f32 / self.height as f32;
            let view_proj = scene.camera.view_proj(aspect);
            check(
                reconl::reconlCmdPushConstants(
                    self.commands,
                    0,
                    view_proj.as_ptr() as *const core::ffi::c_void,
                    64,
                ),
                "reconlCmdPushConstants (view-proj)",
            )?;
            for (chunk_index, model) in instances {
                let (chunk, (vb, ib)) = match (scene.chunks.get(*chunk_index), self.buffers.get(*chunk_index)) {
                    (Some(c), Some(b)) => (c, b),
                    _ => {
                        reconl::reconlCmdEndRenderPass(self.commands);
                        return Err(format!(
                            "an instance names chunk {chunk_index} but the scene has {} chunks",
                            scene.chunks.len()
                        ));
                    }
                };
                check(reconl::reconlCmdSetPipeline(self.commands, self.pipeline), "reconlCmdSetPipeline")?;
                check(reconl::reconlCmdSetVertexBuffer(self.commands, 0, *vb, 0), "reconlCmdSetVertexBuffer")?;
                check(reconl::reconlCmdSetIndexBuffer(self.commands, *ib, 0, 1), "reconlCmdSetIndexBuffer")?;
                check(
                    reconl::reconlCmdPushConstants(self.commands, 1, model.as_ptr() as *const core::ffi::c_void, 64),
                    "reconlCmdPushConstants (model)",
                )?;
                check(
                    reconl::reconlCmdDrawIndexed(self.commands, chunk.indices.len() as u32, 0, 0),
                    "reconlCmdDrawIndexed",
                )?;
            }
            check(reconl::reconlCmdEndRenderPass(self.commands), "reconlCmdEndRenderPass")?;
            let r = reconl::reconlSubmit(self.device, self.commands, core::ptr::null_mut());
            cost.submit_ns = t1.elapsed().as_nanos() as u64;
            if r != abi::result::OK {
                return Err(failed("reconlSubmit", r));
            }

            let t2 = Instant::now();
            let mut prd = abi::ReconLPresentDesc {
                base: hdr::<abi::ReconLPresentDesc>(),
                out_pixels: pixels.as_mut_ptr() as *mut core::ffi::c_void,
                out_pixels_size: pixels.len() as u64,
                out_row_pitch: self.width * 4,
                out_format: 1,
                flip: 0,
            };
            let r = reconl::reconlPresent(self.device, self.swapchain, &mut prd);
            cost.present_ns = t2.elapsed().as_nanos() as u64;
            if r != abi::result::OK {
                return Err(failed("reconlPresent", r));
            }
            Ok(cost)
        }
    }

    /// Presents one *generated* image - the newest frame warped `ahead` frame
    /// intervals along its camera's motion - and returns what it cost.
    ///
    /// This is a host's other half of the schedule: it renders fewer frames and
    /// presents more images. Nothing is rendered here, so a failure leaves the
    /// device able to render; the caller gets the error and may present a real
    /// frame next, which is why this returns `Result` rather than a cost alone.
    pub fn present_generated(&self, pixels: &mut [u8], ahead: f32) -> Result<u64, String> {
        if pixels.len() < self.frame_bytes() {
            return Err(format!(
                "present buffer is {} bytes, a {}x{} frame needs {}",
                pixels.len(),
                self.width,
                self.height,
                self.frame_bytes()
            ));
        }
        // SAFETY: the device and swapchain are live, and the descriptor is
        // filled from Rust data that outlives the call.
        unsafe {
            let mut prd = abi::ReconLPresentDesc {
                base: hdr::<abi::ReconLPresentDesc>(),
                out_pixels: pixels.as_mut_ptr() as *mut core::ffi::c_void,
                out_pixels_size: pixels.len() as u64,
                out_row_pitch: self.width * 4,
                out_format: 1,
                flip: 0,
            };
            let started = Instant::now();
            let r = reconl::reconlPresentGenerated(self.device, self.swapchain, &mut prd, ahead);
            let took = started.elapsed().as_nanos() as u64;
            if r != abi::result::OK {
                return Err(failed("reconlPresentGenerated", r));
            }
            Ok(took)
        }
    }
}

fn check(r: i32, what: &str) -> Result<(), String> {
    if r == abi::result::OK {
        Ok(())
    } else {
        Err(failed(what, r))
    }
}

/// Releases a partial renderer's handles in the order they were acquired.
///
/// # Safety
/// Each handle must be live and not released elsewhere.
unsafe fn release_all(
    swapchain: *mut reconl::SwapchainHandle,
    pipeline: *mut reconl::PipelineHandle,
    commands: *mut reconl::CommandListHandle,
    buffers: &[(*mut reconl::BufferHandle, *mut reconl::BufferHandle)],
) {
    for (vb, ib) in buffers {
        // SAFETY: live handles, as the caller's contract states.
        unsafe {
            reconl::reconlRelease(*vb as *mut core::ffi::c_void);
            reconl::reconlRelease(*ib as *mut core::ffi::c_void);
        }
    }
    // SAFETY: live handles, as the caller's contract states.
    unsafe {
        reconl::reconlRelease(commands as *mut core::ffi::c_void);
        reconl::reconlRelease(pipeline as *mut core::ffi::c_void);
        reconl::reconlRelease(swapchain as *mut core::ffi::c_void);
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // SAFETY: every handle was created by `Renderer::new` and is released once.
        unsafe { release_all(self.swapchain, self.pipeline, self.commands, &self.buffers) };
        self.swapchain = core::ptr::null_mut();
        self.pipeline = core::ptr::null_mut();
        self.commands = core::ptr::null_mut();
        self.buffers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Config, Device};
    use crate::math;
    use crate::scene::ShadowMode;

    fn reference_device() -> Device {
        Device::create(&Config {
            backend: abi::backend::SOFT_CPU,
            tier: 2,
            allow_downgrade: abi::allow_downgrade::NONE,
            threads: 1,
            ..Config::default()
        })
        .expect("soft-cpu device")
    }

    fn renderer_for(device: &Device, scene: &Scene, repeat: u32) -> Renderer {
        Renderer::new(device, scene, Options { repeat, ..Options::default() }).expect("renderer")
    }

    /// A frame is a frame: it presents the frame's bytes, advances the counters,
    /// and reports a cost whose parts add up.
    #[test]
    fn a_frame_presents_pixels_and_costs_what_it_reports() {
        let device = reference_device();
        let scene = Scene::reference();
        let renderer = renderer_for(&device, &scene, 1);
        let mut pixels = vec![0u8; renderer.frame_bytes()];
        let cost = renderer.frame(&scene, 1, &mut pixels).expect("frame");
        assert!(pixels.iter().any(|b| *b != 0), "a frame presents pixels");
        assert_eq!(cost.total_ns(), cost.begin_ns + cost.submit_ns + cost.present_ns);
        let stats = device.stats().expect("stats");
        assert_eq!(stats.frames_presented, 1);
        assert_eq!(stats.frames_dropped, 0);
        assert!(stats.shadows.cascades_active > 0, "the reference scene has shadows");
    }

    /// Two frames of a deterministic scene are the same frame, which is what
    /// makes a golden a golden.
    #[test]
    fn two_frames_of_the_same_scene_are_byte_identical() {
        let device = reference_device();
        let scene = Scene::reference();
        let renderer = renderer_for(&device, &scene, 1);
        let mut first = vec![0u8; renderer.frame_bytes()];
        let mut second = vec![0u8; renderer.frame_bytes()];
        renderer.frame(&scene, 1, &mut first).expect("first");
        renderer.frame(&scene, 1, &mut second).expect("second");
        assert_eq!(first, second);
    }

    /// The shadow mode changes the shadow pass and not the scene, so the two
    /// frames differ - and turning shadows off must not fail or empty the frame.
    #[test]
    fn shadow_modes_render_and_differ() {
        let device = reference_device();
        let on = Scene::with_shadow_mode(ShadowMode::On);
        let off = Scene::with_shadow_mode(ShadowMode::Off);
        let renderer = renderer_for(&device, &on, 1);
        let mut shadowed = vec![0u8; renderer.frame_bytes()];
        let mut unshadowed = vec![0u8; renderer.frame_bytes()];
        renderer.frame(&on, 1, &mut shadowed).expect("shadows on");
        renderer.frame(&off, 1, &mut unshadowed).expect("shadows off");
        assert_ne!(shadowed, unshadowed, "the shadow must change pixels");
        let stats = device.stats().expect("stats");
        assert_eq!(stats.frames_presented, 2);
        assert_eq!(stats.frames_dropped, 0);
    }

    /// A short present buffer is refused by the *host* before any ABI call, so
    /// no frame is opened, nothing is dropped, and the next frame renders
    /// normally. (The library refuses the same case with a consumed frame if a
    /// host presents a bad buffer itself; `ffi/tests/abi.rs` pins that.)
    #[test]
    fn an_undersized_present_buffer_is_refused_before_the_abi_is_touched() {
        let device = reference_device();
        let scene = Scene::reference();
        let renderer = renderer_for(&device, &scene, 1);
        let mut too_small = vec![0u8; 16];
        let e = renderer.frame(&scene, 1, &mut too_small).unwrap_err();
        assert!(e.contains("present buffer"), "{e}");
        assert_eq!(device.stats().expect("stats").frames_dropped, 0, "no frame was opened");
        let mut pixels = vec![0u8; renderer.frame_bytes()];
        renderer.frame(&scene, 1, &mut pixels).expect("the next frame renders");
        let stats = device.stats().expect("stats");
        assert_eq!(stats.frames_presented, 1);
    }

    /// An instanced plan moves a mesh without re-uploading it, and every
    /// instance gets the model it names: translating one chunk draws it at the
    /// translated position, with everything else unmoved.
    #[test]
    fn an_instanced_plan_moves_each_mesh_by_its_own_model() {
        let device = reference_device();
        let scene = Scene::reference();
        let renderer = renderer_for(&device, &scene, 1);
        let mut baseline = vec![0u8; renderer.frame_bytes()];
        renderer.frame(&scene, 1, &mut baseline).expect("baseline");

        // A plan that draws the caster (chunk 1) at a translation instead.
        let mut moved = math::IDENTITY;
        moved[12] = 0.0;
        moved[13] = 0.0;
        moved[14] = 6.0; // +6 on Z, toward the camera
        let plan = vec![(0usize, math::IDENTITY), (1usize, moved)];
        let mut pixels = vec![0u8; renderer.frame_bytes()];
        renderer
            .frame_instances(&scene, 1, &mut pixels, &plan)
            .expect("instanced frame");
        assert_ne!(baseline, pixels, "a translated instance changes the frame");

        // The same plan twice is the same frame: the per-instance model is
        // carried through the command list, not leaked into device state.
        let mut again = vec![0u8; renderer.frame_bytes()];
        renderer.frame_instances(&scene, 1, &mut again, &plan).expect("repeat");
        assert_eq!(pixels, again);
    }

    /// A plan naming a chunk the scene does not have is a host-side error
    /// before the frame is submitted, not a device failure.
    #[test]
    fn an_instance_naming_a_missing_chunk_is_refused_by_the_host() {
        let device = reference_device();
        let scene = Scene::reference();
        let renderer = renderer_for(&device, &scene, 1);
        let mut pixels = vec![0u8; renderer.frame_bytes()];
        let plan = vec![(7usize, math::IDENTITY)];
        let e = renderer.frame_instances(&scene, 1, &mut pixels, &plan).unwrap_err();
        assert!(e.contains("chunk 7"), "{e}");
        assert_eq!(device.stats().expect("stats").frames_dropped, 0);
    }

    /// A list sized for the repeat count accepts the frame it was sized for: the
    /// capacity is a reservation, and recording past it is refused rather than
    /// reallocated, so "the renderer works at repeat N" has to be a test.
    #[test]
    fn a_list_sized_for_its_repeat_count_records_the_whole_frame() {
        let device = reference_device();
        let scene = Scene::reference();
        for repeat in [1u32, 16, 64] {
            let renderer = renderer_for(&device, &scene, repeat);
            let mut pixels = vec![0u8; renderer.frame_bytes()];
            let cost = renderer.frame(&scene, 1, &mut pixels).expect("frame at this repeat");
            assert!(cost.submit_ns > 0);
            assert_eq!(renderer.repeat(), repeat);
        }
        assert_eq!(device.stats().expect("stats").frames_dropped, 0);
    }
}
