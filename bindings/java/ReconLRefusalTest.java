// What the library refuses, and why that matters to a foreign caller.
//
// probes/src/hostile*.c assert these same refusals in C. A refusal that only
// ever gets exercised from one language is a refusal nobody has checked from
// the other, and an FFI layer that gets the descriptor wrong should be stopped
// by the library's own guards - which is only true if the guards are actually
// reached. So this class is mostly a walk through the ways a descriptor can be
// wrong, plus the two behaviours that are easy to get wrong as a *host*: a
// present attempt consumes the frame whether it succeeds or not, and the tier
// ladder only goes down.

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class ReconLRefusalTest {

    private static final int SIZE = 32;
    private static final int PIXELS = SIZE * SIZE * 4;

    private static void base(MemorySegment s, int type, int structSize) {
        ReconL.BASE.varHandle(groupElement("struct_size")).set(s, structSize);
        ReconL.BASE.varHandle(groupElement("type")).set(s, type);
        ReconL.BASE.varHandle(groupElement("next")).set(s, MemorySegment.NULL);
    }

    /** begin -> draw -> submit, i.e. a frame that is ready to be presented. */
    private static void submittedFrame(ReconL.Rig rig) {
        assertEquals(ReconL.OK, rig.beginFrame(), "beginFrame");
        assertEquals(ReconL.OK, rig.draw(), "draw");
        assertEquals(ReconL.OK, rig.submit(), "submit");
    }

    // ------------------------------------------------------------- handles

    @Test
    @DisplayName("a NULL handle is refused, not dereferenced")
    void nullHandlesAreRefused() {
        assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                ReconL.reconlCreateSwapchain(MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL));
        assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                ReconL.reconlCreateCommandList(MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL));
        assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                ReconL.reconlCreateBuffer(MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL));
        assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                ReconL.reconlCreatePipeline(MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL));
        assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                ReconL.reconlBeginFrame(MemorySegment.NULL, MemorySegment.NULL));
        assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                ReconL.reconlGetStats(MemorySegment.NULL, MemorySegment.NULL));
        // Releasing nothing is not an error: teardown paths call it
        // unconditionally, and making them all null-check first is worse.
        assertEquals(ReconL.OK, ReconL.reconlRelease(MemorySegment.NULL));
    }

    @Test
    @DisplayName("a NULL descriptor is refused")
    void nullDescriptorsAreRefused() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                     ReconL.TIER_T2_CPU_RAM)) {
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlCreateSwapchain(device.handle(), MemorySegment.NULL, out));
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlCreateBuffer(device.handle(), MemorySegment.NULL, out));
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlCreatePipeline(device.handle(), MemorySegment.NULL, out));
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlBeginFrame(device.handle(), MemorySegment.NULL));
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlGetStats(device.handle(), MemorySegment.NULL));
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlGetLastError(device.handle(), MemorySegment.NULL));
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlSubmit(device.handle(), MemorySegment.NULL, MemorySegment.NULL));
        }
    }

    /**
     * The one entry point where a NULL descriptor is not an error, and the only
     * reason this test exists: a host that assumes the whole ABI refuses NULL
     * will get a real command list here and never notice, because nothing about
     * the call looks wrong.
     *
     * ffi/src/lib.rs treats `desc == NULL` as `capacity_bytes == 0`, which the
     * header documents ("0 = the library's default (1024 commands)"). The
     * behaviour is deliberate; the fact that reconl.h does not mention it at
     * this declaration is not. See Context.md - the follow-up is to document it
     * next to the prototype, or refuse it like every sibling does.
     */
    @Test
    @DisplayName("a NULL command-list descriptor means the default capacity, not an error")
    void aNullCommandListDescriptorMeansDefaults() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                     ReconL.TIER_T2_CPU_RAM)) {
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            assertEquals(ReconL.OK,
                    ReconL.reconlCreateCommandList(device.handle(), MemorySegment.NULL, out),
                    "a NULL descriptor is the documented default capacity, unlike every "
                            + "sibling call");

            MemorySegment defaulted = out.get(ValueLayout.ADDRESS, 0);
            assertTrue(defaulted.address() != 0, "and it actually produced a command list");
            assertEquals(ReconL.OK, ReconL.reconlRelease(defaulted));
        }
    }

    @Test
    @DisplayName("a device with a zeroed allocator is refused to start")
    void aZeroedAllocatorIsRefused() {
        // reconl.h: "allocator required; zeroed allocator = refuse start".
        // Without this a library with no way to reach the host's allocator would
        // either fall back to its own or start with no idea who owns the memory.
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment desc = ReconL.zeroed(arena, ReconL.DEVICE_DESC);
            base(desc, ReconL.STRUCT_DEVICE_DESC, (int) ReconL.SIZE_DEVICE_DESC);
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlCreateDevice(desc, arena.allocate(ValueLayout.ADDRESS)));
        }
    }

    @Test
    @DisplayName("a refused device leaves a working one alone")
    void refusalsDoNotDisturbAHealthyDevice() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                     ReconL.TIER_T2_CPU_RAM)) {
            MemorySegment bad = ReconL.zeroed(arena, ReconL.FRAME_DESC);
            base(bad, ReconL.STRUCT_FRAME_DESC, 8);
            assertEquals(ReconL.ERR_STRUCT_SIZE, ReconL.reconlBeginFrame(device.handle(), bad));

            assertEquals(ReconL.BACKEND_SOFT_CPU, device.backend(), "backend survives a refusal");
            assertEquals(ReconL.TIER_T2_CPU_RAM, device.tier(), "tier survives a refusal");
        }
    }

    // -------------------------------------------------------- frame lifecycle

    @Test
    @DisplayName("a second begin before the first is presented is refused")
    void twoOpenFramesAreRefused() {
        try (ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            assertEquals(ReconL.OK, rig.beginFrame());
            assertEquals(ReconL.ERR_FRAME_IN_PROGRESS, rig.beginFrame(),
                    "a second open frame would leave the first's work unowned");
        }
    }

    @Test
    @DisplayName("presenting a frame that was never submitted is refused")
    void presentWithoutSubmitIsRefused() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            MemorySegment out = arena.allocate(PIXELS, 1);
            assertEquals(ReconL.ERR_NO_FRAME,
                    rig.present(rig.presentDesc(out, PIXELS, SIZE * 4)),
                    "nothing has ever been submitted on this device");

            assertEquals(ReconL.OK, rig.beginFrame());
            assertEquals(ReconL.ERR_NO_FRAME,
                    rig.present(rig.presentDesc(out, PIXELS, SIZE * 4)),
                    "an open frame is not a submitted one");
        }
    }

    @Test
    @DisplayName("submitting a command list with nothing in it is refused")
    void submittingNothingIsRefused() {
        try (ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            assertEquals(ReconL.OK, rig.beginFrame());
            assertEquals(ReconL.ERR_INVALID_ARGUMENT, rig.submit(),
                    "an empty frame has nothing to present and is not a frame");
        }
    }

    // ------------------------------------------------------- present guards

    @Test
    @DisplayName("present checks the descriptor type before anything else")
    void presentChecksTheDescriptorType() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            submittedFrame(rig);
            MemorySegment bad = rig.presentDesc(arena.allocate(PIXELS, 1), PIXELS, SIZE * 4);
            ReconL.BASE.varHandle(groupElement("type")).set(bad, ReconL.STRUCT_STATS);
            assertEquals(ReconL.ERR_WRONG_STRUCT_TYPE, rig.present(bad));
        }
    }

    @Test
    @DisplayName("present checks its own struct_size")
    void presentChecksStructSize() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            submittedFrame(rig);
            MemorySegment bad = rig.presentDesc(arena.allocate(PIXELS, 1), PIXELS, SIZE * 4);
            ReconL.BASE.varHandle(groupElement("struct_size")).set(bad, 8);
            assertEquals(ReconL.ERR_STRUCT_SIZE, rig.present(bad));
        }
    }

    @Test
    @DisplayName("a row pitch narrower than a row is refused")
    void aNarrowRowPitchIsRefused() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            submittedFrame(rig);
            MemorySegment out = arena.allocate(PIXELS, 1);
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    rig.present(rig.presentDesc(out, PIXELS, SIZE * 4 - 1)));
        }
    }

    @Test
    @DisplayName("an output buffer too small for the frame is refused")
    void aShortOutputBufferIsRefused() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            submittedFrame(rig);
            MemorySegment out = arena.allocate(PIXELS, 1);
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    rig.present(rig.presentDesc(out, PIXELS - 1, SIZE * 4)),
                    "the last row would be written past the end of the host's buffer");
        }
    }

    @Test
    @DisplayName("a NULL output buffer with a size is refused")
    void aNullOutputBufferIsRefused() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            submittedFrame(rig);
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    rig.present(rig.presentDesc(MemorySegment.NULL, PIXELS, SIZE * 4)));
        }
    }

    @Test
    @DisplayName("a NULL swapchain is refused")
    void aNullSwapchainIsRefused() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            submittedFrame(rig);
            MemorySegment desc = rig.presentDesc(arena.allocate(PIXELS, 1), PIXELS, SIZE * 4);
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlPresent(rig.device.handle(), MemorySegment.NULL, desc));
        }
    }

    /**
     * The subtle one. A present attempt consumes the frame whether the library
     * accepted it or not, so a host that retries a refused present with a fixed
     * descriptor presents nothing and silently reads back a stale buffer. This
     * is asserted from both ends - the refusal, and the emptiness after it -
     * because a binding that got either one wrong would pass a test of the
     * other.
     */
    @Test
    @DisplayName("a refused present still consumes the frame")
    void aRefusedPresentConsumesTheFrame() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            submittedFrame(rig);
            MemorySegment out = arena.allocate(PIXELS, 1);
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    rig.present(rig.presentDesc(out, PIXELS, 1)),
                    "the refusal under test");
            assertEquals(ReconL.ERR_NO_FRAME,
                    rig.present(rig.presentDesc(out, PIXELS, SIZE * 4)),
                    "a retry after a refusal has no frame left to present");
        }
    }

    @Test
    @DisplayName("a good present also consumes the frame, and the next frame works")
    void aGoodPresentConsumesTheFrame() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Rig rig = new ReconL.Rig(ReconL.BACKEND_SOFT_CPU, SIZE)) {
            submittedFrame(rig);
            MemorySegment out = arena.allocate(PIXELS, 1);
            assertEquals(ReconL.OK, rig.present(rig.presentDesc(out, PIXELS, SIZE * 4)));
            assertEquals(ReconL.ERR_NO_FRAME, rig.present(rig.presentDesc(out, PIXELS, SIZE * 4)));

            // ...and the device is idle, not broken: the next frame presents.
            submittedFrame(rig);
            assertEquals(ReconL.OK, rig.present(rig.presentDesc(out, PIXELS, SIZE * 4)));
        }
    }

    // ------------------------------------------------------------- the budget

    @Test
    @DisplayName("a swapchain past the allocation ceiling is refused, not attempted")
    void anOversizedSwapchainIsRefused() {
        // reconl.h: the check runs before the host allocator is called, so this
        // costs the host an error and not a machine that starts swapping.
        try (Arena arena = Arena.ofConfined();
             ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                     ReconL.TIER_T2_CPU_RAM)) {
            MemorySegment desc = ReconL.zeroed(arena, ReconL.SWAPCHAIN_DESC);
            base(desc, ReconL.STRUCT_SWAPCHAIN_DESC, (int) ReconL.SIZE_SWAPCHAIN_DESC);
            ReconL.SWAPCHAIN_DESC.varHandle(groupElement("width")).set(desc, 65535);
            ReconL.SWAPCHAIN_DESC.varHandle(groupElement("height")).set(desc, 65535);
            ReconL.SWAPCHAIN_DESC.varHandle(groupElement("format"))
                    .set(desc, ReconL.FORMAT_R8G8B8A8_UNORM);
            ReconL.SWAPCHAIN_DESC.varHandle(groupElement("image_count")).set(desc, 2);
            ReconL.SWAPCHAIN_DESC.varHandle(groupElement("present_to_memory")).set(desc, 1);
            assertEquals(ReconL.ERR_BUDGET_EXCEEDED,
                    ReconL.reconlCreateSwapchain(device.handle(), desc,
                            arena.allocate(ValueLayout.ADDRESS)));
            assertEquals(ReconL.BACKEND_SOFT_CPU, device.backend(), "the device is still fine");
        }
    }

    @Test
    @DisplayName("a degenerate swapchain is refused")
    void aDegenerateSwapchainIsRefused() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                     ReconL.TIER_T2_CPU_RAM)) {
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);

            MemorySegment noImages = swapchainDesc(arena, 64, 64, 0);
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlCreateSwapchain(device.handle(), noImages, out),
                    "image_count is documented 1..3");

            MemorySegment empty = swapchainDesc(arena, 0, 0, 2);
            assertEquals(ReconL.ERR_INVALID_ARGUMENT,
                    ReconL.reconlCreateSwapchain(device.handle(), empty, out));
        }
    }

    private static MemorySegment swapchainDesc(Arena arena, int w, int h, int images) {
        MemorySegment d = ReconL.zeroed(arena, ReconL.SWAPCHAIN_DESC);
        base(d, ReconL.STRUCT_SWAPCHAIN_DESC, (int) ReconL.SIZE_SWAPCHAIN_DESC);
        ReconL.SWAPCHAIN_DESC.varHandle(groupElement("width")).set(d, w);
        ReconL.SWAPCHAIN_DESC.varHandle(groupElement("height")).set(d, h);
        ReconL.SWAPCHAIN_DESC.varHandle(groupElement("format")).set(d, ReconL.FORMAT_R8G8B8A8_UNORM);
        ReconL.SWAPCHAIN_DESC.varHandle(groupElement("image_count")).set(d, images);
        ReconL.SWAPCHAIN_DESC.varHandle(groupElement("present_to_memory")).set(d, 1);
        return d;
    }

    // ------------------------------------------------------- the tier ladder

    /**
     * reconlRequestTier only ever walks a device *down* the ladder. Asking a T2
     * device for T0 does not promote it - the loop in ffi/src/lib.rs steps down
     * from whatever it is at until it runs out of rungs, which is the offload
     * contract: a host may shed quality, never invent it. A binding that got the
     * argument order wrong would promote instead, and this is what would catch it.
     */
    @Test
    @DisplayName("requestTier only ever walks a device down the ladder")
    void requestTierOnlyWalksDown() {
        try (ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                ReconL.TIER_T1_GPU_SHARED)) {
            assertEquals(ReconL.TIER_T2_CPU_RAM, device.tier(),
                    "soft-cpu cannot serve a GPU tier, so T1 lands on T2");
            assertEquals(ReconL.OK, device.requestTier(ReconL.TIER_T0_GPU_DISCRETE));
            assertEquals(ReconL.TIER_T4_OUT_OF_CORE, device.tier(),
                    "asking for a stronger tier walks it to the weakest, never up");
        }
    }

    @Test
    @DisplayName("requestTier clamps an out-of-range tier into the ladder")
    void requestTierClampsOutOfRange() {
        try (ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                ReconL.TIER_T4_OUT_OF_CORE)) {
            // The ABI takes the tier as a uint32, so -1 arrives as 2^32-1 and 99
            // as 99. Neither is refused; both are clamped, because the ladder is
            // the host's safety net and a panic here would take the frame with it.
            assertEquals(ReconL.OK, device.requestTier(99));
            assertEquals(ReconL.OK, device.requestTier(-1));
            assertEquals(ReconL.TIER_T4_OUT_OF_CORE, device.tier());
        }
    }

    @Test
    @DisplayName("a device survives every tier the ladder offers")
    void everyTierInTheLadderIsServable() {
        for (int tier = ReconL.TIER_T0_GPU_DISCRETE; tier <= ReconL.TIER_T4_OUT_OF_CORE; tier++) {
            try (ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU, tier)) {
                assertEquals(ReconL.BACKEND_SOFT_CPU, device.backend(), "backend for hint " + tier);
                assertNotEquals(ReconL.BACKEND_NONE, device.backend());
                assertTrue(device.tier() >= ReconL.TIER_T0_GPU_DISCRETE
                                && device.tier() <= ReconL.TIER_T4_OUT_OF_CORE,
                        "tier for hint " + tier + " was " + device.tier());
            }
        }
    }
}
