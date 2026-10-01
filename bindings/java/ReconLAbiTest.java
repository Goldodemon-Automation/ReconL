// Does this binding agree with include/reconl/reconl.h?
//
// Everything here compares something written down in Java against something the
// library itself reports or enforces. The library validates `struct_size` on
// every descriptor it accepts, so a layout that disagrees with the header cannot
// silently render - it is refused. These tests make that refusal the thing being
// asserted rather than an accident, and pin the enum values to the names the
// library hands back, so a binding that transposed two constants fails loudly
// instead of rendering a black frame.
//
// The struct sizes below are the same numbers `cargo run -p reconl-ffi
// --example pixel_hash -- --sizes` prints, which are themselves the header's.

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

class ReconLAbiTest {

    // ------------------------------------------------------- struct sizes

    @ParameterizedTest(name = "{0} is {1} bytes")
    @CsvSource({
        "BASE, 16",
        "ALLOCATOR, 32",
        "DEVICE_DESC, 96",
        "SWAPCHAIN_DESC, 48",
        "COMMAND_LIST_DESC, 32",
        "BUFFER_DESC, 56",
        "PIPELINE_DESC, 96",
        "RENDER_PASS_DESC, 176",
        "FRAME_DESC, 64",
        "PRESENT_DESC, 48",
        "STATS, 4304",
        "ERROR_INFO, 544",
        "VERTEX, 48",
    })
    @DisplayName("every layout is the header's sizeof")
    void layoutSizesMatchTheHeader(String name, long expected) {
        StructLayout layout = (StructLayout) switch (name) {
            case "BASE" -> ReconL.BASE;
            case "ALLOCATOR" -> ReconL.ALLOCATOR;
            case "DEVICE_DESC" -> ReconL.DEVICE_DESC;
            case "SWAPCHAIN_DESC" -> ReconL.SWAPCHAIN_DESC;
            case "COMMAND_LIST_DESC" -> ReconL.COMMAND_LIST_DESC;
            case "BUFFER_DESC" -> ReconL.BUFFER_DESC;
            case "PIPELINE_DESC" -> ReconL.PIPELINE_DESC;
            case "RENDER_PASS_DESC" -> ReconL.RENDER_PASS_DESC;
            case "FRAME_DESC" -> ReconL.FRAME_DESC;
            case "PRESENT_DESC" -> ReconL.PRESENT_DESC;
            case "STATS" -> ReconL.STATS;
            case "ERROR_INFO" -> ReconL.ERROR_INFO;
            default -> ReconL.VERTEX;
        };
        assertEquals(expected, layout.byteSize(), name + " byteSize");
    }

    /**
     * The offsets that a wrong field order would move. `allocator` sits 56 bytes
     * into the device descriptor, not 0, and `message` sits 20 bytes into the
     * error record, not 16 - both are places a binding that "looks right"
     * writes over something else.
     */
    @Test
    @DisplayName("field offsets are the header's")
    void fieldOffsetsMatchTheHeader() {
        assertEquals(0, ReconL.BASE.byteOffset(groupElement("struct_size")));
        assertEquals(4, ReconL.BASE.byteOffset(groupElement("type")));
        assertEquals(8, ReconL.BASE.byteOffset(groupElement("next")));

        assertEquals(16, ReconL.DEVICE_DESC.byteOffset(groupElement("backend_hint")));
        assertEquals(56, ReconL.DEVICE_DESC.byteOffset(groupElement("allocator")),
                "the allocator callbacks must not be written over the descriptor's base");
        assertEquals(88, ReconL.DEVICE_DESC.byteOffset(groupElement("backend_desc")));

        assertEquals(16, ReconL.PRESENT_DESC.byteOffset(groupElement("out_pixels")));
        assertEquals(24, ReconL.PRESENT_DESC.byteOffset(groupElement("out_pixels_size")));

        assertEquals(20, ReconL.ERROR_INFO.byteOffset(groupElement("message")),
                "char[192] message, so the byte right after `result`");
        assertEquals(16, ReconL.ERROR_INFO.byteOffset(groupElement("result")));

        assertEquals(24, ReconL.RENDER_PASS_DESC.byteOffset(groupElement("color")));
        assertEquals(144, ReconL.RENDER_PASS_DESC.byteOffset(groupElement("clear_color")));

        // ReconLVertex is 12 floats, and the conformance frame's whole triangle
        // is written through these offsets.
        assertEquals(0, ReconL.VERTEX.byteOffset(groupElement("position")));
        assertEquals(12, ReconL.VERTEX.byteOffset(groupElement("normal")));
        assertEquals(24, ReconL.VERTEX.byteOffset(groupElement("uv")));
        assertEquals(32, ReconL.VERTEX.byteOffset(groupElement("color")));
    }

    // -------------------------------------------------------- result codes

    @ParameterizedTest(name = "result {0} is {1}")
    @CsvSource({
        "0,  RECONL_OK",
        "-1, RECONL_ERR_INVALID_ARGUMENT",
        "-2, RECONL_ERR_OUT_OF_MEMORY",
        "-3, RECONL_ERR_NOT_SUPPORTED",
        "-4, RECONL_ERR_BACKEND_UNAVAILABLE",
        "-5, RECONL_ERR_BUDGET_EXCEEDED",
        "-6, RECONL_ERR_DEVICE_LOST",
        "-7, RECONL_ERR_INVALID_HANDLE",
        "-8, RECONL_ERR_STRUCT_SIZE",
        "-9, RECONL_ERR_WRONG_STRUCT_TYPE",
        "-10, RECONL_ERR_ABI_VERSION",
        "-11, RECONL_ERR_FRAME_IN_PROGRESS",
        "-12, RECONL_ERR_NO_FRAME",
        "-13, RECONL_ERR_NOT_READY",
        "-14, RECONL_ERR_IO",
        "-15, RECONL_ERR_CORRUPT_CACHE",
        "-16, RECONL_ERR_DEGRADED",
        "-17, RECONL_ERR_PANIC",
        "-18, RECONL_ERR_EMPTY_FRAME",
    })
    @DisplayName("every result constant is the name the library reports")
    void resultConstantsMatchTheLibrary(int code, String name) {
        assertEquals(name, ReconL.resultName(code), "reconlResultName(" + code + ")");
    }

    @ParameterizedTest(name = "tier {0} is {1}")
    @CsvSource({
        "TIER_T0_GPU_DISCRETE, 0, T0/gpu-discrete",
        "TIER_T1_GPU_SHARED,   1, T1/gpu-shared",
        "TIER_T2_CPU_RAM,      2, T2/cpu-ram",
        "TIER_T3_CPU_THRIFTY,  3, T3/cpu-thrifty",
        "TIER_T4_OUT_OF_CORE,  4, T4/out-of-core",
    })
    @DisplayName("the tier ladder is T0..T4, and the constants are those numbers")
    void tierConstantsMatchTheLibrary(String constant, int tier, String name) {
        int actual = switch (constant) {
            case "TIER_T0_GPU_DISCRETE" -> ReconL.TIER_T0_GPU_DISCRETE;
            case "TIER_T1_GPU_SHARED" -> ReconL.TIER_T1_GPU_SHARED;
            case "TIER_T2_CPU_RAM" -> ReconL.TIER_T2_CPU_RAM;
            case "TIER_T3_CPU_THRIFTY" -> ReconL.TIER_T3_CPU_THRIFTY;
            default -> ReconL.TIER_T4_OUT_OF_CORE;
        };
        assertEquals(tier, actual, constant + " is the enum value the header writes down");
        assertEquals(name, ReconL.tierName(tier), "reconlTierName(" + tier + ")");
    }

    @ParameterizedTest(name = "backend {0} is {1}")
    @CsvSource({
        "0, none",
        "1, soft-cpu",
        "2, null",
        "3, d3d11",
    })
    @DisplayName("the backend constants are the numbers the library reports")
    void backendConstantsMatchTheLibrary(int backend, String name) {
        assertEquals(name, ReconL.backendName(backend), "reconlBackendName(" + backend + ")");
    }

    @Test
    @DisplayName("a backend id outside the enum reads as none, not as garbage")
    void unknownValuesAreClampedNotGuessed() {
        // A binding must not invent a name for a value the library does not
        // know: a host that printed "d3d12" for a made-up id would be lying.
        assertEquals("none", ReconL.backendName(99));
        assertEquals("none", ReconL.backendName(-1));
        assertEquals("unknown", ReconL.resultName(1));
        assertEquals("unknown", ReconL.resultName(-99));
        // The tier table clamps to its last rung rather than reading off the end.
        assertEquals(ReconL.tierName(ReconL.TIER_T4_OUT_OF_CORE), ReconL.tierName(5));
    }

    // ------------------------------------------------- struct_size refusal

    @ParameterizedTest(name = "{0} with struct_size 8 is refused")
    @ValueSource(strings = {"DEVICE_DESC", "SWAPCHAIN_DESC", "FRAME_DESC", "STATS"})
    @DisplayName("a descriptor shorter than the library reads is refused")
    void shortDescriptorsAreRefused(String which) {
        try (Arena arena = Arena.ofConfined();
             ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                     ReconL.TIER_T2_CPU_RAM)) {
            int rc;
            switch (which) {
                case "DEVICE_DESC" -> {
                    MemorySegment d = ReconL.zeroed(arena, ReconL.DEVICE_DESC);
                    ReconL.BASE.varHandle(groupElement("struct_size")).set(d, 8);
                    rc = ReconL.reconlCreateDevice(d, arena.allocate(
                            java.lang.foreign.ValueLayout.ADDRESS));
                }
                case "SWAPCHAIN_DESC" -> {
                    MemorySegment d = ReconL.zeroed(arena, ReconL.SWAPCHAIN_DESC);
                    ReconL.BASE.varHandle(groupElement("struct_size")).set(d, 8);
                    rc = ReconL.reconlCreateSwapchain(device.handle(), d,
                            arena.allocate(java.lang.foreign.ValueLayout.ADDRESS));
                }
                case "FRAME_DESC" -> {
                    MemorySegment d = ReconL.zeroed(arena, ReconL.FRAME_DESC);
                    ReconL.BASE.varHandle(groupElement("struct_size")).set(d, 8);
                    rc = ReconL.reconlBeginFrame(device.handle(), d);
                }
                default -> {
                    MemorySegment d = ReconL.zeroed(arena, ReconL.STATS);
                    ReconL.BASE.varHandle(groupElement("struct_size")).set(d, 8);
                    rc = ReconL.reconlGetStats(device.handle(), d);
                }
            }
            assertEquals(ReconL.ERR_STRUCT_SIZE, rc, which + " with struct_size = 8");
        }
    }

    @ParameterizedTest(name = "{0} typed as a camera is refused")
    @ValueSource(strings = {"DEVICE_DESC", "SWAPCHAIN_DESC", "FRAME_DESC"})
    @DisplayName("a descriptor with the wrong type tag is refused")
    void wrongTypeIsRefused(String which) {
        try (Arena arena = Arena.ofConfined();
             ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                     ReconL.TIER_T2_CPU_RAM)) {
            int rc;
            switch (which) {
                case "DEVICE_DESC" -> {
                    MemorySegment d = ReconL.zeroed(arena, ReconL.DEVICE_DESC);
                    ReconL.setBase(d, ReconL.DEVICE_DESC, ReconL.STRUCT_CAMERA,
                            ReconL.SIZE_DEVICE_DESC);
                    rc = ReconL.reconlCreateDevice(d, arena.allocate(
                            java.lang.foreign.ValueLayout.ADDRESS));
                }
                case "SWAPCHAIN_DESC" -> {
                    MemorySegment d = ReconL.zeroed(arena, ReconL.SWAPCHAIN_DESC);
                    ReconL.setBase(d, ReconL.SWAPCHAIN_DESC, ReconL.STRUCT_CAMERA,
                            ReconL.SIZE_SWAPCHAIN_DESC);
                    rc = ReconL.reconlCreateSwapchain(device.handle(), d,
                            arena.allocate(java.lang.foreign.ValueLayout.ADDRESS));
                }
                default -> {
                    MemorySegment d = ReconL.zeroed(arena, ReconL.FRAME_DESC);
                    ReconL.setBase(d, ReconL.FRAME_DESC, ReconL.STRUCT_CAMERA,
                            ReconL.SIZE_FRAME_DESC);
                    rc = ReconL.reconlBeginFrame(device.handle(), d);
                }
            }
            assertEquals(ReconL.ERR_WRONG_STRUCT_TYPE, rc, which + " typed as a camera");
        }
    }

    @Test
    @DisplayName("the library says what it expected, in words")
    void aRefusalCarriesAnExplanation() {
        try (Arena arena = Arena.ofConfined();
             ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                     ReconL.TIER_T2_CPU_RAM)) {
            MemorySegment d = ReconL.zeroed(arena, ReconL.FRAME_DESC);
            ReconL.BASE.varHandle(groupElement("struct_size")).set(d, 8);
            assertEquals(ReconL.ERR_STRUCT_SIZE, ReconL.reconlBeginFrame(device.handle(), d));

            String message = device.lastError();
            assertTrue(message != null && message.contains("struct_size"),
                    "lastError should name the field, was: " + message);
            assertTrue(message.contains("8"), "lastError should name the value, was: " + message);
        }
    }

    @Test
    @DisplayName("a fresh device has recorded no error at all")
    void aFreshDeviceHasNoError() {
        try (ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                ReconL.TIER_T2_CPU_RAM)) {
            assertEquals("no error has been recorded", device.lastError());
        }
    }

    /**
     * The guard that makes the rest of the suite meaningful: if setBase ever
     * stopped checking, a layout could drift from the header and the next test
     * to touch it would fail for the wrong reason.
     */
    @Test
    @DisplayName("a layout whose size drifts from the header is caught before a call")
    void aWrongLayoutIsCaughtLocally() {
        MemoryLayout wrong = MemoryLayout.structLayout(
                java.lang.foreign.ValueLayout.JAVA_INT.withName("not_enough"));
        IllegalStateException thrown = assertThrows(IllegalStateException.class,
                () -> ReconL.setBase(MemorySegment.NULL, (StructLayout) wrong,
                        ReconL.STRUCT_DEVICE_DESC, ReconL.SIZE_DEVICE_DESC));
        assertTrue(thrown.getMessage().contains("the header says 96"), thrown.getMessage());
    }
}
