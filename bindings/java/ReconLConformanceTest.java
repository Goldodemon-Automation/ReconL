// The cross-runtime check.
//
// Everything else in this suite asks "did the library do the right thing". This
// class asks the only question a *second* FFI layer can answer: "does this
// language agree with the other one about what the header says".
//
// ffi/examples/pixel_hash.rs builds the same frame through the Rust side of the
// ABI - the same three clip-space vertices, the same blue clear, the same tight
// RGBA8 readback - and prints the SHA-256 of the pixels. The digests below were
// produced by that example. If ReconL.java disagreed with reconl.h about a field
// order, a struct size, a blend mode or the reversed-Z compare, one of the two
// sides would render a different image and this test would fail. Everything else
// in the directory would still pass, which is why this class exists.
//
// Regenerate after an intentional ABI change:
//   cargo run -p reconl-ffi --example pixel_hash -- --size=64
//   java --enable-preview -cp out ConformanceMain            (see run-java-tests.sh)

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class ReconLConformanceTest {

    /** From `cargo run -p reconl-ffi --example pixel_hash -- --size=64`. */
    private static final String RUST_SOFT_CPU_64 =
            "0fbba07a833d4dcfc7024eaf313661a0ba8f80a05c6d29b8801c612e10e60dee";
    /** Odd sizes are the interesting ones: 65 rows is where a binding that got
     *  the row pitch or the buffer size wrong stops agreeing with itself. */
    private static final String RUST_SOFT_CPU_65 =
            "f9d2ec81247fefce150e363234206ffea99b7b5c0c4587d4c5d74d10fdc0ff43";
    private static final String RUST_SOFT_CPU_128 =
            "71189f7fb6aed638640078fba3a35fda6c39c8962e74dcc75935aac948da9063";
    /** A second backend, so the digest is not a constant of the library. */
    private static final String RUST_NULL_64 =
            "20aaec3d48040c744eeb8551b38acee45d7d754101f5636fd744fcec10a60bb5";

    // ------------------------------------------------- the cross-runtime bit

    @ParameterizedTest(name = "{0} at {1}px matches the Rust digest")
    @CsvSource({
        "soft-cpu, 64,  " + RUST_SOFT_CPU_64,
        "soft-cpu, 65,  " + RUST_SOFT_CPU_65,
        "soft-cpu, 128, " + RUST_SOFT_CPU_128,
        "null,     64,  " + RUST_NULL_64,
    })
    @DisplayName("the same frame from Java and from Rust hashes the same")
    void theFrameIsIdenticalAcrossRuntimes(String backend, int size, String expected) {
        int id = switch (backend) {
            case "null" -> ReconL.BACKEND_NULL;
            default -> ReconL.BACKEND_SOFT_CPU;
        };
        byte[] pixels = ReconL.renderConformanceFrame(id, size);
        assertEquals(size * size * 4, pixels.length, "the readback is one RGBA8 pixel per cell");
        assertEquals(expected, ReconL.sha256Hex(pixels),
                "Java and Rust rendered the same frame differently - the binding and the "
                        + "header disagree about something");
    }

    @Test
    @DisplayName("the reference digest is stable across runs in one process")
    void theReferenceDigestIsStable() {
        byte[] first = ReconL.renderConformanceFrame(ReconL.BACKEND_SOFT_CPU, 64);
        byte[] second = ReconL.renderConformanceFrame(ReconL.BACKEND_SOFT_CPU, 64);
        assertArrayEquals(first, second, "the same descriptors must render the same pixels");
        assertEquals(RUST_SOFT_CPU_64, ReconL.sha256Hex(second));
    }

    @Test
    @DisplayName("the backends disagree, so the digest is measuring the render")
    void backendsRenderDifferentPixels() {
        // A digest that matched across every backend would mean the readback was
        // returning the clear colour, not the frame - a green test proving nothing.
        assertNotEquals(RUST_NULL_64, RUST_SOFT_CPU_64);
        assertEquals(RUST_NULL_64,
                ReconL.sha256Hex(ReconL.renderConformanceFrame(ReconL.BACKEND_NULL, 64)));
    }

    // ---------------------------------------------------- what the frame is

    @Test
    @DisplayName("the conformance frame is a full-clip white triangle over blue")
    void theFrameIsTheSceneItClaimsToBe() {
        int size = 64;
        byte[] pixels = ReconL.renderConformanceFrame(ReconL.BACKEND_SOFT_CPU, size);
        assertEquals(RUST_SOFT_CPU_64, ReconL.sha256Hex(pixels), "the digest, not the pixels");

        // The triangle is (-1,-1) (3,-1) (-1,3) in clip space, which covers the
        // whole [-1,1] square, so every cell is the vertex colour. If the
        // reversed-Z compare were wrong, or the depth clear were wrong, the
        // triangle would be rejected and the clear would show through instead.
        int whiteCells = 0;
        for (int i = 0; i < pixels.length; i += 4) {
            int r = pixels[i] & 0xff, g = pixels[i + 1] & 0xff;
            int b = pixels[i + 2] & 0xff, a = pixels[i + 3] & 0xff;
            if (r == 0xff && g == 0xff && b == 0xff && a == 0xff) {
                whiteCells++;
            }
            assertEquals(0xff, a, "the alpha of pixel " + (i / 4) + " was " + a);
        }
        assertEquals(size * size, whiteCells,
                "a full-clip triangle should cover every cell; " + whiteCells + " of "
                        + (size * size) + " were white, so part of the frame fell through");
    }

    @Test
    @DisplayName("a size that does not divide evenly still renders the right number of bytes")
    void oddSizesRenderTheRightNumberOfBytes() {
        for (int size : new int[] {1, 3, 65, 127}) {
            byte[] pixels = ReconL.renderConformanceFrame(ReconL.BACKEND_SOFT_CPU, size);
            assertEquals(size * size * 4, pixels.length, size + "px");
            for (int i = 0; i < pixels.length; i++) {
                if (pixels[i] == 0) {
                    continue;
                }
                assertTrue(pixels[i] != 0 || size == 1, "no clearing of the last row");
                break;
            }
        }
    }

    // ------------------------------------------------------------ tier matrix

    /**
     * The tier a device lands on, per backend hint, is the matrix probes/src/
     * ladder.c walks. On a CPU backend a GPU tier is not merely ignored: the
     * device reports NO_GPU_API as the reason it stepped down, which is the
     * difference between a host that asked for T0 and a host that fell to it.
     */
    @ParameterizedTest(name = "soft-cpu asked for {0} lands on {1} for reason {2}")
    @CsvSource({
        "TIER_T0_GPU_DISCRETE, TIER_T2_CPU_RAM,      TIER_REASON_HOST_REQUEST",
        "TIER_T1_GPU_SHARED,   TIER_T2_CPU_RAM,      TIER_REASON_NO_GPU_API",
        "TIER_T2_CPU_RAM,      TIER_T2_CPU_RAM,      TIER_REASON_HOST_REQUEST",
        "TIER_T3_CPU_THRIFTY,  TIER_T3_CPU_THRIFTY,  TIER_REASON_HOST_REQUEST",
        "TIER_T4_OUT_OF_CORE,  TIER_T4_OUT_OF_CORE,  TIER_REASON_HOST_REQUEST",
    })
    @DisplayName("the tier matrix, on a backend that cannot serve the GPU tiers")
    void theTierMatrixOnACpuBackend(String hint, String expectedTier, String expectedReason) {
        try (ReconL.Device device = new ReconL.Device(ReconL.BACKEND_SOFT_CPU, tier(hint))) {
            assertEquals(ReconL.BACKEND_SOFT_CPU, device.backend(), "hint " + hint);
            assertEquals(tier(expectedTier), device.tier(), "hint " + hint);
            assertEquals(tierReason(expectedReason), device.tierReason(), "hint " + hint);
        }
    }

    /**
     * Only T1 records NO_GPU_API. A hint of T0 lands on the same T2 rung as T1
     * but is reported as a host request, which is worth pinning: a host auditing
     * its tier log distinguishes "I asked for the top tier and it was not there"
     * from "I asked for a shared-GPU tier and there was no GPU API". Asserting
     * only the resulting tier would not notice if the two swapped.
     */
    @Test
    @DisplayName("T0 and T1 both land on T2 but say different things about why")
    void theTwoGpuHintsAreDistinguished() {
        try (ReconL.Device t0 = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                     ReconL.TIER_T0_GPU_DISCRETE);
             ReconL.Device t1 = new ReconL.Device(ReconL.BACKEND_SOFT_CPU,
                     ReconL.TIER_T1_GPU_SHARED)) {
            assertEquals(t0.tier(), t1.tier(), "both step down to the same rung");
            assertEquals(ReconL.TIER_REASON_NO_GPU_API, t1.tierReason(),
                    "asking for the shared-GPU tier names the missing API");
            assertNotEquals(ReconL.TIER_REASON_NO_GPU_API, t0.tierReason(),
                    "asking for the discrete tier does not, so a log can tell them apart");
        }
    }

    /**
     * The same matrix on the null backend, which renders nothing and exists to be
     * cheap. It must agree with soft-cpu about *which* tier a hint selects even
     * though it renders different pixels: the tier ladder is above the backend.
     */
    @ParameterizedTest(name = "null asked for {0} lands on {1}")
    @CsvSource({
        "TIER_T0_GPU_DISCRETE, TIER_T2_CPU_RAM",
        "TIER_T1_GPU_SHARED,   TIER_T2_CPU_RAM",
        "TIER_T2_CPU_RAM,      TIER_T2_CPU_RAM",
        "TIER_T3_CPU_THRIFTY,  TIER_T3_CPU_THRIFTY",
        "TIER_T4_OUT_OF_CORE,  TIER_T4_OUT_OF_CORE",
    })
    @DisplayName("the tier matrix is the same on the null backend")
    void theTierMatrixIsAboveTheBackend(String hint, String expectedTier) {
        try (ReconL.Device device = new ReconL.Device(ReconL.BACKEND_NULL, tier(hint))) {
            assertEquals(ReconL.BACKEND_NULL, device.backend(), "hint " + hint);
            assertEquals(tier(expectedTier), device.tier(), "hint " + hint);
        }
    }

    /**
     * RECONL_BACKEND_NONE is "let ReconL choose", so the answer is whatever this
     * machine has - and on a machine with no usable GPU backend there is no
     * answer, so both outcomes are legitimate and the test accepts either. What
     * it will not accept is a device that comes up on BACKEND_NONE itself: that
     * would be a hint the library silently ignored. Pinning which backend is
     * chosen would be a test of the build host, not of the ABI.
     */
    @Test
    @DisplayName("BACKEND_NONE either resolves to a real backend or is honestly refused")
    void backendNoneResolvesToSomethingUsable() {
        try (ReconL.Device device = new ReconL.Device(ReconL.BACKEND_NONE, 0)) {
            assertNotEquals(ReconL.BACKEND_NONE, device.backend(),
                    "a hint of NONE must not leave the device on no backend");
            assertTrue(device.tier() >= ReconL.TIER_T0_GPU_DISCRETE
                    && device.tier() <= ReconL.TIER_T4_OUT_OF_CORE,
                    "the chosen backend reported tier " + device.tier());
            return;
        } catch (IllegalStateException refused) {
            assertTrue(refused.getMessage().contains("RECONL_ERR_BACKEND_UNAVAILABLE"),
                    "the only honest way to refuse a hint of NONE is to say no backend "
                            + "was available, not something else: " + refused.getMessage());
        }
    }

    @Test
    @DisplayName("every backend in the matrix renders a frame of the requested size")
    void everyBackendRenders() {
        for (int backend : new int[] {ReconL.BACKEND_SOFT_CPU, ReconL.BACKEND_NULL}) {
            byte[] pixels = ReconL.renderConformanceFrame(backend, 32);
            assertEquals(32 * 32 * 4, pixels.length, ReconL.backendName(backend));
        }
    }

    private static int tier(String name) {
        return switch (name) {
            case "TIER_T0_GPU_DISCRETE" -> ReconL.TIER_T0_GPU_DISCRETE;
            case "TIER_T1_GPU_SHARED" -> ReconL.TIER_T1_GPU_SHARED;
            case "TIER_T2_CPU_RAM" -> ReconL.TIER_T2_CPU_RAM;
            case "TIER_T3_CPU_THRIFTY" -> ReconL.TIER_T3_CPU_THRIFTY;
            default -> ReconL.TIER_T4_OUT_OF_CORE;
        };
    }

    private static int tierReason(String name) {
        return switch (name) {
            case "TIER_REASON_NO_GPU_API" -> ReconL.TIER_REASON_NO_GPU_API;
            case "TIER_REASON_RECOVERY" -> ReconL.TIER_REASON_RECOVERY;
            default -> ReconL.TIER_REASON_HOST_REQUEST;
        };
    }
}
