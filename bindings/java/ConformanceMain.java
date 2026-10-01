// Prints the conformance digest from Java, for the shell side of the
// cross-runtime check. The JUnit suite asserts the same value; this exists so
// `diff <(java ...) <(cargo run ...)` is a one-liner when the two disagree and
// no test runner is in the way.
public final class ConformanceMain {
    public static void main(String[] args) {
        int backend = ReconL.BACKEND_SOFT_CPU;
        int size = 64;
        for (String a : args) {
            if (a.startsWith("--backend=")) {
                backend = switch (a.substring("--backend=".length())) {
                    case "null" -> ReconL.BACKEND_NULL;
                    case "d3d11" -> ReconL.BACKEND_D3D11;
                    default -> ReconL.BACKEND_SOFT_CPU;
                };
            } else if (a.startsWith("--size=")) {
                size = Integer.parseInt(a.substring("--size=".length()));
            }
        }
        byte[] pixels = ReconL.renderConformanceFrame(backend, size);
        System.out.println("size=" + size + " backend=" + backend + " bytes=" + pixels.length);
        System.out.println("sha256=" + ReconL.sha256Hex(pixels));
    }
}
