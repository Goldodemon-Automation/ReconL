import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/** Minimal Java 21 Panama consumer: python spike/verify_java.py builds and runs it. */
public final class ReconLUiSmoke {
    private static final int WIDTH = 320;
    private static final int HEIGHT = 180;
    private static final int ABI_VERSION = 1;

    private static final MemoryLayout INPUT_LAYOUT = MemoryLayout.structLayout(
            JAVA_FLOAT.withName("px"),
            JAVA_FLOAT.withName("py"),
            JAVA_INT.withName("down"),
            JAVA_INT.withName("pressed"),
            JAVA_INT.withName("released"),
            JAVA_FLOAT.withName("wheel"),
            JAVA_FLOAT.withName("dt_ms"));

    private final MethodHandle abiVersion;
    private final MethodHandle create;
    private final MethodHandle destroy;
    private final MethodHandle begin;
    private final MethodHandle label;
    private final MethodHandle button;
    private final MethodHandle endFrame;
    private final MethodHandle render;
    private final MethodHandle frameIndex;

    private ReconLUiSmoke(Linker linker, SymbolLookup library) {
        abiVersion = bind(linker, library, "reconlUiAbiVersion",
                FunctionDescriptor.of(JAVA_INT));
        create = bind(linker, library, "reconlUiCreate",
                FunctionDescriptor.of(ADDRESS, JAVA_INT, JAVA_INT));
        destroy = bind(linker, library, "reconlUiDestroy",
                FunctionDescriptor.ofVoid(ADDRESS));
        begin = bind(linker, library, "reconlUiBegin",
                FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
        label = bind(linker, library, "reconlUiLabel",
                FunctionDescriptor.of(JAVA_FLOAT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
        button = bind(linker, library, "reconlUiButton",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT));
        endFrame = bind(linker, library, "reconlUiEndFrame",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));
        render = bind(linker, library, "reconlUiRender",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));
        frameIndex = bind(linker, library, "reconlUiFrameIndex",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));
    }

    private static MethodHandle bind(Linker linker, SymbolLookup library,
                                     String name, FunctionDescriptor descriptor) {
        MemorySegment address = library.find(name)
                .orElseThrow(() -> new UnsatisfiedLinkError("Missing C ABI symbol: " + name));
        return linker.downcallHandle(address, descriptor);
    }

    private void run(Arena arena) throws Throwable {
        int version = (int) abiVersion.invokeExact();
        require(version == ABI_VERSION, "Expected ABI version 1, got " + version);

        MemorySegment ui = (MemorySegment) create.invokeExact(WIDTH, HEIGHT);
        require(ui.address() != 0, "reconlUiCreate returned NULL");
        try {
            MemorySegment input = arena.allocate(INPUT_LAYOUT);
            setInput(input, -1_000_000.0f, -1_000_000.0f, 0, 0, 0, 0.0f);

            begin.invokeExact(ui, input);
            float labelHeight = (float) label.invokeExact(
                    ui, arena.allocateUtf8String("Hello from Java 21 Panama"), 2, MemorySegment.NULL);
            require(labelHeight > 0.0f, "reconlUiLabel did not lay out text");

            int hasMesh = (int) endFrame.invokeExact(ui);
            require(hasMesh == 1, "reconlUiEndFrame produced no mesh");

            long pixelBytes = (long) WIDTH * HEIGHT * 4;
            MemorySegment pixels = arena.allocate(pixelBytes, 4);
            int result = (int) render.invokeExact(ui, pixels, pixelBytes);
            require(result == 0, "reconlUiRender returned " + result);
            require(hasNonBackgroundPixel(pixels, pixelBytes), "rendered frame contains only the clear colour");

            int renderedFrames = (int) frameIndex.invokeExact(ui);
            require(renderedFrames == 1, "Expected one successful render, got " + renderedFrames);

            // The root column's first button occupies y=0..34. Press must not
            // click; releasing over that same widget must report exactly one.
            setInput(input, WIDTH / 2.0f, 17.0f, 1, 1, 0, 0.0f);
            begin.invokeExact(ui, input);
            MemorySegment id = arena.allocateUtf8String("java-click");
            MemorySegment text = arena.allocateUtf8String("Click");
            int pressed = (int) button.invokeExact(ui, id, text, 0);
            require(pressed == 0, "button reported a click on press");
            require((int) endFrame.invokeExact(ui) == 1, "press frame produced no mesh");

            setInput(input, WIDTH / 2.0f, 17.0f, 0, 0, 1, 0.0f);
            begin.invokeExact(ui, input);
            int released = (int) button.invokeExact(ui, id, text, 0);
            require(released == 1, "button did not report the Java-driven release click");
            require((int) endFrame.invokeExact(ui) == 1, "release frame produced no mesh");
        } finally {
            destroy.invokeExact(ui);
        }
        System.out.println("Java 21 Panama FFM: rendered " + WIDTH + "x" + HEIGHT
                + " frame and verified button press/release through the C ABI.");
        System.out.println("Context destruction completed.");
    }

    private static long inputOffset(String field) {
        return INPUT_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement(field));
    }

    private static void setInput(MemorySegment input, float px, float py,
                                 int down, int pressed, int released, float wheel) {
        input.set(JAVA_FLOAT, inputOffset("px"), px);
        input.set(JAVA_FLOAT, inputOffset("py"), py);
        input.set(JAVA_INT, inputOffset("down"), down);
        input.set(JAVA_INT, inputOffset("pressed"), pressed);
        input.set(JAVA_INT, inputOffset("released"), released);
        input.set(JAVA_FLOAT, inputOffset("wheel"), wheel);
        input.set(JAVA_FLOAT, inputOffset("dt_ms"), 1000.0f / 60.0f);
    }

    private static boolean hasNonBackgroundPixel(MemorySegment pixels, long bytes) {
        for (long offset = 0; offset < bytes; offset += 4) {
            int red = Byte.toUnsignedInt(pixels.get(ValueLayout.JAVA_BYTE, offset));
            int green = Byte.toUnsignedInt(pixels.get(ValueLayout.JAVA_BYTE, offset + 1));
            int blue = Byte.toUnsignedInt(pixels.get(ValueLayout.JAVA_BYTE, offset + 2));
            if (red != 0x0B || green != 0x0D || blue != 0x0C) return true;
        }
        return false;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalStateException(message);
    }

    public static void main(String[] args) throws Throwable {
        if (INPUT_LAYOUT.byteSize() != 28) {
            throw new IllegalStateException("ReconLUiInput layout is " + INPUT_LAYOUT.byteSize() + " bytes, expected 28");
        }
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: ReconLUiSmoke <path-to-reconl_ui.dll>");
        }

        Linker linker = Linker.nativeLinker();
        try (Arena arena = Arena.ofConfined()) {
            SymbolLookup library = SymbolLookup.libraryLookup(Path.of(args[0]), arena);
            new ReconLUiSmoke(linker, library).run(arena);
        }
    }
}
