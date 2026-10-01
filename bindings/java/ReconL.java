// Java 21 FFM bindings for the shipped ReconL C ABI.
//
// This is the second half of the Java story: the FFM smoke in
// frontend/spike/java drives reconl_ui.dll, the Zig frontend's C API, and
// proves a foreign runtime can reach this project at all. This file drives
// reconl.dll - the renderer's own ABI, the one the 17 C probes use - so the
// product's surface is exercised from a second language with a second FFI
// layer.
//
// Everything below is transcribed from include/reconl/reconl.h. Nothing here
// is allowed to be an opinion: where a constant or an offset is written down,
// the conformance suite checks it back against the library itself
// (reconlResultName / reconlTierName / reconlBackendName) or against the
// library's own struct_size refusal, because a binding that silently
// disagrees with the header is worse than no binding.
//
// Build and run:
//   javac --enable-preview --release 21 -d out bindings/java/*.java
//   java  --enable-preview --enable-native-access=ALL-UNNAMED ...
//
// `zig-out/bin` must be on PATH: SymbolLookup opens reconl.dll, and the
// dynamic loader then has to find reconl.dll's own dependency, reconl.dll.
// Passing the absolute path to the library is not enough on its own.

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.VarHandle;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HexFormat;

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/** Bindings for the shipped ReconL C ABI, plus the canonical conformance frame. */
public final class ReconL {

    private ReconL() {}

    // ---------------------------------------------------------------- results

    public static final int OK = 0;
    public static final int ERR_INVALID_ARGUMENT = -1;
    public static final int ERR_OUT_OF_MEMORY = -2;
    public static final int ERR_NOT_SUPPORTED = -3;
    public static final int ERR_BACKEND_UNAVAILABLE = -4;
    public static final int ERR_BUDGET_EXCEEDED = -5;
    public static final int ERR_DEVICE_LOST = -6;
    public static final int ERR_INVALID_HANDLE = -7;
    public static final int ERR_STRUCT_SIZE = -8;
    public static final int ERR_WRONG_STRUCT_TYPE = -9;
    public static final int ERR_ABI_VERSION = -10;
    public static final int ERR_FRAME_IN_PROGRESS = -11;
    public static final int ERR_NO_FRAME = -12;
    public static final int ERR_NOT_READY = -13;
    public static final int ERR_IO = -14;
    public static final int ERR_CORRUPT_CACHE = -15;
    public static final int ERR_DEGRADED = -16;
    public static final int ERR_PANIC = -17;
    public static final int ERR_EMPTY_FRAME = -18;

    // ----------------------------------------------------------------- tiers

    public static final int TIER_T0_GPU_DISCRETE = 0;
    public static final int TIER_T1_GPU_SHARED = 1;
    public static final int TIER_T2_CPU_RAM = 2;
    public static final int TIER_T3_CPU_THRIFTY = 3;
    public static final int TIER_T4_OUT_OF_CORE = 4;

    public static final int TIER_REASON_HOST_REQUEST = 1;

    // -------------------------------------------------------------- backends

    public static final int BACKEND_NONE = 0;
    public static final int BACKEND_SOFT_CPU = 1;
    public static final int BACKEND_NULL = 2;
    public static final int BACKEND_D3D11 = 3;

    // ---------------------------------------------------------------- formats

    public static final int FORMAT_UNKNOWN = 0;
    public static final int FORMAT_R8G8B8A8_UNORM = 1;
    public static final int FORMAT_D32_FLOAT = 8;

    // ----------------------------------------------------------- struct types

    public static final int STRUCT_DEVICE_DESC = 6;
    public static final int STRUCT_BUFFER_DESC = 7;
    public static final int STRUCT_PIPELINE_DESC = 10;
    public static final int STRUCT_COMMAND_LIST_DESC = 11;
    public static final int STRUCT_RENDER_PASS_DESC = 12;
    public static final int STRUCT_SWAPCHAIN_DESC = 13;
    public static final int STRUCT_FRAME_DESC = 14;
    public static final int STRUCT_PRESENT_DESC = 15;
    public static final int STRUCT_STATS = 17;
    public static final int STRUCT_ERROR_INFO = 20;
    public static final int STRUCT_CAMERA = 21;
    public static final int STRUCT_SOFTCPU_DESC = 64;
    public static final int STRUCT_NULL_DESC = 65;
    public static final int STRUCT_D3D11_DESC = 66;

    // ------------------------------------------------------------- buffer use

    public static final int BUFFER_VERTEX = 1;
    public static final int BUFFER_INDEX = 2;

    public static final int INDEX_UINT32 = 1;

    // ------------------------------------------------ pipeline state (raw values)

    public static final int SHADING_UNLIT = 0;
    public static final int BLEND_OPAQUE = 0;
    public static final int CULL_NONE = 0;
    /** Reversed-Z: the library's default compare, and the only one built here. */
    public static final int COMPARE_GREATER = 1;

    // ------------------------------------------------------ tier reasons (raw)

    public static final int TIER_REASON_NO_GPU_API = 9;
    public static final int TIER_REASON_RECOVERY = 10;
    public static final int MAX_TEXTURE_SLOTS = 8;
    public static final int MAX_ATTACHMENTS = 4;
    public static final int MAX_NAME = 64;
    public static final int MAX_MESSAGE = 192;
    public static final int MAX_PATH = 260;

    /** sizeof(ReconLStats) from the header, asserted by the suite against Rust. */
    public static final long SIZE_STATS_TOTAL = 4304;

    // ---------------------------------------------------------------- layouts

    /** ReconLBase: struct_size, type, next. Every descriptor below starts here. */
    public static final StructLayout BASE = MemoryLayout.structLayout(
            JAVA_INT.withName("struct_size"),
            JAVA_INT.withName("type"),
            ADDRESS.withName("next"));

    public static final StructLayout ALLOCATOR = MemoryLayout.structLayout(
            ADDRESS.withName("alloc"),
            ADDRESS.withName("realloc"),
            ADDRESS.withName("free"),
            ADDRESS.withName("user"));

    public static final StructLayout DEVICE_DESC = MemoryLayout.structLayout(
            BASE.withName("base"),
            JAVA_INT.withName("backend_hint"),
            JAVA_INT.withName("tier_hint"),
            JAVA_INT.withName("allow_downgrade"),
            JAVA_INT.withName("worker_threads"),
            JAVA_INT.withName("target_frame_ms"),
            JAVA_INT.withName("downgrade_after_frames"),
            JAVA_INT.withName("seed"),
            JAVA_INT.withName("flags"),
            ADDRESS.withName("budget"),
            ALLOCATOR.withName("allocator"),
            ADDRESS.withName("backend_desc"));

    public static final StructLayout SWAPCHAIN_DESC = MemoryLayout.structLayout(
            BASE.withName("base"),
            JAVA_INT.withName("width"),
            JAVA_INT.withName("height"),
            JAVA_INT.withName("format"),
            JAVA_INT.withName("image_count"),
            JAVA_INT.withName("present_to_memory"),
            JAVA_INT.withName("depth_format"),
            JAVA_INT.withName("flags"),
            JAVA_INT.withName("reserved"));

    public static final StructLayout COMMAND_LIST_DESC = MemoryLayout.structLayout(
            BASE.withName("base"),
            JAVA_INT.withName("capacity_bytes"),
            JAVA_INT.withName("reserved"),
            ADDRESS.withName("debug_name"));

    public static final StructLayout BUFFER_DESC = MemoryLayout.structLayout(
            BASE.withName("base"),
            JAVA_LONG.withName("size_bytes"),
            JAVA_INT.withName("usage"),
            JAVA_INT.withName("reserved"),
            ADDRESS.withName("data"),
            JAVA_LONG.withName("data_size"),
            ADDRESS.withName("debug_name"));

    public static final StructLayout PIPELINE_DESC = MemoryLayout.structLayout(
            BASE.withName("base"),
            JAVA_INT.withName("shading"),
            JAVA_INT.withName("blend"),
            JAVA_INT.withName("cull"),
            JAVA_INT.withName("depth_compare"),
            JAVA_INT.withName("depth_write"),
            JAVA_INT.withName("texture_slots"),
            MemoryLayout.sequenceLayout(MAX_TEXTURE_SLOTS, JAVA_INT).withName("texture_formats"),
            JAVA_INT.withName("receives_shadow"),
            JAVA_INT.withName("casts_shadow"),
            JAVA_INT.withName("flags"),
            JAVA_INT.withName("reserved"),
            ADDRESS.withName("debug_name"));

    /** ReconLColorAttachment: texture, resolve, mip, layer - 24 bytes. */
    private static final StructLayout COLOR_ATTACHMENT = MemoryLayout.structLayout(
            ADDRESS.withName("texture"),
            ADDRESS.withName("resolve"),
            JAVA_INT.withName("mip"),
            JAVA_INT.withName("layer"));

    public static final StructLayout RENDER_PASS_DESC = MemoryLayout.structLayout(
            BASE.withName("base"),
            JAVA_INT.withName("color_count"),
            JAVA_INT.withName("reserved"),
            MemoryLayout.sequenceLayout(MAX_ATTACHMENTS, COLOR_ATTACHMENT).withName("color"),
            ADDRESS.withName("depth"),
            JAVA_INT.withName("viewport_width"),
            JAVA_INT.withName("viewport_height"),
            JAVA_INT.withName("load_color"),
            JAVA_INT.withName("load_depth"),
            MemoryLayout.sequenceLayout(4, JAVA_FLOAT).withName("clear_color"),
            JAVA_FLOAT.withName("clear_depth"),
            JAVA_INT.withName("stencil_clear"),
            JAVA_INT.withName("reserved2"),
            // C rounds a struct up to its own alignment; a Java StructLayout
            // does not. 172 laid out, 176 in the header, and the library
            // refuses anything else with ERR_STRUCT_SIZE.
            MemoryLayout.paddingLayout(4));

    public static final StructLayout FRAME_DESC = MemoryLayout.structLayout(
            BASE.withName("base"),
            JAVA_INT.withName("width"),
            JAVA_INT.withName("height"),
            JAVA_INT.withName("seed"),
            JAVA_INT.withName("reserved"),
            ADDRESS.withName("lights"),
            ADDRESS.withName("shadows"),
            ADDRESS.withName("camera"),
            ADDRESS.withName("framegen"));

    public static final StructLayout PRESENT_DESC = MemoryLayout.structLayout(
            BASE.withName("base"),
            ADDRESS.withName("out_pixels"),
            JAVA_LONG.withName("out_pixels_size"),
            JAVA_INT.withName("out_row_pitch"),
            JAVA_INT.withName("out_format"),
            JAVA_INT.withName("flip"),
            MemoryLayout.paddingLayout(4));

    /**
     * Only the leading fields are named; the rest is opaque but the size is not.
     * The eight named uint32_t are counted off explicitly rather than folded
     * into the padding, so a field the header grows shows up as a size mismatch
     * here rather than as silently dead bytes in front of the padding.
     */
    public static final StructLayout STATS = MemoryLayout.structLayout(
            BASE.withName("base"),
            JAVA_INT.withName("backend"),
            JAVA_INT.withName("tier"),
            JAVA_INT.withName("tier_reason"),
            JAVA_INT.withName("tier_locked"),
            JAVA_INT.withName("frames_presented"),
            JAVA_INT.withName("frames_dropped"),
            JAVA_INT.withName("downgrade_count"),
            JAVA_INT.withName("downgrade_capacity"),
            MemoryLayout.paddingLayout(SIZE_STATS_TOTAL - 48));

    /**
     * `char[192]`, `char[260]`, `char[64]` - bytes, not uint32_t. Reading the
     * header's `char message[RECONL_MAX_MESSAGE]` as an int array would put the
     * message field at four times its real offset and the struct at 1120 bytes,
     * which the library's own struct_size check refuses.
     */
    public static final StructLayout ERROR_INFO = MemoryLayout.structLayout(
            BASE.withName("base"),
            JAVA_INT.withName("result"),
            MemoryLayout.sequenceLayout(MAX_MESSAGE, JAVA_BYTE).withName("message"),
            MemoryLayout.sequenceLayout(MAX_PATH, JAVA_BYTE).withName("file"),
            JAVA_INT.withName("line"),
            JAVA_INT.withName("function_name_index"),
            MemoryLayout.sequenceLayout(MAX_NAME, JAVA_BYTE).withName("function"));

    /** ReconLVertex: position[3], normal[3], uv[2], color[4] - 12 floats. */
    public static final StructLayout VERTEX = MemoryLayout.structLayout(
            MemoryLayout.sequenceLayout(3, JAVA_FLOAT).withName("position"),
            MemoryLayout.sequenceLayout(3, JAVA_FLOAT).withName("normal"),
            MemoryLayout.sequenceLayout(2, JAVA_FLOAT).withName("uv"),
            MemoryLayout.sequenceLayout(4, JAVA_FLOAT).withName("color"));

    // The header sizes, from `pixel_hash --sizes`. A mismatch is not a warning:
    // the library refuses the call with ERR_STRUCT_SIZE, so a wrong layout here
    // cannot reach a frame.
    public static final long SIZE_BASE = 16, SIZE_ALLOCATOR = 32, SIZE_DEVICE_DESC = 96;
    public static final long SIZE_SWAPCHAIN_DESC = 48, SIZE_COMMAND_LIST_DESC = 32;
    public static final long SIZE_BUFFER_DESC = 56, SIZE_PIPELINE_DESC = 96;
    public static final long SIZE_RENDER_PASS_DESC = 176, SIZE_FRAME_DESC = 64;
    public static final long SIZE_PRESENT_DESC = 48, SIZE_STATS = 4304, SIZE_ERROR_INFO = 544;
    public static final long SIZE_VERTEX = 48;

    /**
     * Every layout, checked against the C `sizeof` at class-init. The library
     * checks struct_size too, but only on the calls a test happens to make; a
     * layout nothing has touched yet should still not be able to be wrong, and
     * a binding that is quietly 4 bytes short is the exact failure this whole
     * suite exists to catch.
     */
    static {
        Object[][] expected = {
            {BASE, SIZE_BASE}, {ALLOCATOR, SIZE_ALLOCATOR}, {DEVICE_DESC, SIZE_DEVICE_DESC},
            {SWAPCHAIN_DESC, SIZE_SWAPCHAIN_DESC}, {COMMAND_LIST_DESC, SIZE_COMMAND_LIST_DESC},
            {BUFFER_DESC, SIZE_BUFFER_DESC}, {PIPELINE_DESC, SIZE_PIPELINE_DESC},
            {RENDER_PASS_DESC, SIZE_RENDER_PASS_DESC}, {FRAME_DESC, SIZE_FRAME_DESC},
            {PRESENT_DESC, SIZE_PRESENT_DESC}, {STATS, SIZE_STATS}, {ERROR_INFO, SIZE_ERROR_INFO},
            {VERTEX, SIZE_VERTEX},
        };
        for (Object[] row : expected) {
            StructLayout layout = (StructLayout) row[0];
            long size = (Long) row[1];
            if (layout.byteSize() != size) {
                throw new ExceptionInInitializerError(new IllegalStateException(
                        "ReconL.java layout is " + layout.byteSize() + " bytes; the header says "
                                + size + ". Every field offset below it is also suspect."));
            }
        }
    }

    // ------------------------------------------------------------- linkage

    private static final Arena LIBRARY = Arena.global();
    private static final Linker LINKER = Linker.nativeLinker();

    private static final SymbolLookup LOOKUP = openLibrary();

    private static SymbolLookup openLibrary() {
        String explicit = System.getProperty("reconl.dll");
        String name = (explicit != null && !explicit.isBlank())
                ? explicit
                : "reconl.dll";
        return SymbolLookup.libraryLookup(name, LIBRARY);
    }

    /**
     * Downcall handles are resolved once and kept. `Linker.downcallHandle` is
     * not free, and a conformance suite calls these from a loop.
     */
    private static final java.util.Map<String, MethodHandle> DOWN_CALLS = new java.util.HashMap<>();

    private static MethodHandle downcall(String symbol, FunctionDescriptor type) {
        synchronized (DOWN_CALLS) {
            MethodHandle cached = DOWN_CALLS.get(symbol);
            if (cached != null) {
                return cached;
            }
        }
        MethodHandle handle = LINKER.downcallHandle(
                LOOKUP.find(symbol).orElseThrow(
                        () -> new IllegalStateException("reconl.dll has no symbol " + symbol)), type);
        synchronized (DOWN_CALLS) {
            DOWN_CALLS.put(symbol, handle);
        }
        return handle;
    }

    // The library calls back for every allocation, so the three allocator
    // functions are real upcalls rather than a description of one.
    private static final MemorySegment STUB_ALLOC;
    private static final MemorySegment STUB_REALLOC;
    private static final MemorySegment STUB_FREE;

    static {
        try {
            MethodHandles.Lookup lookup = MethodHandles.lookup();
            MemorySegment alloc = LINKER.upcallStub(
                    lookup.findStatic(ReconL.class, "nativeAlloc",
                            MethodType.methodType(MemorySegment.class, MemorySegment.class, long.class, long.class)),
                    FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG), LIBRARY);
            MemorySegment realloc = LINKER.upcallStub(
                    lookup.findStatic(ReconL.class, "nativeRealloc",
                            MethodType.methodType(MemorySegment.class, MemorySegment.class,
                                    MemorySegment.class, long.class, long.class, long.class)),
                    FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG), LIBRARY);
            MemorySegment free = LINKER.upcallStub(
                    lookup.findStatic(ReconL.class, "nativeFree",
                            MethodType.methodType(void.class, MemorySegment.class, MemorySegment.class, long.class)),
                    FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG), LIBRARY);
            STUB_ALLOC = alloc;
            STUB_REALLOC = realloc;
            STUB_FREE = free;
        } catch (Throwable t) {
            throw new ExceptionInInitializerError(t);
        }
    }

    /** Backing store for one library allocation; freed by {@link #nativeFree}. */
    private record Allocation(Arena arena, MemorySegment segment) {}

    private static final java.util.Map<MemorySegment, Allocation> LIVE = new java.util.HashMap<>();

    static MemorySegment nativeAlloc(MemorySegment user, long size, long align) {
        long a = Math.max(Math.max(align, 16), 8);
        // Shared, not confined: the library owns which thread calls back, and a
        // worker thread arriving at a confined arena's allocate would throw
        // WrongThreadException from inside native code.
        Arena arena = Arena.ofShared();
        MemorySegment s = arena.allocate(Math.max(size, 1), a);
        synchronized (LIVE) {
            LIVE.put(s, new Allocation(arena, s));
        }
        return s;
    }

    static MemorySegment nativeRealloc(MemorySegment user, MemorySegment ptr, long oldSize, long newSize, long align) {
        MemorySegment fresh = nativeAlloc(user, newSize, align);
        // A downcall's ADDRESS argument arrives as a zero-length segment over
        // the real pointer, so the copy has to re-establish the extent from
        // the size the library just handed us.
        long moved = Math.min(Math.max(oldSize, 0), Math.max(newSize, 0));
        if (moved > 0) {
            MemorySegment.copy(ptr.reinterpret(moved), 0, fresh, 0, moved);
        }
        nativeFree(user, ptr, oldSize);
        return fresh;
    }

    static void nativeFree(MemorySegment user, MemorySegment ptr, long size) {
        Allocation held;
        synchronized (LIVE) {
            held = LIVE.remove(ptr);
        }
        if (held != null) {
            held.arena().close();
        }
    }

    // ---------------------------------------------------------------- helpers

    static void setBase(MemorySegment s, StructLayout layout, int type, long expectedSize) {
        if (layout.byteSize() != expectedSize) {
            throw new IllegalStateException("layout for type " + type + " is " + layout.byteSize()
                    + " bytes, the header says " + expectedSize);
        }
        VarHandle size = BASE.varHandle(groupElement("struct_size"));
        VarHandle kind = BASE.varHandle(groupElement("type"));
        VarHandle next = BASE.varHandle(groupElement("next"));
        size.set(s, (int) expectedSize);
        kind.set(s, type);
        next.set(s, MemorySegment.NULL);
    }

    static MemorySegment zeroed(Arena arena, StructLayout layout) {
        MemorySegment s = arena.allocate(layout);
        s.fill((byte) 0);
        return s;
    }

    /**
     * Writes a `float[N]` C array member. The offset comes from the layout rather
     * than a running counter so that reordering the struct in Java is caught
     * against the layout instead of quietly writing the wrong field.
     */
    static void writeFloats(MemorySegment s, StructLayout layout, String field, float[] values) {
        long at = layout.byteOffset(groupElement(field));
        for (float v : values) {
            s.set(JAVA_FLOAT, at, v);
            at += Float.BYTES;
        }
    }

    /** Reads a NUL-terminated C string out of a returned `const char *`. */
    static String cString(MemorySegment p) {
        if (p.address() == 0) {
            return null;
        }
        MemorySegment s = p.reinterpret(256);
        int n = 0;
        while (n < 256 && s.get(JAVA_BYTE, n) != 0) {
            n++;
        }
        byte[] bytes = new byte[n];
        MemorySegment.copy(s, 0, MemorySegment.ofArray(bytes), 0, n);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    public static String resultName(int code) {
        try {
            return cString((MemorySegment) downcall("reconlResultName",
                    FunctionDescriptor.of(ADDRESS, JAVA_INT)).invokeExact(code));
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    public static String tierName(int tier) {
        try {
            return cString((MemorySegment) downcall("reconlTierName",
                    FunctionDescriptor.of(ADDRESS, JAVA_INT)).invokeExact(tier));
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    public static String backendName(int backend) {
        try {
            return cString((MemorySegment) downcall("reconlBackendName",
                    FunctionDescriptor.of(ADDRESS, JAVA_INT)).invokeExact(backend));
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    // ------------------------------------------------------------- the ABI

    static int reconlCreateDevice(MemorySegment desc, MemorySegment out) {
        try {
            return (int) downcall("reconlCreateDevice", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS))
                    .invokeExact(desc, out);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCreateSwapchain(MemorySegment device, MemorySegment desc, MemorySegment out) {
        try {
            return (int) downcall("reconlCreateSwapchain", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
                    .invokeExact(device, desc, out);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCreateCommandList(MemorySegment device, MemorySegment desc, MemorySegment out) {
        try {
            return (int) downcall("reconlCreateCommandList", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
                    .invokeExact(device, desc, out);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCreateBuffer(MemorySegment device, MemorySegment desc, MemorySegment out) {
        try {
            return (int) downcall("reconlCreateBuffer", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
                    .invokeExact(device, desc, out);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCreatePipeline(MemorySegment device, MemorySegment desc, MemorySegment out) {
        try {
            return (int) downcall("reconlCreatePipeline", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
                    .invokeExact(device, desc, out);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlBeginFrame(MemorySegment device, MemorySegment desc) {
        try {
            return (int) downcall("reconlBeginFrame", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS))
                    .invokeExact(device, desc);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCmdReset(MemorySegment list) {
        try {
            return (int) downcall("reconlCmdReset", FunctionDescriptor.of(JAVA_INT, ADDRESS)).invokeExact(list);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCmdBeginRenderPass(MemorySegment list, MemorySegment desc) {
        try {
            return (int) downcall("reconlCmdBeginRenderPass", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS))
                    .invokeExact(list, desc);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCmdSetPipeline(MemorySegment list, MemorySegment pipeline) {
        try {
            return (int) downcall("reconlCmdSetPipeline", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS))
                    .invokeExact(list, pipeline);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCmdPushConstants(MemorySegment list, int slot, MemorySegment data, int size) {
        try {
            return (int) downcall("reconlCmdPushConstants",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT))
                    .invokeExact(list, slot, data, size);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCmdSetVertexBuffer(MemorySegment list, int stream, MemorySegment buffer, long offset) {
        try {
            return (int) downcall("reconlCmdSetVertexBuffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG))
                    .invokeExact(list, stream, buffer, offset);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCmdSetIndexBuffer(MemorySegment list, MemorySegment buffer, long offset, int format) {
        try {
            return (int) downcall("reconlCmdSetIndexBuffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT))
                    .invokeExact(list, buffer, offset, format);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCmdDrawIndexed(MemorySegment list, int indexCount, int firstIndex, int vertexOffset) {
        try {
            return (int) downcall("reconlCmdDrawIndexed",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT))
                    .invokeExact(list, indexCount, firstIndex, vertexOffset);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlCmdEndRenderPass(MemorySegment list) {
        try {
            return (int) downcall("reconlCmdEndRenderPass", FunctionDescriptor.of(JAVA_INT, ADDRESS))
                    .invokeExact(list);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlSubmit(MemorySegment device, MemorySegment list, MemorySegment fence) {
        try {
            return (int) downcall("reconlSubmit", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
                    .invokeExact(device, list, fence);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlPresent(MemorySegment device, MemorySegment swapchain, MemorySegment desc) {
        try {
            return (int) downcall("reconlPresent", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
                    .invokeExact(device, swapchain, desc);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlGetStats(MemorySegment device, MemorySegment out) {
        try {
            return (int) downcall("reconlGetStats", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS))
                    .invokeExact(device, out);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlGetLastError(MemorySegment device, MemorySegment out) {
        try {
            return (int) downcall("reconlGetLastError", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS))
                    .invokeExact(device, out);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlRequestTier(MemorySegment device, int tier, int reason) {
        try {
            return (int) downcall("reconlRequestTier", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT))
                    .invokeExact(device, tier, reason);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    static int reconlRelease(MemorySegment handle) {
        try {
            return (int) downcall("reconlRelease", FunctionDescriptor.of(JAVA_INT, ADDRESS)).invokeExact(handle);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    // ------------------------------------------------- device + conformance frame

    /** A device plus the scratch every descriptor needs, kept alive together. */
    public static final class Device implements AutoCloseable {
        final Arena arena = Arena.ofConfined();
        final MemorySegment handle;
        final int backendHint;
        private boolean released;

        public Device(int backendHint, int tierHint) {
            this.backendHint = backendHint;
            MemorySegment desc = zeroed(arena, DEVICE_DESC);
            setBase(desc, DEVICE_DESC, STRUCT_DEVICE_DESC, SIZE_DEVICE_DESC);
            VarHandle f = DEVICE_DESC.varHandle(groupElement("backend_hint"));
            f.set(desc, backendHint);
            DEVICE_DESC.varHandle(groupElement("tier_hint")).set(desc, tierHint);
            DEVICE_DESC.varHandle(groupElement("seed")).set(desc, 7);
            // Through the *device* layout, not the bare allocator layout: the
            // allocator sits 56 bytes into the descriptor, so a VarHandle taken
            // from ALLOCATOR would write the callbacks over the descriptor's
            // own base.struct_size.
            DEVICE_DESC.varHandle(groupElement("allocator"), groupElement("alloc")).set(desc, STUB_ALLOC);
            DEVICE_DESC.varHandle(groupElement("allocator"), groupElement("realloc")).set(desc, STUB_REALLOC);
            DEVICE_DESC.varHandle(groupElement("allocator"), groupElement("free")).set(desc, STUB_FREE);
            DEVICE_DESC.varHandle(groupElement("allocator"), groupElement("user")).set(desc, MemorySegment.NULL);

            MemorySegment out = arena.allocate(ADDRESS);
            int rc = reconlCreateDevice(desc, out);
            if (rc != OK) {
                arena.close();
                throw new IllegalStateException("reconlCreateDevice: " + resultName(rc));
            }
            this.handle = out.get(ADDRESS, 0);
        }

        /** The opaque `ReconLDevice *`, for the tests that exercise the guards. */
        public MemorySegment handle() {
            return handle;
        }

        public int backend() {
            return statsField("backend");
        }

        public int tier() {
            return statsField("tier");
        }

        public int tierReason() {
            return statsField("tier_reason");
        }

        private int statsField(String field) {
            MemorySegment s = zeroed(arena, STATS);
            setBase(s, STATS, STRUCT_STATS, SIZE_STATS);
            if (reconlGetStats(handle, s) != OK) {
                return -1;
            }
            return (int) STATS.varHandle(groupElement(field)).get(s);
        }

        public String lastError() {
            MemorySegment s = zeroed(arena, ERROR_INFO);
            setBase(s, ERROR_INFO, STRUCT_ERROR_INFO, SIZE_ERROR_INFO);
            if (reconlGetLastError(handle, s) != OK) {
                return null;
            }
            long base = ERROR_INFO.byteOffset(groupElement("message"));
            MemorySegment msg = s.asSlice(base, MAX_MESSAGE);
            return cString(msg);
        }

        public int requestTier(int tier) {
            return reconlRequestTier(handle, tier, TIER_REASON_HOST_REQUEST);
        }

        @Override
        public void close() {
            // Idempotent: a test that closes in a finally *and* through
            // try-with-resources would otherwise release the handle twice.
            if (!released) {
                released = true;
                if (handle.address() != 0) {
                    reconlRelease(handle);
                }
            }
            arena.close();
        }
    }

// ---------------------------------------------------------------- the rig

    /**
     * One device, swapchain, command list, geometry and pipeline, kept alive
     * together, with the frame split into the calls the host actually makes.
     *
     * The conformance frame used to be one long method, which is the right
     * shape for "render it and hash it" and the wrong shape for a suite: every
     * refusal path in the ABI lives *between* two of those calls, and a method
     * that always runs all of them cannot be stopped half way to ask what the
     * library does at the half-way mark. So the calls are separated here and
     * `renderConformanceFrame` is the special case that runs them in order.
     */
    public static final class Rig implements AutoCloseable {
        private final Arena arena = Arena.ofConfined();
        private final MemorySegment sc, cl, vb, ib, pipe;
        private final MemorySegment swapchain, list, vertexBuffer, indexBuffer, pipeline;
        private boolean closed;

        public final Device device;
        public final int size;
        public final int backendHint;

        public Rig(int backend, int size) {
            this(backend, size, 0);
        }

        public Rig(int backend, int size, int tierHint) {
            this.size = size;
            this.backendHint = backend;
            this.device = new Device(backend, tierHint);

            sc = arena.allocate(ADDRESS);
            MemorySegment sd = zeroed(arena, SWAPCHAIN_DESC);
            setBase(sd, SWAPCHAIN_DESC, STRUCT_SWAPCHAIN_DESC, SIZE_SWAPCHAIN_DESC);
            SWAPCHAIN_DESC.varHandle(groupElement("width")).set(sd, size);
            SWAPCHAIN_DESC.varHandle(groupElement("height")).set(sd, size);
            SWAPCHAIN_DESC.varHandle(groupElement("format")).set(sd, FORMAT_R8G8B8A8_UNORM);
            SWAPCHAIN_DESC.varHandle(groupElement("image_count")).set(sd, 2);
            SWAPCHAIN_DESC.varHandle(groupElement("present_to_memory")).set(sd, 1);
            SWAPCHAIN_DESC.varHandle(groupElement("depth_format")).set(sd, FORMAT_D32_FLOAT);
            require(reconlCreateSwapchain(device.handle, sd, sc), "reconlCreateSwapchain");
            swapchain = sc.get(ADDRESS, 0);

            cl = arena.allocate(ADDRESS);
            MemorySegment cd = zeroed(arena, COMMAND_LIST_DESC);
            setBase(cd, COMMAND_LIST_DESC, STRUCT_COMMAND_LIST_DESC, SIZE_COMMAND_LIST_DESC);
            COMMAND_LIST_DESC.varHandle(groupElement("capacity_bytes")).set(cd, 8192);
            require(reconlCreateCommandList(device.handle, cd, cl), "reconlCreateCommandList");
            list = cl.get(ADDRESS, 0);

            // One triangle covering the whole clip-space cube, unlit white.
            float[][] pos = {{-1f, -1f, 0.5f}, {3f, -1f, 0.5f}, {-1f, 3f, 0.5f}};
            MemorySegment verts = arena.allocateArray(VERTEX, 3);
            for (int i = 0; i < 3; i++) {
                MemorySegment v = verts.asSlice(i * SIZE_VERTEX, SIZE_VERTEX);
                writeFloats(v, VERTEX, "position", pos[i]);
                writeFloats(v, VERTEX, "normal", new float[] {0f, 0f, 1f});
                writeFloats(v, VERTEX, "uv", new float[] {0f, 0f});
                writeFloats(v, VERTEX, "color", new float[] {1f, 1f, 1f, 1f});
            }
            MemorySegment idx = arena.allocate(3L * Integer.BYTES, Integer.BYTES);
            for (int i = 0; i < 3; i++) {
                idx.set(JAVA_INT, i * (long) Integer.BYTES, i);
            }

            vb = arena.allocate(ADDRESS);
            MemorySegment bd = zeroed(arena, BUFFER_DESC);
            setBase(bd, BUFFER_DESC, STRUCT_BUFFER_DESC, SIZE_BUFFER_DESC);
            BUFFER_DESC.varHandle(groupElement("size_bytes")).set(bd, 3 * SIZE_VERTEX);
            BUFFER_DESC.varHandle(groupElement("usage")).set(bd, BUFFER_VERTEX);
            BUFFER_DESC.varHandle(groupElement("data")).set(bd, verts);
            BUFFER_DESC.varHandle(groupElement("data_size")).set(bd, 3 * SIZE_VERTEX);
            require(reconlCreateBuffer(device.handle, bd, vb), "reconlCreateBuffer(vertex)");
            vertexBuffer = vb.get(ADDRESS, 0);

            ib = arena.allocate(ADDRESS);
            MemorySegment bdi = zeroed(arena, BUFFER_DESC);
            setBase(bdi, BUFFER_DESC, STRUCT_BUFFER_DESC, SIZE_BUFFER_DESC);
            BUFFER_DESC.varHandle(groupElement("size_bytes")).set(bdi, 3L * Integer.BYTES);
            BUFFER_DESC.varHandle(groupElement("usage")).set(bdi, BUFFER_INDEX);
            BUFFER_DESC.varHandle(groupElement("data")).set(bdi, idx);
            BUFFER_DESC.varHandle(groupElement("data_size")).set(bdi, 3L * Integer.BYTES);
            require(reconlCreateBuffer(device.handle, bdi, ib), "reconlCreateBuffer(index)");
            indexBuffer = ib.get(ADDRESS, 0);

            pipe = arena.allocate(ADDRESS);
            MemorySegment pd = zeroed(arena, PIPELINE_DESC);
            setBase(pd, PIPELINE_DESC, STRUCT_PIPELINE_DESC, SIZE_PIPELINE_DESC);
            PIPELINE_DESC.varHandle(groupElement("shading")).set(pd, SHADING_UNLIT);
            PIPELINE_DESC.varHandle(groupElement("blend")).set(pd, BLEND_OPAQUE);
            PIPELINE_DESC.varHandle(groupElement("cull")).set(pd, CULL_NONE);
            // Reversed-Z is the default, and the conformance frame is built in
            // clip space with no camera, so the compare has to be GREATER.
            PIPELINE_DESC.varHandle(groupElement("depth_compare")).set(pd, COMPARE_GREATER);
            PIPELINE_DESC.varHandle(groupElement("depth_write")).set(pd, 1);
            require(reconlCreatePipeline(device.handle, pd, pipe), "reconlCreatePipeline");
            pipeline = pipe.get(ADDRESS, 0);
        }

        public MemorySegment swapchain() { return swapchain; }
        public MemorySegment list() { return list; }
        public MemorySegment pipeline() { return pipeline; }

        /** A frame descriptor for this rig, so a test can corrupt one field. */
        public MemorySegment frameDesc() {
            MemorySegment fd = zeroed(arena, FRAME_DESC);
            setBase(fd, FRAME_DESC, STRUCT_FRAME_DESC, SIZE_FRAME_DESC);
            FRAME_DESC.varHandle(groupElement("width")).set(fd, size);
            FRAME_DESC.varHandle(groupElement("height")).set(fd, size);
            FRAME_DESC.varHandle(groupElement("seed")).set(fd, 1);
            return fd;
        }

        public int beginFrame() {
            return reconlBeginFrame(device.handle, frameDesc());
        }

        /** The conformance draw: one pass, a full-clip unlit white triangle. */
        public int draw() {
            MemorySegment rp = zeroed(arena, RENDER_PASS_DESC);
            setBase(rp, RENDER_PASS_DESC, STRUCT_RENDER_PASS_DESC, SIZE_RENDER_PASS_DESC);
            RENDER_PASS_DESC.varHandle(groupElement("load_color")).set(rp, 1);
            RENDER_PASS_DESC.varHandle(groupElement("load_depth")).set(rp, 1);
            writeFloats(rp, RENDER_PASS_DESC, "clear_color", new float[] {0f, 0f, 1f, 1f});
            RENDER_PASS_DESC.varHandle(groupElement("clear_depth")).set(rp, 0f);

            reconlCmdReset(list);
            int rc = reconlCmdBeginRenderPass(list, rp);
            if (rc != OK) {
                return rc;
            }
            MemorySegment ident = arena.allocate(16L * Float.BYTES, Float.BYTES);
            for (int i = 0; i < 16; i++) {
                ident.set(JAVA_FLOAT, i * (long) Float.BYTES, i % 5 == 0 ? 1f : 0f);
            }
            reconlCmdSetPipeline(list, pipeline);
            reconlCmdPushConstants(list, 0, ident, 64);
            reconlCmdPushConstants(list, 1, ident, 64);
            rc = reconlCmdSetVertexBuffer(list, 0, vertexBuffer, 0);
            if (rc != OK) {
                return rc;
            }
            rc = reconlCmdSetIndexBuffer(list, indexBuffer, 0, INDEX_UINT32);
            if (rc != OK) {
                return rc;
            }
            rc = reconlCmdDrawIndexed(list, 3, 0, 0);
            if (rc != OK) {
                return rc;
            }
            return reconlCmdEndRenderPass(list);
        }

        public int submit() {
            return reconlSubmit(device.handle, list, MemorySegment.NULL);
        }

        /** A present descriptor for this rig's size, tight row pitch. */
        public MemorySegment presentDesc(MemorySegment out, int outBytes, int rowPitch) {
            MemorySegment pr = zeroed(arena, PRESENT_DESC);
            setBase(pr, PRESENT_DESC, STRUCT_PRESENT_DESC, SIZE_PRESENT_DESC);
            PRESENT_DESC.varHandle(groupElement("out_pixels")).set(pr, out);
            PRESENT_DESC.varHandle(groupElement("out_pixels_size")).set(pr, outBytes);
            PRESENT_DESC.varHandle(groupElement("out_row_pitch")).set(pr, rowPitch);
            PRESENT_DESC.varHandle(groupElement("out_format")).set(pr, FORMAT_R8G8B8A8_UNORM);
            return pr;
        }

        public int present(MemorySegment desc) {
            return reconlPresent(device.handle, swapchain, desc);
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            reconlRelease(pipeline);
            reconlRelease(vertexBuffer);
            reconlRelease(indexBuffer);
            reconlRelease(list);
            reconlRelease(swapchain);
            device.close();
            arena.close();
        }
    }

    /**
     * The canonical conformance frame: a full-clip unlit white triangle over a
     * blue clear, RGBA8, tight row pitch, read back through Present.
     *
     * This is the frame `ffi/examples/pixel_hash.rs` renders on the Rust side.
     * Both are built from the same header and must hash the same, which is the
     * only assertion in the suite that checks the *binding* rather than the
     * library: every other test would still pass if this one FFI layer had
     * quietly disagreed with reconl.h about a field order.
     */
    public static byte[] renderConformanceFrame(int backend, int size) {
        try (Rig rig = new Rig(backend, size)) {
            require(rig.beginFrame(), "reconlBeginFrame");
            require(rig.draw(), "conformance draw");
            require(rig.submit(), "reconlSubmit");
            byte[] pixels = new byte[size * size * 4];
            int rc;
            try (Arena out = Arena.ofConfined()) {
                MemorySegment seg = out.allocate(pixels.length, 1);
                rc = rig.present(rig.presentDesc(seg, pixels.length, size * 4));
                require(rc, "reconlPresent");
                MemorySegment.copy(seg, 0, MemorySegment.ofArray(pixels), 0, pixels.length);
            }
            return pixels;
        }
    }

    static void require(int rc, String what) {
        if (rc != OK) {
            throw new IllegalStateException(what + ": " + resultName(rc) + " (" + rc + ")");
        }
    }

    public static String sha256Hex(byte[] data) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(data));
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}
