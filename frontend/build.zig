//! Build for the ReconL frontend (UI core).
//!
//! Artifacts:
//!
//! * `zig build test`      - core unit tests. No ReconL, no GPU, no window.
//!                           Font tests read the bundled Inter/Outfit.
//! * `zig build`           - installs nothing by default; tests are the
//!                           default-ish step people run. Add `install` for
//!                           artifacts.
//! * `zig build demo`      - the animated showcase, rendered *through* ReconL.
//! * `zig build abi-test`  - tests that drive the UI through the C ABI and the
//!                           real reconl import library.
//! * `zig build shared`    - installs `reconl_ui.dll`: the C ABI surface Java
//!                           (Panama), Kotlin, C and other Zig hosts consume
//!                           (`include/reconl_ui.h`).
//!
//! The ReconL-linked artifacts need `cargo build` (or `cargo build --release`
//! for `-Dreconl-profile=release`) to have produced the matching import library
//! and DLL. When they are missing the build says so and skips them rather than
//! failing - the same "skips with a printed reason" rule the C probes follow.
//!
//! Module graph (type identity matters: `geom.Vec2` imported two ways would
//! be two *different* types, so everything reaches shared code through the
//! one `reconl-frontend` module):
//!
//!   reconl-frontend (src/root.zig)  - geom theme anim tess polygon font
//!                                      text ui png; never links reconl.
//!   reconl-frontend-backend         - src/backend.zig, imports the core
//!                                      module by name, links libreconl.
//!   demo / abi-test / shared lib    - import core + backend.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The pure UI core: layout, animation, text, tessellation, widgets.
    const core = b.addModule("reconl-frontend", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{},
    });
    core.addAnonymousImport("resources", .{
        .root_source_file = b.path("assets/fonts/resources.zig"),
    });
    // `@cImport` of reconl.h lives only in backend.zig, but the include path
    // is attached here so any module that reaches backend through the core
    // sees the same header. The header is always in the repo; the *library*
    // is the optional part.
    core.addIncludePath(b.path("../include"));
    core.link_libc = true;

    // ---- core tests: never link ReconL ---------------------------------
    const core_tests = b.addTest(.{ .root_module = core });
    const run_core_tests = b.addRunArtifact(core_tests);
    const test_step = b.step("test", "Core unit tests (no ReconL required)");
    test_step.dependOn(&run_core_tests.step);

    // ---- ReconL presence probe -----------------------------------------
    // The probe is the only cwd-relative bit (LazyPaths resolve against the
    // build root on their own), so accept either invocation directory: the
    // package root or the repo root.
    const reconl_profile = b.option([]const u8, "reconl-profile", "ReconL dependency profile: debug (default) or release") orelse "debug";
    if (!std.mem.eql(u8, reconl_profile, "debug") and !std.mem.eql(u8, reconl_profile, "release")) {
        @panic("-Dreconl-profile must be debug or release");
    }
    const reconl_dir = b.fmt("../target/{s}", .{reconl_profile});
    const root_reconl_dir = b.fmt("target/{s}", .{reconl_profile});
    const reconl_import = b.fmt("{s}/libreconl.dll.a", .{reconl_dir});
    const reconl_dll = b.fmt("{s}/reconl.dll", .{reconl_dir});
    const root_reconl_import = b.fmt("{s}/libreconl.dll.a", .{root_reconl_dir});
    const root_reconl_dll = b.fmt("{s}/reconl.dll", .{root_reconl_dir});
    const probe_io = std.Io.Threaded.global_single_threaded.io();
    const probe_dir = std.Io.Dir.cwd();
    const exists = struct {
        fn go(io: std.Io, dir: std.Io.Dir, paths: []const []const u8) bool {
            for (paths) |p| {
                dir.access(io, p, .{}) catch continue;
                return true;
            }
            return false;
        }
    }.go;
    const have_reconl = exists(probe_io, probe_dir, &.{ reconl_import, root_reconl_import }) and
        exists(probe_io, probe_dir, &.{ reconl_dll, root_reconl_dll });

    const install_step = b.step("shared", "Install reconl_ui shared library (needs cargo build first)");
    const demo_step = b.step("demo", "Render the animated UI showcase through ReconL");
    const abi_step = b.step("abi-test", "Tests that drive the frontend through the C ABI and reconl");

    if (!have_reconl) {
        std.debug.print(
            \\note: reconl is not built - skipping the shared library, the demo
            \\      and the ABI tests. Run `cargo build` in the repo root first.
            \\\n
        , .{});
        // Keep the steps addressable so `zig build demo` explains itself
        // instead of "missing dependency" noise.
        const explain = b.addSystemCommand(&.{ "sh", "-c",
            "echo 'reconl not built: run cargo build in the repo root, then re-run zig build.'" });
        demo_step.dependOn(&explain.step);
        install_step.dependOn(&explain.step);
        abi_step.dependOn(&explain.step);
        return;
    }

    // Backend module: reaches shared types through the core module by name,
    // so `geom.Vec2` here is the same type as `geom.Vec2` in widgets.
    const backend = b.createModule(.{
        .root_source_file = b.path("src/backend.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "reconl-frontend", .module = core }},
    });
    backend.addIncludePath(b.path("../include"));
    backend.link_libc = true;
    backend.addLibraryPath(b.path(reconl_dir));
    backend.linkSystemLibrary("reconl", .{ .use_pkg_config = .no });

    // ---- shared library: the C ABI other languages consume --------------
    const c_api_mod = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "reconl-frontend", .module = core },
            .{ .name = "reconl-backend", .module = backend },
        },
    });
    c_api_mod.addIncludePath(b.path("include"));
    c_api_mod.link_libc = true;
    const shared = b.addLibrary(.{
        .name = "reconl_ui",
        .linkage = .dynamic,
        .root_module = c_api_mod,
    });
    // `dependOn(&shared.step)` would only build into the cache; the Install
    // step is what lands zig-out/lib/reconl_ui.dll where foreign hosts load it.
    const install_shared = b.addInstallArtifact(shared, .{});
    b.getInstallStep().dependOn(&install_shared.step);
    install_step.dependOn(&install_shared.step);

    // Windows loads DLLs from the executable's directory first: copy the
    // renderer next to whatever links us, so no PATH editing is needed.
    const dll_install = b.addInstallFileWithDir(b.path(reconl_dll), .bin, "reconl.dll");
    b.getInstallStep().dependOn(&dll_install.step);

    // ---- showcase demo --------------------------------------------------
    const demo_mod = b.createModule(.{
        .root_source_file = b.path("examples/showcase.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "reconl-frontend", .module = core },
            .{ .name = "reconl-backend", .module = backend },
        },
    });
    demo_mod.addIncludePath(b.path("include"));
    demo_mod.link_libc = true;
    const demo = b.addExecutable(.{ .name = "reconl-ui-showcase", .root_module = demo_mod });
    const run_demo = b.addRunArtifact(demo);
    run_demo.step.dependOn(b.getInstallStep());
    // The demo links libreconl.dll.a, so reconl.dll must be loadable from the
    // cache-directory binary: PATH is how the loader finds it.
    run_demo.addPathDir(b.pathFromRoot(reconl_dir));
    if (b.args) |args| run_demo.addArgs(args);
    demo_step.dependOn(&run_demo.step);

    // ---- ABI tests: the C header consumed through Zig -------------------
    const abi_mod = b.createModule(.{
        .root_source_file = b.path("src/abi_test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "reconl-frontend", .module = core },
            .{ .name = "reconl-backend", .module = backend },
        },
    });
    abi_mod.addIncludePath(b.path("include"));
    abi_mod.link_libc = true;
    const abi_tests = b.addTest(.{ .root_module = abi_mod });
    const run_abi_tests = b.addRunArtifact(abi_tests);
    run_abi_tests.addPathDir(b.pathFromRoot(reconl_dir));
    abi_step.dependOn(&run_abi_tests.step);
}
