"""Build and run the Java 21 Panama FFM consumer against reconl_ui.dll.

From frontend/: python spike/verify_java.py
Requires Java 21, Python 3, Zig, and the ReconL DLL/import library produced by
`cargo build` in the repository root. Java classes are compiled in a temporary
folder; `zig build shared` builds and installs the DLL and renderer dependency.
"""
from pathlib import Path
import os
import subprocess
import tempfile

FRONTEND = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent / "java" / "ReconLUiSmoke.java"
DLL_DIR = FRONTEND / "zig-out" / "bin"
DLL = DLL_DIR / "reconl_ui.dll"
RECONL_IMPORT = FRONTEND.parent / "target" / "debug" / "libreconl.dll.a"
RECONL_DLL = FRONTEND.parent / "target" / "debug" / "reconl.dll"


def run(command, *, cwd, env=None):
    print("$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_reconl_build(import_library=RECONL_IMPORT, renderer_dll=RECONL_DLL):
    missing = [path for path in (import_library, renderer_dll) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Build ReconL first with `cargo build` from the repository root; missing: "
            + ", ".join(str(path) for path in missing)
        )


def main():
    require_reconl_build()
    run(["zig", "build", "shared"], cwd=FRONTEND)
    if not DLL.is_file():
        raise FileNotFoundError(f"zig build shared did not produce {DLL}")

    env = os.environ.copy()
    env["PATH"] = str(DLL_DIR) + os.pathsep + env.get("PATH", "")

    with tempfile.TemporaryDirectory(prefix="reconl-java-ffm-") as classes:
        run([
            "javac", "--enable-preview", "--release", "21",
            "-d", classes, str(SOURCE),
        ], cwd=FRONTEND, env=env)
        run([
            "java", "--enable-preview", "--enable-native-access=ALL-UNNAMED",
            "-cp", classes, "ReconLUiSmoke", str(DLL),
        ], cwd=FRONTEND, env=env)

    print("Java 21 Panama FFM consumer check: ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
