#!/usr/bin/env bash
# Build and run the Java conformance suite.
#
# Deliberately javac plus the JUnit console-standalone jar rather than Gradle:
# the suite is one binding file, three test files and one dependency, and a
# build tool here would be a second thing to keep in step with the header.
#
# The one thing that is not optional is PATH. SymbolLookup opens reconl.dll by
# name, the dynamic loader then has to find reconl.dll's own dependency out of
# the same directory, and that directory is not a system directory - so the
# library is reached by putting its directory on PATH, not by passing an absolute
# path. `--enable-preview` is because FFM was still a preview API in 21;
# `--enable-native-access` is because restricting it would be a lie about what
# this needs.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="$here/out"
jar="$here/lib/junit-platform-console-standalone.jar"

if [[ ! -f "$jar" ]]; then
    echo "missing $jar" >&2
    echo "fetch it with:" >&2
    echo "  curl -sSLo '$jar' \\" >&2
    echo "    https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.11.4/junit-platform-console-standalone-1.11.4.jar" >&2
    exit 2
fi

# The library the tests link against. `reconl.dll` on PATH wins if there is one,
# so a plain run uses whatever the build last produced.
if [[ -d "$root/frontend/zig-out/bin" && -f "$root/frontend/zig-out/bin/reconl.dll" ]]; then
    export PATH="$root/frontend/zig-out/bin:$PATH"
elif [[ -f "$root/target/debug/reconl.dll" ]]; then
    export PATH="$root/target/debug:$PATH"
fi

rm -rf "$out"
mkdir -p "$out"

javac --enable-preview --release 21 -cp "$jar" -d "$out" "$here"/*.java

exec java --enable-preview --enable-native-access=ALL-UNNAMED \
    -jar "$jar" execute \
    --class-path "$out" \
    --select-class=ReconLAbiTest \
    --select-class=ReconLRefusalTest \
    --select-class=ReconLConformanceTest \
    --details=summary \
    --disable-banner \
    "$@"
