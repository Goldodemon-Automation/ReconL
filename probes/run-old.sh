#!/usr/bin/env bash
# The C probes, run the way a host runs the library.
#
# ReconL's behavioural claims are claims about the shipped artifact, so the
# evidence for them is a C program driving the exported functions of the built
# DLL/SO - not a Rust test reaching the same code in-process. This script builds
# the release library, builds every probe in `probes/src` against the exact
# binary that build produced, runs each one, and prints a single summary.
#
# The md5 of the library is printed with the summary and every probe's output is
# kept under `probes/.build/`, so a result is always traceable to the library
# that produced it. That is the failure mode this replaces: the probes used to
# live in a scratch directory next to a hand-copied DLL, and a stale copy there
# once made a pass report on a library it had not built.
#
# Usage:
#   probes/run.sh              # build the library, run every probe, summarise
#   probes/run.sh fghostile    # only the named probes (still builds the library)
#
# Exit: 0 when every row passes, 1 when a row fails, 2 when the build does.
# See probes/README.md for what each probe is for.

set -u

root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/probes/src"
build="$root/probes/.build"
cc="${CC:-cc}"
filter="$*"

# ----------------------------------------------------------------- the plan
# One row per probe invocation: probe|arguments|expected checks|required lines.
#
# `expected checks` is the count the probe's own verdict prints (`-` for the few
# probes that print a verdict but no count). Pinning it means a probe that stops
# running its checks cannot pass by printing nothing: a row is red unless the
# count matches *and* every required line is present.
#
# `required lines` are literal substrings, `;`-separated, that the output must
# contain. They are what makes a red row diagnosable - the count alone would not
# say which property moved.
fg_state="(the documented order holds in every cell)"
fg_state="$fg_state;refusals that wrote into the host's buffer: 0"
fg_state="$fg_state;cells that left the device somewhere unusable: 0"
fg_state="$fg_state;cells that could not be set up: 0 of 90"
fg_state="$fg_state;descriptors that crashed or wedged the device: 0"

plan=$(cat <<PLAN
fghostile|--backend=soft-cpu|32|
fghostile|--backend=d3d11|32|
framegen|--backend=soft-cpu|21|
framegen|--backend=d3d11|21|
framestate||43|
gpu_probe||-|OK;usable=1
hostile||11|
hostile2||13|
hostile3||11|
hostile4||-|a full frame afterwards: 0 (recovered);present(too small) -1
hostile5||16|
ladder||6|
narrowpitch|--backend=soft-cpu|11|
narrowpitch|--backend=d3d11|11|
narrowpitch|--backend=null|11|
offload||33|
onewriter||8|
overtarget||-|presented 12;failures 0
shadowconfig||40|
viewport|--backend=soft-cpu --size=64 --vp=16|-|lit pixels 256 of 4096;lit pixels outside the requested 16x16 rect: 0
viewport|--backend=d3d11 --size=64 --vp=16|-|lit pixels 256 of 4096;lit pixels outside the requested 16x16 rect: 0
fgstate|--backend=soft-cpu|-|$fg_state
fgstate|--backend=d3d11|-|$fg_state
fgstate|--backend=null|-|$fg_state
PLAN
)

# ------------------------------------------------------------------- hashing
hash() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | cut -d' ' -f1
    else
        md5 -q "$1"
    fi
}

# ------------------------------------------------------------ build the library
echo "== building the shipped library =="
(cd "$root" && cargo build --release -p reconl-ffi) || {
    echo "run.sh: the library did not build"
    exit 2
}

case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) libname=reconl.dll exe=.exe ;;
    Darwin) libname=libreconl.dylib exe= ;;
    *) libname=libreconl.so exe= ;;
esac

lib="$root/target/release/$libname"
if [ ! -f "$lib" ]; then
    echo "run.sh: no $libname in $root/target/release"
    exit 2
fi

mkdir -p "$build"
cp -f "$lib" "$build/$libname"
lib_hash="$(hash "$lib")"
copy_hash="$(hash "$build/$libname")"
if [ "$lib_hash" != "$copy_hash" ]; then
    echo "run.sh: the copied $libname does not match the built one"
    exit 2
fi

# The probe binaries go in a run-scoped directory, and are run from `$build` so
# the DLL/SO beside them resolves. Linking over a name a previous run's process
# still holds is a real failure on Windows (the image stays locked until the
# process is gone), and it would make this gate intermittently red for a reason
# that has nothing to do with the library.
bin="$build/bin.$$"
rm -rf "$bin"
mkdir -p "$bin"
trap 'rm -rf "$bin"' EXIT

# ------------------------------------------- build the probes against that binary
# Directly against the DLL/SO: the probes link the artifact, not a Rust crate,
# and no import library or .def file has to be kept in step with it.
probes=""
while IFS='|' read -r probe _args _want _need; do
    [ -n "$probe" ] || continue
    case " $probes " in *" $probe "*) continue ;; esac
    probes="$probes $probe"
done <<EOF
$plan
EOF

# A source with no plan row would be compiled and never run, which is how a
# probe stops being evidence without anyone deciding to stop running it.
unplanned=""
for f in "$src"/*.c; do
    p="$(basename "$f" .c)"
    case " $probes " in *" $p "*) ;; *) unplanned="$unplanned $p" ;; esac
done
if [ -n "$unplanned" ]; then
    echo "run.sh: no plan row for:$unplanned"
    exit 2
fi

for probe in $probes; do
    if [ ! -f "$src/$probe.c" ]; then
        echo "run.sh: no source for $probe"
        exit 2
    fi
    if ! "$cc" -O2 -I "$root/include" -o "$bin/$probe$exe" "$src/$probe.c" "$build/$libname" \
        2>"$build/$probe.build.log"; then
        sleep 1
        if ! "$cc" -O2 -I "$root/include" -o "$bin/$probe$exe" "$src/$probe.c" "$build/$libname" \
            2>"$build/$probe.build.log"; then
            echo "run.sh: $probe did not compile"
            sed 's/^/  /' "$build/$probe.build.log"
            exit 2
        fi
    fi
done

# --------------------------------------------------------------- run the plan
# Whether this host has a usable D3D11 device, from the probe that reports the
# backend table rather than from an assumption about the OS. Rows that name
# `--backend=d3d11` are skipped when it does not, the way the in-repo tests skip
# their d3d11 legs, instead of being reported as failures the machine cannot fix.
have_gpu=no
if (cd "$build" && "./bin.$$/gpu_probe$exe" 2>&1 | grep -qE 'd3d11 +usable=1'); then
    have_gpu=yes
fi

rows=0
bad=0
skip=0
summary=""
failures=""
while IFS='|' read -r probe args want need; do
    [ -n "$probe" ] || continue
    if [ -n "$filter" ]; then
        case " $filter " in *" $probe "*) ;; *) continue ;; esac
    fi
    rows=$((rows + 1))

    tag="$(printf '%s %s' "$probe" "$args" | tr -d ' ' | tr '=' '_')"
    out="$build/$tag.out"
    : >"$out"

    case "$probe $args" in
        *d3d11* | overtarget\ *)
            if [ "$have_gpu" = no ]; then
                skip=$((skip + 1))
                printf -v row '%-10s %-34s %6s %6s   %s\n' "$probe" "$args" "-" "-" \
                    "SKIP (no usable d3d11 device on this host)"
                summary="$summary$row"
                continue
            fi
            ;;
    esac

    (cd "$build" && "./bin.$$/$probe$exe" $args) >>"$out" 2>&1
    rc=$?

    got="-"
    n="$(grep -oE '[0-9]+ checks' "$out" | head -1 | grep -oE '[0-9]+' || true)"
    [ -n "$n" ] && got="$n"

    findings="$(grep -cE 'FAIL|FIND' "$out" || true)"
    why=""

    if [ "$rc" -ne 0 ]; then
        why="exit $rc"
    fi
    if [ "$want" != "-" ] && [ "$got" != "$want" ]; then
        why="${why:+$why; }checks $got, expected $want"
    fi
    if [ "$findings" -ne 0 ]; then
        why="${why:+$why; }$findings failure marker(s)"
    fi
    if [ -n "$need" ]; then
        missing=""
        # A `while read` fed by a here-document, not a pipeline: the loop runs in
        # this shell, so `missing` survives it. Required lines contain spaces.
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            grep -qF -- "$line" "$out" || missing="${missing:+$missing, }$line"
        done <<NEED
$(echo "$need" | tr ';' '\n')
NEED
        [ -n "$missing" ] && why="${why:+$why; }missing: $missing"
    fi

    verdict=PASS
    if [ -n "$why" ]; then
        bad=$((bad + 1))
        verdict=FAIL
        printf -v line '  %-10s %-34s %s\n' "$probe" "$args" "$why"
        failures="$failures$line"
    fi
    printf -v row '%-10s %-34s %6s %6s   %s %s\n' "$probe" "$args" "$got" "$findings" "$verdict" "${why:+($why)}"
    summary="$summary$row"
done <<EOF
$plan
EOF

# -------------------------------------------------------------------- summary
echo
echo "reconl C probes - $libname md5 $lib_hash"
echo "  linked against: $build/$libname (md5 confirmed by copy of $lib)"
echo "  host: $(uname -s) $(uname -m), d3d11 $( [ "$have_gpu" = yes ] && echo "usable" || echo "not usable")"
echo
printf '%-10s %-34s %6s %6s   %s\n' probe arguments checks finds verdict
[ -n "$summary" ] && printf '%s\n' "$summary"
echo
echo "$rows rows: $((rows - bad - skip)) pass, $bad fail, $skip skip   (per-row output in probes/.build/*.out)"

if [ "$bad" -ne 0 ]; then
    echo
    echo "failing rows:"
    printf '%s\n' "$failures"
    exit 1
fi
exit 0
