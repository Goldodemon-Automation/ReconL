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
# One row per probe invocation: probe|arguments|minimum checks|required lines|
# alternate short path|D3D11 required.
#
# `minimum checks` is the smallest count a complete run reports (`-` for probes
# pinned by required lines instead). It is a floor: optional passing branches can
# add checks, but missing checks below the floor need an exact alternate path in
# the next column. For offload, the return trip adds two checks (35 vs 33). When
# the first hardware frame is within target, it can finish at 18; that exact count
# and the probe's no-miss explanation is the only accepted short path.
#
# `required lines` are literal substrings, `;`-separated, that the output must
# contain. They are what makes a red row diagnosable - the count alone would not
# say which property moved.
#
# `alternate short path` entries are `count:reason`, `;`-separated. A row below
# its floor passes only at that exact branch count and when the probe prints its
# reason; all other short counts remain red. `D3D11 required` rows are skipped
# when gpu_probe reports no usable device; their row remains in the 24-row total.
fg_state="(the documented order holds in every cell)"
fg_state="$fg_state;refusals that wrote into the host's buffer: 0"
fg_state="$fg_state;cells that left the device somewhere unusable: 0"
fg_state="$fg_state;cells that could not be set up: 0 of 90"
fg_state="$fg_state;descriptors that crashed or wedged the device: 0"

plan=$(cat <<PLAN
fghostile|--backend=soft-cpu|32|||no
fghostile|--backend=d3d11|32|||yes
framegen|--backend=soft-cpu|21|||no
framegen|--backend=d3d11|21|||yes
framestate||43|||yes
gpu_probe||-|OK;usable=1||no
hostile||11|||yes
hostile2||13|||yes
hostile3||11|||yes
hostile4||-|a full frame afterwards: 0 (recovered);present(too small) -1||yes
hostile5||16|||yes
ladder||6|||yes
narrowpitch|--backend=soft-cpu|11|||no
narrowpitch|--backend=d3d11|11|||yes
narrowpitch|--backend=null|11|||no
offload||33||18:no miss here for the trigger to act on|yes
onewriter||8|||yes
overtarget||-|presented 12;failures 0||yes
shadowconfig||40|||yes
viewport|--backend=soft-cpu --size=64 --vp=16|-|lit pixels 256 of 4096;lit pixels outside the requested 16x16 rect: 0||no
viewport|--backend=d3d11 --size=64 --vp=16|-|lit pixels 256 of 4096;lit pixels outside the requested 16x16 rect: 0||yes
fgstate|--backend=soft-cpu|-|$fg_state||no
fgstate|--backend=d3d11|-|$fg_state||yes
fgstate|--backend=null|-|$fg_state||no
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
while IFS='|' read -r probe _args _want _need _alternate _requires_gpu; do
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
# backend table rather than from an assumption about the OS. The plan marks any
# row that depends on that device, including probes that mix hardware and CPU
# arms, so missing hardware is a skip rather than a silent partial run.
have_gpu=no
if (cd "$build" && "./bin.$$/gpu_probe$exe" 2>&1 | grep -qE 'd3d11 +usable=1'); then
    have_gpu=yes
fi

rows=0
bad=0
skip=0
summary=""
failures=""
while IFS='|' read -r probe args want need alternate requires_gpu; do
    [ -n "$probe" ] || continue
    if [ -n "$filter" ]; then
        case " $filter " in *" $probe "*) ;; *) continue ;; esac
    fi
    rows=$((rows + 1))

    tag="$(printf '%s %s' "$probe" "$args" | tr -d ' ' | tr '=' '_')"
    out="$build/$tag.out"
    : >"$out"

    if [ "$requires_gpu" = yes ] && [ "$have_gpu" = no ]; then
        skip=$((skip + 1))
        printf -v row '%-10s %-34s %6s %6s   %s\n' "$probe" "$args" "-" "-" \
            "SKIP (no usable d3d11 device on this host)"
        summary="$summary$row"
        continue
    fi

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
    if [ "$probe" = gpu_probe ] && [ "$have_gpu" = no ]; then
        need="no usable D3D11 device: nothing further to check"
    fi
    setup_issue="$(grep -m1 -iE '\(unavailable\)|device not created|device unavailable|capped rig unavailable|ring unreadable|d3d11 unavailable|could not be piloted' "$out" || true)"
    short=""
    if [ "$want" != "-" ]; then
        case "$got" in
            '' | *[!0-9]*)
                why="${why:+$why; }no check count in the verdict (see $out)"
                ;;
            *)
                if [ "$got" -lt "$want" ]; then
                    # A legitimate branch may have a lower minimum, but only at
                    # its declared count and with its own identifying output.
                    while IFS= read -r path; do
                        [ -n "$path" ] || continue
                        path_min="${path%%:*}"
                        path_reason="${path#*:}"
                        case "$path_min" in '' | *[!0-9]*) continue ;; esac
                        [ -n "$path_reason" ] || continue
                        if [ "$got" = "$path_min" ] && grep -qF -- "$path_reason" "$out"; then
                            short="${got} of ${want}, ${path_reason}"
                            break
                        fi
                    done <<ALTERNATES
$(echo "$alternate" | tr ';' '\n')
ALTERNATES
                    [ -n "$short" ] ||
                        why="${why:+$why; }checks $got, below the minimum $want (see $out)"
                fi
                ;;
        esac
    fi
    if [ -n "$setup_issue" ]; then
        why="${why:+$why; }required setup unavailable: $setup_issue"
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
    printf -v row '%-10s %-34s %6s %6s   %s %s\n' "$probe" "$args" "$got" "$findings" "$verdict" "${why:+($why)}${short:+ (short: $short)}"
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
