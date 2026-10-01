#!/usr/bin/env bash
# Throwaway: build every probe once and measure its check count over N runs.
# $1 = runs per row (default 30), $2 = probe filter (default all).
set -u
ROOT=$(cd "$(dirname "$0")" && pwd)
N=${1:-30}
ONLY=${2:-}
BIN=$(mktemp -d)

cc -O2 -I "$ROOT/include" -o "$BIN/gpu_probe.exe" "$ROOT/probes/src/gpu_probe.c" "$ROOT/target/release/reconl.dll" 2>/dev/null
cp -f "$ROOT/target/release/reconl.dll" "$BIN/" 2>/dev/null

# probe|arguments, one line each, exactly as run.sh's plan invokes them.
rows=$(cat <<ROWS
fghostile|--backend=soft-cpu
fghostile|--backend=d3d11
framegen|--backend=soft-cpu
framegen|--backend=d3d11
framestate|
gpu_probe|
hostile|
hostile2|
hostile3|
hostile4|
hostile5|
ladder|
narrowpitch|--backend=soft-cpu
narrowpitch|--backend=d3d11
narrowpitch|--backend=null
offload|
onewriter|
overtarget|
shadowconfig|
viewport|--backend=soft-cpu --size=64 --vp=16
viewport|--backend=d3d11 --size=64 --vp=16
fgstate|--backend=soft-cpu
fgstate|--backend=d3d11
fgstate|--backend=null
ROWS
)

printf '%-12s %-34s %-7s %s\n' probe arguments time min/mode/max "distinct counts"
while IFS='|' read -r probe args; do
    [ -n "$probe" ] || continue
    case " $ONLY " in *" $probe "*) ;; *) [ -n "$ONLY" ] && continue ;; esac
    if [ ! -x "$BIN/$probe.exe" ]; then
        cc -O2 -I "$ROOT/include" -o "$BIN/$probe.exe" "$ROOT/probes/src/$probe.c" "$ROOT/target/release/reconl.dll" 2>/dev/null || {
            printf '%-12s %s\n' "$probe" "DID NOT BUILD"
            continue
        }
    fi
    t0=$(date +%s%N)
    first=$(cd "$BIN" && ./"$probe.exe" $args 2>&1)
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    counts=""
    bails=0
    for i in $(seq 1 "$N"); do
        out=$(cd "$BIN" && ./"$probe.exe" $args 2>&1)
        c=$(echo "$out" | grep -oE '[0-9]+ checks' | head -1 | grep -oE '[0-9]+' || echo "-")
        counts="$counts $c"
        echo "$out" | grep -qE "\(d3d11 unavailable\)|no miss here|\(this host" && bails=$((bails + 1))
    done
    dist=$(echo "$counts" | tr ' ' '\n' | grep -v '^$' | sort -n | uniq -c | awk '{printf "%sx%s ", $1, $2}')
    mn=$(echo "$counts" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1)
    mx=$(echo "$counts" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1)
    printf '%-12s %-34s %-7s %s| host-branch/bail lines: %s/%s\n' "$probe" "$args" "${ms}ms" "$dist" "$bails" "$N"
done <<EOF
$rows
EOF

rm -rf "$BIN"
