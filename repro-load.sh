#!/usr/bin/env bash
# Throwaway: how often does the offload probe take its return-trip branch (35
# checks) under load, and does that branch ever carry failures?
set -u
N=${1:-40}
ROOT=$(cd "$(dirname "$0")" && pwd)

# CPU pressure: the ladder returns to hardware when the reference tier measures
# slower, and contention is what makes that happen.
for i in 1 2 3 4 5 6; do ( while :; do :; done ) & done
BURN=$(jobs -p)

B=$(mktemp -d)
cp "$ROOT/target/release/reconl.dll" "$B/"
cc -O2 -I "$ROOT/include" -o "$B/offload.exe" "$ROOT/probes/src/offload.c" "$ROOT/target/release/reconl.dll"

for i in $(seq 1 "$N"); do
    out=$(cd "$B" && ./offload.exe 2>&1)
    printf '%s | %s | %s\n' \
        "$(echo "$out" | grep -oE '^[0-9]+ checks, [0-9]+ failures')" \
        "$(echo "$out" | grep -oE 'came back|stayed' | head -1)" \
        "$(echo "$out" | grep -oE '\(d3d11 unavailable\)' | head -1)"
done | sort | uniq -c | sed 's/^/  /'

kill $BURN 2>/dev/null
rm -rf "$B"
