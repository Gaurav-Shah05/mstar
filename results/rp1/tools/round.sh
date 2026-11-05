#!/usr/bin/env bash
# A/B rounds: round.sh <tag> <base_ck> <cand_ck> <config> "<order, e.g. C B B C C B>"
# Each arm holds the GPU lock for its whole life. Results under logs/<tag>/<i>_<arm>.
set -uo pipefail
TAG=$1; BASE=$2; CAND=$3; CFG=$4; ORDER=$5
L=/workspace/results/rp1-pi05-logs/$TAG
mkdir -p "$L"
i=0
for a in $ORDER; do
  i=$((i+1))
  ck=$BASE; [ "$a" = C ] && ck=$CAND
  out=$L/${i}_$a
  echo "=== arm $i $a $ck $(date -u +%FT%TZ)"
  flock /workspace/gpu.lock /workspace/results/rp1-pi05-tools/arm.sh "$ck" "$CFG" "$out" ${NREQ:-200} ${CS:-1,4,8} 2>&1 | grep -E "RESULT|FAILED"
done
echo "=== done $(date -u +%FT%TZ)"
