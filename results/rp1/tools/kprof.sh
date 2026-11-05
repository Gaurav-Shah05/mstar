#!/usr/bin/env bash
# Kernel profile under load: kprof.sh <checkout> <config> <outdir> <c>
# Run under flock /workspace/gpu.lock.
set -uo pipefail
CK=$1; CFG=$2; OUT=$3; C=$4
PORT=${PORT:-8313}
mkdir -p "$OUT"; rm -f "$OUT/go"
source /workspace/venv/bin/activate
export MSTAR_SHM_ARENA=1 PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$CK RP1_PROF_DIR=$OUT RP1_PROF_SECONDS=${SECS:-3}
export TORCHINDUCTOR_CACHE_DIR=${IND:-$(mktemp -d /tmp/rp1ind.XXXX)}
unset TRITON_CACHE_DIR
SS=$CK/.claude/skills/benchmarking/scripts/server.sh
export SERVER_STATE_DIR=/tmp/servers_rp1k
cd "$CK"
$SS start rp1k $PORT --timeout 3600 --log "$OUT/server.log" -- \
  python -m mstar.cli.main serve pi05 --gpus 0 --port $PORT --config "$CK/configs/$CFG" || { $SS stop rp1k; exit 1; }
trap '$SS stop rp1k >/dev/null 2>&1' EXIT
run() { python -m benchmark.runner --url http://127.0.0.1:$PORT --model pi05 --request-type vision_language_action \
  --profiling-type closed_loop --max-concurrency $C --num-requests $1 --num-warmup 2 \
  --local-cache /workspace/results/rp1-pi05-tools/droid-cache --output-dir "$OUT/res$1"; }
run 20 > "$OUT/warm.log" 2>&1
( for i in 1 2 3 4; do run 96; done ) > "$OUT/bench.log" 2>&1 &
BP=$!
sleep 25; touch "$OUT/go"
wait $BP
sleep 5
ls "$OUT"
