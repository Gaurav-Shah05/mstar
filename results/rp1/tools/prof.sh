#!/usr/bin/env bash
# Profiling server run: phase timing + per-request stats. usage: prof.sh <checkout> <config> <outdir> <nreq> <c>
set -uo pipefail
CK=$1; CFG=$2; OUT=$3; NREQ=$4; C=$5; shift 5
PORT=${PORT:-8312}
mkdir -p "$OUT"
source /workspace/venv/bin/activate
export MSTAR_SHM_ARENA=1 PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$CK MSTAR_PHASE_TIMING=${PHASE:-50} RP1_GRAPH_TIMING=$OUT
export TORCHINDUCTOR_CACHE_DIR=${IND:-$(mktemp -d /tmp/rp1ind.XXXX)}
unset TRITON_CACHE_DIR
SS=$CK/.claude/skills/benchmarking/scripts/server.sh
export SERVER_STATE_DIR=/tmp/servers_rp1p
cd "$CK"
$SS start rp1p $PORT --timeout 3600 --log "$OUT/server.log" -- \
  python -m mstar.cli.main serve pi05 --gpus 0 --port $PORT --config "$( [ "${CFG:0:1}" = / ] && echo "$CFG" || echo "$CK/configs/$CFG" )" --log-stats-file "$OUT/stats.txt" "$@" || { $SS stop rp1p; exit 1; }
trap '$SS stop rp1p >/dev/null 2>&1' EXIT
python -m benchmark.runner --url http://127.0.0.1:$PORT --model pi05 --request-type vision_language_action \
  --profiling-type closed_loop --max-concurrency $C --num-requests $NREQ --num-warmup 3 \
  --local-cache /workspace/results/rp1-pi05-tools/droid-cache --output-dir "$OUT/res" > "$OUT/bench.log" 2>&1
eval "${POSTHOOK:-true}"
