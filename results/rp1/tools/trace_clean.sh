#!/usr/bin/env bash
# fresh server, hook tracer only (no phase timing, no per-request stats). usage: trace_clean.sh <checkout> <outdir> <nreq>
CK=$1; OUT=$2; N=$3; PORT=${PORT:-8340}; R=/workspace/results/rp1-pi05-tools
mkdir -p "$OUT"; source /workspace/venv/bin/activate
export MSTAR_SHM_ARENA=1 PYTHONPATH=$R/pyshim:$CK RP1_TRACE_DIR=$OUT RP1_GRAPH_TIMING=$OUT TORCHINDUCTOR_CACHE_DIR=${IND:-$R/ind-head} SERVER_STATE_DIR=/tmp/servers_rp1t
unset TRITON_CACHE_DIR; SS=$CK/.claude/skills/benchmarking/scripts/server.sh; cd "$CK"
$SS start trc $PORT --timeout 3600 --log "$OUT/server.log" -- python -m mstar.cli.main serve pi05 --gpus 0 --port $PORT --config "$CK/configs/pi05_droid.yaml" || { $SS stop trc; exit 1; }
trap '$SS stop trc >/dev/null 2>&1' EXIT
python -m benchmark.runner --url http://127.0.0.1:$PORT --model pi05 --request-type vision_language_action --profiling-type closed_loop --max-concurrency 1 --num-requests $N --num-warmup 3 --local-cache $R/droid-cache --output-dir "$OUT/res" > "$OUT/bench.log" 2>&1
sleep 3
