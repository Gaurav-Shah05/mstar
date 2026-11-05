#!/usr/bin/env bash
# One measurement arm: fresh server from CHECKOUT, untimed warm pass, then timed
# closed-loop runs at each concurrency. Must be invoked under flock /workspace/gpu.lock.
# usage: arm.sh <checkout> <config.yaml> <outdir> <nreq_per_c> <c1,c2,...> [extra serve args]
set -uo pipefail
R_IND=$([ "$(basename $1)" = wt-rp1-pi05-b4 ] && echo /workspace/results/rp1-pi05-tools/ind-head || echo /workspace/results/rp1-pi05-tools/ind-h28b)
CK=$1; CFG=$2; OUT=$3; NREQ=$4; CS=$5; shift 5
PORT=${PORT:-8311}
mkdir -p "$OUT"
source /workspace/venv/bin/activate
export MSTAR_SHM_ARENA=1 PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$CK
export TORCHINDUCTOR_CACHE_DIR=$R_IND
unset TRITON_CACHE_DIR
SS=$CK/.claude/skills/benchmarking/scripts/server.sh
export SERVER_STATE_DIR=/tmp/servers_rp1
cd "$CK"
( cd "$CK" && python -c 'import mstar; print("IMPORT", mstar.__file__)' ) > "$OUT/import.txt"
git -C "$CK" rev-parse --short HEAD > "$OUT/commit.txt"
uptime > "$OUT/load.txt"
t0=$(date +%s)
$SS start rp1 $PORT --timeout 3600 --log "$OUT/server.log" -- \
  python -m mstar.cli.main serve pi05 --gpus 0 --port $PORT --config "$CK/configs/$CFG" "$@" \
  || { echo "START FAILED"; $SS stop rp1; exit 1; }
echo "startup_s $(( $(date +%s) - t0 ))" >> "$OUT/load.txt"
cleanup() { $SS stop rp1 >/dev/null 2>&1; }
trap cleanup EXIT
python /workspace/results/rp1-pi05-tools/gate_client.py $PORT "$OUT/gate.npz" > "$OUT/gate.log" 2>&1
# untimed warm pass: exercise every concurrency once
for c in ${CS//,/ }; do
  python -m benchmark.runner --url http://127.0.0.1:$PORT --model pi05 --request-type vision_language_action \
    --profiling-type closed_loop --max-concurrency $c --num-requests $((c*2)) --num-warmup 1 \
    --local-cache /workspace/results/rp1-pi05-tools/droid-cache > "$OUT/warm_c$c.log" 2>&1
done
for c in ${CS//,/ }; do
  n=$NREQ; [ "$c" -gt 1 ] && n=96
  python -m benchmark.runner --url http://127.0.0.1:$PORT --model pi05 --request-type vision_language_action \
    --profiling-type closed_loop --max-concurrency $c --num-requests $n --num-warmup 1 \
    --local-cache /workspace/results/rp1-pi05-tools/droid-cache --output-dir "$OUT/c$c" > "$OUT/bench_c$c.log" 2>&1
  python - "$OUT/c$c/results.json" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
print("RESULT", sys.argv[1], "p50=%.2f p99=%.2f mean=%.2f thr=%.3f ok=%s fail=%s" % (d.get("jct_median_ms",0), d.get("jct_p99_ms",0), d.get("jct_mean_ms",0), d.get("request_throughput",0), d.get("completed"), d.get("failed")))
EOF
done
