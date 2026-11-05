#!/usr/bin/env bash
# fresh single server of the H26 tree: gate, then a c=1 run. Holds the GPU lock throughout.
source /workspace/venv/bin/activate; CK=/workspace/wt-rp1-pi05-h27; OUT=/workspace/results/rp1-pi05-logs/h27_probe; mkdir -p $OUT; cd $CK
export MSTAR_SHM_ARENA=1 PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$CK SERVER_STATE_DIR=/tmp/servers_rp1h27; unset TRITON_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/rp1ind.XXXX); SS=$CK/.claude/skills/benchmarking/scripts/server.sh
$SS start h27 8350 --timeout 3600 --log $OUT/server.log -- python -m mstar.cli.main serve pi05 --gpus 0 --port 8350 --config $CK/configs/pi05_droid.yaml || { echo START FAILED; $SS stop h27; rm -rf $TORCHINDUCTOR_CACHE_DIR; exit 1; }
python /workspace/results/rp1-pi05-tools/gate_client.py 8350 $OUT/gate.npz > $OUT/gate.log 2>&1; python /workspace/results/rp1-pi05-tools/gate_check.py $OUT/gate.npz 2>&1 | grep -E "GATE|Error"
for c in 1 4 8; do python -m benchmark.runner --url http://127.0.0.1:8350 --model pi05 --request-type vision_language_action --profiling-type closed_loop --max-concurrency $c --num-requests $((c*32)) --num-warmup 3 --local-cache /workspace/results/rp1-pi05-tools/droid-cache --output-dir $OUT/c$c > $OUT/bench_c$c.log 2>&1; python - $OUT/c$c/results.json $c <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print("RESULT c",sys.argv[2],"p50 %.2f p99 %.2f thr %.2f ok %d fail %d"%(d['jct_median_ms'],d['jct_p99_ms'],d['request_throughput'],d['completed'],d['failed']))
PY
done
$SS stop h27 >/dev/null 2>&1; rm -rf $TORCHINDUCTOR_CACHE_DIR; echo finished
