#!/usr/bin/env bash
# usage: bs_probe.sh <name> <checkout> <inductor cache>: per-step batch sizes at c=4 and c=8, one fresh server. Holds the lock throughout.
NAME=$1; CK=$2; IND=$3; source /workspace/venv/bin/activate; OUT=/workspace/results/rp1-pi05-logs/bs/$NAME; mkdir -p $OUT; cd $CK
export MSTAR_SHM_ARENA=1 PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$CK SERVER_STATE_DIR=/tmp/servers_rp1bs TORCHINDUCTOR_CACHE_DIR=$IND; unset TRITON_CACHE_DIR
SS=$CK/.claude/skills/benchmarking/scripts/server.sh
$SS start bs 8354 --timeout 3600 --log $OUT/server.log -- env RP1_BS_DIR=$OUT python -m mstar.cli.main serve pi05 --gpus 0 --port 8354 --config $CK/configs/pi05_droid.yaml || { echo "$NAME START FAILED"; $SS stop bs; exit 1; }
grep -c "these run eagerly" $OUT/server.log | sed "s/^/$NAME dropped /"
for c in 4 8; do python -m benchmark.runner --url http://127.0.0.1:8354 --model pi05 --request-type vision_language_action --profiling-type closed_loop --max-concurrency $c --num-requests 192 --num-warmup 1 --local-cache /workspace/results/rp1-pi05-tools/droid-cache --output-dir $OUT/c$c > $OUT/b$c.log 2>&1; echo "mark c$c $(python -c 'import time;print(time.monotonic())')" >> $OUT/marks.txt; sleep 2; python - $OUT/c$c/results.json $NAME $c <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print("RESULT",sys.argv[2],"c",sys.argv[3],"p50 %.2f p99 %.2f thr %.2f fail %d"%(d['jct_median_ms'],d['jct_p99_ms'],d['request_throughput'],d['failed']))
PY
done
$SS stop bs >/dev/null 2>&1
