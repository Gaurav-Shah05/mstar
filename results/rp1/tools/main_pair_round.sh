#!/usr/bin/env bash
# pair_round.sh for main: both arms from venv-main, MSTAR_SHM_ARENA=1, knobs step_thread_core: siblings, FE_FLAG for --rust-frontend.
# usage: main_pair_round.sh <tag> <ckA> <ckB> <config name> [rounds=3]
set -uo pipefail
TAG=$1; CKA=$2; CKB=$3; CFG=$4; ROUNDS=${5:-3}
R=/workspace/results/rp1-pi05-tools; L=/workspace/results/rp1-pi05-logs/$TAG; mkdir -p "$L"
export PATH=/workspace/venv-main/bin:$PATH VIRTUAL_ENV=/workspace/venv-main MSTAR_SHM_ARENA=1 SERVER_STATE_DIR=/tmp/servers_rp1mp
export MSTAR_SERVER_BIN=/workspace/wt-rp1-main/rust/server/target/release/mstar-server
unset TRITON_CACHE_DIR
SS=$CKA/.claude/skills/benchmarking/scripts/server.sh; PA=8361; PB=8362; FE=${FE_FLAG:-}
cfg() { python - "$1/configs/$CFG" "$2" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); k = c.setdefault("knobs", {}); k["step_thread_core"] = "siblings"
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
PY
}
one_round() {
  local k=$1 out=$L/r$k; mkdir -p "$out/A" "$out/B"
  git -C "$CKA" rev-parse --short HEAD > "$out/A/commit.txt"; git -C "$CKB" rev-parse --short HEAD > "$out/B/commit.txt"
  uptime > "$out/load.txt"
  cfg "$CKA" "$out/A/$CFG"; cfg "$CKB" "$out/B/$CFG"
  local ia ib; ia=$(mktemp -d /tmp/rp1ind.XXXX); ib=$(mktemp -d /tmp/rp1ind.XXXX)
  (cd "$CKA" && PYTHONPATH=$R/main_pyshim:$CKA TORCHINDUCTOR_CACHE_DIR=$ia $SS start pairA $PA --timeout 3600 --log "$out/A/server.log" -- python -m mstar.cli.main serve pi05 --gpus 0 --port $PA --config "$out/A/$CFG" $FE) &
  local pa=$!
  (cd "$CKB" && PYTHONPATH=$R/main_pyshim:$CKB TORCHINDUCTOR_CACHE_DIR=$ib $SS start pairB $PB --timeout 3600 --log "$out/B/server.log" -- python -m mstar.cli.main serve pi05 --gpus 0 --port $PB --config "$out/B/$CFG" $FE) &
  local pb=$!
  wait $pa; local ra=$?; wait $pb; local rb=$?
  if [ $ra -ne 0 ] || [ $rb -ne 0 ]; then echo "START FAILED A=$ra B=$rb"; $SS stop pairA; $SS stop pairB; rm -rf "$ia" "$ib"; return 1; fi
  echo "started $(date -u +%T)" >> "$out/load.txt"
  cd "$CKA"; export PYTHONPATH=$CKA
  python $R/gate_client.py $PA "$out/A/gate.npz" > "$out/A/gate.log" 2>&1
  python $R/gate_client.py $PB "$out/B/gate.npz" > "$out/B/gate.log" 2>&1
  for side in A B; do
    port=$PA; [ $side = B ] && port=$PB
    for c in 1 4 8; do
      python -m benchmark.runner --url http://127.0.0.1:$port --model pi05 --request-type vision_language_action --profiling-type closed_loop --max-concurrency $c --num-requests $((c*2)) --num-warmup 1 --local-cache $R/droid-cache > "$out/$side/warm_c$c.log" 2>&1
    done
  done
  python $R/paired_c1.py $PA $PB ${NREQ:-200} "$out" 2>&1 | grep -E "RESULT|FAILED"
  for c in 4 8; do
    if [ $((k % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
    for side in $order; do
      port=$PA; [ $side = B ] && port=$PB
      python -m benchmark.runner --url http://127.0.0.1:$port --model pi05 --request-type vision_language_action --profiling-type closed_loop --max-concurrency $c --num-requests 96 --num-warmup 1 --local-cache $R/droid-cache --output-dir "$out/$side/c$c" > "$out/$side/bench_c$c.log" 2>&1
      python - "$out/$side/c$c/results.json" $side $c <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"RESULT {sys.argv[2]} c{sys.argv[3]} p50={d['jct_median_ms']:.2f} p99={d['jct_p99_ms']:.2f} mean={d['jct_mean_ms']:.2f} thr={d['request_throughput']:.3f} ok={d['completed']} fail={d['failed']}")
PY
    done
  done
  uptime >> "$out/load.txt"
  $SS stop pairA >/dev/null 2>&1; $SS stop pairB >/dev/null 2>&1; rm -rf "$ia" "$ib"
}
for k in $(seq 1 $ROUNDS); do
  echo "=== round $k $(date -u +%FT%TZ)"
  flock /workspace/gpu.lock bash -c "$(declare -f one_round cfg); L=$L R=$R SS=$SS PA=$PA PB=$PB CKA=$CKA CKB=$CKB CFG=$CFG NREQ=${NREQ:-200} FE='$FE'; one_round $k"
done
echo "=== done $(date -u +%FT%TZ)"
