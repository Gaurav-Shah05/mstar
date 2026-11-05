#!/usr/bin/env bash
# Paired A/B rounds: both servers up side by side per round (fresh servers, fresh
# inductor caches), gate cases and an untimed warm pass on each, then c=1 as
# alternating single requests (paired_c1.py) and c=4 / c=8 as back-to-back runner
# runs alternating which arm goes first. A = base, B = candidate.
# usage: pair_round.sh <tag> <ckA> <ckB> <config> [rounds=3]
# Holds /workspace/gpu.lock for each whole round.
set -uo pipefail
TAG=$1; CKA=$2; CKB=$3; CFG=$4; ROUNDS=${5:-3}
R=/workspace/results/rp1-pi05-tools
L=/workspace/results/rp1-pi05-logs/$TAG
mkdir -p "$L"
source /workspace/venv/bin/activate
export MSTAR_SHM_ARENA=1 SERVER_STATE_DIR=/tmp/servers_rp1pair
unset TRITON_CACHE_DIR
SS=$CKA/.claude/skills/benchmarking/scripts/server.sh
PA=8321; PB=8322

one_round() {
  local k=$1
  local out=$L/r$k
  mkdir -p "$out/A" "$out/B"
  git -C "$CKA" rev-parse --short HEAD > "$out/A/commit.txt"; git -C "$CKB" rev-parse --short HEAD > "$out/B/commit.txt"
  git -C "$CKA" diff --stat > "$out/A/diffstat.txt"; git -C "$CKB" diff --stat > "$out/B/diffstat.txt"
  uptime > "$out/load.txt"
  local ia ib; ia=$(mktemp -d /tmp/rp1ind.XXXX); ib=$(mktemp -d /tmp/rp1ind.XXXX)
  (cd "$CKA" && PYTHONPATH=$R/pyshim:$CKA TORCHINDUCTOR_CACHE_DIR=$ia $SS start pairA $PA --timeout 3600 --log "$out/A/server.log" -- \
      env ${ENV_A:-} python -m mstar.cli.main serve pi05 --gpus 0 --port $PA --config "$CKA/configs/$CFG") &
  local pa=$!
  (cd "$CKB" && PYTHONPATH=$R/pyshim:$CKB TORCHINDUCTOR_CACHE_DIR=$ib $SS start pairB $PB --timeout 3600 --log "$out/B/server.log" -- \
      env ${ENV_B:-} python -m mstar.cli.main serve pi05 --gpus 0 --port $PB --config "$CKB/configs/$CFG") &
  local pb=$!
  wait $pa; local ra=$?; wait $pb; local rb=$?
  if [ $ra -ne 0 ] || [ $rb -ne 0 ]; then echo "START FAILED A=$ra B=$rb"; $SS stop pairA; $SS stop pairB; rm -rf "$ia" "$ib"; return 1; fi
  echo "started $(date -u +%T)" >> "$out/load.txt"
  cd "$CKA"
  export PYTHONPATH=$R/pyshim:$CKA
  python $R/gate_client.py $PA "$out/A/gate.npz" > "$out/A/gate.log" 2>&1
  python $R/gate_client.py $PB "$out/B/gate.npz" > "$out/B/gate.log" 2>&1
  for side in A B; do
    port=$PA; [ $side = B ] && port=$PB
    for c in 1 4 8; do
      python -m benchmark.runner --url http://127.0.0.1:$port --model pi05 --request-type vision_language_action \
        --profiling-type closed_loop --max-concurrency $c --num-requests $((c*2)) --num-warmup 1 \
        --local-cache $R/droid-cache > "$out/$side/warm_c$c.log" 2>&1
    done
  done
  python $R/paired_c1.py $PA $PB ${NREQ:-200} "$out" 2>&1 | grep -E "RESULT|FAILED"
  for c in 4 8; do
    if [ $((k % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
    for side in $order; do
      port=$PA; [ $side = B ] && port=$PB
      python -m benchmark.runner --url http://127.0.0.1:$port --model pi05 --request-type vision_language_action \
        --profiling-type closed_loop --max-concurrency $c --num-requests 96 --num-warmup 1 \
        --local-cache $R/droid-cache --output-dir "$out/$side/c$c" > "$out/$side/bench_c$c.log" 2>&1
      python - "$out/$side/c$c/results.json" $side $c <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"RESULT {sys.argv[2]} c{sys.argv[3]} p50={d['jct_median_ms']:.2f} p99={d['jct_p99_ms']:.2f} mean={d['jct_mean_ms']:.2f} thr={d['request_throughput']:.3f} ok={d['completed']} fail={d['failed']}")
EOF
    done
  done
  uptime >> "$out/load.txt"
  $SS stop pairA >/dev/null 2>&1; $SS stop pairB >/dev/null 2>&1
  rm -rf "$ia" "$ib"
}

for k in $(seq 1 $ROUNDS); do
  echo "=== round $k $(date -u +%FT%TZ)"
  flock /workspace/gpu.lock bash -c "$(declare -f one_round); L=$L R=$R SS=$SS PA=$PA PB=$PB CKA=$CKA CKB=$CKB CFG=$CFG NREQ=${NREQ:-200} ENV_A='${ENV_A:-}' ENV_B='${ENV_B:-}'; one_round $k"
done
echo "=== done $(date -u +%FT%TZ)"
