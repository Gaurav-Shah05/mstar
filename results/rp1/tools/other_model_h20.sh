#!/usr/bin/env bash
# Qwen3.5-0.8B with the frequency request off vs on, same tree, alternating fresh servers. Under the GPU lock per arm.
R=/workspace/results/rp1-pi05-tools; L=/workspace/results/rp1-pi05-logs/other_h20; CK=/workspace/wt-rp1-pi05-best
mkdir -p $L; source /workspace/venv/bin/activate; export MSTAR_SHM_ARENA=1 PYTHONPATH=$R/pyshim:$CK SERVER_STATE_DIR=/tmp/servers_rp1o; unset TRITON_CACHE_DIR
SS=$CK/.claude/skills/benchmarking/scripts/server.sh; P=8331
arm() { # name value
  flock /workspace/gpu.lock bash -c "cd $CK; export TORCHINDUCTOR_CACHE_DIR=\$(mktemp -d /tmp/rp1ind.XXXX) MSTAR_FULL_CPU_FREQUENCY=$2; $SS start oth $P --timeout 3600 --log $L/$1.server.log -- python -m mstar.cli.main serve qwen3_5_0.8b --gpus 0 --port $P || exit 1
   for c in 1 32; do python -m benchmark.runner --url http://127.0.0.1:$P --model qwen3.5 --model-id Qwen/Qwen3.5-0.8B --request-type text_to_text --profiling-type closed_loop --max-concurrency \$c --num-requests \$((c==1?32:256)) --num-warmup 2 --ignore-eos --output-len-min 128 --output-len-max 128 --local-cache $R/bench-cache --output-dir $L/$1_c\$c > $L/$1_c\$c.log 2>&1; python - $L/$1_c\$c/results.json $1 \$c <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); a=d['aggregate']; t=a['ttft_s'].get('text') or {}
print('RESULT',sys.argv[2],'c',sys.argv[3],'tok/s %.1f'%(a['text_token_throughput'] or 0),'ttft_p50_ms %.2f'%((t.get('p50') or 0)*1e3),'ok',d['completed'],'fail',d['failed'])
PY
   done; rm -rf \$TORCHINDUCTOR_CACHE_DIR; $SS stop oth >/dev/null 2>&1"
}
for spec in "off1 0" "on1 1" "on2 1" "off2 0" "off3 0" "on3 1"; do set -- $spec; echo "== $1 $(date -u +%T)"; arm $1 $2; done
