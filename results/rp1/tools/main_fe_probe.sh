#!/usr/bin/env bash
# Does --rust-frontend serve pi0.5's requests? Port tree, droid config + knob; gate client and a short c=1 and c=4 run.
R=/workspace/results/rp1-pi05-tools; OUT=/workspace/results/rp1-pi05-logs/main_fe; mkdir -p $OUT; CK=/workspace/wt-rp1-main; cd $CK
export PATH=/workspace/venv-main/bin:$PATH VIRTUAL_ENV=/workspace/venv-main MSTAR_SHM_ARENA=1 SERVER_STATE_DIR=/tmp/servers_rp1fe PYTHONPATH=$R/main_pyshim:$CK TORCHINDUCTOR_CACHE_DIR=/tmp/rp1ind_m MSTAR_SERVER_BIN=$CK/rust/server/target/release/mstar-server
python - <<'PY'
import yaml
c=yaml.safe_load(open("configs/pi05_droid.yaml")); c.setdefault("knobs",{})["step_thread_core"]="siblings"; yaml.safe_dump(c,open("/workspace/results/rp1-pi05-logs/main_fe/cfg.yaml","w"),sort_keys=False)
PY
SS=$CK/.claude/skills/benchmarking/scripts/server.sh
$SS start fe 8363 --timeout 3600 --log $OUT/server.log -- python -m mstar.cli.main serve pi05 --gpus 0 --port 8363 --config $OUT/cfg.yaml --rust-frontend || { echo START FAILED; $SS stop fe; exit 1; }
export PYTHONPATH=$CK
python $R/gate_client.py 8363 $OUT/gate.npz > $OUT/gate.log 2>&1; echo "gate client exit $?"; tail -2 $OUT/gate.log | cut -c1-160
for c in 1 4; do python -m benchmark.runner --url http://127.0.0.1:8363 --model pi05 --request-type vision_language_action --profiling-type closed_loop --max-concurrency $c --num-requests $((c*16)) --num-warmup 1 --local-cache $R/droid-cache --output-dir $OUT/c$c > $OUT/b$c.log 2>&1; python - $OUT/c$c/results.json $c <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print("RESULT c",sys.argv[2],"p50 %.2f ok %d fail %d"%(d['jct_median_ms'],d['completed'],d['failed']))
PY
done
$SS stop fe >/dev/null 2>&1; echo finished
