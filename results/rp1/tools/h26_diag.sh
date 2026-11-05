#!/usr/bin/env bash
# usage: h26_diag.sh <name> <ENV=VAL ...>: H26 tree with a shared Inductor cache, gate only. Holds the lock throughout.
NAME=$1; shift; source /workspace/venv/bin/activate; CK=/workspace/wt-rp1-pi05-h26; OUT=/workspace/results/rp1-pi05-logs/h26_diag/$NAME; mkdir -p $OUT; cd $CK
export MSTAR_SHM_ARENA=1 PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$CK SERVER_STATE_DIR=/tmp/servers_rp1h26d TORCHINDUCTOR_CACHE_DIR=/workspace/results/rp1-pi05-tools/ind-h26; unset TRITON_CACHE_DIR
SS=$CK/.claude/skills/benchmarking/scripts/server.sh
$SS start h26d 8351 --timeout 3600 --log $OUT/server.log -- env "$@" python -m mstar.cli.main serve pi05 --gpus 0 --port 8351 --config $CK/configs/pi05_droid.yaml || { echo "$NAME START FAILED"; $SS stop h26d; exit 1; }
python /workspace/results/rp1-pi05-tools/gate_client.py 8351 $OUT/gate.npz > $OUT/gate.log 2>&1; python /workspace/results/rp1-pi05-tools/gate_client.py 8351 $OUT/gate2.npz > /dev/null 2>&1
echo "$NAME [$*] $(python /workspace/results/rp1-pi05-tools/gate_check.py $OUT/gate.npz 2>&1 | grep -E 'GATE|Error' | cut -c1-70) | run2 $(python /workspace/results/rp1-pi05-tools/gate_check.py $OUT/gate2.npz 2>&1 | grep -E 'GATE|Error' | cut -c1-48)"
$SS stop h26d >/dev/null 2>&1
