#!/usr/bin/env bash
# usage: planlog.sh <name> <config>: H26 tree, plan contents logged during gate cases. Holds the lock throughout.
NAME=$1; CFG=$2; source /workspace/venv/bin/activate; CK=/workspace/wt-rp1-pi05-h26; OUT=/workspace/results/rp1-pi05-logs/planlog/$NAME; mkdir -p $OUT; cd $CK
export MSTAR_SHM_ARENA=1 PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$CK SERVER_STATE_DIR=/tmp/servers_rp1pl TORCHINDUCTOR_CACHE_DIR=/workspace/results/rp1-pi05-tools/ind-h26; unset TRITON_CACHE_DIR
SS=$CK/.claude/skills/benchmarking/scripts/server.sh
$SS start pl 8353 --timeout 3600 --log $OUT/server.log -- env RP1_PLANLOG_DIR=$OUT python -m mstar.cli.main serve pi05 --gpus 0 --port 8353 --config $CFG || { echo "$NAME START FAILED"; $SS stop pl; exit 1; }
python /workspace/results/rp1-pi05-tools/gate_client.py 8353 $OUT/gate.npz > $OUT/gate.log 2>&1; sleep 2
echo "$NAME $(python /workspace/results/rp1-pi05-tools/gate_check.py $OUT/gate.npz 2>&1 | grep -E 'GATE|Error' | cut -c1-70)"
$SS stop pl >/dev/null 2>&1
