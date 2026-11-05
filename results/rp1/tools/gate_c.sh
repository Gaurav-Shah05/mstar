#!/usr/bin/env bash
# usage: gate_c.sh <name> <checkout> <inductor cache>: gate cases at c=1,4,8,16, 3 repeats each, one fresh server. Holds the lock throughout.
NAME=$1; CK=$2; IND=$3; source /workspace/venv/bin/activate; OUT=/workspace/results/rp1-pi05-logs/gate_c/$NAME; mkdir -p $OUT; cd $CK
export MSTAR_SHM_ARENA=1 PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$CK SERVER_STATE_DIR=/tmp/servers_rp1gc TORCHINDUCTOR_CACHE_DIR=$IND; unset TRITON_CACHE_DIR
SS=$CK/.claude/skills/benchmarking/scripts/server.sh
$SS start gc 8352 --timeout 3600 --log $OUT/server.log -- python -m mstar.cli.main serve pi05 --gpus 0 --port 8352 --config $CK/configs/pi05_droid.yaml || { echo "$NAME START FAILED"; $SS stop gc; exit 1; }
for c in 1 4 8 16; do python /workspace/results/rp1-pi05-tools/gate_client_c.py 8352 $OUT/c$c $c 3 > $OUT/c$c.log 2>&1; for r in 0 1 2; do echo "$NAME c=$c rep $r $(python /workspace/results/rp1-pi05-tools/gate_check.py $OUT/c${c}_$r.npz 2>&1 | grep -E 'GATE|Error' | cut -c1-62)"; done; done
$SS stop gc >/dev/null 2>&1; echo "$NAME finished"
