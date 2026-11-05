#!/usr/bin/env bash
# usage: main_gate.sh <name> <checkout> <config> <client cwd>: one fresh server, gate cases sequential and at c=8. Holds the lock throughout.
NAME=$1; CK=$2; CFG=$3; CW=$4; OUT=/workspace/results/rp1-pi05-logs/main_gate/$NAME; mkdir -p $OUT
export PATH=/workspace/venv-main/bin:$PATH VIRTUAL_ENV=/workspace/venv-main MSTAR_SHM_ARENA=1 PYTHONPATH=/workspace/results/rp1-pi05-tools/main_pyshim:$CK SERVER_STATE_DIR=/tmp/servers_rp1main TORCHINDUCTOR_CACHE_DIR=/tmp/rp1ind_main_$NAME; unset TRITON_CACHE_DIR
mkdir -p $TORCHINDUCTOR_CACHE_DIR; SS=$CK/.claude/skills/benchmarking/scripts/server.sh; cd $CK
$SS start mg 8360 --timeout 3600 --log $OUT/server.log -- python -m mstar.cli.main serve pi05 --gpus 0 --port 8360 --config $CFG || { echo "$NAME START FAILED"; $SS stop mg; exit 1; }
cd $CW; PYTHONPATH=$CW python /workspace/results/rp1-pi05-tools/gate_client.py 8360 $OUT/gate.npz > $OUT/gate.log 2>&1; PYTHONPATH=$CW python /workspace/results/rp1-pi05-tools/gate_client_c.py 8360 $OUT/c8 8 3 > $OUT/c8.log 2>&1
cd $CK; $SS stop mg >/dev/null 2>&1; grep -c "these run eagerly" $OUT/server.log | sed "s/^/$NAME dropped-captures /"; echo "$NAME finished"
