#!/usr/bin/env bash
source /workspace/venv/bin/activate; CK=/workspace/wt-rp1-pi05-ck3; cd $CK
export PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$CK MSTAR_SHM_ARENA=1
for arm in nocublas base; do
 export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/rp1ind.XXXX)
 S=/workspace/results/rp1-pi05-tools/h24_run.py; [ $arm = base ] && S=
 for w in prefill action_gen; do
  python $S /workspace/results/rp1-pi05-tools/llm_harness.py --checkout $CK --walk $w --bs 1 --mode graph --profile --tag h24$arm > /workspace/results/rp1-pi05-logs/h24_${arm}_$w.log 2>&1
 done
 rm -rf $TORCHINDUCTOR_CACHE_DIR
done
