#!/usr/bin/env bash
source /workspace/venv/bin/activate; CK=/workspace/wt-rp1-pi05-ck3; cd $CK
export PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$CK MSTAR_SHM_ARENA=1 TORCHINDUCTOR_CACHE_DIR=/workspace/results/rp1-pi05-tools/ind-harness
python /workspace/results/rp1-pi05-tools/llm_harness.py --checkout $CK --walk action_gen --bs 1 --mode graph --profile > /workspace/results/rp1-pi05-logs/h_action.log 2>&1
python /workspace/results/rp1-pi05-tools/llm_harness.py --checkout $CK --walk prefill --bs 1 --mode graph --profile > /workspace/results/rp1-pi05-logs/h_prefill.log 2>&1
