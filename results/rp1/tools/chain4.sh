#!/usr/bin/env bash
R=/workspace/results/rp1-pi05-tools
$R/pair_round.sh ckpt2_droid /workspace/wt-rp1-pi05-base /workspace/wt-rp1-pi05-best pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/ckpt2_droid.out 2>&1
$R/pair_round.sh ckpt2_base /workspace/wt-rp1-pi05-base /workspace/wt-rp1-pi05-best pi05.yaml 3 > /workspace/results/rp1-pi05-logs/ckpt2_base.out 2>&1
cd /workspace/wt-rp1-pi05-test && git checkout -q --detach ec5e8627f && source /workspace/venv/bin/activate && \
  PYTHONPATH=$R/pyshim:/workspace/wt-rp1-pi05-test OMP_NUM_THREADS=2 flock /workspace/gpu.lock \
  timeout -s INT 5400 python -m pytest test/modular -q -n 8 --dist loadfile -p no:cacheprovider --basetemp=/tmp/rp1bt3 \
  > /workspace/results/rp1-pi05-logs/modular_ckpt2.log 2>&1
