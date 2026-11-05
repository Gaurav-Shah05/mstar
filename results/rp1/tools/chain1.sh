#!/usr/bin/env bash
# checkpoint sweep (droid, base config), full modular suite, then H5 round. Strictly in order.
R=/workspace/results/rp1-pi05-tools
$R/round.sh ckpt1_droid /workspace/wt-rp1-pi05-base /workspace/wt-rp1-pi05-best pi05_droid.yaml "B C C B B C" > /workspace/results/rp1-pi05-logs/ckpt1_droid.out 2>&1
$R/round.sh ckpt1_base /workspace/wt-rp1-pi05-base /workspace/wt-rp1-pi05-best pi05.yaml "C B B C C B" > /workspace/results/rp1-pi05-logs/ckpt1_base.out 2>&1
cd /workspace/wt-rp1-pi05-test && source /workspace/venv/bin/activate && \
  PYTHONPATH=$R/pyshim:/workspace/wt-rp1-pi05-test OMP_NUM_THREADS=2 flock /workspace/gpu.lock \
  timeout -s INT 5400 python -m pytest test/modular -q -n 8 --dist loadfile -p no:cacheprovider --basetemp=/tmp/rp1bt2 \
  > /workspace/results/rp1-pi05-logs/modular_ckpt1.log 2>&1
$R/round.sh r4_h5 /workspace/wt-rp1-pi05-best /workspace/wt-rp1-pi05-cand pi05_droid.yaml "C B B C C B" > /workspace/results/rp1-pi05-logs/r4_h5.out 2>&1
