#!/usr/bin/env bash
R=/workspace/results/rp1-pi05-tools; L=/workspace/results/rp1-pi05-logs; CK=/workspace/wt-rp1-pi05-ck3
$R/pair_round.sh ckpt3_droid /workspace/wt-rp1-pi05-base $CK pi05_droid.yaml 3 > $L/ckpt3_droid.out 2>&1
$R/pair_round.sh ckpt3_base /workspace/wt-rp1-pi05-base $CK pi05.yaml 3 > $L/ckpt3_base.out 2>&1
cd /workspace/wt-rp1-pi05-test && git checkout -q --detach aeda84155 && source /workspace/venv/bin/activate && \
  PYTHONPATH=$R/pyshim:/workspace/wt-rp1-pi05-test OMP_NUM_THREADS=2 flock /workspace/gpu.lock \
  timeout -s INT 5400 python -m pytest test/modular -q -n 8 --dist loadfile -p no:cacheprovider --basetemp=/tmp/rp1bt4 \
  > $L/modular_ckpt3.log 2>&1
