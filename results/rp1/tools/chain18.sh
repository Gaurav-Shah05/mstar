#!/usr/bin/env bash
R=/workspace/results/rp1-pi05-tools; L=/workspace/results/rp1-pi05-logs; B=/workspace/wt-rp1-pi05-base; C=/workspace/wt-rp1-pi05-b5
$R/round.sh ckpt5_droid $B $C pi05_droid.yaml "B C C B B C" > $L/ckpt5_droid.out 2>&1
$R/round.sh ckpt5_base $B $C pi05.yaml "C B B C C B" > $L/ckpt5_base.out 2>&1
cd /workspace/wt-rp1-pi05-test && git checkout -q --detach 3c4fdfa9a && source /workspace/venv/bin/activate && \
  PYTHONPATH=$R/pyshim:/workspace/wt-rp1-pi05-test OMP_NUM_THREADS=2 flock /workspace/gpu.lock \
  timeout -s INT 5400 python -m pytest test/modular -q -n 8 --dist loadfile -p no:cacheprovider --basetemp=/tmp/rp1bt6 \
  > $L/modular_ckpt5.log 2>&1
echo finished > $L/ckpt5.done
