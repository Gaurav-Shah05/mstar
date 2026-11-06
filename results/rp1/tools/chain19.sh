#!/usr/bin/env bash
while [ ! -f /workspace/results/rp1-pi05-logs/ckpt5.done ]; do sleep 30; done
/workspace/results/rp1-pi05-tools/round.sh s_h29 /workspace/wt-rp1-pi05-b5 /workspace/wt-rp1-pi05-h29 pi05_droid.yaml "C B B C C B" > /workspace/results/rp1-pi05-logs/s_h29.out 2>&1
/workspace/results/rp1-pi05-tools/round.sh s_h30 /workspace/wt-rp1-pi05-b5 /workspace/wt-rp1-pi05-h30 pi05_droid.yaml "B C C B B C" > /workspace/results/rp1-pi05-logs/s_h30.out 2>&1
echo finished > /workspace/results/rp1-pi05-logs/chain19.done
