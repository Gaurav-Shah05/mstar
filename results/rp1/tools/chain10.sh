#!/usr/bin/env bash
while kill -0 76532 2>/dev/null; do sleep 30; done
/workspace/results/rp1-pi05-tools/pair_round.sh p_h22 /workspace/wt-rp1-pi05-p1 /workspace/wt-rp1-pi05-h22 pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/p_h22.out 2>&1
