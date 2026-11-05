#!/usr/bin/env bash
while kill -0 33439 2>/dev/null; do sleep 30; done
/workspace/results/rp1-pi05-tools/pair_round.sh p_p1 /workspace/wt-rp1-pi05-best /workspace/wt-rp1-pi05-p1 pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/p_p1.out 2>&1
/workspace/results/rp1-pi05-tools/pair_round.sh p_h21 /workspace/wt-rp1-pi05-best /workspace/wt-rp1-pi05-h6 pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/p_h21.out 2>&1
