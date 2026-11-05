#!/usr/bin/env bash
while kill -0 80160 2>/dev/null; do sleep 30; done
/workspace/results/rp1-pi05-tools/pair_round.sh p_h23 /workspace/wt-rp1-pi05-ck3 /workspace/wt-rp1-pi05-h23 pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/p_h23.out 2>&1
