#!/usr/bin/env bash
/workspace/results/rp1-pi05-tools/pair_round.sh p_h28 /workspace/wt-rp1-pi05-b4 /workspace/wt-rp1-pi05-h28b pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/p_h28.out 2>&1
flock /workspace/gpu.lock /workspace/results/rp1-pi05-tools/gate_c.sh h28 /workspace/wt-rp1-pi05-h28b /workspace/results/rp1-pi05-tools/ind-h28b > /workspace/results/rp1-pi05-logs/gate_c_h28.out 2>&1
