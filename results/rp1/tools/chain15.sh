#!/usr/bin/env bash
/workspace/results/rp1-pi05-tools/pair_round.sh p_h27 /workspace/wt-rp1-pi05-b4 /workspace/wt-rp1-pi05-h27 pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/p_h27.out 2>&1
flock /workspace/gpu.lock /workspace/results/rp1-pi05-tools/gate_c.sh h27 /workspace/wt-rp1-pi05-h27 /workspace/results/rp1-pi05-tools/ind-h26 > /workspace/results/rp1-pi05-logs/gate_c_h27.out 2>&1
