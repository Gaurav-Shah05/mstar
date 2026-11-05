#!/usr/bin/env bash
while kill -0 640534 2>/dev/null; do sleep 20; done
flock /workspace/gpu.lock /workspace/results/rp1-pi05-tools/gate_c.sh h26 /workspace/wt-rp1-pi05-h26 /workspace/results/rp1-pi05-tools/ind-h26 > /workspace/results/rp1-pi05-logs/gate_c_h26.out 2>&1
flock /workspace/gpu.lock /workspace/results/rp1-pi05-tools/gate_c.sh best /workspace/wt-rp1-pi05-b4 /workspace/results/rp1-pi05-tools/ind-head > /workspace/results/rp1-pi05-logs/gate_c_best.out 2>&1
