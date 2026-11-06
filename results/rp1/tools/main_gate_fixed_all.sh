#!/usr/bin/env bash
B=/workspace/wt-rp1-main-base; M=/workspace/wt-rp1-main
flock /workspace/gpu.lock /workspace/results/rp1-pi05-tools/main_gate_fixed.sh base15_h28_fixed $M /workspace/results/rp1-pi05-tools/main_gate_base15b.yaml > /workspace/results/rp1-pi05-logs/main_gate_base15_h28_fixed.out 2>&1
flock /workspace/gpu.lock /workspace/results/rp1-pi05-tools/main_gate_fixed.sh base15_base_fixed $B /workspace/results/rp1-pi05-tools/main_gate_base15.yaml > /workspace/results/rp1-pi05-logs/main_gate_base15_base_fixed.out 2>&1
echo finished > /workspace/results/rp1-pi05-logs/main_gate_fixed.done
