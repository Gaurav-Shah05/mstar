#!/usr/bin/env bash
# paired rounds: H5, then H9 (base = H5 tree), then H6 (base = H5+H9 tree)
R=/workspace/results/rp1-pi05-tools
while pgrep -f "chain1.sh|pair_round.sh pairtest" >/dev/null; do sleep 30; done
$R/pair_round.sh p_h5 /workspace/wt-rp1-pi05-best /workspace/wt-rp1-pi05-cand pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/p_h5.out 2>&1
$R/pair_round.sh p_h9 /workspace/wt-rp1-pi05-cand /workspace/wt-rp1-pi05-h0 pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/p_h9.out 2>&1
$R/pair_round.sh p_h6 /workspace/wt-rp1-pi05-h0 /workspace/wt-rp1-pi05-h6 pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/p_h6.out 2>&1
