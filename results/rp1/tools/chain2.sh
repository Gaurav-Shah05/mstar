#!/usr/bin/env bash
# after chain1: H9 round (base = H5 tree), then H6 round (base = H5+H9 tree)
R=/workspace/results/rp1-pi05-tools
while pgrep -f "chain1.sh" >/dev/null; do sleep 30; done
$R/round.sh r5_h9 /workspace/wt-rp1-pi05-cand /workspace/wt-rp1-pi05-h0 pi05_droid.yaml "B C C B B C" > /workspace/results/rp1-pi05-logs/r5_h9.out 2>&1
$R/round.sh r6_h6 /workspace/wt-rp1-pi05-h0 /workspace/wt-rp1-pi05-h6 pi05_droid.yaml "C B B C C B" > /workspace/results/rp1-pi05-logs/r6_h6.out 2>&1
