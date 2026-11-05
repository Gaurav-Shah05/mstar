#!/usr/bin/env bash
while kill -0 491753 2>/dev/null; do sleep 30; done
ENV_B=RP1_PAR_DECODE=1 /workspace/results/rp1-pi05-tools/pair_round.sh diag_pardec /workspace/wt-rp1-pi05-best /workspace/wt-rp1-pi05-best pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/diag_pardec.out 2>&1
