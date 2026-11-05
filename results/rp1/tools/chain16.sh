#!/usr/bin/env bash
export FE_FLAG=--rust-frontend
B=/workspace/wt-rp1-main-base; M=/workspace/wt-rp1-main
/workspace/results/rp1-pi05-tools/main_pair_round.sh mck_droid $B $M pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/mck_droid.out 2>&1
/workspace/results/rp1-pi05-tools/main_pair_round.sh mck_base $B $M pi05.yaml 3 > /workspace/results/rp1-pi05-logs/mck_base.out 2>&1
