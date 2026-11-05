#!/usr/bin/env bash
source /workspace/venv/bin/activate
cd /workspace/results/rp1-pi05-tools
for ck in /workspace/wt-rp1-pi05-best /workspace/wt-rp1-pi05-h0; do
  for bs in 1 4; do
    tag=$(basename $ck)
    PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$ck timeout 1500 python llm_harness.py --checkout $ck --walk action_gen --bs $bs --mode graph $([ $ck = /workspace/wt-rp1-pi05-h0 ] && [ $bs = 1 ] && echo --profile) --tag _$tag 2>&1 | grep -E "SUMMARY|action_gen bs=|by class|gemm |attention |elementwise |norm/|Traceback|Error" 
  done
done
