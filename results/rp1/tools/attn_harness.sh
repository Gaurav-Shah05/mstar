#!/usr/bin/env bash
source /workspace/venv/bin/activate
cd /workspace/results/rp1-pi05-tools
for ck in /workspace/wt-rp1-pi05-cand /workspace/wt-rp1-pi05-h0; do
  for walk in action_gen prefill; do
    PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$ck timeout 1500 python llm_harness.py --checkout $ck --walk $walk --bs 1 --mode graph --tag _attn_$(basename $ck) 2>&1 | grep -E "RESULT graph|Traceback|Error"
  done
done
