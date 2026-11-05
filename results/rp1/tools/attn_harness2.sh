#!/usr/bin/env bash
source /workspace/venv/bin/activate
cd /workspace/results/rp1-pi05-tools
for ck in /workspace/wt-rp1-pi05-cand /workspace/wt-rp1-pi05-h0; do
  for spec in "action_gen 4" "prefill 2" "prefill 4"; do set -- $spec
    PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$ck timeout 1500 python llm_harness.py --checkout $ck --walk $1 --bs $2 --mode graph --tag _attn2_$(basename $ck) 2>&1 | grep -E "RESULT graph|Traceback|Error" | sed "s#^#$(basename $ck) #"
  done
done
