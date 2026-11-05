#!/usr/bin/env bash
export R=/workspace/results/rp1-pi05-tools
while kill -0 477839 2>/dev/null; do sleep 30; done
source /workspace/venv/bin/activate
cd $R
flock /workspace/gpu.lock bash -c 'for spec in "/workspace/wt-rp1-pi05-best prefill" "/workspace/wt-rp1-pi05-h6 prefill" "/workspace/wt-rp1-pi05-h6 action_gen"; do set -- $spec; PYTHONPATH=$R/pyshim:$1 timeout 1500 python llm_harness.py --checkout $1 --walk $2 --bs 1 --text-tokens 150 --mode graph --tag _h11_$(basename $1) 2>&1 | grep "RESULT graph" | sed "s#^#$(basename $1) #"; done' > /workspace/results/rp1-pi05-logs/h11_harness.txt 2>&1
cd /workspace/results && PORT=8318 flock /workspace/gpu.lock $R/arm.sh /workspace/wt-rp1-pi05-h6 pi05_droid.yaml /workspace/results/rp1-pi05-logs/gate_h11 10 1 > /workspace/results/rp1-pi05-logs/gate_h11.out 2>&1
$R/pair_round.sh p_h11 /workspace/wt-rp1-pi05-best /workspace/wt-rp1-pi05-h6 pi05_droid.yaml 3 > /workspace/results/rp1-pi05-logs/p_h11.out 2>&1
