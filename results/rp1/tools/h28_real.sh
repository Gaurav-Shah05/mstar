#!/usr/bin/env bash
source /workspace/venv/bin/activate
for ck in /workspace/wt-rp1-pi05 /workspace/wt-rp1-pi05-b4; do cd $ck; echo "== $ck"; PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$ck OMP_NUM_THREADS=2 timeout -s INT 900 python -m pytest test/integration/test_pi05_real_weights.py -q -p no:cacheprovider --basetemp=/tmp/h28r 2>&1 | grep -E "^(FAILED|ERROR) |passed|failed|skipped" | grep -v Deprec | sed -E 's/ - .*//' | cut -c1-120 | tail -12; done; echo finished
