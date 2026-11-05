#!/usr/bin/env bash
source /workspace/venv/bin/activate
for ck in /workspace/wt-rp1-pi05-base /workspace/wt-rp1-pi05-test; do
  cd $ck; echo "== $ck $(git rev-parse --short HEAD)"
  PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:$ck OMP_NUM_THREADS=2 timeout -s INT 2400 python -m pytest test/modular/test_waypoint_gpu.py test/modular/test_gdn_no_sink_slot.py test/modular/test_gdn_prefill_stage.py test/modular/test_gdn_prefix_snapshots.py test/modular/test_gdn_cuda_graph.py test/modular/test_blackwell_ptxas.py test/modular/test_tp_read_failure_acceptance.py -q -p no:cacheprovider --basetemp=/tmp/rp1rc4 2>&1 | grep -E "^(FAILED|ERROR) |passed|failed" | sed -E 's/ - .*//' | cut -c1-125
done
echo finished
