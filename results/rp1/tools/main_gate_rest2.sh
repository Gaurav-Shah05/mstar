#!/usr/bin/env bash
B=/workspace/wt-rp1-main-base; M=/workspace/wt-rp1-main
SERVER_STATE_DIR=/tmp/servers_rp1main $B/.claude/skills/benchmarking/scripts/server.sh stop mg >/dev/null 2>&1
flock /workspace/gpu.lock /workspace/results/rp1-pi05-tools/main_gate.sh droid_base $B $B/configs/pi05_droid.yaml $B > /workspace/results/rp1-pi05-logs/main_gate_droid_base.out 2>&1
cd $M; PYTHONPATH=/workspace/results/rp1-pi05-tools/main_pyshim:$M OMP_NUM_THREADS=2 flock /workspace/gpu.lock timeout -s INT 900 /workspace/venv-main/bin/python -m pytest test/modular/test_pi05_reference_equivalence.py test/modular/test_pi05_model.py -q -p no:cacheprovider --basetemp=/tmp/mgp > /workspace/results/rp1-pi05-logs/main_parity_gpu.log 2>&1
echo finished > /workspace/results/rp1-pi05-logs/main_gate_rest.done
