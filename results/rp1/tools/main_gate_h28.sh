#!/usr/bin/env bash
flock /workspace/gpu.lock /workspace/results/rp1-pi05-tools/main_gate.sh base15_h28 /workspace/wt-rp1-main /workspace/results/rp1-pi05-tools/main_gate_base15b.yaml /workspace/wt-rp1-main > /workspace/results/rp1-pi05-logs/main_gate_base15_h28.out 2>&1
flock /workspace/gpu.lock /workspace/results/rp1-pi05-tools/main_gate.sh droid_h28 /workspace/wt-rp1-main /workspace/wt-rp1-main/configs/pi05_droid.yaml /workspace/wt-rp1-main > /workspace/results/rp1-pi05-logs/main_gate_droid_h28.out 2>&1
cd /workspace/wt-rp1-main; PYTHONPATH=/workspace/results/rp1-pi05-tools/main_pyshim:/workspace/wt-rp1-main OMP_NUM_THREADS=2 flock /workspace/gpu.lock timeout -s INT 1500 /workspace/venv-main/bin/python -m pytest test/modular/test_pi05_reference_equivalence.py test/modular/test_pi05_model.py -q -p no:cacheprovider --basetemp=/tmp/mgp2 > /workspace/results/rp1-pi05-logs/main_parity_h28.log 2>&1
echo finished > /workspace/results/rp1-pi05-logs/main_gate_h28.done
