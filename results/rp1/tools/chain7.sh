#!/usr/bin/env bash
R=/workspace/results/rp1-pi05-tools
L=/workspace/results/rp1-pi05-logs
cd /workspace/wt-rp1-pi05-test && source /workspace/venv/bin/activate
PYTHONPATH=$R/pyshim:/workspace/wt-rp1-pi05-test OMP_NUM_THREADS=2 flock /workspace/gpu.lock timeout -s INT 2400 python -m pytest $(cat $R/h20_tests.txt) test/modular/test_cpu_frequency.py -q -n 8 --dist loadfile -p no:cacheprovider --basetemp=/tmp/rp1t > $L/tests_h20.log 2>&1
$R/pair_round.sh a_h9 /workspace/wt-rp1-pi05-cand /workspace/wt-rp1-pi05-best pi05_droid.yaml 3 > $L/a_h9.out 2>&1
$R/pair_round.sh a_h5 /workspace/wt-rp1-pi05-h0 /workspace/wt-rp1-pi05-cand pi05_droid.yaml 3 > $L/a_h5.out 2>&1
