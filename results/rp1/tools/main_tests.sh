#!/usr/bin/env bash
# targeted CPU tests at the port and at 9c6f1ad
for ck in /workspace/wt-rp1-main /workspace/wt-rp1-main-base; do
  cd $ck; F=$(for f in $(cat /tmp/main_tests.txt); do [ -f $f ] && echo $f; done)
  n=$(basename $ck)
  PYTHONPATH=/workspace/results/rp1-pi05-tools/main_pyshim:$ck OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= timeout -s INT 1700 /workspace/venv-main/bin/python -m pytest $F -q -n 6 --dist loadfile -p no:cacheprovider --basetemp=/tmp/m${n: -4} > /workspace/results/rp1-pi05-logs/main_tests_$n.log 2>&1
  echo "$n $(tail -1 /workspace/results/rp1-pi05-logs/main_tests_$n.log | cut -c1-110)"
done
echo finished
