#!/usr/bin/env bash
mkdir -p /workspace/results/rp1-pi05-tools/ind-h28a /workspace/results/rp1-pi05-tools/ind-h28b
/workspace/results/rp1-pi05-tools/bs_probe.sh h28a /workspace/wt-rp1-pi05-h28a /workspace/results/rp1-pi05-tools/ind-h28a
/workspace/results/rp1-pi05-tools/bs_probe.sh h28b /workspace/wt-rp1-pi05-h28b /workspace/results/rp1-pi05-tools/ind-h28b
echo finished
