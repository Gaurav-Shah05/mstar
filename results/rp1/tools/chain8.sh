#!/usr/bin/env bash
while kill -0 33439 2>/dev/null; do sleep 30; done
/workspace/results/rp1-pi05-tools/other_model_h20.sh > /workspace/results/rp1-pi05-logs/other_h20.out 2>&1
