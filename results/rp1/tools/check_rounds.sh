#!/usr/bin/env bash
# Round validity: any server of a paired run that dropped a graph capture. usage: check_rounds.sh <tag>
cd /workspace/results/rp1-pi05-logs; grep -l "these run eagerly" $1/r*/[AB]/server.log 2>/dev/null || echo "$1: all captures ok"
