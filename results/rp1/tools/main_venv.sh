#!/usr/bin/env bash
cd /workspace
uv venv /workspace/venv-main --python 3.12 --seed
VIRTUAL_ENV=/workspace/venv-main uv pip install --torch-backend=auto -e "/workspace/wt-rp1-main[all,dev]"
echo "venv exit $?"
