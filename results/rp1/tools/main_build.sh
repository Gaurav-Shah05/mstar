#!/usr/bin/env bash
export VIRTUAL_ENV=/workspace/venv-main PATH=/workspace/venv-main/bin:$PATH CARGO_BUILD_JOBS=4
cd /workspace/wt-rp1-main
uv pip install maturin
maturin develop --release -m rust/Cargo.toml; echo "maturin exit $?"
cargo build --release --manifest-path rust/server/Cargo.toml; echo "cargo exit $?"
