IN PROGRESS: bucket cut A/B on 9B, then review, gates, numbers

# PR Qwen3.5 facts

## SHAs
- Base: 305e80ef46248ade6e6670aa20c9bb67f4ac2e26. Branch: perf/rp3-pr-qwen35. Head: moving until READY.

## Environment notes
- torchaudio 2.11.1+cu132 (what `[all]` resolves to) cannot load against torch 2.12.1 (OSError on libtorchaudio.so). Uninstalled from venv-main305 and venv-pr, as on earlier venvs; audio test modules then fail at collection, so full runs use --continue-on-collection-errors on both arms.
- pytest-xdist added to both venvs for the -n 8 runs.
- Qwen3-Omni weights removed from this pod's HF cache to fit Qwen3.5-9B (re-downloadable; /workspace had 24 GB free).
