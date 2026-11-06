IN PROGRESS: review fixes under test (head d96154d), then sweep, ladder, gates, references

# PR Qwen3.5 facts

## SHAs
- Base: 305e80ef46248ade6e6670aa20c9bb67f4ac2e26. Branch: perf/rp3-pr-qwen35. Head: moving until READY.

## Environment notes
- torchaudio 2.11.1+cu132 (what `[all]` resolves to) cannot load against torch 2.12.1 (OSError on libtorchaudio.so). Uninstalled from venv-main305 and venv-pr, as on earlier venvs; audio test modules then fail at collection, so full runs use --continue-on-collection-errors on both arms.
- pytest-xdist added to both venvs for the -n 8 runs.
- Qwen3-Omni weights removed from this pod's HF cache to fit Qwen3.5-9B (re-downloadable; /workspace had 24 GB free).

## Bucket cut decision (9B Random 1024/256, 3 alternating rounds, on the three concepts; rows hypothesis "PR check: qwen35 bucket cut"; arm 440391c, before the review's refactor)
- TTFT p50 c=1: without 59.9, 58.0, 56.8 ms; with 42.2, 42.9, 42.7 ms. All three beat. KEPT.
- c=16 TTFT p50: without 198.7, 198.0, 196.9; with 200.1, 197.3, 201.2 (overlap). tok/s without 2170.5, 2165.4, 2167.4; with 2171.5, 2175.5, 2171.7.
- c=64 tok/s: without 4536.7, 4523.6, 4532.0; with 4533.9, 4544.8, 4540.5 (overlap). TTFT p50 without 218.8, 218.2, 219.5; with 218.9, 220.6, 220.8 (overlap).

## Baseline tests
- Serial and -n 8 --dist loadfile at the base agree: 50 failed, 7 errors, 10898 passed, 1290 skipped, same ids. Parallel used from then on.
