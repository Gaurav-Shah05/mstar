## Part 3 status

- Base 9c6f1ad. Best commit afbf77c. Kept, each with a test or stated gaps in its commit: r1 9a5c390 (test c39528b) auto takes siblings where cores sleep deep; r5 917eb8a (test b851763) GDN one-launch stage on Hopper; r10 f35f5c3 (test 9a228e9) GatherPolicy for Omni's audio encoder; r15 afe4633 8192 vision prefill bucket; r18 78d41d6 (test afbf77c) no Talker masks without audio output.
- Checkpoint 4 (best afe4633 / faster ref, tok/s): 0.8B ShareGPT c1 1.28, c16 1.38, c64 1.35; Random c16 0.97, c64 0.97. 2B 1.16, 1.29, 1.45; Random 1.03, 1.07. TTFT p50 0.36-0.70 of the reference. Best / 9c6f1ad: 0.8B ShareGPT 1.90-2.07, Random 1.48-1.61; 2B 1.40-1.72, 1.20-1.27.
- 9B at b851763 / faster ref: 1.03, 1.12, 1.16; Random 1.01, 1.01.
- Omni Thinker A2T / vLLM (SGLang crashes on batched audio): at afe4633 c32 0.78 (63.3 vs 80.9 req/s, TTFT p50 113 vs 76 ms), +21% over 9c6f1ad; c8 0.98. r18 then 68.0 req/s, TTFT 106 ms: about 0.84 (against checkpoint 4's vLLM, not alternated), about +31% over 9c6f1ad. The one remaining gap; proposal 10.
- Images / vLLM cache off, MMMU: 0.8B c4 1.30, c16 1.35; 2B c16 1.19, c4 warm 1.21 (before r15, which fixes the first-pass stall).
- Open risks: r1's H200 rows in docs where siblings lost, not reproducible here, one host class measured; r1 and the Python frontend: c64 TTFT about 350 ms; r10 trades TTFT (c32 +16%); r15's 9B MMMU c16 about 2% lower than before it (1836 vs 1874-1879; later rounds show no stall). Not run: TP, 27B, Qwen3.8, A2A, Blackwell.
- Reverted or not kept: r2, r3, r4, r6, r7, r8, r11, r12, r13, r14, r16, r17.
- In progress: nothing. No checkpoint at afbf77c (sweep longer than the time left); last checkpoint afe4633. Omni at afbf77c is GPU-led (re-profile at the end of the file); finer decode batch sizes showed nothing in a one-round probe. Queue: proposal 10, kernels on the Thinker decode step.

## Status (2026-11-03 15:10 UTC)

- Best commit: HEAD c5251738b on perf/rp3-qwen3vl (Qwen3-VL 2B/4B/8B in mstar/model/qwen3_vl/, registered, configs, docs, tests). Parity gate (64 ShareGPT + 16 MMMU, teacher-forced top-1 vs transformers sdpa, bar = transformers eager) passes on 2B, 4B and 8B at HEAD; log-prob fidelity vs fp32 shows the served path at least as close as transformers bf16 for text, images and video; negative control (deepstack off) fails the gate.
- Runtime: production Rust runtime (mstar_rust built, MSTAR_SHM_ARENA=1) since ~10:40 UTC; earlier rows are pure-Python runtime.
- Kept changes after bring-up: c1744fc63 serving fix (start-up warmup of the uncaptured >4096-token / >16384-patch paths; MMMU c1 TTFT p99 2963 -> ~1150 ms). Loop rounds 1-2 (capture ladder ratio 1.5) reverted.
- Headline (8B MMMU c16, n=384 since round 2): checkpoint 3 at HEAD 1845.7 [1835.4-1857.8] tok/s, MMMU c1 TTFT p50 47.0 ms; 3v3 noise 3-11% on this shared host (load average 20-50 from other tenants). GPU time: decode ~71% (GEMMs ~88% of HBM roofline), vision encoder ~13% (attention-bound, fastest available kernel), image prefill ~16% (compute-bound). No model-lane candidate found with an expected effect above the noise; measured bounds and rejections below.
- Shared-code changes (qwen3_lm.py, qwen_vision.py) measured on Qwen3.5-2B and Qwen3-ASR-1.7B vs run-base: no non-overlapping regression.

## Hypothesis queue (by expected gain on Image MMMU c=16, 8B)

State 2026-11-03 15:10 UTC, base HEAD c5251738b (checkpoint 3: 1845.7 [1835.4-1857.8] tok/s). No model-lane candidate left whose expected headline effect the 3v3 all-beat rule can resolve on this host:

1. Mixing prefill_vision with decode: bound 2.2% (209 displaced decode steps of 5.54 ms in 52 s); eager routing is not viable (eager prefill_vision: c1 TTFT x2, c16 -1.3%), so only an Omni-style mixed capture remains, est. net 1-1.5%. Dropped before implementation (design map: /workspace/results/rp3-notes/chunked_prefill_vision_map.md).
2. Capture ladder ratio 1.5: ~1.3%, inconclusive in rounds 1 and 2; dropped (patch /workspace/rp3-bench/round1_ladder.patch).
3. Opaque-RGBA fast path in media_limits._on_white: ~2 ms of ~47 ms c1 TTFT; TTFT-only (cannot be kept on the headline rule); shared model code.
4. Fused norm + table RoPE + KV write for prefill_vision: <1% (patch /workspace/rp3-bench/h1_fused_table_rope.patch).
5. Engine / shared-kernel items with measured shares: /workspace/results/rp3-qwen3vl-proposals.md.

Rejected with evidence: head_dim 72 padding (no faster kernel), CUDA-core decode attention (slower at bs>=16), split vision/LLM workers with MPS (-25% on the Rust runtime), API-server preprocessing proposal (withdrawn after attribution).

## Base profile (42312c3e8, 8B)

- Uninstrumented round (probe r0): ShareGPT c1 207.4 / c16 2175.6 / c64 5788.3 tok/s; MMMU c1 180.0 / c4 512.8 / c16 1515.9 tok/s; TTFT p50 ShareGPT 21.4/30.2/60.7 ms, MMMU 51.5/127.5/232.2 ms. Load avg 31 on the host.
- worker_phases (MMMU c16, steady bs~15): iter_total 5.83 ms, event_sync 2.16 ms (GPU-bound component present).
- Prefill-only MMMU c16 (1 output token): 11.8 s first pass, 8.1-8.3 s warm; full (256 tokens) 21.4 s.
- TORCH_LOGS=recompiles: one recompile in the first timed prefill-only pass (prefill_vision eager step, new batch size after the graph break at the sampler call; dynamo then marks the shape dynamic); none after.
- MMMU first 128 requests: image tokens mean 944, median 308, max 5360; 14 exceed the 4096-token prefill_vision bucket and the 16384-patch block-loop bucket.
- Kernel table (torch.profiler, 600 steps): skinny decode GEMMs ~4.0 ms/decode step (weights 15.1 GB -> ~3.5 ms at 4.3 TB/s: ~88% of roofline); decode attention + split-KV merge ~0.74 ms/step (KV ~2.65 GB/step -> ~0.66 ms); fused_add_rmsnorm 0.24 ms/step; split-K reduce 0.14; qk_norm_rope_kv 0.09; silu 0.08; sampler 0.04.

## Rejected / tested hypotheses

- 2026-11-03: "image-subset parity deficit comes from the fused qk-norm-RoPE-KV kernel's fast sin/cos at large positions". Tested: fused kernel vs fp64 at positions 0..20000 has max error equal to transformers' own fp32 path (pos 6000: 2.6e-4 vs 2.3e-4). Rejected.
- 2026-11-03: "the 12288-patch block-loop bucket replays wrong". One MMMU image showed served tower cos 0.896 vs fp32 (HF bf16 0.995). Re-tested the same grid (11600 patches) and 4096..16384 patches: all 0.9956-0.9972 (HF bf16 range). Cause was the parity harness: it gave transformers `Image.convert("RGB")` (drops alpha) while M* and transformers' own `convert_to_rgb` composite RGBA on white. Harness fixed (served_parity.py passes the image as opened).

## Setup log

- 2026-11-03 04:2x UTC. Worktree /workspace/wt-rp3-qwen3vl, branch perf/rp3-qwen3vl, run-base = e8826a77754347f37804505117125991cdbce308 (verified). GPU NVIDIA H200, no other process.
- Kirill's commits cherry-picked with his authorship: 0b5d307f -> e98c5dd1d, 6b54a8f3 -> 42cb59991. Conflicts in encoder.py, weight_loading.py, qwen3_5/components/vision.py, test_qwen_vision_tower.py resolved to his own merge tree 6e5236766 (that merge's main parent 3e98682a8 is an ancestor of run-base and none of the 10 files changed between it and run-base; final tree == 6e5236766 for those files).
- His tests failed at 42cb59991 (5 of 8: stale `device` arg to main's grid helpers; one asserts a binding main's VisionAttention does not keep). Fixed / skipped with reason in 4b36fe844.
- Env: pytest-xdist was missing from the venv; installed (3.8.0).
- Env: torchaudio 2.11.1+cu132 is ABI-incompatible with torch 2.12.1+cu132 (`undefined symbol: torch_exception_get_what_without_backtrace`); no 2.12 build on the cu132 index; pod install.log shows this pairing from setup. Every transformers image/audio processor import failed with OSError. Uninstalled torchaudio from /workspace/venv (reinstall: `pip install torchaudio==2.11.1+cu132 --index-url https://download.pytorch.org/whl/cu132`). mstar/api_server/data_worker.py:22-25 catches ImportError and OSError alike.
- Env: the venv's `mstar` is the editable checkout /workspace/mstar (run-base). Arms run with PYTHONPATH=<tree>; each measured commit is a `git archive` snapshot under /workspace/rp3-arms/<sha>. mstar_rust is not built, so MSTAR_SHM_ARENA stays 0 (file SHM) for every arm.
- Benchmark driver /workspace/rp3-bench/qwen3vl_bench.py: benchmark/runner.py's Benchmark (OurSystem /generate, closed loop, 1 warmup wave) with two datasets the runner lacks: ShareGPT (vLLM sampling: first human turn, output len = first reply's tokens, prompt 4-1024, total <= 2048, seed 0) and MMMU validation (lmms-lab/MMMU, image_1, options inlined, 256 output tokens). ignore_eos, temperature 0. Throughput = expected output tokens of successful requests / wall time. Round script: /workspace/rp3-bench/bench_round.sh.
- No /v1/chat/completions for Qwen3-VL: ADAPTER_REGISTRY lives in mstar/api_server (core).

## Correctness

- Component parity (test/qwen3_vl/component_parity.py, dense stand-in resources), 2B and 8B: tower merged/deepstack as close to fp32 as transformers' bf16 tower; text-position logits closer to fp32 than transformers sdpa (2B cos 0.999854 vs 0.999807; 8B 0.997454 vs 0.993669). PASS.
- Served parity (test/qwen3_vl/served_parity.py, 64 ShareGPT + 16 MMMU, 64 greedy tokens, teacher-forced top-1 vs transformers sdpa; bar = transformers eager greedy scored the same way), pre-commit tree, harness before the alpha fix:
  - 2B: served 0.9848 vs eager 0.9846 PASS
  - 8B: served 0.9886 vs eager 0.9884 PASS
  - 4B: served 0.9910 vs eager 0.9914 FAIL (image subset 1013 vs 1018 of 1024) -> traced to the harness alpha bug above; rescoring.
- Modular unit tests: test/modular/test_qwen3_vl.py 8 passed.

### 2026-11-03 ~05:40 UTC update

- Harness: served_parity.py now scores the served ids (`return_token_ids`, stop token included), commit 42312c3e8. Retokenizing was not the cause of the 4B gap: same totals with exact ids.
- Served parity at 42312c3e8 (exact ids, alpha-correct HF input):
  - 2B: served 0.9854 (4938/5011) vs eager 0.9848 PASS; image 1010 vs 1010, text 3928 vs 3925
  - 4B: served 0.9912 (4971/5015) vs eager 0.9914 FAIL by 1 position; image 1014 vs 1018, text 3957 vs 3954. Per item: served misses 1-2 positions on 4 images where eager does not; eager misses on 2 where served does not.
  - 8B: served 0.9896 (4947/4999) vs eager 0.9888 PASS; image 1015 vs 1014
- INVALID negative control: every server so far loaded /workspace/wt-rp3-qwen3vl (python -m puts cwd first on sys.path; conductor log `yaml /workspace/wt-rp3-qwen3vl/configs/...`). The "deepstack off" control therefore ran the real code; its 5/16 diverging image replies are cross-process variation (text replies 64/64 identical across the two processes). Parity numbers above are still 42312c3e8's (worktree == HEAD then). Scripts now cd into the arm tree and abort unless the server log shows that tree's config.
- test/modular on CPU (CUDA hidden, -n 8 --dist loadfile, OMP_NUM_THREADS=2): 42cb59991 67 failed / 6 errors; 42312c3e8 60 failed / 6 errors. New failures at HEAD: none. Fixed at HEAD: Kirill's 5 vision tests, 2 test_eval_compare (timing). Logs: /workspace/results/rp3-logs/modular_cpu_*.log

### Parity verdict for the loop base (42312c3e8)

- Log-prob fidelity, 4B (diag_logprob.py; served raw logprob of each sampled token vs transformers fp32 teacher-forced on the same tokens; arm tree verified in server log):
  - real: image prefill served 0.03485 vs hf-bf16 0.03508; image decode 0.01037 vs 0.01237; text prefill 0.00856 vs 0.01456; text decode 0.00982 vs 0.01125 (mean |lp - lp_fp32|). Served is closer to fp32 than transformers bf16 in every bucket.
  - deepstack-off control: image prefill 0.37096 (vs 0.04063), image decode 0.09041 (vs 0.01459); text unchanged. Detected, ~9x.
- Top-1 gate control (2B, deepstack off, tree verified): served 0.9747 vs eager 0.9848, image 965 vs 1010 -> FAIL. The gate can fail.
- Decision: 2B and 8B pass the top-1 gate; 4B is short by 1 position of 5015 (4971 vs 4972), which the log-prob measure shows is near-tie noise (served closer to fp32 than transformers bf16). Recorded as a gate FAIL for 4B; 4B gate re-run after each kept change. Loop base = 42312c3e8 (headline size 8B passes).

### 2026-11-03 06:3x UTC, round 0 diagnosis (no code change)

- Prefill-only MMMU c16 8B GPU windows (warm, 8.1 s): vision_encoder 3.94 s (49%, 50.6 ms/step, 1.92 images/step), prefill_vision replay 2.22 s (27%), prefill_vision eager 1.94 s (24%).
- Hypothesis "8B tower head_dim 72 padded to 128 wastes ~1.5 s; a native-72 kernel would recover it" -> REJECTED by microbenchmark (/workspace/rp3-bench/attn_hd_bench.py, non-causal, 16 heads, bf16): n=21440 flashinfer ragged d128 5.37 ms (395 TF/s useful at d72), torch SDPA d72 cudnn 6.35 / flash 9.05 / efficient 21.63 ms. No available kernel beats the padded one; no proposal.
- Conclusion for the headline: decode ~88% of bandwidth roofline, vision attention at kernel optimum, prefill compute-bound. Remaining model-lane levers are each est. <=3%.

## Rounds

### Round 1: prefill_vision capture ladder ratio 1.5 instead of 2 -> reverted

- Hypothesis: a prefill_vision step pads to its capture bucket and this prefill is compute-bound; buckets 2^k and 1.5*2^k (64..4096) cut padded tokens on MMMU's first 128 prompts from 26.8% to 16.0% (first 32: 31.6% -> 18.0%). Expected: MMMU c16 8B +1-1.5% throughput (captured image prefill is 9.7% of GPU time), MMMU c1 TTFT p50 ~-4%.
- Candidate: working tree on 42312c3e8 + mstar/model/qwen3_vl/submodules.py `PREFILL_VISION_TOKEN_BUCKETS`; snapshot /workspace/rp3-arms/cand-b807f7312 (`git stash create` object b807f7312). Recompile headroom checked: busiest frame at 40 of 84 compiles in the base.
- Commands: /workspace/rp3-bench/round1.sh (parity.sh 8 and 4 on the candidate; then bench_round.sh base/cand alternating, rounds 1-3, 8B, all six workloads, OUTROOT /workspace/results/rp3-bench/round1). Summary: report_round.py /workspace/results/rp3-bench/round1 8 42312c3 <commit> "<hypothesis>".
- Parked: fused norm + table RoPE + KV write for prefill_vision, /workspace/rp3-bench/h1_fused_table_rope.patch (est. <1%).
- Review (independent subagent): approach defensible (a ratio-bounded ladder, same kind of declaration as Qwen3.5's); flagged a recompile-limit risk from test_qwen3_5_bucket_knobs.py's accounting (estimate 80 -> 98 of 84) and asked for explicit list + shorter comment (done in worktree; values unchanged).
- Recompile check, measured (TORCH_LOGS=recompiles, prefill-only MMMU c16 after startup): candidate busiest frame forward_batched 46 entries (base 40), 0 `recompile_limit` hits. The test's per-(rows, tokens) formula overcounts for this node; the limit is not reached.
- Parity: 8B candidate served 0.9898 (4948/4999) vs eager 0.9888 PASS.
- Parity: 4B candidate served 0.9920 (4975/5015) vs eager 0.9914 PASS.
- Results (3 rounds each, alternating base/cand, fresh server per round, 8B; mean [min-max] tok/s; TTFT p50 ms):
  - MMMU c1: base 184.9 [180.5-193.1], ttft 50.4 [46.3-52.8] | cand 182.7 [179.2-188.3], ttft 49.0 [47.9-49.6] | -1.2%
  - MMMU c4: base 556.0 [549.1-563.6] | cand 524.3 [512.7-540.4] | -5.7%, non-overlapping (regression)
  - MMMU c16 (headline): base 1525.2 [1464.3-1564.2] | cand 1536.9 [1497.6-1590.2] | +0.8%, overlapping
  - ShareGPT c1: 211.9 [207.0-221.6] | 210.6 [207.4-216.7]; c16: 2188.9 [2177.7-2198.1] | 2187.1 [2176.0-2193.9]; c64: 5430.9 [5114.4-5747.9] | 5324.9 [5293.2-5388.2]
  - base r2 ShareGPT c64: 2 of 256 failed, "Server disconnected" on request ids 0 and 1 (first timed requests after warmup), no server-side error: client keep-alive artifact; lowers that arm's number.
  - Load average 8-22 across arms.
- Decision: REVERTED (headline not all-beat; MMMU c4 regresses with non-overlapping rounds). Patch: /workspace/rp3-bench/round1_ladder.patch. Rows: mstar-report batch rp3-qwen3vl-loop, commits 42312c3 (baseline) / b807f73 (candidate, `git stash create` object).

## Checkpoint 1 (2026-11-03 07:29-~08:15 UTC): best commit 42312c3e8, all sizes

Run-base cannot serve Qwen3-VL, so only the best commit (role candidate). 3 rounds, sizes interleaved per round, fresh server per arm, /workspace/rp3-bench/checkpoint.sh -> /workspace/results/rp3-bench/ckpt1. 54 runs, 0 failed requests. Rows: mstar-report batch h200_1103, status counted. Mean [min-max] tok/s, TTFT p50 mean:
- 2B: MMMU c1 329.1 [304.7-349.1] 45.4 ms; c4 938.6 [905.5-991.3]; c16 2292.8 [2228.9-2370.2]; ShareGPT c1 396.1 [381.3-423.9] 18.0 ms; c16 3123.0 [2759.5-3745.5]; c64 6934.4 [6494.3-7480.3]
- 4B: MMMU c1 262.5 [260.6-264.2] 49.8 ms; c4 724.3 [711.9-741.6]; c16 1838.6 [1747.5-1945.6]; ShareGPT c1 320.6 [320.0-321.5] 20.4 ms; c16 2729.4 [2588.1-2971.3]; c64 5659.7 [5514.3-5824.1]
- 8B: MMMU c1 187.1 [180.8-194.3] 48.4 ms; c4 487.4 [457.4-508.1]; c16 1467.2 [1390.9-1548.1]; ShareGPT c1 215.8 [206.8-225.1] 21.6 ms; c16 2198.2 [2179.0-2219.2]; c64 5609.3 [5533.5-5699.6]
- test/modular (CPU, CUDA hidden): 58 failed / 6 errors; new failures vs 42cb59991: none. Log: /workspace/results/rp3-logs/modular_cpu_ckpt1_42312c3.log
- Note: 2B/4B are far below their weight-bandwidth bound (2B ShareGPT c16 3123 tok/s vs 8B 2198 with ~1/4 the weights): candidates for a host-bound profile.

### Serving fix (not a perf round): warm the uncaptured prefill_vision / block-loop paths at start-up

- Problem: the first >4096-token image prompt on a fresh server compiles the eager prefill_vision forward on a live request (one Dynamo recompile in the timed MMMU c1 pass; request 30, the 5360-token image, TTFT ~4.2 s in base rounds; MMMU c1 TTFT p99 2.96 s).
- Change: configs/qwen3_vl_{2,4,8}b.yaml add two warmup_requests with 7 and 8 copies of the 608-token Cosmos3 warmup image (~4.3k and ~4.9k tokens; >16384 patches) so both uncaptured paths compile at start-up at two lengths. Snapshot /workspace/rp3-arms/cand-36a78f46e.
- Evidence (8B, MMMU c1 32 requests, fresh server each, alternating, /workspace/rp3-bench/warm_exp.sh): TTFT p99 base 2963.6 / 2962.1 ms vs cand 1136.8 / 1198.5 ms; request 30 TTFT base 4166 / 4190 ms vs cand 1546 / 1631 ms; throughput base 196.0 / 182.6 vs cand 201.3 / 193.8 tok/s (overlapping); TTFT p50 49.2 / 52.0 vs 49.8 / 44.0 ms; start-up 148/138 s vs 151/148 s. TORCH_LOGS=recompiles on the candidate: 279 at start-up, 279 after the live run (0 live recompiles). All 4 warmup requests complete (conductor log warmup-0..3 done).
- Remaining ~1.6 s for request 30 is not Dynamo (no recompiles); on an already-warm base server the same request was ~0.87 s; the other ~0.7 s first-time cost is unattributed.
- Kept as a serving fix rather than an optimization round: it does not change the headline workload (compiles were already paid before MMMU c16 in the round order).
- Review (independent subagent): no blocking issue; counts verified (608 tokens / 2432 patches per image; 7 is the smallest count past both buckets; 64-segment cap not reached; prefix cache cannot shorten a prefill_vision prompt). Applied: name both bucket constants in the comment, YAML anchor for the repeated URL, Dynamo rationale stated as its static-then-dynamic behaviour. Noted: warmup failures only log a warning (existing design); Qwen3.5 configs share the same 4096 / 16384 tops and likely the same first-request stall (other model's lane; not changed).
- Committed c1744fc63 "model: Qwen3-VL warms its uncaptured vision paths at start-up". This is the base for later rounds. Reported as a one-time ~3 s stall removed, not a steady-state speedup.
- Vision block-loop bucket padding (MMMU first 128, per image): 25.7% of captured patches are padding (74.5k of 290.6k), but captured images carry only 1034M of 6155M attention pair-products; the 14 uncaptured big images carry 5121M. Padding est. ~0.5% of the headline. Not pursued.
- Chunked/combined prefill_vision with decode (Omni precedent: split_inputs at qwen3_omni/submodules.py:634-652, combined walk :405-408): needs embeds/positions/deepstack cut per chunk and a mixed capture taking embeds; est. ~3%, below the measured 3v3 noise. Not pursued now.

### Video path check (no code change to the model)

- served_parity.py now takes `--videos` (two prompts each), scored as a subset apart from the text+image gate; commit after c1744fc63. Videos: ffmpeg lavfi testsrc2 448x320 10fps 4s, mandelbrot 640x360 12fps 6s, life 320x320 8fps 3s (/workspace/data/videos).
- 8B (model code == c1744fc63): video subset served 0.9746 (346/355) vs eager 0.9681 (334/345) PASS; gate served 0.9890 (4944/4999) vs eager 0.9888 PASS (image 1007 vs 1014, text 3937 vs 3928 this run; same code gave image 1015 vs 1014 before: near-tie variation).

### Noise floor check (method, no code change)

- Base c1744fc63, 8B, MMMU c16 alone, 4 fresh servers sharing one inductor cache (SHARED_INDUCTOR_CACHE in bench_round.sh; r0 warmed it): 1371.2, 1440.8, 1319.2, 1452.2 tok/s (spread 10%), load average 18-36. Fresh-cache spread in round 1 was 1464-1564 (7%). Kernel selection is not the dominant noise; host load is. Shared cache cut start-up 144 s -> ~103 s.
- Consequence: under the 3v3 all-beat rule only effects of roughly >= 8-10% on MMMU c16 can be kept on this host. Model-lane headline candidates left are ~1-3% each.
- MMMU c16 run first on a fresh server: TTFT p99 3.0-4.9 s despite the large-prompt warmup (checking for batch-size recompiles of the eager prefill_vision step next).
- Fresh server, MMMU c16 first (c1744fc63, TORCH_LOGS=recompiles): 1490.2 tok/s; 279 recompiles at start-up, 1 live: the compiled tail of Qwen3VLLLMSubmodule._forward after the select_last_hidden graph break, re-specialised for a 2-row eager prefill_vision step (small frame). TTFT p99 at c16 is queueing behind large prefills, not compiles.
- 384 requests for MMMU c16 (base c1744fc63, 3 fresh servers, preceded by MMMU c1 n=16 and c4 n=32 as warm shapes): 1791.1, 1770.0, 1770.3 tok/s (spread 1.2%) at load 20-49, vs 7-10% at 128 requests. The 128-request spread was the heavy-tailed drain, not host load. From round 2 on, loop rounds run MMMU c16 with 384 requests (both arms); numbers are not comparable with the 128-request rows (different MMMU subject mix: ~1780 vs ~1500 tok/s).

### Round 2: capture ladder ratio 1.5, retested at 384 MMMU c16 requests -> reverted, dropped

- Hypothesis: as round 1 (padded image-prefill tokens 27% -> 16%); expected MMMU c16 +1-1.5%. Round 1's verdict was taken at 7-10% noise; the 384-request headline measures at ~1.2%.
- Base c1744fc63 (/workspace/rp3-arms/c1744fc63); candidate = base + /workspace/rp3-bench/round1_ladder.patch (/workspace/rp3-arms/cand-bf5b75ab4). Commands: /workspace/rp3-bench/round2.sh (parity 8B, then 3 alternating rounds, six workloads, MMMU c16 n=384), OUTROOT /workspace/results/rp3-bench/round2.
- Parity 8B candidate: gate served 0.9890 (4944/4999) vs eager 0.9888 PASS.
- Results (3 alternating rounds, 8B, mean [min-max] tok/s, TTFT p50 ms): MMMU c1 203.1 [193.5-209.7] / 195.1 [194.3-196.4] (-4.0%), ttft 49.2 / 48.0; MMMU c4 520.2 [500.1-537.9] / 500.2 [458.6-562.4] (-3.8%); MMMU c16 n=384 1753.9 [1647.0-1836.6] / 1760.6 [1696.2-1827.3] (+0.4%, overlapping); ShareGPT c1 218.0 [207.2-224.8] / 207.2 [206.8-207.4]; c16 2189.7 / 2177.3; c64 5429.2 / 5473.8. No non-overlapping regression. Load average 18-42.
- Decision: REVERTED; dropped (inconclusive twice, effect below what this host resolves). Noise: base MMMU c16 at n=384 spread 11% this round under load 18-42; the 1.2% spread of the noise study was a quieter period, not the request count. Rows: rp3-qwen3vl-loop, commits c1744fc (baseline) / bf5b75a (candidate, stash-create object).

### Probe: vision_encoder and LLM in separate workers on one GPU with MPS (config only, not a round)

- Arm /workspace/rp3-arms/split-exp = c1744fc63 with configs/qwen3_vl_8b.yaml: `rank_devices: {0: 0, 1: 0}`, `mps: true`, vision_encoder on rank 0, LLM on rank 1 (the pattern of configs/qwen3_asr_split.yaml; docs/serving.rst:586-592). Mechanism (hypothesis): the compute-bound encoder overlaps the bandwidth-bound decode instead of stalling it.
- 2 alternating rounds, 8B, /workspace/rp3-bench/split_exp.sh: MMMU c16 n=384 base 1596.8 / 1628.8 vs split 1706.0 / 1698.4 tok/s (+5.6%); MMMU c4 n=32 base 648.6 / 663.5 vs split 621.0 / 625.6 (-4.3%); MMMU c1 TTFT p50 base 48.3 / 47.3 vs split 64.2 / 61.0 ms (+30%); c1 throughput 200.3 / 194.7 vs 196.2 / 196.4.
- The c1/c4 cost is the new cross-process edge (vision embeds + 3 deepstack tensors per request) over file SHM: mstar_rust was never built on this pod, so MSTAR_SHM_ARENA could not be set (the benchmarking skill says to benchmark with the arena).
- Finding: every measurement so far ran the pure-Python ZMQ / graph runtime and file SHM; production Linux wheels ship mstar_rust, whose AUTO selection moves workers to the Rust runtime. Building mstar_rust (maturin develop --release -m rust/Cargo.toml; rust/ unchanged vs run-base) and re-measuring the base with MSTAR_SHM_ARENA=1 before any further comparison. Rows before this point are Python-runtime numbers.

### Rust runtime (mstar_rust built, MSTAR_SHM_ARENA=1) from 2026-11-03 ~10:40 UTC

- maturin develop --release -m rust/Cargo.toml (mstar-rust 0.1.1; rust/ unchanged vs run-base); test/rust 570 passed, 107 skipped. Server logs `graph runtime: rust (MSTAR_RUST_GRAPH=AUTO)`. All scripts export MSTAR_SHM_ARENA=1; arm.txt records mstar_rust and the arena flag.
- Parity 8B at c1744fc63 on the Rust runtime: gate served 0.9896 vs eager 0.9888 PASS.
- Base c1744fc63, 2 rounds: MMMU c1 196.0 / 193.8 tok/s, TTFT p50 53.7 / 52.9 ms; c4 699.8 / 668.2; c16 n=384 1839.2 / 1806.4 (Python runtime base in the same session: 1596.8 / 1628.8, so ~+13%).
- Split vision_encoder / LLM workers + MPS on the Rust runtime: MMMU c1 172.7 / 173.4 tok/s, TTFT p50 156.6 / 155.5 ms; c4 537.3 / 570.8; c16 1392.0 / 1358.7 (-25%). REJECTED.
- Checkpoint 1's counted rows (h200_1103) are Python-runtime numbers; checkpoint 2 re-measures the best commit on the Rust runtime.

## Checkpoint 2 (2026-11-03 11:00-~11:50 UTC): best commit c1744fc63 on the Rust runtime, all sizes

/workspace/rp3-bench/checkpoint.sh -> /workspace/results/rp3-bench/ckpt2; 3 rounds, sizes interleaved, fresh server per arm, mstar_rust + MSTAR_SHM_ARENA=1 on all 9 servers; MMMU c16 at 384 requests (not comparable with checkpoint 1's 128). 54 runs, 0 failed requests. Rows: mstar-report batch h200_1103, status counted, commit c1744fc. Mean [min-max] tok/s, TTFT p50 mean:
- 2B: MMMU c1 409.9 [385.4-444.3] 43.2 ms; c4 965.0 [906.5-1014.5]; c16 2949.6 [2856.3-3057.1]; ShareGPT c1 471.0 [445.6-499.7] 16.9 ms; c16 3768.9 [3500.3-3971.0]; c64 7528.3 [7006.5-8241.8]
- 4B: MMMU c1 295.1 [294.5-296.2] 43.5 ms; c4 732.1 [689.7-775.0]; c16 2430.8 [2340.6-2510.4]; ShareGPT c1 321.2 [320.1-322.8] 19.6 ms; c16 3025.8 [2973.8-3066.2]; c64 7394.9 [7034.0-7763.5]
- 8B: MMMU c1 203.8 [193.7-209.1] 49.8 ms; c4 515.6 [472.9-560.1]; c16 1814.6 [1785.6-1864.7]; ShareGPT c1 218.2 [207.3-223.7] 22.8 ms; c16 2200.9 [2184.5-2209.8]; c64 5801.6 [5657.5-5887.0]
- test/modular (CPU, CUDA hidden) at b79898b6e: 90 failed / 6 errors. New vs 42cb59991: 32 in test_loop_output_ownership.py, which now run because mstar_rust is installed; the same 32 fail at run-base e8826a777 with the extension (AttributeError at test_loop_output_ownership.py:116): pre-existing. No other new failure.
- Rust runtime, 8B MMMU c16 n=384 GPU split (scratch timing tree with c1744fc63 model code): decode replay 70.7% (6647 steps, 102076 rows, 5.54 ms/step), vision_encoder 13.3% (204 steps, 33.8 ms), prefill_vision replay 11.7% (191 steps, 31.8 ms), prefill_vision eager 4.3% (18 steps, 125.7 ms). Upper bound for mixing prefill_vision with decode: 209 displaced decode steps x 5.54 ms = 1.16 s of 52.0 s (2.2%), under the 4.4% checkpoint-2 spread. Not pursued.
- Hypothesis "CUDA-core decode attention (decode_tensor_cores=False) beats tensor-core for 8B's GQA 4" -> REJECTED by microbenchmark (/workspace/rp3-bench/decode_attn_bench.py, FlashInfer BatchDecodeWithPagedKVCacheWrapper, 32/8 heads, d128, page 128, bf16): bs1 ctx1200 23.7 vs 21.2 us; bs16 ctx1200 29.3 vs 38.5; bs16 ctx2500 48.0 vs 67.7; bs64 ctx600 48.5 vs 79.2 (tensor-core vs CUDA-core). Current declaration stays. (decode_fmha=True has no trtllm-gen kernel for this GPU/geometry: the profile shows FlashInfer's paged kernel with split-KV merges.)
- Loop status: no model-lane candidate left with an expected headline effect above the measured 3v3 noise (4.4% at checkpoint 2). Remaining items are each <=2.2% (mixing prefill_vision with decode: bound measured), ~1.3% (capture ladder, dropped), or TTFT-only (opaque RGBA fast path), which the keep rule cannot keep. Continuing with checkpoints, correctness coverage and core proposals.

### Full-branch review (independent subagent, 2026-11-03 ~12:20 UTC) and fixes

- Fixed (commits dbecfdd..c525173): reuse Qwen3.5's metadata resolver (tokenizer_config.json already carries the chat template on all three checkpoints; verified); refuse tensor parallelism at load (deepstack / 3D rotary validated on one rank only); text rotary frequencies as an fp32 buffer restored in _apply (was a lazily moved plain attribute: a one-off synchronous copy on the first 3D step); docs wording (the fused 1D kernel is the attention resource's); deleted the obsolete skipped vision-tower test instead of skipping it and corrected FrameSDPA's docstring; new tests driving the served code: Qwen3.5's `_vision_inputs` + deepstack rows vs transformers' get_rope_index / embeddings / image rows, and the vision node splitting a two-request batch (both new layout tests fail when the deepstack rows are shifted by one: checked); served parity exits nonzero on a failing video subset.
- Recorded, not changed: (a) two vision tower classes exist (components/qwen_vision.py QwenVisionModel, used by Qwen3-VL as the brief's "shared vision tower", and qwen3_5/components/vision.py Qwen3_5VisionModel, which Kirill's merge with main kept for Qwen3.5); unifying means moving Qwen3.5 onto the shared tower, which changes Qwen3.5's numerics (it rounds its rotary tables to bf16) in another model's package: left to Kirill's PR. (b) `_deepstack_rows` mirrors Qwen3.5's `_vision_inputs` walk (cannot be derived without editing Qwen3.5); its totals check plus the new layout test guard drift. (c) The deepstack loop recomputes the next norm on the 3 deepstack layers of a vision step (removing it changes rounding order; negligible cost). (d) Warmup counts are tied to the two bucket tops, named in the config comment.
- test/modular related files: 51 passed. Parity at the new HEAD queued after the importer rounds.

### Shared-code no-regression check (rule: measure one other importer)

- Shared files changed by this branch after Kirill's commits: components/qwen3_lm.py (`attention_cls` argument, defaulting to Qwen3Attention) and components/qwen_vision.py (QwenVisionModel stages + fp32 `_apply`; Qwen3.5 imports only QwenVisionConfig from it).
- run-base (/workspace/mstar, e8826a777) vs HEAD snapshot b79898b6e, 3 alternating rounds, fresh server each, Rust runtime + arena, /workspace/rp3-bench/importer_round.sh:
  - Qwen3.5-2B MMMU c16 n=384: run-base 2212.8 / 2530.1 / 2259.2 vs HEAD 2377.2 / 2361.6 / 2616.1 tok/s (overlapping, mean +5%); ShareGPT c16: 3218.8 / 3363.3 / 3619.3 vs 4065.7 / 3273.1 / 3448.4 (overlapping). 0 failed.
  - Qwen3-ASR-1.7B LibriSpeech c16 n=128 (benchmark.runner, closed loop): run-base 1186.1 / 1693.0 / 1564.6 vs HEAD 1309.2 / 1398.5 / 1410.5 text tok/s (HEAD range inside run-base's; mean -7%, overlapping); TTFT p50 111 / 76 / 86 vs 106 / 98 / 92 ms. 128/128 completed each.
- Verdict: no regression with non-overlapping rounds on either importer.
- Parity at HEAD c5251738b (8B): gate served 0.9898 (4948/4999) vs eager 0.9888 PASS.
- Parity at HEAD c5251738b with the 3 videos: 2B gate served 0.9854 (4938/5011) vs eager 0.9848 PASS, video subset 373/382 (0.9764) vs 377/382 (0.9869); 4B gate 0.9922 (4976/5015) vs 0.9914 PASS (the earlier 1-position 4B shortfall does not recur), video 368/376 (0.9787) vs 376/384 (0.9792). Video subsets fall short by ~4 and ~1 positions on ~380.
- Video prompts are identical to transformers' (ids incl. timestamps and frame groups, grids, pixel shapes) on all three clips.
- Log-prob fidelity, 2B with the videos (diag_logprob.py + videos; served raw logprob vs fp32 teacher-forced; scratch tree with sampler.enable_logprobs): video prefill served 0.00815 vs hf-bf16 0.01911, video decode 0.01587 vs 0.01719; image prefill 0.02646 vs 0.04068, image decode 0.01178 vs 0.01266; text decode 0.01216 vs 0.01313; text prefill 0.01947 vs 0.01612 (n=64). Served is closer to fp32 than transformers bf16 on video; the video top-1 shortfall is near-tie variation on a small sample.
- Back-to-back headline noise, Rust runtime, HEAD c5251738b, 3 fresh servers (MMMU c1 n=16, c4 n=32, then c16 n=384): 1659.0, 1709.2, 1697.0 tok/s (3%), load 24-54. Mixing prefill_vision with decode (<= 2.2%) stays parked. These are ~7% below checkpoint 2's c1744fc63 mean (1814.6, different warm sequence); A/B c1744fc63 vs c5251738b queued to rule out a regression from the review fixes.
- A/B c1744fc63 (base) vs HEAD c5251738b, 3 alternating rounds, MMMU c1 n=16 / c4 n=32 / c16 n=384: c1 205.5 [200.3-215.6] vs 207.4 [201.2-218.8] tok/s, TTFT p50 60.4 vs 53.8 ms; c4 660.6 [650.8-675.4] vs 649.4 [646.7-652.3]; c16 1786.2 [1762.4-1832.4] vs 1732.3 [1680.1-1786.6] (-3.0%, overlapping). No non-overlapping regression; the review-fix commits have no runtime path (fp32 buffer moved at load, load-time TP check, metadata helper, docs/tests). The earlier ~7% gap to checkpoint 2 was session drift (c1744fc63 measures 1762-1832 here).
- Best commit is now HEAD c5251738b; checkpoint 3 re-measures it.

## Checkpoint 3 (2026-11-03 13:55-~14:45 UTC): best commit HEAD c5251738b, Rust runtime, all sizes

/workspace/rp3-bench/checkpoint.sh -> /workspace/results/rp3-bench/ckpt3; 3 rounds, sizes interleaved; 54 runs, 0 failed. Rows: mstar-report batch h200_1103, status counted, commit c525173. Mean [min-max] tok/s, TTFT p50 mean:
- 2B: MMMU c1 450.5 [423.9-480.3] 41.2 ms; c4 1011.3 [829.1-1146.6]; c16 3154.4 [3086.0-3280.4]; ShareGPT c1 455.3 [425.7-472.3] 17.8 ms; c16 3840.7 [3529.7-4138.8]; c64 8104.0 [7243.6-8914.3]
- 4B: MMMU c1 294.2 [293.7-294.9] 43.6 ms; c4 755.1 [708.1-778.7]; c16 2445.4 [2404.0-2490.8]; ShareGPT c1 322.2 [322.0-322.4] 17.5 ms; c16 3033.9 [2894.0-3108.6]; c64 7187.6 [6961.7-7486.6]
- 8B: MMMU c1 208.8 [203.8-212.9] 47.0 ms; c4 529.5 [440.5-574.8]; c16 1845.7 [1835.4-1857.8]; ShareGPT c1 221.2 [217.4-224.9] 20.5 ms; c16 2222.2 [2212.3-2227.2]; c64 5979.2 [5923.6-6063.3]
- test/modular (CPU) at c5251738b: 90 failed / 6 errors, identical failing set to checkpoint 2 (the 32 Rust-enabled test_loop_output_ownership failures are pre-existing at run-base).

### Probe for round 3 design: eager vs captured prefill_vision (scratch arm, not a candidate)

- Question: can mixed (prefill_vision + decode) steps run eager without losing the captured prefill's speed? If eager prefill_vision costs ~nothing at MMMU sizes, round 3 can route vision-holding mixed steps to eager (cg_key_info) instead of building an Omni-style mixed capture with embeds / 3D tables / deepstack.
- Arm /workspace/rp3-arms/nocap-vision = c5251738b with the prefill_vision PackedAcceleratorGraphConfig dropped; 2 alternating rounds vs c5251738b, MMMU c1 n=16 / c4 n=32 / c16 n=384 (/workspace/rp3-bench/nocap_exp.sh).
- Result (2 alternating rounds, 8B; mean [min-max] tok/s, TTFT p50 ms): MMMU c1 base 213.4 [210.1-216.8], 50.4 ms | eager 199.0 [196.2-201.9], 100.0 ms (TTFT x2; -6.8%, non-overlapping); c4 645.5 / 656.5, TTFT 64.0 / 147.9 ms; c16 n=384 1777.8 [1727.5-1828.1] / 1753.9 [1724.3-1783.6] (-1.3%). Load 19-57.
- Conclusion: eager prefill_vision is launch-bound at MMMU sizes (c1 TTFT doubles) and costs ~1.3% at c16, so routing vision-holding mixed steps to eager would give back most of the <=2.2% mixing bound and hurt TTFT. Only an Omni-style mixed capture (embeds, 3D tables, deepstack as static inputs; qwen3_omni/submodules.py:977-1029) could net a gain, est. 1-1.5% after decode rows run on the prefill kernel and capture batches pad up. Round 3 (mixing) DROPPED before implementation: not resolvable by the 3v3 rule even at checkpoint 3's 1.2% spread.

### Round 4 candidate search (15:10-15:15 UTC): sampler cost -> proposal, no round

- Hypothesis checked: "the sampler (greedy benchmark, enable_repetion_penalty inherited from Qwen3.5) costs a measurable share of each decode step". Kernel profile (400 steps, MMMU c16 n=384, HEAD c5251738b): sampler ~0.14 ms of 5.58 ms per decode step (split softmax with penalty history 0.10 ms, top-k renorm + top-p 0.04 ms). The lever (a greedy fast path) is in the engine's sampler resource: proposal 5. Dropping enable_repetion_penalty from Qwen3-VL's SamplerSpec would remove a feature to save part of 0.10 ms: not done.
- No model-lane round started: the queue holds no candidate whose expected headline effect the keep rule can resolve (see the queue above).

### Round 5 (in progress): capture ladder ratio 1.5, retested on the Rust runtime

- Why again: rounds 1-2 ran on the pure-Python runtime with base MMMU c16 spreads of 7% and 11%; on the Rust runtime checkpoint 3 measured 1835.4-1857.8 (1.2%). Hypothesis unchanged: padded image-prefill tokens 27% -> 16% (prefill_vision is compute-bound and ~16% of GPU time) -> est. +1-1.5% on MMMU c16 8B.
- Base HEAD c5251738b; candidate = base + /workspace/rp3-bench/round1_ladder.patch (/workspace/rp3-arms/cand5-e735afe4b). Commands: /workspace/rp3-bench/round5.sh (parity 8B, then 3 alternating rounds, six workloads, MMMU c16 n=384), OUTROOT /workspace/results/rp3-bench/round5.
- Review: same diff as rounds 1-2, already reviewed by an independent subagent in round 1 (approach accepted; comment shortened per review; recompile headroom measured at 46 of 84 entries).

## Part 2

### Final summary (2026-11-03 ~19:00 ET)

- Where M* stands (checkpoint 1, best commit 360590298, vs vLLM 0.33.0 with its defaults; vLLM was the faster reference on every row of the 1a sweep except 2B ShareGPT c64, where SGLang 0.5.23 led by 1.5%): M* leads at c1 on every size and workload (tok/s 1.02-1.23x, TTFT p50 0.61-0.83x; 2B ShareGPT c1 TTFT 1.13x), at ShareGPT c16 on 4B and 8B (1.07-1.11x), and at MMMU c4 on all sizes (1.02-1.07x). It trails on ShareGPT c64 (2B 0.68, 4B 0.90, 8B 0.93; vs SGLang 2B 0.65), 2B ShareGPT c16 (0.85) and MMMU c16 (2B 0.89, 4B 0.98, 8B 0.93). Against SGLang (1a sweep means) M* is ahead everywhere except ShareGPT c64 on 2B and 4B (0.65, 0.95) and 2B ShareGPT c16 (0.93).
- Most of the MMMU c16 deficit is vLLM's image prefix cache hitting the benchmark's repeated images; on identical work (vLLM cache off) 8B MMMU c16 is ~1.02 and 2B ~0.94.
- Kept: a72867145 "model: qwen3 vit frame interpolation runs in numpy on the calling thread" (+ test 360590298). Removes a 7-25 ms/image GPU-thread stall (torch CPU ops fork-joining over the worker's 128-thread intra-op pool in a 27-CPU quota). Measured: 2B MMMU c16 +11.9%, c4 +12.4%, 8B MMMU c16 +4.4%, c4 +4.2% (all-beat); Qwen3.5-2B MMMU c16 +19.7% (shared code, all-beat). Bit-identical output; parity passes on 2B and 8B.
- Needs the core (proposals file): 4c worker host time per step (ShareGPT c64/c16 gaps; model hooks are ~0.8 of 7.8 ms per step); 7 size the worker's torch intra-op pool from the CPU plan (the cause behind the kept round; affects every model's host-side CPU torch ops); 8 prefix cache for MRoPE image prompts; 6 step_thread_core auto as a default once its c64 TTFT regression is fixed (+7% / +27% throughput, half the spread, but TTFT p50 x5.6 at 2B c64); 5 greedy sampler fast path; 1 OpenAI chat route for qwen3_vl.
- Remaining model-lane items are each at or below the measured 3v3 spread (2.5% at 8B MMMU c16): image-prefill bucket padding (<=2%, inconclusive 3 times in part 1), prefill_vision/decode mixing (<=2.2% bound), remaining image host work (~3% of the 2B pass, now compute, not stalls). Loop stopped.

### Gap table (reference sweep 2026-11-03 11:32-13:35 ET, M* c5251738b, placement off; mean of 3 rounds; ratio = M* / faster reference)

| size | workload | c | M* tok/s | best ref tok/s | tok/s ratio | M* TTFT p50 | best ref TTFT p50 | TTFT ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2B | ShareGPT | 64 | 8018.5 | 12703.5 (SGLang) | 0.631 | 92.0 | 47.7 (vLLM) | 1.93 |
| 2B | MMMU | 16 | 3243.6 | 3998.1 (vLLM) | 0.811 | 130.0 | 128.1 (vLLM) | 1.01 |
| 2B | ShareGPT | 16 | 4072.0 | 4684.4 (vLLM) | 0.869 | 25.3 | 23.1 (vLLM) | 1.10 |
| 8B | MMMU | 16 | 1796.6 | 2054.7 (vLLM) | 0.874 | 187.7 | 138.1 (vLLM) | 1.36 |
| 2B | MMMU | 4 | 1060.5 | 1188.5 (vLLM) | 0.892 | 79.9 | 63.5 (vLLM) | 1.26 |
| 4B | MMMU | 16 | 2406.6 | 2692.1 (vLLM) | 0.894 | 165.3 | 121.6 (vLLM) | 1.36 |
| 8B | MMMU | 4 | 516.1 | 567.3 (vLLM) | 0.910 | 105.0 | 89.9 (vLLM) | 1.17 |
| 4B | ShareGPT | 64 | 7404.4 | 8130.1 (vLLM) | 0.911 | 71.2 | 42.5 (vLLM) | 1.67 |
| 2B | MMMU | 1 | 394.7 | 426.1 (vLLM) | 0.926 | 42.6 | 49.4 (vLLM) | 0.86 |
| 8B | ShareGPT | 64 | 5851.1 | 6290.9 (vLLM) | 0.930 | 63.1 | 47.0 (vLLM) | 1.34 |
| 4B | MMMU | 4 | 731.6 | 782.3 (vLLM) | 0.935 | 78.3 | 78.2 (vLLM) | 1.00 |
| 2B | ShareGPT | 1 | 488.8 | 473.8 (SGLang) | 1.032 | 18.4 | 14.7 (vLLM) | 1.25 |
| 8B | ShareGPT | 16 | 2208.8 | 2085.5 (vLLM) | 1.059 | 28.9 | 30.9 (vLLM) | 0.94 |
| 4B | ShareGPT | 16 | 3035.1 | 2794.8 (vLLM) | 1.086 | 27.7 | 31.5 (vLLM) | 0.88 |
| 8B | MMMU | 1 | 202.6 | 179.3 (vLLM) | 1.130 | 50.7 | 62.4 (SGLang) | 0.81 |
| 8B | ShareGPT | 1 | 214.9 | 189.3 (SGLang) | 1.135 | 22.3 | 30.3 (vLLM) | 0.74 |
| 4B | MMMU | 1 | 294.1 | 255.1 (vLLM) | 1.153 | 45.1 | 61.5 (SGLang) | 0.73 |
| 4B | ShareGPT | 1 | 321.4 | 271.6 (SGLang) | 1.183 | 20.1 | 25.8 (vLLM) | 0.78 |

TTFT reference = the lower-TTFT engine. Per-engine means and ranges: /workspace/rp3-ref/sweep_report.py /workspace/results/rp3-bench/p2_sweep. vLLM is the faster reference everywhere except 2B ShareGPT c64 (SGLang 12703.5 vs vLLM 12512.5) and c1 rows.

### Hypothesis queue (2026-11-03 ~14:30 ET; ordered by expected gain)

Headline switched from 8B MMMU c16 to **2B MMMU c16 output tok/s** (second: 8B MMMU c16): on identical work (vLLM with prefix caching off, gap analysis 1) 8B MMMU c16 is 0.96 (1796.6 vs 1878, within ~2x the 2.5% spread) while 2B MMMU c16 is 0.86 (3243.6 vs 3770); 2B ShareGPT c64 (0.63) is core host overhead (gap analysis 2).

1. Done (round 1, KEPT a72867145): vision node host grid maths in numpy (2B MMMU c16 +11.9%).
2. Remaining image host work on the GPU thread after round 1: vision prepare_inputs 1.63 ms/image (wall = CPU, compute), LLM _vision_inputs 0.80 ms/image: ~3% of the 2B pass; below the spread.
3. 8B image prefill padding to capture buckets (~27% padded tokens) and the uncaptured >4096-token path: est. <=2% on 8B (rounds 1-2, 5 of part 1 inconclusive).
4. Core (proposals): worker host overhead per step (4b, 4c; 2B ShareGPT c64 0.63-0.68), worker intra-op pool sizing (7), multimodal prefix cache (8), step_thread_core default (6).

### Setup (2026-11-03 10:31 ET start)

- Head c5251738b, run-base e8826a777 (verified). H200, no other GPU process. Round 5 of part 1 (ladder 1.5, Rust runtime) was interrupted after one base round; its diff (identical to /workspace/rp3-bench/round1_ladder.patch) was removed from the tree. Not a result.
- test/modular (CPU, CUDA hidden, -n 8 --dist loadfile, OMP_NUM_THREADS=2) at c5251738b: 90 failed / 6 errors, failing set identical to checkpoint 3 (pre-existing). Log /workspace/results/rp3-logs/modular_cpu_part2_c525173.log.
- Reference engines: vLLM 0.33.0 (/workspace/venv-vllm, torch 2.14.0+cu132, flashinfer 0.7.2.post1, transformers 5.20.2); SGLang 0.5.23 (/workspace/venv-sglang, torch 2.14.1+cu132, sglang-kernel 0.4.10, flashinfer 0.7.2, transformers 5.19.0). Installed with uv; no errors.
- Launch: `vllm serve Qwen/Qwen3-VL-<N>B-Instruct --port 8200 --dtype bfloat16 --max-model-len 32768 --seed 0` and `python -m sglang.launch_server --model-path Qwen/Qwen3-VL-<N>B-Instruct --port 8200 --dtype bfloat16 --context-length 32768 --random-seed 0` (32768 = M*'s max_seq_len, configs/qwen3_vl_8b.yaml:2); everything else engine defaults (no speculative decoding in any engine). Script /workspace/rp3-ref/ref_server.sh.
- Client: /workspace/rp3-bench/qwen3vl_bench.py `--system openai` uses benchmark/runner.py's VLLM_OMNI client (benchmark/request.py:998, streamed /v1/chat/completions, image as data URL); output length from the runner's ignore_eos stamping (benchmark/runner.py:439-462) with max_tokens = the same per-request length M* gets as max_output_tokens. Same request lists, temperature 0, closed loop, one warmup wave. Rounds: /workspace/rp3-ref/ref_round.sh (references), /workspace/rp3-bench/bench_round.sh (M*).
- Thread placement option verified: `knobs: step_thread_core` / MSTAR_STEP_THREAD_CORE (mstar/utils/knobs.py:193), worker pins main/GPU/plan threads (mstar/worker/worker.py:2246-2268), API server result threads (mstar/api_server/entrypoint.py:2724-2746); `auto` places by GPU rank on its NUMA node (mstar/utils/cpus.py module doc). Host: EPYC 9554, 2 NUMA nodes, cgroup quota 27.2 CPUs.

### Same-work check (16 MMMU images + 64 ShareGPT, 8B)

- Commands: /workspace/rp3-ref/ref_parity.sh <engine> 8 (ref_record.py record, then test/qwen3_vl/served_parity.py score); ref_record.py compare against M*'s record /workspace/results/rp3-parity/8b_head-c5251738b.json.
- vLLM 0.33.0: prompt tokens (usage.prompt_tokens) equal transformers' processor on 80/80 prompts, all 16 images (162-5411 tokens). M* uses the same processor with its defaults (mstar/model/qwen3_5/qwen3_5_model.py:582-588) and the same chat template. Teacher-forced gate: vLLM 0.9902 (4950/4999; image 1014/1024) vs hf eager 0.9888 PASS (M* c5251738b: 0.9898). Replies identical to M*'s: 45/80; mean common-prefix fraction text 0.783, image 0.837 (bf16 near-tie divergence; both pass the gate).
- SGLang 0.5.23: prompt tokens equal transformers' on 80/80, all 16 images (same values as vLLM). Gate: 0.9882 (4940/4999; image 1015/1024, text 3925/3975) vs hf eager 0.9888 -> FAIL by 3 positions of 4999, text only. Replies identical to M*'s 44/80; common-prefix text 0.814, image 0.799. Kept in the speed comparison: same prompt tokens and image resolution, and ignore_eos fixes every output length, so the work per request is identical; the shortfall is recorded as near-tie size (part 1's 4B was short by 1 at the same scale).

### 1b. Noise floor: step_thread_core auto vs off (c5251738b)

- Command: /workspace/rp3-bench/noise_place.sh (bench_round.sh with MSTAR_STEP_THREAD_CORE=off|auto, alternating, fresh server each; 8B MMMU c1 n=16, c4 n=32, then c16 n=384; 2B ShareGPT c64 n=256). Server log confirms `auto`: worker step threads on CPUs 64,192 (GPU's NUMA node 1), `off`: not pinned. 0 failed requests.
- 8B MMMU c16: off 1773.7 / 1796.3 / 1819.3 tok/s, mean 1796.4, spread 2.5%, TTFT p50 199.8 ms (203.0/200.7/195.8), p99 2276-2589 | auto 1919.4 / 1916.8 / 1933.6, mean 1923.3 (+7.1%), spread 0.9%, TTFT p50 170.3 ms (171.3/159.0/180.6), p99 1979-2211.
- 2B ShareGPT c64: off 8174.2 / 7938.6 / 9032.0, mean 8381.6, spread 13.0%, TTFT p50 97.4 ms (59.3/150.6/82.3), p99 248-658 | auto 11071.4 / 10403.7 / 10441.4, mean 10638.8 (+26.9%), spread 6.3%, TTFT p50 546.7 ms (410.2/683.9/546.0; 5.6x), p99 899-1094.
- Decision: placement lowers the spread but hurts TTFT (2B c64 p50 x5.6, reproducing rp2's observation), so the brief's condition fails: every part-2 M* measurement runs with placement OFF (the engine default; references are unpinned too). Proposal 6 written. Not a gain of this lane.

### Probe: vision-tower attention kernel, M* vs vLLM (no code change)

- vLLM 0.33.0 runs Qwen3-VL's ViT on FLASH_ATTN (server log: "Using backend AttentionBackendEnum.FLASH_ATTN for vit attention") at native head_dim 72; M* runs FlashInfer ragged prefill at d128 (padded).
- /workspace/rp3-ref/vit_attn_bench.sh (16 heads, non-causal, bf16, ms): [21440] fa3-d72 4.49 / fa2-d72 8.30 / flashinfer-d128 4.97; [8192] 0.70 / 1.41 / 0.72; 4x2432 0.28 / 0.48 / 0.31; 8x1024 0.15 / 0.19 / 0.14; [4096,1200,800,3000] 0.29 / 0.62 / 0.34.
- FA3 at d72 is 0-15% faster than the padded FlashInfer kernel; vision attention is part of the 13% vision-encoder share, so <1% of the headline. Not the gap; not pursued (and FA3 is not a dependency M* has).

### 1a. Reference sweep (2026-11-03 11:32-13:35 ET)

- Command: `MSTAR_STEP_THREAD_CORE=off /workspace/rp3-ref/sweep.sh` (3 rounds; per round sizes 8, 2, 4; engine order rotated per round: M*/vLLM/SGLang, vLLM/SGLang/M*, SGLang/M*/vLLM; fresh server per arm; six workloads each: ShareGPT c1 n=32, c16 n=128, c64 n=256, MMMU c1 n=32, c4 n=64, c16 n=384; one warmup wave each). Output /workspace/results/rp3-bench/p2_sweep; log /workspace/results/rp3-logs/p2_sweep.log. 162 runs, 0 failed requests. 162 rows reported (batch h200_1103, status counted). Load average 15-50 from other tenants.
- Full table: gap table above. M* leads at c1 on every size (tok/s and TTFT, except 2B ShareGPT c1 TTFT) and at ShareGPT c16 on 4B/8B; trails at high concurrency and on every MMMU c4/c16 row.

### Gap analysis 1: MMMU rows compare different work (prefix cache)

- The runner's warmup wave replays the first c requests of the timed list (benchmark/runner.py:418-426), and the sweep's MMMU c1 n=32 / c4 n=64 / c16 n=384 all take MMMU in file order, so in one server the first 64 images repeat. vLLM 0.33 and SGLang 0.5.23 prefix-cache by default, image tokens included (vLLM log "Prefix cache hit rate"; SGLang radix cache), so a repeated image skips encoder and prefill. M* keys only text prompts (PrefixKeys("text_inputs", ("prefill_text",)), mstar/model/qwen3_5/qwen3_5_model.py:660; `keyed_spans=0` for prefill_vision, :641), and core prefix-cache eligibility excludes custom (MRoPE) position ids (.claude/skills/prefix-caching/SKILL.md). MMMU rows therefore credit the references with skipped work M* recomputes; ShareGPT rows give both sides text cache hits.
- vLLM 8B profile, cache on (/workspace/rp3-ref/prof_ref.sh vllm 8; MMMU c16 n=128 after a warm pass of the same 128): prefill-only pass 2.21 s (all hits), full 14.81 s (2212 tok/s), profiled 14.81 s. The profiled pass is decode only (no vision kernels): 2508 sampling steps, kernel busy 14879 ms (5.93 ms/step at ~13 rows): GEMMs ~4.03 ms/step (nvjet 192x16 / 128x16 splitK / 64x16, same family M* runs; lm_head 302 us), FA3 paged decode attention + combine ~1.20 ms/step, fused add+RMSNorm ~0.14 ms/step. M* (part 1 kernel table): GEMMs ~4.0-4.08, decode attention ~0.64-0.74 ms/step, step 5.54-5.58 ms at ~15 rows. M*'s decode step is not slower than vLLM's. /workspace/rp3-ref/kernels.py <trace>.
- Next: vLLM with --disable-prefix-caching --mm-processor-cache-gb 0 on the sweep's MMMU sequence (8B, 2B, 2 rounds) and a cache-off profile (/workspace/rp3-ref/nocache.sh), to measure the gap on identical work.

### Gap analysis 2: identical work, and where the 2B gaps sit

- vLLM cache off (--disable-prefix-caching --mm-processor-cache-gb 0), sweep's MMMU sequence, 2 rounds (/workspace/rp3-ref/nocache.sh, /workspace/results/rp3-bench/p2_nocache): 8B c1 176.0 / 174.9, c4 542.7 / 542.5, c16 1870.8 / 1885.x tok/s (TTFT p50 c16 193 ms); 2B c1 426.9 / 416.4, c4 1160.6 / 1106.0, c16 3882.2 / 3658.x. Against the sweep's M* means: 8B c16 0.96 (M* TTFT p50 187.7 vs 193), c4 0.95; 2B c16 0.86, c4 0.94. The cached sweep numbers overstated the 8B headline gap (0.874).
- 8B profile on identical work (MMMU c16 n=128, both warm, cache off): vLLM prefill-only 7.04 s, full 20.11 s, kernel busy 20.45 s; M* prefill-only 8.64 s, full 23.06 s, GPU 21.57 s (decode replay 13.62 s, 5.68 ms/step; vision encoder 3.87 s; image prefill captured 2.24 s + eager 1.84 s). vLLM image work ~5.6 s (LLM prefill GEMMs ~2.7 s, ViT FA3 attention ~1.74 s, ViT linears ~0.69 s) vs M* ~7.95 s; M*'s decode is ~1.3 s cheaper. Tower kernels are of the same kind and ratio (attention ~half of tower time in both). Commands: /workspace/rp3-ref/prof_ref_nocache.sh vllm 8, /workspace/rp3-bench/prof_mstar.sh 8, /workspace/rp3-ref/kernels.py, vit_split.py.
- worker_phases (MSTAR_PHASE_TIMING=100, /workspace/rp3-bench/profile.sh, /workspace/results/rp3-profile/p2/phases): 2B ShareGPT c64 steady (iters 1000-1600, bs 60.8): iter_total 7.81 ms, event_sync 0.37 (host-bound); core phases gpu_thread.exec 3.04, postprocess_batch 2.32, plan_thread.pre_plan 2.04, engine.commit 0.67, build_speculation 0.77; model phases prepare_inputs 0.23, preprocess 0.23, forward 0.33 ms -> core (proposal 4b). 2B MMMU c16 steady (iters 4300-6500, bs 13.8): iter_total 3.95 (p50 3.05), event_sync 0.23 (host-bound); gpu_thread.prepare_inputs mean 0.30, p95 1.93 ms -> probed below.
- Probe (scratch tree timers, not a candidate): LLM `_vision_inputs` + `_deepstack_rows` 0.63 + 0.10 ms per image row; text_inputs already on cuda:0 (no pageable copy). Hypothesis "pageable text copy in _vision_inputs" REJECTED. Vision node `prepare_inputs` 6.8 ms (run 1) / 25.0 ms (run 2) mean per image; of which `vision_interpolation` 23.7 ms wall / 7.7 ms thread CPU (p50 0.70, p90 25.3, max 535); torch intra-op threads in the worker = 128 (cgroup quota 27.2). Offline: 0.44 ms at 1 thread vs 9.95 ms wall at 128 threads (64x58 grid). -> round 1; core side proposal 7.

### Round 1 (part 2): vision node's frame interpolation in numpy on the calling thread

- Hypothesis: the vision node's `prepare_inputs` holds the GPU thread 7-25 ms per image (2B MMMU c16, measured above) in `_frame_interpolation`'s torch CPU ops fork-joining over a 128-thread intra-op pool in a 27-CPU quota; numpy on the calling thread (as `_tile_frames` already does) removes ~1-3 s of GPU-thread host time per 128 requests. Expected: 2B MMMU c16 +5-15% (host-bound), 8B MMMU c16 ~0-2% (GPU-bound).
- Change: mstar/model/components/qwen3_vit.py `_axis_taps_weights` / `_frame_interpolation` in numpy (shared model code: imported by Qwen3.5, Qwen3-Omni, Qwen3-VL). Bit-identical to the torch version on 918 grids (sides 48/32/6, 1-3 entries, t 1-3, up to 240x240; int64 indices, float32 weights). Offline at 128 threads: 9.95 -> 0.43 ms (64x58), 15.8 -> 1.68 ms (120x160).
- Tests: ruff clean; every test/modular file importing qwen3_vit / qwen3_5.submodules / qwen3_omni.submodules / the vision towers (24 files): 1123 passed, 5 failed = chatterbox test_model_kv_layouts cases already failing at c5251738b.
- Candidate arm /workspace/rp3-arms/r1p2-numpy-interp (c5251738b + the file). Command: /workspace/rp3-bench/p2_round1.sh (parity 2B and 8B on the candidate; 3 alternating rounds x 2B and 8B, six workloads, fresh server each, placement off), OUTROOT /workspace/results/rp3-bench/p2_round1.
- Parity (candidate arm): 2B gate served 0.9856 (4939/5011) vs hf eager 0.9848 PASS; 8B gate 0.9894 (4946/4999) vs 0.9888 PASS.
- Results (3 alternating rounds, fresh server each, placement off; mean [min-max] tok/s, TTFT p50 ms), base c5251738b vs candidate (snapshot a76760a37 = `git stash create`; rows label it r1p2-numpy):
  - 2B MMMU c16 (headline): 3136.0 [3106.8-3167.5], 129.8 | 3509.1 [3304.6-3695.3], 112.4 | +11.9% all-beat
  - 2B MMMU c4: 1046.2 [1016.3-1072.6] | 1175.6 [1133.3-1197.4] | +12.4% all-beat; c1: 422.6 | 419.5 (-0.8%, overlapping)
  - 2B ShareGPT c1 424.0 | 440.4; c16 4089.7 [3597.3-4761.3] | 3657.6 [3407.8-3833.9] (-10.6%, overlapping; text path does not run the changed code); c64 7832.0 | 7632.8 (overlapping)
  - 8B MMMU c16: 1833.1 [1802.1-1862.3], 185.5 | 1913.1 [1905.4-1922.0], 163.2 | +4.4% all-beat; c4 551.9 | 575.0 +4.2% all-beat; c1 200.3 | 197.5 (TTFT p50 51.8 | 45.3)
  - 8B ShareGPT c1 212.9 | 207.2; c16 2205.5 | 2202.4; c64 5896.5 | 5862.8 (all overlapping)
- Shared-code importer check (Qwen3.5-2B, /workspace/rp3-bench/p2_round1_imp.sh -> p2_importer_round.sh, 3 alternating rounds, base c5251738b arm): MMMU c16 n=384 base 2488.1 / 2548.1 / 2208.4 (mean 2414.9) vs cand 2906.0 / 2950.0 / 2818.9 (mean 2891.7, +19.7%, all-beat); ShareGPT c16 base 3528.4 / 3364.3 / 3527.4 vs cand 3271.8 / 3153.0 / 3564.0 (overlapping; candidate r3 at load 50). No non-overlapping regression.
- Review (independent subagent): no rule of section 3 broken; bit-exact confirmed on 10,200 grids; asked for an in-server measurement (above) and two text fixes (double backticks, comment on the float32 cast), applied. It also measured, on an idle host at 8 threads, the 256x256 frame 2.7 ms slower in numpy than torch's threaded version: the served measurement above is the deciding one.
- Rows: mstar-report batch rp3-qwen3vl-loop, 72 rows (2B and 8B, both arms).
- Decision: KEPT. Commits a72867145 (model) and 360590298 (test: the bilinear-reference test also at side 48 with MMMU-like and large grids; float32-coordinate tolerance 5e-5, which the previous torch code needs too). ruff clean; test_qwen3_5_video.py 41 passed.

## Part 2 checkpoint 1 (2026-11-03 ~16:45-18:55 ET): best commit 360590298 vs vLLM 0.33.0 (defaults)

- Command: /workspace/rp3-ref/ckpt.sh (3 rounds, sizes 8/2/4, M*/vLLM order alternated per round, fresh server each, six workloads, placement off). 108 runs, 0 failed. Rows: batch h200_1103, status counted, commit 3605902 (M*, candidate) / vLLM 0.33.0 (baseline). Summary: /workspace/rp3-ref/sweep_report2.py /workspace/results/rp3-bench/p2_ckpt1.
- M* / vLLM tok/s (TTFT p50 ratio): 2B ShareGPT c64 0.675 (1.31), c16 0.848 (1.05), c1 1.021 (1.13); 2B MMMU c16 0.888 (0.81), c4 1.024 (1.22), c1 1.072 (0.83); 4B ShareGPT c64 0.896 (1.78), c16 1.107 (0.81), c1 1.225 (0.77); 4B MMMU c16 0.978 (0.82), c4 1.068 (1.01), c1 1.198 (0.68); 8B ShareGPT c64 0.931 (1.34), c16 1.070 (0.89), c1 1.183 (0.70); 8B MMMU c16 0.929 (1.26), c4 1.025 (1.08), c1 1.156 (0.61).
- M* means: 2B MMMU c16 3540.2 [3455.4-3584.0] (sweep at c5251738b: 3243.6), c4 1211.7 (1060.5); 4B MMMU c16 2611.1 (2406.6), c4 825.0 (731.6); 8B MMMU c16 1911.5 [1902.6-1918.3] (1796.6), c4 580.7 (516.1). ShareGPT rows unchanged within spread.
- vLLM's MMMU rows include its image prefix-cache hits on repeated images (gap analysis 1); on identical work (cache off) vLLM measured 8B c16 1870.8 / 1885 and 2B c16 3882 / 3658 tok/s, i.e. M* 8B ~1.02, 2B ~0.94.
- test/modular (CPU, CUDA hidden, -n 8 --dist loadfile, run during the GPU sweep): 93 failed / 6 errors; vs c5251738b's set, 3 extra: test_eval_compare_multigpu.py (2 hang-cap tests) and test_eval_compare_transport.py (1). All 3 pass when rerun alone at HEAD (230 s; host loaded by the sweep); neither file imports a touched module. Load-induced timeouts, not regressions. Log /workspace/results/rp3-logs/modular_cpu_p2ckpt1_360590298.log.

## Part 3

### A5. Gap table (M* main 9c6f1ad, auto = not pinned on this host; 3 rounds; batch h200_1103)

| size | workload | c | M* tok/s | best ref | tok/s ratio | M* TTFT | best ref TTFT | TTFT ratio | M*/run-base tok/s | TTFT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8b | random | 16 | 5561 | 8301 (sglang) | 0.670 | 67.2 | 97.0 (sglang) | 0.69 | - | - |
| 0.8b | random | 64 | 10509 | 17582 (sglang) | 0.598 | 91.9 | 211.1 (sglang) | 0.44 | 1.004 | 1.01 |
| 0.8b | sharegpt | 1 | 475 | 684 (sglang) | 0.695 | 18.4 | 23.1 (vllm) | 0.80 | - | - |
| 0.8b | sharegpt | 16 | 4375 | 5866 (sglang) | 0.746 | 23.3 | 34.8 (sglang) | 0.67 | - | - |
| 0.8b | sharegpt | 64 | 8044 | 11368 (sglang) | 0.708 | 36.9 | 48.4 (sglang) | 0.76 | 1.177 | 0.88 |
| 2b | random | 16 | 5567 | 5984 (sglang) | 0.930 | 83.6 | 124.5 (sglang) | 0.67 | - | - |
| 2b | random | 64 | 11674 | 13257 (sglang) | 0.881 | 94.6 | 276.5 (vllm) | 0.34 | 1.218 | 1.05 |
| 2b | sharegpt | 1 | 452 | 538 (sglang) | 0.840 | 18.7 | 25.3 (vllm) | 0.74 | - | - |
| 2b | sharegpt | 16 | 4764 | 5011 (sglang) | 0.951 | 23.0 | 33.7 (sglang) | 0.68 | - | - |
| 2b | sharegpt | 64 | 8587 | 9681 (sglang) | 0.887 | 35.8 | 46.5 (sglang) | 0.77 | 1.215 | 0.90 |

Same engines, M* at `step_thread_core siblings` (grid, 3 rounds, rows rp3-main-loop) over the same faster reference: 0.8B ShareGPT c1 885/684 = 1.29, c16 8108/5866 = 1.38, c64 15188/11368 = 1.34; Random c16 7999/8301 = 0.96, c64 15853/17582 = 0.90. 2B ShareGPT c1 620/538 = 1.15, c16 6308/5011 = 1.26, c64 12866/9681 = 1.33; Random c16 5992/5984 = 1.00, c64 13325/13257 = 1.01. TTFT p50 over the lower reference: 0.8B 0.64 (c1), 0.49 (c16), 0.54 (c64), Random 0.75, 0.50; 2B 0.64, 0.53, 0.63, Random 0.72, 0.45.

Time per step at c64 from tok/s (64 / tok/s): 0.8B ShareGPT M* 7.96, vLLM 6.96, SGLang 5.63 ms; Random 6.09, 5.19, 3.64. 2B ShareGPT 7.45, 6.92, 6.61; Random 5.48, 5.43, 4.83. M* at siblings: 0.8B ShareGPT 4.21, Random 4.04; 2B 4.97, 4.80. A3 below is with phase timing on, which lowers tok/s by ~20% (0.8B ShareGPT c64 6243 vs 8044).

A3 phases, longest steady segment, ms per iteration (main vs run-base; both unpinned). Worker CPU over the three threads is about 100% of wall (iter_total.cpu + gpu_thread + plan_thread .cpu), event_sync 0.2-0.5 ms: host-bound. Full phase files: results/rp3-main/a3/<arm>/<size>_<wl>_c64/phases.txt.


0.8b_sharegpt_c64: main iters 1000-2700 bs 62.29 | run-base iters 1000-2700 bs 62.21
| phase | main ms | main n | run-base ms | run-base n |
| --- | --- | --- | --- | --- |
| iter_total | 8.666 | 1800 | 8.776 | 1800 |
| iter_total.cpu | 4.116 | 1800 | 3.673 | 1800 |
| worker.postprocess.event_sync | 0.284 | 1800 | 0.187 | 1800 |
| worker.gpu_thread.total | 5.567 | 1800 | 6.540 | 1798 |
| worker.gpu_thread.exec | 4.120 | 1801 | 4.440 | 1799 |
| worker.gpu_thread.prepare_inputs | 0.279 | 1800 | 0.278 | 1800 |
| worker.gpu_thread.stage_host_rows | 0.656 | 1800 | 0.727 | 1798 |
| worker.gpu_thread.await_plan | 0.239 | 1799 | 0.847 | 1799 |
| worker.plan_thread.pre_plan | 2.015 | 1333 | 2.488 | 1337 |
| worker.plan_thread.await_commit | 0.070 | 1799 | 0.072 | 1799 |
| worker.postprocess_batch | 2.309 | 1800 | 2.096 | 1800 |
| worker.postprocess.runtime | 0.778 | 1800 | 0.749 | 1800 |
| worker.postprocess.store_tensors | 0.753 | 467 | 0.727 | 462 |
| worker.postprocess.send_prep | 0.363 | 800 | 0.348 | 782 |
| worker.postprocess.check_stop | 0.269 | 467 | 0.247 | 462 |
| worker.postprocess.prematerialize | 0.449 | 1800 | 0.422 | 1800 |
| rust.postprocess.nogil | 0.403 | 1800 | 0.402 | 1800 |
| worker.main.build_speculation | 0.941 | 1800 | 0.837 | 1800 |
| worker.main.process_messages | 0.870 | 1800 | 0.856 | 1800 |
| speculate | 0.857 | 1799 | 0.760 | 1799 |
| submit_spec | 2.654 | 1799 | 3.382 | 1799 |
| await_gpu | 0.345 | 1800 | 0.535 | 1800 |
| engine.commit | 0.929 | 1800 | 0.975 | 1800 |
| engine.admit | 0.159 | 1800 | 0.174 | 1800 |
| engine.pre_admit | 0.313 | 1333 | 0.376 | 1337 |
| engine.preprocess | 0.510 | 1801 | 0.471 | 1800 |
| engine.forward | 0.404 | 1801 | 0.345 | 1800 |
| engine.forward.replay | 0.250 | 1801 | 0.234 | 1800 |
| engine.collect_outputs | 0.413 | 1800 | 0.360 | 1800 |
| engine.plan.promoted | 0.189 | 1333 | 0.217 | 1337 |
| engine.plan.fresh | 3.607 | 467 | 4.731 | 463 |
| res.plan_fresh.linear_attn | 1.133 | 467 | 1.469 | 462 |
| res.plan_fresh.attn | 1.118 | 467 | 1.174 | 463 |
| res.plan_fresh.kv_cache | 0.359 | 467 | 0.880 | 463 |
| res.plan_fresh.rope | 0.308 | 467 | 0.381 | 463 |
| res.plan_fresh.sampler | 0.347 | 467 | 0.402 | 463 |
| res.plan_fresh.gdn_state | 0.284 | 467 | 0.363 | 463 |
| res.pre_plan.kv_cache | 0.403 | 1333 | 0.810 | 1337 |
| res.pre_plan.attn | 0.574 | 1333 | 0.175 | 1337 |
| res.pre_plan.resident | 0.321 | 1333 | 0.000 | 0 |
| res.commit.gdn_state | 0.434 | 1800 | 0.409 | 1800 |
| res.commit.kv_cache | 0.339 | 1800 | 0.437 | 1800 |
| gpu_exec | 5.523 | 1795 | 5.741 | 1800 |

0.8b_random_c64: main iters 500-1800 bs 62.82 | run-base iters 500-1800 bs 62.99
| phase | main ms | main n | run-base ms | run-base n |
| --- | --- | --- | --- | --- |
| iter_total | 7.250 | 1400 | 7.182 | 1400 |
| iter_total.cpu | 3.357 | 1400 | 3.151 | 1400 |
| worker.postprocess.event_sync | 0.209 | 1400 | 0.160 | 1400 |
| worker.gpu_thread.total | 5.102 | 1399 | 5.575 | 1396 |
| worker.gpu_thread.exec | 3.452 | 1400 | 3.524 | 1399 |
| worker.gpu_thread.prepare_inputs | 0.144 | 1400 | 0.131 | 1400 |
| worker.gpu_thread.stage_host_rows | 0.887 | 1399 | 1.001 | 1396 |
| worker.gpu_thread.await_plan | 0.340 | 1399 | 0.673 | 1399 |
| worker.plan_thread.pre_plan | 1.579 | 1226 | 1.803 | 1224 |
| worker.plan_thread.await_commit | 0.025 | 1399 | 0.018 | 1399 |
| worker.postprocess_batch | 2.123 | 1400 | 1.972 | 1400 |
| worker.postprocess.runtime | 0.764 | 1400 | 0.745 | 1400 |
| worker.postprocess.store_tensors | 0.738 | 173 | 0.603 | 175 |
| worker.postprocess.send_prep | 0.725 | 197 | 0.733 | 195 |
| worker.postprocess.check_stop | 0.191 | 173 | 0.176 | 175 |
| worker.postprocess.prematerialize | 0.458 | 1400 | 0.432 | 1400 |
| rust.postprocess.nogil | 0.547 | 1400 | 0.547 | 1400 |
| worker.main.build_speculation | 0.750 | 1400 | 0.704 | 1400 |
| worker.main.process_messages | 0.675 | 1400 | 0.685 | 1400 |
| speculate | 0.680 | 1399 | 0.635 | 1399 |
| submit_spec | 2.280 | 1399 | 2.464 | 1399 |
| await_gpu | 0.206 | 1400 | 0.308 | 1400 |
| engine.commit | 0.951 | 1400 | 0.973 | 1400 |
| engine.admit | 0.137 | 1400 | 0.143 | 1400 |
| engine.pre_admit | 0.311 | 1226 | 0.348 | 1224 |
| engine.preprocess | 0.267 | 1400 | 0.210 | 1400 |
| engine.forward | 0.476 | 1400 | 0.449 | 1400 |
| engine.forward.replay | 0.347 | 1400 | 0.341 | 1400 |
| engine.collect_outputs | 0.106 | 1400 | 0.101 | 1399 |
| engine.plan.promoted | 0.210 | 1226 | 0.209 | 1224 |
| engine.plan.fresh | 6.555 | 173 | 7.578 | 176 |
| res.plan_fresh.linear_attn | 3.145 | 174 | 2.580 | 176 |
| res.plan_fresh.attn | 1.575 | 174 | 2.672 | 176 |
| res.plan_fresh.kv_cache | 0.444 | 174 | 0.973 | 176 |
| res.plan_fresh.rope | 0.546 | 173 | 0.415 | 176 |
| res.plan_fresh.sampler | 0.441 | 174 | 0.479 | 176 |
| res.plan_fresh.gdn_state | 0.336 | 174 | 0.404 | 176 |
| res.pre_plan.kv_cache | 0.255 | 1226 | 0.659 | 1224 |
| res.pre_plan.attn | 0.475 | 1226 | 0.188 | 1224 |
| res.pre_plan.resident | 0.158 | 1226 | 0.000 | 0 |
| res.commit.gdn_state | 0.325 | 1400 | 0.324 | 1400 |
| res.commit.kv_cache | 0.463 | 1400 | 0.520 | 1400 |
| gpu_exec | 5.448 | 1398 | 5.563 | 1398 |

2b_sharegpt_c64: main iters 1000-2700 bs 62.33 | run-base iters 1000-2700 bs 62.3
| phase | main ms | main n | run-base ms | run-base n |
| --- | --- | --- | --- | --- |
| iter_total | 8.956 | 1800 | 10.082 | 1800 |
| iter_total.cpu | 4.144 | 1800 | 4.322 | 1800 |
| worker.postprocess.event_sync | 0.305 | 1800 | 0.362 | 1800 |
| worker.gpu_thread.total | 6.217 | 1800 | 7.160 | 1800 |
| worker.gpu_thread.exec | 4.585 | 1800 | 4.921 | 1799 |
| worker.gpu_thread.prepare_inputs | 0.309 | 1800 | 0.324 | 1800 |
| worker.gpu_thread.stage_host_rows | 0.766 | 1800 | 0.700 | 1800 |
| worker.gpu_thread.await_plan | 0.242 | 1799 | 0.936 | 1799 |
| worker.plan_thread.pre_plan | 1.948 | 1329 | 2.945 | 1326 |
| worker.plan_thread.await_commit | 0.080 | 1799 | 0.083 | 1799 |
| worker.postprocess_batch | 2.410 | 1800 | 2.540 | 1800 |
| worker.postprocess.runtime | 0.882 | 1800 | 0.809 | 1800 |
| worker.postprocess.store_tensors | 0.704 | 470 | 0.740 | 474 |
| worker.postprocess.send_prep | 0.361 | 820 | 0.379 | 832 |
| worker.postprocess.check_stop | 0.245 | 470 | 0.251 | 474 |
| worker.postprocess.prematerialize | 0.429 | 1800 | 0.556 | 1800 |
| rust.postprocess.nogil | 0.473 | 1800 | 0.426 | 1800 |
| worker.main.build_speculation | 0.905 | 1800 | 0.896 | 1800 |
| worker.main.process_messages | 0.881 | 1800 | 0.960 | 1800 |
| speculate | 0.822 | 1799 | 0.813 | 1799 |
| submit_spec | 2.816 | 1799 | 3.632 | 1799 |
| await_gpu | 0.384 | 1800 | 0.474 | 1800 |
| engine.commit | 0.984 | 1800 | 1.065 | 1801 |
| engine.admit | 0.174 | 1800 | 0.205 | 1800 |
| engine.pre_admit | 0.319 | 1329 | 0.440 | 1326 |
| engine.preprocess | 0.573 | 1799 | 0.387 | 1801 |
| engine.forward | 0.460 | 1800 | 0.417 | 1801 |
| engine.forward.replay | 0.290 | 1800 | 0.291 | 1801 |
| engine.collect_outputs | 0.423 | 1800 | 0.425 | 1801 |
| engine.plan.promoted | 0.213 | 1329 | 0.225 | 1326 |
| engine.plan.fresh | 4.260 | 471 | 5.725 | 474 |
| res.plan_fresh.linear_attn | 1.365 | 471 | 1.998 | 473 |
| res.plan_fresh.attn | 1.405 | 471 | 1.600 | 474 |
| res.plan_fresh.kv_cache | 0.351 | 471 | 0.918 | 474 |
| res.plan_fresh.rope | 0.342 | 471 | 0.298 | 474 |
| res.plan_fresh.sampler | 0.416 | 471 | 0.456 | 474 |
| res.plan_fresh.gdn_state | 0.315 | 471 | 0.378 | 474 |
| res.pre_plan.kv_cache | 0.234 | 1329 | 0.961 | 1326 |
| res.pre_plan.attn | 0.633 | 1329 | 0.215 | 1326 |
| res.pre_plan.resident | 0.334 | 1329 | 0.000 | 0 |
| res.commit.gdn_state | 0.432 | 1800 | 0.473 | 1801 |
| res.commit.kv_cache | 0.378 | 1800 | 0.454 | 1801 |
| gpu_exec | 6.423 | 1799 | 6.659 | 1800 |

2b_random_c64: main iters 500-1900 bs 62.21 | run-base iters 500-1900 bs 62.23
| phase | main ms | main n | run-base ms | run-base n |
| --- | --- | --- | --- | --- |
| iter_total | 6.866 | 1500 | 6.908 | 1500 |
| iter_total.cpu | 3.217 | 1500 | 3.120 | 1500 |
| worker.postprocess.event_sync | 0.467 | 1500 | 0.412 | 1500 |
| worker.gpu_thread.total | 4.728 | 1498 | 4.685 | 1497 |
| worker.gpu_thread.exec | 3.051 | 1500 | 2.907 | 1499 |
| worker.gpu_thread.prepare_inputs | 0.125 | 1500 | 0.116 | 1500 |
| worker.gpu_thread.stage_host_rows | 0.927 | 1498 | 0.831 | 1497 |
| worker.gpu_thread.await_plan | 0.352 | 1499 | 0.606 | 1499 |
| worker.plan_thread.pre_plan | 1.483 | 1324 | 1.890 | 1326 |
| worker.plan_thread.await_commit | 0.020 | 1499 | 0.017 | 1499 |
| worker.postprocess_batch | 2.187 | 1500 | 2.084 | 1500 |
| worker.postprocess.runtime | 0.713 | 1500 | 0.663 | 1500 |
| worker.postprocess.store_tensors | 0.596 | 175 | 0.581 | 173 |
| worker.postprocess.send_prep | 0.495 | 225 | 0.503 | 221 |
| worker.postprocess.check_stop | 0.158 | 175 | 0.173 | 173 |
| worker.postprocess.prematerialize | 0.422 | 1500 | 0.419 | 1500 |
| rust.postprocess.nogil | 0.520 | 1500 | 0.479 | 1500 |
| worker.main.build_speculation | 0.688 | 1500 | 0.683 | 1500 |
| worker.main.process_messages | 0.507 | 1500 | 0.523 | 1500 |
| speculate | 0.622 | 1499 | 0.618 | 1499 |
| submit_spec | 2.283 | 1499 | 2.495 | 1499 |
| await_gpu | 0.307 | 1500 | 0.289 | 1500 |
| engine.commit | 0.910 | 1500 | 0.887 | 1500 |
| engine.admit | 0.139 | 1500 | 0.127 | 1500 |
| engine.pre_admit | 0.302 | 1324 | 0.337 | 1326 |
| engine.preprocess | 0.256 | 1500 | 0.248 | 1500 |
| engine.forward | 0.427 | 1500 | 0.368 | 1500 |
| engine.forward.replay | 0.285 | 1500 | 0.266 | 1500 |
| engine.collect_outputs | 0.089 | 1500 | 0.083 | 1500 |
| engine.plan.promoted | 0.192 | 1324 | 0.200 | 1326 |
| engine.plan.fresh | 4.960 | 176 | 5.005 | 174 |
| res.plan_fresh.linear_attn | 1.567 | 176 | 1.355 | 174 |
| res.plan_fresh.attn | 1.666 | 176 | 1.538 | 174 |
| res.plan_fresh.kv_cache | 0.405 | 176 | 0.853 | 174 |
| res.plan_fresh.rope | 0.490 | 176 | 0.409 | 174 |
| res.plan_fresh.sampler | 0.426 | 176 | 0.439 | 174 |
| res.plan_fresh.gdn_state | 0.341 | 176 | 0.362 | 174 |
| res.pre_plan.kv_cache | 0.180 | 1324 | 0.817 | 1326 |
| res.pre_plan.attn | 0.467 | 1324 | 0.176 | 1326 |
| res.pre_plan.resident | 0.152 | 1324 | 0.000 | 0 |
| res.commit.gdn_state | 0.313 | 1500 | 0.303 | 1500 |
| res.commit.kv_cache | 0.434 | 1500 | 0.460 | 1500 |
| gpu_exec | 5.574 | 1494 | 5.437 | 1495 |


Appendix A re-ranked, measured ms per step x share of steps on main (0.8B, ShareGPT c64 / Random c64; A3 is unpinned):

| # | item | main ms x share = ms per step | run-base | change |
| --- | --- | --- | --- | --- |
| 9 | thread placement: idle exit on every handoff | c64 step 7.96 -> 4.21 ms at siblings (all steps), measured across rounds | same class | unchanged by main; removed by placement |
| 2 | output postprocessing | postprocess_batch 2.31 / 2.12 (all steps) | 2.10 / 1.97 | unchanged (slightly larger) |
| 3 | build_speculation + engine.commit | 0.94 + 0.93 / 0.75 + 0.95 | 0.84 + 0.98 / 0.70 + 0.97 | unchanged |
| 10 | per-step planning | pre_plan 2.02 x 74% = 1.49 / 1.58 x 88% = 1.39 | 2.49 x 74% = 1.84 / 1.80 x 87% = 1.57 | smaller |
| 1 | fresh plans, inline | 3.61 x 26% = 0.94 / 6.56 x 12.4% = 0.81 | 4.73 x 26% = 1.22 / 7.58 x 12.6% = 0.95 | smaller, still inline |
| 12 | API intake | process_messages 0.87 / 0.68; Python frontend c64 TTFT p50 69 ms off, 476 ms at siblings, Rust 37 / 26 | 0.86 / 0.69 | unchanged on the worker; Python saturates at ~12k tok/s |
| 11 | graph exec on GPU thread (decode steps) | exec 2.14 (cpu 1.70) x 74% = 1.58 | not split on run-base | overall exec 4.12 vs 4.44: smaller |
| 7 | all-greedy sampler | res.plan.sampler 0.06 + pre_plan 0.05 host; GPU not re-measured | - | not measured on GPU |
| 4, 5, 6, 8, 13 | intra-op pool, image decode, 9B prefill padding, MoE batching, conductor hops | not on these workloads (text only, one GPU, no MoE, no 9B yet) | - | not measured |

### Setup (2026-11-04 01:28 UTC start)

- Worktree /workspace/wt-rp3-main, branch perf/rp3-main, pinned 9c6f1ad757. venv /workspace/venv-main (torch 2.12.1+cu132, flashinfer 0.7.2), mstar_rust 0.1.1 + mstar-server built from the tree. Each measured commit runs from a git archive under /workspace/rp3-arms/ with PYTHONPATH; server log confirms the arm's yaml and the extension path.
- Env: venv-main pulled torchaudio 2.11.1+cu132, ABI-incompatible with torch 2.12.1 (no 2.12 build on the index); uninstalled, as part 1 did for /workspace/venv. test/modular's audio failures below come from that.
- run-base arm: /workspace/venv imports /workspace/mstar (HEAD e8826a777), so it runs run-base itself; rust/ is unchanged between run-base and perf/rp3-qwen3vl. No mstar-server existed: built from /workspace/mstar/rust/server with CARGO_TARGET_DIR=/workspace/rp3-main/target-runbase (tree untouched). run-base has --rust-frontend and MSTAR_SHM_ARENA; its step_thread_core default differs (recorded in each row's config).
- Qwen3.5-0.8B weights downloaded (were not cached).
- Host: EPYC 9554 x2 (256 CPUs in mask), cgroup quota 27.2 CPUs, H200 on NUMA node 1, idle states POLL 0 / C1 1 / C2 800 us, governor schedutil. auto's log line: "API server: result threads not pinned (not pinned: auto is measured on Intel hosts only and this host's is AuthenticAMD (step_thread_core always pins anyway))". Load average 20-30 from other tenants.
- Client: /workspace/rp3-main/bench.py: benchmark/runner.py's Benchmark (closed loop, ignore_eos, temperature 0, Qwen3_5_Dense kwargs: enable_thinking on for all engines), OurSystem /generate for M*, VLLMOmni client for references, always venv-main. One untimed wave at the workload's concurrency from a disjoint slice; timed prompts never repeat in one server, so no timed request is a prefix-cache hit on any engine. ShareGPT (vLLM sampling): c1 n=32, c16 n=256, c64 n=640. Random 1024/256 (random ids decoded, re-encoding to exactly 1024): c16 n=128, c64 n=384. Records client CPU share; 0.4-0.5 at c64.
- Compile level: M* at its default -O1 (2-4x shorter cold start). The benchmarking skill asks for -O2 for numbers quoted against another engine; checked separately below.
- test/modular at 9c6f1ad (CPU, CUDA hidden, -n 8 --dist loadfile, OMP_NUM_THREADS=2): 64 failed, 6 errors, 10349 passed, 982 skipped; set in rp3-logs/modular_main_9c6f1ad.set (audio files need torchaudio; startup_graph_validation, send_registration_failure, eval_compare*, etc.).

### A1. Same-work check (2026-11-04, /workspace/rp3-main/a1.sh, a1b.sh, parity.py; results/rp3-main/parity)

- Launch: `vllm serve Qwen/Qwen3.5-<N>B --port 8200 --dtype bfloat16 --max-model-len 32768 --seed 0`, `python -m sglang.launch_server --model-path Qwen/Qwen3.5-<N>B --port 8200 --dtype bfloat16 --context-length 32768 --random-seed 0 --mamba-ssm-dtype float32` (M*'s and vLLM's GDN state is the checkpoint's float32; SGLang needs the flag, benchmarking skill). Prefix cache on in all three. No speculative decoding. Both engines served both sizes. vLLM 0.33.0 (FLASH_ATTN), SGLang 0.5.23 (fa3).
- Prompt tokens: vLLM and SGLang usage.prompt_tokens equal on all prompts; equal transformers' apply_chat_template(enable_thinking=True) (M*'s path, mstar/model/qwen3_5/qwen3_5_model.py:570-584) on the sample checked. M*'s /v1/chat/completions under --rust-frontend reports usage.prompt_tokens = 0 (seen on all 64; not a benchmark input). Total output tokens over the 64 replies equal to within 1 across engines (0.8B 7528, 2B 7528-7529).
- Greedy agreement, identical of 64 / mean common-prefix: M* vs M* across boots 0.8B 38 / 0.807, 2B 38 / 0.798. vLLM vs M*: 0.8B 35-36 / 0.758-0.772, 2B 33-40 / 0.745-0.790. SGLang vs M*: 0.8B 32-34 / 0.772, 2B 34-35 / 0.750-0.756. Short of the M* self-floor on 0.8B by 2-6 prompts and 0.04 of prefix: bf16 near-tie divergence between engines of the size part 2 recorded; kept in the speed comparison because prompts are token-equal and ignore_eos fixes every output length, so the work per request is identical.
- Correctness-gate baseline for later rounds: main_{0.8,2}b_boot{1,2}.json.

### A2. Sweep (2026-11-03 21:30 - 23:45 ET; /workspace/rp3-main/sweep.sh; results/rp3-main/a2; report: gap.py)

- 3 rounds, rotated order, fresh server per round, fresh inductor cache per boot, 102 rows batch h200_1103, 0 failed. The M* c64 runs' client CPU 0.43-0.54 (not saturated).
- Commands per boot: `python -m mstar.cli.main serve qwen3_5_<N>b --gpus 0 --port 8100 --rust-frontend --rust-frontend-bin ...`, MSTAR_SHM_ARENA=1, placement auto (log "not pinned: auto is measured on Intel hosts only and this host's is AuthenticAMD").
- Mishap: my wait loop's pgrep matched itself, so A3 started 1.7 h late. No data affected.

### A4. Placement, first rounds (2026-11-04 ~02:40-03:35 ET; a4.sh; results/rp3-main/a4; rows rp3-main-loop)

- 0.8B ShareGPT c64 n=640: off 7972 / 8789 / 7612 (mean 8124), TTFT p50 36.7 / 36.6 / 38.2; siblings 15636 / 13313 / 15677 (14875), p50 26.2 / 29.7 / 24.8; always 10099 / 9767 / 9243 (9703), p50 35.8 / 32.3 / 34.5. (The frontend A/B later rewrote siblings_rust/r1-r2 on disk; the rows above are the reported ones in results.jsonl.)

### A4. Placement, frontend, references (2026-11-04; rows rp3-main-loop; results/rp3-main/{a4,grid,mask,refspin})

- Grid 3 rounds alternating, five workloads a boot. 0.8B off / siblings / numa-mask (taskset 64-127,192-255, no pin): ShareGPT c1 472 [453-488] / 885 [871-895] / 457; c16 4336 [4093-4625] / 8108 [7867-8261] / 4489; c64 8001 [7260-8612] / 15188 [14482-15764] / 7884; Random c16 5130 [4759-5377] / 7999 [7876-8195] / 5352; c64 11500 [11000-12355] / 15853 [15591-16044] / 11114. TTFT p50 ms: 18.2 / 14.8; 24.0 / 16.9; 38.0 / 26.2; Random 63.2 / 72.7, 80.5 / 104.8.
- 2B off / siblings / numa: ShareGPT c1 488 / 620 / 496; c16 4601 / 6308 / 4781; c64 8246 [8057-8489] / 12866 [12603-13134] / 9164 [7853-10690]; Random c16 5337 / 5992 / 5523; c64 10880 / 13325 / 11704. TTFT p50: 18.3 / 16.2; 24.0 / 18.0; 37.8 / 29.5; Random 83.5 / 90.0, 92.2 / 124.3.
- 16-CPU mask (64-71,192-199), 2 rounds, 0.8B: off / siblings / off + SCHED_IDLE spinners on those CPUs: c1 442 / 919 / 865; c16 4249 / 8444 / 8151; c64 8031 / 15814 / 14846; Random c64 11061 / 16537 / 17855. TTFT p50 16.7 / 12.8 / 10.9; 22.9 / 15.3 / 13.1; 36.8 / 24.6 / 21.2; 84.1 / 92.7 / 61.4. Holding the cores awake with no pin gets the gain: the cost is idle exit (C2 800 us) on each handoff. The socket is not it (numa arm level).
- Frontend at siblings, 0.8B ShareGPT c64: Python 13416, 10930 tok/s, TTFT p50 422, 531 ms vs Rust 15351 mean, p50 26. Python off: 7725, 8081, p50 71, 68. Python siblings without pinning the API result thread (scratch arm): 13868, 12014, p50 453, 669, so the pin is not the cause. Rust siblings without API pin: 15962, 15762, p50 28, 26 (level with pinned). The TTFT regression belongs to the Python frontend, whose process stops keeping up at this token rate (API process CPU 0.72 of a core over the whole run; not narrowed further).
- API process per thread at c64, Rust (share of a CPU, including the warm wave): off main thread 0.08, message thread 0.15, bridge 0.13, mstar-server 0.23; siblings 0.08, 0.23, 0.17, 0.32; always 0.08, 0.18, 0.15, 0.27. Files a4/<arm>/r*/threads.txt.
- References with cores held awake (spinners, same 16-CPU mask, 2 rounds, 0.8B): SGLang ShareGPT c1 581, c16 4922, c64 8820; Random c16 7292, c64 14990. vLLM 485, 3880, 6541; 4961, 9815. Both lower than unpinned on the full mask (SGLang 684, 5866, 11368, 8301, 17582). Not a test of their placement options: the mask and the spinners may both cost them. Their unpinned numbers stay the comparison.

### Other models, off vs siblings on main (2026-11-04; others.sh, klein.sh; results/rp3-main/others; 3 alternating rounds)

- Qwen3-ASR-1.7B, benchmark.runner, tok/s: c8 off 1166 [1059-1269], siblings 1757 [1679-1826]; c32 1568 [1388-1860], 2472 [2444-2490]; TTFT p50 ms c8 31 / 19, c32 296 / 186. All-beat.
- Qwen3.5-2B MMMU: c1 334 / 418; c4 1040 [933-1216] / 1410 [1389-1438]; c16 3296 [3184-3422] / 4359 [4338-4377]. All-beat.
- Kokoro, audio sec/s: c1 127 / 132, c16 488 [446-525] / 521 [463-558]. Level.
- FLUX.2 klein 4 steps 1024 px, Python frontend (the Rust frontend answers 404 for it), img/s: c1 off 2.888, 2.733, 2.706; siblings 2.731, 2.742, 2.665; c8 3.681, 3.669, 3.666; 3.666, 3.670, 3.661. Level.
- Whisper large-v3: weights not cached (config.json missing), not run.

### Round 1: step_thread_core auto takes siblings where cores sleep deep

- Commit 9a5c390 (code), c39528b (test). Hypothesis: every handoff between the worker's threads waits on C2 exit (0.8B ShareGPT c64 7.96 ms per step, grid siblings 4.21 ms); expected ~+80% on the headline.
- Commands: /workspace/rp3-main/loop_round.sh r1 <base> 9c6f1ad <cand> r1-auto (results/rp3-main/loop/r1), r1py.sh (Python frontend).
- 0.8B base | cand, tok/s mean [min-max], TTFT p50: ShareGPT c1 443 [417-482] 18.5 | 893 [879-912] 14.2; c16 4128 [3727-4450] 25.0 | 8290 [8030-8431] 16.6; c64 7720 [7567-8002] 38.9 | 15965 [15893-16004] 26.3 (+107%); Random c16 5375 [5242-5513] 65.7 | 7946 [7858-8055] 75.7; c64 10593 [10354-10770] 94.3 | 15865 [15557-16050] 102.3.
- 2B: ShareGPT c1 486 19.3 | 624 15.6; c16 4438 24.5 | 6382 17.8; c64 7862 [7701-8035] 39.4 | 13239 [13073-13357] 29.0; Random c16 5466 82.3 | 6019 88.0; c64 11011 96.6 | 13345 130.7. All-beat on every row. The second metric, TTFT p50 on Random c16, gets worse with non-overlapping rounds (0.8B 66 -> 76 ms, 2B 82 -> 88): the arrival rate rises with tok/s, and it stays under SGLang's 97 / 125 ms. Recorded, kept.
- Python frontend 0.8B ShareGPT: c16 base 5931, 4735, 4732, p50 20.4, 23.2, 23.7 | cand 8154, 8350, 7681, p50 16.4, 31.5, 35.9. c64 base 11113, 7858, 8551, p50 48, 61, 57 | cand 14573, 13086, 11183, p50 110, 345, 599. Tok/s up, TTFT worse at c64.
- Correctness: test_step_threads.py + test_knobs.py 360 passed; ruff clean. Greedy 64 prompts: 2B candidate vs base 39-40 identical (self 38), prefix 0.784-0.818 (0.798). 0.8B first came in under the two-boot floor (33-34 vs 38, 0.747-0.750 vs 0.807), so the floor was extended (par_extra.sh, 4 base and 3 candidate boots): base-base 6 pairs 32-38 identical, prefix 0.758-0.807; candidate-base 12 pairs 32-39, prefix 0.746-0.804; candidate-candidate 34-35. Same distribution; placement changes no arithmetic. PASS.
- Review (subagent): raised XPU/CPU workers reaching auto, missing declined flag on the too-small path, a docs claim about separate cores on EPYC past what was measured, stale docstrings, "older code" implying a cause, agreement test only on always. All fixed. Open and stated in the commit: one EPYC SKU standing in for AMD, the two earlier H200 rows that lost, Python-frontend TTFT, TP not run.
- Decision: KEPT.

### Headline switch (2026-11-04 18:20 UTC)

- At c39528b M* leads every ShareGPT point (1.15-1.38 over the faster reference); the largest gap on Qwen3.5 is 0.8B Random 1024/256 c64 (0.90) then c16 (0.96). Headline moves to output tok/s on 0.8B Random c64; second stays TTFT p50 on Random c16.

### Profile at c39528b (results/rp3-main/p1), 0.8B, phases on

- ShareGPT c64: iter_total 3.96, event_sync 0.46, gpu_thread.exec 1.60, postprocess_batch 1.35, plan.fresh 1.42 on 27%, pre_plan 0.75, commit 0.37, build_speculation 0.37. Random c64: iter_total 3.88 (p50 2.63); decode (84% of steps) gpu_exec 2.49, event_sync 0.36, exec 0.86, postprocess 1.02; mixed (16%) gpu_exec 10.4, event_sync 3.2, exec 4.3, postprocess 4.7, plan.fresh 1.77. Mixed steps ~40% of wall.
- Padding probe (scratch arm, results/rp3-main/padprobe, 894 mixed steps across Random c64, c16, ShareGPT c64): bucket 2048 at real tokens [1024,1152) 208 steps, [1152,1280) 47, [2048,2176) 164; bucket 1024 at 512-1023 104; bucket 512 at 256-511 101; bucket 256 71; bucket 128 at under 128 156.

## Part 3 checkpoint 1 (2026-11-04 13:15-15:00 ET): best c39528b vs SGLang 0.5.23 and 9c6f1ad

- Command: /workspace/rp3-main/ckpt.sh c39528b ckpt1 (3 rounds, sections 2 workloads, both sizes, order rotated, fresh server each). Rows batch h200_1104, status counted, 90 rows, 0 failed. SGLang stayed the faster reference on all rows in A2 and is the arm here.
- mean [min-max] tok/s, TTFT p50 (9c6f1ad | SGLang | best):
  - 0.8B ShareGPT c1 469 17.5 | 685 25.2 | 909 13.9; c16 4527 23.1 | 6133 34.3 | 8298 15.7; c64 7864 38.2 | 9951 50.8 | 15544 [15148-16068] 25.2; Random c16 5384 61.5 | 8319 96.2 | 8062 [7980-8110] 71.2; c64 12210 79.7 | 17137 211.2 | 16101 [15854-16339] 98.7.
  - 2B ShareGPT c1 455 19.0 | 538 25.1 | 618 16.5; c16 4439 25.1 | 4985 33.9 | 6311 18.1; c64 8045 38.2 | 9839 48.7 | 12975 29.3; Random c16 5399 79.8 | 5956 124.3 | 6002 88.5; c64 11365 89.2 | 12940 325.8 | 13357 121.2.
- test/modular at c39528b (CPU, CUDA hidden, -n 8 --dist loadfile; it ran alongside the first boots of round 2, 1184 s vs 572 s): 62 failed, 6 errors, 10361 passed; no failure outside the 9c6f1ad set; 2 of that set pass here (load-sensitive). Log rp3-logs/modular_c39528b.log.

### Round 2: reverted. A 1536 token bucket for Qwen3.5 prefill and mixed steps

- Hypothesis: a mixed step of one 1024-token prompt plus decode rows replays the 2048 graph (208 of 894 mixed steps, ~52% used; mixed gpu_exec 10.4 ms on 16% of Random steps); a 1536 bucket should cut ~0.2 ms per step, +5% on 0.8B Random c64.
- Command: /workspace/rp3-main/loop_round.sh r2 ... (0.8B, 3 alternating; arm: PREFILL_TOKEN_BUCKETS gains 1536). The three first arms overlapped checkpoint 1's test/modular; the result is level with or without them (base r3 and cand r3 ran clean: Random c64 15671, 15740).
- base | cand: Random c64 15929 [15671-16178] 100.4 ms | 15724 [15643-15789] 98.5; c16 7963 [7855-8047] 72.3 | 7913 [7733-8034] 64.3; ShareGPT c1 891 | 879; c16 8200 | 7909; c64 14985 [14144-15707] | 14712 [13718-15261].
- Correctness passed (0.8B 36-37 identical, 2B 38-40), not needed. Decision: reverted, no commit. Headline did not move; padding is not what those steps cost, and the hypothesis that GPU work scales with the bucket is not supported here. TTFT p50 on Random c16 improved ~11% (overlapping rounds), not pursued alone.

### Probes at c39528b (no code kept; results/rp3-main/{spy2,prefill,gputime})

- Prefill-only rate (0.8B, Random 1024-token prompts, max output 1, c16, n=240, 2 rounds): M* 171214, 152034; SGLang 167989, 199164; vLLM 160745, 167416 tok/s. Prefill is not M*'s gap, within this noise.
- Replay timing (scratch arm: timing events around the forward, ShareGPT c64 two passes, tok/s 15581, 16913): replay interval 16418 ms of 16638 ms wall (99%), but that interval runs from an event recorded before staging and the pre-plan wait, so it counts host gaps inside a step and is not GPU busy. CORRECTED by the kernel trace below: not fully GPU-bound. ms per replay: decode bs64 2.19, bs48 1.94, bs32 1.68, bs16 1.45, bs8 1.31, bs4 1.27, bs2 1.24, bs1 1.01. Mixed (bs64 bucket): tokens 128 3.86, 256 4.12, 512 4.82, 1024 6.63, 2048 9.57. A mixed step's share of replay time: 4.2 of 16.4 s.
- In-process stack sampler (scratch arm; biased to GIL release points, so used only as a pointer): main thread 63% in wait_for_work.
- Kernel trace (torch profiler, CUDA activity, 700 steps across a warm wave and the start of a timed pass, 0.8B ShareGPT c64): merged kernel busy 930 ms of 1355 ms active span (69%, with the profiler's overhead, not steady state), 1.34 ms of kernels per step. Gaps: 86 ms in 100-500 us, 282 ms in 500-5000 us. Top: GDN decode kernel 161 ms, attention prefill 49, fused add+RMSNorm 49, GEMMs ~350 in total, sampler (TopP 32.4, RadixTopK 16.3, split partials 12.1, split write 9.2 over 499 steps) 70 ms = 140 us per step.

### Round 3: reverted. GPU worker's torch intra-op pool capped from the CPU plan (Appendix A item 4)

- Hypothesis: part 2 measured 7-25 ms per image of torch CPU fork-join on the GPU thread at 128 threads in a 27-CPU quota; main's `_frame_interpolation` (qwen3_vit.py:106) is still torch; capping (4 threads) should lift Qwen3.5-2B MMMU c16 ~+10%. Not shown in a main profile first (rule 4 of section 3: it should have been; the measurement replaced it).
- Command: /workspace/rp3-main/r3.sh (Qwen3.5-2B, 3 alternating, c39528b vs + `torch.set_num_threads(plan.worker_threads)` in Worker._plan_step_cpus).
- base | cand: MMMU c16 4445 [4419-4476] 85.6 ms | 4335 [4253-4427] 94.0; c4 958 [955-959] 61.6 | 948 [927-978] 65.2; Random c64 13412 [13404-13422] | 13322 [13179-13454]; ShareGPT c64 13256 [13180-13325] | 13045 [12877-13296].
- Decision: reverted. No gain on main at siblings. Spread at siblings is now ~0.3% on base. Not checked further: why part 2's cost is absent (its round ran placement off, on Qwen3-VL).

### Open PRs on this host

- #1172 (fused decode, merged onto c39528b, model_kwargs.fused_decode_batch_sizes [1, 2] on 0.8B; 3 alternating; results/rp3-main/prs/1172): the log says "fused decode stays off on LLM: compute capability (9, 0)". H200 gets nothing from it: level (c39528b | PR ShareGPT c1 894 | 907, c64 15559 | 15435; Random c64 16110 | 16068). #1163 is microbenchmarks only, nothing to serve.
- #1143 could not be merged onto c39528b (conflicts in 23 files, it is 34 behind main). Runs next on its own base a0470c3 with an extension and frontend built from its tree, at MSTAR_STEP_THREAD_CORE=siblings; pyzmq added to venv-main for that base (main doesn't import it).
- Not run: #1121 (398 ahead, 96 behind; would need its own venv), #1132 (arena by default: this setup already sets MSTAR_SHM_ARENA=1, so nothing to measure), #1093, #1175 (Omni and side lane: none of these workloads).

### Round 4: reverted. Tuned (not deterministic) compile for Qwen3.5 mixed captures

- Hypothesis: mixed steps compile deterministic while decode keeps tuned kernels; mixed replay at under 128 prompt tokens 3.86 ms vs decode 2.19 ms; tuned should cut ~15% of mixed replay, ~+4% on Random c64.
- Command: /workspace/rp3-main/loop_round.sh r4 (0.8B, 3 alternating; compile_mode=decode_compile_mode for the mixed config). base | cand: Random c64 16578 [15925-17095] 83.0 ms | 16575 [15723-17502] 101.5; c16 8213 | 8159; ShareGPT c1 914 | 909; c16 8409 | 8348; c64 15954 [15687-16437] | 15503 [14948-15787]. Correctness passed, moot. Decision: reverted.

### Mixed step split (kernel trace, 22 mixed vs 477 decode steps, small mixed sample)

- decode 335 kernels, 1722 us of kernels; mixed 932 kernels, 5406 us: elementwise 87x 313 us, cat 90x 234 us, indexSelect 32x 230 us, index_copy 48x 228 us, on top of the GDN kernels. docs: 340160530's message says this eager path around the chunked kernel is 40 kernels a layer, and gdn_stage_applies (mstar/engine/resources/linear_attn/gdn_stage.py:40-49) keeps H200 on it "until someone measures them".

### Round 5: GDN one-launch prefill input stage on Hopper (Appendix A: found by the kernel trace, not listed)

- Commits: code, then test; best SHA b851763.
- Hypothesis: H200 takes the eager GDN prefill input path, ~250 small kernels and ~1 ms of kernel time a mixed step, because gdn_stage_applies gates on Blackwell; one launch should give +5-10% on 0.8B Random c64.
- Command: /workspace/rp3-main/loop_round.sh r5 (0.8B and 2B, 3 alternating, one-line gate change), img_gate.sh r5 (2B, 16 MMMU images plus MMMU c16, 2 boots each).
- 0.8B base | cand: Random c64 (headline) 16315 [16256-16382] 93.6 ms | 17009 [16517-17899] 73.5 (+4.3%, all-beat); c16 8070 [8051-8095] 67.7 | 8319 [8173-8589] 55.6 (+3.1%, all-beat); ShareGPT c1 921 | 909; c16 8427 | 8629 (+2.4%, all-beat); c64 16079 [15830-16300] | 16397 [16145-16709].
- 2B: Random c64 13312 [13260-13413] 113.2 | 13835 [13765-13875] 104.1 (+3.9%, all-beat); c16 6004 | 6147 (all-beat); ShareGPT c1 622 | 623; c16 6287 | 6468; c64 12940 [12282-13307] | 13479 [13124-13815]. MMMU c16 wall 20.56, 20.58 | 19.86, 19.91 s.
- Correctness: test_gdn_prefill_stage, test_gdn_mixed_split (new staged cases against the chunked path), test_gdn_cuda_graph, test_step_threads on the H200 (see log rp3-logs/gdn_gpu_tests.log and the rerun). Greedy 64 prompts: 0.8B 35 identical, prefix 0.789 (floor 32-38, 0.758-0.807); 2B 37-38, 0.810 (38, 0.798). 16 images: base-base 7, 0.616; cand-base 8-10, 0.643-0.781.
- Review (subagent): no correctness problem on SM90 by reading the code; asked for the test changes and a stale comment in config.py, done. Open and stated in the commit: 9B TP2, 27B TP4, Qwen3.8 not run.
- Decision: KEPT.

## Part 3 checkpoint 2 (2026-11-04 20:30-22:00 ET): best b851763 vs SGLang 0.5.23 and 9c6f1ad

- Command: /workspace/rp3-main/ckpt.sh b851763 ckpt2. Rows batch h200_1104, status counted, 90 rows, 0 failed.
- mean tok/s, TTFT p50 (9c6f1ad | SGLang | best):
  - 0.8B ShareGPT c1 454 16.7 | 689 24.3 | 868 13.6; c16 4566 23.0 | 6097 34.8 | 8275 15.6; c64 8031 37.6 | 11008 52.2 | 15566 25.1; Random c16 5306 64.3 | 8215 99.2 | 8092 63.4; c64 10992 84.8 | 16853 225.9 | 16413 [15718-17500] 81.2.
  - 2B ShareGPT c1 480 18.5 | 537 25.5 | 619 16.4; c16 4646 23.6 | 4989 33.8 | 6421 17.4; c64 8588 36.5 | 10195 49.9 | 13709 27.8; Random c16 5338 80.5 | 5984 125.8 | 6099 85.1; c64 11903 91.9 | 12978 314.9 | 13753 114.2.
  - Best / SGLang tok/s: 0.8B 1.26, 1.36, 1.41; Random 0.985, 0.974. 2B 1.15, 1.29, 1.34; Random 1.02, 1.06. TTFT p50 best / SGLang 0.56, 0.45, 0.48; 0.64, 0.36; 0.64, 0.51, 0.56; 0.68, 0.36.
- test/modular at b851763: 59 failed, 6 errors, 10364 passed, all in the 9c6f1ad set (no GPU job ran alongside this time, 260 s). Log rp3-logs/modular_b851763.log.

### Qwen3.5-9B gap table (2026-11-04 ~22:10-23:20 ET; sweep9.sh b851763; 3 rotated rounds, 60 rows batch h200_1104 counted, 0 failed)

| workload | c | 9c6f1ad | best b851763 | vLLM | SGLang | best / faster ref | TTFT p50 best / lower ref |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ShareGPT | 1 | 202 | 203 | 183 | 198 | 1.03 | 24.8 / 31.9 |
| ShareGPT | 16 | 2440 | 2487 | 2107 | 2228 | 1.12 | 28.5 / 37.0 |
| ShareGPT | 64 | 5237 | 5396 | 4663 | 4139 | 1.16 | 47.7 / 58.2 |
| Random | 16 | 2139 | 2167 | 2081 | 2140 | 1.01 | 229.5 / 347.0 |
| Random | 64 | 4444 | 4524 | 4477 | 4381 | 1.01 | 264.8 / 965.9 |

M* leads every 9B point, already at 9c6f1ad on most. 9B is GPU-bound on this host: placement and the stage add 1-3%. No round taken on 9B. Appendix item 6 (prompt padding) has no gap to close here at the benchmark's prompts.

### Round 6: not kept. MSTAR_INGRAPH_DECODE_ROPE=1 (existing knob, off by default; no code)

- Hypothesis: mixed steps build rotary tables before the replay (engine.preprocess.cos_sin 0.20 ms on 16-27% of steps); in-graph should give +1-2% on 0.8B Random c64.
- Command: /workspace/rp3-main/r6.sh (b851763, 0.8B, 3 alternating; arms differ only in the env).
- base | cand: Random c64 16247 [15965-16503] 85.6 ms | 16880 [15929-18580] 80.3 (+3.9%, overlapping: one round at 18580); c16 8067 | 8175; ShareGPT c1 886 | 888; c16 8501 | 8628; c64 16430 | 16643. No regression, headline not all-beat. Decision: not kept by the rule. A rerun might separate it: note for later.
- #1143 on its own base a0470c3 (34 commits behind main), step_thread_core auto there (siblings layout), extension and frontend built from its tree, 0.8B, 3 alternating: base | PR ShareGPT c1 791 | 781; c16 7299 [7003-7448] | 7617 [7519-7791] (+4.4%, all-beat); c64 13325 [13003-13536] | 13643 [13238-13857] (+2.4%, overlapping); Random c16 7853 | 7853; c64 15656 [15196-15953] | 15811 [15538-16153] (+1.0%, overlapping), TTFT p50 87 | 108 ms. Its base lacks 9/N-era commits on main that it conflicts with, so these are not additive with c39528b. Recorded; not built on.

### Disk (2026-11-05)

- Removed for Omni's 70.5 GB (all re-downloadable): Qwen3.5-9B, Qwen3-VL-4B and 8B weights, FLUX.2 klein weights, uv cache. rmmodel.py removes only blobs no other model links. A rerun of part 1/2 or the 9B table downloads them again.

### Qwen3-Omni Thinker A2T gap table (2026-11-05 ~01:15-02:00 ET; sweep_om.sh b851763; configs/qwen3omni_colocated.yaml; LibriSpeech via benchmark.runner, greedy Thinker, max 256 tokens, c8 n=200, c32 n=600; 3 rotated rounds; rows batch h200_1105 counted, 0 failed)

- req/s, TTFT p50 ms. c8: 9c6f1ad 24.1 [23.9-24.3] 66 | vLLM 25.4 [25.1-26.2] 70 | best 25.1 [24.9-25.3] 64. c32: 9c6f1ad 51.3 [50.3-53.1] 109 | vLLM 80.4 [79.6-81.7] 78 | best 57.8 [57.3-58.8] 96. best / vLLM: c8 0.99, c32 0.72. Text tok/s in the rows: the two clients count chunks differently for the same text (556 of 600 replies identical, characters within 0.1%), so req/s is the fair metric.
- SGLang 0.5.23 crashes on batched audio at c8: "RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 483 but got size 659" then SIGQUIT (results/rp3-main/omni_smoke/sglang). Recorded; vLLM is the reference.
- Worker phases at b851763 c32 n=600 (results/rp3-main/p2, phases on, iters 400-1000, bs 19.4 although c=32): iter_total 13.08, event_sync 5.77; by total time over 700 steps of 9155 ms: mixed 215 steps gpu_exec 19.83 ms (4263 ms), decode 270 steps gpu_exec 11.09 (2994), audio encoder 214 steps 3.64 (778). Thinker prepare_inputs on mixed steps 10.55 ms (cpu 9.2), which #1151, #1157, #1093 rewrite.
- M* kernel trace (500 steps, profiler on): merged busy 3477 of 4796 ms (73%); fused_moe_kernel 1713 ms (92.9 us x 36.9 per step, 49%), _gate_up 162, decode attention 156, prefill attention 152, _down 118.

### Omni probes (2026-11-05 ~02:15-03:00 ET; results/rp3-main/{prs/omni1157,omtrace})

- Open PR stack #1157 tip (includes #1151; 491 commits ahead of a base 96 behind main; extension and frontend built from its tree, step_thread_core auto there) on qwen3omni_colocated.yaml: compile raises ConstraintViolationError on cu_seqlens, nearly every request answers 500, 0.13 req/s. b851763 alongside: 57.4, 57.4, 55.6 req/s. Nothing to measure on this config and GPU. The three rp3-main-loop rows with commit pr1157 say failed 0: wrong, om_row.py's regex missed it (the runner's counts were ~598 of 600 failed); treat those three as failed.
- vLLM 0.33 kernel trace, c32 n=600 under its torch profiler (77.1 req/s): merged busy 8225 of 9157 ms (90%); fused_moe_kernel 5320 ms (58.3 us x 91200, 1900 steps at 24 layers x 2), FA3 attention 515, topk gating 179. M* (b851763, 500 steps, 35 req/s under the profiler): busy 3477 of 4796 ms (73%); fused_moe 1713 ms (92.9 us x 18432). Normalized per request the two engines are within about 20% in kernel time: no single kernel explains the gap. M*'s steps: bs 19 at c32, one chain carrying audio encoder, mixed and decode steps in turn.
- Next: re-sweep the H200 moe_gemm table with the current harness (benchmark.kernels.sweep; table dated 2026-10-31 from the older tuner, B300's was re-swept in #1160).

### Headline switch (2026-11-05 03:00 ET)

- 0.8B Random c64 is 0.974 at b851763 and every other Qwen3.5 point is at or above 0.985; Omni A2T c32 is 0.72. Headline moves to Omni Thinker A2T c32 req/s (the runner's text tok/s differs between clients on the same text). Second: TTFT p50 at the same point.

### Round 7: reverted (no code). H200 moe_gemm table re-swept with the current harness

- Hypothesis: sweep shows 500 -> 442 us at M=2048, 843 -> 697 at M=4096; +1-3% on Omni A2T c32. Sweep: benchmark.kernels.sweep moe_gemm --shape E=128,N=1536,K=2048,top_k=8 (8.5 min; below M=1024 the table is within 2%).
- Command: /workspace/rp3-main/r7.sh (MSTAR_KERNEL_TABLES on b851763, 3 alternating). base | cand: c32 58.1 [57.9-58.3] 96.7 ms | 58.2 [57.6-58.6] 97.7; c8 24.9 | 25.0. Decision: reverted. Table in /workspace/rp3-main/ktables, not committed.

### Round 8: not kept. 64 and 96 token buckets for the Thinker's prefill and mixed captures

- Hypothesis: 84% of Thinker mixed steps replay the 128-token bucket with ~58 real tokens (padding probe, results/rp3-main/padprobe2, 200 steps: bucket 128 169, 512 13, 1024 8, 256 7; real under 64: 102, 64-127: 67); +10-20% on Omni A2T c32 req/s.
- Command: /workspace/rp3-main/r8.sh (PREFILL_TOKEN_BUCKETS gains 64, 96; 3 alternating; 86 vs 74 Thinker captures). base | cand: c32 59.1 [58.3-59.6] 93.7 ms | 59.5 [58.9-60.6] 92.7; c8 24.7 [24.2-25.1] 64.0 | 25.7 [25.3-26.1] 60.3 (+4%, all-beat). Decision: not kept, headline flat. c8's gain is real and a later headline at c8 could take it.

### Round 9, first pass (scratch arm, not a commit): hold the audio encoder for 4 rows or 15 ms

- Hypothesis: the audio encoder runs as soon as one row is ready, so a mixed step carries ~1.4 prompts at 19.8 ms gpu_exec (47% of wall at c32); holding should merge prompts, +10%. Correction to an earlier note: AudioEncoder batches up to 8 (AuTEncoderSubmodule.MAX_BATCH_SIZE, aut_submodule.py:39); the max_batch_size of 1 at submodules.py:242 is the vision encoder's.
- Command: /workspace/rp3-main/r9.sh; arm r9-hold patches MicroScheduler (_scratch_hold, env SCRATCH_HOLD_K, SCRATCH_HOLD_MS). base | cand: c32 57.6 [57.0-58.3] 97.0 ms | 61.0 [60.2-61.6] 101.0 (+5.9%, all-beat); c8 24.8 [24.3-25.2] 64.7 | 25.8 [25.6-26.1] 80.7.
- Design question open: TTFT at c8 +25%. Settings sweep next. Audio encoder on a side lane instead would remove its 214 steps of 700 from the chain, but mstar/worker/engine_manager.py:158-186 refuses a walk that isn't the node's alone (prefill_audio holds Thinker too) and a node that holds resources: design note, not one round.

### Hold settings (scratch arm, 2 alternating rounds; results/rp3-main/loop/r9b)

- req/s, TTFT p50. c32: base 58.5 98.5; 3 rows or 6 ms 59.5 97.5; 8 rows or 6 ms 59.9 101.5; 8 rows or 30 ms 64.0 113.5. c8: base 24.9 64.5; 25.7 72.5; 25.3 75.0; 25.6 93.5. Longer holds buy more throughput; a busy gate removes the c8 cost.

### Round 10: GatherPolicy, opt-in, Qwen3-Omni's audio encoder at 8 rows or 30 ms while 16 requests are live

- Commits: code, then test; best SHA 9a228e9. Hypothesis: as round 9; scratch gave +9.4% at c32.
- Command: /workspace/rp3-main/r10.sh (3 alternating, c8 n=200, c16 n=300, c32 n=600). base | cand: c32 58.5 [58.0-59.2] 96.0 ms | 64.3 [63.1-65.0] 111.3 (+9.9%, all-beat; p99 lower, 301, 169, 266 | 256, 212, 173); c16 38.6 [38.1-39.5] 74.0 | 40.3 [39.6-41.7] 92.0 (+4.4%, all-beat); c8 25.1 60.7 | 24.9 63.0.
- Second metric (TTFT p50 at c32) is worse with non-overlapping rounds: the policy trades it, stated in the commit. M*'s TTFT against vLLM at c32 goes from 96 to 111 vs 78 ms.
- Correctness: test_gather_policy.py, scheduler tests 336 passed, test/modular run 59 failed, 6 errors, all in the 9c6f1ad set. Omni replies c32 of 600 identical: base-base 570-579; cand-base 564-588. Qwen3.5 greedy (r10q): 0.8B 36, 0.786-0.792 (floor 32-38, 0.758-0.807). 2B 34-39, 0.727-0.796 against a floor from 4 base boots (6 pairs) 35-41, 0.780-0.822: one boot slightly under on prefix. The change doesn't touch Qwen3.5's arithmetic or batching (no policy declared); not rerun after that.
- No-regression: r10q Qwen3.5 0.8B read ShareGPT c64 -5.2% (overlapping); rerun on final code (r10r, 4 alternating): ShareGPT c64 15585 [14811-16202] | 15562 [14990-16127]; Random c64 16605 [16001-17584] | 16327 [15961-16864]. Noise.
- Review (subagent): no hang or lockstep bug. Fixed: a followed TP head no longer waits in the peek, rows from MAX_BATCH_SIZE, sentinel, note_ran in the test, docs/adding_models.rst. Stated in the commit: A2A unmeasured, multi-clip prompts wait per clip, TP not run. Left: peek counts graph-ready rows where the scan counts engine-ready ones, so one speculation can be lost in a rare disagreement.
- Decision: KEPT.

## Part 3 checkpoint 3 (2026-11-05 08:45-12:00 ET): best 9a228e9 vs SGLang 0.5.23, vLLM 0.33.0 (Omni) and 9c6f1ad

- Commands: /workspace/rp3-main/ckpt.sh 9a228e9 ckpt3; sweep_om.sh 9a228e9 ckpt3_om. Batch h200_1105, counted, 0 failed.
- 0.8B (best vs SGLang, tok/s): ShareGPT c1 892 / ~685; c16 8570; c64 16144 [15947-16479]; Random c16 8315 [8224-8464], c64 16625 [16016-17412]. 2B: ShareGPT 621; 6453; 13548 [13486-13595]; Random 6116; 13834 [13764-13906]. Full rows: rp3-main/ckpt3 (cmp.py).
- Omni A2T, req/s, TTFT p50: c32 9c6f1ad 54.0 104.3 | vLLM 80.9 75.7 | best 63.6 111.3 (0.79 vs vLLM, +18% vs 9c6f1ad). c8 23.9 67.0 | 25.6 71.0 | 25.1 64.0.
- test/modular at 9a228e9: 59 failed, 6 errors, 10370 passed, none outside the 9c6f1ad set.

### Round 11: not kept. kv_plan_cache on Qwen3.5-2B (env, no code)

- Hypothesis: 2B's config leaves it off where 0.8B's sets it; +2-4% on ShareGPT c64. Commands /workspace/rp3-main/r11.sh. base | cand: ShareGPT c64 13639 [13508-13885] | 13291 [12825-13653]; c16 6493 | 6450; c1 619 | 621; Random c64 13798 | 13635; c16 6135 | 6122. Reverted.

### Round 12: not kept. MSTAR_INGRAPH_DECODE_ROPE=1 rerun, 4 alternating at 9a228e9

- base | cand 0.8B: Random c64 15906 [15476-16299] | 16067 [15693-16606] (+1.0%); ShareGPT c64 15132 | 15452; c16 8276 | 8425; c1 879 | 878; Random c16 8015 | 8031. Overlapping; round 6's +3.9% was mostly a single round. TTFT p50 on Random 5-8% lower, overlapping. Not kept.

### Round 13: not kept. gc_gen0_threshold 50000 on qwen3omni_colocated.yaml (env, no code)

- Hypothesis: profile shows gc.gen0 150 and gc.gen1 14 (mean 3.2 ms) in the window and a stall in om.prep.mask (mean 1.68, p50 0.05, p95 8.9 ms); +2-5% on Omni A2T c32. Commands /workspace/rp3-main/r13.sh. base | cand: c32 63.5 [62.9-64.4] 114.7 ms | 63.9 [62.8-64.6] 111.7; c16 39.8 | 39.9; c8 24.9 | 24.7. Flat. Reverted.

### Thinker prepare_inputs per op (scratch arm, c32 n=600, phases on, results/rp3-main/p3)

- Per audio row, ms mean (p50): m_w0 `mm_mask[0] = False` 1.80 (0.03), p95 8.7, CPU the same; rope 0.26 (0.10); spliced 0.62 (0.52; talker masks 0.45 of it); text_around 0.04; others under 0.07. 4.4 rows per mixed step after round 10.
- Hypothesis, unverified: m_w0 is the first device launch after N's replay and blocks behind it, spinning, so it is GPU wait charged to a host phase (benchmark/worker_phases README: a wait lands in the enclosing span). Its share matches a step that is GPU-bound (gpu_exec 88% of wall). Removing it would move the wait. Not pursued. Talker masks for a text-only request (~2 ms a mixed step, on the GPU thread) are open to the same doubt; #1151 and #1157 change them.

### Round 14: not kept. Thinker 64 and 96 token buckets at 9a228e9

- Hypothesis: round 8's c8 +4% and TTFT -6%, with GatherPolicy now in. Commands /workspace/rp3-main/r14.sh, 3 alternating. base | cand: c32 63.9 [63.1-64.5] 111.3 ms | 64.8 [64.4-65.4] 109.3; c16 39.8 [38.9-40.3] 95.0 | 40.8 [39.9-41.4] 91.3; c8 24.8 [24.5-25.1] 64.3 | 25.4 [25.0-25.8] 61.7. All positive, +1-2.5%, overlapping: headline not all-beat. Costs 12 more captures. Not kept; a candidate if someone prefers lower TTFT.

### Image gap on main at 9a228e9 (off-brief; img_sweep.sh, imgc4.sh; results/rp3-main/{imgsweep,imgc4})

- Qwen3.5-2B MMMU, 2 rotated rounds, tok/s, TTFT p50 (first workloads on a fresh server: c4 n=64 then c16 n=256): M* c4 971 [962, 980] 66 | vLLM no cache 1186 109 | vLLM default 1231 122 | SGLang 945 208. c16 M* 4399 92 | 3705 177 | 3791 128 | 2126 386. M* leads c16 by 1.16 and TTFT at both. 0.8B did not run: my script's helper overwrote the loop's list.
- c4 is a first-pass artifact: three passes on one server, wall M* 17.2, 11.4, 11.4 s (1432-1440 tok/s warm) against vLLM no cache 14.0, 13.8, 13.7 (1190). Warm c4 is 1.21.
- The first pass's 5.8 s is one stall: requests 28-31 of 64, jct 6.3-6.7 s against 0.7-1.1 warm, TTFT of requests 30, 31 about 6 s, all others within 150 ms of warm. Request 30 is a 2560x2133 image, 5360 tokens, above PREFILL_VISION_TOKEN_BUCKETS' top 4096 (qwen3_5/submodules.py:145), the first such image on the server. Hypothesis (from the comment there, untested): eager first-compile. Requests 33, 37, 38, 49 are also above 4096.
- Round 15 tests an 8192 bucket.

### Round 15: 8192 token bucket for Qwen3.5 vision prefill (best afe4633)

- Hypothesis: the first image over 4096 tokens on a server runs eagerly and compiles, 5.6 s with all rows waiting (imgc4 above); a bucket holds it. +30-50% on 2B MMMU c4 first pass.
- 2B (r15.sh, 3 alternating, first pass on a fresh server; base | cand): MMMU c4 963 [949-989], TTFT p99 2432-2614 | 1374 [1371-1377], p99 535-559 (+42.6%, all-beat); c16 4372 [4314-4411] | 4332 [4315-4366]; ShareGPT c64 13533 | 13245, Random c64 13716 | 13630 (overlapping). Capture +44 MiB. 2B image gate wall 20.6 | 15.1 s for c16 after the image record.
- 9B (r15_9b.sh, 2 alternating): MMMU c4 486, 479 | 569, 563 (+18%, TTFT p99 6491, 3116 | 1119, 1127); c16 1874, 1879 | 1569, 1822 (TTFT p99 1973, 2358 | 8202, 2410; lower in both rounds); ShareGPT c64 5411, 5399 | 5406, 5398; Random c64 4531, 4510 | 4534, 4533. c4 + c16 wall 68.7, 69.1 | 70.6, 65.1 s. Capture 94709 -> 94273 MiB free, 100 -> 103 captures, largest eager step 352 -> 705 MiB. Hypothesis: on 9B the stall moves to the first step packing over 8192 tokens at c16; not verified.
- Review (subagent): wording fixes applied. Dynamo: 83 compiles against its limit of 84 (was 80), tight. Inherited by all Qwen3.5 sizes and Qwen3.8; 27B not run. BLOCK_LOOP_PATCH_BUCKETS still tops at 16384 patches (4096 tokens), so such an image's tower runs uncaptured.
- Correctness: 16 images base-base 9, 0.678; cand-base 6-10, 0.559-0.767. 64 prompts 2B 33-35, 0.728-0.784 (floor 35-41, 0.780-0.822, one boot); 0.8B 31-34, 0.709-0.738 (floor 32-38, 0.758-0.807): under the floor on one boot, so 3 more boots are being recorded.
- 3 more 0.8B boots on the final tree against 4 base boots: 33-39 identical, prefix 0.744-0.803. The low boot was a draw (floor 32-38, 0.746-0.807). test_qwen3_5_bucket_knobs, mixed, video: 70 passed. Decision: KEPT, off the section 2 headline (off-brief image work; Random and ShareGPT unchanged), stating 9B MMMU c16's lower reading (2 rounds, non-overlapping, explained only by hypothesis) in the commit.

### Qwen3.5 images on main (img_sweep.sh; 2 rotated rounds; first on a fresh server; rows not reported as counted: off the brief's sweep)

- 0.8B MMMU at afe4633, tok/s, TTFT p50: c4 M* 1949 57 | vLLM no cache 1503 124 | vLLM default 1417 136 | SGLang 1081 240. c16 6050 101 | 4488 166 | 4311 171 | 2305 470. M* leads 1.30 and 1.35.

## Part 3 checkpoint 4 (2026-11-05 ~18:00-20:30 ET): best afe4633 vs SGLang 0.5.23, vLLM 0.33.0 (Omni) and 9c6f1ad

- Commands: ckpt.sh afe4633 ckpt4; sweep_om.sh afe4633 ckpt4_om. Batch h200_1105, counted, 0 failed.
- 0.8B best / SGLang: ShareGPT c1 882 / 687 (1.28), c16 8253 / 5976 (1.38), c64 15645 / 11590 (1.35); Random c16 8058 / 8350 (0.97), c64 15821 / 16331 (0.97). 2B: 623 / 536 (1.16), 6419 / 4976 (1.29), 13539 / 9321 (1.45; SGLang had one low round, 7901); Random 6098 / 5926 (1.03), 13762 / 12851 (1.07). TTFT p50 best / SGLang 0.55, 0.45, 0.51; 0.70, 0.38; 0.58, 0.50, 0.56; 0.67, 0.36. Best / 9c6f1ad tok/s: 0.8B 2.07, 1.90, 2.02; 1.61, 1.48. 2B 1.40, 1.43, 1.72; 1.20, 1.27.
- test/modular at afe4633: 59 failed, 6 errors, 10369 passed, all in the 9c6f1ad set (log modular_afe4633).
- Omni A2T at afe4633, req/s, TTFT p50: c32 9c6f1ad 52.1 109.7 | vLLM 80.9 75.7 | best 63.3 113.3 (0.78 vs vLLM, +21% vs 9c6f1ad). c8 23.7 70.0 | 25.3 73.0 | 24.8 65.3.

### Round 16: reverted (scratch arm, no commit). Tower block loop to 24576 patches

- Hypothesis: BLOCK_LOOP_PATCH_BUCKETS tops at 16384 patches, so an image past 4096 tokens runs its tower uncaptured; a 24576 bucket should capture it and could remove the rest of the stall. Command /workspace/rp3-main/r16.sh (2B, 3 alternating). base | cand: MMMU c16 4334 [4316-4345], TTFT p99 1077-1169 | 3586 [3152-4348], p99 1143, 6251, 6582 (-17%: a 6 s stall in 2 of 3 rounds, cause not looked into); c4 1379 | 1374; ShareGPT c64 13157 | 13469; Random c64 13659 | 13511. Decision: reverted.

### Row audit (2026-11-06)

- r3, r15, r15_9b, r16, imgsweep and the Qwen3.5-2B image half of others/ had not been reported (their scripts didn't call mstar-report). Backfilled by /workspace/rp3-main/backfill.py into rp3-main-loop, status probe, `started` from the result file's mtime. r8/r9/r10/r13/r14 and Omni rounds reported as they ran. Kokoro, ASR and klein stayed in this file only (unit differs). Three correction rows mark #1157's failed rounds.

### Round 17: reverted. Token budget for prefill_vision (keep a step inside the 8192 bucket)

- Hypothesis (proposal 11): r15's 9B MMMU c16 stall is a step packing past 8192 tokens. Change: ChunkingPolicy(walks=("prefill_vision",), max_batch_tokens=8192) and get_input_sequence_len from the grids. Commands /workspace/rp3-main/r17.sh, r17_9b.sh, 3 alternating each.
- 2B base | cand: MMMU c16 4318 [4268-4357] | 4351 [4329-4367]; c4 1384 | 1368; Random c64 13546 | 13671; ShareGPT c64 13113 | 13147. Flat.
- 9B base | cand: MMMU c16 1836 [1833-1840], TTFT p99 2341-2503 | 1593 [1589-1600], p99 5097-8044 (-13.2%, non-overlapping); c4 567 | 566; Random c64 4526 | 4521; ShareGPT c64 5388 | 5396.
- Base at afe4633 has no stall on 9B in 3 rounds, so the 8.2 s in r15's single candidate round was a one-off and the hypothesis has nothing to fix. The candidate's stall comes with "step of 8275 tokens over its budget of 8192": the length undercounts, so a fuller pack spilled past the bucket (mechanism read from that log line). Decision: reverted, not patched. r15's commit message's 9B c16 numbers (1695 mean) stand as measured; these 3 rounds read 1836 against 9a228e9's 1874, 1879, a 2% gap, TTFT p99 unchanged.

### Round 18: no Talker masks for a request with no audio output (best afbf77c)

- Commits 78d41d6 (code), afbf77c (test). Hypothesis: 0.45 ms of mask ops per audio row plus mm_mask's launches, 4.4 rows a mixed step, on the GPU thread (om.prep, om.splice); +1-3% on Omni A2T c32.
- Command: /workspace/rp3-main/r18.sh (3 alternating, c8 n=200, c16 n=300, c32 n=600). base | cand, req/s, TTFT p50: c32 63.8 [63.2-64.7] 112.3 ms | 68.0 [67.5-68.4] 106.0 (+6.6%, all-beat); c16 39.6 [39.4-39.7] 94.7 | 41.7 [40.8-43.0] 90.0 (+5.3%, all-beat); c8 25.0 [25.0-25.1] 64.3 | 26.2 [25.9-26.5] 60.3 (+4.8%, all-beat).
- Larger than expected, and not explained. CORRECTION to the "Thinker prepare_inputs per op" note above: it held that m_w0's stall (1.80 ms mean) was GPU wait that removal would only move, and doubted the masks for the same reason. Removing it did help, so that hypothesis was wrong or incomplete. No profile was taken after the change.
- Correctness: test/modular 59 failed, 6 errors, 10372 passed, all in the 9c6f1ad set (modular_final); new test in test_qwen3_omni_prefix_cache.py; ruff clean. Replies at c32, of 600: base-base 575-581; cand-base 573-586. Audio output: A2S 8 of 8 on the candidate, 14.6 s mean audio, no errors in the log (results/rp3-main/loop/r18/a2a); no base A2S alongside. The 64-prompt Qwen3.5 gate was not run: this file is Omni's Thinker only.
- Review (subagent): by reading the code, every consumer handles an absent mask; `audio_output` is an explicit False for text-only requests and set on every Thinker prefill step. Decision: KEPT.
- Not rerun: no checkpoint at afbf77c, since a section 2 sweep takes longer than was left before the deadline. The 0.84 below is this round's 68.0 over checkpoint 4's vLLM 80.9, not an alternating comparison.

### Omni re-profile at afbf77c (2026-11-06 ~22:50 ET; prof_om.sh, results/rp3-main/p4; phases on, c32 n=600, iters 400-900, bs 20.9)

- iter_total 10.92 ms (was 13.08), event_sync 6.64 (61%). Decode 370 steps gpu_exec 10.71 (3962 ms), mixed 116 steps 17.61 (2043), audio encoder 114 steps 4.59 (524). Mixed gpu_thread.total 14.81 -> 4.48 ms, exec 4.02 -> 2.86: the prepare_inputs time did not reappear elsewhere, which is what round 18's correction note left open. Mixed event_sync 14.3 -> 14.0 ms, unchanged. The step is now GPU-led.
- Next lead, hypothesis: rows average 21 while Thinker decode captures are 16 and 32, so most decode steps replay the 32-row graph; the moe_gemm sweep times M=24 at 222 us against M=32 at 248 us (-10%). Finer decode batch sizes (12, 20, 24, 28) could cut decode's 60% of wall. Not a full round: too little time left. Single-round probe below; a full 3-round A/B is the next round.
- Probe (one round, candidate then base, /workspace/rp3-main/p19.sh; not a round by section 4): decode captures at 1, 2, 4, 8, 12, 16, 20, 24, 28, 32 rows (82 Thinker captures vs 74, 136 MiB) | afbf77c, c32 n=600: 67.7 req/s, TTFT p50 107 ms | 68.9, 96 ms. No sign of a gain: dropped, no commit. Supports the same result as rounds 2 and 8: padding is not what these steps cost on this GPU.
