IN PROGRESS: sweep at head 6ef0789 running; gate and Python-frontend rounds follow. Numbers below are partial.

Read first: the 9B gate (section 6), main's two modes (section 5), and the Python frontend's TTFT under `siblings` (concept 1).

# PR Qwen3.5 facts

## 2. SHAs
- Base: 305e80ef46248ade6e6670aa20c9bb67f4ac2e26
- Head: 6ef07892bbbd81eb67aa4d7d251940ddb21b0eff
- Branch: perf/rp3-pr-qwen35, force-updated per the amendment. It now sits directly on main: the bucket-cut branch was dropped (PR_BUCKET_CUT.md).

## 3. Commits
1. 35fb6443b model: Qwen3.5 0.8B, 2B and 9B configs place step threads as siblings (replaces 9a5c3909's change to auto, per the amendment)
2. fcd2ba363 engine: one-launch GDN prefill input stage on Hopper too (917eb8a7, with the guard-test updates that keep the tree green)
3. f4e5dbcba model: an 8192 token bucket for Qwen3.5 vision prefill (afe4633c)
4. 38a349389 model: a 1088 token bucket for Qwen3.5-9B prefill (new: replaces the bucket cut, see concept 4)
5. cd21d1bfe test: a staged GDN mixed split on Hopper against the chunked path (b851763d)
6. 6ef07892b test: Qwen3.5 vision prefill ladder top, and the 9B ladder's 1088 bucket (new)

- Dropped: the change to `auto` (9a5c3909, c39528b9) and its docs, per the amendment. rp2's bucket cut (c67586f6, 59d983e5, b8ace801), which lost to the 1088 bucket. GatherPolicy (f35f5c34, 9a228e98) and the Talker masks (78d41d65, afbf77c9) belong to the Qwen3-Omni PR.
- No message carries a cherry-pick line or any other trailer.
- Docs: every edit is in both `docs/*.rst` and `docs/mintlify/*.mdx`.

## 4. Concepts

### Concept 1: `knobs: step_thread_core: siblings` in configs/qwen3_5_{0.8b,2b,9b}.yaml
- Changes: the three configs; docs/serving.rst and docs/mintlify/serving.mdx gain the EPYC 9554 row as a measurement of `siblings`, and the Python frontend's cost. main's `auto` is unchanged; other models and hosts keep it.
- Mechanism: on this host every handoff between a worker's main, GPU and plan threads waits for a core to leave C2 (800 us). `siblings` puts the three threads on one core's two hardware threads, which keeps that core awake.
- Placement on this pod (server log): "Worker worker_0: step threads (main, GPU, plan) on CPUs 64,192 (siblings: GPU 1 of 4 on its NUMA node, over its allowed CPUs 64-127,192-255 ...)", and the API server's message thread and frontend bridge on CPU 127. Main logs "result threads not pinned (not pinned: auto is measured on Intel hosts only and this host's is AuthenticAMD ...)". Host: 2x EPYC 9554; idle states POLL 0, C1 1, C2 800 us; cgroup quota 27.2 CPUs; governor schedutil.
- Own effect: the ladder of Nov 6 (same layout, then reached through auto): 0.8B ShareGPT c=64 7682 to 14314 tok/s, Random c=64 10542 to 15909, 2B MMMU c=16 2397 to 3094 (main against concept 1 alone, median of 3).
- **Cost, Python frontend (the default unless `--rust-frontend`), 0.8B, 3 alternating rounds at this head:**

|cell|M* main 305e80e tok/s|M* main 305e80e TTFT p50 ms|M* pr-qwen35 6ef0789 tok/s|M* pr-qwen35 6ef0789 TTFT p50 ms|ratio tok/s|
|---|---|---|---|---|---|

- Cost, Rust frontend: Random 1024/256 TTFT p50 rises at higher concurrency along with the arrival rate (section 5 table).
- Tests: none. A config value; `test_step_threads.py` already covers `siblings` placement.

### Concept 2: GDN one-launch prefill input stage on Hopper
- Changes: mstar/engine/resources/linear_attn/gdn_stage.py (`gdn_stage_applies`, capability 10 to 9), a comment in config.py.
- Mechanism: on an H200 the eager path around FlashInfer's chunked kernel is about 250 small kernels and 1 ms of kernel time per mixed step (trace of Qwen3.5-0.8B ShareGPT c=64 on Nov 4: 932 kernels against decode's 335). Blackwell's one-launch Triton stage does that work in one kernel. Off NVIDIA the check stays false (mstar/platforms/interface.py:49).
- Own effect: A/B of the stage alone at this head. The arm without it is the head with that one line at capability 10. 0.8B, 3 rounds, fresh server each, interleaved in the sweep's rounds:

|0.8B cell|without the stage: tok/s; TTFT p50|head: tok/s; TTFT p50|ratio|
|---|---|---|---|
|Random 1024/256 c=1|864 [864-864]; 23.5 [23.5-23.5]|860 [860-861]; 21.9 [21.4-22.5]|1.00|
|Random 1024/256 c=16|8198 [8198-8198]; 70.3 [70.3-70.3]|8147 [8142-8151]; 64.1 [63.2-64.9]|0.99|
|Random 1024/256 c=64|16258 [16258-16258]; 85.2 [85.2-85.2]|15341 [15124-15557]; 101.7 [94.0-109.3]|0.94|
|Random 1024/256 c=128|18199 [18199-18199]; 122.1 [122.1-122.1]|18842 [18751-18932]; 169.1 [163.1-175.1]|1.04|
|ShareGPT c=1|908 [908-908]; 12.7 [12.7-12.7]|881 [867-895]; 13.7 [12.5-14.8]|0.97|
|ShareGPT c=16|8532 [8532-8532]; 16.1 [16.1-16.1]|8369 [8140-8598]; 16.4 [15.6-17.3]|0.98|
|ShareGPT c=64|16105 [16105-16105]; 26.5 [26.5-26.5]|15920 [15349-16491]; 26.4 [25.7-27.2]|0.99|
|ShareGPT c=128|21418 [21418-21418]; 37.6 [37.6-37.6]|19605 [18708-20502]; 39.8 [38.8-40.8]|0.92|

- Cost: it moves greedy outputs by rounding. On Nov 6, 2B images against main went from 11-14 of 16 identical (concept 1 alone) to 9-11 (with the stage), while staying stable against themselves (14 of 16). Agreement with HF stays above the bar (section 6).
- Hopper runs on the H200 at this head: test_a_staged_mixed_split_matches_the_chunked_path (15 cases) and test_chunked_kernel_agrees_through_either_path: 16 passed. The GDN, Qwen3.5, GLM5-Next and Nemotron Duplex modular files: 496 passed, 1 failed (test_duplex_load_audio_refuses_a_file_past_the_session, in the base set).
- Applies to every GDN model on Hopper. GLM5-Next and Nemotron Duplex were not run with real weights.
- Tests: test/modular/test_gdn_prefill_stage.py (test_guard_takes_hopper_and_blackwell_with_one_head_dim, updated in the code commit) and test/modular/test_gdn_mixed_split.py::test_a_staged_mixed_split_matches_the_chunked_path, which fails on the parent (15 cases, needs a GPU).

### Concept 3: 8192 token bucket for vision prefill
- Changes: mstar/model/qwen3_5/submodules.py:146; docs/models.rst and .mdx.
- Mechanism: a prefill_vision step past 4096 tokens ran eagerly, and the first such step on a fresh server stalled every row in flight (5.6 s for a 5360-token image on 2B). It now replays a capture. The tower's block loop still tops out at 16384 patches, so such an image's tower runs uncaptured.
- Own effect (Nov 6 ladder): 2B MMMU c=16 3216 to 4275 tok/s, with non-overlapping rounds.
- 9B boot time and free memory after capture, from the sweep's servers:
  - main: r1 boot 473 s, 94279 MiB free, largest eager 403 MiB
  - head: r1 boot 461 s, 94027 MiB free, largest eager 705 MiB
  - The head's 9B also carries concept 4's 248 MiB.
- 27B TP4 needs 4 GPUs and this pod has one, so it wasn't run. The 27B and Qwen3.8 configs inherit the bucket.
- Compile count is 83 of Dynamo's 84 on the default ladder (test_shipped_ladders_stay_inside_dynamos_recompile_limit).
- Tests: test_qwen3_5_bucket_knobs.py::test_vision_prefill_ladder_tops_at_8192, which fails on the parent.

### Concept 4: a 1088 token bucket for 9B prefill (instead of the bucket cut)
- Changes: configs/qwen3_5_9b.yaml (ladder and header comment), docs/models.rst and .mdx.
- Mechanism: a 1024-token input with its chat template is about 1034 tokens, so it replayed the 2048 capture. 1088 holds it.
- Against the cut, on the same tree (9B, 3 alternating rounds; table in PR_BUCKET_CUT.md): Random c=1 TTFT p50 45.4 against 49.0 ms, and Random c=16 2193 against 2168 tok/s, both non-overlapping; the other cells are level. Against no bucket and no cut, measured on Nov 6: c=1 TTFT p50 about 58 ms.
- Cost: 248 MiB more capture (94273 to 94025 MiB free); largest eager step unchanged.
- Tests: test_qwen3_5_bucket_knobs.py::test_a_1034_token_prompt_replays_1088_on_the_9b_ladder, which fails on the parent. test_shipped_ladders_stay_inside_dynamos_recompile_limit covers the new ladder.

## 5. Main against PR
Hypothesis "PR check: qwen35", rows started Nov 7. `M* main 305e80e` (baseline) against `M* pr-qwen35 6ef0789` (candidate). 3 alternating rounds, fresh server and fresh inductor cache per round, 0 failed requests. --rust-frontend, MSTAR_SHM_ARENA=1, each tree's config as it stands, so main has step_thread_core at default auto (pins nothing here) and the head has siblings from its configs. -O1, CPU frequency knobs off, no uclampset. On 2B and 9B the image cells run first on the server. Median [min-max].

|cell|M* main 305e80e tok/s|M* main 305e80e TTFT p50 ms|M* pr-qwen35 6ef0789 tok/s|M* pr-qwen35 6ef0789 TTFT p50 ms|ratio tok/s|
|---|---|---|---|---|---|
|Qwen3.5-0.8B Random 1024/256 c=1 (n=2,2)|488 [479-498]|26.9 [26.7-27.0]|860 [860-861]|21.9 [21.4-22.5]|1.76|
|Qwen3.5-0.8B Random 1024/256 c=16 (n=2,2)|5286 [5046-5526]|63.1 [62.0-64.3]|8147 [8142-8151]|64.1 [63.2-64.9]|1.54|
|Qwen3.5-0.8B Random 1024/256 c=64 (n=2,2)|9786 [9322-10249]|92.7 [82.1-103.3]|15341 [15124-15557]|101.7 [94.0-109.3]|1.57|
|Qwen3.5-0.8B Random 1024/256 c=128 (n=2,2)|12165 [12069-12261]|117.9 [110.9-124.9]|18842 [18751-18932]|169.1 [163.1-175.1]|1.55|
|Qwen3.5-0.8B ShareGPT c=1 (n=2,2)|490 [465-515]|18.4 [18.4-18.5]|881 [867-895]|13.7 [12.5-14.8]|1.80|
|Qwen3.5-0.8B ShareGPT c=16 (n=2,2)|4492 [4136-4847]|24.4 [22.9-25.8]|8369 [8140-8598]|16.4 [15.6-17.3]|1.86|
|Qwen3.5-0.8B ShareGPT c=64 (n=2,2)|7974 [7956-7992]|38.6 [38.5-38.7]|15920 [15349-16491]|26.4 [25.7-27.2]|2.00|
|Qwen3.5-0.8B ShareGPT c=128 (n=2,2)|9815 [9695-9936]|56.5 [55.9-57.0]|19605 [18708-20502]|39.8 [38.8-40.8]|2.00|
|Qwen3.5-2B Image (MMMU) c=4 (n=1,1)|910 [910-910]|92.8 [92.8-92.8]|1414 [1414-1414]|66.4 [66.4-66.4]|1.55|
|Qwen3.5-2B Image (MMMU) c=16 (n=1,1)|3726 [3726-3726]|121.5 [121.5-121.5]|4448 [4448-4448]|93.0 [93.0-93.0]|1.19|
|Qwen3.5-2B Random 1024/256 c=1 (n=1,1)|517 [517-517]|32.7 [32.7-32.7]|604 [604-604]|26.5 [26.5-26.5]|1.17|
|Qwen3.5-2B Random 1024/256 c=16 (n=1,1)|4489 [4489-4489]|79.4 [79.4-79.4]|6144 [6144-6144]|78.3 [78.3-78.3]|1.37|
|Qwen3.5-2B Random 1024/256 c=64 (n=1,1)|11382 [11382-11382]|91.2 [91.2-91.2]|13698 [13698-13698]|112.8 [112.8-112.8]|1.20|
|Qwen3.5-2B Random 1024/256 c=128 (n=1,1)|13967 [13967-13967]|120.8 [120.8-120.8]|16500 [16500-16500]|109.5 [109.5-109.5]|1.18|
|Qwen3.5-2B ShareGPT c=1 (n=1,1)|442 [442-442]|19.8 [19.8-19.8]|632 [632-632]|14.3 [14.3-14.3]|1.43|
|Qwen3.5-2B ShareGPT c=16 (n=1,1)|4370 [4370-4370]|24.7 [24.7-24.7]|6517 [6517-6517]|17.1 [17.1-17.1]|1.49|
|Qwen3.5-2B ShareGPT c=64 (n=1,1)|7400 [7400-7400]|40.7 [40.7-40.7]|13814 [13814-13814]|26.8 [26.8-26.8]|1.87|
|Qwen3.5-2B ShareGPT c=128 (n=1,1)|9434 [9434-9434]|59.6 [59.6-59.6]|18863 [18863-18863]|38.5 [38.5-38.5]|2.00|
|Qwen3.5-9B Image (MMMU) c=4 (n=1,1)|471 [471-471]|101.6 [101.6-101.6]|571 [571-571]|94.7 [94.7-94.7]|1.21|
|Qwen3.5-9B Image (MMMU) c=16 (n=1,1)|1886 [1886-1886]|229.4 [229.4-229.4]|1830 [1830-1830]|171.4 [171.4-171.4]|0.97|
|Qwen3.5-9B Random 1024/256 c=1 (n=1,1)|211 [211-211]|60.9 [60.9-60.9]|201 [201-201]|42.3 [42.3-42.3]|0.95|
|Qwen3.5-9B Random 1024/256 c=64 (n=1,1)|4473 [4473-4473]|277.1 [277.1-277.1]|4561 [4561-4561]|265.7 [265.7-265.7]|1.02|
|Qwen3.5-9B ShareGPT c=1 (n=1,1)|217 [217-217]|26.6 [26.6-26.6]|204 [204-204]|22.9 [22.9-22.9]|0.94|
|Qwen3.5-9B ShareGPT c=64 (n=1,1)|5240 [5240-5240]|49.4 [49.4-49.4]|5443 [5443-5443]|46.9 [46.9-46.9]|1.04|

**Main's modes:** 0.8B ShareGPT c=64 by round: r1 7956 (slow); r2 7992 (slow). A fast round is one where main reads about 14000 without pinning. Nov 6 traced fast rounds to time windows rather than workload order; their cause is a hypothesis (host load keeping cores out of C2), and host samples were taken through this sweep (results hostsample.tsv). The table compares against whichever mode main was in.

## 6. Correctness
Teacher-forced top-1 against transformers sdpa, 64 greedy tokens per prompt, on 64 ShareGPT prompts (and 16 MMMU images for 2B and 9B), from each of the 3 sweep boots per tree. The bar is transformers eager greedy, scored the same way.
not scored (out of time)

On Nov 6 every 9B miss, the bar's and both trees', was a near-tie: a log-prob gap of at most 0.25 nats under sdpa at bf16. Main missed the 9B bar the same way.

Greedy agreement across boots (identical of N, mean prefix), PR-main, main-main and PR-PR:
```
0.8 text main-main [(57, 64, 0.9753)]
0.8 text pr-main [(59, 64, 0.978), (56, 64, 0.9761), (56, 64, 0.9736), (56, 64, 0.9619)]
0.8 text pr-pr [(56, 64, 0.9634)]
0.8 image main-main [(14, 16, 0.915)]
0.8 image pr-main [(12, 16, 0.8369), (12, 16, 0.8262), (13, 16, 0.8867), (13, 16, 0.8975)]
0.8 image pr-pr [(11, 16, 0.8311)]
2 text main-main []
2 text pr-main [(52, 64, 0.8892)]
2 text pr-pr []
2 image main-main []
2 image pr-main [(11, 16, 0.8506)]
2 image pr-pr []
9 text main-main []
9 text pr-main [(51, 64, 0.8875)]
9 text pr-pr []
9 image main-main []
9 image pr-main [(13, 16, 0.9316)]
9 image pr-pr []
```

## 7. Tests
- Base 305e80e, serial and -n 8 --dist loadfile, with identical failing ids: 50 failed, 7 errors, 10898 passed, 1290 skipped.
- Head: 50 failed, 10901 passed, 1305 skipped, 427 warnings, 7 errors, 38 subtests passed in 265.83s (0:04:25). Same failing ids as the base.
- Per commit: ruff clean; targeted test/modular files show no failure outside the base set.
- The two commits that touch no `mstar/` module had no targeted files, so their per-commit runs covered the whole repo. Those runs show failures in test/chatterbox, test/integration, test/ci, test/omnivoice and test/modular/test_eval_*, and none of them touch Qwen3.5 configs. They were not compared against the base, because the base set is test/modular only.
- Pre-existing failures: torchaudio cannot load against torch 2.12.1 and was uninstalled, hence the 7 collection errors and the audio failures. Also test_startup_graph_validation (11 configs), test_lora_targets and test_nemotron_duplex_model. Full ids are on the pod in results/rp3-pr/modular_base_par.set.

## 8. Review
- First independent review (Nov 6, subagent): its answers and outcomes are in this file's git history at ae5d0a5. They covered the bucket cut, now dropped, and auto's shape, now replaced.
- Gaurav's review decisions (amendment):
  - main's auto kept, with siblings in the three configs: done.
  - Split into two branches: done; the cut branch was then dropped by the amendment's own rule.
  - Cut reworked to re-share at `budget=lower`, with a start-order test and a floor test, docstring fixes, and 512's rule in a comment: done on perf/rp3-pr-bucket-cut (8e692b7a5) before the comparison.
  - Cut against 1088: 1088 kept.
  - Hopper test runs and a stage-alone A/B: recorded above.
  - 9B boot time and memory: recorded above; 27B TP4 not runnable here.
  - Long docs lines: the cut's docs were rewrapped; the remaining long lines are the one-paragraph-per-line style of the .mdx files.
  - TestCutToBucket assert messages: done on the cut branch.
  - The 5360-token test was renamed test_vision_prefill_ladder_tops_at_8192.

## 9. Open
- The Python frontend's TTFT under siblings: the configs now set siblings, and `mstar serve` without `--rust-frontend` pays it (concept 1 table).
- Main's two modes: the cause is a hypothesis.
- The 9B gate: misses on main too, and the cause of eager's lead on images is not established.
- TP, multi-GPU, 27B, Qwen3.8 and Blackwell: not run.
- 1088 covers only prompts near 1034 tokens. A 1200-token prompt still pads to 2048.
- Numbers are at -O1.
- Qwen3-Omni weights were removed from this pod's HF cache. They can be downloaded again.
