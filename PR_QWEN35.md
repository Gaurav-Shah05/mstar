IN PROGRESS: Python-frontend rounds at 6ef0789 running; every other number below is final.

Read first:
- **The gate misses on a strict reading.** 9B images fail on all 3 PR boots (1015-1017 against a bar of 1019; main's boots read 1013-1019). One boot each of 0.8B and 9B text misses by 1-2 positions; main misses the same way. Section 6.
- **9B MMMU c=16 is 1.5% lower on the PR with non-overlapping rounds**: 1838 [1830-1842] against 1866 [1863-1886] tok/s (concept 3).
- **Python-frontend TTFT under `siblings`**: concept 1.
- **Main was in its slow mode in all 3 rounds**, so the ratios are against that mode. Section 5.

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
- Own effect: A/B of the stage alone at this head. Random c=16 TTFT p50 goes from 72.0 to 63.2 ms and Random c=128 tok/s rises 3% (18259 to 18751), both non-overlapping. The other cells overlap. The arm without it is the head with that one line at capability 10. 0.8B, 3 rounds, fresh server each, interleaved in the sweep's rounds:

|0.8B cell|without the stage: tok/s; TTFT p50|head: tok/s; TTFT p50|ratio|
|---|---|---|---|
|Random 1024/256 c=1|864 [855-869]; 23.5 [23.1-24.8]|861 [860-864]; 22.5 [21.4-22.5]|1.00|
|Random 1024/256 c=16|8023 [8005-8198]; 72.0 [70.3-74.1]|8151 [8142-8209]; 63.2 [56.6-64.9]|1.02|
|Random 1024/256 c=64|15573 [15442-16258]; 101.0 [85.2-106.8]|15557 [15124-16172]; 94.0 [82.1-109.3]|1.00|
|Random 1024/256 c=128|18259 [18199-18616]; 172.9 [122.1-183.9]|18751 [18692-18932]; 163.1 [145.2-175.1]|1.03|
|ShareGPT c=1|896 [880-908]; 13.3 [12.7-16.1]|895 [867-903]; 13.2 [12.5-14.8]|1.00|
|ShareGPT c=16|8164 [7913-8532]; 16.3 [16.1-17.5]|8567 [8140-8598]; 15.6 [15.2-17.3]|1.05|
|ShareGPT c=64|15330 [14728-16105]; 27.8 [26.5-28.9]|15703 [15349-16491]; 27.2 [25.7-27.5]|1.02|
|ShareGPT c=128|20480 [19222-21418]; 38.1 [37.6-38.8]|20502 [18708-21411]; 38.8 [37.2-40.8]|1.00|

- Cost: it moves greedy outputs by rounding. On Nov 6, 2B images against main went from 11-14 of 16 identical (concept 1 alone) to 9-11 (with the stage), while staying stable against themselves (14 of 16). Agreement with HF stays above the bar (section 6).
- Hopper runs on the H200 at this head: test_a_staged_mixed_split_matches_the_chunked_path (15 cases) and test_chunked_kernel_agrees_through_either_path: 16 passed. The GDN, Qwen3.5, GLM5-Next and Nemotron Duplex modular files: 496 passed, 1 failed (test_duplex_load_audio_refuses_a_file_past_the_session, in the base set).
- Applies to every GDN model on Hopper. GLM5-Next and Nemotron Duplex were not run with real weights.
- Tests: test/modular/test_gdn_prefill_stage.py (test_guard_takes_hopper_and_blackwell_with_one_head_dim, updated in the code commit) and test/modular/test_gdn_mixed_split.py::test_a_staged_mixed_split_matches_the_chunked_path, which fails on the parent (15 cases, needs a GPU).

### Concept 3: 8192 token bucket for vision prefill
- Changes: mstar/model/qwen3_5/submodules.py:146; docs/models.rst and .mdx.
- Mechanism: a prefill_vision step past 4096 tokens ran eagerly, and the first such step on a fresh server stalled every row in flight (5.6 s for a 5360-token image on 2B). It now replays a capture. The tower's block loop still tops out at 16384 patches, so such an image's tower runs uncaptured.
- Own effect (Nov 6 ladder): 2B MMMU c=16 3216 to 4275 tok/s, with non-overlapping rounds.
- 9B boot time and free memory after capture, from the sweep's servers:
  - main: r1 boot 473 s, 94279 MiB free, largest eager 403 MiB; r2 boot 458 s, 94279 MiB free, largest eager 403 MiB; r3 boot 410 s, 94279 MiB free, largest eager 403 MiB
  - head: r1 boot 461 s, 94027 MiB free, largest eager 705 MiB; r2 boot 410 s, 94027 MiB free, largest eager 705 MiB; r3 boot 462 s, 94027 MiB free, largest eager 705 MiB
  - The head's 9B also carries concept 4's 248 MiB.
- Cost: 9B MMMU c=16 tok/s is 1.5% lower than main with non-overlapping rounds (section 5), and c=4 is 21% higher. On Nov 5 this bucket alone measured 9B c=16 about 2% lower than without it; the cause was not found.
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
|Qwen3.5-0.8B Random 1024/256 c=1 (n=3,3)|479 [444-498]|27.0 [26.7-27.4]|861 [860-864]|22.5 [21.4-22.5]|1.80|
|Qwen3.5-0.8B Random 1024/256 c=16 (n=3,3)|5214 [5046-5526]|64.3 [62.0-64.6]|8151 [8142-8209]|63.2 [56.6-64.9]|1.56|
|Qwen3.5-0.8B Random 1024/256 c=64 (n=3,3)|10249 [9322-10444]|97.4 [82.1-103.3]|15557 [15124-16172]|94.0 [82.1-109.3]|1.52|
|Qwen3.5-0.8B Random 1024/256 c=128 (n=3,3)|12261 [12069-13926]|124.9 [110.9-141.4]|18751 [18692-18932]|163.1 [145.2-175.1]|1.53|
|Qwen3.5-0.8B ShareGPT c=1 (n=3,3)|480 [465-515]|18.4 [17.9-18.5]|895 [867-903]|13.2 [12.5-14.8]|1.86|
|Qwen3.5-0.8B ShareGPT c=16 (n=3,3)|4354 [4136-4847]|23.5 [22.9-25.8]|8567 [8140-8598]|15.6 [15.2-17.3]|1.97|
|Qwen3.5-0.8B ShareGPT c=64 (n=3,3)|7956 [7327-7992]|38.7 [38.5-40.2]|15703 [15349-16491]|27.2 [25.7-27.5]|1.97|
|Qwen3.5-0.8B ShareGPT c=128 (n=3,3)|9695 [9664-9936]|57.0 [55.9-58.3]|20502 [18708-21411]|38.8 [37.2-40.8]|2.11|
|Qwen3.5-2B Image (MMMU) c=4 (n=3,3)|880 [870-910]|91.3 [82.8-92.8]|1385 [1380-1414]|65.5 [64.0-66.4]|1.57|
|Qwen3.5-2B Image (MMMU) c=16 (n=3,3)|3726 [3381-3770]|121.5 [104.1-132.9]|4354 [4334-4448]|95.9 [93.0-100.5]|1.17|
|Qwen3.5-2B Random 1024/256 c=1 (n=3,3)|499 [442-517]|32.7 [31.9-32.9]|590 [582-604]|28.0 [26.5-28.7]|1.18|
|Qwen3.5-2B Random 1024/256 c=16 (n=3,3)|4736 [4489-5054]|82.4 [79.4-83.7]|6118 [6085-6144]|80.2 [78.3-85.4]|1.29|
|Qwen3.5-2B Random 1024/256 c=64 (n=3,3)|11114 [10998-11382]|91.2 [90.3-97.5]|13698 [13599-13766]|119.8 [112.8-120.0]|1.23|
|Qwen3.5-2B Random 1024/256 c=128 (n=3,3)|12569 [11257-13967]|123.2 [120.8-148.7]|16500 [16373-16516]|133.0 [109.5-164.4]|1.31|
|Qwen3.5-2B ShareGPT c=1 (n=3,3)|442 [439-471]|20.2 [19.8-20.4]|618 [605-632]|16.3 [14.3-17.9]|1.40|
|Qwen3.5-2B ShareGPT c=16 (n=3,3)|4494 [4370-4558]|24.2 [23.5-24.7]|6352 [6334-6517]|18.3 [17.1-18.5]|1.41|
|Qwen3.5-2B ShareGPT c=64 (n=3,3)|7686 [7400-7740]|40.4 [39.7-40.7]|13089 [13050-13814]|29.4 [26.8-30.4]|1.70|
|Qwen3.5-2B ShareGPT c=128 (n=3,3)|9728 [9434-9830]|57.7 [57.6-59.6]|18203 [18128-18863]|40.7 [38.5-40.9]|1.87|
|Qwen3.5-9B Image (MMMU) c=4 (n=3,3)|471 [471-473]|103.3 [101.6-114.9]|571 [566-571]|94.7 [94.3-111.4]|1.21|
|Qwen3.5-9B Image (MMMU) c=16 (n=3,3)|1866 [1863-1886]|201.9 [190.8-229.4]|1838 [1830-1842]|184.3 [171.4-190.8]|0.98|
|Qwen3.5-9B Random 1024/256 c=1 (n=3,3)|198 [198-211]|63.1 [60.9-64.8]|201 [200-201]|42.8 [42.3-43.0]|1.02|
|Qwen3.5-9B Random 1024/256 c=64 (n=3,3)|4474 [4473-4478]|276.9 [275.2-277.1]|4561 [4552-4561]|268.1 [265.7-269.4]|1.02|
|Qwen3.5-9B ShareGPT c=1 (n=3,3)|204 [204-217]|25.7 [24.7-26.6]|204 [204-205]|23.3 [22.9-25.2]|1.00|
|Qwen3.5-9B ShareGPT c=64 (n=3,3)|5257 [5240-5292]|49.4 [48.3-50.1]|5438 [5430-5443]|46.9 [46.6-48.6]|1.03|

**Main's modes:** 0.8B ShareGPT c=64 by round: r1 7956 (slow); r2 7992 (slow); r3 7327 (slow). A fast round is one where main reads about 14000 without pinning. Nov 6 traced fast rounds to time windows rather than workload order; their cause is a hypothesis (host load keeping cores out of C2), and host samples were taken through this sweep (results hostsample.tsv). The table compares against whichever mode main was in.

## 6. Correctness
Teacher-forced top-1 against transformers sdpa, 64 greedy tokens per prompt, on 64 ShareGPT prompts (and 16 MMMU images for 2B and 9B), from each of the 3 sweep boots per tree. The bar is transformers eager greedy, scored the same way.

0.8B:
```
hf eager   text  4081/4096 = 0.99634

PASS text  4085/4096 = 0.99731 6ef0789/0.8b/r1/parity.json

PASS text  4086/4096 = 0.99756 6ef0789/0.8b/r2/parity.json

FAIL text  4080/4096 = 0.99609 6ef0789/0.8b/r3/parity.json

PASS text  4082/4096 = 0.99658 main/0.8b/r1/parity.json

PASS text  4081/4096 = 0.99634 main/0.8b/r2/parity.json

FAIL text  4079/4096 = 0.99585 main/0.8b/r3/parity.json

```
2B:
```
hf eager   text  4075/4096 = 0.99487

PASS text  4083/4096 = 0.99683 6ef0789/2b/r1/parity.json

PASS text  4084/4096 = 0.99707 6ef0789/2b/r2/parity.json

PASS text  4080/4096 = 0.99609 6ef0789/2b/r3/parity.json

PASS text  4079/4096 = 0.99585 main/2b/r1/parity.json

PASS text  4080/4096 = 0.99609 main/2b/r2/parity.json

PASS text  4076/4096 = 0.99512 main/2b/r3/parity.json

hf eager   image 1010/1024 = 0.98633

PASS image 1015/1024 = 0.99121 6ef0789/2b/r1/parity.json

PASS image 1015/1024 = 0.99121 6ef0789/2b/r2/parity.json

PASS image 1014/1024 = 0.99023 6ef0789/2b/r3/parity.json

PASS image 1015/1024 = 0.99121 main/2b/r1/parity.json

PASS image 1012/1024 = 0.98828 main/2b/r2/parity.json

PASS image 1015/1024 = 0.99121 main/2b/r3/parity.json

```
9B:
```
hf eager   text  4080/4096 = 0.99609

PASS text  4081/4096 = 0.99634 6ef0789/9b/r1/parity.json

PASS text  4083/4096 = 0.99683 6ef0789/9b/r2/parity.json

FAIL text  4078/4096 = 0.99561 6ef0789/9b/r3/parity.json

PASS text  4080/4096 = 0.99609 main/9b/r1/parity.json

FAIL text  4078/4096 = 0.99561 main/9b/r2/parity.json

PASS text  4081/4096 = 0.99634 main/9b/r3/parity.json

hf eager   image 1019/1024 = 0.99512

FAIL image 1015/1024 = 0.99121 6ef0789/9b/r1/parity.json

FAIL image 1015/1024 = 0.99121 6ef0789/9b/r2/parity.json

FAIL image 1017/1024 = 0.99316 6ef0789/9b/r3/parity.json

FAIL image 1017/1024 = 0.99316 main/9b/r1/parity.json

FAIL image 1013/1024 = 0.98926 main/9b/r2/parity.json

PASS image 1019/1024 = 0.99512 main/9b/r3/parity.json

```

On Nov 6 every 9B miss, the bar's and both trees', was a near-tie: a log-prob gap of at most 0.25 nats under sdpa at bf16. Main missed the 9B bar the same way.

Greedy agreement across boots (identical of N, mean prefix), PR-main, main-main and PR-PR:
```
0.8 text main-main [(57, 64, 0.9753), (53, 64, 0.9534), (55, 64, 0.9651)]
0.8 text pr-main [(59, 64, 0.978), (56, 64, 0.9761), (52, 64, 0.95), (56, 64, 0.9736), (56, 64, 0.9619), (52, 64, 0.9604), (58, 64, 0.97), (53, 64, 0.9524), (52, 64, 0.939)]
0.8 text pr-pr [(56, 64, 0.9634), (56, 64, 0.9551), (55, 64, 0.9536)]
0.8 image main-main [(14, 16, 0.915), (12, 16, 0.8037), (13, 16, 0.8477)]
0.8 image pr-main [(12, 16, 0.8369), (12, 16, 0.8262), (13, 16, 0.8936), (13, 16, 0.8867), (13, 16, 0.8975), (12, 16, 0.8525), (14, 16, 0.8984), (14, 16, 0.9092), (13, 16, 0.8643)]
0.8 image pr-pr [(11, 16, 0.8311), (12, 16, 0.8428), (15, 16, 0.9883)]
2 text main-main [(53, 64, 0.9346), (51, 64, 0.9001), (49, 64, 0.9077)]
2 text pr-main [(52, 64, 0.8892), (49, 64, 0.884), (47, 64, 0.8772), (52, 64, 0.8872), (51, 64, 0.8953), (49, 64, 0.8826), (47, 64, 0.8662), (49, 64, 0.8831), (43, 64, 0.8452)]
2 text pr-pr [(56, 64, 0.9424), (54, 64, 0.9319), (52, 64, 0.9148)]
2 image main-main [(11, 16, 0.9121), (12, 16, 0.959), (13, 16, 0.9375)]
2 image pr-main [(11, 16, 0.8506), (12, 16, 0.8467), (13, 16, 0.876), (11, 16, 0.8867), (9, 16, 0.8223), (12, 16, 0.8848), (12, 16, 0.9248), (11, 16, 0.8643), (14, 16, 0.9268)]
2 image pr-pr [(10, 16, 0.8428), (12, 16, 0.9033), (13, 16, 0.9326)]
9 text main-main [(51, 64, 0.8921), (53, 64, 0.9343), (52, 64, 0.9238)]
9 text pr-main [(51, 64, 0.8875), (51, 64, 0.8994), (57, 64, 0.9336), (53, 64, 0.9075), (49, 64, 0.8784), (55, 64, 0.9375), (52, 64, 0.9124), (49, 64, 0.8879), (54, 64, 0.8979)]
9 text pr-pr [(60, 64, 0.9771), (55, 64, 0.9082), (55, 64, 0.9263)]
9 image main-main [(12, 16, 0.9111), (14, 16, 0.9736), (11, 16, 0.8916)]
9 image pr-main [(13, 16, 0.9316), (14, 16, 0.9355), (12, 16, 0.9238), (13, 16, 0.9141), (14, 16, 0.9531), (12, 16, 0.9062), (13, 16, 0.9453), (12, 16, 0.9062), (14, 16, 0.9531)]
9 image pr-pr [(14, 16, 0.9395), (14, 16, 0.9707), (12, 16, 0.9102)]
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
- 9B MMMU c=16: 1.5% lower than main; cause not found.
- TP, multi-GPU, 27B, Qwen3.8 and Blackwell: not run.
- 1088 covers only prompts near 1034 tokens. A 1200-token prompt still pads to 2048.
- Numbers are at -O1.
- Qwen3-Omni weights were removed from this pod's HF cache. They can be downloaded again.
