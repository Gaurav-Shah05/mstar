IN PROGRESS: amendment of Nov 7, split into two branches; numbers below are from the earlier head 6bfc962 and will be replaced

Read first: the 9B correctness gate misses its bar on most sets, on main as well as on the PR (section 6). Concept 1's 2x reading is against main's slow mode, and main isn't always in it (section 4). Python-frontend TTFT at c=64 gets about 12x worse (concept 1).

# PR Qwen3.5 facts

## 2. SHAs
- Base: 305e80ef46248ade6e6670aa20c9bb67f4ac2e26
- Head: 6bfc96288efcb83a3bc557e7c276166e82a53fa5
- Branch: perf/rp3-pr-qwen35
- Rows are labelled 362e11d, the head before a docs-only correction of the Python-frontend numbers in docs/serving.rst and docs/mintlify/serving.mdx. `git diff 362e11d46 6bfc96288` touches only those two files. The test run at the end is on 6bfc962.

## 3. Commits (in order, with source)
1. 9d52854f2 engine: step_thread_core auto takes siblings where cores sleep deep (9a5c3909, plus the test updates from c39528b9 that keep the tree green)
2. f4f71d5f5 test: step_thread_core auto declines without an address or room, and takes the GPU lock (c39528b9)
3. 65c8b32eb engine: one-launch GDN prefill input stage on Hopper too (917eb8a7, plus the guard test updates from b851763d)
4. 8ab3ddb1d test: a staged GDN mixed split on Hopper against the chunked path (b851763d)
5. 255fe3107 model: an 8192 token bucket for Qwen3.5 vision prefill (afe4633c)
6. de4a86936 test: a 5360-token image prefill has a captured Qwen3.5 bucket (new)
7. 719990db6 worker: cut a step at the capture bucket below when that avoids padding (rp2 c67586f6 + 59d983e5)
8. 6bfc96288 test: a bucket cut needs a step of chunkable rows alone, through the real builder (rp2's tests from those commits and b8ace801)

Dropped: f35f5c34, 9a228e98 (GatherPolicy) and 78d41d65, afbf77c9 (Talker masks): assigned to the Qwen3-Omni PR. Nothing dropped because main already did it, and no commit dropped with a revert.

Ported or changed:
- Docs: main moved guides to Mintlify (#1135) and AGENTS.md requires .rst and .mdx together, so every doc edit now also lands in docs/mintlify/.
- Commit 1: "With the cores held awake, siblings loses" now names the 6767P, since the row this PR adds shows siblings ahead of the spinner on the EPYC. Python-frontend numbers remeasured on this base.
- Commits 1 and 3 carry the updates to existing tests that asserted the old behaviour. The source commits left the code commit red (248 and 2 failures).
- Commits 2 and 4: strengthened so they fail without the code. The small-mask test also asserts the reason; the staged split test now goes through gdn_stage_applies. Both passed on the parent before.
- Commit 6: new. Concept 3 had no test.
- Commit 7: squashed from two rp2 commits, and reshaped after the review (section 8). Subject scope moved from Scheduler to worker. rp2's docs numbers replaced with this pod's.

## 4. Concepts

### Concept 1: step_thread_core auto takes siblings where cores sleep deep
- Changes: mstar/utils/cpus.py, mstar/utils/knobs.py, mstar/api_server/entrypoint.py, docs (environment variables, serving; .rst and .mdx). Default behaviour for every model on Intel or AMD hosts with deep idle.
- Mechanism: a step hands work between a worker's three threads several times. On a host with deep idle states each handoff can wait for a core to exit idle. auto now places a worker's three threads on one core's hardware threads, which keeps that core awake, and declines with one log line where it doesn't apply.
- What auto decides on this pod (server logs): "API server: message thread and frontend bridge on CPU 127 (siblings: GPU 1 of 4 on its NUMA node, over its allowed CPUs 64-127,192-255 ...)" and "Worker worker_0: step threads (main, GPU, plan) on CPUs 64,192". Main logs "result threads not pinned (not pinned: auto is measured on Intel hosts only and this host's is AuthenticAMD ...)".
- Values it reads: /sys/devices/system/cpu/cpu*/cpuidle/state*/latency, name and disable for CPUs in the mask (cpus.py:416, compared with DEEP_IDLE_US=100, cpus.py:73). Here POLL 0, C1 1, C2 800 us, none disabled. Also /proc/cpuinfo vendor_id (cpus.py:394, AUTO_VENDORS cpus.py:84), AuthenticAMD here. Placement reads /sys/bus/pci/devices (numa_node, local_cpulist) and CPU topology as siblings already did. Host: 2x EPYC 9554, 256 CPUs in the mask, cgroup quota 27.2 CPUs, governor schedutil.
- Own effect (ladder, main to c1, median of 3): 0.8B ShareGPT c=64 7682 to 14314 tok/s; 0.8B Random c=64 10542 to 15909; 2B MMMU c=16 2397 to 3094.
- **Main is bimodal, and the gain is conditional.** On identical code and sequence, main 0.8B ShareGPT c=64 read 14642 in ladder round 1 and 13407 then 14370 in one diagnostic server, against 6900-7700 on every other main boot (diagnostic rows: batch h200_1106, status probe, hypothesis "PR diag: workload order"). In its fast mode main matches the PR: ladder round 1 main 14642 against head 14455. Arms with concept 1 stayed at 14287-16861 throughout. All 3 sweep rounds of main landed in the slow mode, so section 5's ratios on 0.8B and 2B are against the slow mode.
  - Workload order was tested and is not the cause: the mode holds for a whole server, across c=64, c=1, c=16 and c=64 again.
  - Hypothesis, not verified: other tenants' load keeps cores out of deep idle during fast periods. Fast boots cluster in time (13:27-14:25 UTC), when the load average read 25-50, though that figure includes this pod's own compiles. Slow boots sampled at 15:46 and 16:43 had 16-23 busy CPUs and C2 residency about 0.9 on the GPU's node. No fast boot was sampled. An earlier run on this host supports the direction: no pin plus SCHED_IDLE spinners on the server's CPUs got most of the gain (0.8B c=64 8031 no pin, 14846 spinner, 15814 siblings; results file, part 3, A4).
- Costs:
  - **Python frontend, the default, 0.8B ShareGPT, 3 rounds:** c=64 TTFT p50 81 to 940 ms (tok/s 8102 to 10358); c=16 TTFT p50 26 to 36 ms (tok/s 4281 to 7978). Table below. Pinning its threads or not made no difference at 9c6f1ad. Rust frontend, same cell: 41 to 24 ms.
  - Random 1024/256 TTFT p50 rises at high concurrency as the arrival rate rises: 0.8B c=64 92.5 to 112.6 ms, 2B c=64 90.2 to 109.5, 2B c=128 122.0 to 160.7. Those medians include the other concepts.
- Other models, off against siblings, measured at 9c6f1ad and not rerun here: Qwen3-ASR c=8 1166 to 1757, c=32 1568 to 2472 tok/s; Kokoro and FLUX.2 klein level.
- Tests: test/modular/test_step_threads.py: test_auto_under_a_mask_too_small_declines_without_a_warning, test_auto_declines_a_device_with_no_pci_address, test_auto_takes_the_lock_when_it_pins, test_auto_on_a_deep_idle_host_takes_siblings, test_always_takes_separate_cores_and_auto_takes_siblings; test_the_api_server_and_the_conductors_workers_build_one_plan gains an auto case. The 10 cases of the first four fail on the base.

Python frontend, 0.8B (rows "PR check: qwen35 python frontend"):

|cell|M* main 305e80e tok/s; TTFT p50|M* pr-qwen35 362e11d tok/s; TTFT p50|ratio last/first|
|---|---|---|---|
|Qwen3.5-0.8B ShareGPT c=16|4281 [4223-4576] n=3; 26.3 [25.8-28.2]|7978 [7696-8102] n=3; 36.4 [35.4-86.6]|1.86|
|Qwen3.5-0.8B ShareGPT c=64|8102 [7964-8131] n=3; 81.3 [79.6-123.3]|10358 [9400-10708] n=3; 939.5 [802.5-1044.8]|1.28|

### Concept 2: GDN one-launch prefill input stage on Hopper
- Changes: mstar/engine/resources/linear_attn/gdn_stage.py:40 (capability 10 to 9), a comment in config.py.
- Mechanism: on a mixed step the eager path around FlashInfer's chunked kernel is about 250 small kernels and 1 ms of kernel time on an H200 (trace at c39528b, 932 kernels against decode's 335). The Triton stage that Blackwell already uses does that work in one launch. Off NVIDIA the check stays false (mstar/platforms/interface.py:49).
- Own effect (ladder, c1 to c2): 0.8B Random c=64 15909 to 18507; 0.8B ShareGPT c=64 14314 to 15150; 2B MMMU c=16 3094 to 3216. On 0.8B the rounds overlap; on 2B MMMU they don't (c1 3092-3118, c2 3211-3240). At c39528b, in a 3-round A/B: 0.8B Random c=64 16315 to 17009, TTFT p50 94 to 74 ms.
- Cost: it moves greedy outputs by rounding. 2B images against main: identical 9-11 of 16, prefix 0.757-0.815, against 11-14 and 0.852-0.969 at c1; stable against itself (14 of 16, 0.947). HF agreement stays above the bar (section 6). Capture memory goes down on 0.8B and 2B: free after capture main to PR 123189 to 123613 MiB and 120193 to 120535; largest eager step 595 to 492 and 645 to 542 MiB. That is the whole PR's reading, attributed here by hypothesis.
- Applies to every GDN model on Hopper (Qwen3.5 sizes, GLM5-Next, Nemotron Duplex). Their modular tests pass on the H200; no real-weights run of the other two.
- Tests: test/modular/test_gdn_prefill_stage.py: test_guard_takes_hopper_and_blackwell_with_one_head_dim, test_guard_refuses_cpu_and_older_gpus. test/modular/test_gdn_mixed_split.py: test_a_staged_mixed_split_matches_the_chunked_path (15 cases fail on the parent, needs a GPU).

### Concept 3: 8192 token bucket for vision prefill
- Changes: mstar/model/qwen3_5/submodules.py:146, docs/models.rst, docs/mintlify/models.mdx. Inherited by every Qwen3.5 size and Qwen3.8.
- Mechanism: a prefill_vision step past the top bucket ran eagerly, and the first such step on a fresh server stalled every row in flight (5.6 s for a 5360-token image on 2B at 9c6f1ad). Now it replays a capture. The tower's block loop still tops out at 16384 patches, so such an image's tower runs uncaptured.
- Own effect (ladder, c2 to c3): 2B MMMU c=16 3216 to 4275 tok/s, TTFT p50 94.0 to 103.3 ms, non-overlapping on tok/s. Text cells overlap.
- Cost: 83 compiles against the limit of 84 (test_shipped_ladders_stay_inside_dynamos_recompile_limit checks it; it was 80), so 1 is spare. 9B: largest eager step 403 to 705 MiB. Free after capture on 9B 94279 to 94273 MiB. Startup not separately timed.
- Tests: test/modular/test_qwen3_5_bucket_knobs.py: test_a_5360_token_image_prefill_has_a_captured_bucket.

### Concept 4: bucket cut (rp2), kept
- Changes: mstar/worker/batch_builder.py:483, mstar/worker/micro_scheduler.py:1327, mstar/engine/chunking.py:197, engine.py, accelerator_graph_runner.py, mstar/model/submodule_base.py, qwen3_5_model.py, configs/qwen3_5_9b.yaml, docs (adding_models, models; .rst and .mdx).
- Mechanism: a step just past a capture bucket pads to the next one up, so a lone 1034-token prompt replays 2048. With `ChunkingPolicy.bucket_cut_min_saved` set, one row is cut back to the lower bucket when that avoids at least that many padded tokens, net of the bucket the remainder pads to. Only a step of chunkable rows alone cuts. Only configs/qwen3_5_9b.yaml sets it, to 512, a hand-set number.
- Keep test on the three concepts (67b69d6 against 440391c; 9B Random 1024/256, 3 alternating; hypothesis "PR check: qwen35 bucket cut"):
  - c=1 TTFT p50: 59.9, 58.0, 56.8 without; 42.2, 42.9, 42.7 with. All three beat. Tok/s 196.9, 211.3, 197.4 without; 200.1, 199.7, 200.0 with.
  - c=16 TTFT p50: 198.7, 198.0, 196.9 without; 200.1, 197.3, 201.2 with (overlap). Tok/s 2170, 2165, 2167 without; 2172, 2176, 2172 with.
  - c=64 tok/s: 4537, 4524, 4532 without; 4534, 4545, 4541 with (overlap). TTFT p50 218.8, 218.2, 219.5 without; 218.9, 220.6, 220.8 with (overlap).
  - The 440391c arm predates the review's refactor. Section 5's 9B Random c=1 at head reads 47.5 [43.4-48.9] ms, against main's 63.0, so the cut still fires. That is up to 6 ms above the arm tested, in a sweep where the cell runs after other workloads on the server.
- Ladder: on the ladder cells it is off (0.8B and 2B don't set it): c3 to head 0.8B ShareGPT c=64 14882 to 14957, Random c=64 15673 to 16143, 2B MMMU c=16 4275 to 4376; rounds overlap.
- Cost: a prompt cut this way keeps its prefix-cache GDN snapshot at an earlier page boundary, expected to make a resend re-run more tokens once. Not measured.
- Tests: test/modular/test_chunked_prefill.py: TestCutToBucket, test_captured_token_counts_read_the_walks_buckets, test_a_combined_walk_answers_a_bucket_cut_from_the_walk_that_asks. test/modular/test_micro_scheduler.py: test_the_builder_cuts_at_a_bucket, test_a_speculation_chain_beside_the_row_stops_the_bucket_cut. test/modular/test_qwen3_5_bucket_knobs.py: test_the_bucket_cut_is_off_unless_a_config_sets_it, test_a_bad_bucket_cut_is_refused. Fail on the parent (collection error in the first two files, 6 failures in the third).

## Ladder
Hypothesis "PR check: qwen35 ladder". 3 rounds rotated, fresh server each, the cell alone on it. Tok/s median [r1, r2, r3]. Round 1 of 0.8B fell in the fast host period above, so read the spread: Random c=64 readings at 17-18.6k are the round 1 of every arm, plus c2's round 2.

|cell|main|c1 c2ba5e6|c2 7a4394d|c3 c21ca1b|cut 362e11d|
|---|---|---|---|---|---|
|Qwen3.5-0.8B Random 1024/256 c=64|10542 [17298, 10098, 10542]; TTFT p50 87.1|15909 [17503, 15909, 15383]; TTFT p50 106.1|18507 [18507, 18584, 15607]; TTFT p50 70.2|15673 [17960, 15673, 15608]; TTFT p50 88.2|16143 [18210, 16143, 15438]; TTFT p50 99.4|
|Qwen3.5-0.8B ShareGPT c=64|7682 [14642, 7682, 7552]; TTFT p50 40.0|14314 [14287, 15744, 14314]; TTFT p50 27.4|15150 [15898, 15150, 14318]; TTFT p50 21.9|14882 [14858, 14882, 15150]; TTFT p50 28.3|14957 [14455, 15235, 14957]; TTFT p50 29.2|
|Qwen3.5-2B Image (MMMU) c=16|2397 [3038, 2397, 2393]; TTFT p50 135.6|3094 [3094, 3118, 3092]; TTFT p50 100.0|3216 [3211, 3216, 3240]; TTFT p50 94.0|4275 [4278, 4275, 4258]; TTFT p50 103.3|4376 [4376, 4388, 4326]; TTFT p50 96.8|

## 5. Main against PR
Hypothesis "PR check: qwen35". `M* main 305e80e` (baseline) and `M* pr-qwen35 362e11d` (candidate), 3 alternating rounds, fresh server per round, 0 failed requests across all 243 rows of batch h200_1106. --rust-frontend, MSTAR_SHM_ARENA=1, each tree's config, step_thread_core at its default, -O1, CPU frequency knobs off (main has neither knob), no uclampset. Fresh inductor cache per boot. On 2B and 9B the image cells run first on the server. Median [min-max] of tok/s; TTFT p50 in ms. Ratio is PR over main on tok/s. Main was in its slow mode for all of these (concept 1).

|cell|M* main 305e80e tok/s; TTFT p50|M* pr-qwen35 362e11d tok/s; TTFT p50|ratio last/first|
|---|---|---|---|
|Qwen3.5-0.8B Random 1024/256 c=1|423 [373-528] n=3; 29.0 [28.3-29.4]|864 [863-870] n=3; 24.0 [21.5-24.0]|2.04|
|Qwen3.5-0.8B Random 1024/256 c=16|5015 [4606-5198] n=3; 67.2 [57.3-70.6]|8215 [7373-8264] n=3; 64.1 [63.7-64.3]|1.64|
|Qwen3.5-0.8B Random 1024/256 c=64|9958 [9878-10683] n=3; 92.5 [85.7-102.2]|15744 [15610-16066] n=3; 112.6 [96.7-117.3]|1.58|
|Qwen3.5-0.8B Random 1024/256 c=128|11694 [9354-11768] n=3; 123.1 [117.3-169.1]|18725 [18626-19740] n=3; 139.3 [128.5-172.1]|1.60|
|Qwen3.5-0.8B ShareGPT c=1|364 [363-425] n=3; 20.3 [19.3-20.7]|783 [774-884] n=3; 13.2 [13.1-16.6]|2.15|
|Qwen3.5-0.8B ShareGPT c=16|4268 [3994-4525] n=3; 24.9 [23.3-26.9]|8680 [8396-8690] n=3; 15.2 [13.9-16.8]|2.03|
|Qwen3.5-0.8B ShareGPT c=64|7540 [6923-7653] n=3; 41.3 [41.2-42.4]|16859 [15882-16861] n=3; 23.8 [21.4-26.8]|2.24|
|Qwen3.5-0.8B ShareGPT c=128|8886 [8654-9902] n=3; 62.5 [58.1-64.3]|21219 [20964-21818] n=3; 38.3 [33.2-38.6]|2.39|
|Qwen3.5-2B Image (MMMU) c=4|837 [686-851] n=3; 89.3 [79.1-90.6]|1397 [1356-1398] n=3; 64.8 [56.0-67.3]|1.67|
|Qwen3.5-2B Image (MMMU) c=16|3260 [2723-3378] n=3; 134.4 [128.9-137.4]|4393 [4282-4429] n=3; 88.5 [75.8-97.5]|1.35|
|Qwen3.5-2B Random 1024/256 c=1|440 [432-457] n=3; 32.9 [30.4-33.3]|598 [590-599] n=3; 26.7 [26.1-27.5]|1.36|
|Qwen3.5-2B Random 1024/256 c=16|4987 [4983-5178] n=3; 80.0 [79.3-80.7]|6160 [6042-6177] n=3; 82.8 [67.9-83.3]|1.24|
|Qwen3.5-2B Random 1024/256 c=64|10433 [8464-10611] n=3; 90.2 [85.5-97.1]|13848 [13825-13852] n=3; 109.5 [85.8-111.0]|1.33|
|Qwen3.5-2B Random 1024/256 c=128|12209 [11974-12429] n=3; 122.0 [119.8-123.3]|16522 [16499-16627] n=3; 160.7 [114.5-163.8]|1.35|
|Qwen3.5-2B ShareGPT c=1|435 [424-440] n=3; 20.2 [20.1-20.3]|626 [619-630] n=3; 14.8 [14.7-15.0]|1.44|
|Qwen3.5-2B ShareGPT c=16|4398 [3654-4957] n=3; 24.1 [21.6-27.9]|6396 [6349-6509] n=3; 18.0 [14.9-18.2]|1.45|
|Qwen3.5-2B ShareGPT c=64|7568 [7560-8094] n=3; 40.3 [38.4-40.6]|13992 [13799-14034] n=3; 27.5 [23.5-28.1]|1.85|
|Qwen3.5-2B ShareGPT c=128|9807 [9713-9850] n=3; 58.2 [57.8-58.2]|19044 [18838-19445] n=3; 38.1 [33.9-38.2]|1.94|
|Qwen3.5-9B Image (MMMU) c=4|464 [419-478] n=3; 112.0 [96.6-117.3]|569 [567-575] n=3; 95.2 [83.9-107.3]|1.23|
|Qwen3.5-9B Image (MMMU) c=16|1814 [1713-1862] n=3; 207.6 [153.9-235.4]|1837 [1827-1843] n=3; 192.5 [179.7-199.3]|1.01|
|Qwen3.5-9B Random 1024/256 c=1|197 [196-197] n=3; 63.0 [62.3-64.5]|213 [212-213] n=3; 47.5 [43.4-48.9]|1.08|
|Qwen3.5-9B Random 1024/256 c=64|4444 [4442-4450] n=3; 277.7 [232.7-280.9]|4526 [4519-4535] n=3; 271.7 [217.6-273.1]|1.02|
|Qwen3.5-9B ShareGPT c=1|202 [202-203] n=3; 24.5 [23.7-25.2]|217 [217-218] n=3; 25.7 [22.1-25.9]|1.07|
|Qwen3.5-9B ShareGPT c=64|5215 [5108-5273] n=3; 50.0 [44.0-52.3]|5422 [5405-5428] n=3; 48.2 [45.1-48.7]|1.04|

## 6. Correctness
Teacher-forced top-1 against transformers sdpa, 64 greedy tokens per prompt, on 64 ShareGPT prompts and 16 MMMU images, from each of 3 sweep boots per tree. Bar: transformers eager greedy, scored the same way. Harness adapted from test/qwen3_vl/served_parity.py; not on the branch.

|set|bar (HF eager)|PR r1, r2, r3|main r1, r2, r3|PR result|
|---|---|---|---|---|
|0.8B text /4096|4081|4086, 4083, 4082|4082, 4086, 4082|pass 3 of 3|
|2B text /4096|4075|4085, 4078, 4079|4087, 4074, 4078|pass 3 of 3|
|2B image /1024|1010|1013, 1012, 1014|1016, 1013, 1014|pass 3 of 3|
|9B text /4096|4080|4081, 4079, 4079|4079, 4080, 4076|**pass 1 of 3**|
|9B image /1024|1019|1016, 1016, 1016|1016, 1016, 1017|**pass 0 of 3**|

- **9B misses the gate on a strict reading, and main misses it too.** The PR is at or above main's range on text and 1 position under main's best on images. Every miss on 9B, the bar's and both trees', has a log-prob gap of at most 0.25 nats under sdpa at bf16, median 0 to 0.125 (results gate/margin_9.json). The first-token miss count is 1 on all of them. Four of the 8 image misses are ones HF eager also has. So near-ties, by this measure; I didn't establish why eager scores higher on these images.
- Greedy agreement (the earlier brief's gate, with main 305e80e), identical of N and mean prefix, across boots:
  - 0.8B text: PR-main 52-57, 0.937-0.976; main-main 51-57, 0.939-0.967. Within.
  - 2B text: PR-main 44-55, 0.864-0.926; main-main 47-52, 0.888-0.915. Overlaps, lower end below.
  - **2B image: PR-main 9-13 of 16, 0.765-0.919; main-main 13-15, 0.922-0.971. Below.** Attributed to concept 2 (section 4), since c1 is 11-14 and c2 is 9-11. PR-PR in the sweep reads 10 of 16, 0.805-0.883; c2 and c3 against themselves read 14 of 16, 0.945.
  - **9B text: PR-main 47-53, 0.864-0.912; main-main 49-57, 0.917-0.959. Below.** Not attributed: no intermediate arms on 9B. PR-PR 50-51, 0.886-0.931.
  - 9B image: PR-main 11-16, 0.879-1.0; main-main 13-14, 0.932-0.964. Overlaps.
  - On a strict reading that earlier gate fails on 2B image and 9B text. Teacher-forced agreement with HF, this PR's own gate, is the table above.

## 7. Tests
- Base: 50 failed, 7 errors, 10898 passed, 1290 skipped, serial and -n 8 --dist loadfile with identical ids. Parallel used after that. CUDA hidden, OMP_NUM_THREADS=2, --continue-on-collection-errors.
- Head 6bfc962: 50 failed, 7 errors, 10929 passed, 1305 skipped. Same failing ids as the base. ruff clean.
- Each commit: ruff plus targeted files, no failure outside the base set.
- GPU, H200, at head, torch 2.12.1+cu132, flashinfer 0.7.2: test_gdn_prefill_stage, test_gdn_mixed_split, test_gdn_cuda_graph, test_gdn_pinned_stage, test_gdn_no_sink_slot, test_gdn_prefill_padding_rows, test_gdn_prefix_snapshots, test_gdn_gate_pad_alignment, test_gdn_tensor_parallel, test_qwen3_5_mixed, test_qwen3_5_bucket_knobs, test_chunked_prefill, test_step_threads, test_micro_scheduler: 735 passed. test_glm5next_*, test_nemotron_duplex_*: 366 passed, 1 failed (the failure is test_duplex_load_audio_refuses_a_file_past_the_session, in the base set). Real-weights integration tests were not run; the serving gate in section 6 stands in for them.
- Pre-existing failures: torchaudio-dependent audio tests and modules (7 collection errors: test_chatterbox_model, test_chatterbox_watermark_import, test_cosyvoice3_flow_capture, test_cosyvoice3_model, test_omnilingual_asr_model, test_piecewise_config_signature, test_resample_rate_bound), plus failures in test_audio_decode_limit, test_audio_features, test_audio_watermark, test_base_load_audio, test_client_error_text, test_fish_s2_pro_model, test_lora_targets, test_model_kv_layouts, test_nemotron_duplex_model, test_qwen3_tts_model, test_qwen3_tts_reference_audio, test_request_media_totals, test_startup_graph_validation (11 configs), test_whisper_model. Full ids: results/rp3-pr/modular_base_par.set on the pod; the list is below.
- Environment: `[all]` resolves torchaudio 2.11.1+cu132, which cannot load against torch 2.12.1, so it was uninstalled from both venvs. That is why those modules can't collect, and why --continue-on-collection-errors is used. pytest-xdist added to both venvs.

## 8. Review
A subagent saw only the diff and base, on 362e11d's predecessor (head 3910a49). Run before the numbers, so its fixes were measured. Its answers, and what happened:

|piece|its answer|outcome|
|---|---|---|
|Engine.bucket_cut keyed on the combined walk|wrong ladder: reads `mixed` while a prompt-only step replays `prefill_text`; latent, the ladders match on 9B|fixed: resolved against the walk inside the combined walk that asks|
|Engine.bucket_cut, _bucket_cuts|belongs in ChunkingManagement|moved; Engine keeps a one-line forwarder like its siblings|
|chain_tokens field and parameter|exist only for an early return|removed; the scheduler passes None. New test on the real _build_step|
|Policy on LLM_MIXED carrying the cut|leak of the scheduler's key into the model|removed; only prefill_text sets it|
|`_dense` test helper|aborts __init__ by exception|replaced with __new__ plus attributes|
|docs "one pair of the earlier form"|describes a discarded design|removed|
|env docs: a server that finds its cores locked pins nothing|contradiction: claim_cores tries the next set|fixed in .rst and .mdx|
|cpus._siblings docstring numbers|conflict with the table, too long|cut, points at the table|
|rule stated in many places|keep the docstring and one doc pair|cut from the builder field, policy comment, yaml, deployment_plan. The scheduler keeps one line for the chain case|
|auto_pins default idle_hold="off"|could drop the hold silently|parameter made required|
|5.6 s in a code comment|more than a reader needs|moved to the commit body|
|test asserting 8192|asserts a constant|rewritten against the motivating 5360-token image through the capture configs. Still close to a constant; kept so the concept has a test|
|Qwen3.5 class attribute bucket_cut_min_saved|only for tests using __new__|**tried removing, restored.** 8 test_startup_graph_validation configs build the model through __new__ and failed|
|auto as siblings as a default|can't justify without more data: Python frontend, klein on the 6740E, an EPYC with the same C2 that lost|**not changed:** the brief names this concept. Costs stated above; open below|
|auto_pins shape, "auto places by rank" at three sites|resolve auto once|**not changed:** startup-only, and resolving once means reshaping deployment_plan's callers. Open below|
|gdn_stage_applies|justified on correctness; "measured" needs a Hopper number|numbers in section 4|
|8192 bucket|partly justified; no memory or startup number; tower still stops at 16384 patches|memory numbers in section 4; startup not timed|
|cut_to_bucket, captured_token_counts, BucketCut, ChunkingPolicy field, yaml, declared paths|justified|kept|
|general components calling undeclared methods|none found|-|

## 9. Open
- **Concept 1 as a default, on the Python frontend:** c=64 TTFT p50 about 12x worse, and that is the default frontend. Options a reviewer may ask about: auto declines without --rust-frontend, or the default frontend changes. Neither is built.
- **How big concept 1's gain is on a typical host.** It holds against main's slow mode, is near nothing in its fast mode, and the cause of the two modes is a hypothesis. One host class (EPYC 9554). docs/serving.rst keeps two earlier H200 rows where siblings lost, not repeated here. auto can't tell those hosts from this one.
- **9B gate** (section 6): misses on main too, near-ties by the margin check, cause of eager's lead on images not established. The strict earlier greedy gate fails on 2B image and 9B text.
- 9B greedy distance from main not attributed to a concept.
- TP and multi-GPU: not run for any concept (one GPU here). Plan tests cover auto on those configs. Qwen3.5-27B, Qwen3.8 and Blackwell not run. Concept 2 and 3 inherit there; concept 4 is off there.
- Concept 3: compile count at 83 of 84, one spare. Startup time not separated. A step past 8192 tokens is still eager.
- Concept 4: 512 is hand-set; resend cost after a cut not measured. tp2 and dp2 configs leave it off, unmeasured. Only one A/B of the cut itself, on the arm before the refactor.
- auto_pins' shape (section 8).
- Numbers are at -O1, not the -O2 the benchmarking skill asks for against another engine.
- Qwen3-Omni weights were removed from this pod's HF cache to fit 9B, and can be downloaded again.

## References
Hypothesis "PR check: qwen35 references". One pass, unpinned, fresh server each, same client. vLLM 0.33.0 and SGLang 0.5.23, bf16, max length 32768, prefix cache default, no speculative decoding. For scale, the PR's sweep median beside them, which isn't a same-time A/B and is at -O1.

|cell|SGLang tok/s; TTFT p50|vLLM tok/s; TTFT p50|M* PR|
|---|---|---|---|
|0.8B ShareGPT c=64|10720; 55.0|8667; 79.6|16859; 23.8|
|0.8B Random c=64|16921; 214.4|11950; 258.3|15744; 112.6|
|2B ShareGPT c=64|10105; 48.9|9093; 78.2|13992; 27.5|
|2B Random c=64|12773; 344.3|11207; 310.2|13848; 109.5|

## Pre-existing failure ids
```
ERROR test/modular/test_chatterbox_model.py
ERROR test/modular/test_chatterbox_watermark_import.py
ERROR test/modular/test_cosyvoice3_flow_capture.py
ERROR test/modular/test_cosyvoice3_model.py
ERROR test/modular/test_omnilingual_asr_model.py
ERROR test/modular/test_piecewise_config_signature.py
ERROR test/modular/test_resample_rate_bound.py
FAILED test/modular/test_audio_decode_limit.py::test_a_soundfile_rate_past_the_range_is_refused_before_any_read
FAILED test/modular/test_audio_decode_limit.py::test_omnilingual_asr_decode_keeps_a_clip_under_its_bound
FAILED test/modular/test_audio_decode_limit.py::test_omnilingual_asr_decode_stops_at_its_clip_bound
FAILED test/modular/test_audio_decode_limit.py::test_omnilingual_asr_refuses_a_rate_past_the_range_before_reading
FAILED test/modular/test_audio_decode_limit.py::test_qwen3_tts_reference_decode_stops_at_its_bound
FAILED test/modular/test_audio_decode_limit.py::test_soundfile_loaders_are_capped
FAILED test/modular/test_audio_features.py::test_load_audio_file_decodes_and_resamples
FAILED test/modular/test_audio_watermark.py::test_detect_resamples_to_the_rate_the_mark_was_added_at[16000-24000]
FAILED test/modular/test_audio_watermark.py::test_detect_resamples_to_the_rate_the_mark_was_added_at[48000-24000]
FAILED test/modular/test_base_load_audio.py::test_a_crafted_header_rate_is_refused_before_resampling[100003]
FAILED test/modular/test_base_load_audio.py::test_a_crafted_header_rate_is_refused_before_resampling[48001]
FAILED test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-flac-PCM_16]
FAILED test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-mp3-MPEG_LAYER_III]
FAILED test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-ogg-VORBIS]
FAILED test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-wav-PCM_16]
FAILED test/modular/test_client_error_text.py::test_a_transcription_whose_decode_fails_in_the_frontend_answers_with_an_error_id
FAILED test/modular/test_fish_s2_pro_model.py::test_load_audio_decodes_a_clip_without_a_sample_count[1]
FAILED test/modular/test_fish_s2_pro_model.py::test_load_audio_decodes_a_clip_without_a_sample_count[2]
FAILED test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_clip_without_a_sample_count_at_its_decoded_length[0.3-0\\.\\d\\d]
FAILED test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_clip_without_a_sample_count_at_its_decoded_length[45-30\\.\\d\\d]
FAILED test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[16000-False]
FAILED test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[44100-False]
FAILED test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[44101-True]
FAILED test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[48000-False]
FAILED test/modular/test_lora_targets.py::test_children_run_other_than_through_their_forward_are_refused[chatterbox_positions]
FAILED test/modular/test_model_kv_layouts.py::test_a_reference_clip_is_laid_out_at_the_tokens_the_tokenizer_makes_of_it[chatterbox]
FAILED test/modular/test_model_kv_layouts.py::test_a_reference_clip_is_laid_out_at_the_tokens_the_tokenizer_makes_of_it[turbo]
FAILED test/modular/test_model_kv_layouts.py::test_chatterbox_lays_out_every_label_it_declares[0.0-False]
FAILED test/modular/test_model_kv_layouts.py::test_chatterbox_lays_out_every_label_it_declares[0.5-False]
FAILED test/modular/test_model_kv_layouts.py::test_chatterbox_lays_out_every_label_it_declares[0.5-True]
FAILED test/modular/test_nemotron_duplex_model.py::test_duplex_load_audio_refuses_a_file_past_the_session
FAILED test/modular/test_qwen3_tts_model.py::test_qwen3_tts_load_audio_rejects_a_long_clip_before_decoding
FAILED test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_decodes_a_clip_without_a_sample_count[1]
FAILED test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_decodes_a_clip_without_a_sample_count[2]
FAILED test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_refuses_a_long_clip_without_a_sample_count_after_about_30_s
FAILED test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_refuses_a_short_clip_without_a_sample_count_at_its_length
FAILED test/modular/test_request_media_totals.py::test_a_second_clip_is_refused_before_either_is_decoded[mstar.model.chatterbox.chatterbox_model-ChatterboxModel]
FAILED test/modular/test_request_media_totals.py::test_one_ordinary_clip_still_loads[mstar.model.chatterbox.chatterbox_model-ChatterboxModel]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox.yaml]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_multilingual.yaml]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_multilingual_split.yaml]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_split.yaml]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_turbo.yaml]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_turbo_split.yaml]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[cosyvoice3.yaml]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_1b.yaml]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_300m.yaml]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_3b.yaml]
FAILED test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_7b.yaml]
FAILED test/modular/test_whisper_model.py::test_load_audio_file_decodes_and_resamples
```
