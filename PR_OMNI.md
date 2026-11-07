IN PROGRESS: amendment of Nov 7 — splitting into perf/rp4-pr-gather + perf/rp4-pr-omni, review fixes, re-measuring

# PR facts: Qwen3-Omni

- Base: 305e80ef46248ade6e6670aa20c9bb67f4ac2e26 (upstream main)
- Branch: perf/rp4-pr-omni
- Head: 82ad7e695d2ccaed73f279b82aa7faea61cfd083

## Commits (in order)

| # | Commit | Subject | Source |
|---|---|---|---|
| 1 | c5685b951 | model: set qwen3-omni thinker sentinel slots without host syncs | 68785670 (ported) |
| 2 | fc002c60d | test: qwen3-omni thinker prefill prep writes no scalars by index | 0cf8457d + a5921bde (squashed, ported) |
| 3 | a91d6b869 | model: build a qwen3-omni audio walk row in a few launches | 387d2c4f + a28a4e57 (ported) |
| 4 | 134d42d0f | test: qwen3-omni audio walk row is its spans end to end, embedded in one call | new for main (79fea87f's assertions targeted run-base's row) |
| 5 | e0d9525b0 | model: no qwen3-omni talker masks for a request with no audio output | rp3 78d41d65 (cherry-pick -x, conflicts resolved) |
| 6 | e922f3434 | test: qwen3-omni builds no talker masks without audio output | rp3 afbf77c9 (cherry-pick -x) |
| 7 | a0c06b469 | worker: let a node's ready rows gather before they take a step | rp3 f35f5c34 (cherry-pick -x) + the fake-engine lines of 9a228e98 + Mintlify docs |
| 8 | 82ad7e695 | test: gather policy waits only while busy, and a node without one steps as before | rp3 9a228e98 (test_gather_policy.py) + a new test |

### Dropped
- Concept 1 (60bb897c, a3f3b6a4) and concept 3 (e3c6427c, a3fc3e02): main's #1110 (a0470c32b) already prefills each attachment together with all the text before it and, for the last one, the text after it (`_thinker_steps`, mstar/model/qwen3_omni/qwen3_omni_model.py:180 at base; `_spliced_row`, submodules.py:651). That is a superset of both concepts (no page-size limit on the leading span), and its KV layout keys the steps it runs (#1165). The docs commit 66d60d5a documents concepts 1/3 and is dropped with them.
- a3f3b6a4's remaining point (penalty ids of an audio row survive a whole-walk prefix-cache hit): main carries those ids as `text_token_ids`, not `input_ids` (submodules.py:679 at base: "Ids cannot ride `input_ids`"). Left as main has it; see Open.
- 5609484e + revert c1e93f7b; f2c07a72, 898f7450, revert 4fdfd0e3 (replaced by rp3's concept 5); 11870717, 7b360407 (CPU frequency PR, rp2).

### Ported (what changed)
- Concept 2: main still wrote `mm_mask[0] = False` / `[-1]` on audio and vision rows (submodules.py:552-553, 585-586 at base), under a comment that said the write is not a sync. Now one helper `_media_mask` holds the fill_ and the reason.
- Concept 4: main's audio row is `_spliced_row` (text before + clip + text after, an embed call per side, a position helper per span, masks concatenated). `_audio_row` builds it from one id buffer, one embedding call, one position range and masks filled in place. Bit-identical to main's row on four layouts (checked with /workspace/prwork/scripts/c4_compare.py; the equivalence test passes on main too). Dispatched non-view ops for a text-clip-text row: 66 -> 50 (CPU, TorchDispatchMode count). Main's `text_token_ids` is kept for the penalty. Pads use the start-sentinel id (overwritten by the encoder rows) so no new config field is read. `get_rope_index_audio` and `_wrap_audio_input` deleted (no other user). The "only a request that speaks builds masks" part of 387d2c4f moved to concept 5, where rp3's commit does it.
- Concept 5: rp3's change on main's `_spliced_row`/text rows, plus the same gate in `_audio_row`. rp3's test needed the Rope counter seeded (`_counters[RID]`), as the file's other tests do.
- Concept 6: test_loop_turns.py / test_tp_follow_targeted_gate.py conflicts were adjacent additions (main's `batch_cap_varies` fake and the new `get_gather_policy` fake): kept both. The fake updates moved into the code commit so it leaves the tree green. Docs added to docs/mintlify/adding_models.mdx too (main keeps both doc trees live).

## Concepts

### 2. The Thinker's sentinel slots are set without host syncs (c5685b951, test fc002c60d)
- Changes: mstar/model/qwen3_omni/submodules.py (`ThinkerSubmodule._media_mask`, used by the image/video row; the audio row until concept 4 replaces it).
- Mechanism: `mask[0] = False` on a CUDA tensor is an index write of a Python scalar, which ATen turns into a copy from a pageable host tensor, so it waits for the queued GPU work (the step in flight). The slots are now set with `fill_` on a view, which is a kernel launch. Main's comment there said the opposite.
- Own effect: not isolated by the ladder (the first ladder arm, after concept 4, holds 2 and 4). On run-base in the loop (round 3) it gave A2T +3.9% c=4, +5.2% c=16, +4.0% c=32 (all rounds), when the audio row still used these writes. On this branch the audio row stops using `_media_mask` in concept 4, so what remains is the image/video row, which no measured workload exercises.
- Cost: none measured.
- Tests: test/modular/test_step_prep_host_syncs.py::test_qwen3_omni_thinker_prefill_prep_writes_no_scalar (CPU, TorchFunctionMode records scalar index writes, audio and image rows), ::test_qwen3_omni_thinker_audio_prep_takes_no_host_sync (CUDA sync-debug "error" mode).

### 4. An audio walk row is built in a few launches (a91d6b869, test 134d42d0f)
- Changes: mstar/model/qwen3_omni/submodules.py (`_audio_row` replaces the audio branch of `prepare_inputs`, `_wrap_audio_input` and its cached sentinel embeddings), mstar/model/qwen3_omni/components/rope.py (`get_rope_index_audio` deleted, no other user), a comment in qwen3_omni_model.py naming the new function.
- Mechanism: main built the row span by span (an embedding call per text side, a position helper per span, masks concatenated from per-span pieces) on the GPU thread. The row's ids now fill one buffer (text before, start sentinel, placeholder rows, end sentinel, text after) embedded in one call, then the encoder's rows are copied over the placeholders; positions are one `arange`, since text and audio both advance all three MRoPE components by one per row (as main's helpers already did); the Talker masks are filled in place. Output is bit-identical to main's row.
- Own effect (ladder, A2T c=32, includes concept 2): 1265.0 [1260.2-1302.7] -> 1387.1 [1377.6-1407.2] tok/s (+9.7%, every round above main's); TTFT p50 115.3 -> 106.3 ms (-7.8%). T2S c=16 (text-only, path unchanged): 99.7 -> 98.8 audio-s/s, overlapping. Dispatched non-view ops for a text-clip-text row 66 -> 50 (CPU count). The throughput gain is larger than the op count suggests; that it comes from GPU-thread host time per audio row is a hypothesis (no profile on main).
- Cost: none measured. Embeds the placeholder rows before overwriting them (one embedding row per audio frame, wasted).
- Tests: test/modular/test_qwen3_omni_audio_row.py::test_an_audio_row_is_its_spans_end_to_end (4 layouts: embeds, positions, both mask rows, advance, text ids against the spans laid end to end; also passes on main), ::test_an_audio_row_embeds_its_ids_in_one_call (fails on main: 2 calls).

### 5. No Talker masks for a request with no audio output (e0d9525b0, test e922f3434; rp3's 78d41d65 / afbf77c9)
- Changes: mstar/model/qwen3_omni/submodules.py: text rows, `_audio_row`, `_media_mask` / `_spliced_row` (image/video) build `masks_for_talker` only when `step_metadata["audio_output"]`; `split_inputs` cuts the key only when present.
- Mechanism: only the Talker reads these masks, and a request without audio output never reaches the Talker. Every consumer already reads the key with `.get` (preprocess, unpack_packed_outputs; postprocess returns before reading them for such requests). The per-row `talker_text_mask` passes, stack and fills are skipped.
- Own effect (ladder): A2T c=32 1387.1 [1377.6-1407.2] -> 1403.3 [1390.4-1405.1] tok/s (+1.2%, overlapping); TTFT 106.3 -> 106.6 ms. T2T c=1 (text rows and the scheduler's policy lookup are the only PR code on that path) is +1.5% head over main with every round above (310.2 -> 314.9 tok/s); no ladder arm isolates it. rp3 measured +3-6% A2T on its base, where the audio row still built its masks span by span.
- Cost: none measured.
- Tests: test/modular/test_qwen3_omni_prefix_cache.py::test_a_request_with_no_audio_output_builds_no_talker_masks[prefill_text|prefill_audio|prefill_vision] (row and a cut chunk carry no masks).

### 6. GatherPolicy: a node's ready rows can gather before they take a step (a0c06b469, test 82ad7e695; rp3's f35f5c34 / 9a228e98)
- Changes: mstar/model/submodule_base.py (`GatherPolicy`, `NodeSubmodule.gather_policy` returning None), mstar/engine/engine.py (`Engine.get_gather_policy`), mstar/worker/micro_scheduler.py (`_gathering`, the ready filter in `get_next_batch`'s scan, the peek in `has_ready_excluding`, `note_ran` ends a wait), mstar/model/qwen3_omni/submodules.py (`AudioEncoderSubmodule.gather_policy`: 8 rows, 30 ms, 16 requests), docs/adding_models.rst and docs/mintlify/adding_models.mdx.
- Mechanism: the audio encoder is cheap and ran as soon as one row was ready, so clips reached the Thinker one at a time, each costing the Thinker a mixed prefill step. With a policy, while the worker holds at least `busy_requests` requests, the walk's ready rows wait until `rows` are ready or `wait_s` has passed since they first could have run, so the Thinker prefills several clips per step. The policy is asked once per (node, walk); a node without one is filtered by nothing. The decision is made only for nodes this rank leads; TP followers replay rank 0's steps.
- Own effect (ladder, head against after-5): A2T c=32 1403.3 [1390.4-1405.1] -> 1525.7 [1525.0-1531.4] tok/s (+8.7%, every round above). A2T c=8 is below busy_requests, so the policy is idle there.
- Cost: A2T c=32 TTFT p50 106.6 -> 142.9 ms (+34% against after-5; +24% against main's 115.3). It holds rows back by design, up to 30 ms. Not measured: audio output with audio input (requests in the Talker count towards busy, so a lone clip can wait the full 30 ms), several clips in one prompt (waits once per clip), TP. T2S: after-5 against head at T2S c=1 and c=4, 3 rounds each: level (see "Main against PR").
- Tests: test/modular/test_gather_policy.py (rows wait until the policy lapses; enough rows run at once; a quiet worker does not wait; a node without a policy runs at once and is asked once; the rows after a step wait from their own start; a peek does not start the wait; a wait the load no longer calls for ends), test/modular/test_loop_turns.py::test_a_node_without_a_gather_policy_steps_as_if_nothing_gathered[TURNS|ARRIVAL] (24 requests, more than any busy count: the real scheduler is asked, never gathers, and takes the same steps as with gathering patched out; the other scheduler test files run unchanged with the new fake).

## Main against PR

Arms: `M* main 305e80e` (baseline; /workspace/wt-main305, venv-main305) and `M* pr-omni 82ad7e695` (candidate; venv-pr, PYTHONPATH at a detached worktree of the head). Ladder arms at the headline cells: after concept 4 (134d42d0f), after concept 5 (e922f3434); after concept 6 is the head.
Settings, same on every arm: configs/qwen3omni_colocated.yaml as in each tree (identical), one H200, `--rust-frontend`, MSTAR_SHM_ARENA=1, MSTAR_STEP_THREAD_CORE=siblings (the env form of `step_thread_core: siblings`), no full_cpu_frequency / step_thread_util_min knobs (absent on main) and no uclampset. Fresh server per round, one untimed warm pass per server (T2T, A2T, T2S at c=8 on rows disjoint from the measured ones), each A2T cell on LibriSpeech validation-clean rows no earlier cell on that server sent. Requests per cell: A2T 16 x c (min 64), T2T and T2S 5 x c (min 32). Harness /workspace/prwork/round_pr.sh + ab_pr.sh, client /workspace/results/tools/omni_bench.py (benchmark.runner), one per-arm Inductor cache kept across that arm's rounds. Order: round 1 main, c4, c5, head; round 2 head, c5, c4, main; round 3 main, c4, c5, head.
Round 1 of main ran while CPU test suites were still being stopped (its A2T c=1 and c=8 cells overlapped them: 38.8 and 237 tok/s); it is kept as runs/pr_ab/r1x_main_contended, not counted or reported, and main's round 1 was rerun after round 3.

Median [min-max] of 3 rounds; throughput tok/s (A2T, T2T) or audio-s/s (T2S); latency p50 is TTFT (A2T, T2T) or TTFA (T2S); ratio head/main of medians, "all above/below" when the ranges do not overlap. No request failed in any counted cell.

| cell | metric | main | c4 | c5 | head | head/main |
|---|---|---|---|---|---|---|
| a2t_c1 | tput | 197.3 [195.8-198.3] | n/a | n/a | 199.3 [198.2-200.0] | 1.010 |
| a2t_c1 | p50 ms | 39.5 [39.3-40.7] | n/a | n/a | 39.6 [38.9-40.5] | 1.001 |
| a2t_c8 | tput | 564.6 [562.4-568.8] | n/a | n/a | 593.6 [586.4-598.1] | 1.051 (all above) |
| a2t_c8 | p50 ms | 69.8 [67.5-70.3] | n/a | n/a | 67.7 [66.0-67.8] | 0.969 |
| a2t_c32 | tput | 1265.0 [1260.2-1302.7] | 1387.1 [1377.6-1407.2] | 1403.3 [1390.4-1405.1] | 1525.7 [1525.0-1531.4] | 1.206 (all above) |
| a2t_c32 | p50 ms | 115.3 [114.9-117.1] | 106.3 [104.7-107.7] | 106.6 [106.1-109.1] | 142.9 [137.5-143.3] | 1.239 (all above) |
| t2t_c1 | tput | 310.2 [310.1-310.4] | n/a | n/a | 314.9 [314.7-315.1] | 1.015 (all above) |
| t2t_c1 | p50 ms | 26.4 [26.3-26.8] | n/a | n/a | 26.7 [26.2-26.9] | 1.011 |
| t2t_c32 | tput | 2288.0 [2286.7-2293.2] | n/a | n/a | 2294.6 [2289.5-2296.1] | 1.003 |
| t2t_c32 | p50 ms | 61.4 [60.9-64.8] | n/a | n/a | 61.4 [60.8-65.1] | 1.000 |
| t2s_c1 | tput | 20.1 [20.1-20.1] | n/a | n/a | 20.0 [20.0-20.0] | 0.995 (all below) |
| t2s_c1 | p50 ms | 116.2 [116.2-116.7] | n/a | n/a | 115.7 [115.5-116.4] | 0.995 |
| t2s_c4 | tput | 43.9 [43.6-44.1] | n/a | n/a | 43.3 [43.2-43.6] | 0.986 (all below) |
| t2s_c4 | p50 ms | 177.5 [175.7-181.1] | n/a | n/a | 178.1 [177.7-178.6] | 1.003 |
| t2s_c16 | tput | 99.7 [98.5-100.0] | 98.8 [98.8-99.9] | 99.6 [99.5-99.8] | 99.0 [98.9-100.8] | 0.992 |
| t2s_c16 | p50 ms | 261.5 [261.0-262.2] | 262.1 [255.8-262.5] | 257.7 [252.2-257.8] | 261.1 [256.4-271.6] | 0.998 |
| t2s_c32 | tput | 148.8 [148.4-148.9] | n/a | n/a | 149.9 [147.6-150.4] | 1.007 |
| t2s_c32 | p50 ms | 331.4 [318.6-332.3] | n/a | n/a | 333.9 [326.1-340.1] | 1.007 |

All 66 rows reported with mstar-report (batch h200_1106, status counted, hypothesis "PR check: Qwen3-Omni"; each row's config names every flag and knob).

T2S attribution check. Head is below main at T2S c=1 (-0.5%) and c=4 (-1.4%) with non-overlapping rounds, although text-only T2S requests run no changed model code; the only PR code on their path is concept 6's scheduler lookup. After concept 5 (e922f3434) against head, T2S c=1 and c=4 only, 3 alternating rounds, fresh server each, same settings (runs/pr_t2s):

| cell | metric | after 5 (e922f3434) | head |
|---|---|---|---|
| t2s_c1 | audio-s/s | 20.0 [20.0-20.0] | 20.0 [20.0-20.0] |
| t2s_c1 | TTFA p50 ms | 117.8 [115.8-118.7] | 116.7 [115.8-117.6] |
| t2s_c4 | audio-s/s | 43.8 [43.3-44.4] | 43.7 [43.7-43.8] |
| t2s_c4 | TTFA p50 ms | 178.5 [168.2-179.6] | 170.4 [166.5-179.2] |

Concept 6 does not move T2S, and after-5's T2S c=4 range overlaps main's. Concepts 2, 4 and 5 do not run for text-only speech requests. The head-vs-main gap of 0.6 audio-s/s at c=4 is attributed to server-to-server variation; that is a reading of these numbers, not a verified mechanism. These 12 rows are reported too (78 rows in all).

## Correctness (gate runs: one fresh server per arm, /workspace/prwork/runs/gate; same harness and settings as the A/B)

Thinker, 64 fixed LibriSpeech validation-clean clips (rows 0-63), greedy, exact token ids, HF transformers Qwen3-Omni-30B-A3B-Instruct bf16 teacher-forced on each run's tokens (/workspace/results/tools/hf_ref.py, reference /workspace/results/ref from the loop):

| | main 305e80e | PR head 82ad7e695 | HF sdpa | HF eager |
|---|---|---|---|---|
| corpus WER | 20.84% | 21.13% | 15.29% | 15.19% |
| WER excluding the answered clip | 0.88% | 0.88% | 1.08% | 1.08% |
| clips answered instead of transcribed | 1 (clip 4) | 1 (clip 4) | 1 | 1 |
| transcripts identical to HF sdpa | 61/64 | 60/64 | | |
| teacher-forced top-1 agreement (hf_ref) | 0.99660 | 0.99456 | | sdpa on eager 0.99804 |
| agreement, second HF pass (hf_diag) | 0.99184 (12 diffs) | 0.99252 (11 diffs) | sdpa on itself 0.99739 | |

- The literal bar (agreement >= HF self-agreement 0.99804) is missed by main as well as by the PR, and HF misses it against itself in a second pass (0.99739). Every disagreement in both arms has an HF top-1 margin of 0.0, 0.125 or 0.25 logits, i.e. a bf16 tie (hf_diag.log). Head's mismatches: clip 4 (the free-form answer, 7 tokens) and clip 52 (1 token, margin 0.125). The gate as applied in the loop (no disagreement above 0.25 logits; WER excluding answered clips no worse) passes.
- Corpus WER differs only through clip 4, which both arms answer at length instead of transcribing.

Talker: sampling is seeded per request but differs between server processes, so codec tokens are not comparable across arms. ASR round trip instead (benchmark.answer_audio_wer, Whisper large-v3-turbo, each spoken answer against its own text), T2S c=4, 32 answers per arm:
| | main | head |
|---|---|---|
| mean WER | 0.093 | 0.108 |
| median | 0.045 | 0.059 |
| p90 | 0.209 | 0.376 |
| answers above 0.5 | 1 | 2 (Fisher exact p = 1.0) |
Second sample, the A/B's round 1 T2S c=4 (32 other prompts, audio kept): main 0.055 mean (median 0.040, p90 0.146, 0 above 0.5), head 0.065 (median 0.033, p90 0.145, 0 above 0.5).
Pooled over both samples (64 answers per arm): mean of the two samples' means main 0.074, head 0.087; answers above 0.5: main 1, head 2. The loop measured per-arm means of 0.035-0.085 on identical code, so the difference is inside that spread. Text-only speech requests run no model code the PR changes (prefill_text rows still build their masks since they speak; no audio encoder), so the result is consistent with no change. Greedy codec-token identity was not run.

## Tests

Baseline, `pytest test/modular -q` serial at base in venv-main305 (CPU only, OMP_NUM_THREADS=2, PYTHONHASHSEED=0):
2 failed, 11460 passed, 1291 skipped, 38 subtests passed (58:39). Pre-existing failures:
- test/modular/test_client_error_text.py::test_a_transcription_whose_decode_fails_in_the_frontend_answers_with_an_error_id
- test/modular/test_tp_read_failure_acceptance.py::test_a_request_beside_the_failed_one_is_bit_identical_to_a_run_without_failure[text_only]

Parallel (`-n 8 --dist loadfile`) was not used: on this host, with the GPU server and other suites running, the base parallel run reached 55% in 1 h 40 min (serial finished in 59 min), and `test_eval_compare.py::test_a_job_whose_lease_is_not_renewed_stops_itself_and_its_servers` (a 1.5 s lease around a fake server's startup) failed under that load in two parallel targeted runs and passed alone. Every later run is serial.

Per commit (ruff + the 34 test/modular files that import qwen3_omni, plus mstar/model/qwen3_omni and the new audio-row test, serial):
| commit | ruff | targeted |
|---|---|---|
| c5685b951 | pass | 2052 passed, 40 skipped |
| fc002c60d | pass | 2053 passed, 41 skipped |
| a91d6b869 | pass | 2053 passed, 41 skipped |
| 134d42d0f | pass | 2058 passed, 41 skipped |
| e0d9525b0 | pass | 2058 passed, 41 skipped |
| e922f3434 | pass | 2061 passed, 41 skipped |
| a0c06b469 | pass | scheduler files (test_loop_turns, test_micro_scheduler, test_resident_delta, test_scheduler_room, test_tp_follow_targeted_gate): 252 passed |
| 82ad7e695 | pass | the same + test_gather_policy: 261 passed |

Full `pytest test/modular -q`, serial, at the split point (e922f3434, before concept 6) and at head (82ad7e695), run side by side on a quiet host:
| | failed | passed | skipped |
|---|---|---|---|
| base 305e80e (OMP_NUM_THREADS=2) | 2 | 11460 | 1291 |
| e922f3434 | 4 | 11467 | 1292 |
| 82ad7e695 | 4 | 11476 | 1292 |
The four failures at both commits: test_client_error_text.py::test_a_transcription_whose_decode_fails_in_the_frontend_answers_with_an_error_id (fails at base too) and three test_wan_vae_decode.py cases (test_warm_cache_stream_is_the_whole_stream_decode[wan21], [wan22], test_clip_start_alone_and_a_ragged_tail[1-False-wan22]). These two runs did not set OMP_NUM_THREADS=2 as the baseline did (a wrapper omission); AGENTS.md documents test_wan_vae_decode.py's bit-exact cases as failing at 4-8 intra-op threads. Rerun with the baseline's settings (OMP_NUM_THREADS=2, PYTHONHASHSEED=0) at both commits: test_wan_vae_decode.py, test_client_error_text.py and test_tp_read_failure_acceptance.py give 1 failed, 68 passed each, the failure being the pre-existing test_client_error_text one. test_tp_read_failure_acceptance[text_only], which failed in the baseline run (under load), passes at both commits. No new failures.
Note on these runs: when their 2-hour `timeout` wrappers were removed at 65% so the runs could finish, the wrapper's cleanup deleted their --basetemp directories for about 10 s before they were recreated; no test errored on a missing temp path (no ERROR lines).

New tests fail without their code commit (checked in a worktree at the parent with the test file checked out):
- concept 2: both test_qwen3_omni_thinker_* in test_step_prep_host_syncs.py fail (scalar index writes; sync-debug error on CUDA).
- GPU tests at head (82ad7e695), H200, torch 2.12.1+cu132: test_step_prep_host_syncs.py -k qwen3_omni and test_qwen3_omni_audio_row.py, 8 passed (the CUDA sync-debug test included).
- concept 4: test_an_audio_row_embeds_its_ids_in_one_call fails on main (2 calls); the equivalence test passes on main once main's lazy sentinel attributes are set, i.e. the row is unchanged.
- concept 5: the three test_a_request_with_no_audio_output_builds_no_talker_masks cases fail.
- concept 6: test_gather_policy.py fails to import GatherPolicy; test_a_node_without_a_gather_policy_steps_as_if_nothing_gathered fails (no `_gathering`).

## Review (independent subagent, given only the diff and the base)

Its answers, by piece:
- GatherPolicy / NodeSubmodule.gather_policy: belongs beside step_order/max_batch_size (model declares, scheduler acts); base class declares the hook, so Engine.get_gather_policy calls a declared method; no new coupling beyond submodule -> engine -> scheduler. Docstring's motivation sentence more than needed.
- AudioEncoderSubmodule.gather_policy: policy belongs in the model, but wait_s=0.030 / busy_requests=16 are deployment tuning hardcoded in Python (invariants 2, 9); rows=MAX_BATCH_SIZE couples to the class cap.
- Scheduler gating: belongs in MicroScheduler; couples the decision to len(self.arrival) and wall-clock time. No lockstep hazard: only leader-run nodes consult it; followers replay rank 0's broadcast (`_try_schedule_tp_follow`, which never calls `_gathering`). Found: `has_ready_excluding` ("does NOT modify queue state") started gather timers; its row count (graph-ready rows per walk) can differ from get_next_batch's (engine-ready rows per key, combined walks) -> at worst a wasted pass. Timers were cleared only in note_ran, so a wait that ended by load dropping left a stale start (errs toward running). Held rows are re-checked every pass while they wait (invariant 6 pattern; bounded by wait_s).
- `_audio_row`: matches the old path (positions, advance, masks); no host syncs; fine as model code; `copy_` would silently cast a dtype mismatch where cat raised. Docstring narrated more than needed.
- audio_output=False drops masks_for_talker: every consumer reads it with .get; fine.
- Tests: the first version of the no-policy scheduler test compared a run with itself (adds no coverage); the CUDA sync test's docstring was stale after concept 4 (no sentinel cache any more).

Changed after it (history rewritten so each fix sits in the commit that introduced the code):
- `_gathering(key, rows, start=True)`: the peek in has_ready_excluding passes start=False and starts no wait; a pass that finds the policy no longer applies ends the wait. Tests: test_a_peek_does_not_start_the_wait, test_a_wait_the_load_no_longer_calls_for_ends.
- The no-policy test now records every `_gathering` answer through the real method and asserts it was asked and never said yes, then that the steps equal a run with gathering patched out.
- Trimmed docstrings: GatherPolicy, `_audio_row`, `_media_mask`; fixed the CUDA test docstring.
- Not changed: the policy's numbers stay in the model (see Open).

## Open

- GatherPolicy's numbers (8 rows, 30 ms, 16 busy requests) are constants in AudioEncoderSubmodule. The reviewer pointed at AGENTS.md invariants 2 and 9 (deployment tuning belongs in the serving config); moving them would add a config surface, so they stay where rp3 put them.
- GatherPolicy trades TTFT for throughput: A2T c=32 TTFT p50 +24% against main (+34% against after-5) for +8.7% throughput over after-5. Not measured: audio in with audio out (Talker requests count towards busy, so a lone clip can wait the full 30 ms), several clips in one prompt (waits once per clip), TP, other models with a policy (none declares one).
- The peek in `has_ready_excluding` counts graph-ready rows per walk; `get_next_batch` counts engine-ready rows per scheduling key (a combined walk). They can disagree, costing a pass with no step; not observed or measured.
- Rows a policy holds are re-scanned for readiness every pass while they wait (AGENTS.md invariant 6 pattern), bounded by wait_s; per-pass CPU cost not measured with benchmark/worker_phases.
- Concept 2's own effect on main is not isolated (the ladder's first arm holds 2 and 4), and on this branch it only remains on image/video rows, which no measured workload uses.
- Concept 4's +9.7% at A2T c=32 is larger than the 66 -> 50 op count suggests; attributing it to GPU-thread host time per audio row is a hypothesis without a profile on main.
- From the dropped concept 1 (a3f3b6a4): on main an audio/vision row's text ids ride `text_token_ids`, not `input_ids`, so the sampler's `apply_cached_prefix` (mstar/engine/resources/sampler/resource.py:311 at base) keeps nothing for a walk the prefix cache serves whole, and that walk runs no plan; read from the code, not reproduced. It only matters for a repeated prompt with a repetition penalty. Left out because main's design comment says ids cannot ride `input_ids` there.
- Thinker literal agreement bar (>= 0.99804) is missed by main and by the PR alike; all disagreements are bf16 ties.
- Absolute throughput here is below the loop's for comparable cells because CPU frequency boosting is off on both arms, as the brief requires.
- test_eval_compare.py::test_a_job_whose_lease_is_not_renewed_stops_itself_and_its_servers fails under heavy host load (1.5 s lease around a fake server's startup) and passes alone; it does not touch PR code. The full serial runs passed it.

## Environment notes
- venvs /workspace/venv-main305, /workspace/venv-pr; pytest-xdist installed into both (not in [all]). torchaudio's C extension fails to load against the installed torch in both venvs (as in the loop's venv); every run puts the shim /workspace/results/shim on PYTHONPATH (both arms).
