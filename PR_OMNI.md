IN PROGRESS: full test runs at the split point and head; T2S attribution check (after concept 5 vs head) next

# PR facts: Qwen3-Omni

- Base: 305e80ef46248ade6e6670aa20c9bb67f4ac2e26 (upstream main)
- Branch: perf/rp4-pr-omni
- Head: 82ad7e695

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

## Environment notes
- venvs /workspace/venv-main305, /workspace/venv-pr; pytest-xdist installed into both (not in [all]). torchaudio's C extension fails to load against the installed torch in both venvs (as in the loop's venv); every run puts the shim /workspace/results/shim on PYTHONPATH (both arms).

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

New tests fail without their code commit (checked in a worktree at the parent with the test file checked out):
- concept 2: both test_qwen3_omni_thinker_* in test_step_prep_host_syncs.py fail (scalar index writes; sync-debug error on CUDA).
- GPU tests at head (82ad7e695), H200, torch 2.12.1+cu132: test_step_prep_host_syncs.py -k qwen3_omni and test_qwen3_omni_audio_row.py, 8 passed (the CUDA sync-debug test included).
- concept 4: test_an_audio_row_embeds_its_ids_in_one_call fails on main (2 calls); the equivalence test passes on main once main's lazy sentinel attributes are set, i.e. the row is unchanged.
- concept 5: the three test_a_request_with_no_audio_output_builds_no_talker_masks cases fail.
- concept 6: test_gather_policy.py fails to import GatherPolicy; test_a_node_without_a_gather_policy_steps_as_if_nothing_gathered fails (no `_gathering`).

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
Text-only T2S prompts run code the PR does not change (prefill_text rows; masks still built since they speak; no audio encoder). The loop's noise floor for this measure was per-arm means 0.035-0.085 on identical code. A second sample per arm comes from the A/B's round 1 (audio kept).

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
