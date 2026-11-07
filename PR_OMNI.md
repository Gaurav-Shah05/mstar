IN PROGRESS: final gather T2T recheck; writing

# PR facts: Qwen3-Omni

- Base: perf/rp4-pr-gather (fffa8931f, PR_GATHER.md), itself on upstream main 305e80ef46248ade6e6670aa20c9bb67f4ac2e26
- Branch: perf/rp4-pr-omni (force-updated on Nov 7; the first draft was 82ad7e695, kept locally as backup/rp4-pr-omni-82ad7e6)
- Head: 7d2480e01 (git rev-parse perf/rp4-pr-omni)

## Commits (in order, on top of perf/rp4-pr-gather)

| # | Commit | Subject | Source |
|---|---|---|---|
| 1 | 7da2681c3 | model: set qwen3-omni thinker sentinel slots without host syncs | rp4 68785670, ported |
| 2 | 873c54830 | model: build a qwen3-omni audio walk row in a few launches | rp4 387d2c4f + a28a4e57, ported; a comment in `get_input_sequence_len` now names `_audio_row` beside `_spliced_row` |
| 3 | b6bfaa0d8 | model: no qwen3-omni talker masks for a request with no audio output | rp3 78d41d65, ported (conflicts with 1 and 2 resolved) |
| 4 | 2571ca8cc | model: qwen3-omni's audio encoder gathers its rows while the worker is busy | the Qwen3-Omni lines of rp3 f35f5c34 |
| 5 | 987c3b16d | test: qwen3-omni thinker prefill prep writes no scalars by index | rp4 0cf8457d + a5921bde, ported |
| 6 | 99804f867 | test: qwen3-omni audio walk row is its spans end to end, embedded in one call | new; the fixture now sets the sentinel-embedding attributes __init__ sets, so on the parent only the call count fails |
| 7 | 7d2480e01 | test: qwen3-omni builds no talker masks without audio output | rp3 afbf77c9 (Rope counter seeded, as the file's other tests do) |

No message carries a trailer.

### Dropped
- Concept 1 (60bb897c, a3f3b6a4) and concept 3 (e3c6427c, a3fc3e02): main's #1110 (a0470c32b) already prefills each attachment together with all the text before it and, for the last one, the text after it (`_thinker_steps`, mstar/model/qwen3_omni/qwen3_omni_model.py:180 at base; `_spliced_row`, submodules.py:651). That is a superset of both concepts (no page-size limit on the leading span), and its KV layout keys the steps it runs (#1165). The docs commit 66d60d5a documents concepts 1/3 and is dropped with them.
- a3f3b6a4's remaining point (penalty ids of an audio row survive a whole-walk prefix-cache hit): main carries those ids as `text_token_ids`, not `input_ids` (submodules.py:679 at base: "Ids cannot ride `input_ids`"). Left as main has it; see Open.
- 5609484e + revert c1e93f7b; f2c07a72, 898f7450, revert 4fdfd0e3 (replaced by rp3's concept 5); 11870717, 7b360407 (CPU frequency PR, rp2).


### Ported (what changed)
- Concept 2: main still wrote `mm_mask[0] = False` / `[-1]` on audio and vision rows (submodules.py:552-553, 585-586 at main), under a comment that said the write is not a sync. One helper `_media_mask` holds the `fill_` and the reason.
- Concept 4: main's audio row went through `_spliced_row` (an embed call per text side, a position helper per span, masks concatenated). `_audio_row` builds it from one id buffer, one embedding call, one position range and masks filled in place; bit-identical to main's row (/workspace/prwork/scripts/c4_compare.py, four layouts). Dispatched non-view ops for a text-clip-text row 66 -> 50. `get_rope_index_audio` and `_wrap_audio_input` deleted (no other user). The "only a request that speaks builds masks" part of 387d2c4f is concept 5's.
- Concept 5: rp3's change on main's `_spliced_row` and text rows, plus the same gate in `_audio_row`.
- Concept 6: rp3's mechanism moved to perf/rp4-pr-gather with the review fixes; only the declaration stays here, as its own `model:` commit.

## Concepts

### 2. The Thinker's sentinel slots are set without host syncs (7da2681c3, test 987c3b16d)
- Changes: mstar/model/qwen3_omni/submodules.py (`ThinkerSubmodule._media_mask`, used by the image/video row; the audio row until concept 4 replaces it).
- Mechanism: `mask[0] = False` on a CUDA tensor is an index write of a Python scalar, which ATen turns into a copy from a pageable host tensor, so it waits for the queued GPU work (the step in flight). The slots are now set with `fill_` on a view, which is a kernel launch. Main's comment there said the opposite.
- Own effect: not isolated by the ladder (the first ladder arm, after concept 4, holds 2 and 4). On run-base in the loop (round 3) it gave A2T +3.9% c=4, +5.2% c=16, +4.0% c=32 (all rounds), when the audio row still used these writes. On this branch the audio row stops using `_media_mask` in concept 4, so what remains is the image/video row, which no measured workload exercises.
- Cost: none measured.
- Tests: test/modular/test_step_prep_host_syncs.py::test_qwen3_omni_thinker_prefill_prep_writes_no_scalar (CPU, TorchFunctionMode records scalar index writes, audio and image rows), ::test_qwen3_omni_thinker_audio_prep_takes_no_host_sync (CUDA sync-debug "error" mode).

### 4. An audio walk row is built in a few launches (873c54830, test 99804f867)
- Changes: mstar/model/qwen3_omni/submodules.py (`_audio_row` replaces the audio branch of `prepare_inputs`, `_wrap_audio_input` and its cached sentinel embeddings), mstar/model/qwen3_omni/components/rope.py (`get_rope_index_audio` deleted, no other user), a comment in qwen3_omni_model.py naming the new function.
- Mechanism: main built the row span by span (an embedding call per text side, a position helper per span, masks concatenated from per-span pieces) on the GPU thread. The row's ids now fill one buffer (text before, start sentinel, placeholder rows, end sentinel, text after) embedded in one call, then the encoder's rows are copied over the placeholders; positions are one `arange`, since text and audio both advance all three MRoPE components by one per row (as main's helpers already did); the Talker masks are filled in place. Output is bit-identical to main's row.
- Own effect (ladder, A2T c=32, includes concept 2): 1265.0 [1260.2-1302.7] -> 1387.1 [1377.6-1407.2] tok/s (+9.7%, every round above main's); TTFT p50 115.3 -> 106.3 ms (-7.8%). T2S c=16 (text-only, path unchanged): 99.7 -> 98.8 audio-s/s, overlapping. Dispatched non-view ops for a text-clip-text row 66 -> 50 (CPU count). The throughput gain is larger than the op count suggests; that it comes from GPU-thread host time per audio row is a hypothesis (no profile on main).
- Cost: none measured. Embeds the placeholder rows before overwriting them (one embedding row per audio frame, wasted).
- Tests: test/modular/test_qwen3_omni_audio_row.py::test_an_audio_row_is_its_spans_end_to_end (4 layouts: embeds, positions, both mask rows, advance, text ids against the spans laid end to end; also passes on main), ::test_an_audio_row_embeds_its_ids_in_one_call (fails on main: 2 calls).

### 5. No Talker masks for a request with no audio output (b6bfaa0d8, test 7d2480e01; rp3's 78d41d65 / afbf77c9)
- Changes: mstar/model/qwen3_omni/submodules.py: text rows, `_audio_row`, `_media_mask` / `_spliced_row` (image/video) build `masks_for_talker` only when `step_metadata["audio_output"]`; `split_inputs` cuts the key only when present.
- Mechanism: only the Talker reads these masks, and a request without audio output never reaches the Talker. Every consumer already reads the key with `.get` (preprocess, unpack_packed_outputs; postprocess returns before reading them for such requests). The per-row `talker_text_mask` passes, stack and fills are skipped.
- Own effect (ladder): A2T c=32 1387.1 [1377.6-1407.2] -> 1403.3 [1390.4-1405.1] tok/s (+1.2%, overlapping); TTFT 106.3 -> 106.6 ms. T2T c=1 (text rows and the scheduler's policy lookup are the only PR code on that path) is +1.5% head over main with every round above (310.2 -> 314.9 tok/s); no ladder arm isolates it. rp3 measured +3-6% A2T on its base, where the audio row still built its masks span by span.
- Cost: none measured.
- Tests: test/modular/test_qwen3_omni_prefix_cache.py::test_a_request_with_no_audio_output_builds_no_talker_masks[prefill_text|prefill_audio|prefill_vision] (row and a cut chunk carry no masks).

### 6. Qwen3-Omni's audio encoder declares a gather policy (2571ca8cc; the mechanism is perf/rp4-pr-gather, PR_GATHER.md)
- Changes: mstar/model/qwen3_omni/submodules.py, `AudioEncoderSubmodule.gather_policy`: `GatherPolicy(rows=MAX_BATCH_SIZE (8), wait_s=0.030, busy_requests=16)`.
- Mechanism: a clip's row would otherwise take a Thinker prefill step of its own, so once the worker holds 16 requests a ready clip waits up to 30 ms for up to 8 clips, and the Thinker prefills them in fewer steps.
- Own effect: the policy only acts at 16 or more requests. At the head against main (below): A2T +2.1% c=1 and +5.4% c=8 (no policy action there: concepts 2, 4, 5), +6.7% c=16, +19.5% c=24, +17.9% c=32. The Nov 6 ladder (first draft) put the policy's own share at c=32 at +8.7% over after-concept-5.
- Cost: TTFT p50 +15% at c=16 (84.6 -> 97.3 ms) and +6% at c=24 (102.1 -> 108.1); level at c=32 (117.8 -> 118.0). TTFT p99 at c=16-32 is level or lower (203.5 vs 264.3 ms at c=16, overlapping). The first draft, whose peek never started the wait, cost +24% p50 at c=32; with the fix the wait ends on time. Not measured: audio in with audio out (Talker requests count toward busy), several clips per prompt (waits once per clip), TP.
- Tests: the mechanism's are on perf/rp4-pr-gather; the declaration is a constant with no test of its own.

## Main against PR (Nov 7, at the final heads)

Settings, same on every arm: configs/qwen3omni_colocated.yaml (identical in both trees), one H200, `--rust-frontend`, MSTAR_SHM_ARENA=1, MSTAR_STEP_THREAD_CORE=siblings, no full_cpu_frequency / step_thread_util_min knobs (absent on main) and no uclampset. Fresh server per round, one untimed warm pass per server, A2T rows never repeated within a server, requests per cell A2T 16 x c (min 64), T2T/T2S 5 x c (min 32). Each arm keeps its own Inductor cache across its rounds. 3 alternating rounds: main, omni, gather; gather, omni, main; main, omni, gather (harness /workspace/prwork/round_pr.sh, ab3.sh; client /workspace/results/tools/omni_bench.py; runs/pr_ab3). No request failed. Median [min-max]; ratio omni/main of medians, "all above/below" when the ranges do not overlap.

| cell | metric | main 305e80e | pr-omni 7d2480e01 | (unused) | pr-gather fffa8931f | omni/main |
|---|---|---|---|---|---|---|
| a2t_c1 | tput | 196.2 [193.4-197.5] | 200.4 [198.8-200.5] | n/a | n/a | 1.021 (all above) |
| a2t_c1 | p50 ms | 40.9 [39.7-43.0] | 39.3 [38.9-39.5] | n/a | n/a | 0.961 (all below) |
| a2t_c1 | p99 ms | 58.4 [56.3-62.4] | 58.5 [56.4-61.6] | n/a | n/a | 1.001 |
| a2t_c8 | tput | 568.8 [559.9-571.6] | 599.2 [599.2-605.1] | n/a | n/a | 1.054 (all above) |
| a2t_c8 | p50 ms | 70.3 [67.6-73.9] | 67.1 [65.5-67.6] | n/a | n/a | 0.955 (all below) |
| a2t_c8 | p99 ms | 166.2 [140.3-193.9] | 122.8 [117.6-136.9] | n/a | n/a | 0.739 (all below) |
| a2t_c16 | tput | 900.1 [878.7-910.5] | 960.7 [949.9-981.6] | n/a | n/a | 1.067 (all above) |
| a2t_c16 | p50 ms | 84.6 [81.6-87.2] | 97.3 [95.6-98.0] | n/a | n/a | 1.150 (all above) |
| a2t_c16 | p99 ms | 264.3 [212.5-321.2] | 203.5 [183.0-300.5] | n/a | n/a | 0.770 |
| a2t_c24 | tput | 1044.4 [1032.5-1067.0] | 1248.0 [1223.6-1248.5] | n/a | n/a | 1.195 (all above) |
| a2t_c24 | p50 ms | 102.1 [100.4-103.3] | 108.1 [107.4-108.3] | n/a | n/a | 1.060 (all above) |
| a2t_c24 | p99 ms | 289.3 [282.5-447.2] | 263.7 [231.5-347.7] | n/a | n/a | 0.912 |
| a2t_c32 | tput | 1211.5 [1210.6-1214.8] | 1428.7 [1407.7-1432.8] | n/a | n/a | 1.179 (all above) |
| a2t_c32 | p50 ms | 117.8 [112.2-117.8] | 118.0 [116.2-119.4] | n/a | n/a | 1.002 |
| a2t_c32 | p99 ms | 314.8 [293.3-348.1] | 317.7 [303.2-340.7] | n/a | n/a | 1.009 |
| t2t_c1 | tput | 310.3 [309.7-310.4] | 315.2 [314.7-315.4] | n/a | n/a | 1.016 (all above) |
| t2t_c1 | p50 ms | 26.6 [26.5-27.3] | 26.0 [25.8-27.4] | n/a | n/a | 0.976 |
| t2t_c1 | p99 ms | 31.7 [30.7-32.1] | 30.6 [30.2-32.3] | n/a | n/a | 0.967 |
| t2t_c32 | tput | 2289.3 [2288.2-2296.2] | 2293.6 [2286.9-2298.3] | n/a | 2257.9 [2251.7-2262.9] | 1.002 |
| t2t_c32 | p50 ms | 58.5 [56.1-62.6] | 60.2 [58.9-62.6] | n/a | 62.2 [56.9-62.4] | 1.030 |
| t2t_c32 | p99 ms | 153.3 [147.4-155.5] | 135.6 [123.6-148.9] | n/a | 165.1 [160.9-191.6] | 0.885 |
| t2s_c4 | tput | 44.1 [44.0-44.8] | 43.8 [43.7-45.4] | n/a | n/a | 0.992 |
| t2s_c4 | p50 ms | 194.9 [184.7-195.4] | 170.5 [168.4-180.4] | n/a | n/a | 0.875 (all below) |
| t2s_c4 | p99 ms | 244.1 [236.4-250.1] | 245.5 [237.8-247.8] | n/a | n/a | 1.006 |
| t2s_c16 | tput | 96.8 [96.1-97.3] | 97.0 [95.3-98.7] | n/a | 99.9 [96.8-100.1] | 1.002 |
| t2s_c16 | p50 ms | 256.8 [247.1-259.1] | 250.2 [248.5-255.3] | n/a | 255.9 [249.9-259.9] | 0.975 |
| t2s_c16 | p99 ms | 367.6 [355.5-373.2] | 369.9 [358.4-376.1] | n/a | 412.6 [381.2-444.5] | 1.006 |

60 rows reported with mstar-report (batch h200_1106, status counted, hypothesis "PR check: Qwen3-Omni" / "PR check: gather policy"; engines `M* main 305e80e`, `M* pr-omni 7d2480e01`, `M* pr-gather fffa8931f`; each row's config names every flag and knob). Earlier sessions (same harness, also reported): 4e64ad1ca and the step-count variant (runs/pr_ab2, 87 rows), the first draft 82ad7e69 (Nov 6, 78 rows).

Reading: A2T gains grow with concurrency, every round above main from c=1 to c=32. T2T c=1 +1.6% (all above; text rows build no masks), T2T c=32 level, T2S c=4 and c=16 level. T2S c=4 TTFA p50 is lower (170.5 vs 194.9 ms, all below) with p99 level; text-only speech runs no changed model code, so that is not attributed to the PR.

Concept ladder (Nov 6, on the first draft, whose code for concepts 2, 4 and 5 is the same as here): A2T c=32 main 1265.0 -> after concepts 2+4 1387.1 (+9.7%) -> after 5 1403.3 (+1.2%) -> with the old gather policy 1525.7.

## Correctness (at 7d2480e01, against main 305e80e; the A/B's round 1 servers)

Thinker, 64 fixed LibriSpeech validation-clean clips (rows 0-63), greedy, exact token ids, HF transformers bf16 teacher-forced on each run's tokens (/workspace/results/tools/hf_ref.py, hf_diag.py; reference /workspace/results/ref):

| | main | head | HF sdpa | HF eager |
|---|---|---|---|---|
| corpus WER | 19.47% | 20.35% | 15.29% | 15.19% |
| WER excluding the answered clip | 0.88% | 0.88% | 1.08% | 1.08% |
| clips answered instead of transcribed | 1 (clip 4) | 1 (clip 4) | 1 | 1 |
| transcripts identical to HF sdpa | 59/64 | 60/64 | | |
| teacher-forced top-1 agreement (hf_ref) | 0.99660 | 0.99592 | | sdpa on eager 0.99804 |
| agreement, second HF pass (hf_diag) | 0.98980 (15 diffs) | 0.99048 (14 diffs) | sdpa on itself 0.99739 | |

The literal bar (>= 0.99804) is missed by main and head alike, and by HF against itself in a second pass. Head's 14 disagreements have HF top-1 margins of 0.0 (8), 0.125 (5) and 0.25 (1): bf16 ties. Head mismatches sit in clip 4 (the free-form answer) and clip 54 (one token). Corpus WER moves only with clip 4's answer. Earlier sessions: 4e64ad1ca 15.00% vs main 15.58%, first draft 21.13% vs 20.84%; WER excluding clip 4 0.88% in every arm and session.

Talker: ASR round trip (benchmark.answer_audio_wer, Whisper large-v3-turbo, each spoken answer against its own text), T2S c=4, 32 answers per arm:

| | main | head |
|---|---|---|
| mean WER | 0.092 | 0.229 |
| median | 0.046 | 0.048 |
| p90 | 0.168 | 0.211 |
| answers above 0.5 | 2 | 2 (Fisher exact p = 1.0) |

Head's mean comes from one answer (request 28, a reply listing weather URLs, read back as noise: WER 4.64, 99.7 s of audio); without it head's mean is 0.087. Main speaks the same text at WER 0.12 (60.1 s). That pair repeated in every session at these heads, so I tested where it comes from (runs/nowarm, fresh servers with no warm pass, T2T and T2S at c=1, 12 prompts each, text compared byte for byte):

| comparison | T2T identical text | T2S identical text |
|---|---|---|
| main vs main, same Inductor cache, two servers | 12/12 | 12/12 (and audio lengths 12/12) |
| head vs head, same cache, two servers | 12/12 | 12/12 (audio lengths 12/12) |
| main vs head | 4/12 | 1/12 |
| main with a fresh cache vs main with its cache | 7/12 (8/12 second fresh run) | 6/12 (7/12) |
| main fresh vs main fresh, two compiles | 8/12 | 6/12 |

Main's own code diverges from itself as much across two compiles as it does from the head, so sampled text and speech here depend on each server's compiled kernels. Each arm reuses its own cache across rounds, which is why its outputs repeat. The head's garbled answer is therefore not attributable to the PR, and text-only speech requests run no model code the PR changes. Codec tokens cannot be compared across compiles; greedy codec identity was not run.

## Tests

Full `pytest test/modular -q`, serial, CPU only, OMP_NUM_THREADS=2, PYTHONHASHSEED=0, on a quiet host:
| tree | failed | passed | skipped |
|---|---|---|---|
| main 305e80e | 1 | 11461 | 1291 |
| perf/rp4-pr-gather fffa8931f | 1 | 11472 | 1291 |
| perf/rp4-pr-omni 7d2480e01 | 1 | 11481 | 1292 |
The one failure in every tree is pre-existing: test/modular/test_client_error_text.py::test_a_transcription_whose_decode_fails_in_the_frontend_answers_with_an_error_id. (An earlier, loaded base run also failed test_tp_read_failure_acceptance[text_only], which passes on the quiet runs.)

Per commit (ruff + targeted, serial): 7afdf86a2 scheduler files 252 passed; 7da2681c3, 873c54830, b6bfaa0d8, 2571ca8cc: 2052 passed, 40 skipped (the 34 test/modular files importing qwen3_omni + mstar/model/qwen3_omni); 987c3b16d 2053 passed, 41 skipped; 99804f867 2058 passed, 41 skipped; ruff clean on every commit. Both heads are covered by the full runs.

New tests on the parent of their code commit (test file checked out there): test_step_prep_host_syncs.py at fffa8931f fails test_qwen3_omni_thinker_prefill_prep_writes_no_scalar; test_qwen3_omni_audio_row.py at 7da2681c3 fails only test_an_audio_row_embeds_its_ids_in_one_call (the equivalence test passes, so the row is unchanged); test_qwen3_omni_prefix_cache.py at 873c54830 fails the three no-audio-output cases.
GPU, H200, torch 2.12.1+cu132, at 7d2480e01: test_step_prep_host_syncs.py -k qwen3_omni and test_qwen3_omni_audio_row.py, 8 passed (the CUDA sync-debug test included).

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


Second review (Nov 7 amendment) and a third independent review of the two final diffs (given only the diffs and the base):
- Fixed: the peek that never started the wait, the no-policy test that could not fail, `_gathering`'s docstring, the docs' per-item wait (PR_GATHER.md); the audio-row fixture; the `get_input_sequence_len` comment naming only `_spliced_row`.
- Third review, fixed: the peek counted graph-ready rows while `get_next_batch` counted engine-ready rows, so with enough rows on the queue but too few the engine would take, the peek ended the wait and yielded and `get_next_batch` started a new one, pass after pass, until the chain stopped (a bug the item-2 fix introduced; now both count engine-ready rows, test_a_peek_counts_the_rows_the_engine_would_take, which fails without the fix); `Engine.get_gather_policy` gained its docstring; `_audio_row`'s comment on the placeholder id was wrong (the first slot keeps it as the start sentinel); the policy comment stated a performance claim; the fixture comment named the wrong `__init__`; `_media_mask`'s docstring shortened.
- Third review, not changed: the docs' "a prompt with K such items waits once per item" (the amendment asked for it); the policy's numbers in Python (Open).
- Item 3 (step count) measured and not kept (PR_GATHER.md).

## Open

- GatherPolicy's numbers (8 rows, 30 ms, 16 busy requests) are constants in AudioEncoderSubmodule, not serving config (AGENTS.md invariants 2, 9).
- The gather policy costs TTFT p50 at A2T c=16 (+15%) and c=24 (+6%). Not measured: audio in with audio out (Talker requests count toward busy), several clips per prompt, TP, the 2-GPU layout (where `len(self.arrival)` also counts requests speaking on rank 0).
- If every held row of a walk is removed, its wait start stays until the next row's call; the next row may then skip its wait (errs toward running).
- Concept 2's own effect on main is not isolated, and on this branch it only remains on image/video rows, which no measured workload uses.
- Concept 4's gain is larger than its op count suggests; attributing it to GPU-thread host time per audio row is a hypothesis without a profile on main.
- From the dropped concept 1 (a3f3b6a4): an audio/vision row's text ids ride `text_token_ids`, not `input_ids`, so the sampler's `apply_cached_prefix` (mstar/engine/resources/sampler/resource.py:311 at main) keeps nothing for a walk the prefix cache serves whole; read from the code, not reproduced.
- Sampled outputs differ between compiles of the same code (runs/nowarm), so correctness comparisons across server arms can only be statistical; a greedy, fixed-kernel comparison was not run.
- Absolute throughput is below the loop's numbers because CPU frequency boosting is off on both arms.

## Environment notes
- venvs /workspace/venv-main305 (main) and /workspace/venv-pr (PR arms, PYTHONPATH at a detached worktree of the measured commit); identical package sets (uv pip freeze); pytest-xdist installed in both. torchaudio's C extension fails to load against the installed torch in both; every run puts /workspace/results/shim on PYTHONPATH.
