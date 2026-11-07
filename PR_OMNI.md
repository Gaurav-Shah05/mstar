IN PROGRESS: full test runs at both heads; per-commit checks

# PR facts: Qwen3-Omni

- Base: perf/rp4-pr-gather (803c6682e88cc5770930b7a89f17013c045d6820, PR_GATHER.md), itself on upstream main 305e80ef46248ade6e6670aa20c9bb67f4ac2e26
- Branch: perf/rp4-pr-omni (force-updated on Nov 7; the first draft was 82ad7e695, kept locally as backup/rp4-pr-omni-82ad7e6)
- Head: 4e64ad1cae03370976cfb1dc33f6d1a35dbd2b03

## Commits (in order, on top of perf/rp4-pr-gather)

| # | Commit | Subject | Source |
|---|---|---|---|
| 1 | cb4551551 | model: set qwen3-omni thinker sentinel slots without host syncs | rp4 68785670, ported |
| 2 | 98639be8f | model: build a qwen3-omni audio walk row in a few launches | rp4 387d2c4f + a28a4e57, ported; a comment in `get_input_sequence_len` now names `_audio_row` beside `_spliced_row` |
| 3 | 5aeb0e3e4 | model: no qwen3-omni talker masks for a request with no audio output | rp3 78d41d65, ported (conflicts with 1 and 2 resolved) |
| 4 | f8d37db97 | model: qwen3-omni's audio encoder gathers its rows while the worker is busy | the Qwen3-Omni lines of rp3 f35f5c34 |
| 5 | c48397f6d | test: qwen3-omni thinker prefill prep writes no scalars by index | rp4 0cf8457d + a5921bde, ported |
| 6 | 373334988 | test: qwen3-omni audio walk row is its spans end to end, embedded in one call | new; the fixture now sets the sentinel-embedding attributes __init__ sets, so on the parent only the call count fails |
| 7 | 4e64ad1ca | test: qwen3-omni builds no talker masks without audio output | rp3 afbf77c9 (Rope counter seeded, as the file's other tests do) |

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

### 2. The Thinker's sentinel slots are set without host syncs (cb4551551, test c48397f6d)
- Changes: mstar/model/qwen3_omni/submodules.py (`ThinkerSubmodule._media_mask`, used by the image/video row; the audio row until concept 4 replaces it).
- Mechanism: `mask[0] = False` on a CUDA tensor is an index write of a Python scalar, which ATen turns into a copy from a pageable host tensor, so it waits for the queued GPU work (the step in flight). The slots are now set with `fill_` on a view, which is a kernel launch. Main's comment there said the opposite.
- Own effect: not isolated by the ladder (the first ladder arm, after concept 4, holds 2 and 4). On run-base in the loop (round 3) it gave A2T +3.9% c=4, +5.2% c=16, +4.0% c=32 (all rounds), when the audio row still used these writes. On this branch the audio row stops using `_media_mask` in concept 4, so what remains is the image/video row, which no measured workload exercises.
- Cost: none measured.
- Tests: test/modular/test_step_prep_host_syncs.py::test_qwen3_omni_thinker_prefill_prep_writes_no_scalar (CPU, TorchFunctionMode records scalar index writes, audio and image rows), ::test_qwen3_omni_thinker_audio_prep_takes_no_host_sync (CUDA sync-debug "error" mode).

### 4. An audio walk row is built in a few launches (98639be8f, test 373334988)
- Changes: mstar/model/qwen3_omni/submodules.py (`_audio_row` replaces the audio branch of `prepare_inputs`, `_wrap_audio_input` and its cached sentinel embeddings), mstar/model/qwen3_omni/components/rope.py (`get_rope_index_audio` deleted, no other user), a comment in qwen3_omni_model.py naming the new function.
- Mechanism: main built the row span by span (an embedding call per text side, a position helper per span, masks concatenated from per-span pieces) on the GPU thread. The row's ids now fill one buffer (text before, start sentinel, placeholder rows, end sentinel, text after) embedded in one call, then the encoder's rows are copied over the placeholders; positions are one `arange`, since text and audio both advance all three MRoPE components by one per row (as main's helpers already did); the Talker masks are filled in place. Output is bit-identical to main's row.
- Own effect (ladder, A2T c=32, includes concept 2): 1265.0 [1260.2-1302.7] -> 1387.1 [1377.6-1407.2] tok/s (+9.7%, every round above main's); TTFT p50 115.3 -> 106.3 ms (-7.8%). T2S c=16 (text-only, path unchanged): 99.7 -> 98.8 audio-s/s, overlapping. Dispatched non-view ops for a text-clip-text row 66 -> 50 (CPU count). The throughput gain is larger than the op count suggests; that it comes from GPU-thread host time per audio row is a hypothesis (no profile on main).
- Cost: none measured. Embeds the placeholder rows before overwriting them (one embedding row per audio frame, wasted).
- Tests: test/modular/test_qwen3_omni_audio_row.py::test_an_audio_row_is_its_spans_end_to_end (4 layouts: embeds, positions, both mask rows, advance, text ids against the spans laid end to end; also passes on main), ::test_an_audio_row_embeds_its_ids_in_one_call (fails on main: 2 calls).

### 5. No Talker masks for a request with no audio output (5aeb0e3e4, test 4e64ad1ca; rp3's 78d41d65 / afbf77c9)
- Changes: mstar/model/qwen3_omni/submodules.py: text rows, `_audio_row`, `_media_mask` / `_spliced_row` (image/video) build `masks_for_talker` only when `step_metadata["audio_output"]`; `split_inputs` cuts the key only when present.
- Mechanism: only the Talker reads these masks, and a request without audio output never reaches the Talker. Every consumer already reads the key with `.get` (preprocess, unpack_packed_outputs; postprocess returns before reading them for such requests). The per-row `talker_text_mask` passes, stack and fills are skipped.
- Own effect (ladder): A2T c=32 1387.1 [1377.6-1407.2] -> 1403.3 [1390.4-1405.1] tok/s (+1.2%, overlapping); TTFT 106.3 -> 106.6 ms. T2T c=1 (text rows and the scheduler's policy lookup are the only PR code on that path) is +1.5% head over main with every round above (310.2 -> 314.9 tok/s); no ladder arm isolates it. rp3 measured +3-6% A2T on its base, where the audio row still built its masks span by span.
- Cost: none measured.
- Tests: test/modular/test_qwen3_omni_prefix_cache.py::test_a_request_with_no_audio_output_builds_no_talker_masks[prefill_text|prefill_audio|prefill_vision] (row and a cut chunk carry no masks).

### 6. Qwen3-Omni's audio encoder declares a gather policy (f8d37db97; the mechanism is perf/rp4-pr-gather, PR_GATHER.md)
- Changes: mstar/model/qwen3_omni/submodules.py, `AudioEncoderSubmodule.gather_policy`: `GatherPolicy(rows=MAX_BATCH_SIZE (8), wait_s=0.030, busy_requests=16)`.
- Mechanism: a clip's prompt row costs the Thinker most of a mixed step of its own, so once the worker holds 16 requests a ready clip waits up to 30 ms for up to 8 clips, and the Thinker prefills them in fewer steps.
- Own effect: head against main crossing the 16-request threshold (A2T, below): +1.0% c=1, +5.1% c=8 (both below the threshold: concepts 2/4/5), +8.0% c=16, +14.7% c=24, +14.3% c=32. On 82ad7e69 (the policy as rp3 wrote it, before the peek fix) the ladder gave +8.7% at c=32 over after-concept-5; with the fix head is +14.3% over main against +20.6% then, so the fix gave back part of the gain (different sessions; a hypothesis: the old peek held rows longer, so more clips shared a step).
- Cost: TTFT p50 +20% at c=16 (82.8 -> 99.5 ms), +8% at c=24, +8% at c=32; TTFT p99 is lower than main's at c=16-32 (217.0 vs 292.9 ms at c=16, 325.5 vs 337.2 at c=32, overlapping at c=24/32). Not measured: audio in with audio out, several clips per prompt (waits once per clip), TP.
- Tests: the mechanism's tests are on perf/rp4-pr-gather. The declaration has no test of its own (a constant).

## Main against PR (Nov 7, at 4e64ad1ca)

Settings, same on every arm: configs/qwen3omni_colocated.yaml (identical in both trees), one H200, `--rust-frontend`, MSTAR_SHM_ARENA=1, MSTAR_STEP_THREAD_CORE=siblings, no full_cpu_frequency / step_thread_util_min knobs (absent on main) and no uclampset. Fresh server per round, one untimed warm pass per server, A2T rows never repeated within a server, requests per cell A2T 16 x c (min 64), T2T/T2S 5 x c (min 32). 3 alternating rounds: main, head, steps, gather; then gather, steps, head, main; then main, head, steps, gather (harness /workspace/prwork/round_pr.sh, ab2.sh; client /workspace/results/tools/omni_bench.py; runs/pr_ab2). "steps" is the item-3 variant (PR_GATHER.md), "gather" the gather branch alone. No request failed in any cell. Median [min-max]; ratio head/main of medians, "all above/below" when the ranges do not overlap.

| cell | metric | main | head | steps | gather | head/main |
|---|---|---|---|---|---|---|
| a2t_c1 | tput | 196.2 [195.3-198.8] | 198.3 [198.0-199.2] | 200.0 [196.5-200.2] | n/a | 1.010 |
| a2t_c1 | p50 ms | 40.1 [38.5-40.2] | 39.8 [39.1-40.2] | 39.6 [38.8-41.0] | n/a | 0.993 |
| a2t_c1 | p99 ms | 63.9 [55.8-71.1] | 66.9 [59.7-68.9] | 58.9 [56.4-66.8] | n/a | 1.046 |
| a2t_c8 | tput | 569.0 [568.3-570.5] | 598.0 [596.4-600.9] | 584.8 [582.4-590.2] | n/a | 1.051 (all above) |
| a2t_c8 | p50 ms | 69.1 [67.2-69.4] | 68.2 [66.6-70.4] | 76.1 [75.5-78.1] | n/a | 0.987 |
| a2t_c8 | p99 ms | 155.6 [152.2-191.7] | 149.3 [134.1-155.5] | 229.5 [216.7-230.4] | n/a | 0.960 |
| a2t_c16 | tput | 899.7 [897.7-907.3] | 972.0 [956.0-973.7] | 977.6 [970.9-986.6] | n/a | 1.080 (all above) |
| a2t_c16 | p50 ms | 82.8 [81.3-87.8] | 99.5 [97.7-99.9] | 90.6 [90.6-91.7] | n/a | 1.203 (all above) |
| a2t_c16 | p99 ms | 292.9 [211.6-347.8] | 217.0 [196.5-217.6] | 217.8 [199.8-224.5] | n/a | 0.741 |
| a2t_c24 | tput | 1068.9 [1047.5-1072.2] | 1226.3 [1212.0-1244.7] | 1186.2 [1172.8-1198.7] | n/a | 1.147 (all above) |
| a2t_c24 | p50 ms | 103.5 [100.0-106.2] | 112.0 [110.4-112.5] | 102.4 [102.0-103.8] | n/a | 1.083 (all above) |
| a2t_c24 | p99 ms | 289.9 [270.1-292.8] | 282.3 [268.2-367.9] | 270.0 [264.4-297.8] | n/a | 0.974 |
| a2t_c32 | tput | 1226.3 [1209.2-1227.2] | 1401.9 [1393.2-1432.6] | 1368.9 [1355.9-1376.1] | n/a | 1.143 (all above) |
| a2t_c32 | p50 ms | 115.5 [113.4-122.0] | 124.2 [122.9-127.7] | 114.4 [112.2-115.2] | n/a | 1.075 (all above) |
| a2t_c32 | p99 ms | 337.2 [320.9-352.8] | 325.5 [322.1-371.8] | 332.7 [310.4-345.9] | n/a | 0.965 |
| t2t_c1 | tput | 310.2 [310.1-310.4] | 314.9 [314.7-315.0] | 314.5 [314.1-314.7] | n/a | 1.015 (all above) |
| t2t_c1 | p50 ms | 26.5 [26.2-26.8] | 27.1 [26.7-27.3] | 26.7 [26.1-26.8] | n/a | 1.022 |
| t2t_c1 | p99 ms | 30.8 [30.8-31.4] | 31.4 [31.2-31.8] | 30.9 [29.3-31.8] | n/a | 1.019 |
| t2t_c32 | tput | 2284.2 [2282.6-2289.5] | 2290.4 [2286.9-2291.6] | 2290.2 [2287.9-2294.0] | 2261.6 [2260.6-2261.8] | 1.003 |
| t2t_c32 | p50 ms | 59.4 [56.7-59.9] | 62.3 [56.0-64.6] | 62.6 [60.6-64.8] | 59.6 [57.9-61.0] | 1.050 |
| t2t_c32 | p99 ms | 151.2 [149.3-156.4] | 142.3 [140.0-145.2] | 139.7 [130.8-151.9] | 170.0 [152.8-184.0] | 0.941 (all below) |
| t2s_c4 | tput | 44.0 [43.8-44.0] | 43.5 [43.3-44.4] | 45.1 [44.6-45.9] | n/a | 0.989 |
| t2s_c4 | p50 ms | 179.9 [174.4-182.1] | 177.9 [175.8-178.0] | 177.6 [176.7-184.3] | n/a | 0.989 |
| t2s_c4 | p99 ms | 241.0 [225.5-255.5] | 260.8 [258.8-263.7] | 262.6 [238.1-273.4] | n/a | 1.082 (all above) |
| t2s_c16 | tput | 95.9 [95.7-99.7] | 97.4 [95.7-99.1] | 96.4 [96.3-97.7] | 99.2 [97.6-100.8] | 1.015 |
| t2s_c16 | p50 ms | 250.8 [248.6-256.2] | 251.5 [250.7-255.3] | 251.1 [246.4-255.2] | 255.7 [252.0-267.1] | 1.003 |
| t2s_c16 | p99 ms | 389.2 [381.1-390.8] | 385.9 [381.8-389.5] | 388.7 [383.1-388.8] | 406.1 [373.0-417.3] | 0.991 |

All rows reported with mstar-report (batch h200_1106, status counted; engines `M* main 305e80e`, `M* pr-omni 4e64ad1ca`, `M* pr-omni-steps 129a48868`, `M* pr-gather 803c6682e`; each row's config names every flag and knob): 87 rows from this A/B, 6 from the gather recheck.

Reading: A2T gains grow with concurrency (+1.0% c=1, +5.1% c=8, +8.0% c=16, +14.7% c=24, +14.3% c=32, all above from c=8). T2T c=1 +1.5% (all above; text rows build no masks), T2T c=32 level. T2S c=4 and c=16 level within overlapping rounds (T2S c=4 TTFA p99 +8%, non-overlapping: 260.8 [258.8-263.7] vs 241.0 [225.5-255.5]; text-only speech runs no changed model code, and the step-count arm, which shares that code, shows the same 262.6).

Concept ladder (Nov 6, on the first draft's commits, whose code for concepts 2, 4 and 5 is the same as here): A2T c=32 main 1265.0 -> after concepts 2+4 1387.1 (+9.7%) -> after 5 1403.3 (+1.2%) -> with the old gather policy 1525.7. Per-concept effects above cite it.

## Correctness (at 4e64ad1ca, against main 305e80e; A/B round 1 servers)

Thinker, 64 fixed LibriSpeech validation-clean clips (rows 0-63), greedy, exact token ids, HF transformers bf16 teacher-forced on each run's tokens (/workspace/results/tools/hf_ref.py; reference /workspace/results/ref):

| | main | head | HF sdpa | HF eager |
|---|---|---|---|---|
| corpus WER | 15.58% | 15.00% | 15.29% | 15.19% |
| WER excluding the answered clip | 0.88% | 0.88% | 1.08% | 1.08% |
| clips answered instead of transcribed | 1 (clip 4) | 1 (clip 4) | 1 | 1 |
| transcripts identical to HF sdpa | 59/64 | 59/64 | | |
| teacher-forced top-1 agreement (hf_ref) | 0.99592 | 0.99728 | | sdpa on eager 0.99804 |
| agreement, second HF pass (hf_diag) | 0.99184 (12 diffs) | 0.99320 (10 diffs) | sdpa on itself 0.99739 | |

The literal bar (>= 0.99804) is missed by main and by head alike, and by HF against itself in a second pass. Head's 10 disagreements all have HF top-1 margins of 0.0-0.25 logits (bf16 ties); main has one at 0.375. Head mismatches: clips 4 (the free-form answer), 52 and 54 (one token each), the same clips as main.

Talker: codec tokens are not comparable across server processes (sampling seeded per request but process-dependent); ASR round trip instead (benchmark.answer_audio_wer, Whisper large-v3-turbo, each spoken answer against its own text), T2S c=4, 32 answers per arm:

| | main | head |
|---|---|---|
| mean WER | 0.082 | 0.226 |
| median | 0.046 | 0.048 |
| p90 | 0.166 | 0.294 |
| answers above 0.5 | 1 | 1 (Fisher exact p = 1.0) |

Head's mean comes from one answer: a reply listing weather URLs that the Talker read back as noise (WER 4.64, 99.7 s of audio); without it head's mean is 0.083. The prompts are the same in both arms, the Thinker's sampled text differs. Text-only speech requests run no model code the PR changes. Nov 6 samples (first draft, same speech path): main 0.093 / 0.055, head 0.108 / 0.065. Greedy codec-token identity was not run.

## Tests

(pending: full serial runs at both heads, per-commit checks)

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

Second review (Nov 7 amendment), fixed: the peek that never started the wait and the no-policy test (both on perf/rp4-pr-gather, PR_GATHER.md); the audio-row fixture; `_gathering`'s docstring; the `get_input_sequence_len` comment naming only `_spliced_row`; the docs' per-clip wait. Item 3 (step count) measured and not kept (PR_GATHER.md).

## Open

- GatherPolicy's numbers (8 rows, 30 ms, 16 busy requests) are constants in AudioEncoderSubmodule, not serving config (AGENTS.md invariants 2, 9).
- The gather policy trades TTFT p50 (+8-20% at A2T c=16-32) for throughput. Not measured: audio in with audio out (Talker requests count toward busy), several clips per prompt, TP, the 2-GPU layout (where `len(self.arrival)` also counts requests speaking on rank 0).
- Concept 2's own effect on main is not isolated, and on this branch it only remains on image/video rows, which no measured workload uses.
- Concept 4's gain is larger than its op count suggests; attributing it to GPU-thread host time per audio row is a hypothesis without a profile on main.
- From the dropped concept 1 (a3f3b6a4): an audio/vision row's text ids ride `text_token_ids`, not `input_ids`, so the sampler's `apply_cached_prefix` (mstar/engine/resources/sampler/resource.py:311 at main) keeps nothing for a walk the prefix cache serves whole; read from the code, not reproduced.
- The Thinker's literal agreement bar is missed by main and head alike; all of head's disagreements are bf16 ties.
- Absolute throughput is below the loop's numbers because CPU frequency boosting is off on both arms.
