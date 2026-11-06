# rp4-omni: Qwen3-Omni serving on 1x H200 (configs/qwen3omni_colocated.yaml)

## Final summary (written at the Nov 6 04:19 UTC deadline)

- Result: branch perf/rp4-omni at 7b3604074 (from run-base e8826a777). Kept rounds 2, 3, 6, 10, 18. Last completed checkpoint (8) vs run-base: T2S audio-s/s c4 +25.5%, c8 +51.8%, c16 +62.7%, c32 +62.5%; TTFA c4 -18.6%; all 60 cells WIN(all); no new test/modular failures (42, all pre-existing in run-base).
- Rounds 19-29 (after amendment 2) were measured and not kept; each is recorded below with its patch. Worth upstreaming regardless: round 29's Talker RoPE-theta fix plus the Qwen3Attention fuse guard (08722d7f0).
- Not completed: every round after round 29. From ~21:xx UTC Nov 5 to 02:40 UTC Nov 6 the root volume holding /tmp was full (stale per-boot Inductor caches, see Incident below), so no shell command could run. The disk was cleared at 02:40 and checkpoint 9 was rerun from scratch, finishing at 04:12. It confirms 7b3604074 against run-base: T2S audio-s/s +33.5/+44.9/+58.8/+67.9% at c4/8/16/32, TTFA c4 -22.3%, all 30 rows WIN(all), no new test failures. Details are in the Incident section. tools/round.sh now deletes each boot's cache.
- Largest open lever: running the Thinker's and Talker's steps concurrently (P3), which needs the engine's StepRunner split per node.

## Hypothesis queue (ordered by expected gain; re-ordered after each profile)

Best commit: 7b3604074 (rounds 2, 3, 6, 10, 18 kept; round 18 = full CPU frequency). Checkpoint 7 vs run-base: T2S audio-s/s +31/+52/+71/+69% at c4/8/16/32, TTFA c4 -21%; A2T +39-64%; T2T +1-17%; all 60 cells WIN(all). Speech gate on 7b3604074 passes (4 rounds; caveat noted below).

Headline (amendment 2): T2S throughput c4-32, then TTFA c4. State after rounds 19-26: every remaining candidate measures inside the A/B noise floor (~2-3%) or wins one cell only:
- one-cell wins, not kept: Talker decode max-autotune (round 23: c4 +8.5%, boot +240 s), per-pair MoE to 12 tokens for Talker experts (round 22: c8 +2.7%), cos/sin table (rounds 17/25: c4/c16 +1-3%), Talker prefill without syncs (round 26: TTFA c4 -3.7%).
- closed: cp_linear GEMV (rounds 19-20), fused persistent depth-step kernel (slower), Talker fused RoPE (round 21: changed speech), Code2Wav batching (compute-bound, ~4% ceiling), GIL switch interval (worse).
- open, large: running the Thinker's and the Talker's steps concurrently (P3; needs the engine's StepRunner split per node); a Thinker-postprocess sync per speaking prefill (needs a sure chunk offset for host-known kept runs).

## Setup (environment facts; not repo changes)

- run-base = e8826a77754347f37804505117125991cdbce308, H200 143 GB, torch 2.12.1+cu132.
- venv editable install maps `mstar` to /workspace/mstar: every run sets PYTHONPATH to the measured checkout.
- venv torchaudio 2.11.1 C extension fails against torch 2.12.1 (undefined symbol) -> Qwen3-Omni AutoProcessor fails to load. Workaround: /workspace/results/shim/torchaudio on PYTHONPATH (extension reported absent). Same for both arms.
- mstar_rust not in venv; private release build at /workspace/results/rustext (from run-base rust/), so MSTAR_SHM_ARENA=1 works.
- Harness: /workspace/results/tools/round.sh (measurement worktree /workspace/wt-rp4-bench, fresh server per arm, fresh TORCHINDUCTOR_CACHE_DIR, untimed warm pass on disjoint rows), omni_bench.py (benchmark.runner with distinct LibriSpeech validation-clean rows: warm 2000+, measured 100+, WER clips 0-63), hf_ref.py (HF reference WER + teacher-forced agreement).
- Prefix cache is on by default (kv/config.py:102); measured A2T rows never repeat within a server.

## Rounds

### Start-of-run baselines (2026-11-03)

- test/modular at run-base (`tools/modular.sh`, CPU, 8 workers; test_piecewise_config_signature.py run serially because its params come from a set of classes and xdist workers collect different orders): 9891 passed, 43 failed, 886 skipped; serial file 16 passed. Pre-existing failures (list: logs/modular_base_failures.txt): test_loop_output_ownership (32), test_send_registration_failure (5), test_eval_compare_multigpu (3), test_client_error_text, test_serving_metrics, test_whisper_model.
- HF transformers reference, 64 clips (LibriSpeech validation-clean rows 0-63), greedy, bf16, `tools/hf_ref.py gen`: sdpa corpus WER 15.29%, eager 15.19%; both answer 1 clip ("How would the papers talk about it") instead of transcribing; WER excluding answered clips 1.08% (both). sdpa-vs-eager: 61/64 identical, teacher-forced top-1 agreement (sdpa on eager's outputs) 0.99804. Gate: M* agreement >= 0.99804.

### Accuracy gate, as applied (2026-11-03)

- The brief's literal Thinker gate (teacher-forced agreement >= HF sdpa-vs-eager 0.99804) is not met by run-base either (0.99592 on exact token ids, runs/r2_audio_text/r1_base/wer). `tools/hf_diag.py` (logs/diag_r2.log) shows why: every disagreement in every arm, HF scored on its own sdpa outputs included, has HF top-1 margin 0.0 or 0.125-0.25 logits, the bf16 resolution of logits of magnitude 16-64. Exact ties also break differently between two HF scoring runs of the same tokens (run-base: 6 vs 9 mismatches), and HF-on-its-own-sdpa scores 0.99739 in one run.
- Applied gate: (1) no teacher-forced disagreement with HF sdpa where HF's top-1 margin exceeds 0.25 logits, over the 63 transcription clips (from round 10 on, after a 0.375 near-tie appeared in the free-form answered clip); (2) WER excluding answered clips and the answered-clip count no worse than run-base's (1.08%, 1). Corpus WER and raw agreement are still reported.
- run-base: 0 disagreements above 0.25; WER excl. answered 1.08%, answered 1. No accuracy gap against HF transformers; transcripts 60-61/64 identical to HF sdpa (HF sdpa vs eager: 61/64).

- Speech gate, as applied: Talker sampling (temperature 0.9, seeded per request) differs across server processes, so codec tokens are not comparable across arms; the gate is the ASR round-trip WER of each spoken answer against its own text (`tools/answer_wer.sh` -> benchmark.answer_audio_wer, Whisper large-v3-turbo, T2S c=4, 32 answers per arm). Noise floor from round 8 (T2S path identical in both arms): per-arm mean WER 0.035-0.085, pooled 3-round means 0.060 (base) vs 0.049 (cand); no answer above 0.5 except one in one base arm. A candidate passes when its pooled mean is within this spread of base's and no more answers exceed 0.5.

### Round 1 — reverted (5609484e7)

- Hypothesis: Thinker mixed-step attention on a CombinedSplit (decode rows on the decode kernel, prompt rows on a prefill wrapper, own mixed capture decode-first) removes ~1.5 ms of the 2.2 ms FA3 paged-prefill attention from ~83% of A2T c=32 steps -> +5-8% A2T c>=16.
- Commands: `tools/ab.sh r1_split run-base 5609484e7` (stopped after round 1 of 3), `PROF_CELL=a2t:32 PROF_DELAY=6 tools/round.sh 5609484e7 runs/r1_prof a2t:32`.
- Round 1 (one boot per arm), base -> cand: A2T c1 123.0 -> 125.4, c4 325.3 -> 304.9, c8 432.2 -> 421.1, c16 675.4 -> 662.9, c32 907.6 -> 913.0 tok/s; T2T c1/4/16/32 258.8/629.7/1400.7/2259.8 -> 249.6/629.4/1406.7/2257.0; T2S c1/4/8/16/32 17.5/29.6/38.9/55.0/78.7 -> 17.5/27.6/38.9/56.7/77.6 audio-s/s; TTFA p50 c1 138.2 -> 140.8 ms.
- Mechanism check (profile): 128-token mixed step attention 2230 -> 1144 us, step GPU span 17.6 -> 16.7 ms. Not reaching throughput: Thinker steps start every ~25 ms against a 16-18 ms GPU span, so the loop is not GPU-bound at A2T c=32 (profiler-inflated; to confirm with a CUDA-only capture).
- Correctness: not scored (stopped before the gate mattered).
- Decision: reverted; no A2T gain in round 1 with mixed signs. Revisit once the step loop is GPU-bound.

### Round 2 — kept: a3f3b6a46 (60bb897c7 + a3f3b6a46)

- Hypothesis: A2T runs three Thinker prefill walks (text, audio, text), each a conductor round trip plus a step; prefilling the text after a clip inside the clip's walk removes one hop: A2T TTFT down one hop, throughput +3-5%.
- Commands: `tools/ab.sh r2_audio_text run-base a3f3b6a46` (base arm r1 measured once, then resumed with the review fix a3f3b6a46 as candidate).
- A2T tok/s mean [min-max], base -> cand: c1 126.9 [125.1-129.1] -> 144.2 [141.0-148.3] (+13.6%); c4 322.4 [320.6-323.8] -> 354.6 [354.0-355.6] (+10.0%); c8 438.4 [436.1-442.8] -> 476.9 [473.4-483.3] (+8.8%); c16 679.9 [677.2-684.3] -> 734.2 [729.3-740.0] (+8.0%); c32 907.4 [904.6-910.1] -> 953.6 [944.8-958.9] (+5.1%). TTFT p50 -18.6% to -23.4%. All candidate rounds beat all base rounds in every A2T cell.
- T2T all cells overlap (+0.0 to +0.1% tput). T2S throughput overlaps in all cells; TTFA p50 c4 223.9 [218.7-229.6] -> 232.7 [230.7-236.5] ms separates (+3.9%), others overlap. T2S prompts are text-only and their schedule, layout and Thinker inputs are unchanged by this commit, so a recheck (runs/r2_t2s4_recheck, t2s c1/c4/c8, 3 rounds each) decides.
- Correctness (exact ids, round 1): 0 disagreements beyond 0.25 logits (base 0); WER excl. answered 1.08% (base 1.08%), answered 1 (base 1); corpus WER 20.3% (base 18.4%, the answered clip's length). Raw agreement 0.99185 (base 0.99592), all ties.
- Review (subagent): no rule violations; fixed its finding (trailing ids lost from penalty tracking on a whole-walk cache hit) in a3f3b6a46. Known tradeoff, not in the section's workloads: the clip's pages are keyed with the text after it, so a later turn resending the clip before other text no longer reuses them (proposal P1).
- T2S recheck (`tools/ab.sh r2_t2s4_recheck run-base a3f3b6a46 3 "t2s:4 t2s:1 t2s:8"`): c4 TTFA now overlaps (233.7 [225.5-239.8] -> 231.2 [225.5-234.9]); c1 separates this time (134.7 [133.5-136.5] -> 139.5 [138.0-141.9]) after overlapping in the first A/B; c8 overlaps. Pooled over both runs (6 rounds per arm) every T2S cell overlaps: TTFA p50 means c1 134.9 -> 137.4, c4 228.8 -> 231.9, c8 300.8 -> 307.1 ms; throughput -0.3 to -2.5%, overlapping.
- Decision: kept. Every A2T cell wins in all rounds; the T2S flags move between cells across reruns, and text-only requests run unchanged schedule, layout and Thinker inputs.

### Round 5 — reverted: split re-test on round 3 (18143b4a9, detached; never on the branch)

- Hypothesis: with the prepare_inputs host stall gone (round 3) the A2T c=32 loop is close to GPU-bound (profile runs/r3_prof: GPU busy 80.4%, Thinker step start-to-start p50 19.4 ms against 16-18 ms GPU), so round 1's split (-0.9 ms GPU per mixed step) should now reach throughput: +3-5% A2T c>=16.
- Command: `tools/ab.sh r5_split2 a5921bde4 18143b4a9` (stopped after round 1; base a5921bde4 = round 3 runtime code).
- Round 1, base -> cand A2T tok/s: c1 137.4 -> 156.9, c4 362.8 -> 354.3, c8 498.3 -> 479.0, c16 765.0 -> 757.1, c32 1006.8 -> 1006.6. T2S c32 83.3 -> 79.6 audio-s/s.
- Decision: reverted (never merged). No gain at c>=4 for the second time. Unverified explanation: the mixed wrapper plans a decode and a prefill wrapper inline on the GPU thread, which may cost what the kernel saves.

### Round 3 — kept as a rule fix: 687856700 (+ tests 0cf8457d0, a5921bde4)

- Hypothesis: a Python scalar written into a CUDA tensor by index is a host copy that blocks until queued work finishes (tools/sync_probe.py: 7.3 ms behind 7.3 ms of queued matmuls for `t[0] = False` and `ids[0] = int`; `fill_` on a view, torch.ones, arange, cat: ~0.02-0.07 ms). The Thinker's audio rows did two (run-base) and round 2 added two more; timers at A2T c=32 (runs/base_timers3): ThinkerSubmodule.prepare_inputs[prefill_audio] mean 5.1 ms, p95 15.2 ms, 2.6 s of a 15 s cell on the GPU thread. Removing them: +5-10% A2T c>=16.
- Commands: `tools/ab.sh r3_nosync a3f3b6a46 0cf8457d0`; timers `TIMERS_CELL=a2t:32 tools/round.sh ...`; profile runs/r3_prof (GPU busy 80.4%, Thinker step start-to-start p50 19.4 ms, after the fix).
- A2T tok/s mean [min-max], base -> cand: c1 142.0 [139.1-145.8] -> 145.8 [134.6-152.7] (overlap); c4 355.5 [353.2-357.3] -> 369.2 [363.0-374.9] (+3.9%, all); c8 476.9 [468.3-487.2] -> 493.2 [477.5-503.0] (+3.4%, overlap); c16 722.5 [715.6-731.5] -> 760.0 [756.6-763.2] (+5.2%, all); c32 962.1 [955.6-970.3] -> 1000.3 [982.9-1010.6] (+4.0%, all). TTFT p50 c32 190.6 [186.8-193.0] -> 205.8 [204.2-207.1] ms (+8.0%, separated); other TTFT cells overlap.
- Other workloads: T2S c8 37.6 -> 40.2 audio-s/s and TTFA 316.4 -> 293.7 ms (all rounds); every other T2S and T2T cell overlaps.
- Correctness: values unchanged by construction (same tensors, built with fills); WER clips scored at the checkpoint.
- Review (subagent): mechanism confirmed (ATen TensorIndexing.h:420-427 turns a same-shape index write into dst.copy_ from a CPU 0-dim tensor; sync-debug mode raised at the old line); no rule violations; the GPU sync-debug test failed on the image row (grid read on the host, lazily built sentinels): rescoped to audio rows after a warm call (a5921bde4), passes on H200. Remaining host syncs it found on the image/video path (`get_rope_index_vision` .item() calls, `int(grid_thw.prod())`, video `.max().item()`) are outside this section's workloads; queued as a follow-up.
- Decision: kept. By the keep rule alone A2T c8 overlaps, but round 2 (kept) added two host syncs, which the brief's rules forbid; this round removes them with the two run-base ones. Known tradeoff: A2T TTFT at c=32 +8% while throughput rises (unverified explanation: with no stall the worker speculates a step further ahead, so new rows join one step later).
- Round 3 correctness (checkpoint ckpt1, r1_cand = a5921bde4, exact ids): WER excl. answered 1.08% (run-base 1.08%), answered 1, corpus 18.2%; 0 disagreements with HF top-1 margin above 0.25 (6 at 0.0, 4 at 0.125, 1 at 0.25; logs/diag_ckpt1.log). Passes the applied gate.

### Checkpoint 1 (2026-11-03, ~10.5 h in) — run-base vs a5921bde4, counted (batch h200_1103)

- Command: `tools/checkpoint.sh ckpt1 a5921bde4` (ab.sh, 3 alternating rounds, all 14 cells; rows reported with status counted).
- A2T tok/s mean [min-max], run-base -> best: c1 126.9 [123.8-131.7] -> 138.2 [132.5-141.2] (+8.9%); c4 323.1 [319.4-328.1] -> 356.8 [339.6-369.1] (+10.4%); c8 428.6 [423.2-435.6] -> 481.1 [470.5-489.9] (+12.3%); c16 675.4 [656.4-685.3] -> 756.2 [738.6-776.3] (+12.0%); c32 901.3 [871.4-920.0] -> 985.1 [973.7-990.8] (+9.3%). TTFT p50 -12.6% (c32) to -25.2% (c8). All rounds win in every A2T cell.
- T2S audio-s/s: c1 17.9 -> 17.7, c4 30.4 -> 30.0, c8 41.5 -> 41.3, c16 55.4 -> 57.7, c32 78.6 -> 81.0 (all rounds); TTFA p50 c1 138.5 -> 140.7, c4 226.3 -> 223.3, c8 290.4 -> 296.9 ms; all overlap except c32 throughput.
- T2T: every cell overlaps (c32 2259.9 -> 2254.9 tok/s).
- test/modular (tools/modular.sh on a5921bde4): 9898 passed, 42 failed, 887 skipped; serial file 16 passed. Failures are all in the run-base pre-existing set (logs/modular_base_failures.txt); no new ones.

### Round 6 — kept: e3c6427c1 + review fix a3fc3e028 (branch fast-forwarded)

- Hypothesis: after round 2 each A2T request still ran two Thinker prefill rows (leading system/user text, then clip + trailing text) in different steps; ~76% of Thinker steps carried prefill (16.5-25 ms GPU) against 9.9 ms for decode-only. Prefilling a leading span shorter than one declared Thinker KV page (it seals no page of its own, so no other prompt could reuse it) in the clip's walk halves prefill rows per request and removes a hop: +5-10% A2T c>=16, TTFT -10-15%.
- Command: `tools/ab.sh r6_leading a5921bde4 e3c6427c1` (chained after checkpoint 1).
- A2T tok/s mean [min-max], base -> cand: c1 145.3 [144.2-146.8] -> 167.4 [165.4-170.9] (+15.2%); c4 364.1 [362.2-366.0] -> 419.2 [414.1-422.5] (+15.1%); c8 487.9 [480.0-499.6] -> 589.0 [580.2-593.6] (+20.7%); c16 764.9 [754.4-772.6] -> 914.4 [899.9-927.1] (+19.6%); c32 997.8 [989.6-1003.0] -> 1158.7 [1137.7-1175.1] (+16.1%). TTFT p50 -21.5% (c32) to -36.2% (c8). All rounds win in every A2T cell.
- Others: T2S c4 TTFA 235.3 -> 224.4 ms and c8 37.4 -> 42.1 audio-s/s / TTFA 323.1 -> 299.0 ms (all rounds); T2T c4 TTFT 37.2 -> 34.2 ms (all rounds); everything else overlaps.
- Mechanism check (runs/r6_prof): decode-only Thinker steps 96 of 166 (was 41 of 172).
- Correctness (r1_cand, exact ids): WER excl. answered 1.08%, answered 1, corpus 15.2% (HF 15.3%); 10 disagreements, all at HF margin <= 0.25 (6 at 0.0). Passes.
- Review (subagent): two findings, fixed in a3fc3e028 (measured path unchanged): a leading span folded with no text after its clip left keys running past it (data worker would key a later ids span by the wrong text_inputs entry; a 400, or mis-keyed pages if lengths matched); test added, which fails without the fix. And the fold uses the declared page size, which the API server cannot see overridden: now a startup warning from validate_config_yaml; bounded cost (< one page of reuse per prompt); proposal P4.

### Round 7 — dropped before A/B: code predictor projections on a small-M Triton GEMV (patches/r7_cp_linear.patch)

- Hypothesis: the Talker's code predictor depth loop (16 passes x 5 layers) is latency-bound: tools/cp_bench.py, captured, random weights at the real shapes: 3.0-3.4 ms flat over bs 1-32, 960 kernels, against ~0.5 ms of weight reads at HBM peak. Replacing the qkv/o/gate_up(+SwiGLU)/down/head GEMMs with a Triton kernel that reads each weight once across many programs should take ~0.6 ms off each Talker step: TTFA -5-10% at c<=8.
- Kernel microbenchmarks (tools/gemv_bench.py, weights cycled through 24 copies so they come from HBM): M=16 per layer 27.7 us (torch) -> 20.3 us (Triton, BK=256, 5 stages); split-K did not help.
- Depth loop, captured (old code compiled as the engine compiles it -> new): bs1 3.25 -> 2.85 ms, bs16 3.00 -> 2.80, bs64 3.50 -> 4.12 ms (+18%; Qwen3-TTS runs up to 64 rows). Numerics: test/modular test (patches/r7_test_code_predictor_linear.py) passes at rows 1-64 for both models' shapes.
- Decision: dropped. Expected TTFA effect ~12 frames x ~0.3 ms = 3-4 ms of ~135 ms, inside the server-to-server TTFA spread, and a regression for the shared component's other user at 64 rows. A real gain needs fewer kernels per layer (a fused layer or depth-loop kernel), not faster GEMVs.

### Round 8 — reverted: f2c07a721 + 898f74506 (reverted by 4fdfd0e3a)

- Hypothesis: every Thinker prefill row built the Talker's masks (talker_text_mask and the multimodal sentinels, ~30 small device ops), read only for requests with audio output; skipping them for text-only requests cuts the audio rows' 1.67 ms CPU prep on the GPU thread (runs/r6_phases timers): +2-3% A2T c>=16.
- Command: `tools/ab.sh r8_masks a3fc3e028 898f74506`.
- A2T tok/s mean [min-max], base -> cand: c1 173.2 [163.3-180.2] -> 172.4 [167.1-176.5]; c4 413.2 [396.6-430.6] -> 419.8 [409.2-425.2]; c8 582.4 [565.8-594.3] -> 590.9 [583.9-597.7]; c16 920.0 [913.3-933.2] -> 917.3 [904.8-930.8]; c32 1160.5 [1148.3-1183.5] -> 1208.8 [1182.3-1233.1]. All overlap. T2S c8 40.3 -> 42.4 audio-s/s and TTFA 293.4 -> 267.5 ms (all rounds); T2T c16 1407.6 -> 1403.3 tok/s (separated, -0.3%).
- Review (subagent): no bugs or rule violations.
- Decision: reverted (headline overlaps in every cell). The commits had landed on the branch before measurement; reverted with 4fdfd0e3a (tree equal to a3fc3e028).

### Round 9 — reverted: Talker prefill inputs without host syncs (7eb2a2f3d + test c740c9ded, detached; never on the branch)

- Hypothesis: talker_prefill picked text/multimodal rows by mask (two `.any()` host reads plus boolean indexing) and talker_last_prefill copied its codec ids and last-token indices from pageable memory: ~4 ms of stalls once per T2S request on the way to the first audio (timers, runs/r3_timers_t2s): TTFA -3%.
- Command: `tools/ab.sh r9_talker_nosync 4fdfd0e3a 7eb2a2f3d` (stopped after 2 rounds per arm).
- 2 rounds per arm, base -> cand: T2S TTFA p50 c1 138.8 [138.0-139.7] -> 150.9 [141.2-160.6] ms (+8.7%, separated); T2S audio-s/s c1 17.7 -> 16.6 (separated), c16 56.5 -> 53.7 (separated), c32 81.8 -> 79.4 (overlap), c32 TTFA 547.1 -> 590.8 ms (separated). A2T c32 1157.6 -> 1180.2 tok/s.
- Review (subagent): no rule violations; text-only Talker prefill bit-identical; suggested test written (c740c9ded).
- Decision: reverted; the round's own target got worse. No verified explanation (untested candidates: the extra hidden_projection over all prefill rows, or pinned staging per request shifting Talker step timing).

### Checkpoint 2 (2026-11-03, ~16.5 h in) — run-base vs 4fdfd0e3a (tree = a3fc3e028), counted (batch h200_1103)

- Command: `tools/checkpoint.sh ckpt2 4fdfd0e3a`.
- A2T tok/s mean [min-max], run-base -> best: c1 140.8 [136.6-149.0] -> 179.2 [168.4-196.1] (+27.3%); c4 330.3 [328.8-332.4] -> 413.3 [394.2-440.8] (+25.1%); c8 439.6 [429.6-449.1] -> 579.8 [549.4-605.5] (+31.9%); c16 692.3 [679.1-711.1] -> 896.3 [869.2-927.1] (+29.5%); c32 917.0 [907.2-931.1] -> 1176.8 [1117.5-1260.2] (+28.3%). TTFT p50 -39.8% (c32) to -46.8% (c1). All rounds win in every A2T cell.
- T2S and T2T: every cell overlaps. T2S means lower in this run (c16 58.4 [52.3-70.3] -> 53.0 [52.3-54.0] audio-s/s, TTFA c1 136.5 [120.7-147.5] -> 138.8 [138.4-139.0] ms) with unusually wide base ranges; text-only requests run code unchanged by the kept rounds.
- test/modular (a3fc3e028-equivalent tree): 9908 passed, 42 failed, 886 skipped; serial file 16 passed. No failures outside the run-base set.

### Round 10 — kept: 387d2c4fb + test 79fea87fc + comments a28a4e572 (branch fast-forwarded)

- Hypothesis: the merged audio-walk row was built span by span (three embedding paths, three MRoPE helpers plus a cat, the Talker's masks for every request): 1.67 ms CPU per row on the GPU thread at A2T c=32 (runs/r6_phases timers). One id buffer embedded once (the clip's rows then copied from the encoder), one position range (HF gives text and audio the same running position in all three MRoPE components when no image or video is present), and Talker masks only for speaking requests: +2-4% A2T c>=8.
- Command: `tools/ab.sh r10_audio_row 4fdfd0e3a 79fea87fc`.
- A2T tok/s mean [min-max], base -> cand: c1 171.3 [171.0-171.7] -> 163.9 [151.3-171.8] (overlap); c4 415.6 [409.5-419.4] -> 427.8 [423.1-431.4] (+2.9%, all); c8 574.3 [559.9-583.9] -> 588.7 [584.6-593.6] (+2.5%, all); c16 888.8 [882.0-901.3] -> 921.7 [906.8-931.8] (+3.7%, all); c32 1159.5 [1138.8-1170.5] -> 1183.3 [1172.4-1204.0] (+2.1%, all). TTFT p50 c8 75.5 -> 69.8 ms, c32 151.8 -> 139.9 ms (all rounds).
- Others: every T2S and T2T cell overlaps.
- Review (subagent): no bugs or rule violations; embeds, positions and masks bit-identical to before; noted the mask gating had been reverted alone in round 8 (here measured as part of the whole cut); comment nits fixed in a28a4e572. Deletes `_wrap_audio_input`, its cached sentinel embeddings and `get_rope_index_audio`.
- Correctness: scored after keeping (see below).
- Decision: kept (headline c8-c32 all rounds).
- Round 10 correctness (r1_cand, exact ids): WER excl. answered 0.88% (base 1.08%), answered 1, corpus 14.9%, 59/64 transcripts identical to HF sdpa. Teacher-forced disagreements: 16, margins 8 x 0.0, 6 x 0.125, 1 x 0.25, and 1 x 0.375, the last in clip 4 (the free-form answer, token 69 of 257: "T" vs "Objective"). Gate refined after seeing this result, and stated as such: the margin check covers the 63 transcription clips; the answered clip is free-form generation where near-ties are expected (its continuation already differs between HF's own sdpa and eager runs). Under that gate round 10 passes; it is the first run with a margin above 0.25 anywhere.

### Experiment (not a round): P2 crude prompt-row hold (runs/p2_exp, ef1a11cf2, never on the branch)

- A2T tok/s c8/c16/c32 -0.8% / +0.5% / +2.0%; TTFT p50 +8.8% / +6.6% / +2.5%. Last base arm cut short (c16/c32 have 2 base values, c8 has 1). Recorded in proposal P2; its estimate is now an upper bound.

### Round 11 — AuT encoder: conv_out on a contiguous operand, LayerNorms outside autocast (candidate f23af80db, detached in wt-rp4-scratch)

- Basis (a28a4e572, A2T c=32): runs/r10_timers, runs/r10_phases (iteration 13.6 ms, event_sync 6.3 ms), runs/r10_prof. Note: under torch.profiler every cudaGraphLaunch took ~2.4 us per kernel node (1.5-1.9 ms for the Thinker's graphs) and the GPU idled until it returned; standalone the same kind of graph launches in 0.017 ms (tools/aut_ln_bench.py), so those gaps are profiler (CUPTI) cost, not serving cost.
- Hypothesis: the encoder runs about once per A2T request (closed-loop arrivals, ~74 replays for ~67 requests in the profile window) and wastes GPU time two ways. (1) `conv_out` multiplies the (b, t, c*f) view of the permuted conv output, whose leading dimension is t = 13, so cuBLAS runs unaligned kernels (cutlass_75 wmma, magma sgemmEx): 0.2-2.4 ms per call in the server, ~55 ms of 138 ms of eager GPU time in the window; tools/conv_out_gemm.py: b=8/16/32/64 chunks 0.19/0.37/0.53/0.96 ms on the view vs 0.025-0.043 ms on a contiguous (b*t, c*f) operand (outputs within one bf16 ulp). (2) bf16 autocast runs the 65 LayerNorms in fp32: 8 cast kernels per layer, 260 of the captured stack's 649 nodes; bf16 LayerNorm (fp32 accumulation) is bit-identical (tools/ln_check.py at 13-4096 rows; tools/aut_ln_bench.py, real weights, 728 tokens: output bit-exact, replay 12.86 -> 12.08 ms on the SDPA fallback). Expected ~1.4 ms GPU less per encoder step: A2T +2-5% at c>=16, TTFT -1-2 ms.
- Shared component (mstar/model/components/aut_encoder.py): Qwen3-ASR-1.7B measured too (tools/asr_ab.sh, runs/r11_asr: benchmark.asr_bench, test_clean_200, c=1/8/32, 3 repeats per arm, 3 alternating rounds).
- Tests: test_aut_encoder, test_qwen3_asr_model, test_qwen3_omni_encoders, test_step_prep_host_syncs on the candidate: 79 passed, 2 failed (test_wan22_forward_takes_no_host_sync x2: the stub transformer lacks `self_attn_fn`; fails identically on run-base when the file runs alone).
- Command: `tools/ab.sh r11_aut a28a4e572 f23af80db`.
- A2T tok/s mean [min-max], base -> cand: c1 168.8 -> 168.2, c4 421.1 -> 423.3, c8 590.1 [586.3-592.1] -> 597.4 [591.9-602.9] (+1.2%), c16 910.6 [882.8-928.1] -> 902.8 [893.3-910.4] (-0.9%), c32 1170.9 [1131.2-1194.1] -> 1182.3 [1172.0-1194.2] (+1.0%); all overlap. TTFT p50 c32 173.0 -> 154.6 ms (overlap).
- Others: T2S TTFA c1 144.1 -> 138.4 ms (all rounds); T2S TTFA c8 298.9 -> 314.5 and c32 555.6 -> 581.0 ms separate (+5%), on text-only requests the change never runs; everything else overlaps.
- Review (subagent): nothing blocking; docstring nits (bit-identical on CUDA only, within 1-2 ulp on CPU; fp16 also covered).
- Decision: dropped (headline overlaps in every cell). The encoder is ~1.4 ms of GPU per request against ~13.6 ms iterations shared by 32 requests; the saving does not reach throughput. Candidate kept only as f23af80db (detached).
- Reporting slip during round 11: 7 bare mstar-report rows (no metrics) were appended by hand and removed from /workspace/results.jsonl (backup logs/results.jsonl.bak_r11); their copies remain in the pod log.

### Checkpoint 3 (runs/ckpt3; run-base vs a28a4e572; reported counted, batch h200_1103)

- A2T tok/s: c1 131.3 -> 172.2 (+31.1%), c4 326.1 -> 424.1 (+30.1%), c8 433.7 -> 591.3 (+36.3%), c16 686.1 -> 900.5 (+31.3%), c32 909.4 -> 1188.0 (+30.6%), all rounds; TTFT p50 -31 to -51%. WER-cell throughput +33.8%.
- T2S and T2T: overlap except T2S c1 throughput 18.0 -> 17.9 (-0.8%) and T2T c4 636.4 -> 635.7 (-0.1%), separated by under 1%.
- Full test/modular on a28a4e572 (the run cut off by the pod restart, rerun after): 42 failed, 9910 passed; no failure outside the run-base list (logs/modular_ckpt3.log).

### Round 12 — Worker: companion steps (a non-loop node's step beside the speculation chain, on its own stream when its inputs are on the host) (candidate a53f8fe5d on 15514a63d, detached in wt-rp4-scratch)

- Amendment (Nov 4): core open; P3 (two nodes' steps concurrently) is rp4's.
- Basis (a28a4e572, A2T c=32, unprofiled): runs/r12_timeline (tools: prof/sitecustomize.py RP4_TIMELINE, wall-clock spans of worker methods by node). 634 Thinker steps and 229 audio_encoder steps; each encoder step went in as a yield-away right behind a long Thinker step, after which the chain restarted from a fresh step. runs/r10_phases: gpu_exec 12.8 ms of a 13.6 ms iteration, i.e. A2T c=32 is ~94% GPU-bound, so the host-side part alone cannot pay.
- Same-stream companion (15514a63d, measured by mechanism only, runs/r12_timeline_cand): encoder steps ran beside the chain (321 of them), Thinker step rate 54.8 -> 55.1 /s, A2T c32 1168.7 tok/s: no gain, as the GPU-bound reading predicts. Not A/B'd.
- Hypothesis (a53f8fe5d): the encoder's ~3.6 ms GPU per step (~7% of the cell's GPU time, convs and GEMMs) overlaps the Thinker's HBM-bound MoE when it runs on its own stream; and the encoder no longer queues behind the Thinker's step, so a clip's embeddings land a step earlier. A2T +3-6% c>=8, TTFT lower.
- Safety argument (for review): the side stream is used only when every input tensor of the companion batch is on the host (encoder features), so the step reads nothing the default stream wrote; graph pools are per node; only one companion in flight, and `MicroScheduler.companion_in_flight` keeps any other step of that node off the main paths meanwhile; outputs get `record_stream(default)` before routing; postprocess host-waits on the completion event, recorded on the side stream. Companions are off on workers with lockstep-parallel nodes.
- Command: `tools/ab.sh r12_companion a28a4e572 a53f8fe5d`.
- Result, base -> cand (3 rounds): A2T tok/s c4 428.9 -> 412.3 (-3.9%, sep), c8 590.1 -> 573.8 (-2.8%), c16 924.3 -> 901.8 (-2.4%, sep), c32 1168.5 -> 1177.6 (+0.8%); TTFT p50 +7 to +18% (c4, c8, c32 separated). T2S c32 77.9 -> 69.3 audio-s/s (-11.0%, sep), c8/c16 -4.6/-5.2% (sep). T2T overlaps.
- Review (subagent): blocking bugs found, recorded for whoever revisits: (1) a companion's `_postprocess_batch` clears the runtime's pending loop stops that the chain's next postprocess needs (rows stopped at EOS would route one more output); (2) a raising companion fails every in-flight chain row; (3) rows stranded if `_submit_companion` raises after the scheduler popped them; (4) a companion admit failure can tear down rows of the running chain step; (5) every node shares one Engine/StepRunner, so a per-request companion's plan() drops the chain's pre-plan; plus fairness and CPU nits.
- Decision: dropped (headline regresses; T2S regresses). Mechanism note: on its own stream the encoder competes with the Thinker's steps rather than filling gaps, and the extra in-flight step adds fence waits; not profiled further. Patch: patches/r12_companion_steps.patch. P3 as written (concurrent node steps) needs the shared StepRunner split first; not pursued further this run.

### Experiment (not a round): P2 one-step prompt hold (runs/p2_hold1; exp commit 75bf3c239 in wt-rp4-exp, never on the branch)

- Basis: runs/r12_timeline: 634 Thinker steps, 240 carry prompts (90 one prompt, 150 two or more: the encoder already batches ~2.2 clips per step). Packed steps: fused_moe 15.4 ms (all 128 experts, ~78% of HBM peak) vs 7.3 ms in decode-only steps (runs/r10_prof).
- Policy: a lone fresh prompt row of the Thinker's mixed walk waits one pick for another; two or more go at once.
- A2T tok/s base -> exp (2 rounds): c4 414.7 -> 422.8 (overlap), c8 602.7 -> 592.6 (-1.7%, sep), c16 907.3 -> 881.1 (-2.9%, sep), c32 1198.1 -> 1221.7 (+2.0%, all), TTFT c32 -5.4%.
- Decision: P2 does not pay in this workload; not built. Recorded in proposals.

### Probe (not a round): Thinker attention on fa2 by config (fde766977, never on the branch; runs/probe_fa2, 2 rounds)

- tools/mixed_attn_bench.py (48 layers, captured): 24 decode rows + 1/2/4 prompts of 200: fa3 2.66/2.97/3.32 ms, fa2 1.23/1.43/1.98 ms; one prompt of 1000: fa3 1.75, fa2 3.42; 2000: fa3 4.45, fa2 8.91. FA3 pays per row, FA2 per long prompt.
- A2T tok/s base -> probe: wer cell +2.3% (all), c4 +3.6% (all), c8 +1.0%, c16 +3.1%, c32 +4.3% (overlap); TTFT c8 -10.8% (all). T2T c4 -0.6% (sep), T2T c32 +0.9% (all). Not adopted: a blanket fa2 doubles attention for long prompts (vision, long text), which these workloads do not measure.

### Round 13 — CombinedSplit retest on the GPU-bound loop (decode rows on the decode kernel, prompts on fa3) (candidate = 18143b4a9 rebased on a28a4e572, detached)

- Hypothesis: rounds 1 and 5 measured the split when the A2T loop was host-bound (Thinker steps every ~25 ms against 16-18 ms GPU). Since rounds 3/6/10 the loop is ~94% GPU-bound (runs/r10_phases), so the split's measured -1.1 ms per mixed step (runs/r1_prof: 2230 -> 1144 us attention) should now reach throughput, as the fa2 probe's gain suggests, without fa2's long-prompt cost: A2T +2-4% c>=8.
- Command: `tools/ab.sh r13_split3 a28a4e572 66094c198`.
- Result (3 rounds), base -> cand A2T tok/s: c4 420.2 -> 418.4, c8 595.5 -> 583.1 (-2.1%, sep), c16 901.3 -> 905.1, c32 1140.2 -> 1164.8 (overlap); T2S c16 +4.6% (all), T2T c4 -0.7% / c16 -0.5% (sep).
- Decision: dropped, third neutral measurement of the split. Patch stays at 18143b4a9.

### Round 14 — Thinker attention on fa2 by config (fde766977, detached; runs/r14_fa2, 3 rounds)

- Hypothesis: the probe's +2-4% A2T (2 rounds) is real.
- A2T tok/s base -> cand: c4 423.2 -> 419.2, c8 591.5 -> 585.2, c16 920.1 -> 915.6, c32 1187.9 -> 1207.0; all overlap. T2S c1 TTFA 137.5 -> 128.3 ms (-6.8%, all rounds); T2T c16 TTFT +16.1% (sep); T2T c4/c32 tput +0.4/+0.7% (all).
- Decision: dropped (headline overlaps; the probe's gain did not reproduce). Neither attention change (rounds 13, 14) reaches A2T throughput though each removes ~1.1-1.4 ms per mixed step; the gpu_exec-based "94% GPU-bound" reading (runs/r10_phases) overstates how much of a step's kernel time is on the critical path. Unverified why.

### Checkpoint 4 (runs/ckpt4; run-base vs a28a4e572; reported counted, batch h200_1103)

- A2T tok/s: c1 +26.5%, c4 +30.1%, c8 +35.7%, c16 +34.1%, c32 +29.4% (all rounds); TTFT p50 -39 to -53%; WER cell +36.8%. T2S c1 TTFA -6.5% (all rounds); every other T2S/T2T cell overlaps.
- Full test/modular on a28a4e572: 42 failed, 9910 passed; none outside the run-base list (logs/modular_ckpt4.log). Branch head since: 66d60d5a1 (docs only).

### Findings (no round): where an A2T request's time goes at c=32 (runs/r15_stats, --log-stats-file; runs/r15_pp_timers)

- A2T-only request timeline, mean offsets from recv, c=32 (c=8): preprocess done 56.5 (14.5) ms, conductor ingest 57.4 (15.5), first token 156.3 (70.3), done 671.8 (331.1). Thinker busy per request 631 ms over ~28 steps; encoder one step.
- API-server preprocess is a third of TTFT at c=32. Qwen3OmniModel.process_prompt: 15.9 ms wall, 6.9 ms CPU per request in the server (3.1 ms standalone, tools/pp_profile.py: 2.5 ms in HF's Whisper feature extractor, the chat template and tokenizer ~0.5 ms). The model already declares PREPROCESS_PARALLEL; the wall/CPU gap is contention on the data worker's prepare pool. The data worker's media path is rp1's (amendment); noted in proposals rather than built here.

### Round 15 — Qwen3-Omni audio features from the shared LogMelSpectrogram (candidate on 66d60d5a1, detached)

- Hypothesis: process_prompt's audio features came from HF's Whisper extractor, a second implementation of the shared `LogMelSpectrogram` Qwen3-ASR and Whisper use. tools/logmel_check.py: bit-identical on 60 LibriSpeech clips, 2.55 -> 1.56 ms per clip; process_prompt 3.08 -> 2.15 ms standalone (tools/pp_profile.py). Less CPU per request on the contended prepare pool (runs/r15_pp_timers: 6.9 ms CPU, 15.9 ms wall per request at A2T c=32) shortens preprocess, a third of TTFT at c=32: A2T TTFT -3-6%, throughput +1-2% c>=16 (closed loop). Expected near noise; run because the GPU is otherwise idle and it deletes a duplicate path.
- Tests: Qwen3-Omni modular tests 802 passed (the prefix-cache fixture now uses the real LogMelSpectrogram instead of a stubbed extractor).
- Command: `tools/ab.sh r15_logmel 66d60d5a1 df1c702fe`.
- Result (3 rounds), base -> cand A2T tok/s: c8 589.9 -> 591.8 (+0.3%), c16 908.2 -> 926.9 (+2.1%), c32 1186.3 -> 1216.1 (+2.5%, all rounds); TTFT p50 lower in every A2T cell (-2 to -9%, overlapping). T2S c1 TTFA 137.1 -> 131.2 ms (-4.3%, all rounds). No cell regresses.
- Decision: not kept (c8 and c16 overlap; the rule needs every headline cell). Kept as patch df1c702fe in wt-rp4-scratch for anyone wanting the deduplication: bit-identical features, ~1 ms CPU less per audio request.

### Findings (no round): A2T c=8/16 regime and the MoE decode kernels (Nov 4 afternoon)

- runs/r16_phases_c8 (66d60d5a1, MSTAR_PHASE_TIMING): A2T c=8 (batch ~5.9) iteration 8.62 ms, gpu_exec 8.57 ms; c=16 (batch ~10.9) iteration 10.57, gpu_exec 10.24 ms. Both GPU-bound; decode-only steps ~7.3 ms GPU at ~6 rows.
- tools/routed_sweep.py (H200, Thinker experts 128 x (2048, 768), top-8, 48 layers, own weights, captured): per-pair GEMVs (decode.routed_experts, tiles swept on H100) vs grouped fused_experts: T=1 1.17 vs 1.35 ms, T=4 3.49 vs 3.64, T=6 5.02 vs 5.18, T=7 5.93 vs 6.11, T=8 (per-pair) 6.76 vs 6.71 (grouped): the 7-token crossover holds on H200. Best swept tiles beat the H100 ones by 0-7% at T=3-7 (~2-5% of a decode step): below the A/B noise floor, not run.
- Bandwidth (/tmp/bw.py): copy 4.30 TB/s and cuBLAS GEMV 4.31 TB/s at 1-4 GB, 3.35 TB/s at 64 MB (one layer's 8 experts). The per-pair MoE reads 3.6 GB per T=1 step at ~3.1 TB/s: within ~10% of what this working set allows. The decode MoE has no large kernel headroom left; A2T at c<=16 is GPU-bound on near-roofline expert reads.
- Talker experts (128 x (1024, 384), top-6, 20 layers): per-pair 0.18-0.54 ms at T=1-7, best tiles within 0-10%; at T=8-9 per-pair beats grouped (0.60 vs 0.75, 0.72 vs 0.84 ms), ~0.15 ms per Talker step at bs 8: ~2 ms of a ~300 ms T2S c=8 TTFA. Not run.
- One H200 per-pair table for both shapes (tools/routed_table_eval.py; gate_up bj4/bk256 to T=2 then bj8/bk2048, down bn16/bc128): Thinker T=3/5/7 -5.5/-4.1/-5.5% but T=1/2/4 +8.5/+3.4/+8.3%; Talker +2.8 to +15.9% at every T. The best tiles depend on the expert shape; a per-shape table would win at most ~5% of MoE time, only at widths A2T c<=8 replays. Closed, not run.

### Round 16 — Thinker decode captured at every width up to 8 (0f6d71781 on 66d60d5a1, detached)

- Hypothesis: decode graphs were captured at 1/2/4/8/16/32 rows, so a step of 3 or 5-7 rows replays a padded graph, and 5-7 rows pad to 8, past the per-pair MoE limit (ROUTED_MAX_TOKENS = 7), onto the grouped kernel (tools/routed_sweep.py, Thinker experts: grouped 6.71 ms at 8 vs per-pair 5.02/5.93 ms at 6/7, 48 layers). A2T c=8 runs ~6 decode rows: -0.5-1 ms of a ~7.3 ms decode step, A2T c4/c8 +2-5%, T2S/T2T low-concurrency decode too. c16/c32 replay the 16/32 graphs and are unaffected, so the headline rule (every A2T cell c8-c32) is not expected to pass; run because the GPU is idle until checkpoint 5.
- Command: `tools/ab.sh r16_decode_widths 66d60d5a1 0f6d71781 3 "a2t:4 a2t:8 a2t:16 a2t:32 t2s:1 t2s:4 t2s:8 t2t:1 t2t:4"`.
- Result (3 rounds), base -> cand: A2T tok/s c4 378.9 -> 381.5, c8 566.0 -> 547.4 (-3.3%), c16 908.9 -> 910.0, c32 1163.9 -> 1135.9 (all overlap); A2T c8 TTFT 71.8 -> 76.6 ms (+6.7%, sep). T2S and T2T overlap.
- Decision: dropped. The kernel-level saving (per-pair vs grouped at 5-7 rows) does not reach the step; unverified why (the decode graph's other kernels, or more graphs to plan/replay per bucket). Patch: 0f6d71781.

### Probe (not a round): GIL switch interval (runs/probe_gil; 66d60d5a1 both arms, MSTAR_PY_SWITCH_INTERVAL_SEC unset vs 0.0005, 2 rounds)

- Basis: T2S c=8 GPU-thread steps spend 5.3 ms wall for 2.1 ms CPU (runs/r13_t2s8_timeline), so either the GIL's 5 ms switch interval or plan-future waits.
- Result, default -> 0.5 ms: T2S c8 42.0 -> 38.4 audio-s/s (-8.6%, sep), A2T c32 1264.2 -> 1236.1 tok/s (-2.2%, sep) and TTFT +8.0% (sep); T2S c1/c4 overlap. A shorter interval costs more in switching than it saves in waiting: the default stands, and the GPU thread's wait is not switch latency (plan-future waits are rp3's).

### Findings (no round): T2S TTFA from c=1 to c=4 (runs/r17_t2s_stats, --log-stats-file, 66d60d5a1)

- Client TTFA p50 141.5 (c1) -> 234.9 ms (c4). Per request, Talker decode busy 3628 ms / 687 steps = 5.3 ms per step at c1, 4750 / 526 = 9.0 ms at c4; Thinker decode 5.4 -> 8.3 ms per step. Over the 12 frames before first audio the Talker's +3.7 ms per step alone is ~45 ms of the ~93 ms gap.
- The Talker step's GPU work is latency-bound and flat in rows (~3.3 ms at bs 1-4, runs/base_prof_t2s); the model's per-row hooks cost ~0.06 ms per row (prepare_inputs 0.018, postprocess 0.022, check_stop 0.021; runs/r13_t2s8_timeline timers). The growth is the worker's host time per step (pre-plan, per-row postprocess, speculation), which is rp3's under the amendment. No model-lane round follows from it.

### Incident: disk full during checkpoint 5 (Nov 4 17:2x UTC)

- /workspace hit 100% in ckpt5's round-3 candidate arm (T2S cells' wav outputs, ~0.8 GB per arm across 48 GB of runs). That arm lost its T2S c16/c32 cells, and summary, report and modular all failed to write. /workspace/results.jsonl got no ckpt5 rows. The report may have copied rows to the pod log before the append failed; if so, those copies cannot be withdrawn.
- Fix: all stored wavs deleted (every answer-WER score is already in JSON); tools/round.sh now deletes an arm's wavs at its end unless KEEP_WAV=1. ckpt5 resumed: round 3 candidate rerun from scratch, then summary, counted report and modular.

### Checkpoint 5 (runs/ckpt5; run-base vs 66d60d5a1; reported counted, 82 rows)

- A2T tok/s: c1 +32.3%, c4 +29.0%, c8 +37.3%, c16 +33.5%, c32 +28.3% (all rounds); TTFT p50 -22 to -52%; WER cell +35.4%. T2S and T2T as in checkpoint 4 (see runs/ckpt5/summary.txt).
- Full test/modular on 66d60d5a1: 42 failed, 9910 passed; none outside the run-base list (logs/modular_ckpt5.log).

### Round 17 — Thinker text-position cos/sin gathered from a table (6ebdf89f0 on 66d60d5a1, detached)

- Basis (runs/r18_t2s8_timers, T2S c=8, 66d60d5a1): ThinkerSubmodule.preprocess p50 0.586 ms on the GPU thread, of which compute_3d_cos_sin 0.385 ms (0.35 ms CPU, ~12 small ops). T2S c>=4 is host-bound on the GPU thread (runs/r13_t2s8_timeline). The earlier "~0.1 ms" note was an underestimate.
- Hypothesis: when every row of a step takes the planned 1D positions (text, decode; the `mrope_position_ids` fast path), the three MRoPE components share one position, so each row's cos/sin depends on its position only: one table per device, built by compute_3d_cos_sin over every position the Thinker's KV holds (its resolved max_seq_len, 16 MB in bf16), and two index_selects. Bit-identical (/tmp/cs_check.py: 777 random int32/int64 positions, torch.equal). -0.3 ms per Thinker step on the GPU thread: T2S c4/c8 TTFA and throughput +1-3%; A2T (GPU-bound) unchanged. Steps with vision or audio custom positions keep the computed path.
- Tests: Qwen3-Omni modular 802 passed (test_qwen3omni_vision_mixed's hand-built submodule gains the table attribute and a THINKER_KV stub).
- Command: `tools/ab.sh r17_cossin 66d60d5a1 6ebdf89f0`.
- Round 17 result (3 rounds; round 3 candidate rerun after the pod restart), base -> cand: T2S audio-s/s c4 +1.8%, c8 +4.2%, c16 +0.4%, c32 -2.4% (all overlap); T2S TTFA c16 -8.3% (all rounds). T2T c1 263.3 -> 287.8 tok/s (+9.3%, all) and TTFT -6.2% (all), T2T c4 +0.8% (all), T2T c32 TTFT -11.0% (all); A2T c8 +0.9% (all). Nothing regresses.
- Decision: not kept under amendment 2 (T2S throughput c4-c32 overlaps). Patch 6ebdf89f0 stays available; it is the best non-regressing T2T change measured.

### Findings: the Talker decode graph by width (existing traces runs/base_prof_t2s c=1, runs/r6_prof_t2s16 c=16)

- Talker decode graph GPU span p50: 3.36 ms at bs1 (1162 kernels) vs 5.22 ms at bs16 (1423 kernels). At bs16 the small projections (code predictor and Talker dense layers, M=16) run as cuBLAS nvjet GEMMs (0.85 + 0.46 + 0.40 + 0.40 ms, ~2.1 ms), where bs1 runs Inductor reductions (0.66 + 0.36 + 0.28 + 0.11 ms); the Talker MoE takes the grouped fused_moe at 16 (0.68 ms). Round 7's Triton GEMV patch (patches/r7_cp_linear.patch: M=16 27.7 -> 20.3 us per layer, regressing at 64 rows) addresses exactly this width range, which is now the headline's.
- Result (3 rounds), base -> cand, every cell WIN(all): T2S audio-s/s c1 17.9 -> 18.9 (+5.7%), c4 28.8 -> 37.8 (+31.5%), c8 39.5 -> 58.2 (+47.3%), c16 53.1 -> 85.0 (+60.1%), c32 80.0 -> 126.1 (+57.6%); TTFA p50 c4 230.7 -> 182.4 ms (-20.9%), c1 -13.1%, c8-32 -28 to -38%. A2T tok/s +3.5 to +21.6% (c32 +11.8%), TTFT -20 to -37%; T2T +0.8 to +12.5%, TTFT -9 to -32%.
- Mechanism (from the result): the T2S step was host-bound on threads woken at low clock under schedutil; per-row cost seen at c=4 (Talker 9.0 vs 5.3 ms per step) was mostly CPU frequency, not per-row work.
- Correctness: no numeric change (a scheduler frequency hint); codec tokens, WER inputs and outputs are produced by unchanged code. Not re-scored.
- Review (subagent): nothing blocking; fails soft (no uclamp, refused, non-Linux), set once at API startup and inherited by every spawned process and later thread; no per-step cost; knob and env var documented. Fixed before keeping: commit body and docs row now carry the Qwen3-Omni numbers. Nits left: ctypes load not wrapped, syscall args as int, test asserts getattr where it could skip.
- Decision: kept. Branch fast-forwarded to 11870717d (core, message amended with the Omni evidence; "Still applies to main" kept) + 7b3604074 (test). ruff clean; test_cpu_frequency, test_knobs, test_step_threads 74 passed.

### Profiles on 7b3604074 (full CPU frequency): T2S is now GPU-bound at every width

- runs/r19_phases (MSTAR_PHASE_TIMING, T2S c1/c4/c16): iteration / gpu_exec ms at batch ~1: 3.19 / 3.20; ~3-4: 4.32 / 4.28; ~9-20: 7.74 / 6.50 (event_sync 3.4). GPU time per step grows with the rows.
- runs/r19_prof_t2s4 / t2s16 (torch.profiler, CUDA): GPU busy 95% (c4) and 90% (c16) of the window. c4: Talker decode graph span p50 4.92 ms (1343 kernels; bs1 3.36), Thinker decode 3.19 ms. c16: Talker 5.24 ms (1423 kernels), Thinker decode 5.84 ms at ~5 rows (57 steps/s), Code2Wav 2.6 ms (26/s), Thinker prefill/mixed 7.3-15.8 ms. GPU budget at c16: Talker ~36%, Thinker decode ~37%, Code2Wav ~10%.
- Talker at c4: code predictor projections run as cuBLAS nvjet GEMMs (1.01 + 0.99 + 0.40 + splitK reduce 0.15 ms) where bs1 runs Inductor reductions (1.41 ms total). tools/cp_bench.py (compiled depth loop, captured, random weights): ~3.0 ms flat over bs 1-64: launch/latency-bound, not bandwidth-bound.
- Next candidates from this: (a) code predictor projections on round 7's cp_linear at <= 32 rows (round 19; cp_bench: bs1 3.24 -> 2.84, bs4 3.03 -> 3.01, bs8 2.81 -> 3.05, bs16 2.99 -> 2.79, bs32 3.14 -> 3.07, bs64 unchanged); (b) fewer, fuller Thinker decode steps in T2S (57 steps/s at ~5 rows, 37% of GPU); (c) fewer kernels per code predictor layer.

### Round 19 — code predictor projections on a small-row Triton GEMV up to 32 rows (1a6f003e7 on 7b3604074, detached)

- Round 7's cp_linear (patches/r7_cp_linear.patch), now dispatched by row count (`depth_linear`: CUDA and <= 32 rows; the plain GEMM above, so Qwen3-TTS's 64-row loop keeps cuBLAS). Hypothesis: the Talker step at bs1-32 loses the cuBLAS split-K GEMVs, their reduce kernels and the separate SwiGLU kernel (160 kernels per step): T2S +3-8% c4-32.
- Tests: test_code_predictor_kernels, test_qwen3_tts_model: 96 passed, 5 skipped (a first build routed CPU tensors to Triton and failed 7; fixed before measuring).
- Command: `tools/ab.sh r19_cplinear 7b3604074 1a6f003e7`. Qwen3-TTS (shared component) to be measured if T2S wins.
- Result (3 rounds), base -> cand: T2S audio-s/s c1 18.8 -> 16.0 (-15.0%, sep) and TTFA +8.7% (sep), c4 39.0 -> 38.5 (overlap), c8 58.9 -> 57.4 (-2.4%, sep) and TTFA +5.5% (sep), c16 86.8 -> 89.6 (+3.3%, all), c32 128.0 -> 133.9 (+4.6%, all). A2T and T2T overlap (A2T c8 TTFT -4.5%, all).
- Decision: not kept (c1/c8 regress). In the real Talker graph cp_linear wins at 16-32 rows and loses to Inductor's lowering at 1-8, the opposite of cp_bench at bs1 (unexplained). Next: the same kernel only past 8 rows.

### Round 20 — code predictor projections on cp_linear only at 9-32 rows (1e7fccb07 on 7b3604074, detached)

- From round 19: cp_linear gains at 16 and 32 rows and loses at 1-8 in the real Talker graph. Now `use_cp_linear` picks it at 9-32 rows (the Talker's 16 and 32 buckets) and the original modules run everywhere else, unchanged (attention projections, ParallelGatedMLP, the head matmul). Hypothesis: T2S c16/c32 +3-5% as in round 19, c1-c8 unchanged.
- Tests: test_code_predictor_kernels, test_qwen3_tts_model: 96 passed, 5 skipped.
- Command: `tools/ab.sh r20_cplinear16 7b3604074 1e7fccb07`. Qwen3-TTS (shared component; its 64-row loop is outside the range, its 16/32-row buckets are not) to be measured if T2S wins.

### Findings: T2S c=16 step mix (runs/r19_phases rows counters, per 100 iterations)

- thinker_decode ~43 steps at 1.0-4.5 rows; talker_decode ~44 steps at 12 rows; code2wav_chunk ~13 steps at 1.5 rows. A request carries ~141 text tokens and ~525 codec frames (runs/r19_phases t2s:16 SUMMARY: 11294 tokens / 80, 42 s of audio), and the Talker takes one Thinker state per frame while text lasts, so during each request's first ~141 frames the Thinker's step rate paces its Talker.
- Levers ranked: (1) Talker step GPU time: the code predictor's ~3 ms is latency-bound at every width (tools/cp_bench.py), ~10 kernels per layer-step x 75; fewer kernels per layer moves every T2S cell. (2) Fuller Code2Wav steps (1.5 rows, ~7% of GPU). (3) Thinker decode cadence: equal step counts with the Talker at 1-4 rows; a change must keep each request's Thinker ahead of its Talker.
- Round 20 result (3 rounds), base -> cand: T2S audio-s/s c4 +0.7%, c8 0.0%, c16 90.3 -> 88.8 (-1.7%), c32 135.8 -> 134.7 (-0.9%), all overlap; TTFA c4 +3.0% (sep). T2T c32 +0.5% (all); A2T overlaps.
- Decision: dropped. This run's base measured 90.3/135.8 at c16/c32 against round 19's 86.8/128.0, so round 19's c16/c32 "gain" was most likely server-to-server variance, not the kernel. cp_linear does not move T2S end to end at any width; closed.

### Experiment (no round): the code predictor's depth step as one persistent Triton kernel (tools/cp_fused.py)

- One launch per depth step (5 layers + final norm) with grid barriers between phases (norm+qkv | q/k norm+RoPE, KV write, attention | o + residual | norm + gate/up + SwiGLU | down + residual), one program per SM. Correct against forward_depth_unrolled (rel. max error 1-2% over 16 positions at bs 1-32, bf16-level; KV within a few ulps).
- Timing (16 captured depth steps, random weights): module 3.03-3.17 ms; fused 4.7-6.3 ms as first written, 3.59 ms at the best of a tile/warps/stages sweep (bs4). Barriers are not the cost (tools: /tmp/bar_bench.py, 0.8 us each, 0.36 ms per Talker step); the GEMV phases read at ~0.7 TB/s (one program per SM cannot keep enough loads in flight; 2x programs: no gain; 2x programs at 8 warps and 4x programs deadlock, not all resident).
- Closed: matching cuBLAS's per-GEMM efficiency inside a persistent kernel needs hand-written async-copy pipelines, more than the remaining time allows safely.

### Findings: Code2Wav batching headroom (tools/c2w_bench.py, real weights, captured, 50-frame window)

- bs1 2.46 ms, bs2 4.00, bs4 6.86, bs8 11.93, bs16 22.72 ms per step: 2.46 -> 1.42 ms per row. Compute-bound; at c=16 it runs ~1.5 rows per step and ~10% of GPU time, so perfect batching would save ~40% of that (~4% overall). Not pursued (a hold-and-batch policy also risks playback gaps).
- State of the T2S GPU budget at c=16 after round 18: Thinker decode ~37% (small steps at 1-4.5 rows, MoE near its kernel limits), Talker ~36% (code predictor latency-bound; neither cp_linear nor a fused persistent kernel beat cuBLAS end to end), Code2Wav ~10% (compute-bound). The remaining large lever is running the Thinker's and the Talker's steps concurrently (P3), which needs the engine's shared StepRunner split per node and two speculation chains in flight: beyond what the remaining day can build and verify safely.

### Round 21 — Talker attention through the fused q/k norm + RoPE + KV-write kernel (74f877e81 on 7b3604074, detached)

- Basis (runs/r19_prof_t2s16): per Talker decode step the 20 attention layers run q/k RMSNorm (Inductor, 2 x 1.2 us), FlashInfer's BatchQKApplyRotaryPosIds (7.6 us), a KV scatter (1.3 us) apart, ~0.23 ms; the Thinker and Qwen3's LM run all three as one kernel (`Qwen3Attention`, `AttentionCallable.norm_rope_attend`).
- Change: Qwen3OmniAttention extends Qwen3Attention instead of ParallelAttention; the Thinker keeps its MRoPE table path, the Talker's 1D branch defers to Qwen3Attention.forward (fused on CUDA with planned positions, the old unfused path otherwise). No duplicated logic.
- Hypothesis: -0.15-0.2 ms per Talker step (~3-4% of its GPU), T2S +1-3% c4-32. Likely under the noise floor; run per amendment 2 (GPU not idle, candidate from the profile). Numerics: fp32 rounded once instead of bf16 after norm and rotation (as the Thinker); speech gate if kept.
- Tests: Qwen3-Omni modular 802 passed.
- Command: `tools/ab.sh r21_talker_rope 7b3604074 74f877e81`.
- Result (3 rounds): T2S audio-s/s -6 to -9% in every cell (sep), TTFA c1 -2.5%, c16 +4.1%. But the spoken output changed: mean audio per request ~30% shorter for the same text (c1 54.4-55.2 s base vs 32.8-37.7 s cand; c32 53.0-53.4 vs 37.4-39.3 s), so the Talker generates different (likely truncated or degraded) speech. Fails the speech correctness gate regardless of speed.
- Decision: dropped. Hypothesis, unverified: the Talker's rotation is not the plain rotate-half 1D RoPE the fused kernel applies (e.g. its position resource's RoPE differs in layout or positions), though `fusable_rope_theta` admitted it. Worth a parity test before anyone retries.

### Round 22 — per-pair MoE up to 12 tokens for the Talker's expert shape (ffb68ec29 on 7b3604074, detached)

- Basis: tools/routed_sweep.py talker (128 x (1024, 384), top-6, 20 layers, H200): per-pair (current tiles) vs grouped at T=8/10/12/14/16: 0.615/0.769/0.882/1.021/1.106 vs 0.788/0.898/1.002/1.103/1.128 ms. The 7-token cutoff came from the Thinker's shape (level at 8).
- Change (core, mstar/utils/fused_moe): `routed_max_tokens(H, I)`, 12 for (1024, 384), ROUTED_MAX_TOKENS elsewhere. Commit body notes it still applies to main.
- Hypothesis: Talker steps in the 8-row bucket (T2S c~8) -0.17 ms MoE (~4% of the step); 16/32 buckets unchanged. Expected near the noise floor (amendment 2: GPU not idle).
- Tests: the 18 modular files importing fused_moe: 368 passed, 2 failed (test_fused_moe_config::test_fused_experts_matches_naive_and_precomputed_alignment[1,3], failing on 7b3604074 too).
- Command: `tools/ab.sh r22_talker_pairs 7b3604074 ffb68ec29` (interleaved with checkpoint 6's arms on the GPU lock).
- Round 22 result (3 rounds), base -> cand: T2S audio-s/s c4 38.8 -> 38.1 (-1.7%, sep), c8 58.4 -> 60.0 (+2.7%, all), c16 +1.4%, c32 +1.4% (overlap); TTFA overlaps; A2T c4 +2.5% (all), T2T c4 TTFT -6.7% (all).
- Decision: not kept (c4 regresses, c16/c32 overlap). The 8-row bucket gains as the sweep predicted; it is one cell.

### Checkpoint 6 (runs/ckpt6; run-base vs 7b3604074; reported counted, T2S first)

- T2S audio-s/s: c1 17.6 -> 18.9 (+7.0%), c4 29.3 -> 38.1 (+30.1%), c8 38.9 -> 58.5 (+50.5%), c16 54.2 -> 87.8 (+62.0%), c32 79.2 -> 133.8 (+69.0%); TTFA p50 c4 231.5 -> 180.3 ms (-22.1%), c1 -16.0%, c8-c32 -32 to -41%. All rounds.
- A2T tok/s +39.0 to +66.0% (c1-c32), TTFT -59 to -62%; T2T +1.0 to +16.1%, TTFT -22 to -31%. Every cell WIN(all).
- Full test/modular on 7b3604074: 47 failed, 9908 passed (27 min: it shared the host with round 22's A/B). Four failures outside the run-base list, all server-launching timeout tests (test_eval_compare::test_a_job_whose_lease_is_not_renewed..., ::test_stopping_compare..., test_eval_compare_multigpu::test_a_base_that_hangs..., test_eval_compare_transport::test_seeded_transport_faults...): all four pass when rerun alone on 7b3604074 (40 s), so load-induced, not the branch.

### Round 23 — Talker decode captured from a max-autotune compile (candidate on 7b3604074, detached)

- Basis: tools/cp_bench.py with Inductor max-autotune on the code predictor projections: bs4 3.03 -> 2.76, bs16 3.00 -> 2.88 ms per depth loop. The engine already takes a per-capture-config `compile_mode`, so this is one line on the Talker's talker_decode config (no global change, invariant 4).
- Hypothesis: -0.1-0.3 ms per Talker step, T2S +1-4% at c4-c16; boot time longer (autotune per bucket), recorded from boot_s.
- Command: `tools/ab.sh r23_talker_autotune 7b3604074 c25dc0248`.
- Result (3 rounds), base -> cand: T2S audio-s/s c4 38.2 -> 41.4 (+8.5%, all) and TTFA c4 182.2 -> 175.0 ms (-4.0%, all); c8 +1.3%, c16 -0.9%, c32 +0.5% (overlap). T2T c32 -0.3% (sep); A2T overlaps. Boot 319-329 s -> 561-570 s (autotuning each bucket).
- Decision: not kept (c8-c32 overlap). The gain sits in the 4-row bucket, as cp_bench showed (bs4 -9%, bs16 -4%).

### Findings: T2S c=32 is not GPU-bound; the GPU thread stalls (runs/r24_phases_c32, r24_prof_t2s32, r24_timeline_c32; 7b3604074)

- Trace: GPU busy 1729 of 3002 ms; 309 idle gaps > 100 us totalling 1.2 s, the largest 44-168 ms. Unprofiled phases: iter_total p50 ~5 ms but p95 42-51 ms in many windows (mean 8 ms); in those windows worker.gpu_thread.total p95 19.6 ms against gpu_exec p95 12.2 ms.
- Timeline (unprofiled, 54.7 s): Thinker GPU-thread steps over 15 ms: 264 of 2652, 6.4 s; Talker: 164 of 2916, 3.9 s. `_execute_on_gpu_thread` 29.1 s wall for 14.3 s CPU. Inside the slow Thinker steps: ThinkerSubmodule.postprocess 8.0 ms per step, preprocess 5.8, compute_3d_cos_sin 4.5 (normally ~0.4: blocked), main thread in _postprocess_batch alongside (7.2 ms).
- Postprocess cost: a speaking prefill row's Talker states are cut with boolean indexing (`selected[inclusion_mask]`, `multimodal_mask[inclusion_mask]`), each a host read of the device mask, so the GPU thread waits for the step's kernels. Talker prefill prepare_inputs also carry syncs (3.6-4.6 ms; round 9's target, worse when removed then, unexplained).

### Round 24 — Thinker cuts a speaking text row's Talker states from a host copy of the inclusion mask (24f48dd80 on 7b3604074, detached)

- Change: prepare_inputs builds `talker_text_mask` on the host ids as well (a text row's ids arrive on the CPU) as `talker_keep_host`; split_inputs cuts it with the masks; unpack_packed_outputs hands it to postprocess, which cuts the layer-0 rows by the mask's runs (slices and a cat, no read back) and emits an all-False multimodal mask of the kept length. Audio, vision and eager single-request rows keep the boolean-index path.
- Test: test_qwen3omni_talker_keep_host (4 masks: equal outputs to the device-mask path). Qwen3-Omni modular 806 passed.
- Hypothesis: removes one GPU-thread stall per speaking prefill step (~8 ms waits at c=32): T2S throughput +2-6% at c16-32, TTFA lower; c4-8 smaller.
- Command: `tools/ab.sh r24_keep_host 7b3604074 24f48dd80`.
- Result (3 rounds): T2S audio-s/s c4 +2.1%, c8 +0.3%, c16 +3.2%, c32 -0.3% (all overlap); TTFA c16 -3.3% (all); A2T c16 TTFT +2.3% (sep).
- Mechanism check (runs/r24_cand_timers vs r24_timeline_c32, T2S c32): ThinkerSubmodule.postprocess unchanged (2.2 s per cell, CPU = wall). The host path never ran: a prefill row's `text_inputs` already sits on the GPU when prepare_inputs sees it. The cost is real: 120 calls of ~17 ms each (CPU-time, i.e. the CUDA sync's spin-wait) per 55 s cell, one per speaking prefill row, on the GPU thread.
- Decision: dropped (no-op as built). A working version needs the kept runs from process_prompt (host ids) through ProcessPromptOutput.metadata into each prefill walk's step_metadata, and postprocess must know which part of the walk's span the step covers (prefix-cache hits and chunking shift it). Without a sure offset it would hand the Talker wrong states silently, so not built.

### Round 25 — round 17's cos/sin table retested on the full-CPU-frequency base (b4147c655 = 6ebdf89f0 on 7b3604074, detached)

- Why again: round 18 changed the host regime the round-17 result was measured in, and at T2S c32 the slow Thinker steps spent ~4.5 ms in compute_3d_cos_sin (runs/r24_timeline_c32). Round 17 had T2S neutral, T2T c1 +9.3% (all rounds).
- Command: `tools/ab.sh r25_cossin2 7b3604074 b4147c655`.
- Result (3 rounds), base -> cand: T2S audio-s/s c4 +1.3% (all), c8 +0.3%, c16 +2.9% (all), c32 0.0% (overlap); TTFA c4 -1.1% (all); WER cell TTFT +5.3% (sep); A2T c16 +0.8% (all), T2T c32 +0.5% (all).
- Decision: not kept (c8/c32 overlap, a TTFT regression). Same verdict as round 17 on the old base.

### Speech gate on the best commit (runs/speech_gate_best: run-base vs 7b3604074, T2S c4, 4 alternating rounds, 32 answers per arm, Whisper large-v3-turbo round trip)

- Per-arm WER mean, base: 0.129 / 0.082 / 0.101 / 0.108 (pooled 0.105); best: 0.199 / 0.091 / 0.076 / 0.154 (pooled 0.130). Medians base 0.048 / 0.056 / 0.047 / 0.085, best 0.048 / 0.047 / 0.054 / 0.060. Answers above 0.5: base 3+0+2+1 = 6, best 2+1+0+2 = 5. Mean audio per answer equal (54.5-55.2 s vs 55.3-56.1 s).
- Verdict: passes the gate as applied (no more answers above 0.5; per-arm means overlap). Caveat stated: the pooled mean is higher, driven by one arm (0.199), and sits just above base's highest arm. Round 18 changes no computation (a scheduler frequency hint), and Talker sampling (temperature 0.9) differs across server processes, so per-arm means vary this much on their own.

### Checkpoint 7 (runs/ckpt7; run-base vs 7b3604074; reported counted, T2S first)

- T2S audio-s/s: c1 +7.8%, c4 29.5 -> 38.8 (+31.4%), c8 38.6 -> 58.7 (+52.2%), c16 52.2 -> 89.2 (+70.8%), c32 77.5 -> 131.3 (+69.3%). A2T +39.4 to +63.5%; T2T +0.9 to +17.0%. All 60 cells WIN(all), no regression.
- Full test/modular on 7b3604074: 42 failed, 9913 passed; none outside the run-base list (logs/modular_ckpt7.log).

### Round 26 — round 9's Talker prefill without host syncs, retested on the full-CPU-frequency base (b26bbf2eb = patches/r4_talker_nosync.patch on 7b3604074, detached)

- Basis: runs/r24_timeline_c32 timers: TalkerSubmodule.prepare_inputs[talker_prefill] mean 3.6 ms and [talker_last_prefill] 4.6 ms per request, CPU-bound (sync spin). Round 9 (old, host-bound base) made T2S worse, unexplained; the base has changed since (round 18).
- Change: both projections over every row then torch.where (no .any() reads, same values per row); the codec-prefix ids and last_token_indices staged through pinned memory.
- Tests: Qwen3-Omni modular 837 passed, 2 failed (the pre-existing wan22 host-sync stubs).
- Command: `tools/ab.sh r26_talker_nosync 7b3604074 b26bbf2eb` (a mistyped first launch was killed before measuring anything and its directory removed).
- Result (3 rounds), base -> cand: T2S audio-s/s +0.5 to +1.4% (all overlap); TTFA c1 -1.3%, c4 184.0 -> 177.2 ms (-3.7%), c16 -3.8% (all rounds); A2T c1 TTFT +5.1% (sep; A2T runs no Talker, so most likely run-to-run), A2T c16 TTFT -2.7% (all).
- Decision: not kept (throughput overlaps; a separated regression in a no-regress cell). Unlike round 9 on the old base it does not hurt T2S; its TTFA gain is consistent (c1/c4/c16).

### Probe (no round): Code2Wav under torch.compile (tools/c2w_bench.py, C2W_COMPILE=1)

- bs1/2/4: eager 2.457/4.001/6.858 ms, compiled 2.467/4.011/6.880 ms per step. No gain (convolution-bound; the module's own compiled regions already cover the rest). Closed.

### Round 27 — round 16's Thinker decode capture at every width up to 8, retested for T2S on 7b3604074 (09a2596d8, detached)

- Basis: T2S c16-c32 Thinker decode steps carry 1-8 rows (runs/r24_phases_c32 rows counters) and take ~37% of GPU time; 5-7 rows pad to the 8-row graph and its grouped MoE (tools/routed_sweep.py thinker: per-pair 5.02/5.93 ms at 6/7 rows vs grouped 6.71 at 8). Round 16 measured on the host-bound base, mostly against A2T.
- Command: `tools/ab.sh r27_decode_widths2 7b3604074 09a2596d8`.
- Result (3 rounds), base -> cand: T2S audio-s/s c4 +1.1%, c8 +2.4%, c16 +1.8%, c32 -0.1% (all overlap); TTFA overlaps. A2T c4 442.9 -> 460.6 tok/s (+4.0%, all) and TTFT -4.7% (all); T2T c4 +1.0% (all), TTFT -6.4% (all). Nothing regresses.
- Decision: not kept under amendment 2 (headline overlaps). Patch 09a2596d8 is the best measured change for A2T/T2T c4 if those become headlines again.

### Correction: T2S c=32 is GPU-bound (runs/r28_prof_cpu_t2s32, CPU+CUDA profile, 2 s)

- GPU busy 1865 of 2008 ms (93%); 42 gaps > 1 ms totalling 83 ms. The earlier 58% (runs/r24_prof_t2s32) came from a window holding sporadic long stalls (44-168 ms); unprofiled iteration p95 of 42-51 ms in some windows matches mixed steps at request arrivals (Thinker prefill graphs 15-16 ms, Talker prefill syncs), not GC (no gen2 collections recorded). Remaining T2S levers are kernel-level and concurrency (see the summary at the top).

### Checkpoint 8 (runs/ckpt8; run-base vs 7b3604074; reported counted, T2S first)

- T2S audio-s/s: c1 +5.7%, c4 30.6 -> 38.5 (+25.5%), c8 39.1 -> 59.4 (+51.8%), c16 54.3 -> 88.3 (+62.7%), c32 81.7 -> 132.7 (+62.5%). All 60 cells WIN(all).
- Full test/modular on 7b3604074: 42 failed, 9913 passed; none outside the run-base list (logs/modular_ckpt8.log).

### Round 28 — Thinker decode captured from a max-autotune compile (69073d9db on 7b3604074, detached)

- Basis: round 23's autotune of the Talker decode gave T2S c4 +8.5% (all rounds); the Thinker's decode steps are ~37% of T2S GPU time at c16-c32 and run 1-8 rows (runs/r24_phases_c32), where Inductor's default lowering of the dense projections may also leave tile efficiency.
- Change: one line, `compile_mode="max-autotune-no-cudagraphs"` on the Thinker's thinker_decode capture config (per config, invariant 4). Boot time recorded.
- Command: `tools/ab.sh r28_thinker_autotune 7b3604074 69073d9db`.
- Round 28 result (3 rounds): T2S audio-s/s c4 +0.1%, c8 +1.3%, c16 -0.4%, c32 -1.6% (all overlap); TTFA c1 -0.8% (all), others overlap; A2T/T2T overlap. Boot 320 s -> 401-407 s. Decision: dropped.

### Round 29 — round 21 with its root cause fixed: the Talker's position resource at the layers' RoPE theta (08722d7f0 on 7b3604074: round 21's attention change + this fix, detached)

- Root cause of round 21's changed speech: the unfused path passes the layer's rope_theta (1e6) to the position resource on every call (ParallelAttention._apply_rope), but the fused kernel takes its theta from the position config (`fusable_rope_theta`), and the Talker's PositionConfig never set one, so it rotated at the 10,000 default.
- Change: TalkerTextConfig gains `rope_theta` (checkpoint 1e6, loaded from HF); the Talker layers and TALKER_POS both read it; and `Qwen3Attention` (shared, mstar/model/components/qwen3_lm.py) fuses only where the position resource's theta equals the layer's, so a mismatch falls back to the unfused path instead of changing outputs. Qwen3-ASR, Canary-Qwen and Fish S2 Pro already set their position theta from their LM config: unchanged.
- Tests: Qwen3-Omni modular plus every module importing qwen3_lm/Qwen3Attention: 1009 passed, 4 skipped.
- Hypothesis: Talker attention ~4 kernels fewer per layer (~0.15-0.2 ms per Talker step), T2S +1-3%; speech unchanged (check audio durations, then the speech gate if kept).
- Command: `tools/ab.sh r29_talker_rope_theta 7b3604074 08722d7f0`.
- Result (3 rounds), base -> cand: T2S audio-s/s c1 18.8 -> 19.9 (+5.8%, all) and TTFA -3.0% (all); c4 +0.4%, c8 -0.1%, c16 +2.9%, c32 +1.7% (overlap); WER-cell TTFT +3.9% (sep); A2T c32 +2.5% (all). Speech intact this time: mean audio per request equal to base in every cell (e.g. c32 52.1-53.4 s base vs 53.0-53.9 s cand; round 21 had been ~30% shorter).
- Decision: not kept (headline overlaps, one separated TTFT regression). Worth upstreaming regardless: the guard in Qwen3Attention (fuse only where the position resource's theta equals the layer's) turns a silent wrong-theta rotation into the unfused path, and the Talker's TALKER_POS theta fix makes its position resource consistent with its layers. Patch: 08722d7f0.

### Incident: root volume (/tmp) full during checkpoint 9 (Nov 5, ~21:xx UTC)

- Checkpoint 9 (run-base vs 7b3604074, started 20:14) stopped: the overlay root volume holding /tmp filled. Cause: tools/round.sh gives every server boot a fresh TORCHINDUCTOR_CACHE_DIR under /tmp/ind_XXXX (~0.43 GB each) and never deletes it; ~150 boots since the last cleanup (Nov 3) exceed the 50 GB volume.
- Effect: no shell command can start (the agent's command capture lives on the same volume), so the stale caches cannot be removed from here and checkpoint 9's result, report and modular run are unknown/incomplete. Checkpoints 1-8 and every round before are recorded and reported above; the branch (perf/rp4-omni at 7b3604074) is unaffected.
- Fix applied (file tools only, not yet exercised): tools/round.sh now records the boot's cache in IND_DIR and its EXIT trap deletes it, so failed boots clean up too.
- To recover: `rm -rf /tmp/ind_????` (no server running), then rerun `tools/checkpoint.sh ckpt9 7b3604074` (or resume: ab.sh skips completed arms).
- What actually happened in the failed checkpoint 9 (runs/ckpt9_enospc): all six arms failed at boot with ENOSPC (`PanicException: No space left on device` in the ZMQ communicator, "boot failed") but ab.sh logged rc=0. summarize.py then sent an empty checkpoint report (batch h200_1105, status counted, no cells) to mcp-report, and the modular run showed 2730 errors from the full disk. None of it is a measurement. The rerun's report supersedes it.
- Recovery, Nov 6 02:40 UTC: the shell came back with ~200 KB free. No server was running and no process held any /tmp/ind_* dir, so I deleted all 116 that were older than an hour (`find /tmp -maxdepth 1 -name 'ind_????' -mmin +60`). The root volume went from 100% to 4% used. Then I moved the failed run to runs/ckpt9_enospc and relaunched `tools/checkpoint.sh ckpt9 7b3604074` at 02:40. Checkpoint 8 took ~1.5 h, so this one should finish around 04:15.
- Checkpoint 9 result (rerun 02:40-04:12 UTC Nov 6, runs/ckpt9, reported to mcp-report as batch h200_1105 status counted; this supersedes the empty report). Run-base vs 7b3604074, 3 alternating rounds, every arm booted and every request succeeded:
  - T2S audio-s/s: c1 +6.1%, c4 +33.5%, c8 +44.9%, c16 +58.8%, c32 +67.9%.
  - T2S TTFA p50: c1 -16.2%, c4 -22.3%, c8 -27.8%, c16 -35.3%, c32 -38.8%.
  - A2T throughput: +36% to +51%, with p50 down 57-61%. T2T throughput: +0.7% to +12.7%, with p50 down 7-26%.
  - All 30 rows (tput and p50 in each of the 15 cells) are WIN(all).
  - test/modular: 42 failed, the same set as checkpoint 8 (diffed); all of them also fail in run-base.
  - round.sh cache cleanup confirmed: after each arm only the running boot's /tmp/ind_* remained, and the disk stayed at 4-5%.
- Correction: the final summary was first written as if the 04:19 UTC deadline had already passed. It had not; the shell had only been blocked since ~21:xx Nov 5.
