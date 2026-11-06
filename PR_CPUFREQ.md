IN PROGRESS: last 9B rounds, then the full serial test

# PR_CPUFREQ facts

## SHAs
- Base: 305e80ef46248ade6e6670aa20c9bb67f4ac2e26
- Head: 9882faa31132c346ef12a4cb29ca32d3356e779e
- Branch: perf/rp2-pr-cpufreq

## Commits
```
65c50f688 api: full_cpu_frequency, uclamp.min 1024 on every thread of a deployment
9882faa31 test: full_cpu_frequency reaches every thread and process, and a refusal stops startup
```
Sources: the code commit merges rp1 `3f362254` and rp2 `444a2399`. The test commit merges rp1 `8b585eca` and rp2 `b4ac7d1c`. Neither cherry-picks; each is written fresh on main's code.

Dropped:
- `dae9122b` (9B config), `491ab32f` (4B config) and the 0.8B config line in `444a2399`: the brief says no config sets the knob.
- rp2's `docs/models.rst` paragraphs: they described those config lines.
- `c67586f6`, `59d983e5`, `b8ace801` (bucket cut): assigned to rp3's decision, not this PR.

What each source did, and what the merged version does:

| | rp1 `3f362254` | rp2 `444a2399` | merged |
|---|---|---|---|
| knob | `full_cpu_frequency`, default on | `step_thread_util_min` 0-1024, default off | `full_cpu_frequency`, true/false, default off |
| threads | calling thread and what starts after it | each worker's main, GPU and plan threads and the API result threads, set from those threads | every thread in `/proc/self/task` at the call, and what starts after |
| Rust frontend process | covered | not covered | covered |
| refused by host | warning, carries on | warning, carries on | `CpuPlanError` -> `SystemExit` at startup, naming the kernel config and the permission |
| bad value | not checked (reads as off) | `knobs.check` fails at startup | `knobs.check` fails at startup (kept from rp2) |
| other governor | not mentioned | documented, silent | warning in the log |
| dynamo entry point | covered via `apply_cpu_settings` | not covered | covered via `apply_cpu_settings` (kept from rp1) |
| struct | `struct.pack` | `ctypes.Structure` | `ctypes.Structure` (kept from rp2; the test reads it back through the same struct) |

Choices:
- Name `full_cpu_frequency`: the knob now covers every thread, so `step_thread_*` would misdescribe it, and it reads like its neighbour `idle_hold`.
- True/false, not 0-1024: every measurement was at 1024, and no caller has a use for another value.
- `threads.py` and `worker.py` are untouched: setting the clamp once at the API server before the conductor starts reaches workers by inheritance, so no worker code path carries it.
- Main moved both sources' code: both applied by hand to `apply_cpu_settings` at `mstar/api_server/entrypoint.py`, and to main's second doc tree (`docs/mintlify/`) as well as `docs/*.rst`.

## Concept: full_cpu_frequency
**Changes:** `mstar/utils/cpus.py` (`_SchedAttr`, `run_at_full_frequency`), `mstar/utils/knobs.py` (`full_cpu_frequency`, `MSTAR_FULL_CPU_FREQUENCY`), `mstar/api_server/entrypoint.py` (`apply_cpu_settings`), `mstar/integrations/dynamo/worker.py` (calls it), `docs/environment_variables.rst`, `docs/serving.rst`, `docs/mintlify/environment_variables.mdx`, `docs/mintlify/serving.mdx`.

**Mechanism:** with the knob on, the API server calls `sched_setattr` with `uclamp.min` 1024 on every thread it has, before it starts the conductor. Threads and processes started afterwards inherit the clamp through fork and exec. Under `schedutil`, a clamped thread that wakes runs at the top frequency the governor can pick, where a thread that sleeps often would otherwise read as lightly loaded. Idle threads stay asleep, and there is no spinning.

**Evidence for the mechanism, Xeon 8460Y+, `schedutil`, idle host (`loopfreq.py`):** a thread with 1 ms of work between sleeps:
| sleep | no clamp | uclamp.min 1024 |
|---|---|---|
| 10 ms | 800 MHz, 2587 iter/ms | 1986 MHz, 5289 iter/ms (x2.04) |
| 1 ms | 799 MHz, 2664 iter/ms | 1998 MHz, 6487 iter/ms (x2.43) |

**Coverage on a live server (`threadcheck.py`, `uclampset -p` per thread, whole process tree):** knob on, after load: 771 threads in 38 processes, all 1024, the Rust frontend's included. Every knob-on round's thread check is in `runs/*/on_r*/threads_*.txt`; knob off and main read 0 on every thread.

**Effect:** see "Main against PR".

**Cost:** no per-step code; the only new work is one syscall per thread at startup. Off-against-main shows no throughput cost beyond noise on any cell (table below). Power was not measured.

**Tests** (each fails or errors at the parent, checked with the files copied onto a worktree at 305e80ef4; 13 of 13):
- `test/modular/test_cpu_frequency.py`: `test_every_thread_and_process_of_a_deployment_holds_the_clamp`, `test_a_refusal_names_what_the_kernel_needs` (EOPNOTSUPP, EPERM), `test_a_thread_that_exited_since_the_listing_is_not_a_refusal`, `test_only_a_governor_that_ignores_the_clamp_warns`, `test_a_platform_without_the_syscall_is_refused`, `test_the_api_server_asks_only_when_the_knob_is_on`, `test_a_refusal_stops_the_server_with_the_kernels_requirement`, `test_both_entry_points_apply_the_settings_before_the_conductor_starts`
- `test/modular/test_knobs.py`: `test_full_cpu_frequency_is_off_until_the_config_or_the_environment_asks`, `test_a_bad_full_cpu_frequency_fails_the_startup_check` (1024, -1, "max")

The real-kernel test skips where the kernel refuses uclamp, so CPU CI may not exercise it. It ran here: H200 host, Linux 6.8, `schedutil`.

## Main against PR
One H200, Xeon 8460Y+, torch and CUDA as in `venv-pr`/`venv-main305`. Arms: `M* main 305e80e` (main), `M* pr-cpufreq 9882faa31` knob off (off) and knob on (on), and main under `uclampset -m 1024 --` (mainuclamp). All: `--rust-frontend`, `MSTAR_SHM_ARENA=1`, `MSTAR_STEP_THREAD_CORE=siblings`, each tree's config, fresh server per round, arm order rotated, server on NUMA node 0 and client on node 1. Cells held to the host's common memory state (143-160 ns idle probe), with the mean recorded per round. Rows reported under batch `h200_1106`, hypothesis `PR check: cpufreq`.

"Step-core MHz" is the median of `scaling_cur_freq` over the cell, max of cpu0 and cpu80 at 2 s sampling, so it is coarse.

`q08` and `q9` are the brief's cells. Their cell list runs in one server: 0.8B ShareGPT c=1, 16, 64, then Random c=64; 9B ShareGPT c=1, 64. `q08b` and `q08c` are added rounds where every arm runs only ShareGPT c=1 and c=64.

## q08
| cell | arm | tok/s median [min-max] | x main | TTFT p50 ms median [min-max] | step-core MHz by round | mem ns by round | failed |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B Random 1024/256 c=64 | main | 9708.8 [9149.9-9782.3] | 1.000 | 141.3 [139.0-147.6] | 800 800 800 | 149 152 149 | 0 |
| Qwen3.5-0.8B Random 1024/256 c=64 | off | 9227.2 [8717.4-9712.8] | 0.950 | 187.8 [142.4-200.4] | 887 800 800 | 149 151 151 | 0 |
| Qwen3.5-0.8B Random 1024/256 c=64 | on | 13896.1 [13141.2-13938.6] | 1.431 | 61.8 [61.5-83.8] | 2000 1978 2000 | 149 150 151 | 0 |
| Qwen3.5-0.8B ShareGPT c=1 | main | 489.5 [482.6-693.6] | 1.000 | 16.7 [15.2-21.6] | 800 800 3100 | 148 151 149 | 0 |
| Qwen3.5-0.8B ShareGPT c=1 | mainuclamp | 769.9 [608.0-778.7] | 1.573 | 10.4 [10.3-13.4] | 920 2000 2190 | 151 151 150 | 0 |
| Qwen3.5-0.8B ShareGPT c=1 | off | 481.9 [476.9-489.3] | 0.984 | 17.1 [16.8-21.8] | 2000 800 800 | 149 157 151 | 0 |
| Qwen3.5-0.8B ShareGPT c=1 | on | 676.9 [610.3-702.9] | 1.383 | 12.5 [11.9-12.6] | 928 1070 1552 | 150 151 150 | 0 |
| Qwen3.5-0.8B ShareGPT c=16 | main | 5081.3 [5074.5-6995.3] | 1.000 | 20.8 [19.1-21.0] | 800 2000 800 | 148 149 148 | 0 |
| Qwen3.5-0.8B ShareGPT c=16 | off | 5082.0 [4370.1-5142.2] | 1.000 | 30.7 [21.3-36.1] | 800 800 800 | 150 170 151 | 0 |
| Qwen3.5-0.8B ShareGPT c=16 | on | 5284.3 [5171.6-7259.4] | 1.040 | 17.1 [14.1-21.1] | 2000 1362 3046 | 150 150 150 | 0 |
| Qwen3.5-0.8B ShareGPT c=64 | main | 11267.9 [10946.1-11428.8] | 1.000 | 37.5 [35.2-53.2] | 800 800 800 | 150 148 148 | 0 |
| Qwen3.5-0.8B ShareGPT c=64 | mainuclamp | 16190.7 [15999.1-16354.4] | 1.437 | 21.3 [20.9-21.7] | 998 3000 2847 | 150 152 150 | 0 |
| Qwen3.5-0.8B ShareGPT c=64 | off | 10933.2 [10408.7-11354.6] | 0.970 | 53.7 [37.9-54.4] | 800 800 800 | 149 152 180 | 0 |
| Qwen3.5-0.8B ShareGPT c=64 | on | 11924.2 [11721.7-16254.0] | 1.058 | 25.9 [21.0-29.4] | 2000 2000 2900 | 149 151 150 | 0 |
## q08b
| cell | arm | tok/s median [min-max] | x main | TTFT p50 ms median [min-max] | step-core MHz by round | mem ns by round | failed |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B ShareGPT c=1 | mainuclamp | 736.5 [488.3-770.1] | nan | 11.4 [10.9-17.1] | 3100 3100 805 | 149 148 151 | 0 |
| Qwen3.5-0.8B ShareGPT c=1 | on | 711.8 [666.6-737.8] | nan | 12.1 [11.6-13.4] | 3050 1414 2439 | 149 149 148 | 0 |
| Qwen3.5-0.8B ShareGPT c=64 | mainuclamp | 11862.6 [11652.9-16050.8] | nan | 27.3 [21.5-30.4] | 3700 2000 2000 | 149 158 149 | 0 |
| Qwen3.5-0.8B ShareGPT c=64 | on | 15698.8 [11791.6-16150.7] | nan | 22.2 [21.3-27.3] | 3700 2594 1987 | 150 150 150 | 0 |
## q08c
| cell | arm | tok/s median [min-max] | x main | TTFT p50 ms median [min-max] | step-core MHz by round | mem ns by round | failed |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B ShareGPT c=1 | main | 468.4 [449.9-486.8] | 1.000 | 17.5 [16.9-18.2] | 800 2000 | 148 155 | 0 |
| Qwen3.5-0.8B ShareGPT c=64 | main | 10916.6 [10507.6-11325.6] | 1.000 | 38.0 [36.8-39.2] | 2000 2000 | 149 152 | 0 |
## q9
| cell | arm | tok/s median [min-max] | x main | TTFT p50 ms median [min-max] | step-core MHz by round | mem ns by round | failed |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B ShareGPT c=1 | main | 185.3 [183.2-187.4] | 1.000 | 22.4 [21.9-23.0] | 3100 2000 | 149 149 | 0 |
| Qwen3.5-9B ShareGPT c=1 | off | 186.7 [186.7-186.7] | 1.008 | 24.0 [24.0-24.0] | 2000 | 152 | 0 |
| Qwen3.5-9B ShareGPT c=1 | on | 186.6 [185.8-187.4] | 1.007 | 21.2 [17.1-25.4] | 982 2997 | 232 144 | 0 |
| Qwen3.5-9B ShareGPT c=64 | main | 6480.4 [6471.2-6489.6] | 1.000 | 44.6 [44.5-44.8] | 2000 2000 | 149 148 | 0 |
| Qwen3.5-9B ShareGPT c=64 | off | 6490.0 [6490.0-6490.0] | 1.001 | 44.3 [44.3-44.3] | 2000 | 149 | 0 |
| Qwen3.5-9B ShareGPT c=64 | on | 6315.1 [6051.3-6579.0] | 0.975 | 43.9 [36.3-51.5] | 1187 2000 | 251 149 | 0 |

pi0.5 `pi05_droid`: not run. `lerobot/pi05_droid` isn't in `/workspace/hf/hub` on this pod.

The `uclampset` command: util-linux 2.39.3 refuses the brief's `uclampset -m 1024 -a -- cmd` with "missing -p option", since `-a` only works with `-p`. The rounds use `uclampset -m 1024 -- cmd`, and `threadcheck.py` confirmed with `uclampset -p` that every thread carries it.

## Correctness
Gate: teacher-forced top-1 agreement with HF (sdpa) on 64 fixed ShareGPT prompts, 128 tokens each (8192 positions), Qwen3.5-0.8B, passes when M*'s agreement is at least HF eager's against HF sdpa.

| arm | M* vs HF sdpa | HF eager vs sdpa | pass | first token |
|---|---|---|---|---|
| main | 0.99146 | 0.99084 | True | 0.969 |
| off | 0.99133 | 0.99011 | True | 0.953 |
| off2 | 0.99304 | 0.99158 | True | 0.984 |
| off3 | 0.99316 | 0.99170 | True | 0.969 |
| on | 0.99292 | 0.99219 | True | 0.969 |
| on2 | 0.99329 | 0.99084 | True | 0.984 |
| on3 | 0.99280 | 0.99121 | True | 0.984 |

Knob on 0.99280-0.99329, knob off 0.99133-0.99316: the ranges overlap and every run passes.

Free-running tokens at the gate's concurrency 16 differ between any two runs: main-off 36/64 prompts identical, off-on 30/64. One prompt at a time, tokens still aren't deterministic across servers: same-arm off-off and on-on repeats match on 60/64 each; off-on pairs match on 57, 59, 61 and 63 of 64. The knob adds no difference beyond that spread. Why runs differ at concurrency 1 was not investigated.

## Tests
- Parallel and serial baselines at the base don't match, so full runs are serial, per the brief.
- Baseline, serial: 190 failed, 11828 passed, 228 skipped, 405 warnings, 20 errors, 83 subtests passed in 9523.72s (2:38:43)
- Baseline, parallel (`-n 8 --dist loadfile`): 84 failed, 11941 passed, 228 skipped, 748 warnings, 13 errors, 83 subtests passed in 3201.54s (0:53:21)
- Serial needed `--continue-on-collection-errors`: 7 modules fail at collection at the base.
- Head, serial: not finished
- Failing at head and not at the base (serial): 
```
none
```
- Failing at the base and not at head (serial):
```
test/modular/test_accelerator_graph_backend.py::test_chatterbox_shape_graphs_keep_results_across_replays[cuda]
test/modular/test_audio_decode_limit.py::test_a_soundfile_rate_past_the_range_is_refused_before_any_read
test/modular/test_audio_decode_limit.py::test_omnilingual_asr_decode_keeps_a_clip_under_its_bound
test/modular/test_audio_decode_limit.py::test_omnilingual_asr_decode_stops_at_its_clip_bound
test/modular/test_audio_decode_limit.py::test_omnilingual_asr_refuses_a_rate_past_the_range_before_reading
test/modular/test_audio_decode_limit.py::test_qwen3_tts_reference_decode_stops_at_its_bound
test/modular/test_audio_decode_limit.py::test_soundfile_loaders_are_capped
test/modular/test_audio_features.py::test_load_audio_file_decodes_and_resamples
test/modular/test_audio_watermark.py::test_detect_resamples_to_the_rate_the_mark_was_added_at[16000-24000]
test/modular/test_audio_watermark.py::test_detect_resamples_to_the_rate_the_mark_was_added_at[48000-24000]
test/modular/test_audio_watermark.py::test_engine_captures_replay_the_eager_watermark_bit_for_bit
test/modular/test_bagel_cfg_parallel.py::test_kv_shm_directory_is_private_to_the_deployment
test/modular/test_base_load_audio.py::test_a_crafted_header_rate_is_refused_before_resampling[100003]
test/modular/test_base_load_audio.py::test_a_crafted_header_rate_is_refused_before_resampling[48001]
test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-flac-PCM_16]
test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-mp3-MPEG_LAYER_III]
test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-ogg-VORBIS]
test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-wav-PCM_16]
test/modular/test_blackwell_ptxas.py::test_compiled_interleaved_mrope_matches_eager[small table]
test/modular/test_blackwell_ptxas.py::test_floor_division_under_a_range_select[1024-16]
test/modular/test_blackwell_ptxas.py::test_floor_division_under_a_range_select[256-4]
test/modular/test_blackwell_ptxas.py::test_floor_division_under_a_range_select[64-1]
test/modular/test_chatterbox_model.py
test/modular/test_chatterbox_watermark_import.py
test/modular/test_client_error_text.py::test_a_transcription_whose_decode_fails_in_the_frontend_answers_with_an_error_id
test/modular/test_cosyvoice3_flow_capture.py
test/modular/test_cosyvoice3_model.py
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_compiles_with_a_float_eps[plain]
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_compiles_with_a_float_eps[residual]
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_matches_eager[gated_affine]
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_matches_eager[mod]
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_matches_eager[per_sample]
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_matches_eager[plain_mod]
test/modular/test_diffusion_dit_kernels.py::test_gelu_epilogue_matches_eager
test/modular/test_diffusion_dit_kernels.py::test_rms_norm_rope_kernel_matches_eager
test/modular/test_eval_compare_multigpu.py::test_a_head_that_hangs_is_cut_at_the_cap_with_stacks_and_fails
test/modular/test_eval_framework.py::test_sigterm_to_the_runner_stops_the_managed_server
test/modular/test_fish_s2_pro_codec_stream.py::test_compiled_incremental_decode_matches_eager_on_cuda
test/modular/test_fish_s2_pro_model.py::test_load_audio_decodes_a_clip_without_a_sample_count[1]
test/modular/test_fish_s2_pro_model.py::test_load_audio_decodes_a_clip_without_a_sample_count[2]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_clip_without_a_sample_count_at_its_decoded_length[0.3-0\\.\\d\\d]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_clip_without_a_sample_count_at_its_decoded_length[45-30\\.\\d\\d]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[16000-False]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[44100-False]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[44101-True]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[48000-False]
test/modular/test_flex_attention_resource.py::test_a_batched_step_can_share_one_ring_by_stride_0_expand[cpu]
test/modular/test_flex_attention_resource.py::test_a_batched_step_can_share_one_ring_by_stride_0_expand[cuda]
test/modular/test_flex_attention_resource.py::test_a_batched_step_does_not_reach_across_world_spans[cpu]
test/modular/test_flex_attention_resource.py::test_a_batched_step_does_not_reach_across_world_spans[cuda]
test/modular/test_flex_attention_resource.py::test_attend_matches_a_masked_dense_reference[gappy_ring-committed_blocks1]
test/modular/test_flex_attention_resource.py::test_attend_matches_a_masked_dense_reference[one_frame-committed_blocks0]
test/modular/test_flex_attention_resource.py::test_attend_takes_the_compiled_path
test/modular/test_flex_attention_resource.py::test_eager_flex_attention_does_not_honour_the_block_mask
test/modular/test_fused_moe_config.py::test_compiled_fused_experts_takes_the_tables_tiles[captured-cuda_align]
test/modular/test_fused_moe_config.py::test_compiled_fused_experts_takes_the_tables_tiles[captured-torch_align]
test/modular/test_fused_moe_config.py::test_compiled_fused_experts_takes_the_tables_tiles[eager_forward-cuda_align]
test/modular/test_fused_moe_config.py::test_compiled_fused_experts_takes_the_tables_tiles[eager_forward-torch_align]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_B300_SXM6_AC-0]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_B300_SXM6_AC-1]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_GB200-0]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_H100_80GB_HBM3-0]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_H100_80GB_HBM3-1]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_H100_80GB_HBM3-2]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_H200-0]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_H200-1]
test/modular/test_fused_moe_prefill.py::test_compiles_on_a_dynamic_token_count
test/modular/test_fused_moe_route.py::test_the_one_kernel_route_runs_under_a_symbolic_token_count[3]
test/modular/test_fused_moe_route.py::test_the_one_kernel_route_runs_under_a_symbolic_token_count[5]
test/modular/test_gdn_cuda_graph.py::test_gdn_stack_survives_replay
test/modular/test_gdn_mixed_split.py::test_a_mixed_steps_prompt_rows_save_the_snapshots_the_chunked_path_saves[bucket]
test/modular/test_gdn_mixed_split.py::test_a_mixed_steps_prompt_rows_save_the_snapshots_the_chunked_path_saves[eager]
test/modular/test_gdn_mixed_split.py::test_a_replay_of_another_layout_matches_the_chunked_path
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-decode_first-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-decode_only_slot0-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-decode_only_slot0-eager]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-interleaved-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-no_decode-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-slot0_in_run-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-slot0_in_run-eager]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-slot0_prompt-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-slot0_prompt-eager]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-decode_first-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-decode_first-eager]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-interleaved-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-interleaved-eager]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-no_decode-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-no_decode-eager]
test/modular/test_gdn_mixed_split.py::test_the_decode_halfs_width_changes_no_real_rows_result
test/modular/test_gdn_no_sink_slot.py::test_a_batch_of_nothing_but_padding_leaves_the_state_alone
test/modular/test_gdn_no_sink_slot.py::test_padding_rows_read_zeros_and_write_nowhere
test/modular/test_gdn_no_sink_slot.py::test_replay_takes_a_padding_pattern_the_capture_never_saw
test/modular/test_gdn_prefill_stage.py::test_chunked_kernel_agrees_through_either_path
test/modular/test_gdn_prefix_snapshots.py::test_a_captured_prefill_saves_the_state_an_eager_prefill_ending_at_its_boundary_leaves
test/modular/test_gdn_prefix_snapshots.py::test_a_prefill_past_the_largest_bucket_saves_the_state_at_its_widened_boundary
test/modular/test_gdn_prefix_snapshots.py::test_a_prefill_saves_the_state_a_prefill_ending_at_its_boundary_leaves[lengths0]
test/modular/test_gdn_prefix_snapshots.py::test_a_prefill_saves_the_state_a_prefill_ending_at_its_boundary_leaves[lengths1]
test/modular/test_glm52_dense_fp8.py::test_op_compiles_whole
test/modular/test_glm52_mla_prep.py::test_compiled_layer_with_fused_prep_is_the_unfused_layer_bit_for_bit
test/modular/test_glm52_prefill.py::test_router_kernel_matches_the_torch_top_k[1100]
test/modular/test_glm52_prefill.py::test_router_kernel_matches_the_torch_top_k[1]
test/modular/test_glm52_prefill.py::test_router_kernel_matches_the_torch_top_k[65]
test/modular/test_gpu_mla_attention.py::test_captured_plan_takes_rows_that_share_a_prefix
test/modular/test_kv_offload.py::test_oom_still_reports_allocation_failed
test/modular/test_kv_shm_dir_sweep.py::test_every_process_of_one_deployment_joins_one_directory
test/modular/test_kv_shm_dir_sweep.py::test_without_a_namespace_the_directory_is_named_and_swept_by_conductor_pid
test/modular/test_lora_targets.py::test_children_run_other_than_through_their_forward_are_refused[chatterbox_positions]
test/modular/test_model_kv_layouts.py::test_a_reference_clip_is_laid_out_at_the_tokens_the_tokenizer_makes_of_it[chatterbox]
test/modular/test_model_kv_layouts.py::test_a_reference_clip_is_laid_out_at_the_tokens_the_tokenizer_makes_of_it[turbo]
test/modular/test_model_kv_layouts.py::test_chatterbox_lays_out_every_label_it_declares[0.0-False]
test/modular/test_model_kv_layouts.py::test_chatterbox_lays_out_every_label_it_declares[0.5-False]
test/modular/test_model_kv_layouts.py::test_chatterbox_lays_out_every_label_it_declares[0.5-True]
test/modular/test_moe_router.py::test_router_compiles_with_the_fused_topk
test/modular/test_nemotron_duplex_model.py::test_duplex_load_audio_refuses_a_file_past_the_session
test/modular/test_node_fp32_precision_gpu.py::test_a_captured_default_node_keeps_tf32
test/modular/test_node_fp32_precision_gpu.py::test_a_captured_highest_node_matches_its_eager_fp32_forward
test/modular/test_node_fp32_precision_gpu.py::test_a_compiled_capture_of_a_highest_node_is_full_fp32
test/modular/test_node_fp32_precision_gpu.py::test_interleaved_replays_do_not_leak
test/modular/test_node_fp32_precision_gpu.py::test_tf32_is_live_on_this_gpu
test/modular/test_omnilingual_asr_model.py
test/modular/test_piecewise_config_signature.py
test/modular/test_qwen3_fused_qk_norm_rope.py::test_under_compile_from_one_projection[1]
test/modular/test_qwen3_fused_qk_norm_rope.py::test_under_compile_from_one_projection[64]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[False-1024-384-1]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[False-1024-384-4]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[False-2048-768-1]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[False-2048-768-4]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[True-1024-384-1]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[True-1024-384-4]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[True-2048-768-1]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[True-2048-768-4]
test/modular/test_qwen3_tts_model.py::test_qwen3_tts_load_audio_rejects_a_long_clip_before_decoding
test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_decodes_a_clip_without_a_sample_count[1]
test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_decodes_a_clip_without_a_sample_count[2]
test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_refuses_a_long_clip_without_a_sample_count_after_about_30_s
test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_refuses_a_short_clip_without_a_sample_count_at_its_length
test/modular/test_ragged_attention.py::test_a_compiled_region_attends_through_a_rebuilt_resource
test/modular/test_ragged_attention.py::test_callable_built_inside_a_compiled_region
test/modular/test_ragged_attention.py::test_compiled_caller_keeps_attention_in_its_graph[False]
test/modular/test_ragged_attention.py::test_compiled_caller_keeps_attention_in_its_graph[True]
test/modular/test_request_media_totals.py::test_a_second_clip_is_refused_before_either_is_decoded[mstar.model.chatterbox.chatterbox_model-ChatterboxModel]
test/modular/test_request_media_totals.py::test_one_ordinary_clip_still_loads[mstar.model.chatterbox.chatterbox_model-ChatterboxModel]
test/modular/test_resample_rate_bound.py
test/modular/test_rms_norm_op.py::test_fused_add_rms_norm_compiles_without_a_graph_break
test/modular/test_shm_shutdown_cleanup.py::test_a_worker_stalled_mid_cleanup_leaves_a_restarted_servers_files_alone
test/modular/test_shm_shutdown_cleanup.py::test_an_orphaned_worker_the_watchdog_exits_hard_still_unlinks_its_files
test/modular/test_shm_shutdown_cleanup.py::test_an_orphaned_worker_unlinks_its_files_only_when_no_restart_can_reuse_the_names[launch-token]
test/modular/test_shm_shutdown_cleanup.py::test_an_orphaned_worker_unlinks_its_files_only_when_no_restart_can_reuse_the_names[no-token]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_multilingual.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_multilingual_split.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_split.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_turbo.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_turbo_split.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[cosyvoice3.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_1b.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_300m.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_3b.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_7b.yaml]
test/modular/test_step_prep_host_syncs.py::test_wan22_forward_takes_no_host_sync[video_gen]
test/modular/test_step_prep_host_syncs.py::test_wan22_forward_takes_no_host_sync[video_gen_i2v]
test/modular/test_step_threads.py::test_the_lock_file_is_readable_by_other_users_whatever_the_umask
test/modular/test_tp_read_check.py::test_a_prepare_that_runs_a_collective_per_row_stays_in_step[grid]
test/modular/test_tp_read_check.py::test_a_prepare_that_runs_a_collective_per_row_stays_in_step[text_only]
test/modular/test_tp_read_check.py::test_a_step_whose_every_row_failed_runs_empty_on_every_rank
test/modular/test_tp_read_failure_acceptance.py::test_a_request_beside_the_failed_one_is_bit_identical_to_a_run_without_failure[grid]
test/modular/test_tp_read_failure_acceptance.py::test_a_request_beside_the_failed_one_is_bit_identical_to_a_run_without_failure[text_only]
test/modular/test_tp_read_failure_acceptance.py::test_a_run_waits_for_a_follower_that_routes_after_rank_0
test/modular/test_tp_read_failure_acceptance.py::test_nothing_of_the_failed_row_leaves_either_rank[text_only]
test/modular/test_tp_read_failure_acceptance.py::test_the_failed_rows_pages_never_become_leasable
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[1-rnnt]
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[1-tdt]
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[3-rnnt]
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[3-tdt]
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[64-rnnt]
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[64-tdt]
test/modular/test_transport_microbench.py::test_interference_reports_extra_p99
test/modular/test_transport_microbench.py::test_smoke_both_roles_on_localhost
test/modular/test_wan22_fast_decode.py::test_default_fast_path_is_stock_to_rounding[dtype0]
test/modular/test_wan22_fast_decode.py::test_default_fast_path_is_stock_to_rounding[dtype1]
test/modular/test_wan_vae_decode.py::test_compiled_channels_last_stream_cache_owns_its_storage[wan21]
test/modular/test_wan_vae_decode.py::test_compiled_channels_last_stream_cache_owns_its_storage[wan22]
test/modular/test_waypoint_components.py::test_a_fully_visible_ring_makes_eager_and_compiled_agree
test/modular/test_waypoint_components.py::test_compacted_and_full_global_rings_are_bitwise_identical
test/modular/test_waypoint_dit.py::test_a_materialized_dit_reads_the_noise_island_rows_instead_of_running_it
test/modular/test_waypoint_dit.py::test_append_frame_is_the_committing_pass_alone
test/modular/test_waypoint_dit.py::test_cond_head_cache_is_bit_exact_and_replaces_the_live_projection
test/modular/test_waypoint_dit.py::test_generate_frame_clones_the_denoised_latent
test/modular/test_waypoint_dit.py::test_generate_frame_is_four_frozen_denoise_passes_then_one_commit
test/modular/test_waypoint_dit.py::test_layer_zero_v_reaches_every_block_in_the_dit
test/modular/test_waypoint_dit.py::test_meta_built_model_generates_a_frame_in_the_serving_dtypes
test/modular/test_waypoint_dit.py::test_the_ring_only_moves_on_the_committing_pass
test/modular/test_waypoint_dit.py::test_two_frames_advance_the_ring_clock_together
test/modular/test_waypoint_gpu.py::test_a_second_rollout_starts_from_nothing
test/modular/test_waypoint_gpu.py::test_batched_step_matches_the_same_rollouts_run_one_row_at_a_time
test/modular/test_waypoint_gpu.py::test_capture_replays_fixed_address_planned_masks
test/modular/test_waypoint_gpu.py::test_compile_regions_holds_under_fullgraph
test/modular/test_waypoint_gpu.py::test_compiled_matches_eager_to_four_bf16_ulp_of_peak
test/modular/test_waypoint_gpu.py::test_flex_attention_is_bit_exact_in_and_out_of_a_compiled_region
test/modular/test_waypoint_gpu.py::test_prime_replay_matches_the_uncaptured_prime
test/modular/test_waypoint_gpu.py::test_prime_then_rollout_through_both_graphs_matches_eager
test/modular/test_waypoint_gpu.py::test_replay_matches_the_uncaptured_regions_over_two_ring_wraps
test/modular/test_waypoint_gpu.py::test_the_gemms_are_bit_exact_compiled_vs_eager
test/modular/test_waypoint_gpu.py::test_the_prime_graph_returns_its_own_static_input_buffer
test/modular/test_waypoint_gpu.py::test_two_worlds_interleaved_match_the_same_rollouts_run_alone
test/modular/test_waypoint_shell.py::test_fused_decode_histories_are_isolated_interleaved_and_cleaned_up
test/modular/test_waypoint_shell.py::test_prime_and_rollout_use_the_same_fixed_tensor_histories
test/modular/test_waypoint_shell.py::test_the_encoder_and_the_fused_decode_share_only_weights
test/modular/test_waypoint_shell.py::test_the_fused_decode_turns_one_frame_into_one_raw_clip
test/modular/test_whisper_model.py::test_load_audio_file_decodes_and_resamples
```
- Targeted at head (54 `test/modular` files that import a touched module, `-n 8 --dist loadfile`): 16 failed, 1689 passed, 50 skipped, 3 errors; all 19 ids are in the parallel baseline.
- `ruff check .`: clean at each commit.

<details><summary>Pre-existing failing ids (serial baseline)</summary>

```
test/modular/test_accelerator_graph_backend.py::test_chatterbox_shape_graphs_keep_results_across_replays[cuda]
test/modular/test_audio_decode_limit.py::test_a_soundfile_rate_past_the_range_is_refused_before_any_read
test/modular/test_audio_decode_limit.py::test_omnilingual_asr_decode_keeps_a_clip_under_its_bound
test/modular/test_audio_decode_limit.py::test_omnilingual_asr_decode_stops_at_its_clip_bound
test/modular/test_audio_decode_limit.py::test_omnilingual_asr_refuses_a_rate_past_the_range_before_reading
test/modular/test_audio_decode_limit.py::test_qwen3_tts_reference_decode_stops_at_its_bound
test/modular/test_audio_decode_limit.py::test_soundfile_loaders_are_capped
test/modular/test_audio_features.py::test_load_audio_file_decodes_and_resamples
test/modular/test_audio_watermark.py::test_detect_resamples_to_the_rate_the_mark_was_added_at[16000-24000]
test/modular/test_audio_watermark.py::test_detect_resamples_to_the_rate_the_mark_was_added_at[48000-24000]
test/modular/test_audio_watermark.py::test_engine_captures_replay_the_eager_watermark_bit_for_bit
test/modular/test_bagel_cfg_parallel.py::test_kv_shm_directory_is_private_to_the_deployment
test/modular/test_base_load_audio.py::test_a_crafted_header_rate_is_refused_before_resampling[100003]
test/modular/test_base_load_audio.py::test_a_crafted_header_rate_is_refused_before_resampling[48001]
test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-flac-PCM_16]
test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-mp3-MPEG_LAYER_III]
test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-ogg-VORBIS]
test/modular/test_base_load_audio.py::test_common_containers_decode_without_ffmpeg[24000-2-wav-PCM_16]
test/modular/test_blackwell_ptxas.py::test_compiled_interleaved_mrope_matches_eager[small table]
test/modular/test_blackwell_ptxas.py::test_floor_division_under_a_range_select[1024-16]
test/modular/test_blackwell_ptxas.py::test_floor_division_under_a_range_select[256-4]
test/modular/test_blackwell_ptxas.py::test_floor_division_under_a_range_select[64-1]
test/modular/test_chatterbox_model.py
test/modular/test_chatterbox_watermark_import.py
test/modular/test_client_error_text.py::test_a_transcription_whose_decode_fails_in_the_frontend_answers_with_an_error_id
test/modular/test_cosyvoice3_flow_capture.py
test/modular/test_cosyvoice3_model.py
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_compiles_with_a_float_eps[plain]
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_compiles_with_a_float_eps[residual]
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_matches_eager[gated_affine]
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_matches_eager[mod]
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_matches_eager[per_sample]
test/modular/test_diffusion_dit_kernels.py::test_ada_layer_norm_kernel_matches_eager[plain_mod]
test/modular/test_diffusion_dit_kernels.py::test_gelu_epilogue_matches_eager
test/modular/test_diffusion_dit_kernels.py::test_rms_norm_rope_kernel_matches_eager
test/modular/test_eval_compare_multigpu.py::test_a_head_that_hangs_is_cut_at_the_cap_with_stacks_and_fails
test/modular/test_eval_framework.py::test_sigterm_to_the_runner_stops_the_managed_server
test/modular/test_fish_s2_pro_codec_stream.py::test_compiled_incremental_decode_matches_eager_on_cuda
test/modular/test_fish_s2_pro_model.py::test_load_audio_decodes_a_clip_without_a_sample_count[1]
test/modular/test_fish_s2_pro_model.py::test_load_audio_decodes_a_clip_without_a_sample_count[2]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_clip_without_a_sample_count_at_its_decoded_length[0.3-0\\.\\d\\d]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_clip_without_a_sample_count_at_its_decoded_length[45-30\\.\\d\\d]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[16000-False]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[44100-False]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[44101-True]
test/modular/test_fish_s2_pro_model.py::test_load_audio_refuses_a_rate_whose_resampling_kernel_is_unbounded[48000-False]
test/modular/test_flex_attention_resource.py::test_a_batched_step_can_share_one_ring_by_stride_0_expand[cpu]
test/modular/test_flex_attention_resource.py::test_a_batched_step_can_share_one_ring_by_stride_0_expand[cuda]
test/modular/test_flex_attention_resource.py::test_a_batched_step_does_not_reach_across_world_spans[cpu]
test/modular/test_flex_attention_resource.py::test_a_batched_step_does_not_reach_across_world_spans[cuda]
test/modular/test_flex_attention_resource.py::test_attend_matches_a_masked_dense_reference[gappy_ring-committed_blocks1]
test/modular/test_flex_attention_resource.py::test_attend_matches_a_masked_dense_reference[one_frame-committed_blocks0]
test/modular/test_flex_attention_resource.py::test_attend_takes_the_compiled_path
test/modular/test_flex_attention_resource.py::test_eager_flex_attention_does_not_honour_the_block_mask
test/modular/test_fused_moe_config.py::test_compiled_fused_experts_takes_the_tables_tiles[captured-cuda_align]
test/modular/test_fused_moe_config.py::test_compiled_fused_experts_takes_the_tables_tiles[captured-torch_align]
test/modular/test_fused_moe_config.py::test_compiled_fused_experts_takes_the_tables_tiles[eager_forward-cuda_align]
test/modular/test_fused_moe_config.py::test_compiled_fused_experts_takes_the_tables_tiles[eager_forward-torch_align]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_B300_SXM6_AC-0]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_B300_SXM6_AC-1]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_GB200-0]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_H100_80GB_HBM3-0]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_H100_80GB_HBM3-1]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_H100_80GB_HBM3-2]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_H200-0]
test/modular/test_fused_moe_config.py::test_fused_experts_traces_with_a_symbolic_batch_on_every_table[NVIDIA_H200-1]
test/modular/test_fused_moe_prefill.py::test_compiles_on_a_dynamic_token_count
test/modular/test_fused_moe_route.py::test_the_one_kernel_route_runs_under_a_symbolic_token_count[3]
test/modular/test_fused_moe_route.py::test_the_one_kernel_route_runs_under_a_symbolic_token_count[5]
test/modular/test_gdn_cuda_graph.py::test_gdn_stack_survives_replay
test/modular/test_gdn_mixed_split.py::test_a_mixed_steps_prompt_rows_save_the_snapshots_the_chunked_path_saves[bucket]
test/modular/test_gdn_mixed_split.py::test_a_mixed_steps_prompt_rows_save_the_snapshots_the_chunked_path_saves[eager]
test/modular/test_gdn_mixed_split.py::test_a_replay_of_another_layout_matches_the_chunked_path
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-decode_first-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-decode_only_slot0-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-decode_only_slot0-eager]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-interleaved-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-no_decode-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-slot0_in_run-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-slot0_in_run-eager]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-slot0_prompt-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[no_sink-slot0_prompt-eager]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-decode_first-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-decode_first-eager]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-interleaved-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-interleaved-eager]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-no_decode-bucket]
test/modular/test_gdn_mixed_split.py::test_mixed_split_matches_the_chunked_path[sink-no_decode-eager]
test/modular/test_gdn_mixed_split.py::test_the_decode_halfs_width_changes_no_real_rows_result
test/modular/test_gdn_no_sink_slot.py::test_a_batch_of_nothing_but_padding_leaves_the_state_alone
test/modular/test_gdn_no_sink_slot.py::test_padding_rows_read_zeros_and_write_nowhere
test/modular/test_gdn_no_sink_slot.py::test_replay_takes_a_padding_pattern_the_capture_never_saw
test/modular/test_gdn_prefill_stage.py::test_chunked_kernel_agrees_through_either_path
test/modular/test_gdn_prefix_snapshots.py::test_a_captured_prefill_saves_the_state_an_eager_prefill_ending_at_its_boundary_leaves
test/modular/test_gdn_prefix_snapshots.py::test_a_prefill_past_the_largest_bucket_saves_the_state_at_its_widened_boundary
test/modular/test_gdn_prefix_snapshots.py::test_a_prefill_saves_the_state_a_prefill_ending_at_its_boundary_leaves[lengths0]
test/modular/test_gdn_prefix_snapshots.py::test_a_prefill_saves_the_state_a_prefill_ending_at_its_boundary_leaves[lengths1]
test/modular/test_glm52_dense_fp8.py::test_op_compiles_whole
test/modular/test_glm52_mla_prep.py::test_compiled_layer_with_fused_prep_is_the_unfused_layer_bit_for_bit
test/modular/test_glm52_prefill.py::test_router_kernel_matches_the_torch_top_k[1100]
test/modular/test_glm52_prefill.py::test_router_kernel_matches_the_torch_top_k[1]
test/modular/test_glm52_prefill.py::test_router_kernel_matches_the_torch_top_k[65]
test/modular/test_gpu_mla_attention.py::test_captured_plan_takes_rows_that_share_a_prefix
test/modular/test_kv_offload.py::test_oom_still_reports_allocation_failed
test/modular/test_kv_shm_dir_sweep.py::test_every_process_of_one_deployment_joins_one_directory
test/modular/test_kv_shm_dir_sweep.py::test_without_a_namespace_the_directory_is_named_and_swept_by_conductor_pid
test/modular/test_lora_targets.py::test_children_run_other_than_through_their_forward_are_refused[chatterbox_positions]
test/modular/test_model_kv_layouts.py::test_a_reference_clip_is_laid_out_at_the_tokens_the_tokenizer_makes_of_it[chatterbox]
test/modular/test_model_kv_layouts.py::test_a_reference_clip_is_laid_out_at_the_tokens_the_tokenizer_makes_of_it[turbo]
test/modular/test_model_kv_layouts.py::test_chatterbox_lays_out_every_label_it_declares[0.0-False]
test/modular/test_model_kv_layouts.py::test_chatterbox_lays_out_every_label_it_declares[0.5-False]
test/modular/test_model_kv_layouts.py::test_chatterbox_lays_out_every_label_it_declares[0.5-True]
test/modular/test_moe_router.py::test_router_compiles_with_the_fused_topk
test/modular/test_nemotron_duplex_model.py::test_duplex_load_audio_refuses_a_file_past_the_session
test/modular/test_node_fp32_precision_gpu.py::test_a_captured_default_node_keeps_tf32
test/modular/test_node_fp32_precision_gpu.py::test_a_captured_highest_node_matches_its_eager_fp32_forward
test/modular/test_node_fp32_precision_gpu.py::test_a_compiled_capture_of_a_highest_node_is_full_fp32
test/modular/test_node_fp32_precision_gpu.py::test_interleaved_replays_do_not_leak
test/modular/test_node_fp32_precision_gpu.py::test_tf32_is_live_on_this_gpu
test/modular/test_omnilingual_asr_model.py
test/modular/test_piecewise_config_signature.py
test/modular/test_qwen3_fused_qk_norm_rope.py::test_under_compile_from_one_projection[1]
test/modular/test_qwen3_fused_qk_norm_rope.py::test_under_compile_from_one_projection[64]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[False-1024-384-1]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[False-1024-384-4]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[False-2048-768-1]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[False-2048-768-4]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[True-1024-384-1]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[True-1024-384-4]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[True-2048-768-1]
test/modular/test_qwen3_omni_fused_moe.py::test_fused_experts_compiled_matches_eager[True-2048-768-4]
test/modular/test_qwen3_tts_model.py::test_qwen3_tts_load_audio_rejects_a_long_clip_before_decoding
test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_decodes_a_clip_without_a_sample_count[1]
test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_decodes_a_clip_without_a_sample_count[2]
test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_refuses_a_long_clip_without_a_sample_count_after_about_30_s
test/modular/test_qwen3_tts_reference_audio.py::test_load_audio_refuses_a_short_clip_without_a_sample_count_at_its_length
test/modular/test_ragged_attention.py::test_a_compiled_region_attends_through_a_rebuilt_resource
test/modular/test_ragged_attention.py::test_callable_built_inside_a_compiled_region
test/modular/test_ragged_attention.py::test_compiled_caller_keeps_attention_in_its_graph[False]
test/modular/test_ragged_attention.py::test_compiled_caller_keeps_attention_in_its_graph[True]
test/modular/test_request_media_totals.py::test_a_second_clip_is_refused_before_either_is_decoded[mstar.model.chatterbox.chatterbox_model-ChatterboxModel]
test/modular/test_request_media_totals.py::test_one_ordinary_clip_still_loads[mstar.model.chatterbox.chatterbox_model-ChatterboxModel]
test/modular/test_resample_rate_bound.py
test/modular/test_rms_norm_op.py::test_fused_add_rms_norm_compiles_without_a_graph_break
test/modular/test_shm_shutdown_cleanup.py::test_a_worker_stalled_mid_cleanup_leaves_a_restarted_servers_files_alone
test/modular/test_shm_shutdown_cleanup.py::test_an_orphaned_worker_the_watchdog_exits_hard_still_unlinks_its_files
test/modular/test_shm_shutdown_cleanup.py::test_an_orphaned_worker_unlinks_its_files_only_when_no_restart_can_reuse_the_names[launch-token]
test/modular/test_shm_shutdown_cleanup.py::test_an_orphaned_worker_unlinks_its_files_only_when_no_restart_can_reuse_the_names[no-token]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_multilingual.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_multilingual_split.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_split.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_turbo.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[chatterbox_turbo_split.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[cosyvoice3.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_1b.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_300m.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_3b.yaml]
test/modular/test_startup_graph_validation.py::test_shipped_config_passes_startup_validation[omnilingual_asr_ctc_7b.yaml]
test/modular/test_step_prep_host_syncs.py::test_wan22_forward_takes_no_host_sync[video_gen]
test/modular/test_step_prep_host_syncs.py::test_wan22_forward_takes_no_host_sync[video_gen_i2v]
test/modular/test_step_threads.py::test_the_lock_file_is_readable_by_other_users_whatever_the_umask
test/modular/test_tp_read_check.py::test_a_prepare_that_runs_a_collective_per_row_stays_in_step[grid]
test/modular/test_tp_read_check.py::test_a_prepare_that_runs_a_collective_per_row_stays_in_step[text_only]
test/modular/test_tp_read_check.py::test_a_step_whose_every_row_failed_runs_empty_on_every_rank
test/modular/test_tp_read_failure_acceptance.py::test_a_request_beside_the_failed_one_is_bit_identical_to_a_run_without_failure[grid]
test/modular/test_tp_read_failure_acceptance.py::test_a_request_beside_the_failed_one_is_bit_identical_to_a_run_without_failure[text_only]
test/modular/test_tp_read_failure_acceptance.py::test_a_run_waits_for_a_follower_that_routes_after_rank_0
test/modular/test_tp_read_failure_acceptance.py::test_nothing_of_the_failed_row_leaves_either_rank[text_only]
test/modular/test_tp_read_failure_acceptance.py::test_the_failed_rows_pages_never_become_leasable
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[1-rnnt]
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[1-tdt]
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[3-rnnt]
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[3-tdt]
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[64-rnnt]
test/modular/test_transducer_loop.py::test_captured_iterations_match_eager[64-tdt]
test/modular/test_transport_microbench.py::test_interference_reports_extra_p99
test/modular/test_transport_microbench.py::test_smoke_both_roles_on_localhost
test/modular/test_wan22_fast_decode.py::test_default_fast_path_is_stock_to_rounding[dtype0]
test/modular/test_wan22_fast_decode.py::test_default_fast_path_is_stock_to_rounding[dtype1]
test/modular/test_wan_vae_decode.py::test_compiled_channels_last_stream_cache_owns_its_storage[wan21]
test/modular/test_wan_vae_decode.py::test_compiled_channels_last_stream_cache_owns_its_storage[wan22]
test/modular/test_waypoint_components.py::test_a_fully_visible_ring_makes_eager_and_compiled_agree
test/modular/test_waypoint_components.py::test_compacted_and_full_global_rings_are_bitwise_identical
test/modular/test_waypoint_dit.py::test_a_materialized_dit_reads_the_noise_island_rows_instead_of_running_it
test/modular/test_waypoint_dit.py::test_append_frame_is_the_committing_pass_alone
test/modular/test_waypoint_dit.py::test_cond_head_cache_is_bit_exact_and_replaces_the_live_projection
test/modular/test_waypoint_dit.py::test_generate_frame_clones_the_denoised_latent
test/modular/test_waypoint_dit.py::test_generate_frame_is_four_frozen_denoise_passes_then_one_commit
test/modular/test_waypoint_dit.py::test_layer_zero_v_reaches_every_block_in_the_dit
test/modular/test_waypoint_dit.py::test_meta_built_model_generates_a_frame_in_the_serving_dtypes
test/modular/test_waypoint_dit.py::test_the_ring_only_moves_on_the_committing_pass
test/modular/test_waypoint_dit.py::test_two_frames_advance_the_ring_clock_together
test/modular/test_waypoint_gpu.py::test_a_second_rollout_starts_from_nothing
test/modular/test_waypoint_gpu.py::test_batched_step_matches_the_same_rollouts_run_one_row_at_a_time
test/modular/test_waypoint_gpu.py::test_capture_replays_fixed_address_planned_masks
test/modular/test_waypoint_gpu.py::test_compile_regions_holds_under_fullgraph
test/modular/test_waypoint_gpu.py::test_compiled_matches_eager_to_four_bf16_ulp_of_peak
test/modular/test_waypoint_gpu.py::test_flex_attention_is_bit_exact_in_and_out_of_a_compiled_region
test/modular/test_waypoint_gpu.py::test_prime_replay_matches_the_uncaptured_prime
test/modular/test_waypoint_gpu.py::test_prime_then_rollout_through_both_graphs_matches_eager
test/modular/test_waypoint_gpu.py::test_replay_matches_the_uncaptured_regions_over_two_ring_wraps
test/modular/test_waypoint_gpu.py::test_the_gemms_are_bit_exact_compiled_vs_eager
test/modular/test_waypoint_gpu.py::test_the_prime_graph_returns_its_own_static_input_buffer
test/modular/test_waypoint_gpu.py::test_two_worlds_interleaved_match_the_same_rollouts_run_alone
test/modular/test_waypoint_shell.py::test_fused_decode_histories_are_isolated_interleaved_and_cleaned_up
test/modular/test_waypoint_shell.py::test_prime_and_rollout_use_the_same_fixed_tensor_histories
test/modular/test_waypoint_shell.py::test_the_encoder_and_the_fused_decode_share_only_weights
test/modular/test_waypoint_shell.py::test_the_fused_decode_turns_one_frame_into_one_raw_clip
test/modular/test_whisper_model.py::test_load_audio_file_decodes_and_resamples
```
</details>

## Review
An independent subagent read `git diff 305e80ef4..HEAD` and the checkout. It found no correctness bug. It did not run the tests. It takes kernel ABI facts from memory; the thread readback test and `uc_probe` (below) check them here.

Its answers, by piece:
- `cpus.run_at_full_frequency`: belongs beside `hold_idle`, a process-wide CPU request with no engine or worker state. Couples to the kernel ABI, `/proc` and sysfs. Possible miss if a thread not yet clamped starts one after the listing; it found no such thread at this point of startup and would not change it.
- `knobs._full_cpu_frequency_value`: justified; without it 1024 passes the generic integer branch and reads as off.
- `entrypoint.apply_cpu_settings`: belongs there; removes a duplicated try/except. Couples to `deployment_cpu_plan`'s `knobs.configure` side effect, which the comment states.
- Dynamo call site: fine, but its comment was already wrong about `auto`.
- Tests: the real-kernel test is the most valuable; the rest test one behaviour each.

Changed after it:
- Docs: the knob is now the third bullet under "Three knobs deal with it" in both serving docs, and the separate paragraph is gone. Env rows lost "it is a host setting" and the sentence about measurements, and gained a measured number (the table above) in place of "far below".
- Stated why a refusal is fatal, in the env rows and the docstring.
- Fixed the stale comments at both call sites and reworded "wrapped".
- Test nits: `_Libc` docstring, fixture name, helper order. The entry-points test no longer leaves `quiet_noisy_loggers` or the API key env change behind.

Left as they are:
- Its top item, a measurement, is the table above and belongs in the PR description, not the diff.
- `CpuPlanError` as the name for a frequency refusal: reusing it keeps one `except` at both call sites.
- `apply_cpu_settings` doesn't include `hold_idle`, and dynamo never applies `idle_hold`: that gap predates this PR.
- `full_cpu_frequency: 1` is accepted alongside `true`: same for the env var's format, and harmless.

Permission check (`uc_probe.py`): the `sched_setattr` call succeeds as uid 65534 with no capabilities (CapEff 0), which is what the docs' "no capability otherwise" rests on.

## Open
- **Two clamped states at ShareGPT c=64 on this host.** Knob on and `uclampset` both land at about 16k tok/s or about 11.8k tok/s from round to round, and the low rounds read 2000 MHz on the step cores. Main without a clamp stays at 10.5-11.4k. Why the clock tops out at 2000 in some rounds is not known. Hypothesis, unverified: a host-level limit such as turbo budget under other tenants on this shared machine. An earlier guess that the longer cell list caused the low rounds was tested in `q08b`, and the low state also showed up on the short list, so it is dropped.
- **Not measured:** power, any governor other than `schedutil`, step_thread_core other than `siblings`, pi0.5 (weights not on this pod).
- **Whether every thread needs the clamp, or only the step threads.** rp2's step-thread version isn't compared here; the brief asks for one knob. rp2's numbers for that form are in `444a2399`.
