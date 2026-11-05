# rp1-pi05 proposals (changes that need the core)

## P1. Decode a request's image files concurrently in the API data worker

- **Time it would remove (measured):** pi0.5 DROID requests carry 3 camera PNGs (320x180). `Pi05Model.load_image` -> `mstar.model.media_limits.read_image` takes 2.3 ms per image on a quiet CPU (rp1-pi05 decode microbench: `read_image` 2.30 ms, `torchvision.io.decode_image(path)` 1.54 ms, header check 0.47 ms), and the data worker calls it once per file in sequence (`mstar/api_server/data_worker.py:_load_media`, the `for filepath in paths:` loop at ~line 822 at e8826a777). Per-request stats (`--log-stats-file`, logs/prof_head_c1/stats.txt) put `recv -> preprocess done` at 6-10 ms on a quiet host, of which ~7 ms is the three decodes. Decoding them concurrently (libpng releases the GIL in torchvision's decoder) would bring that to ~2.5 ms: about -4.5 ms of an ~50-80 ms c=1 action-chunk latency (6-9%).
- **Missing primitive:** `_load_media` has no way to load a request's files in parallel; the model hook `load_image(filepath, device)` is per file. Extending `_load_media` to submit a request's files of one modality to a small thread pool (the prepare pool already exists for whole requests) keeps the hook unchanged for every model.
- **Evidence still to collect:** an A/B of the concurrent loop on the data worker with the per-request stats; check the `RequestMediaBudget` accounting holds per file under concurrency.

- **Update (2026-11-03, stacked tree e9502f482+H5+H9+H6, logs/prof_h6_c1, c=1, 80 requests, --log-stats-file):** `recv -> preprocess done` p50 23.7 ms of a 67.2 ms total (35%), on a host with other sessions' load (load avg 6-29). On a quiet host the same stage was 6-10 ms (logs/prof_head_c1), so it is also the stage most exposed to host contention. The decode loop is the part identified so far; the rest of the stage (multipart parse, upload file writes, SHM registration 0.87 ms for the three images) is not broken down by the existing per-request stats.

## P2. Host time between conductor and worker per request

- **Measured (same run):** `conductor ingest -> first chunk` 39.4 ms vs `worker span` 33.0 ms -> ~6.4 ms in conductor/worker hand-offs per request; inside the worker, graph step times sum to 28.3 ms (vit 4.7, prefill 10.1, action_gen 13.5 incl. host prepare) vs the 33.0 ms span -> ~4.7 ms between steps (ViT -> LLM hand-off and the prefill -> action_gen walk switch through the conductor). GPU work per request is 21.1 ms (CUDA events around graph replays: action 10.4, prefill 7.3, ViT 3.4).
- **Per-step host prepare in the worker:** LLM prefill `pre` 2.19 ms and action_gen `pre` 1.85 ms per step, of which the model's own prepare_inputs/preprocess tensor work is < 0.1 ms (microbench: embedding+scale+concat 46 us; generator+randn 8.5 us) -> the rest is engine prepare/plan/admit.
- **Not yet a proposal:** needs per-hop timestamps (conductor send/receive, worker message receipt) to say which primitive to change; recorded as measured overhead only.

> **Correction (2026-11-04): P3 as written below is wrong on size.** Its hop numbers came from a tracer doing file I/O in each hook, before H20. Clean trace at 6eba3b986 (logs/trace_clean1, buffered tracer, no stats): conductor switch -> action prep is 0.39 ms; all worker+conductor host time per request is ~4.5 ms. The "~4.9 ms" and "~10%" below are not supported.

## P3. Let a model chain its walks without a conductor round trip (pi0.5 prefill -> action_gen)

- **Time it would remove (measured, 2026-11-04, best ec5e8627f, c=1, 83 requests, harness-only hook tracer `RP1_TRACE_DIR` in rp1-pi05-tools/pyshim/sitecustomize.py, monotonic clock shared across processes; logs/trace_best_c1):**
  - worker LLM prefill prepare -> conductor `get_partition_forward_pass_args(prefill)`: p50 10.36 ms, of which the captured prefill graph is 7.26 ms GPU (CUDA events around replay) -> ~3.1 ms of prefill launch/postprocess/report.
  - conductor switch -> worker action_gen prepare: p50 1.83 ms.
  - So ~4.9 ms per request between the prefill's GPU work ending and the action step starting, with the GPU idle; at c=1 that is ~10% of a ~45-50 ms chunk on a quiet host. (For contrast, ViT -> LLM prefill on the same walk: 5.69 ms of which 3.2 ms ViT GPU.)
- **Why it exists:** pi0.5's action_gen depends only on the prefill having committed the KV; `Pi05Model.get_partition_forward_pass_args` (pi05_model.py) unconditionally maps prefill -> action_gen with fixed input edges, but the transition is decided in the conductor after `WORKER_GRAPHS_DONE`, so the worker cannot speculate or pre-plan the action step while prefill runs (async-worker skill: speculation follows graph edges within a walk).
- **Missing primitive (either):**
  1. A node-to-node edge across walks the worker can follow locally when the model declares the next walk is unconditional (a "chained walk" declaration checked at startup), so the action step is speculated like an in-walk successor; or
  2. A KV/attention/position resource shared by two nodes on one worker, so pi0.5 can be one walk `[vit_encoder, paligemma, action_expert]` (as in pi05-performance-2's `692d43249`, which needed engine/worker edits). Today every KVSpec binds a single node (grep of `KVSpec(` under mstar/model).
- **Evidence still to collect:** an engine-side prototype measured with the paired harness (rp1-pi05-tools/pair_round.sh).

## P4. Pre-plan a packed step whose row lengths the request layout already fixes (pi0.5 ViT -> prefill hand-off)

- **Time it would remove (measured, same trace):** ViT prepare -> LLM prefill prepare p50 5.69 ms against 3.21 ms of ViT GPU; the prefill step's host `pre` (prepare + plan + admit) is 2.19 ms per request (per-request stats) and starts only after the ViT step's outputs are post-processed, so ~2-2.5 ms per request runs with the GPU idle at c=1.
- **Why:** the prefill walk's second node is a packed capture; `select_batched_bucket` (accelerator_graph_runner.py ~line 1341 at e8826a777) only answers for BASIC_BATCHED captures, so pre-plan for the speculated LLM prefill step cannot run before `prepare_inputs` has built the inputs, which needs the ViT outputs. But pi0.5's prefix length per request is fixed at ingest by its `LabelLayout` (`Pi05Model.process_prompt`: 768 image rows + the prompt's token count), so the bucket and the attention/KV/position plan are knowable before the ViT finishes.
- **Missing primitive:** a way for a packed step to take its per-row token counts from the request's declared layout (or a submodule hook returning them without inputs), letting the plan thread pre-plan it while the previous node's graph runs.
- **Model-lane alternative considered and rejected (H17):** folding SigLIP into the captured prefill step needs a scatter of each request's 768 image tokens to packed offsets that vary per batch; `_stage` copies only the real prefix of each static input and leaves stale tail slots (accelerator_graph_runner.py `_stage` docstring), so padding rows' stale indices could overwrite real tokens. Not safe without engine support.

## Status update (2026-11-04, amendment: this file is now the list of candidate rounds)

- **P1: done.** Kept as 0d887ae5e and 833bc8acb: media 5.34 -> 3.97 ms, then a larger pool. c=1 about -2.6 ms over both rounds.
- **P2: measured, no candidate left at this resolution.** Hops at aeda84155 (logs/trace_clean3): send -> conductor ingest 0.77, ingest 0.26, ingest end -> ViT prepare 0.89. Main's #1074 (H21) did not pass the keep rule.
- **P3, one walk (the smaller of its two options): not built.** Verified: shared KV/attention/position across nodes exists (`KVSpec(nodes=..., leader="LLM")`, mstar/model/bagel/bagel_model.py:934), and one walk would route the transition locally. Blockers: the action node needs a real edge from prefill, but `_forward_prefill_batched` (submodules.py) returns no outputs and a packed capture has no per-request output at dynamic offsets, so this needs engine work. Same node twice in a walk is impossible (nodes keyed by name, mstar/graph/base.py:361-397). Open risk, unverified: speculating into a node that shares KV with an uncommitted step may plan against the wrong stored length, which corrupts output rather than failing. Estimated bound from trace_clean3 (estimate, not measured): about 2 ms with speculation, about 0.7 ms without. Prior attempt: origin/pi05-performance-2 692d43249, not in run-base, on an older engine.
- **P4: owned by rp3** under the amendment's split (pre-planning). Not taken here.
- **Resolution limit:** c=1 rounds now spread about +-1 ms between servers, so under about 1.5 ms is unlikely to pass the keep rule. H21, H22 and H25 fell under this; H11 (0.3-0.7 ms) and H12 (under 1 ms) are expected to as well, and are not queued.

