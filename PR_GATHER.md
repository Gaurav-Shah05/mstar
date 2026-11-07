IN PROGRESS: full test run at head; per-commit checks

# PR facts: gather policy (worker)

- Base: 305e80ef46248ade6e6670aa20c9bb67f4ac2e26 (upstream main)
- Branch: perf/rp4-pr-gather
- Head: 803c6682e88cc5770930b7a89f17013c045d6820
- Stacked under perf/rp4-pr-omni (PR_OMNI.md), where Qwen3-Omni's audio encoder declares the first policy. On this branch no model declares one.

## Commits (in order)

| # | Commit | Subject | Source |
|---|---|---|---|
| 1 | c80be2f34 | worker: let a node's ready rows gather before they take a step | rp3 f35f5c34 without its Qwen3-Omni lines, plus the fake-engine lines of rp3 9a228e98, the review fixes below, Mintlify docs |
| 2 | 803c6682e | test: gather policy holds rows only while busy, a chain's peek starts and ends the wait, a node without one steps as before | rp3 9a228e98's test_gather_policy.py, rewritten for the fixes, plus new scheduler-driven tests |

Ported/changed against rp3's commits:
- The scheduler fakes in test_loop_turns, test_micro_scheduler, test_resident_delta, test_scheduler_room and test_tp_follow_targeted_gate answer `get_gather_policy` in the code commit, so it leaves the tree green. Conflicts with main in test_loop_turns.py and test_tp_follow_targeted_gate.py were adjacent additions (main's `batch_cap_varies` fake): both kept.
- Review fix (item 2 of the Nov 7 amendment): rp3's version, and the first PR draft (82ad7e69), let the chain's peek read the wait without starting it (`_gathering_since.get(key, now)`), so with no wait started the peek always said "gathering". Only `get_next_batch` started the clock, and a speculating chain reaches it only when the peek finds other work or after MSTAR_MAX_CONSECUTIVE_SPEC_STEPS (1024) steps. Now any call that holds rows starts the wait, and any call that would not hold them (or `note_ran`) ends it.
- Docs (docs/adding_models.rst, docs/mintlify/adding_models.mdx): one row; says a prompt with K such items waits once per item; the example names no model.
- Messages carry no trailers; the code commit's body is cut to the why.

## Concept

GatherPolicy: a node's ready rows can gather before they take a step.
- Changes: mstar/model/submodule_base.py (`GatherPolicy(rows, wait_s, busy_requests)`, `NodeSubmodule.gather_policy` returning None), mstar/engine/engine.py (`Engine.get_gather_policy`), mstar/worker/micro_scheduler.py (`_gathering`; the ready filter in `get_next_batch`; the chain's peek in `has_ready_excluding`, which skips a followed TP head since rank 0 decides it; `note_ran` ends a wait), the two docs pages.
- Mechanism: a cheap node served as soon as one row is ready sends its consumer a row at a time, and each row costs the consumer a step. With a policy, while the worker holds at least `busy_requests` requests, the walk's ready rows wait until `rows` are ready or `wait_s` has passed since they first could have run. The policy is asked once per (node, walk) and cached; a node without one returns before any clock read. Only leader-scheduled nodes consult it; TP followers replay rank 0's steps.
- Own effect: none by itself (no model declares a policy here). With Qwen3-Omni's audio encoder declaring one, see PR_OMNI.md.
- Cost on a model without a policy: none measured (below).
- Tests:
  - test/modular/test_gather_policy.py: rows wait until the policy lapses; enough rows run at once; a quiet worker does not wait; a node without a policy runs at once and is asked once; the rows after a step wait from their own start; a wait the load no longer calls for ends.
  - test/modular/test_loop_turns.py::test_a_chains_peek_starts_the_wait_and_yields_once_it_lapses (the real `has_ready_excluding` holds a row, then reports it once the wait lapses, and `get_next_batch` from the chain's yield returns it; fails on the 82ad7e69 version), ::test_the_scheduler_holds_a_row_until_its_wait_lapses, ::test_a_node_without_a_gather_policy_steps_as_one_whose_policy_never_applies[TURNS|ARRIVAL] (24 requests: the no-policy node is asked and never holds; it takes the same steps as a node whose policy never applies; the same node under a policy that applies is held, so the comparison can fail).

## Item 3 of the amendment (step count instead of wall clock): measured, not kept

Built as scratch commit 129a48868 on top of the omni head: `GatherPolicy(rows, steps)`, the wait counted in steps admitted on this worker (`note_ran`), "busy" meaning the chain's peek or `get_next_batch(chain_in_flight=True)` from the worker's yield-away call (worker.py, where step N is still in flight); with nothing in flight the rows run at once. Qwen3-Omni declared rows=8 (the encoder's batch cap), steps=2 (about the 30 ms the wall-clock window allowed at the Thinker's 13-20 ms steps at A2T c=32; the encoder's own batch time was not measured, so this is calibration to the old window, not a derived rule).
Same A/B as PR_OMNI.md (3 alternating rounds, fresh server each):

| A2T c=32 | wall clock (head) | step count | main |
|---|---|---|---|
| tok/s | 1401.9 [1393.2-1432.6] | 1368.9 [1355.9-1376.1] | 1226.3 [1209.2-1227.2] |
| TTFT p50 ms | 124.2 [122.9-127.7] | 114.4 [112.2-115.2] | 115.5 [113.4-122.0] |
| TTFT p99 ms | 325.5 [322.1-371.8] | 332.7 [310.4-345.9] | 337.2 [320.9-352.8] |

The step-count version is 2.4% below with no overlap, and its p99 is no better, so by the amendment's rule the wall-clock version stays. It does lower TTFT p50 at c=16-32 (90.6 vs 99.5 ms at c=16, 102.4 vs 112.0 at c=24, 114.4 vs 124.2 at c=32) and costs A2T c=8 (584.8 vs 598.0 tok/s, p99 229.5 vs 149.3 ms). Its rows are reported as `M* pr-omni-steps 129a48868`.

## Main against PR (a model without a policy)

Settings as in PR_OMNI.md (colocated Qwen3-Omni config, one H200, --rust-frontend, MSTAR_SHM_ARENA=1, MSTAR_STEP_THREAD_CORE=siblings, no CPU-frequency knobs or uclampset, fresh server per round, 3 alternating rounds). Qwen3-Omni at this branch declares no policy. Median [min-max]:

| cell | metric | main 305e80e | pr-gather 803c6682e | ratio |
|---|---|---|---|---|
| T2T c=32 (A/B session) | tok/s | 2284.2 [2282.6-2289.5] | 2261.6 [2260.6-2261.8] | 0.990, all below |
| T2T c=32 (recheck session) | tok/s | 2248.7 [2247.8-2256.5] | 2263.1 [2262.1-2268.6] | 1.006, all above |
| T2T c=32 (recheck) | TTFT p50 / p99 ms | 60.3 / 179.6 | 58.6 / 150.4 | |
| T2S c=16 | audio-s/s | 95.9 [95.7-99.7] | 99.2 [97.6-100.8] | 1.034, overlapping |
| T2S c=16 | TTFA p50 / p99 ms | 250.8 / 389.2 | 255.7 / 406.1 | |

The first session put the branch 1.0% below main at T2T c=32 with tight, non-overlapping rounds; a second session of 3 rounds each put it 0.6% above, again non-overlapping, and main itself moved 1.6% between sessions. Read: server-to-server and session variation of about 1-2%, no measurable effect. Mechanism check: with no policy, `_gathering` returns after one dict lookup per ready (node, walk) per scheduling pass. Reported: 9 rows of `M* pr-gather 803c6682e` (hypothesis "PR check: gather policy") and the main rows of the same sessions.

## Tests

(pending)

## Review

Independent review of the first draft (82ad7e69), answers in PR_OMNI.md. Raised by the second review (Nov 7 amendment) and fixed here: the peek that never started the wait; the no-policy test whose second half could not fail; the `_gathering` docstring; the docs' per-clip wait.

## Open

- The policy's numbers live in the declaring model (Qwen3-Omni: 8 rows, 30 ms, 16 requests), not in serving config (AGENTS.md invariants 2, 9).
- "Busy" is `len(self.arrival)`, which counts requests until teardown; on the 2-GPU layout that includes requests already speaking on rank 0. The step-count version (which counts busy from a chain step in flight) was measured and not kept (above).
- The peek counts graph-ready rows per walk, `get_next_batch` engine-ready rows per scheduling key; they can disagree for a pass. Not observed.
- Held rows are re-scanned each pass while they wait (bounded by wait_s); per-pass CPU cost not measured with benchmark/worker_phases.
- Not measured: TP, the 2-GPU layout.
