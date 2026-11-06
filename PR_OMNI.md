IN PROGRESS: per-commit tests running; main-vs-PR A/B queued after them

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
