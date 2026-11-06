IN PROGRESS: branch built (concepts 2, 4, 5, 6); running tests and the benchmark smoke run

# PR facts: Qwen3-Omni

- Base: 305e80ef46248ade6e6670aa20c9bb67f4ac2e26 (upstream main)
- Branch: perf/rp4-pr-omni
- Head: 6e7364cf7 (provisional)

## Commits (in order)

| # | Commit | Subject | Source |
|---|---|---|---|
| 1 | 50f2d8b61 | model: set qwen3-omni thinker sentinel slots without host syncs | 68785670 (ported) |
| 2 | f1fa5c2d9 | test: qwen3-omni thinker prefill prep writes no scalars by index | 0cf8457d + a5921bde (squashed, ported) |
| 3 | a8cd1ffcf | model: build a qwen3-omni audio walk row in a few launches | 387d2c4f + a28a4e57 (ported) |
| 4 | 1a3e7c78a | test: qwen3-omni audio walk row is its spans end to end, embedded in one call | new for main (79fea87f's assertions targeted run-base's row) |
| 5 | d824e79d8 | model: no qwen3-omni talker masks for a request with no audio output | rp3 78d41d65 (cherry-pick -x, conflicts resolved) |
| 6 | 28ccadaba | test: qwen3-omni builds no talker masks without audio output | rp3 afbf77c9 (cherry-pick -x) |
| 7 | dfb615de8 | worker: let a node's ready rows gather before they take a step | rp3 f35f5c34 (cherry-pick -x) + the fake-engine lines of 9a228e98 + Mintlify docs |
| 8 | 6e7364cf7 | test: gather policy waits only while busy, and a node without one steps as before | rp3 9a228e98 (test_gather_policy.py) + a new test |

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
