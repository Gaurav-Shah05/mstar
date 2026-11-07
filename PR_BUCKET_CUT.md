DROPPED: no PR. A 1088 bucket in the 9B ladder does as well or better, so it ships in perf/rp3-pr-qwen35 instead (amendment, section 3).

# Bucket cut facts

- Branch `perf/rp3-pr-bucket-cut`, head 8e692b7a5, base 305e80e. Kept only as a record of the reworked code: nothing stacks on it. Gaurav can delete it; I didn't remove it from the fork.
- That head has the amendment's rework: `cut_budget` plus a second `share_token_budget` at `budget=lower`, so the builder's start order and the oldest row's floor hold, with tests of both.

## Why it was dropped
Same tree (Qwen3.5 stack fc12d58), 9B, Random 1024/256 and ShareGPT, 3 alternating rounds, fresh server each, --rust-frontend, MSTAR_SHM_ARENA=1, step_thread_core: siblings. Arms: `bucket_cut_min_saved: 512` on the shipped ladder, against no cut and a ladder with 1088 added. Hypothesis "PR check: qwen35 cut vs 1088", batch h200_1106.

| cell | 1088 tok/s [min-max]; TTFT p50 [min-max] | cut tok/s [min-max]; TTFT p50 [min-max] |
|---|---|---|
| Random c=1 | 202 [201-202]; 45.4 [42.2-46.0] | 201 [200-214]; 49.0 [46.6-50.4] |
| Random c=16 | 2193 [2188-2197]; 209.3 [200.6-216.1] | 2168 [2162-2169]; 233.1 [201.3-235.2] |
| Random c=64 | 4564 [4547-4582]; 267.4 [267.1-269.0] | 4552 [4546-4576]; 268.0 [267.5-269.6] |
| ShareGPT c=1 | 204 [204-205]; 24.5 [21.1-24.9] | 204 [204-218]; 24.9 [22.0-26.0] |
| ShareGPT c=64 | 5433 [5406-5450]; 48.1 [47.0-48.3] | 5439 [5424-5440]; 48.8 [47.9-49.5] |

- The bucket wins with non-overlapping rounds on Random c=1 TTFT and Random c=16 tok/s. The other cells are level. The cut wins nowhere.
- The bucket's cost: 94273 to 94025 MiB free after capture (248 MiB). Largest eager step unchanged at 705 MiB. Boot level at 274-283 s against 278-281 s, leaving out one 479 s boot of the bucket arm's round 1 (first boot after the pod restarted; cause not looked into).
- It also needs no scheduler, engine or policy change, no hand-set threshold, and no second step.

Not settled: the bucket is specific to this workload's prompt length. A 1200-token prompt still pads to 2048, where the cut would have helped. Neither arm was measured there.
