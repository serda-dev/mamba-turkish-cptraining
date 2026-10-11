# Batch execution optimization, October 11

Instance 55384026, A100 SXM4 80GB, $1.1857/hour including disk.
The user authorized resuming a checkpoint with faster execution while
preserving the CPT learning scope. The original layout was micro batch 2,
accumulation 32, seq_len 2048, gradient checkpointing enabled.

The 500-step checkpoint contained 65,536,000 tokens. A safe operator stop
saved the current model, optimizer, scheduler, RNG and cursor at step 507,
66,437,120 tokens and 32,440 consumed sequences. Its off-host copy was
independently verified at Hub commit
`133f0bc5891376796142fd7acfa6bc6cba40af57`, path
`checkpoints/stopped_step_000507` in the private CPT model repository.

Every benchmark started from that same immutable checkpoint, replayed the
same next examples, and made three optimizer updates without publishing a
checkpoint. The first update was discarded for timing; the two remaining
updates were averaged. These are short comparative measurements, not a
guarantee of sustained full-run throughput.

| Micro batch | Accumulation | Gradient checkpointing | Tokens/s | Peak reserved GiB |
|---|---|---|---:|---:|
| 2 | 32 | enabled, prior production run | ~5,540 | ~25 GB device usage |
| 4 | 16 | enabled | 5,744 | 27.55 |
| 8 | 8 | enabled | 5,820 | 39.19 |
| 4 | 16 | disabled | 7,450 | 68.42 |
| 8 | 8 | disabled | failed: CUDA OOM | nearly full GPU |

Selected: micro batch 4, accumulation 16, gradient checkpointing disabled.
The effective batch remains 64 sequences / 131,072 input tokens. The 1B token
target, sequence length, full-weight training, bf16, AdamW8bit state, LR,
warmup, cosine horizon and source distribution remain the same. The successful
cases had identical token/source counters and closely matching losses on the
same examples (largest observed difference about 0.0002). Batch regrouping and
recomputation changes can cause small numerical differences; the unchanged
100M held-out quality gate checks actual model behavior.

Resume regrouping is explicitly opt-in through
`training.allow_batch_regrouping`. The trainer requires unchanged data/cache
and training signatures, except batch size and accumulation, verifies their
product is unchanged, and converts the cursor by consumed sequence count.
The cursor must align with the new batch. Unrelated configuration changes
remain rejected. CPU tests compare uninterrupted and regrouped shuffled
training weights, token counts, and restored scheduler/optimizer behavior.

Production resumed from step 507 with the selected layout. The runner reuses
the existing baseline and resumes the latest complete training checkpoint
instead of restarting from base weights. Original config/log snapshots are
preserved as `run.before-batch-opt.yaml` and `training.before-batch-opt.log`.
Detailed measurements are in `/home/serda/cpt-run-20261011/batch-benchmark.json`.
Benchmark updates are diagnostic and are not counted as production progress.
The persistent budget/export/deletion supervisor was restarted after launch.
