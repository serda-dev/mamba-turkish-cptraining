# CPT GPU pilot — 2026-10-07

## Reproducible setup

- Code: `fix/classified-cpt-pipeline` at `786a53b0819dd71cb6f041350253e54acb15ea80`, plus Docker dependency/path changes on `pilot/cpt-gpu-readiness`.
- Public image: `serdadev/jamba2-cpt:pilot-786a53b-r2`, manifest digest `sha256:cfa99d7f5d835ad3d781d3204d712e738ec1e1a44dc6793fcbd9245a79da68e2`.
- Starting model: `ai21labs/AI21-Jamba2-3B` at `525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9`; all 3,039,823,232 parameters were trainable.
- Labels: private `serda-dev/filtered-turkish-dataset` at `388300940d3675640553e817e2b6eae1b3b6947b`, using only `data/fineweb2-tur-files-00-14/part-000000000000-000000100000.parquet` for this pilot. The raw source was streamed at its pinned revision and document IDs/hashes verified.
- Token cache: 2,000,000 attended tokens; SHA-256 integrity check passed. The 10-step training used 1,310,720 tokens: 1,042,624 Turkish and 268,096 English. This is a small throughput sample, not the full classified corpus.
- Identical runs: 1 GPU, bf16, sequence length 2048, microbatch 2, gradient accumulation 32, global batch 131,072 input tokens, gradient checkpointing, AdamW8bit, 10 optimizer steps. Pilot checkpoints were disabled. All runs exited 0 at the operator step limit.

## Measurements

Steady speed is the arithmetic mean of step 2–10 `tokens_per_sec` from the raw JSONL logs. The GPU rate includes the selected instance's 100 GB disk quote. Peak VRAM is the highest 2-second `nvidia-smi` sample. Prices and offer availability change.

| GPU | Rate, USD/h | Steady tokens/s | 10-step train time | Peak sampled VRAM | 1B-token train time* | 1B-token GPU cost* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 SXM4 80 GB | 1.089 | 5,484 | 240 s | 23.8 GiB | 50.7 h | $55.14 |
| H100 NVL 94 GB | 3.054 | 8,934 | 147 s | 24.1 GiB | 31.1 h | $94.97 |
| RTX A6000 48 GB | 0.514 | 2,640 | 496 s | 23.6 GiB | 105.2 h | $54.12 |

\* `1,000,000,000 / steady_tokens_per_sec` and the quoted hourly rate. Excludes preprocessing, model download, checkpoint saves, validation, storage beyond 100 GB, network charges, and market changes. The 10-step loss was about 5.19 initially and 5.07 at step 10 on every GPU; that short change does not establish quality improvement.

Raw step metrics: [A100](pilot_evidence/a100.jsonl), [H100](pilot_evidence/h100.jsonl), [RTX A6000](pilot_evidence/a6000.jsonl). Full console logs and 2-second GPU samples remain on the VPS under `/home/serda/cpt-pilot-data/results/`.

## Decision and remaining gates

A100 is the practical choice at these quotes: nearly the same projected GPU cost as RTX A6000 in roughly half the time. H100 is about 1.63 times faster than A100 but about 1.72 times more expensive per token at the measured quotes. All three tests used full-weight training and completed without GPU OOM; the current path fitted on 48 GB.

Before a paid 1B-token run:

1. Select enough classified label shards across intended source families, freeze their exact revision, and measure accepted token yield. This pilot verified 9,069 FineWeb label rows before its 2M-token stop, accepted 3,289, and yielded 1.59M Turkish tokens. If that prefix yield held, an 800M Turkish-token target would require about 4.6M label rows, above the current 2M-row index limit. This is a prefix extrapolation, not a corpus-wide yield claim.
2. Attach persistent storage for checkpoints and cache, or sync checkpoints to durable storage. The pilot's `/mnt/home_extra` was on the rented instance disk and was deleted with the instance. Size the disk for model cache, optimizer checkpoints, token cache and audit index.
3. Use the corrected cache-reader image described in [the exit fix](CACHE_EXIT_FIX_2026-10-07.md). The original r2 pilot's standalone `prepare-token-cache` wrote a complete, SHA-256-verified cache and then exited 139 during Python finalization (`PyGILState_Release`). The GPU training runs consumed that cache successfully. The later fix pins the verified streaming stack and closes nested readers; automation must continue requiring both a clean producer exit and manifest integrity before the full run.
4. Run heldout loss and frozen task/format evaluations around a longer pilot before treating the full CPT as a quality gain. No model checkpoint was retained from these speed tests.

The team account had no remaining running instances after the tests. Its credit read $97.38 immediately after cleanup; billing may settle further.
