# Authorized 1B-token CPT run — 2026-10-07

User merged PR #3 and authorized proceeding. This run uses the merged cache fix;
it does not restart pretraining from scratch or repeat paid throughput pilots.

## Run

- Official AI21-Jamba2-3B, revision `525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9`.
- All weights trainable. Existing append-only 69,632-token tokenizer, bf16,
  AdamW 8-bit, sequence 2048, microbatch 2, accumulation 32.
- One continuous 1B-token schedule, LR 2e-5, 2% warmup, cosine; no corpus repetition.
- 800M Turkish: FineWeb 200M, Mogan 200M, Bella Academic 160M,
  Bella Ozenli 160M, Bella mc4 40M, Bella OSCAR 40M. Cosmos excluded.
- 200M English retention: FineWeb-Edu 140M, OpenWebMath 40M, Python 20M.
- Ratios now use **accepted encoded tokens including EOS**, rather than raw
  document counts. Rejected/truncated/duplicate texts cannot consume a quota.
  Missing source yield fails preparation; it cannot silently change the mixture.
- The candidate pool is 61 contiguous label shards (6,065,862 rows).
  Labels are not the number of accepted training documents. Prefix selection is
  a budget compromise, not a representative sample of the full source corpus.
- Pinned source joins, full UTF-8 hash/doc-ID checks, PREMIUM/KEEP only,
  nontruncated only, global exact Turkish dedup, content-hash validation split.
- 2M-token heldout cache, same tokenizer, document-disjoint train/validation for
  Turkish and English. No claim of near-dedup or general reasoning improvement.

## Automatic stages

1. Prepare both caches on VPS, CPU limit four cores and RAM limit 5 GiB.
   Halt if VPS free space falls below 15 GiB. No GPU rent before both caches pass
   token-count, source-ratio and every-shard SHA256 checks.
2. Team **LinguAI, ID 733063**, initial observed balance $97.3475. Search one
   verified A100 SXM4 80GB, 180GB disk, total rate <= $1.25/h, >=72 hours remaining
   host availability, transfer costs <= $0.01/GB. Wait if no offer qualifies.
   The observed $0.85/h host ends October 8 and is deliberately ineligible.
3. Use public verified image digest
   `serdadev/jamba2-cpt@sha256:5762cb5fffab046aca0c6c3a7a32fbc1733a8f51352b54244f738f0b6f4eb8e4`.
   Copy the production code archive and frozen caches over SSH. No environment
   rebuilding or dependency upgrade on the rented machine.
4. Baseline: full heldout NLL separately for Turkish and English; the existing
   twelve contract tasks are a small diagnostic, not a benchmark.
5. Train to step 763 (~100M tokens), upload a stopped checkpoint including
   optimizer/scheduler/RNG/data cursor, then evaluate again. Keep the **1B**
   schedule horizon. Resume exactly, without resetting optimizer or schedule,
   if TR NLL <= 1.05x baseline, EN NLL <= 1.10x baseline and contract diagnostic
   drops by no more than two correct cases. These are operational regression
   limits, not statistically validated research thresholds. Otherwise export
   and stop the run for review.
6. Continue to 1B tokens. Evaluate final checkpoint and export reports.
7. Every 500 steps: save and synchronously upload a complete checkpoint.
   Verify remote size and SHA256 (LFS) / Git blob hash (small files) before
   updating latest or pruning. Keep two periodic snapshots plus the 100M
   milestone and stopped/final outputs in the current repository tree.
   Hub commit history is not squashed; historic LFS storage may remain.
8. Check instance and credit every ten minutes. Request a safe checkpoint at
   $86 estimated spend or <=$11 remaining balance. At $90 / <=$7, stop GPU and
   retain disk if checkpoint recovery is still pending; reserve covers delayed
   bandwidth charges. Stop-file handling flushes a partial accumulation safely.
9. Delete **only this run's instance**, after independent VPS verification of
   the newest checkpoint on HF and local capture of status/evaluation reports.
   DONE additionally requires exactly 1B tokens and phase_completed=true.
   A failed or quality-stopped run is distinctly reported, never as DONE.

No deletion merely because logs paused for ten minutes: checkpoint export and
validation can legitimately pause training. Repeated quiet checks request safe
stopping after one hour. Export failure prevents deletion; the spending boundary
stops compute while retaining the disk for recovery.

## Storage / operations

- Run directory: `/home/serda/cpt-run-20261007`.
- Code worktree: `/home/serda/cpt-production`, branch `run/cpt-1b-20261007`.
- CPU unit: `linguai-cpt-prepare.service`; persistent GPU controller:
  `linguai-cpt-supervisor.service` (user systemd, linger enabled).
- Logs: `prepare.log`, `supervisor.log`; state: `preparation.status`,
  `supervisor.json`. Remote: `/mnt/home_extra/run_status.json`, `training.log`.
- Private output: `serda-dev/Jamba2-3B-Turkish-CPT-1B-20261007`.
  Full resumable checkpoints under `checkpoints/`; verified pointer at
  `latest_verified.json`. This repository is distinct from the old SFT model.
- Secrets: team key stays on VPS; HF token transferred into a mode-600 file on
  the instance, never into a public image, repository, or command argument.
- Source text is reconstructed in streaming form; VPS retains packed tokens
  and provenance, not the full raw corpus. The original label repository stays
  unchanged. Source licensing remains relevant to later model/data distribution.

## Verified before enabling the controller

- 55 relevant tests passed: source token balancing despite rejected/unequal
  documents, stream cleanup, trainer exact resume, cache integrity, checkpoint
  hashes, and deletion/budget gating.
- Real HF upload probe passed for a Git JSON blob and an LFS optimizer artifact;
  both remote hashes and sizes matched. Temporary probe files were then deleted.
- Team balance/identity, empty instance list, and eligible A100 search checked.
- At activation, CPU preparation is running; GPU training has not yet begun.
  Progress and all later facts must come from live run state, not this document.

Expected GPU training ~51 hours at the prior measured 5,484 tokens/s. At
$1.23/h that is ~$62 before setup/evaluation/uploads; the prior $75–90 all-in
planning envelope remains a reserve, not an amount that must be spent. CPU
preparation time is separate and does not consume Vast credits.
