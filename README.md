# Jamba2-3B Turkish CPT

Current handoff: [docs/CPT_HANDOFF_2026-10-06.md](docs/CPT_HANDOFF_2026-10-06.md).
Research and training rationale: [docs/CPT_RESEARCH_AND_RUN_PLAN.md](docs/CPT_RESEARCH_AND_RUN_PLAN.md).
Implementation status and production gates: [docs/CPT_IMPLEMENTATION_STATUS.md](docs/CPT_IMPLEMENTATION_STATUS.md).

## Current workflow

The default is `configs/cpt_classified.yaml`: one continuous full-weight CPT stream,
not four successive source-specific curricula. Initial engineering settings are
80% Turkish, 14% English educational text, 4% English math, 2% Python, measured by
encoded tokens. These are hypotheses to validate, not established optimal ratios.
The existing extended 69,632-token tokenizer is preserved and audited against the
pinned starting checkpoint. Official Jamba2-3B is already post-trained; do not call
it a raw pretrained base or silently substitute the current Turkish CPT+SFT model.

Private `serda-dev/filtered-turkish-dataset` contains labels, not text. The adapter
streams original pinned sources, verifies row/hash/document identity, retains all
selected labels and routing decisions, filters PREMIUM/KEEP, gates Cosmos and
truncated inputs, deduplicates exact text globally, and holds out content hashes
before packing. Source streams interleave instead of consuming FineWeb first.
Near-duplicate and benchmark contamination removal are not yet implemented.

## CPU preparation

Install `requirements-new.txt` without torch/CUDA. Configure `paths.storage_root`
as an **existing persistent disk directory** and keep paths under it. Also update
`datasets.turkish.classified.audit_db` and `manifest_path` to that disk. No fallback
to the main disk is permitted. Keep HF_HOME on that disk too.

Authenticate using HF_TOKEN in the runtime environment. Obtain representative
Parquet paths from the private label repo; the label index defaults to at most
2M rows and 10 GiB, so it cannot silently index all 214M records on a small VPS.
Choose a representative shard subset spanning every intended source family;
recording only a first shard per source may introduce source-order bias.

```bash
python -m src.cli freeze-inputs --config configs/cpt_classified.yaml \
  --output /mnt/home_extra/jamba-cpt/resolved.yaml \
  --labels-files <representative-parquet-paths>
python -m src.cli prepare-manifests --config /mnt/home_extra/jamba-cpt/resolved.yaml
python -m src.cli validate-datasets --config /mnt/home_extra/jamba-cpt/resolved.yaml --max_documents 64
python -m src.cli prepare-token-cache --config /mnt/home_extra/jamba-cpt/resolved.yaml
python -m src.cli prepare-token-cache --config /mnt/home_extra/jamba-cpt/resolved.yaml \
  --split validation --token_budget 5000000
```

The source plan is pinned in `configs/source_plan_2026-10-02.json`. A mismatch
halts preparation. Validate real producer identities on a small shard before
large preparation; do not change hash logic merely to pass verification.
Validation writes a separate cache and separate audit ledger. Cache v2 records
actual token counts, tokenizer/data identity, exact per-token source attribution,
valid lengths, SHA-256 integrity and a document cursor. Old v1 caches require
rebuilding. A budget stop may cover only a prefix; corpus audit marks partial
coverage rather than claiming every classified document was trained.

## Single GPU pilot and continuation

Use the pinned CUDA environment in `requirements.txt` / Docker. CUDA kernels,
AMP and AdamW8bit must be verified on the actual GPU; CPU tests do not certify
GPU training. The starting candidate is one A100 SXM 80 GB, bf16, sequence 2048,
microbatch 2, accumulation 32 (131,072 input tokens/update), LR 2e-5.

```bash
python -m src.cli dry-run --config /mnt/home_extra/jamba-cpt/resolved.yaml
# About 100M tokens, while preserving the planned 1B scheduler horizon:
python -m src.cli train --config /mnt/home_extra/jamba-cpt/resolved.yaml \
  --max_steps 763 --resume auto
# After baseline/pilot retention and format checks pass:
python -m src.cli train --config /mnt/home_extra/jamba-cpt/resolved.yaml --resume auto
```

`--max_steps` is an absolute operator pause limit, not a scheduler rewrite.
Training does not silently cycle the dataset. Checkpoints retain optimizer,
scheduler, cursor, RNG and immutable data/training signatures. Paused runs do not
advance phases. Legacy checkpoints can be explicit weights-only starting points,
but cannot claim exact cursor resume. Completion requires the requested token
budget, not merely a directory named `final`.

Budget is attended input tokens; shifted loss-bearing tokens are counted
separately. Raw/source-token ratios do not prove every source text is its named
language. Exact token mixtures vary within one document boundary; inspect the
cache/trainer counters. Do not estimate tokens from GB.

## Evaluation and cost

```bash
python -m src.eval.perplexity --model_name_or_path <checkpoint> \
  --cache_manifest /mnt/home_extra/jamba-cpt/cache/token_cache/phase_1_validation/manifest.json
```

Evaluation uses token-weighted NLL, not mean document perplexity. Tiny sample
texts require explicit `--smoke_test` and are not reported as heldout quality.
A small versioned 12-case generation diagnostic also separates recoverable task
answers from JSON parse/schema compliance:

```bash
python -m src.eval.contracts --model <checkpoint> --output <persistent-disk>/contract_results.json
```

Malformed content that cannot be parsed is semantically unscored, not automatically
called wrong reasoning. This small diagnostic does not replace a larger frozen TR
task, EN retention, reasoning, instruction, and JSON/schema evaluation set. CPT and later SFT checkpoints stay separate.

Measure end-to-end input tokens/sec on the intended GPU, including checkpoint
cost. `hours = remaining_tokens / tokens_per_second / 3600`; multiply by the
actual hourly quote and add storage/evaluation overhead. No GPU speed or price
is assumed here. A 1B-token first tranche is conditional on the 100M pilot;
training the entire classified corpus or rebuilding a model from scratch is not
required. Conditional targeted SFT is described in the research plan.

## Tests and historical tools

```bash
python -m pytest tests -q
```

Tests use tiny local fixtures without corpus/model weight downloads. No GitHub
Actions jobs are introduced. `configs/cpt_4phase.yaml`, `train.py`, and standalone
legacy download/fast-tokenize scripts remain historical compatibility paths;
they are not the new default or a verified explanation of the old run mixture.
