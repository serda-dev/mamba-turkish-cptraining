# CPT implementation status — 2026-10-06

## Implemented and tested

- Fixed `english_sources` being ignored; multi-source replay is read explicitly.
- New continuous budgeted CPT config replaces GB-based four-stage defaults.
- Added metadata-only label join to pinned raw sources, full UTF-8 SHA-256 and
  stable document-ID validation, all selected-label preservation and route audit.
- Global exact dedup and content-hash holdout; source interleaving, Cosmos review
  gate, truncation and repair-route exclusions.
- Bounded SQLite index (2M labels / 10 GiB default); raw corpus is streamed.
- Token-balanced TR/general EN/math/Python stream, measured cache/trainer counters.
- Cache v2 provenance, integrity, exact interrupted preparation, added-vocab-safe
  storage, padding masks and per-token source IDs.
- Finite-pass token budget, exact stochastic resume, partial gradient accumulation,
  nonfinite fail-fast, atomic checkpoints, stopped/completed distinction.
- Revision-pinned checkpoint loading and append-only ID/BPE tokenizer audit.
- Explicit heldout cache evaluation and token-weighted NLL/PPL.
- Versioned 12-case generation diagnostic with separate answer/JSON/schema results; not a general capability benchmark.

## Evidence from prior work

Current handoff records 214,716,754 classified documents across 2,151 Parquet
files, a prior approximately six-day single-48GB-GPU run, and the new A10080
preference. Earlier narrative mentions shorter clean-data runs; they are different
runs, not a mandatory duration for this one.

The current project-history document reports corrected 84-case task counts of
27/84 CPT and 28/84 recovery, versus contract counts 6/84 and 5/84. This does not
establish broadly strong reasoning. The SFT-specific Turkish MMLU record 730/2000
must not be assigned to CPT; another run had 1992 unparsable outputs. Parser,
template, merge and semantic outcomes require separate evaluation.

The roughly 200GB mixed historical run lacks complete immutable run/checkpoint
metadata, so noisy data or replay alone cannot be named its proven failure cause.

Local audit against official Jamba2 revision
`525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9` confirms original token IDs are retained
in the 65,536→69,632 extension. Two tiny TR/EN examples round-tripped equivalently;
these are plumbing checks, not corpus-wide tokenizer efficiency evidence.

## Production gates still requiring real data/GPU

1. The connector exposes private label metadata but the local environment has no
   authenticated access to its Parquet bytes. Freeze its exact revision with the
   user's runtime HF_TOKEN. Never put a token in repository files.
2. The upstream main branch exposes `linguai_quality.contracts.document_id` but
   not the full-corpus inference producer code used to write these 214M labels.
   The implemented identity uses that pinned contract. Test representative actual
   labels against original rows; if mismatched, obtain the producer's precise
   algorithm rather than bypassing verification.
3. Select representative explicit label files and measure index/corpus coverage;
   all-label indexing is intentionally blocked by default. Budget-stop audits
   are partial. Near-duplicate and benchmark contamination checks remain separate.
4. Confirm access to the gated StarCoder Python source before paid training.
   Unavailable replay stops preprocessing rather than silently disappearing.
5. Verify actual CUDA 12.1/PyTorch 2.5.1, Mamba/causal-conv kernels, AMP and
   bitsandbytes on one A10080; benchmark throughput, checkpoint time and VRAM.
6. Run fixed task/format/retention baselines and the pilot. The current repo
   implements heldout PPL and a small objective JSON generation diagnostic; a
   broader frozen evaluation set and human semantic review are still needed.
   No GPU training, real quality gain, or full-corpus join is claimed.
7. SFT is a conditional follow-up in the separate SFT pipeline, not synthetic
   instruction generation from corpus quality labels. Keep CPT artifacts intact.

## Validation

CPU regression suite uses local fixtures, mocked model load/eval, real torch
optimizer updates and interrupted stochastic resume. CPU runtime here is
PyTorch 2.14.1+cpu with transformers 4.56.1; production remains pinned separately.
The GPU production runtime is not verified by these CPU results.
