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
2. The full student producer is now verified at upstream commit
   `53b6abd197ba2fee7f2bda69f684c8173adafdaa`; its identity matches this reader.
   Test representative actual private Parquet labels against pinned original rows
   before training. Producer-code compatibility does not verify archive contents.
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

## Follow-up producer inspection — 2026-10-06

Historical snapshot below; superseded by the newly published producer inspection
at the end of this document.

The user confirmed `serda-dev/linguai-dataset-quality` as the classification
project. Re-inspected current main `8ce7c69855e018fb697c7c6872347a4d354d171a`
and all six other exposed branches. This snapshot's README explicitly places
student training outside the implemented scope; its export writes
`original_text`, `teacher_label_json`, `route`, and `split=train`, rather than the
handoff's full-corpus `source_id`, `row_ordinal`, `predicted_label_json`,
`predicted_route`, and `input_truncated` schema. The main code and those branches
contain no matching full-corpus student inference writer.

Confirmed source contracts:
- `src/linguai_quality/contracts.py`: full UTF-8 text SHA-256; stable ID from
  compact Unicode JSON `[source, revision, config, split, ordinal, text_sha256]`.
- `src/linguai_quality/runs.py`: raw source is repo/config/split at pinned revision;
  ingestion uses original text without whitespace normalization. Its separate
  content-family whitespace hash is not document_hash.
- `src/linguai_quality/export.py`: bounded teacher export preserves text hash and
  source coordinates; it is not the 214M-row prediction writer.

Four independent literal expected-value vectors were generated from the pinned
upstream identity function and added under `tests/fixtures/upstream_identity_v1.json`.
Tests cover Turkish Unicode, preserved whitespace, decomposed Unicode and ordinal
changes. These validate CPT compatibility with the confirmed control-plane
contract, but cannot prove the unseen full-corpus writer reused that contract.

The separately located `serda-dev/linguai-teacher-model` main README describes a
Qwen inference/API service and explicitly excludes corpus/database/routing storage.
Its purpose does not resolve the missing student writer. No relabeling or teacher
training was launched. The remaining identity gate is actual producer code or
representative real prediction rows matched to pinned raw originals.

Validation after this follow-up: 88 CPU regression tests passed.

## Published student producer inspection — 2026-10-07 (Istanbul)

Upstream main `53b6abd197ba2fee7f2bda69f684c8173adafdaa` now contains
`train_student.py`, `classify_student_stream.py`, and the verified HF archive
publisher. Its README still describes the earlier control-plane scope; the new
scripts provide the previously missing implementation evidence.

Confirmed full-corpus writer contracts:
- `prepare_batch` emits the exact eight fields used by the CPT reader. Hash and
  ID use the full original UTF-8 text, without whitespace normalization.
- Empty/non-string source texts are skipped but retain gaps in global ordinals.
- FineWeb file-range jobs begin at the pinned file's `start_ordinal`; their
  output directories have shard suffixes, while row `source_id` remains
  `fineweb2-tur`. Do not derive source ID from the directory name or part start.
- `input_bytes` is the full original byte count even when `input_truncated=true`.
  The latter describes head/tail token sampling at the classifier's max length,
  not a shortened raw artifact. CPT now verifies the full byte count for every
  row and retains the conservative exclusion of truncated predictions.
- `predicted_route` is a policy applied to student predictions, not ground truth.
  The routing audit script computes heldout disagreement and false acceptance,
  but the actual audit report is not committed; quality thresholds cannot be
  inferred from its existence.
- Publisher paths are `data/<job-directory>/part-<start>-<end>.parquet`, with
  per-job `metadata/.../identity.json` and `DONE.json`. Size and LFS SHA-256 are
  verified before removing local output copies. This does not establish a
  finished join or clean-data yield in the CPT reader.

`tests/fixtures/upstream_student_writer_v1.json` records independently generated
producer metadata from the actual pinned `prepare_batch`, encoding and identity
functions, with their source hashes. A deterministic fake tokenizer exercises
the truncation branch; synthetic labels do not measure student quality. Tests
cover literal identity/bytes including a high global ordinal, a real Parquet
roundtrip with an empty-row gap, truncation exclusion, and corrupt truncated-row
byte counts. Actual private archive bytes and GPU training remain untested.

Cost caveat: selecting a late label shard still requires scanning raw source
prefixes with the current generic join. Benchmark preprocessing separately and
prefer a representative pilot that fits the index budget; the pinned FineWeb
file plan permits a future direct-file optimization without changing ordinals.

Validation after the producer update: 93 CPU tests passed; no paid jobs launched.
