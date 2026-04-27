# Jamba2 3B Turkish CPT

This repository runs continued pre-training of `ai21labs/AI21-Jamba2-3B` for Turkish adaptation. The primary workflow is now a 4-phase Turkish + English curriculum that keeps the existing extended Turkish tokenizer, low-RAM token cache, frequent checkpointing, Docker/Vast.ai execution support, and resume behavior.

The extended tokenizer in `./customtokenizer` is intentional. Model loading resizes embeddings when the tokenizer vocab differs from the base Jamba2 tokenizer.

## Curriculum

The default config is [configs/cpt_4phase.yaml](/media/serda/home_extra/projects/mamba-cpt-tr/configs/cpt_4phase.yaml).

Each phase mixes about 75% Turkish and 25% English by source:

- Phase 1: 30 GB Turkish + 10 GB FineWeb-Edu with `score >= 4.0`
- Phase 2: 30 GB Turkish + 10 GB FineWeb-Edu with `score >= 4.5`
- Phase 3: 30 GB Turkish + 10 GB OpenWebMath
- Phase 4: 30 GB Turkish + 10 GB StarCoder Python

The English mix reduces catastrophic forgetting and preserves reasoning, math, structured text, and code-like dependency tracking while the model adapts to Turkish morphology and syntax.

## Datasets

Turkish source:

```text
MRBeDev/MC4veOSCARTekrarsizBirlestirilmis
```

Expected Turkish shard layout:

```text
paket_X/parca_YYYYY.jsonl.gz
{"text": "..."}
```

The pipeline lists shards with Hugging Face Hub metadata or from a local root, then creates deterministic size-aware partitions in:

```text
artifacts/dataset_manifests/turkish_partitions.json
artifacts/dataset_manifests/phase_1_manifest.json
...
```

Large `.jsonl.gz` shards are read line by line with gzip streaming. The code does not require fully decompressing the dataset.

Remote English streaming uses the optional `datasets` package. For tests or offline development, set `datasets.english.<name>.local_path` to tiny local JSONL/JSONL.GZ fixtures.

## Commands

Activate the project environment first:

```bash
conda activate tr_mamba_cpt
```

Prepare manifests only:

```bash
python -m src.cli prepare-manifests --config configs/cpt_4phase.yaml
```

Validate dataset availability:

```bash
python -m src.cli validate-datasets --config configs/cpt_4phase.yaml
```

Dry-run the training plan:

```bash
python -m src.cli dry-run --config configs/cpt_4phase.yaml
```

Prepare a reusable token cache for one phase:

```bash
TOKENIZERS_PARALLELISM=true \
python -m src.cli prepare-token-cache --config configs/cpt_4phase.yaml --phase 1
```

Prepare all phase token caches:

```bash
TOKENIZERS_PARALLELISM=true \
python -m src.cli prepare-token-cache --config configs/cpt_4phase.yaml
```

Run all phases:

```bash
python -m src.cli train --config configs/cpt_4phase.yaml --resume auto
```

Run one phase:

```bash
python -m src.cli train --config configs/cpt_4phase.yaml --phase 2 --resume auto
```

Legacy single-phase training is still available through:

```bash
python train.py --config configs/train.yaml
```

## Hyperparameters

Phase learning rates, warmup ratios, targets, and mix ratios live under `phases:` in `configs/cpt_4phase.yaml`.

Global batch tokens are logged as:

```text
seq_len * micro_batch_size * gradient_accumulation_steps * WORLD_SIZE
```

The default 4-phase config uses `seq_len: 1024`, `micro_batch_size: 1`, and `gradient_accumulation_steps: 2048`, giving about 2.1M tokens on one GPU.

## Token Cache

Raw JSONL/GZ text is not fed directly to the GPU loop. Each phase is packed into a reusable sharded token cache:

```text
/cache/token_cache/phase_1/
  manifest.json
  shard_000000.bin
  shard_000000.source_ids.bin
  shard_000001.bin
  ...
```

`prepare-token-cache` uses batched tokenizer calls and writes completed shards incrementally. If the process is interrupted, rerun the same command; completed shards are reused and preparation continues from the last manifest checkpoint. The final `manifest.json` is marked with `"complete": true`.

Training checks this cache first. If a complete cache exists for a phase, `train` loads it directly and does not tokenize raw text again. If no complete cache exists, `train` prepares it on the current machine before loading the model.

Useful config knobs in `configs/cpt_4phase.yaml`:

```yaml
token_cache:
  batch_size: 1024
  chunks_per_shard: 8192
  reuse_if_complete: true
```

Increase `batch_size` on CPU-heavy machines with enough RAM. Reduce it if memory usage is high.

## Cheap VPS Preprocessing

The cheapest reliable workflow is to tokenize on a CPU/VPS box, then move only the finished token cache to the GPU server.

On the VPS:

```bash
git clone <YOUR_REPO_URL> mamba-cpt-tr
cd mamba-cpt-tr
conda activate tr_mamba_cpt
python -m pip install -r requirements.txt
export HF_TOKEN=hf_xxx
export HF_HOME=/cache/hf
export CACHE_DIR=/cache
export TOKENIZERS_PARALLELISM=true

python -m src.cli prepare-manifests --config configs/cpt_4phase.yaml
python -m src.cli prepare-token-cache --config configs/cpt_4phase.yaml --phase 1
```

Repeat `--phase 2`, `--phase 3`, and `--phase 4` when needed, or omit `--phase` to prepare all phases.

Copy the cache and manifests to the GPU server:

```bash
rsync -avh --partial --progress /cache/token_cache/ gpu-server:/cache/token_cache/
rsync -avh --partial --progress artifacts/dataset_manifests/ gpu-server:/workspace/mamba-cpt-tr/artifacts/dataset_manifests/
```

On the GPU server:

```bash
cd /workspace/mamba-cpt-tr
python -m src.cli train --config configs/cpt_4phase.yaml --phase 1 --resume auto
```

The GPU server will detect `/cache/token_cache/phase_1/manifest.json` and skip raw dataset tokenization.

## Checkpoints and Resume

Curriculum checkpoints are stored by phase:

```text
/checkpoints/
  latest.json
  phase_1/
    step_000100/
    final/
  phase_2/
    ...
```

Each checkpoint includes model weights, tokenizer files, optimizer state, scheduler state, scaler state when applicable, token counters, phase metadata, and source-ratio counters. `latest.json` points to the latest checkpoint for automatic resume.

Use:

```bash
python -m src.cli train --config configs/cpt_4phase.yaml --resume auto
```

Rerunning that command after a GPU interruption resumes from the latest recorded checkpoint. A completed phase final checkpoint advances the next all-phase run to the following phase.

## Docker

Build:

```bash
docker build -t jamba2-tr-cpt:latest .
```

Run with external volumes:

```bash
docker run --gpus all \
  -v /mnt/data:/data \
  -v /mnt/checkpoints:/checkpoints \
  -v /mnt/cache:/cache \
  -v /mnt/logs:/logs \
  --env HF_TOKEN=... \
  jamba2-tr-cpt:latest \
  python -m src.cli train --config configs/cpt_4phase.yaml --resume auto
```

Do not bake large datasets into the Docker image. Mount `/data`, `/cache`, `/checkpoints`, and `/logs`.

For Vast.ai, `scripts/start_train_vast.sh` defaults to the curriculum entrypoint. Set `TRAIN_ENTRYPOINT=legacy` to use `train.py` and `configs/train.yaml`.

## Vast.ai Template

Use a persistent disk large enough for the HF cache, token cache, and checkpoints. `300 GB` is a practical minimum for smoke runs; real 4-phase CPT can require more depending on how much data you cache locally.

Template fields:

```text
Docker Image:
serdadev/jamba2-cpt:cu121

Docker Options:
-e HF_HOME=/workspace/hf-cache -e TRANSFORMERS_CACHE=/workspace/hf-cache/transformers -e HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub -e TOKENIZERS_PARALLELISM=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e AUTO_INSTALL=1 -e AUTO_ENV_CHECK=1 -e AUTO_RESUME=1 -e AUTO_TRAIN=0 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

On-start Script:
echo "vast startup ok"
cd /workspace/mamba-cpt-tr
bash scripts/vast_onstart.sh

Extra Filters, CLI format:
verified=true gpu_display_active=true
```

If the Turkish dataset is private or gated, add your token as another Docker option:

```text
-e HF_TOKEN=hf_xxx
```

Full Vast CLI example:

```bash
vastai create instance <OFFER_ID> \
  --image serdadev/jamba2-cpt:cu121 \
  --env '-e HF_HOME=/workspace/hf-cache -e TRANSFORMERS_CACHE=/workspace/hf-cache/transformers -e HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub -e TOKENIZERS_PARALLELISM=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e AUTO_INSTALL=1 -e AUTO_ENV_CHECK=1 -e AUTO_RESUME=1 -e AUTO_TRAIN=0 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864' \
  --onstart-cmd 'echo "vast startup ok";cd /workspace/mamba-cpt-tr;bash scripts/vast_onstart.sh;;' \
  --disk 300 \
  --ssh \
  --direct
```

`AUTO_TRAIN=0` starts the machine and runs setup/checks only. After SSH login, start training manually:

```bash
cd /workspace/mamba-cpt-tr
python -m src.cli train --config configs/cpt_4phase.yaml --resume auto
```

Set `AUTO_TRAIN=1` only when you want the instance to begin training immediately after boot.

## Logging

Startup logs include Python, PyTorch, CUDA, GPU name/VRAM, model/tokenizer paths, dataset config, phase config, and checkpoint settings.

Training logs include phase, local step, loss, LR, tokens/sec, samples/sec, GPU memory allocated/reserved, Turkish/English sample and token counts, actual Turkish ratio, ETA, and checkpoint events. JSONL metrics are written under the configured log directory.

## Tests

Fast smoke tests do not download full datasets:

```bash
conda activate tr_mamba_cpt
pytest -q
```

Tiny dry-run smoke test:

```bash
python -m src.cli dry-run --config configs/cpt_4phase.yaml
```

The real config will query the Hugging Face Hub to list Turkish shards when manifests do not exist.

## Troubleshooting

- No Turkish shards found: check `datasets.turkish.repo`, `local_root`, and `file_pattern`.
- Remote English streaming fails: install `datasets` from `requirements.txt` or use a local fixture path.
- Resume starts from the wrong place: inspect `/checkpoints/latest.json` and the phase subdirectories.
- CUDA OOM: reduce `micro_batch_size`, increase accumulation only if needed, keep `seq_len` at 1024, and keep `optimizer: adamw_8bit`.
- Docker cannot resume: make sure `/checkpoints` is mounted to persistent storage.
- Hugging Face private dataset access fails: pass `HF_TOKEN` into the container or login on the host.

## Legacy Notes

[configs/train.yaml](/media/serda/home_extra/projects/mamba-cpt-tr/configs/train.yaml) and `train.py` preserve the older single-phase path that downloads one dataset snapshot, tokenizes it to a memmap cache, and trains with one learning rate. Use this only for compatibility or small experiments; the curriculum CLI is the main workflow.
