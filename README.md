# Jamba2 3B Turkish CPT on Vast.ai

This repository is for continued pretraining of `ai21labs/AI21-Jamba2-3B` on single-GPU, interruptible Vast.ai machines. The main target is `RTX 6000 Ada 48GB`, with resumable training as the default operating mode.

## What The Repo Actually Does

- Uses `./customtokenizer` instead of the stock Jamba2 tokenizer.
- Resizes embeddings automatically to match the extended tokenizer vocab.
- Downloads dataset shards directly from the Hugging Face Hub when `data.hf_dataset_id` is set.
- Packs tokenized samples into a memmap dataset for low RAM usage.
- Saves checkpoints frequently and auto-resumes from the newest checkpoint by default.

Current defaults live in [configs/train.yaml](/media/serda/home_extra/projects/mamba-cpt-tr/configs/train.yaml):

- model: `ai21labs/AI21-Jamba2-3B`
- tokenizer: `./customtokenizer`
- dataset: `serda-dev/turkish-raw-text-cleaned`
- `seq_len: 1024`
- `micro_batch_size: 1`
- `gradient_accumulation_steps: 8`
- `torch_dtype: bfloat16`
- `optimizer: adamw_8bit`
- `checkpoint_every_steps: 100`
- `save_total_limit: 6`

## Entry Points

There are two scripts you should care about in the Docker/Vast flow.

### `scripts/start_train_vast.sh`

This is the main training launcher.

- Changes into `/workspace/mamba-cpt-tr`
- Creates `output/`, `dataset/`, and HF cache directories
- Optionally overrides `data.hf_dataset_id`
- Auto-resumes from the newest checkpoint when `AUTO_RESUME=1`
- Forwards the rest of the CLI args to `train.py`

Use this for manual starts:

```bash
cd /workspace/mamba-cpt-tr
bash scripts/start_train_vast.sh
```

Examples:

```bash
cd /workspace/mamba-cpt-tr
bash scripts/start_train_vast.sh --max_steps 1000
```

```bash
cd /workspace/mamba-cpt-tr
bash scripts/start_train_vast.sh --hf_dataset_id serda-dev/turkish-raw-text-cleaned
```

If you want to use a different config file:

```bash
cd /workspace/mamba-cpt-tr
CONFIG_PATH=configs/train.yaml bash scripts/start_train_vast.sh --max_steps 1000
```

### `scripts/vast_onstart.sh`

This is a boot helper for Vast.ai `On-start Script`.

By default it does three things:

1. creates the HF cache directories
2. runs `scripts/install_env.sh` if `AUTO_INSTALL=1`
3. runs `python -m src.utils.env_check` if `AUTO_ENV_CHECK=1`

If `AUTO_TRAIN=1`, it then executes:

```bash
bash scripts/start_train_vast.sh
```

Recommended Vast `On-start Script`:

```bash
cd /workspace/mamba-cpt-tr
bash scripts/vast_onstart.sh
```

## Important Behavior: `AUTO_INSTALL`

This is the part that commonly annoys people during custom startup flows.

The Docker image already runs `bash scripts/install_env.sh` at build time. But `scripts/vast_onstart.sh` also runs it again on every boot when `AUTO_INSTALL=1`.

That means:

- `AUTO_INSTALL=1`: safer for mutable hosts, slower and repetitive on every restart
- `AUTO_INSTALL=0`: better if your image is already baked correctly and you do not want reinstallation on every boot

If you are using your own custom startup script and the image already contains the dependencies, set:

```text
AUTO_INSTALL=0
```

## Dataset Loading

Default dataset source:

```text
serda-dev/turkish-raw-text-cleaned
```

The training code does not use `datasets.load_dataset(...)`. It uses `huggingface_hub.snapshot_download(...)` with `repo_type="dataset"` and reads matching `*.jsonl` shards from the downloaded snapshot.

Expected schema:

```text
{"text": "..."}
```

Default config:

```yaml
data:
  hf_dataset_id: "serda-dev/turkish-raw-text-cleaned"
  hf_cache_dir: "./dataset/hf-cache"
  dataset_dir: "./dataset"
  file_pattern: "*.jsonl"
  text_field: "text"
```

If `hf_dataset_id` is present, the repo pulls the dataset snapshot from Hugging Face.

If `hf_dataset_id` is removed or empty, the repo falls back to local files under `dataset_dir`.

Manual override without editing YAML:

```bash
cd /workspace/mamba-cpt-tr
HF_DATASET_ID=serda-dev/turkish-raw-text-cleaned bash scripts/start_train_vast.sh
```

## Resume Behavior

`scripts/start_train_vast.sh` auto-resumes when `AUTO_RESUME=1`.

It looks for:

- `output/checkpoints/final`
- otherwise the newest `output/checkpoints/step_*`

If found, it appends `--resume <checkpoint>` automatically.

For the common case, just rerun the same command:

```bash
cd /workspace/mamba-cpt-tr
bash scripts/start_train_vast.sh
```

If you want to resume from a specific checkpoint manually, disable auto-resume first:

```bash
cd /workspace/mamba-cpt-tr
AUTO_RESUME=0 bash scripts/start_train_vast.sh --resume output/checkpoints/step_000500
```

Using `--resume` together with `AUTO_RESUME=1` can produce duplicate `--resume` arguments. Do not do that.

## Docker Build

Build command:

```bash
docker build -t YOUR_DOCKERHUB_USER/jamba2-cpt:cu121 .
```

Push:

```bash
docker push YOUR_DOCKERHUB_USER/jamba2-cpt:cu121
```

Shortcut:

```bash
docker login
IMAGE_NAME=YOUR_DOCKERHUB_USER/jamba2-cpt IMAGE_TAG=cu121 bash scripts/docker_push.sh
```

What the Docker build does:

- pulls `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
- copies the repo into `/workspace/mamba-cpt-tr`
- runs `bash scripts/install_env.sh`
- precreates `/workspace/hf-cache`, `output`, and `dataset`

### Build Prerequisites

The build requires outbound network access for:

- Docker Hub base image pull
- PyPI packages
- GitHub-hosted wheel downloads for `mamba-ssm` and `causal-conv1d`

If any of those are blocked, the build will fail before training ever starts.

### A Real Failure Mode To Expect

In this environment, the build currently fails before the first Dockerfile step completes because Docker cannot reach Docker Hub auth:

```text
failed to fetch oauth token
```

That error is not caused by your training code. It means Docker cannot pull the base image. If you see that class of error, fix registry/network access first.

### Wheel Compatibility Assumptions

`scripts/install_env.sh` is pinned to:

- PyTorch `2.5.1+cu121`
- CUDA 12.1 wheel builds
- specific `mamba-ssm` and `causal-conv1d` binary wheel URLs

If you change the base image, Python version, CUDA version, or Torch version, those wheel URLs may stop matching. Keep the Dockerfile pinned unless you are intentionally revalidating the stack.

## Vast.ai Recommended Settings

### Docker Image

```text
YOUR_DOCKERHUB_USER/jamba2-cpt:cu121
```

### Docker Options

```text
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864
```

### Environment Variables

Suggested defaults:

```text
HF_HOME=/workspace/hf-cache
TRANSFORMERS_CACHE=/workspace/hf-cache/transformers
HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub
TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AUTO_INSTALL=0
AUTO_ENV_CHECK=1
AUTO_RESUME=1
AUTO_TRAIN=0
```

Use `AUTO_INSTALL=1` only when you explicitly want startup-time reinstall behavior.

If you want the machine to begin training automatically on boot:

```text
AUTO_TRAIN=1
```

### Persistent Paths

These paths should be backed by persistent storage:

- `/workspace/mamba-cpt-tr/output`
- `/workspace/mamba-cpt-tr/dataset`
- `/workspace/hf-cache`

If `output/` is not persistent, resume is effectively broken.

### GPU Choice

Primary target:

1. `RTX 6000 Ada 48GB`

Reasonable fallback:

2. `L40S 48GB`
3. `RTX 4090 48GB` class offers

## Tokenizer And Legacy `<|endoftext|>` Handling

This repo uses `./customtokenizer`, not the stock tokenizer from the base model.

The tokenizer vocab is extended, so model loading resizes embeddings and LM head automatically before training.

The dataset may contain trailing legacy:

```text
<|endoftext|>
```

The preprocessing path strips trailing legacy `<|endoftext|>` and then relies on normal packing boundaries. This avoids learning duplicated end markers such as:

```text
... <|endoftext|> <|im_end|>
```

## Sanity Checks

Environment:

```bash
cd /workspace/mamba-cpt-tr
python -m src.utils.env_check
```

Inference after a checkpoint exists:

```bash
cd /workspace/mamba-cpt-tr
python -m src.eval.infer --checkpoint_dir ./output/checkpoints/final
```

Inference directly from the published Turkish model on Hugging Face:

```bash
cd /workspace/mamba-cpt-tr
python -m src.eval.infer --model_name_or_path serda-dev/Jamba2-3B-Turkish --prompt "Kisa bir tanitim yazar misin?"
```

Shortcut script:

```bash
cd /workspace/mamba-cpt-tr
PROMPT="Turkce bir paragraf yaz." bash scripts/run_infer.sh
```

For fragile checkpoints, try deterministic decoding first:

```bash
cd /workspace/mamba-cpt-tr
NO_SAMPLE=1 MAX_NEW_TOKENS=64 bash scripts/run_infer.sh
```

Notes:

- The original `ai21labs/AI21-Jamba2-3B` model card loads the model with `AutoModelForCausalLM.from_pretrained(...)` and `AutoTokenizer.from_pretrained(...)`.
- This repo now supports the same Hugging Face style call path for both local checkpoints and Hub model IDs.
- If the tokenizer exposes a chat template, inference uses it automatically.
- If the tokenizer is missing a chat template but still has Jamba chat special tokens, the loader injects a Jamba-compatible fallback template automatically.

Perplexity:

```bash
cd /workspace/mamba-cpt-tr
python -m src.eval.perplexity --checkpoint_dir ./output/checkpoints/final
```

## Notes On Legacy Scripts

`scripts/run_train.sh` and `scripts/run_resume.sh` are older conda-oriented helpers. They are not the primary Docker/Vast path in this repo.

For Docker and Vast.ai, prefer:

- `scripts/vast_onstart.sh`
- `scripts/start_train_vast.sh`

## Short Cheat Sheet

There is also a shorter Vast.ai note here:

- [docs/VAST_AI.md](/media/serda/home_extra/projects/mamba-cpt-tr/docs/VAST_AI.md)
