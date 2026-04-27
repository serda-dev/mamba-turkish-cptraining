# Vast.ai Quickstart

This is the short version. The full operational notes are in [README.md](/media/serda/home_extra/projects/mamba-cpt-tr/README.md).

## Docker Push

```bash
cd /path/to/mamba-cpt-tr
docker login
IMAGE_NAME=YOUR_DOCKERHUB_USER/jamba2-cpt IMAGE_TAG=cu121 bash scripts/docker_push.sh
```

## Vast Template

Docker image:

```text
YOUR_DOCKERHUB_USER/jamba2-cpt:cu121
```

Docker options:

```text
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864
```

On-start script:

```bash
cd /workspace/mamba-cpt-tr
bash scripts/vast_onstart.sh
```

Recommended environment variables:

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

Set `AUTO_TRAIN=1` if you want training to start automatically on boot.

Persistent paths:

```text
/workspace/mamba-cpt-tr/output
/workspace/mamba-cpt-tr/dataset
/workspace/hf-cache
```

## Manual Training Start

```bash
cd /workspace/mamba-cpt-tr
bash scripts/start_train_vast.sh
```

With overrides:

```bash
cd /workspace/mamba-cpt-tr
CONFIG_PATH=configs/train.yaml bash scripts/start_train_vast.sh --max_steps 1000
```

Override dataset source:

```bash
cd /workspace/mamba-cpt-tr
bash scripts/start_train_vast.sh --hf_dataset_id serda-dev/turkish-raw-text-cleaned
```

## Resume

Normal case:

```bash
cd /workspace/mamba-cpt-tr
bash scripts/start_train_vast.sh
```

The launcher auto-resumes from the newest checkpoint when `AUTO_RESUME=1`.

If you need a specific checkpoint:

```bash
cd /workspace/mamba-cpt-tr
AUTO_RESUME=0 bash scripts/start_train_vast.sh --resume output/checkpoints/step_000500
```

## Sanity Checks

```bash
cd /workspace/mamba-cpt-tr
python -m src.utils.env_check
```
