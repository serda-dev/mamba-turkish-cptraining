FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache/transformers \
    HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    AUTO_INSTALL=1 \
    AUTO_ENV_CHECK=1 \
    AUTO_RESUME=1 \
    AUTO_TRAIN=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    git-lfs \
    tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/mamba-cpt-tr

COPY . /workspace/mamba-cpt-tr

RUN mkdir -p /workspace/hf-cache /workspace/mamba-cpt-tr/output /workspace/mamba-cpt-tr/dataset

RUN chmod +x scripts/install_env.sh scripts/vast_onstart.sh scripts/run_train.sh scripts/run_resume.sh scripts/docker_push.sh scripts/start_train_vast.sh
RUN bash scripts/install_env.sh

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
