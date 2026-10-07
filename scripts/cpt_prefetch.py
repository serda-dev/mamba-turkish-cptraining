"""Download the pinned base model while the VPS finishes token preparation."""
import json
import os
from pathlib import Path
import torch
import datasets
import pyarrow
from huggingface_hub import snapshot_download

os.environ['HF_TOKEN'] = Path('/mnt/home_extra/hf_token').read_text().strip()
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable on the rented GPU')
print(json.dumps({'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
                  'datasets':datasets.__version__, 'pyarrow':pyarrow.__version__}), flush=True)
model = snapshot_download('ai21labs/AI21-Jamba2-3B', revision='525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9',
                          ignore_patterns=['*.gguf','*.onnx','*.h5','*.msgpack'])
Path('/mnt/home_extra/prefetch_done.json').write_text(json.dumps({'model_path':model,'complete':True}))
print('MODEL_PREFETCH_DONE', flush=True)
