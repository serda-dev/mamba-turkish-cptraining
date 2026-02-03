# Mamba CPT Pipeline - Walkthrough

## Summary

Implemented a complete, modular Continued Pretraining pipeline for **Mamba 130M** on Turkish JSONL data, optimized for **RTX 4060 8GB** with a ~3 hour training budget.

## Project Structure

```
mamba-cpt-tr/
├── configs/
│   └── train.yaml              # Training configuration
├── src/
│   ├── data/
│   │   ├── jsonl_reader.py     # JSONL streaming + validation
│   │   ├── preprocess.py       # Text cleaning + filtering
│   │   ├── tokenize_pack.py    # Tokenization + sequence packing
│   │   └── dataloader.py       # PyTorch Dataset/DataLoader
│   ├── model/
│   │   └── load.py             # Mamba model + tokenizer loading
│   ├── train/
│   │   └── trainer.py          # Training loop
│   └── utils/
│       ├── time_budget.py      # Time-based stopping
│       ├── logging.py          # Metrics logging
│       └── env_check.py        # Environment verification
├── train.py                    # Main entry point
├── scripts/
│   └── run_train.sh            # Shell launcher
└── requirements.txt            # Dependencies
```

---

## Verification Results

### Environment Check ✓

```
Python:        3.10.19
PyTorch:       2.5.1+cu121
CUDA (torch):  12.1
CUDA avail:    True
GPU:           NVIDIA GeForce RTX 4060 Laptop GPU
GPU Memory:    8.2 GB
Transformers:  4.57.6
mamba-ssm:     2.2.4
causal_conv1d: 1.6.0
einops:        0.8.2
```

### Module Imports ✓

All imports successful, tokenizer loads correctly (vocab_size=50254).

### Training Startup ✓

Pipeline runs and correctly detects missing dataset, exits with clear message:
> "No data files found in ./dataset/*.jsonl"

---

## Usage

### 1. Add Dataset

Place JSONL files in `./dataset/`:
```bash
mkdir -p dataset
# Add your Turkish text JSONL files (each line: {"text": "...", "meta": {...}})
```

### 2. Run Training

```bash
# Using the shell script
./scripts/run_train.sh

# Or directly with Python
conda activate tr_mamba_cpt
python train.py

# Override settings
python train.py --max_steps 1000 --time_budget 3600
```

### 2.1 Continue Training

# Continue training from step 5000 to step 10000
python train.py --resume output/checkpoints/step_005000 --max_steps 10000

# Continue for another 3 hours
python train.py --resume output/checkpoints/final --time_budget 10800

### 3. Monitor

- Logs: `output/logs/`
- Checkpoints: `output/checkpoints/`
- Config: `output/resolved_config.yaml`

---

## Key Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `seq_len` | 1024 | Sequence length |
| `micro_batch_size` | 1 | Fits in 8GB VRAM |
| `gradient_accumulation_steps` | 16 | Effective batch = 16 |
| `max_steps` | 5000 | ~3 hours estimate |
| `time_budget_seconds` | 10800 | 3 hour hard limit |
| `checkpoint_every_steps` | 500 | Save frequency |
| `mixed_precision` | true | fp16 for memory |
