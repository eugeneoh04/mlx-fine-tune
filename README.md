# MLX Fine-Tuning Pipeline

LoRA fine-tuning of a Gemma model on the OpenCodeInstruct dataset using MLX-LM.

## Files

- `data.py` — loads the OpenCodeInstruct dataset, shuffles it, and exports train/validation splits to JSONL.
- `export_utils.py` — converts a dataset split into MLX-LM's chat JSONL format.
- `dataset_utils.py` — wraps a dataset so MLX-LM's trainer can index into it.
- `fine-tune.py` — loads the base model, applies LoRA, runs training, and saves the adapter.

## Usage

```bash
# 1. Prepare the data
python data.py

# 2. Run fine-tuning
python fine-tune.py
```

## Output

- `data/train.jsonl`, `data/valid.jsonl` — training and validation sets
- `adapters/adapters.safetensors` — trained LoRA weights
- `adapters/adapter_config.json` — LoRA config used for training

## Requirements

- `mlx-lm`
- `datasets`
- `tqdm`
