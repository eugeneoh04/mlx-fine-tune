from datasets import load_dataset
from export_utils import export_to_mlx_jsonl
import json
import os

cache_directory = "/Users/eoh/.cache/huggingface/hub/datasets--nvidia--OpenCodeInstruct"
opencodeinstruct = load_dataset(cache_directory)

train = opencodeinstruct["train"]

print(f"Total rows: {len(train)}")

shuffled_dataset = train.shuffle(seed=1234)

train_subset = shuffled_dataset.select(range(50000))
valid_subset = shuffled_dataset.select(range(50000, 51000))

export_to_mlx_jsonl(train_subset, "data/train.jsonl")
export_to_mlx_jsonl(valid_subset, "data/valid.jsonl")

print("Dataset preparation complete.")