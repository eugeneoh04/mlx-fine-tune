import json
import os
from tqdm import tqdm

def export_to_mlx_jsonl(dataset_split, output_filepath):
    """
    Converts a Hugging Face dataset split into the MLX-LM JSONL format.
    Expects the dataset to have 'input' and 'output' columns.
    """
    print(f"Exporting to {output_filepath}...")

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        for row in tqdm(dataset_split, desc=f"Writing {os.path.basename(output_filepath)}"):
            mlx_row = {
                "messages": [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]}
                ]
            }
            f.write(json.dumps(mlx_row) + "\n")
            
    print(f"Successfully saved {len(dataset_split)} rows to {output_filepath}")