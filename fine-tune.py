import os
import json
import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner import train, TrainingArgs, linear_to_lora_layers
from mlx_lm.tuner.datasets import load_dataset
from mlx.utils import tree_flatten
from dataset_utils import DatasetWrapper 

model_path = "mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit"
data_dir = "./data"
adapter_path = "./adapters"

os.makedirs(adapter_path, exist_ok=True)

print("Loading model")
model, tokenizer = load(model_path)

model.freeze()

lora_config = {
    "num_layers": 16,
    "lora_parameters": {
        "rank": 8,
        "scale": 20.0,
        "dropout": 0.0,
    },
}

linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])

num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
print(f"Number of trainable paratmers: {num_train_params}")

model.train()

class MockArgs:
    data = data_dir
    train = True
    test = False

args = MockArgs()
train_set, valid_set, test_set = load_dataset(args, tokenizer)

if train_set: train_set = DatasetWrapper(train_set)
if valid_set: valid_set = DatasetWrapper(valid_set)

print(f"Train set: {len(train_set)} examples | Valid set: {len(valid_set)} examples")
 
adapter_file = os.path.join(adapter_path, "adapters.safetensors")
 
training_args = TrainingArgs(
    adapter_file=adapter_file,
    iters=300,
    steps_per_eval=100,
    steps_per_save=200,
    val_batches=25,
    batch_size=2,
)
 
print("Starting training...")
 
train(
    model=model,
    optimizer=optim.Adam(learning_rate=1e-5),
    train_dataset=train_set,
    val_dataset=valid_set,
    args=training_args,
)
 
model.save_weights(adapter_file)
 
with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
    json.dump(lora_config, f, indent=4)
 
print("Training complete. Adapters saved to:", adapter_path)
print("\nRunning a quick sanity-check generation...")

model.eval()
test_prompt = "Write a Python function that returns the nth Fibonacci number."
messages = [{"role": "user", "content": test_prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=256, verbose=True)
