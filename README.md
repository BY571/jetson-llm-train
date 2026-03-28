# jetson-llm-train

Fast GRPO fine-tuning for LLMs on NVIDIA Jetson and other GPUs. Custom C++/CUDA inference engine for generation, PyTorch for training.

## What this does

GRPO (Group Relative Policy Optimization) trains language models using reinforcement learning: generate multiple completions, score them with reward functions, update the model to produce better outputs. The bottleneck is generation — our C++ engine makes it **3.6x faster** end-to-end than TRL on Jetson Orin.

| Setup | Device | Step time | 300 steps |
|---|---|---|---|
| TRL + HF generate | Jetson Orin 8GB | ~131s/step | ~11h |
| **Ours (C++ engine)** | **Jetson Orin 8GB** | **~36s/step** | **~3h** |
| TRL + vLLM | Jetson Orin 8GB | OOM | -- |

Output: standard HuggingFace PEFT LoRA adapters. Load with `PeftModel.from_pretrained()`, deploy anywhere.

## Quick start

Define a reward function, call `grpo_train()`:

```python
from train import grpo_train
from datasets import Dataset

def my_reward(completions, answer, **kwargs):
    return [2.0 if ans in text else -1.0
            for text, ans in zip(completions, answer)]

dataset = Dataset.from_list([
    {"prompt": "What is 2+2?", "answer": "4"},
    {"prompt": "Capital of France?", "answer": "Paris"},
])

grpo_train(
    dataset=dataset,
    reward_funcs=[my_reward],
    model="Qwen/Qwen3-0.6B",
    max_steps=300,
)
```

See `examples/` for complete working examples:
- `examples/gsm8k.py` — math reasoning with structured output
- `examples/custom_reward.py` — custom reward functions

## Running on Jetson

```bash
# Build Docker image
docker build --network=host -t jetson-llm-train .

# Convert weights (one-time)
docker run --runtime nvidia --network=host \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  jetson-llm-train \
  python3 engine/convert_weights.py --model Qwen/Qwen3-0.6B --output engine/weights_q4l --mode q4l

# Train
docker run --runtime nvidia --network=host \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  jetson-llm-train \
  bash -c 'PYTHONPATH=engine/build2 python3 train.py --max-steps 300'
```

## Running without a Jetson (dry run)

Test the full training loop on any machine using HuggingFace generate:

```bash
pip install torch transformers peft datasets bitsandbytes
python3 train.py --max-steps 5 --dry-run
python3 examples/gsm8k.py --dry-run --max-steps 5
```

## Architecture

```
Training step:
  1. C++ engine generates G completions per prompt (Q4L dp4a, 83 tok/s)
  2. Reward functions score completions (Python)
  3. PyTorch forward pass computes log-probs
  4. GRPO loss: clipped surrogate with per-group advantage normalization
  5. PyTorch backward + optimizer step
  6. LoRA weights synced back to C++ engine (~2ms)
```

The C++ engine handles generation (the bottleneck). PyTorch handles loss computation and gradient updates (cheap, <1s per step). LoRA adapters are synced bidirectionally between PyTorch and the engine after each step.

## Supported models

All Qwen3 variants (same architecture, runtime config):

| Model | Validated | Notes |
|---|---|---|
| Qwen3-0.6B | Yes | Fits on Jetson Orin 8GB |
| Qwen3-1.7B | Config ready | Needs 16GB+ |
| Qwen3-4B | Config ready | Needs 16GB+ |
| Qwen3-8B | Config ready | Needs 24GB+ |

The engine reads model dimensions from a JSON config file at runtime — no recompilation needed.

## Supported hardware

- **Validated**: NVIDIA Jetson Orin Nano 8GB (sm_87)
- **Target**: Any NVIDIA GPU with sm_61+ (Pascal and later)
- JetPack 6.x (CUDA 12.6) on Jetson

## Repo structure

```
train.py                  # Standalone GRPO trainer (no TRL dependency)
engine/                   # C++/CUDA inference engine
  model.h                 #   Model config + weight structs
  engine.cpp              #   Forward pass, generation loop
  kernels.cu              #   CUDA kernels (Q4L dp4a GEMV, attention, RoPE, sampling)
  nf4_gemv_fast.cu        #   Optimized GEMV kernels (dp4a, NF4, fused variants)
  weights.cpp             #   Weight loader
  engine_py.cpp           #   Python bindings (pybind11)
  convert_weights.py      #   HF model -> Q4L weight format
  test_all.py             #   Engine test suite
  bench_nf4.py            #   Generation speed benchmark
  bench_gemv.cu           #   GEMV microbenchmark
benchmark/                # TRL comparison benchmark
  config.py               #   Shared hyperparameters (both scripts use this)
  bench_trl.py            #   TRL GRPOTrainer baseline
  bench_ours.py           #   Our engine benchmark
  compare.py              #   Print comparison table
examples/                 # Usage examples
  gsm8k.py                #   GSM8K math reasoning
  custom_reward.py        #   Custom reward function
lora_sync.py              # LoRA weight sync (PyTorch <-> engine)
jetson_compat.py          # Jetson AMP/dtype patches
```

## Jetson setup

On 8GB Jetson, every MB counts. Free ~1GB RAM before training:

```bash
# Disable desktop GUI (~800MB saved)
sudo systemctl set-default multi-user.target

# Add 16GB NVMe swap
sudo fallocate -l 16G /home/$USER/16GB.swap
sudo chmod 600 /home/$USER/16GB.swap
sudo mkswap /home/$USER/16GB.swap
sudo swapon /home/$USER/16GB.swap
```

See [Jetson AI Lab RAM optimization guide](https://www.jetson-ai-lab.com/tutorials/ram-optimization/).

## Key constraints

- **No vLLM on Jetson** (ARM, not supported) — our C++ engine replaces it
- **No bf16 on Jetson** — `jetson_compat.py` patches AMP for fp16
- **8GB unified memory** — CPU and GPU share RAM, 4-bit quantization is essential
- **PyTorch from Jetson AI Lab index** — `pip install` with `--no-deps` to avoid overwriting with x86 builds
