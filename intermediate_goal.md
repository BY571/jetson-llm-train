# Goal: Fast GRPO Training Library for Qwen on NVIDIA GPUs

## What this is

- Fast GRPO adapter fine-tuning for **Qwen models** on NVIDIA GPUs
- Independent C++/CUDA inference engine (no TRL/HuggingFace at runtime for generation)
- LoRA adapters exported in **HuggingFace PEFT-compatible format**
- Users train with our engine, deploy with HuggingFace/vLLM/llama.cpp

## Scope

### Phase 1 (current): Qwen3 on single GPU
- Qwen3-0.6B (validated on Jetson Orin 8GB)
- Qwen3-1.7B, 4B, 8B (same architecture, runtime config)

### Phase 2: generalize
- Multi-architecture support (LLaMA, Gemma, Mistral)
- pip-installable package with pre-built CUDA wheels
- Multi-GPU / distributed training

## Onboarding principle

A user with a GPU and a reward function should go from zero to training in under 5 minutes:

```python
from train import grpo_train

grpo_train(
    dataset=my_dataset,
    reward_funcs=[my_reward],
    model="Qwen/Qwen3-0.6B",
    max_steps=300,
)
```

## Architecture

```
Training step:
  1. C++ engine generates G completions per prompt (Q4L dp4a, ~60 tok/s single)
  2. Reward functions score completions (Python)
  3. PyTorch forward pass computes log-probs
  4. GRPO loss: clipped surrogate with per-group advantage normalization
  5. PyTorch backward + optimizer step
  6. LoRA weights synced back to C++ engine (~2ms)
```

## Benchmark results (Jetson Orin 8GB, Qwen3-0.6B, G=4, 512 tokens)

| Setup | Step time | 300-step estimate |
|---|---|---|
| TRL + HF generate | ~131s/step | ~11h |
| **Ours (C++ engine)** | **~36s/step** | **~3h** |

3.6x faster end-to-end with identical hyperparameters.

## Remaining work

- [ ] Support Qwen3-1.7B, 4B, 8B (weight conversion + testing)
- [ ] pip-installable package
- [ ] Docker one-command install
- [ ] Hardware matrix benchmarks (RTX 4060, RTX 4090)
