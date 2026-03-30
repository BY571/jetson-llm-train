# Triebwerk

Fast LoRA fine-tuning engine for LLMs. Custom C++/CUDA inference engine with CUDA graphs, dp4a 4-bit quantization, and zero-overhead batched generation. PyTorch handles gradients, Triebwerk handles speed.

## Performance

| Setup | Device | Step time | tok/s |
|---|---|---|---|
| TRL + HF generate | RTX 4060 | 68.3s | ~15 |
| TRL + HF generate | Jetson Orin 8GB | 130.6s | ~7 |
| vLLM | RTX 4060 | 5.2s | ~375 |
| **Triebwerk** | **RTX 4060** | **5.8s** | **~300** |
| **Triebwerk** | **Jetson Orin 8GB** | **16.5s** | **~104** |

Output: standard HuggingFace PEFT LoRA adapters. Load with `PeftModel.from_pretrained()`, deploy anywhere.

## Quick start

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

## What makes it fast

- **Chunked prefill**: all prompt tokens in one forward pass per layer (weights read once, not T times)
- **CUDA graphs**: decode + sampling captured as a single graph, ~1us launch overhead per token
- **Pointer-indirection sampling**: device-side `float**` lets the graph read different random values each step
- **Q4L 4-bit with dp4a**: integer dot product, 4 MACs per instruction on Jetson
- **TurboQuant KV cache**: 4x-8x KV cache compression via random rotation + Lloyd-Max quantization ([paper](https://arxiv.org/abs/2504.19874))
- **Batched generation**: all G completions in parallel via cuBLAS GEMM with tensor cores
- **Arena allocator**: single cudaMalloc for all batch buffers, zero fragmentation with PyTorch
- **Amortized stop checking**: sync every 8 tokens instead of every token, GPU-side position increment
- **GPU-GPU LoRA sync**: direct device-to-device memcpy, no CPU roundtrip

## Architecture

```
Training step:
  1. Triebwerk generates G completions (chunked prefill + CUDA graph decode, ~300 tok/s)
  2. Reward functions score completions (Python)
  3. PyTorch forward pass computes log-probs (batched, one call for all G)
  4. GRPO loss: clipped surrogate with per-group advantage normalization
  5. PyTorch backward + optimizer step
  6. LoRA weights synced to engine via GPU-GPU memcpy (~0.02ms)
```

## Running

```bash
# Build the engine (one-time)
cd engine && mkdir -p build_local && cd build_local
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89  # 87 for Jetson, 89 for RTX 4060
make -j$(nproc)

# Convert weights (one-time)
python3 engine/convert_weights.py --model Qwen/Qwen3-0.6B --output engine/weights_q4l --mode q4l

# Train
PYTHONPATH=engine/build_local python3 train.py --max-steps 300

# Quick test (3 steps)
PYTHONPATH=engine/build_local python3 train.py --max-steps 3

# Dry run (no C++ engine needed)
python3 train.py --max-steps 5 --dry-run
```

## TurboQuant: compressed KV cache

Compresses the KV cache using random rotation + optimal scalar quantization ([Zandieh et al. 2025](https://arxiv.org/abs/2504.19874)). The rotation decorrelates coordinates so each can be quantized independently with Lloyd-Max centroids. Only 32 KB of overhead for the shared rotation matrix.

| Mode | KV bits | Memory reduction | Quality (0.6B) |
|---|---|---|---|
| `kv_bits=0` | 16 (fp16) | 1x (baseline) | Perfect |
| `kv_bits=4` | 4 | **4x** | Good — coherent, slight accuracy loss |
| `kv_bits=2` | 2 | **8x** | Needs 8B+ models (too aggressive for 0.6B) |

### Max context length (Qwen3-0.6B, G=4 GRPO training)

| Device | fp16 | TurboQuant 4-bit | Gain |
|---|---|---|---|
| Jetson Orin 8GB | ~7K tokens | ~24K tokens | 3.4x |
| RTX 4060 8GB | ~7K tokens | ~24K tokens | 3.4x |
| RTX 4090 24GB | ~20K tokens | ~65K tokens | 3.3x |

### Usage

```bash
# Enable 4-bit KV cache (recommended for Qwen3-0.6B)
PYTHONPATH=engine/build_local python3 train.py --max-steps 300 --kv-bits 4

# 2-bit for larger models (8B+)
PYTHONPATH=engine/build_local python3 train.py --max-steps 300 --kv-bits 2
```

```python
# Python API
engine = jetson_engine.Engine(max_seq_len=8192, kv_bits=4)
engine.load_weights("engine/weights_q4l")
```

## Supported models

All Qwen3 variants (runtime config, no recompilation):

| Model | Validated | Notes |
|---|---|---|
| Qwen3-0.6B | Yes | Fits on 8GB |
| Qwen3-1.7B | Config ready | Needs 16GB+ |
| Qwen3-4B | Config ready | Needs 16GB+ |
| Qwen3-8B | Config ready | Needs 24GB+ |

## Supported hardware

- **Validated**: RTX 4060 Laptop 8GB (sm_89), Jetson Orin Nano 8GB (sm_87)
- **Target**: Any NVIDIA GPU with sm_61+ (Pascal and later)

## Repo structure

```
train.py                  # GRPO trainer (no TRL dependency)
engine/                   # C++/CUDA inference engine
  engine.cpp              #   Forward pass, CUDA graphs, generation loop
  kernels.cu              #   CUDA kernels (attention, RoPE, sampling, norms)
  nf4_gemv_fast.cu        #   4-bit GEMV/GEMM kernels (dp4a, fused variants)
  model.h                 #   Model config, weight structs, arena allocator
  weights.cpp             #   Weight loader
  engine_py.cpp           #   Python bindings (pybind11)
  convert_weights.py      #   HF model -> Q4L weight format
benchmark/                # Speed comparison (vs TRL, vLLM)
examples/                 # Training examples
  countdown.py            #   Countdown numbers game (recommended for validation)
  letter_counting.py      #   Letter counting (continuous reward signal)
  gsm8k.py                #   GSM8K math (granular format + correctness rewards)
  custom_reward.py        #   Custom reward function template
lora_sync.py              # LoRA weight sync (PyTorch <-> engine)
jetson_compat.py          # Jetson AMP/dtype patches
```
