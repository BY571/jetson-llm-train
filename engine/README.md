# Jetson LLM Inference Engine

Custom C++/CUDA inference engine for small transformer models on Jetson Orin.
Designed to replace HuggingFace + bitsandbytes for the generation phase of GRPO training.

## Target: Qwen3-0.6B

```
hidden_size:         1024
intermediate_size:   3072
num_hidden_layers:   28
num_attention_heads: 16  (GQA: 2 Q heads per KV head)
num_key_value_heads: 8
head_dim:            128
vocab_size:          151936
rope_theta:          1000000
rms_norm_eps:        1e-6
activation:          SiLU (gate * silu(up))
lm_head:             tied to embedding
```

## Architecture

```
Python (GRPO training loop)
  |
  v
engine.generate(token_ids, max_tokens) -> list[int]  (pybind11)
  |
  v
C++ decode loop (no Python per token)
  |
  v
CUDA kernels:
  - nf4_dequant_gemv: fused NF4 dequant + matmul (replaces 196 bitsandbytes calls)
  - rms_norm: fused RMSNorm
  - rope_attention: RoPE + GQA attention + KV cache update
  - silu_gate_mul: fused SiLU(gate) * up
  - embedding_lookup: vocab embedding
  - top_p_sampling: temperature + top-p + multinomial
```

## Files

```
model.h          - Model structure, weight layout, KV cache
kernels.cu       - All CUDA kernels
engine.cpp       - C++ inference engine (prefill + decode loop)
engine_py.cpp    - pybind11 Python bindings
weights.cpp      - Load weights from safetensors (HF format)
CMakeLists.txt   - Build configuration
test_correctness.py - Verify output matches HuggingFace
```

## Key Design Decisions

1. **Single monolithic forward pass**: One C++ function call per token, not 1400 kernel launches
2. **Pre-allocated everything**: KV cache, scratch buffers, all allocated at init
3. **NF4 weights loaded directly**: Read safetensors, store as uint8 + lookup table
4. **LoRA support**: Load A/B matrices, apply during matmul (dequant + base + lora_B @ lora_A)
5. **Jetson sm_87 specific**: Hardcoded for Ampere tensor cores, unified memory
