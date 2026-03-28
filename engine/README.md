# C++/CUDA Inference Engine

Custom inference engine for Qwen3 models. Replaces HuggingFace generate for the generation phase of GRPO training.

## Performance

- **60 tok/s** single sequence (dp4a, Q4L 4-bit)
- **83 tok/s** batched G=4 (cuBLAS GEMM with tensor cores)
- Validated on Jetson Orin Nano 8GB (sm_87)

## Key features

- Q4L quantization with dp4a integer dot product (4 MACs/clock vs 1 for fp32)
- Batched generation with cuBLAS GEMM for parallel completions
- LoRA adapter support with live weight sync from PyTorch (~2ms)
- Top-p nucleus sampling (parallel 256-thread max extraction)
- Shared embedding with PyTorch (saves 311MB on unified memory)
- Runtime model config from JSON (no recompilation for different model sizes)

## Building

```bash
mkdir -p engine/build2 && cd engine/build2
cmake .. -DCMAKE_CUDA_ARCHITECTURES=87  # 87 for Jetson Orin, 89 for RTX 4060
make -j$(nproc)
```

## Files

```
engine.cpp          Forward pass, generation loop, batched GEMM
kernels.cu          Attention, RoPE, RMSNorm, sampling, embedding
nf4_gemv_fast.cu    Optimized GEMV (dp4a, NF4, fused 2-proj/3-proj variants)
model.h             Model config, weight structs, buffer management
weights.cpp         Weight loader (binary format with index)
engine_py.cpp       Python bindings (pybind11)
convert_weights.py  HuggingFace model -> Q4L weight format
```
