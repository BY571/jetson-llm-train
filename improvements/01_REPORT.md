# Profiling Report: GRPO Baseline on Jetson Orin 8GB

## Setup

| Parameter | Value |
|---|---|
| Model | Qwen3-0.6B (4-bit NF4, fp16 compute) |
| LoRA | rank=16, 7 target modules, 10M trainable params |
| Generation | G=4 completions, max 512 tokens, temperature=1.0 |
| Hardware | Jetson Orin Nano 8GB unified memory |
| Docker | grpo-jetson (torch 2.8, bitsandbytes, peft, trl 0.29) |

## Time Breakdown

| Phase | Time | % of step |
|---|---|---|
| **Generation (4 completions)** | **312.1s** | **99%** |
| Backward + optimizer | 3.9s | 1% |
| **Total step** | **316.0s** | 100% |

Generation dominates completely. The backward pass is already fast.

## Generation Details

| Metric | Value |
|---|---|
| Throughput | **4.2 tok/s** |
| Latency per token | 237.8ms |
| Avg tokens per completion | 328 |
| Time per completion | 78.0s |
| Tokenization overhead | 4.1ms (negligible) |
| Peak memory | 1000 MB |

At 4.2 tok/s, generating 4 completions of ~328 tokens takes 312 seconds.

## Backward Pass Details

| Operation | Time (per completion) |
|---|---|
| Forward pass | 421ms |
| Backward pass | 385ms |
| Optimizer step | 159ms |
| **Total** | **966ms** |
| Peak memory | 1444 MB |

The backward pass for 4 completions takes ~3.9s. This is the hard floor.

## Memory

| Phase | Peak allocated |
|---|---|
| After model load | 903 MB |
| During generation | 1000 MB |
| During backward | 1444 MB |

Total 8GB unified memory, so ~6.5GB available after OS. The model + training fits comfortably. The OOM issues we had earlier were from HF generate allocating KV cache for too many tokens (768+), not from the model itself.

## Speedup Potential

| Generation speed | Step time | Speedup vs baseline |
|---|---|---|
| **4.2 tok/s (baseline)** | **312s** | **1x** |
| 10 tok/s | 135s | 2.3x |
| 20 tok/s | 70s | 4.5x |
| 40 tok/s | 37s | 8.5x |
| 80 tok/s | 20s | 15.6x |
| 160 tok/s | 12s | 26x |
| 400 tok/s | 7s | 44x |
| Instant | 3.9s | 81.8x |

**Target: 40-80 tok/s would make the 300-step baseline run in 1.5-3 hours instead of 11.**

## Where to Optimize

1. **Generation forward pass (237ms/token)**: This is one full model forward pass per token. The model has 24 transformer layers, each with attention + FFN. The 4-bit dequantization + matmul is the hottest path.

2. **NOT the backward pass**: At 3.9s total, it's already fast. No optimization needed.

3. **NOT tokenization**: At 4ms, it's negligible.

4. **NOT memory**: Peak 1.4GB out of 6.5GB available. We have headroom.

## Conclusion

The single bottleneck is autoregressive token generation speed. Every optimization dollar should go toward making the forward pass faster: fused kernels, better attention, reduced overhead in the generation loop.
