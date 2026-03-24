# Improvements

Each improvement is a standalone script that can be tested independently against the baseline (`train_gsm8k.py`).

## Baseline
- `train_gsm8k.py`: HF generate, ~132s/step, ~3 tok/s on Jetson Orin 8GB

## Improvements

| # | Script | Technique | Expected speedup | Status |
|---|--------|-----------|-----------------|--------|
| 1 | `01_llamacpp_generation.py` | Replace HF generate with llama.cpp server (GGUF, CUDA) | 10-50x generation | WIP |
| 2 | `02_spec_rl.py` | Reuse prior trajectories as speculative prefixes (SPEC-RL) | 2-3x rollout | Planned |
| 3 | `03_simple_grpo.py` | Minimal GRPO loop without TRL overhead | 10-30% overhead | Planned |
| 4 | `04_fast_grpo.py` | Speculative decoding with online draft learning | 2-3x end-to-end | Planned |
| 5 | `05_custom_kernels.py` | Fused dequant+matmul, custom attention for sm_87 | 2x compute | Planned |

## Testing

Each improvement measures:
- **s/step**: Wall-clock time per training step
- **tok/s**: Token generation throughput
- **memory**: Peak GPU memory usage
- **reward**: Training reward curve (to verify quality isn't degraded)
