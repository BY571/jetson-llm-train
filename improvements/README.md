# Improvements

Each improvement is a standalone script that can be tested independently against the baseline (`train_gsm8k.py`). The goal: build a fully optimized GRPO training pipeline from the ground up for Jetson Orin, no external inference servers.

## Baseline
- `train_gsm8k.py`: HF generate + TRL GRPOTrainer, ~132s/step, ~3 tok/s on Jetson Orin 8GB

## Improvements

| # | Script | Technique | Target | Status |
|---|--------|-----------|--------|--------|
| 1 | `01_profile_baseline.py` | Profile where time is spent (generation vs training, which ops) | Understand | WIP |
| 2 | `02_fused_generate.py` | Custom autoregressive loop with persistent KV cache, skip HF overhead | 2-3x gen | Planned |
| 3 | `03_custom_linear.py` | Fused dequant+matmul Triton kernel for 4-bit NF4 | 2x forward | Planned |
| 4 | `04_flash_attn_sm87.py` | Custom attention kernel optimized for Jetson Orin sm_87 | 2x attn | Planned |
| 5 | `05_simple_grpo.py` | Minimal GRPO loop, drop TRL/HF Trainer overhead | 20-30% | Planned |
| 6 | `06_spec_rl.py` | Reuse prior trajectories (SPEC-RL) | 2-3x rollout | Planned |

## Philosophy

No external servers, no format conversions, no extra processes. Everything runs in one PyTorch process with custom CUDA/Triton kernels for the hot paths. The LoRA weights stay in PyTorch, generation uses the same model, no sync problems.
