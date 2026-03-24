# jetson-llm-train

Fine-tune and train small language models (0.5B-3B) efficiently on NVIDIA Jetson Orin devices (8-16GB unified memory).

## Goal

Make RL-based LLM training (GRPO, SFT) practical on edge hardware. The Jetson Orin has 8GB unified memory shared between CPU and GPU, which means every byte counts. This repo provides:

- Docker container with full training stack (unsloth + TRL + bitsandbytes + torch 2.8)
- Optimized generation and training for memory-constrained Jetson devices
- Workarounds for Jetson-specific issues (no bf16 AMP, no vLLM, bitsandbytes compat)
- Baseline: GRPO training on GSM8K with Qwen3-0.6B (4-bit quantized)

## Quick Start

```bash
# Build the Docker image
docker build --network=host -t jetson-llm-train .

# Run the GSM8K baseline
docker run --runtime nvidia --network=host \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  jetson-llm-train \
  python3 train_gsm8k.py
```

## Hardware

- NVIDIA Jetson Orin Nano (8GB) / AGX Orin (32/64GB)
- JetPack 6.x (CUDA 12.6)
- NVMe SSD recommended for swap (16GB+)

## Jetson RAM Optimization (do this first)

On 8GB Jetson, every MB counts. These steps free ~1GB+ of RAM and add 16GB NVMe swap.
See: https://www.jetson-ai-lab.com/tutorials/ram-optimization/

```bash
# Disable desktop GUI (saves ~800MB RAM)
sudo systemctl set-default multi-user.target
sudo systemctl stop gdm3 2>/dev/null; sudo systemctl stop lightdm 2>/dev/null

# Disable camera daemon (not needed for training)
sudo systemctl disable nvargus-daemon.service
sudo systemctl stop nvargus-daemon.service

# Replace ZRAM with 16GB NVMe swap
sudo systemctl disable nvzramconfig
sudo swapoff -a
sudo fallocate -l 16G /home/$USER/16GB.swap
sudo chmod 600 /home/$USER/16GB.swap
sudo mkswap /home/$USER/16GB.swap
sudo swapon /home/$USER/16GB.swap
echo "/home/$USER/16GB.swap  none  swap  sw 0  0" | sudo tee -a /etc/fstab

# Verify
free -h
```

After this you should see ~6.5GB available RAM and 16GB swap.

## Known Jetson Issues & Workarounds

| Issue | Workaround |
|---|---|
| bf16 not supported in AMP | Monkey-patch `_amp_foreach_non_finite_check_and_unscale_` to cast bf16 grads to fp32 |
| vLLM not available (ARM) | Use HF generate (slower but works) |
| unsloth bnb-4bit models have bf16 compute dtype | Load base model with explicit `bnb_4bit_compute_dtype=torch.float16` |
| `torch.pip` overwrites Jetson CUDA torch | Install all ML packages with `--no-deps`, restore Jetson torch after |
| Flash Attention 2 broken on some Jetson builds | Falls back to xformers automatically |

## Optimization Roadmap

- [ ] Custom KV cache manager (persistent, pre-allocated)
- [ ] Fused dequant+matmul CUDA kernel for 4-bit inference
- [ ] Speculative decoding with tiny draft model
- [ ] Optimized attention kernel for Jetson sm_87
- [ ] Profile and eliminate memory allocation hotspots
