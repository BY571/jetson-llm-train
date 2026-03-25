"""Convert HuggingFace model weights to flat binary for the C++ engine.

For NF4 quantized models: dequantizes MLP weights to fp16 in Python
(using bitsandbytes reference code for correctness).
Non-quantized weights (attention, norms, embedding) saved as fp16.

Usage:
    python3 engine/convert_weights.py --model unsloth/Qwen3-0.6B-unsloth-bnb-4bit --output engine/weights
    python3 engine/convert_weights.py --model Qwen/Qwen3-0.6B --output engine/weights
"""
import argparse
import json
import os

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
    parser.add_argument("--output", default="engine/weights")
    args = parser.parse_args()

    print(f"Loading {args.model}...")

    if "bnb-4bit" in args.model or "nf4" in args.model:
        # Load quantized model, dequantize everything to fp16
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        import bitsandbytes as bnb

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config,
            device_map="cuda", torch_dtype=torch.float16,
        )

        weights = {}
        for name, param in model.named_parameters():
            clean = name[6:] if name.startswith("model.") else name
            if hasattr(param, "quant_state"):
                # Dequantize NF4 to fp16 using bitsandbytes (known correct)
                data = bnb.functional.dequantize_4bit(
                    param.data, param.quant_state
                ).to(torch.float16).contiguous().cpu()
            else:
                data = param.data.to(torch.float16).contiguous().cpu()
            weights[clean] = data
            print(f"  {clean}: {list(data.shape)} fp16")
    else:
        # Load fp16 model directly
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cpu"
        )
        weights = {}
        for name, param in model.named_parameters():
            clean = name[6:] if name.startswith("model.") else name
            data = param.data.to(torch.float16).contiguous().cpu()
            weights[clean] = data
            print(f"  {clean}: {list(data.shape)} fp16")

    # Save as flat binary + text index
    bin_path = args.output + ".bin"
    idx_path = args.output + ".idx"
    offset = 0
    lines = []

    with open(bin_path, "wb") as f:
        for name in sorted(weights.keys()):
            data = weights[name]
            raw = data.numpy().tobytes()
            nbytes = len(raw)
            shape = ",".join(str(s) for s in data.shape)
            lines.append(f"{name} {offset} {nbytes} float16 {shape}")
            f.write(raw)
            offset += nbytes

    with open(idx_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    total_mb = os.path.getsize(bin_path) / 1e6
    total_params = sum(w.numel() for w in weights.values())
    print(f"\nSaved {len(lines)} tensors ({total_mb:.1f}MB)")
    print(f"Total params: {total_params:,}")


if __name__ == "__main__":
    main()
