"""Convert HuggingFace model weights to a flat binary format for the C++ engine.

Dequantizes NF4 weights to fp16 and stores everything in a single binary file
with a simple index. This avoids implementing safetensors + NF4 parsing in C++.

Format:
  - Header: JSON index (tensor_name -> {offset, shape, dtype}) + padding to 4096
  - Body: raw tensor data, all fp16, contiguous

Usage:
    python3 engine/convert_weights.py --model Qwen/Qwen3-0.6B --output engine/weights.bin
    python3 engine/convert_weights.py --model unsloth/Qwen3-0.6B-unsloth-bnb-4bit --output engine/weights.bin
"""
import argparse
import json
import struct
import os

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def dequantize_model(model_name):
    """Load model and dequantize all weights to fp16."""
    print(f"Loading {model_name}...")

    # Check if it's a pre-quantized model
    if "bnb-4bit" in model_name or "nf4" in model_name:
        # Load quantized, then dequantize
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cpu",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float16,
        )

    # Extract all weights as fp16
    weights = {}
    for name, param in model.named_parameters():
        # Dequantize if needed
        if hasattr(param, 'data'):
            data = param.data
        else:
            data = param

        # Handle bitsandbytes quantized params
        if hasattr(param, 'quant_state'):
            import bitsandbytes as bnb
            data = bnb.functional.dequantize_4bit(
                param.data, param.quant_state
            ).to(torch.float16)
        else:
            data = data.to(torch.float16)

        # Flatten name (remove "model." prefix)
        clean_name = name
        if clean_name.startswith("model."):
            clean_name = clean_name[6:]

        weights[clean_name] = data.contiguous().cpu()
        print(f"  {clean_name}: {list(data.shape)} fp16")

    return weights


def save_binary(weights, output_path):
    """Save weights as a flat binary file with JSON index header."""
    # Build index
    index = {}
    offset = 0
    for name, tensor in weights.items():
        nbytes = tensor.numel() * 2  # fp16 = 2 bytes
        index[name] = {
            "offset": offset,
            "shape": list(tensor.shape),
            "nbytes": nbytes,
        }
        offset += nbytes

    # Serialize header
    header_json = json.dumps(index, indent=2).encode("utf-8")
    # Pad header to 4096 boundary
    header_size = len(header_json)
    padded_size = ((header_size + 8 + 4095) // 4096) * 4096
    padding = padded_size - header_size - 8

    with open(output_path, "wb") as f:
        # 8 bytes: header size (uint64 LE)
        f.write(struct.pack("<Q", header_size))
        # Header JSON
        f.write(header_json)
        # Padding
        f.write(b"\x00" * padding)
        # Tensor data
        for name, tensor in weights.items():
            f.write(tensor.numpy().tobytes())

    total_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved {len(weights)} tensors to {output_path} ({total_mb:.1f}MB)")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="engine/weights.bin")
    args = parser.parse_args()

    weights = dequantize_model(args.model)
    index = save_binary(weights, args.output)

    # Print summary
    total_params = sum(w.numel() for w in weights.values())
    print(f"\nTotal parameters: {total_params:,} ({total_params * 2 / 1e6:.0f}MB fp16)")
    print(f"\nTensor index saved. Load in C++ with:")
    print(f'  engine.load_weights("{args.output}");')


if __name__ == "__main__":
    main()
