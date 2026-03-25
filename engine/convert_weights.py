"""Convert HuggingFace model weights to flat binary for the C++ engine.

Saves two files:
  weights.bin - raw tensor data (fp16 for non-quantized, uint8 for NF4)
  weights.idx - text index: "name offset nbytes dtype shape"

NF4 quantized layers (MLP) are saved as raw uint8 with their absmax scales.
Non-quantized layers (attention, norms, embedding) are saved as fp16.

Usage:
    python3 engine/convert_weights.py --model unsloth/Qwen3-0.6B-unsloth-bnb-4bit --output engine/weights
    python3 engine/convert_weights.py --model Qwen/Qwen3-0.6B --output engine/weights
"""
import argparse
import os

import torch
import safetensors.torch as st
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
    parser.add_argument("--output", default="engine/weights")
    args = parser.parse_args()

    print(f"Downloading/loading {args.model}...")
    model_dir = snapshot_download(args.model)

    # Find safetensors files
    st_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    print(f"  Found {len(st_files)} safetensors files")

    # Load all tensors
    all_tensors = {}
    for fname in st_files:
        path = os.path.join(model_dir, fname)
        tensors = st.load_file(path)
        for name, tensor in tensors.items():
            # Strip "model." prefix
            clean = name[6:] if name.startswith("model.") else name
            all_tensors[clean] = tensor

    print(f"  Total tensors: {len(all_tensors)}")

    # Write binary + index
    bin_path = args.output + ".bin"
    idx_path = args.output + ".idx"
    offset = 0
    lines = []

    with open(bin_path, "wb") as f:
        for name in sorted(all_tensors.keys()):
            t = all_tensors[name]
            # bf16 doesn't have numpy support, convert to fp16 first
            t_cpu = t.contiguous().cpu()
            if t_cpu.dtype == torch.bfloat16:
                t_cpu = t_cpu.to(torch.float16)
            data = t_cpu.numpy().tobytes()
            nbytes = len(data)
            dtype = str(t.dtype).replace("torch.", "")
            if dtype == "bfloat16":
                dtype = "float16"  # we converted to fp16 above
            shape = ",".join(str(s) for s in t.shape)

            lines.append(f"{name} {offset} {nbytes} {dtype} {shape}")
            f.write(data)
            offset += nbytes

    with open(idx_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    total_mb = os.path.getsize(bin_path) / 1e6
    print(f"\nSaved {len(lines)} tensors:")
    print(f"  {bin_path} ({total_mb:.1f}MB)")
    print(f"  {idx_path}")

    # Print summary by category
    nf4_count = sum(1 for n in all_tensors if n.endswith(".weight") and all_tensors[n].dtype == torch.uint8 and all_tensors[n].numel() > 10000)
    fp16_count = sum(1 for n in all_tensors if all_tensors[n].dtype in (torch.float16, torch.bfloat16))
    print(f"\n  NF4 packed weights: {nf4_count}")
    print(f"  FP16/BF16 weights: {fp16_count}")
    print(f"  Other (absmax, quant_map, etc.): {len(all_tensors) - nf4_count - fp16_count}")


if __name__ == "__main__":
    main()
