"""Convert HuggingFace model weights to flat binary for the C++ engine.

Two modes:
  --mode fp16: dequantize everything to fp16 (1.2GB, for correctness testing)
  --mode nf4:  keep MLP as NF4, attention as fp16 (saves ~900MB)

NF4 mode saves per MLP layer:
  .weight          (uint8 packed, 2 values per byte)
  .absmax          (float32, dequantized from double-quant)
  .quant_map       (float32, 16 entries)
  Block size = 64, shape = (out_dim, in_dim)

Usage:
    python3 engine/convert_weights.py --model unsloth/Qwen3-0.6B-unsloth-bnb-4bit --output engine/weights --mode nf4
"""
import argparse
import json
import os

import torch
import numpy as np
import safetensors.torch as st
from huggingface_hub import snapshot_download


def convert_nf4(args):
    """Save NF4 MLP weights as-is, dequant absmax to float32, attention as fp16."""
    print(f"Converting {args.model} (NF4 mode)...")
    model_dir = snapshot_download(args.model)

    tensors = {}
    for fname in os.listdir(model_dir):
        if fname.endswith(".safetensors"):
            for name, t in st.load_file(os.path.join(model_dir, fname)).items():
                clean = name[6:] if name.startswith("model.") else name
                tensors[clean] = t

    bin_path = args.output + ".bin"
    idx_path = args.output + ".idx"
    offset = 0
    lines = []

    with open(bin_path, "wb") as f:
        for name in sorted(tensors.keys()):
            t = tensors[name]

            # Skip quant_state (we extract offset from it)
            if "quant_state" in name:
                continue
            # Skip nested quant map (we dequant absmax ourselves)
            if "nested_quant_map" in name or "nested_absmax" in name:
                continue

            # For absmax: dequantize from uint8 to float32 using nested quant info
            if name.endswith(".weight.absmax"):
                base = name.replace(".weight.absmax", ".weight")
                nested_absmax = tensors.get(base + ".nested_absmax")
                nested_qmap = tensors.get(base + ".nested_quant_map")
                qs_raw = tensors.get(base + ".quant_state.bitsandbytes__nf4")

                nested_offset = 0.0
                if qs_raw is not None:
                    qs_json = bytes(qs_raw.numpy().tolist()).decode("utf-8")
                    nested_offset = json.loads(qs_json).get("nested_offset", 0.0)

                absmax_u8 = t.numpy()
                na = nested_absmax.numpy() if nested_absmax is not None else np.zeros(1)
                nq = nested_qmap.numpy() if nested_qmap is not None else np.zeros(256)

                # Dequantize: val = nq[absmax_u8[i]] * na[i // 256] + offset
                n_blocks = len(absmax_u8)
                absmax_f32 = np.zeros(n_blocks, dtype=np.float32)
                for i in range(n_blocks):
                    group = i // 256
                    absmax_f32[i] = nq[absmax_u8[i]] * na[group] + nested_offset

                data = absmax_f32.tobytes()
                dtype = "float32"
                shape = f"{n_blocks}"
                lines.append(f"{name} {offset} {len(data)} {dtype} {shape}")
                f.write(data)
                offset += len(data)
                continue

            # For NF4 packed weights: save as uint8
            if t.dtype == torch.uint8 and t.numel() > 10000:
                data = t.numpy().flatten().tobytes()
                dtype = "uint8"
                shape = ",".join(str(s) for s in t.shape)
                lines.append(f"{name} {offset} {len(data)} {dtype} {shape}")
                f.write(data)
                offset += len(data)
                continue

            # For quant_map (16 float32 entries): save as-is
            if "quant_map" in name:
                data = t.numpy().tobytes()
                dtype = "float32"
                shape = str(t.shape[0])
                lines.append(f"{name} {offset} {len(data)} {dtype} {shape}")
                f.write(data)
                offset += len(data)
                continue

            # Everything else: convert to fp16
            t_fp16 = t.to(torch.float16) if t.dtype == torch.bfloat16 else t
            if t_fp16.dtype == torch.float16:
                data = t_fp16.contiguous().cpu().numpy().tobytes()
                dtype = "float16"
            else:
                data = t.contiguous().cpu().numpy().tobytes()
                dtype = str(t.dtype).replace("torch.", "")

            shape = ",".join(str(s) for s in t.shape)
            lines.append(f"{name} {offset} {len(data)} {dtype} {shape}")
            f.write(data)
            offset += len(data)

    with open(idx_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    total_mb = os.path.getsize(bin_path) / 1e6
    print(f"Saved {len(lines)} tensors ({total_mb:.1f}MB)")


def convert_fp16(args):
    """Dequantize everything to fp16 using bitsandbytes."""
    print(f"Converting {args.model} (fp16 mode)...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import bitsandbytes as bnb

    if "bnb-4bit" in args.model:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config,
            device_map="cuda", torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cpu"
        )

    bin_path = args.output + ".bin"
    idx_path = args.output + ".idx"
    offset = 0
    lines = []

    with open(bin_path, "wb") as f:
        for name, param in model.named_parameters():
            clean = name[6:] if name.startswith("model.") else name
            if hasattr(param, "quant_state"):
                data = bnb.functional.dequantize_4bit(
                    param.data, param.quant_state
                ).to(torch.float16).contiguous().cpu()
            else:
                data = param.data.to(torch.float16).contiguous().cpu()

            raw = data.numpy().tobytes()
            shape = ",".join(str(s) for s in data.shape)
            lines.append(f"{clean} {offset} {len(raw)} float16 {shape}")
            f.write(raw)
            offset += len(raw)

    with open(idx_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    total_mb = os.path.getsize(bin_path) / 1e6
    print(f"Saved {len(lines)} tensors ({total_mb:.1f}MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
    parser.add_argument("--output", default="engine/weights")
    parser.add_argument("--mode", choices=["fp16", "nf4"], default="fp16")
    args = parser.parse_args()

    if args.mode == "nf4":
        convert_nf4(args)
    else:
        convert_fp16(args)


if __name__ == "__main__":
    main()
