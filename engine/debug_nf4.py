"""Debug NF4 dequantization: compare our logic with bitsandbytes reference."""
import struct
import numpy as np
import torch

# Load our saved weights
with open("engine/weights.idx") as f:
    lines = f.readlines()

idx = {}
for line in lines:
    parts = line.strip().split()
    name, offset, nbytes, dtype = parts[0], int(parts[1]), int(parts[2]), parts[3]
    idx[name] = (offset, nbytes, dtype)

with open("engine/weights.bin", "rb") as f:
    data = f.read()

def load(name):
    off, nb, dt = idx[name]
    raw = data[off:off+nb]
    if dt == "uint8":
        return np.frombuffer(raw, dtype=np.uint8)
    elif dt == "float32":
        return np.frombuffer(raw, dtype=np.float32)
    elif dt == "float16":
        return np.frombuffer(raw, dtype=np.float16)
    return raw

# Load gate_proj NF4 components
packed = load("layers.0.mlp.gate_proj.weight")
absmax_u8 = load("layers.0.mlp.gate_proj.weight.absmax")
nested_absmax = load("layers.0.mlp.gate_proj.weight.nested_absmax")
nested_qmap = load("layers.0.mlp.gate_proj.weight.nested_quant_map")
qmap = load("layers.0.mlp.gate_proj.weight.quant_map")
offset = float(load("layers.0.mlp.gate_proj.weight.nested_offset")[0])

print(f"Offset: {offset:.6f}")

# Dequant absmax
block_size = 64
nested_bs = 256
n_blocks = len(absmax_u8)
absmax_f = np.zeros(n_blocks, dtype=np.float32)
for b in range(n_blocks):
    group = b // nested_bs
    absmax_f[b] = nested_qmap[absmax_u8[b]] * nested_absmax[group] + offset

# Dequant first 10 weights
our_vals = []
for j in range(10):
    byte_idx = j // 2
    if j % 2 == 0:
        nib = (packed[byte_idx] >> 4) & 0x0F
    else:
        nib = packed[byte_idx] & 0x0F
    block = j // block_size
    our_vals.append(float(qmap[nib]) * float(absmax_f[block]))

# BNB reference
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    quantization_config=bnb_config, device_map="cuda", torch_dtype=torch.float16
)
param = model.model.layers[0].mlp.gate_proj.weight
ref = bnb.functional.dequantize_4bit(param.data, param.quant_state).float().cpu().numpy()

print("\nComparison (first row, first 10):")
print(f"  Our:  {[round(v, 5) for v in our_vals]}")
print(f"  BNB:  {[round(v, 5) for v in ref[0, :10]]}")
print(f"  Match: {np.allclose(our_vals, ref[0, :10], atol=0.002)}")

# If not matching, check the mapping
if not np.allclose(our_vals, ref[0, :10], atol=0.002):
    print("\n  Debugging mismatch:")
    for j in range(10):
        print(f"    [{j}] our={our_vals[j]:.6f}  ref={ref[0,j]:.6f}  diff={abs(our_vals[j]-ref[0,j]):.6f}")
