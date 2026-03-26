"""Test Q4L dequantization correctness: compare dp4a vs dequant+GEMM."""
import sys, struct, numpy as np
sys.path.insert(0, "engine/build2")

# Load the Q4L weights manually and verify dequantization
idx_path = "engine/weights_q4l.idx"
bin_path = "engine/weights_q4l.bin"

# Parse index
tensors = {}
with open(idx_path) as f:
    for line in f:
        parts = line.strip().split()
        name, offset, nbytes, dtype = parts[0], int(parts[1]), int(parts[2]), parts[3]
        tensors[name] = (offset, nbytes, dtype)

with open(bin_path, "rb") as f:
    data = f.read()

# Load first Q4L weight: layers.1.self_attn.q_proj.weight (should be Q4L)
w_name = "layers.1.self_attn.q_proj.weight"
s_name = "layers.1.self_attn.q_proj.weight.absmax"
print(f"Testing: {w_name}")
print(f"  dtype: {tensors[w_name][2]}, size: {tensors[w_name][1]} bytes")
print(f"  scales: {tensors[s_name][2]}, size: {tensors[s_name][1]} bytes")

off_w, nb_w, dt_w = tensors[w_name]
off_s, nb_s, dt_s = tensors[s_name]

# Load packed bytes and scales
packed = np.frombuffer(data[off_w:off_w+nb_w], dtype=np.uint8)
scales = np.frombuffer(data[off_s:off_s+nb_s], dtype=np.float32)

# Q4L dimensions: q_proj is (2048, 1024)
out_dim, in_dim = 2048, 1024
total_elems = out_dim * in_dim
n_blocks = total_elems // 64

print(f"  packed: {packed.shape}, scales: {scales.shape}")
print(f"  expected: {total_elems//2} packed bytes, {n_blocks} blocks")

# Dequant in Python using dp4a packing convention
# For each group of 8 elements (4 bytes):
#   byte[k] = elem[k] (lo nibble) | elem[k+4] (hi nibble)
# So: packed[g*4 + k] lo = elem[g*8 + k], hi = elem[g*8 + k + 4]

dequant = np.zeros(total_elems, dtype=np.float32)
n_groups = total_elems // 8
for g in range(min(n_groups, 10)):  # Just check first 10 groups
    for k in range(4):
        byte_idx = g * 4 + k
        b = packed[byte_idx]
        lo_nib = b & 0x0F
        hi_nib = b >> 4
        elem_lo = g * 8 + k
        elem_hi = g * 8 + k + 4
        blk = elem_lo // 64
        scale = scales[blk]
        val_lo = (float(lo_nib) - 8.0) * scale
        val_hi = (float(hi_nib) - 8.0) * scale
        dequant[elem_lo] = val_lo
        dequant[elem_hi] = val_hi

# Print first 16 dequantized values
print(f"\nFirst 16 dequanted values (Python):")
print(f"  {dequant[:16]}")

# Also print what the non-dp4a "original" packing would give
# In the original NF4 packing: byte[k] = elem[2k] (hi) | elem[2k+1] (lo)
# If someone reads our dp4a-packed bytes as if they were original packing:
print(f"\nIf read as original packing (elem[2k]=hi, elem[2k+1]=lo):")
wrong = np.zeros(16, dtype=np.float32)
for i in range(8):
    b = packed[i]
    hi = b >> 4
    lo = b & 0xF
    wrong[2*i] = (float(hi) - 8.0) * scales[0]
    wrong[2*i+1] = (float(lo) - 8.0) * scales[0]
print(f"  {wrong}")

# Check scales
print(f"\nFirst 4 scales: {scales[:4]}")
print(f"Scale for block 0: {scales[0]}")
