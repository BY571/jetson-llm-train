"""Benchmark raw decode speed vs decode+sample."""
import sys
sys.path.insert(0, "engine/build")
import jetson_engine
import time
import torch

engine = jetson_engine.Engine(1024)
engine.load_weights("engine/weights")

# Warmup
engine.reset()
for i in range(8):
    engine.decode_token(i)
engine.sample(1.0, 1.0)

# Raw decode only (no sampling)
engine.reset()
for i in range(8):
    engine.decode_token(i)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(100):
    engine.decode_token(i % 100)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"Raw decode:    {100/elapsed:.1f} tok/s ({elapsed/100*1000:.1f} ms/tok)")

# Decode + sample (CPU logit copy every token)
engine.reset()
for i in range(8):
    engine.decode_token(i)
    engine.sample(0.7, 0.9)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(50):
    engine.decode_token(i % 100)
    engine.sample(0.7, 0.9)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"Decode+sample: {50/elapsed:.1f} tok/s ({elapsed/50*1000:.1f} ms/tok)")

# GPU multinomial sample
engine.reset()
for i in range(8):
    engine.decode_token(i)
    engine.sample_gpu(0.7, 0.9)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(50):
    engine.decode_token(i % 100)
    engine.sample_gpu(0.7, 0.9)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"Decode+GPU sample: {50/elapsed:.1f} tok/s ({elapsed/50*1000:.1f} ms/tok)")

# Full generate with GPU sampling (temp=0.7)
engine.reset()
t0 = time.perf_counter()
tokens = engine.generate([3838, 374, 220, 17, 488, 220, 17, 30],
                          max_new_tokens=100, temperature=0.7, top_p=0.9, eos_token_id=-1)
elapsed = time.perf_counter() - t0
print(f"Generate(0.7): {len(tokens)/elapsed:.1f} tok/s ({elapsed/len(tokens)*1000:.1f} ms/tok)")

# Full generate greedy
engine.reset()
t0 = time.perf_counter()
tokens = engine.generate([3838, 374, 220, 17, 488, 220, 17, 30],
                          max_new_tokens=100, temperature=0.01, top_p=0.9, eos_token_id=-1)
elapsed = time.perf_counter() - t0
print(f"Generate(greedy): {len(tokens)/elapsed:.1f} tok/s ({elapsed/len(tokens)*1000:.1f} ms/tok)")
