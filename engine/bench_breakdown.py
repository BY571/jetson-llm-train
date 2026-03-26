"""Measure per-token decode time breakdown for NF4 model."""
import sys, time, torch
sys.path.insert(0, "engine/build2")
try:
    import jetson_engine
except ImportError:
    sys.path.insert(0, "engine/build")
    import jetson_engine

engine = jetson_engine.Engine(512)
engine.load_weights("engine/weights_nf4")

# Warmup
for i in range(5):
    engine.decode_token(i)
engine.sample(0.001, 1.0)
engine.reset()

# Test 1: Pure decode time (one call = embedding + 28 layers + LM head)
print("=== Per-token decode time ===")
N = 50
engine.reset()
# Prefill 8 tokens
prompt = [3838, 374, 220, 17, 488, 220, 17, 30]
for t in prompt:
    engine.decode_token(t)
torch.cuda.synchronize()

times = []
for i in range(N):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.decode_token(i % 100)  # dummy token
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

avg_decode = sum(times) / len(times)
print(f"Decode (sync per token): {avg_decode:.2f} ms/tok")
print(f"  Min={min(times):.2f}  Max={max(times):.2f}  Median={sorted(times)[N//2]:.2f}")

# Test 2: Decode + sample time (how generate() works)
engine.reset()
for t in prompt:
    engine.decode_token(t)
torch.cuda.synchronize()

times2 = []
for i in range(N):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tok = engine.sample(0.001, 1.0)
    engine.decode_token(tok)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times2.append((t1 - t0) * 1000)

avg_full = sum(times2) / len(times2)
print(f"\nDecode+sample (sync per token): {avg_full:.2f} ms/tok")
print(f"  Sample overhead: {avg_full - avg_decode:.2f} ms/tok")

# Test 3: Pipelined (no sync between tokens, like generate())
engine.reset()
for t in prompt:
    engine.decode_token(t)
torch.cuda.synchronize()

t0 = time.perf_counter()
for i in range(N):
    tok = engine.sample(0.001, 1.0)
    engine.decode_token(tok)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

avg_pipe = elapsed / N * 1000
print(f"\nPipelined (sync only at end): {avg_pipe:.2f} ms/tok ({N/elapsed:.1f} tok/s)")

# Test 4: Sample-only time
engine.decode_token(0)
torch.cuda.synchronize()
sample_times = []
for i in range(N):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tok = engine.sample(0.001, 1.0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    sample_times.append((t1 - t0) * 1000)

avg_sample = sum(sample_times) / len(sample_times)
print(f"\nSample only: {avg_sample:.2f} ms/tok")

print(f"\n=== Summary ===")
print(f"Decode:         {avg_decode:.2f} ms")
print(f"Sample:         {avg_sample:.2f} ms")
print(f"Decode+Sample:  {avg_full:.2f} ms")
print(f"Pipelined:      {avg_pipe:.2f} ms ({N/elapsed:.1f} tok/s)")
print(f"Sync overhead:  {avg_full - avg_pipe:.2f} ms/tok")
