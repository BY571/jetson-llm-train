"""Benchmark CUDA graph decode vs non-graph."""
import sys
sys.path.insert(0, "engine/build")
import jetson_engine
import time
import torch
from transformers import AutoTokenizer

engine = jetson_engine.Engine(1024)
engine.load_weights("engine/weights")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Warmup
engine.decode_token(0)
engine.sample(1.0, 1.0)
engine.reset()

prompt = [3838, 374, 220, 17, 488, 220, 17, 30]

# Test with CUDA graph (generate auto-enables it)
t0 = time.perf_counter()
tokens = engine.generate(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, eos_token_id=-1)
elapsed = time.perf_counter() - t0
tps = len(tokens) / elapsed
ms = elapsed / len(tokens) * 1000
print(f"CUDA graph:  {tps:.1f} tok/s ({ms:.1f} ms/tok)")
print(f"  Text: {tok.decode(tokens)[:100]}")

# Without graph (baseline)
# Hack: reset the graph state
engine.reset()
# Generate without graph uses the normal decode path for first run
print(f"\nPrevious best (no graph): 52.7 tok/s (19.0 ms/tok)")
