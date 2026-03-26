"""Benchmark NF4 with and without CUDA graph."""
import sys, os, time, torch
sys.path.insert(0, "engine/build2")

# Fallback: try build directory
try:
    import jetson_engine
except ImportError:
    sys.path.insert(0, "engine/build")
    import jetson_engine

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
prompt = [3838, 374, 220, 17, 488, 220, 17, 30]

# --- Test 1: NF4 without CUDA graph ---
engine = jetson_engine.Engine(1024)
engine.load_weights("engine/weights_nf4")

# Warmup
engine.decode_token(0)
engine.sample(1.0, 1.0)
engine.reset()

t0 = time.perf_counter()
tokens = engine.generate(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, eos_token_id=-1)
elapsed = time.perf_counter() - t0
tps1 = len(tokens) / elapsed
print(f"NF4 no graph:   {tps1:.1f} tok/s ({elapsed/len(tokens)*1000:.1f} ms/tok)")
text1 = tok.decode(tokens)[:60]
print(f"  Text: {text1}")

del engine
torch.cuda.empty_cache()

# --- Test 2: NF4 with CUDA graph ---
engine = jetson_engine.Engine(1024)
engine.load_weights("engine/weights_nf4")

# Prefill
for t in prompt:
    engine.decode_token(t)
torch.cuda.synchronize()

# Enable CUDA graph (captures on first call)
try:
    engine.enable_cuda_graph()
    print("CUDA graph captured!")

    # Generate with graph
    generated = []
    t0 = time.perf_counter()
    for i in range(100):
        tok_id = engine.sample(0.001, 1.0)
        engine.decode_token(tok_id)
        generated.append(tok_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps2 = 100 / elapsed
    print(f"NF4 with graph: {tps2:.1f} tok/s ({elapsed/100*1000:.1f} ms/tok)")
    text2 = tok.decode(generated)[:60]
    print(f"  Text: {text2}")
    print(f"\nSpeedup: {tps2/tps1:.2f}x")
except Exception as e:
    print(f"CUDA graph FAILED: {e}")
