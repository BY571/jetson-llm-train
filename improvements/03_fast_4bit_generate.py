"""Improvement 03: Fast 4-bit generation — CUDA graphs + static KV cache.

Targets the CPU dispatch bottleneck (138ms/tok with only 17ms GPU time).
Three optimizations tested independently and combined:

1. Static KV cache: pre-allocate, avoid aten::cat per token
2. CUDA graphs: record forward pass, replay without CPU dispatch
3. Minimal generate loop: strip all HF overhead

Usage:
    docker run --runtime nvidia --network=host \\
      -v $(pwd):/workspace \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      -e CUDA_HOME=/usr/local/cuda-12.6 \\
      -e CPATH=/usr/local/cuda-12.6/targets/aarch64-linux/include \\
      -e PATH=/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \\
      grpo-jetson bash -c 'export TRITON_PTXAS_PATH=/usr/local/cuda-12.6/bin/ptxas && \\
      python3 improvements/03_fast_4bit_generate.py'
"""
import os
import sys
import time
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import StaticCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16

patch_amp_for_jetson()


def load_4bit_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = cast_model_to_fp16(model)
    model.eval()
    return model, tokenizer


# ── Test 1: Baseline (HF generate) ──

def bench_hf_generate(model, tokenizer, prompt, n_tokens=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    # Warm up
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=5, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=n_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    generated = out.shape[1] - input_ids.shape[1]
    return elapsed, generated


# ── Test 2: Manual loop with DynamicCache (no HF generate overhead) ──

def bench_manual_dynamic(model, tokenizer, prompt, n_tokens=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Prefill
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_tokens


# ── Test 3: Static KV cache ──

def bench_static_cache(model, tokenizer, prompt, n_tokens=50, max_cache_len=512):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    batch_size = input_ids.shape[0]

    try:
        cache = StaticCache(
            config=model.config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=model.device,
            dtype=torch.float16,
        )
    except Exception as e:
        return None, f"StaticCache init failed: {e}"

    # Prefill
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cache = out.past_key_values

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_tokens):
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([input_ids.shape[1] + i], device=model.device),
            )
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_tokens


# ── Test 4: CUDA graph capture on decode step ──

def bench_cuda_graph(model, tokenizer, prompt, n_tokens=50, max_cache_len=512):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    try:
        cache = StaticCache(
            config=model.config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=model.device,
            dtype=torch.float16,
        )
    except Exception as e:
        return None, f"StaticCache init failed: {e}"

    # Prefill (not graphed, variable length)
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cache = out.past_key_values

    # Static input buffers for graph capture
    static_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=model.device)
    static_cache_pos = torch.zeros(1, dtype=torch.long, device=model.device)

    # Warm up decode step with static inputs
    static_input_ids.copy_(next_token)
    static_cache_pos.fill_(prompt_len)
    with torch.no_grad():
        out = model(input_ids=static_input_ids, past_key_values=cache,
                    use_cache=True, cache_position=static_cache_pos)
    static_logits = out.logits

    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            with torch.no_grad():
                out = model(input_ids=static_input_ids, past_key_values=cache,
                            use_cache=True, cache_position=static_cache_pos)
            static_logits = out.logits
    except Exception as e:
        return None, f"CUDA graph capture failed: {e}"

    # Benchmark with graph replay
    next_token_val = next_token.clone()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_tokens):
        static_input_ids.copy_(next_token_val)
        static_cache_pos.fill_(prompt_len + 1 + i)
        graph.replay()
        next_token_val = static_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_tokens


# ── Test 5: torch.no_grad + inference_mode ──

def bench_inference_mode(model, tokenizer, prompt, n_tokens=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_tokens):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_tokens


def main():
    print("=" * 60)
    print("Improvement 03: Fast 4-bit Generation")
    print("=" * 60)

    model, tokenizer = load_4bit_model()
    prompt = "Solve step by step: What is 25 * 4 + 10?"
    N = 50

    print(f"  Prompt: '{prompt}' ({len(tokenizer(prompt).input_ids)} tokens)")
    print(f"  Generating {N} tokens per test\n")

    results = {}

    tests = [
        ("hf_generate", "HF generate (baseline)", bench_hf_generate),
        ("manual_dynamic", "Manual loop + DynamicCache", bench_manual_dynamic),
        ("inference_mode", "inference_mode (vs no_grad)", bench_inference_mode),
        ("static_cache", "Static KV cache", bench_static_cache),
        ("cuda_graph", "CUDA graph + static cache", bench_cuda_graph),
    ]

    for key, name, fn in tests:
        print(f"{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        try:
            elapsed, n = fn(model, tokenizer, prompt, N)
            if elapsed is None:
                print(f"  FAILED: {n}")
                results[key] = {"error": str(n)}
            else:
                ms = elapsed / n * 1000
                tps = n / elapsed
                print(f"  {n} tokens in {elapsed:.2f}s — {ms:.1f}ms/tok, {tps:.1f} tok/s")
                results[key] = {"ms_per_token": ms, "tok_s": tps, "total_s": elapsed}
        except Exception as e:
            print(f"  FAILED: {e}")
            results[key] = {"error": str(e)[:200]}

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    baseline_ms = results.get("hf_generate", {}).get("ms_per_token", 138)
    for key, name, _ in tests:
        r = results.get(key, {})
        if "error" in r:
            print(f"  {name:<40} FAILED: {r['error'][:50]}")
        else:
            speedup = baseline_ms / r["ms_per_token"]
            print(f"  {name:<40} {r['ms_per_token']:>6.1f}ms/tok  {r['tok_s']:>6.1f} tok/s  {speedup:>5.2f}x")

    with open("improvements/03_fast_4bit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: improvements/03_fast_4bit_results.json")


if __name__ == "__main__":
    main()
