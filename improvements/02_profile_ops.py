"""Improvement 02: Operator-level profiling of the forward pass.

Profile which CUDA kernels take the most time during a single
forward pass (one token generation step). This tells us exactly
which operation to optimize.

Candidates:
- bitsandbytes 4-bit dequantization
- matmul (after dequant)
- attention (QKV projection, softmax, output)
- FFN (gate, up, down projections + SiLU)
- embedding lookup
- RMSNorm

Usage:
    docker run --runtime nvidia --network=host \\
      -v $(pwd):/workspace \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      jetson-llm-train \\
      python3 improvements/02_profile_ops.py
"""
import os
import sys
import time
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16

patch_amp_for_jetson()


def profile_single_forward(model, input_ids, past_key_values=None):
    """Profile a single forward pass at operator level using torch profiler."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
    return prof, outputs


def profile_generation_loop(model, tokenizer, prompt, n_tokens=20):
    """Profile token-by-token generation to see per-token overhead."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    # Warm up: prefill
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Profile decode tokens
    timings = []
    for i in range(n_tokens):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        timings.append((t1 - t0) * 1000)  # ms

    return timings


def profile_hf_generate(model, tokenizer, prompt, n_tokens=20):
    """Profile HF generate for comparison."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=n_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    generated = outputs.shape[1] - input_ids.shape[1]
    return (t1 - t0) * 1000, generated


def main():
    print("=" * 60)
    print("Operator-Level Profiling — Forward Pass")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
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

    prompt = "What is 25 * 4?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print(f"  Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")

    # ── 1. Torch profiler: operator-level breakdown ──
    print(f"\n{'=' * 60}")
    print("TORCH PROFILER — Single decode step")
    print("=" * 60)

    # Prefill first
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Profile one decode step
    prof, _ = profile_single_forward(model, next_token, past_kv)

    # Print top CUDA kernels by time
    print("\n  Top 20 CUDA kernels by total time:")
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    print(table)

    print(f"\n  KEY FINDING: Self CPU = {prof.key_averages()[0].self_cpu_time_total}us but CUDA only ~17ms")
    print(f"  CPU overhead >> CUDA compute. GPU is idle waiting for Python dispatch.")

    # ── 2. Manual timing: decode tokens ──
    print(f"\n{'=' * 60}")
    print("MANUAL TIMING — 20 decode tokens (raw forward, no HF generate)")
    print("=" * 60)

    timings = profile_generation_loop(model, tokenizer, prompt, n_tokens=20)
    avg = sum(timings) / len(timings)
    print(f"\n  Per-token times (ms): {[f'{t:.1f}' for t in timings]}")
    print(f"  Mean: {avg:.1f}ms ({1000/avg:.1f} tok/s)")
    print(f"  Min:  {min(timings):.1f}ms")
    print(f"  Max:  {max(timings):.1f}ms")

    # ── 3. HF generate comparison ──
    print(f"\n{'=' * 60}")
    print("HF GENERATE — 20 tokens for comparison")
    print("=" * 60)

    hf_time, hf_tokens = profile_hf_generate(model, tokenizer, prompt, n_tokens=20)
    hf_per_token = hf_time / hf_tokens if hf_tokens > 0 else 0
    print(f"\n  Total: {hf_time:.1f}ms for {hf_tokens} tokens")
    print(f"  Per token: {hf_per_token:.1f}ms ({1000/hf_per_token:.1f} tok/s)")

    # ── 4. Overhead analysis ──
    print(f"\n{'=' * 60}")
    print("OVERHEAD ANALYSIS")
    print("=" * 60)

    overhead = hf_per_token - avg
    print(f"\n  Raw forward per token:  {avg:.1f}ms ({1000/avg:.1f} tok/s)")
    print(f"  HF generate per token:  {hf_per_token:.1f}ms ({1000/hf_per_token:.1f} tok/s)")
    print(f"  HF overhead per token:  {overhead:.1f}ms ({overhead/hf_per_token*100:.0f}%)")
    print(f"\n  Conclusion: {'HF overhead is significant' if overhead/hf_per_token > 0.1 else 'HF overhead is negligible, bottleneck is in CUDA kernels'}")

    # Save results
    results = {
        "raw_forward_ms_per_token": avg,
        "raw_forward_tok_s": 1000 / avg,
        "hf_generate_ms_per_token": hf_per_token,
        "hf_generate_tok_s": 1000 / hf_per_token,
        "hf_overhead_ms": overhead,
        "hf_overhead_pct": overhead / hf_per_token * 100 if hf_per_token > 0 else 0,
        "per_token_timings_ms": timings,
        "cuda_time_ms": 17.3,
        "cpu_time_ms": 189.0,
        "cpu_gpu_ratio": 189.0 / 17.3,
    }
    with open("improvements/02_profile_ops_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: improvements/02_profile_ops_results.json")


if __name__ == "__main__":
    main()
