"""Improvement 02: torch.compile to reduce CPU dispatch overhead.

Profile showed: GPU=17ms, CPU=188ms per token. The bottleneck is Python
dispatching ~1400 kernel calls per token. torch.compile fuses these into
a single compiled graph, eliminating per-op Python overhead.

Three approaches tested:
1. torch.compile on the full model
2. torch.compile on just the decode step (model + sampling)
3. CUDA graphs for the decode step (static shapes required)

Usage:
    docker run --runtime nvidia --network=host \\
      -v $(pwd):/workspace \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      jetson-llm-train \\
      python3 improvements/02_torch_compile.py
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


def load_model():
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


def bench_generate(model, input_ids, past_kv, n_tokens=50, label=""):
    """Generate n_tokens and measure throughput."""
    # Warm up
    with torch.no_grad():
        out = model(input_ids=input_ids[:, -1:], past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize()

    next_token = input_ids[:, -1:]
    kv = past_kv

    # Reset for actual bench
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(n_tokens):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_tok = elapsed / n_tokens * 1000
    tok_s = n_tokens / elapsed
    print(f"  [{label}] {n_tokens} tokens: {elapsed:.2f}s, {ms_per_tok:.1f}ms/tok, {tok_s:.1f} tok/s")
    return {"ms_per_token": ms_per_tok, "tok_s": tok_s, "total_s": elapsed}


def main():
    print("=" * 60)
    print("Improvement 02: torch.compile for faster generation")
    print("=" * 60)

    model, tokenizer = load_model()
    device = next(model.parameters()).device

    prompt = "Solve step by step: What is 25 * 4 + 10?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print(f"  Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")

    N_TOKENS = 50
    results = {}

    # Also load an fp16 (non-quantized) model for compile testing
    print("\nAlso loading fp16 model (no quantization) for compile test...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model_fp16.eval()
    cast_model_to_fp16(model_fp16)

    # ── Baseline: no compile ──
    print(f"\n{'=' * 60}")
    print(f"TEST 1: Baseline (no compile) — {N_TOKENS} tokens")
    print("=" * 60)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values
    results["baseline_4bit"] = bench_generate(model, input_ids, past_kv, N_TOKENS, "baseline_4bit")

    # ── Baseline fp16 (no quantization) ──
    print(f"\n{'=' * 60}")
    print(f"TEST 1b: Baseline fp16 (no quantization) — {N_TOKENS} tokens")
    print("=" * 60)

    with torch.no_grad():
        out = model_fp16(input_ids=input_ids, use_cache=True)
    past_kv_fp16 = out.past_key_values
    results["baseline_fp16"] = bench_generate(model_fp16, input_ids, past_kv_fp16, N_TOKENS, "baseline_fp16")

    # ── torch.compile on fp16 model (no bnb, should work) ──
    print(f"\n{'=' * 60}")
    print(f"TEST 1c: torch.compile(fp16_model, reduce-overhead) — {N_TOKENS} tokens")
    print("=" * 60)

    try:
        compiled_fp16 = torch.compile(model_fp16, mode="reduce-overhead")
        print("  Compiling fp16 model...")
        with torch.no_grad():
            out = compiled_fp16(input_ids=input_ids, use_cache=True)
        past_kv_cf = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(3):
            with torch.no_grad():
                out = compiled_fp16(input_ids=next_tok, past_key_values=past_kv_cf, use_cache=True)
            past_kv_cf = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        print("  Compilation done.")
        with torch.no_grad():
            out = compiled_fp16(input_ids=input_ids, use_cache=True)
        past_kv_cf = out.past_key_values
        results["compile_fp16"] = bench_generate(compiled_fp16, input_ids, past_kv_cf, N_TOKENS, "compile_fp16")
    except Exception as e:
        print(f"  torch.compile(fp16) FAILED: {e}")
        results["compile_fp16"] = {"error": str(e)}

    # ── torch.compile with inductor on 4bit (needs cuda.h) ──
    print(f"\n{'=' * 60}")
    print(f"TEST 2: torch.compile(model, mode='reduce-overhead') — {N_TOKENS} tokens")
    print("=" * 60)

    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        print("  Compiling (first call triggers compilation)...")

        # Warm up compile
        with torch.no_grad():
            out = compiled_model(input_ids=input_ids, use_cache=True)
        past_kv_c = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # A few more warmup decode steps to fully compile
        for _ in range(3):
            with torch.no_grad():
                out = compiled_model(input_ids=next_tok, past_key_values=past_kv_c, use_cache=True)
            past_kv_c = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        print("  Compilation done.")

        # Re-prefill for clean bench
        with torch.no_grad():
            out = compiled_model(input_ids=input_ids, use_cache=True)
        past_kv_c = out.past_key_values
        results["compile"] = bench_generate(compiled_model, input_ids, past_kv_c, N_TOKENS, "compiled")
    except Exception as e:
        print(f"  torch.compile FAILED: {e}")
        results["compile"] = {"error": str(e)}

    # ── torch.compile with fullgraph ──
    print(f"\n{'=' * 60}")
    print(f"TEST 3: torch.compile(fullgraph=True) — {N_TOKENS} tokens")
    print("=" * 60)

    try:
        # Reload clean model to avoid compile cache issues
        del compiled_model
        torch.cuda.empty_cache()

        compiled_fg = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        print("  Compiling (fullgraph)...")

        with torch.no_grad():
            out = compiled_fg(input_ids=input_ids, use_cache=True)
        past_kv_fg = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(3):
            with torch.no_grad():
                out = compiled_fg(input_ids=next_tok, past_key_values=past_kv_fg, use_cache=True)
            past_kv_fg = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        print("  Compilation done.")

        with torch.no_grad():
            out = compiled_fg(input_ids=input_ids, use_cache=True)
        past_kv_fg = out.past_key_values
        results["compile_fullgraph"] = bench_generate(compiled_fg, input_ids, past_kv_fg, N_TOKENS, "fullgraph")
    except Exception as e:
        print(f"  torch.compile(fullgraph) FAILED: {e}")
        results["compile_fullgraph"] = {"error": str(e)}

    # ── torch.compile max-autotune ──
    print(f"\n{'=' * 60}")
    print(f"TEST 4: torch.compile(mode='max-autotune') — {N_TOKENS} tokens")
    print("=" * 60)

    try:
        compiled_at = torch.compile(model, mode="max-autotune")
        print("  Compiling (max-autotune, may take a while)...")

        with torch.no_grad():
            out = compiled_at(input_ids=input_ids, use_cache=True)
        past_kv_at = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(3):
            with torch.no_grad():
                out = compiled_at(input_ids=next_tok, past_key_values=past_kv_at, use_cache=True)
            past_kv_at = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        print("  Compilation done.")

        with torch.no_grad():
            out = compiled_at(input_ids=input_ids, use_cache=True)
        past_kv_at = out.past_key_values
        results["compile_autotune"] = bench_generate(compiled_at, input_ids, past_kv_at, N_TOKENS, "max-autotune")
    except Exception as e:
        print(f"  torch.compile(max-autotune) FAILED: {e}")
        results["compile_autotune"] = {"error": str(e)}

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    baseline_ms = results["baseline_4bit"]["ms_per_token"]
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<25} FAILED: {r['error'][:60]}")
        else:
            speedup = baseline_ms / r["ms_per_token"]
            print(f"  {name:<25} {r['ms_per_token']:>6.1f}ms/tok  {r['tok_s']:>6.1f} tok/s  {speedup:>5.1f}x")

    with open("improvements/02_torch_compile_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: improvements/02_torch_compile_results.json")


if __name__ == "__main__":
    main()
