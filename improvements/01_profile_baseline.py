"""Improvement 01: Profile the baseline to find where time is spent.

Runs a few training steps and breaks down time into:
- Generation phase (HF generate per completion)
  - Tokenization
  - Forward passes (per token)
  - Sampling
  - KV cache operations
- Reward computation
- GRPO loss + backward
- Optimizer step

Also profiles memory: peak usage, KV cache size, activation size.

Usage:
    docker run --runtime nvidia --network=host \\
      -v $(pwd):/workspace \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      jetson-llm-train \\
      python3 improvements/01_profile_baseline.py
"""
import os
import sys
import time
import json

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16

patch_amp_for_jetson()

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""


def profile_generation(model, tokenizer, prompts, max_new_tokens=512, num_generations=4):
    """Profile the generation phase in detail."""
    device = next(model.parameters()).device
    model.eval()

    timings = {
        "tokenize": [],
        "generate_per_token": [],
        "total_generate": [],
        "tokens_generated": [],
    }

    for prompt_msgs in prompts:
        for g in range(num_generations):
            # Tokenize
            t0 = time.time()
            input_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True,
            )
            input_ids = tokenizer(input_text, return_tensors="pt").to(device)
            prompt_len = input_ids["input_ids"].shape[1]
            t_tok = time.time() - t0
            timings["tokenize"].append(t_tok)

            # Generate
            t1 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            t_gen = time.time() - t1
            new_tokens = outputs.shape[1] - prompt_len
            timings["total_generate"].append(t_gen)
            timings["tokens_generated"].append(new_tokens)
            if new_tokens > 0:
                timings["generate_per_token"].append(t_gen / new_tokens)

    return timings


def profile_backward(model, tokenizer, prompt_msgs, completion_text, device):
    """Profile the backward pass."""
    model.train()

    t0 = time.time()
    full_text = tokenizer.apply_chat_template(
        prompt_msgs + [{"role": "assistant", "content": completion_text}],
        tokenize=False,
    )
    tokens = tokenizer(full_text, return_tensors="pt", truncation=True,
                       max_length=1024).to(device)
    t_tok = time.time() - t0

    t1 = time.time()
    outputs = model(**tokens, labels=tokens["input_ids"])
    t_fwd = time.time() - t1

    t2 = time.time()
    outputs.loss.backward()
    t_bwd = time.time() - t2

    return {"tokenize": t_tok, "forward": t_fwd, "backward": t_bwd}


def profile_memory():
    """Get current memory stats."""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
    return {}


def main():
    print("=" * 60)
    print("Profiling Baseline — Jetson Orin")
    print("=" * 60)

    # Load model (same as baseline)
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

    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    device = next(model.parameters()).device

    print(f"  Memory after model load: {profile_memory()}")

    # Prepare prompts
    data = load_dataset("openai/gsm8k", "main")["train"]
    prompts = []
    for item in list(data)[:3]:
        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ])

    # ── Profile generation ──
    print(f"\n{'=' * 60}")
    print("GENERATION PROFILING (3 prompts x 4 completions x 512 tokens)")
    print("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    gen_timings = profile_generation(model, tokenizer, prompts,
                                      max_new_tokens=512, num_generations=4)

    gen_mem = profile_memory()
    print(f"\n  Peak memory during generation: {gen_mem['max_allocated_mb']:.0f} MB")

    avg_gen = sum(gen_timings["total_generate"]) / len(gen_timings["total_generate"])
    avg_tokens = sum(gen_timings["tokens_generated"]) / len(gen_timings["tokens_generated"])
    avg_per_token = sum(gen_timings["generate_per_token"]) / len(gen_timings["generate_per_token"]) if gen_timings["generate_per_token"] else 0
    avg_tokenize = sum(gen_timings["tokenize"]) / len(gen_timings["tokenize"])
    total_tokens = sum(gen_timings["tokens_generated"])
    total_time = sum(gen_timings["total_generate"])

    print(f"\n  Avg per completion: {avg_gen:.2f}s ({avg_tokens:.0f} tokens)")
    print(f"  Avg per token: {avg_per_token*1000:.1f}ms ({1/avg_per_token:.1f} tok/s)")
    print(f"  Tokenization: {avg_tokenize*1000:.1f}ms avg")
    print(f"  Total: {total_time:.1f}s for {total_tokens} tokens ({total_tokens/total_time:.1f} tok/s)")

    # ── Profile backward ──
    print(f"\n{'=' * 60}")
    print("BACKWARD PASS PROFILING")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-6,
    )

    # Use a sample completion
    sample_completion = "<reasoning>\nLet me solve this step by step.\n1 + 1 = 2\n</reasoning>\n<answer>\n2\n</answer>"

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()
    bwd_timings = profile_backward(model, tokenizer, prompts[0], sample_completion, device)

    t_opt = time.time()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    t_opt = time.time() - t_opt

    bwd_mem = profile_memory()
    print(f"\n  Peak memory during backward: {bwd_mem['max_allocated_mb']:.0f} MB")
    print(f"  Tokenize:  {bwd_timings['tokenize']*1000:.1f}ms")
    print(f"  Forward:   {bwd_timings['forward']*1000:.1f}ms")
    print(f"  Backward:  {bwd_timings['backward']*1000:.1f}ms")
    print(f"  Optimizer: {t_opt*1000:.1f}ms")
    print(f"  Total backward+opt: {(bwd_timings['forward']+bwd_timings['backward']+t_opt)*1000:.1f}ms")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY — Time breakdown for one GRPO step")
    print("=" * 60)

    gen_per_step = avg_gen * 4  # 4 completions per prompt, 1 prompt per step
    bwd_per_step = (bwd_timings["forward"] + bwd_timings["backward"] + t_opt) * 4  # 4 completions

    total_step = gen_per_step + bwd_per_step
    print(f"\n  Generation:  {gen_per_step:.1f}s ({gen_per_step/total_step*100:.0f}%)")
    print(f"  Backward:    {bwd_per_step:.1f}s ({bwd_per_step/total_step*100:.0f}%)")
    print(f"  Total:       {total_step:.1f}s/step")
    print(f"\n  Generation tok/s: {1/avg_per_token:.1f}")
    print(f"  Theoretical min (if gen were instant): {bwd_per_step:.1f}s/step")
    print(f"  Speedup potential from faster gen: {total_step/bwd_per_step:.1f}x")

    # Save results
    results = {
        "generation": {
            "avg_per_completion_s": avg_gen,
            "avg_tokens_per_completion": avg_tokens,
            "avg_ms_per_token": avg_per_token * 1000,
            "tok_per_s": 1 / avg_per_token if avg_per_token > 0 else 0,
            "total_time_s": total_time,
            "total_tokens": total_tokens,
            "peak_memory_mb": gen_mem["max_allocated_mb"],
        },
        "backward": {
            "forward_ms": bwd_timings["forward"] * 1000,
            "backward_ms": bwd_timings["backward"] * 1000,
            "optimizer_ms": t_opt * 1000,
            "peak_memory_mb": bwd_mem["max_allocated_mb"],
        },
        "step_estimate": {
            "generation_s": gen_per_step,
            "backward_s": bwd_per_step,
            "total_s": total_step,
            "generation_pct": gen_per_step / total_step * 100,
            "speedup_if_gen_instant": total_step / bwd_per_step,
        },
    }

    out_path = "improvements/01_profile_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
