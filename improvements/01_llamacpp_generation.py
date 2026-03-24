"""Improvement 1: Replace HF generate with llama.cpp server for generation.

Architecture:
  1. Export the current LoRA weights to GGUF periodically
  2. Load GGUF in llama-server (CUDA, optimized for Jetson sm_87)
  3. GRPO generation phase: send batch requests to llama-server via OpenAI API
  4. GRPO training phase: compute loss and backward with PyTorch (unchanged)

Why this helps:
  - llama.cpp with CUDA on Jetson Orin: ~300+ tok/s for 7B, ~1000+ for 0.6B
  - HF generate on Jetson: ~3 tok/s (our baseline)
  - Generation is ~90% of step time (120s of 132s)
  - Expected: 120s generation -> 2-5s, total step: 132s -> 15-20s (7-10x speedup)

Prerequisites:
  - llama.cpp compiled for Jetson (CUDA, sm_87)
  - llama-server running as background process
  - GGUF model file (converted from HF format)

Usage:
  # Step 1: Convert model to GGUF
  python3 01_llamacpp_generation.py --mode convert --model Qwen/Qwen3-0.6B --output model.gguf

  # Step 2: Start llama-server
  python3 01_llamacpp_generation.py --mode serve --gguf model.gguf

  # Step 3: Run training with llama.cpp generation
  python3 01_llamacpp_generation.py --mode train --gguf model.gguf --max-steps 10

  # Or all-in-one:
  python3 01_llamacpp_generation.py --mode all --max-steps 300
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Optional

import requests
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add parent dir to path for jetson_compat
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

LLAMA_SERVER_URL = "http://localhost:8080"


# ── llama.cpp generation client ──

class LlamaCppGenerator:
    """Generate completions via llama-server's OpenAI-compatible API."""

    def __init__(self, server_url=LLAMA_SERVER_URL, max_tokens=512, temperature=1.0):
        self.server_url = server_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def health_check(self):
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def generate_batch(self, prompts, n_completions=4):
        """Generate n_completions for each prompt via llama-server.

        Uses the /v1/chat/completions endpoint with n parameter for
        multiple completions per prompt.

        Returns: list of list of strings (outer=prompts, inner=completions)
        """
        all_completions = []
        for prompt_messages in prompts:
            completions = []
            # llama.cpp server may not support n>1 in all versions,
            # so we send n_completions individual requests
            for _ in range(n_completions):
                try:
                    r = requests.post(
                        f"{self.server_url}/v1/chat/completions",
                        json={
                            "messages": prompt_messages,
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature,
                            "stream": False,
                        },
                        timeout=120,
                    )
                    r.raise_for_status()
                    text = r.json()["choices"][0]["message"]["content"]
                    completions.append(text)
                except Exception as e:
                    print(f"  llama-server error: {e}")
                    completions.append("")
            all_completions.append(completions)
        return all_completions


# ── Reward functions (same as baseline) ──

def extract_xml_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def correctness_reward(completions, answers):
    rewards = []
    for comp, ans in zip(completions, answers):
        extracted = extract_xml_answer(comp)
        rewards.append(2.0 if extracted == ans else -1.0)
    return rewards


def format_reward(completions):
    rewards = []
    for comp in completions:
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", comp, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", comp, re.DOTALL))
        if has_reasoning and has_answer:
            rewards.append(1.0)
        elif has_answer:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


# ── Model export to GGUF ──

def export_lora_to_gguf(base_model_name, lora_path, output_path):
    """Merge LoRA into base model and convert to GGUF for llama.cpp.

    This is the bridge between PyTorch (training) and llama.cpp (generation).
    Called periodically during training to update the generation model.
    """
    print(f"  Exporting to GGUF: {output_path}")
    # Step 1: Merge LoRA into base model in fp16
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="cpu",
    )
    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    # Step 2: Save merged model
    merged_path = output_path + ".merged"
    model.save_pretrained(merged_path)
    AutoTokenizer.from_pretrained(base_model_name).save_pretrained(merged_path)

    # Step 3: Convert to GGUF using llama.cpp's convert script
    convert_script = "/workspace/llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print(f"  WARNING: {convert_script} not found. Install llama.cpp first.")
        print(f"  Merged model saved to: {merged_path}")
        return merged_path

    subprocess.run([
        sys.executable, convert_script,
        merged_path,
        "--outfile", output_path,
        "--outtype", "q4_k_m",  # 4-bit quantization
    ], check=True)
    print(f"  GGUF saved: {output_path}")
    return output_path


# ── Custom GRPO training loop with llama.cpp generation ──

def grpo_step(model, tokenizer, generator, prompts, answers, device, lr=5e-6):
    """One GRPO training step using llama.cpp for generation.

    1. Generate G completions per prompt via llama-server (fast)
    2. Score with reward functions
    3. Compute GRPO loss and update model (PyTorch backward)

    This is a simplified GRPO implementation inspired by simple_GRPO.
    """
    G = 4  # completions per prompt
    batch_size = len(prompts)

    # ── Generation phase (llama.cpp, fast) ──
    t0 = time.time()
    all_completions = generator.generate_batch(prompts, n_completions=G)
    gen_time = time.time() - t0

    # ── Reward phase ──
    all_rewards = []
    for i, (completions, ans) in enumerate(zip(all_completions, answers)):
        fmt_r = format_reward(completions)
        cor_r = correctness_reward(completions, [ans] * G)
        # Combined reward (format + correctness)
        combined = [f + c for f, c in zip(fmt_r, cor_r)]
        all_rewards.append(combined)

    # ── Compute advantages (GRPO: normalize within group) ──
    # For each prompt, advantage = (reward - mean) / (std + eps)
    advantages = []
    for rewards in all_rewards:
        r = torch.tensor(rewards, dtype=torch.float32)
        mean_r = r.mean()
        std_r = r.std() + 1e-8
        adv = (r - mean_r) / std_r
        advantages.append(adv)

    # ── Training phase (PyTorch backward) ──
    t1 = time.time()
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )
    optimizer.zero_grad()

    total_loss = 0.0
    n_valid = 0

    for i, (prompt_msgs, completions) in enumerate(zip(prompts, all_completions)):
        for j, (completion, adv) in enumerate(zip(completions, advantages[i])):
            if adv.item() == 0:
                continue  # skip zero-advantage completions

            # Tokenize prompt + completion
            full_text = tokenizer.apply_chat_template(
                prompt_msgs + [{"role": "assistant", "content": completion}],
                tokenize=False,
            )
            tokens = tokenizer(full_text, return_tensors="pt", truncation=True,
                               max_length=1024).to(device)

            # Forward pass
            outputs = model(**tokens, labels=tokens["input_ids"])
            loss = outputs.loss * adv.item()
            loss.backward()
            total_loss += loss.item()
            n_valid += 1

    if n_valid > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_time = time.time() - t1

    # ── Metrics ──
    flat_rewards = [r for group in all_rewards for r in group]
    metrics = {
        "gen_time": gen_time,
        "train_time": train_time,
        "total_time": gen_time + train_time,
        "mean_reward": sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0,
        "loss": total_loss / max(n_valid, 1),
        "n_valid": n_valid,
        "gen_tok_s": sum(len(c) for cs in all_completions for c in cs) / max(gen_time, 0.01),
    }
    return metrics


# ── Dataset ──

def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_prompts(n=None):
    data = load_dataset("openai/gsm8k", "main")["train"]
    prompts, answers = [], []
    for item in data:
        p = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        a = extract_hash_answer(item["answer"])
        prompts.append(p)
        answers.append(a)
        if n and len(prompts) >= n:
            break
    return prompts, answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["convert", "serve", "train", "bench", "all"],
                        default="bench", help="What to do")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gguf", default="./model.gguf", help="GGUF model path")
    parser.add_argument("--server-url", default=LLAMA_SERVER_URL)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    if args.mode == "convert":
        export_lora_to_gguf(args.model, None, args.gguf)
        return

    if args.mode == "serve":
        print(f"Starting llama-server with {args.gguf}...")
        os.execvp("llama-server", [
            "llama-server",
            "-m", args.gguf,
            "--host", "0.0.0.0",
            "--port", "8080",
            "-ngl", "999",  # all layers on GPU
            "-c", "2048",
            "--flash-attn",
        ])
        return

    if args.mode == "bench":
        # Benchmark: just test generation speed
        generator = LlamaCppGenerator(args.server_url, args.max_tokens)
        if not generator.health_check():
            print("ERROR: llama-server not running. Start it first:")
            print(f"  llama-server -m {args.gguf} -ngl 999 -c 2048 --flash-attn")
            return

        prompts, _ = get_gsm8k_prompts(n=5)
        print(f"Benchmarking generation speed ({len(prompts)} prompts, G=4)...")
        t0 = time.time()
        completions = generator.generate_batch(prompts, n_completions=4)
        elapsed = time.time() - t0
        total_tokens = sum(len(c) for cs in completions for c in cs)
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Tokens: {total_tokens}")
        print(f"  Speed: {total_tokens/elapsed:.1f} tok/s")
        print(f"  Per prompt: {elapsed/len(prompts):.1f}s")
        return

    if args.mode in ("train", "all"):
        generator = LlamaCppGenerator(args.server_url, args.max_tokens, temperature=1.0)

        if args.mode == "all":
            # Convert and start server first
            export_lora_to_gguf(args.model, None, args.gguf)
            # TODO: start server in background subprocess

        if not generator.health_check():
            print("ERROR: llama-server not running.")
            return

        # Load PyTorch model for training (backward pass only)
        print(f"Loading {args.model} for training...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = cast_model_to_fp16(model)

        lora_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        device = next(model.parameters()).device

        prompts, answers = get_gsm8k_prompts()
        print(f"Training: {args.max_steps} steps")
        print("=" * 60)

        all_metrics = []
        for step in range(args.max_steps):
            # Sample a batch of prompts
            idx = step % len(prompts)
            batch_prompts = [prompts[idx]]
            batch_answers = [answers[idx]]

            metrics = grpo_step(
                model, tokenizer, generator,
                batch_prompts, batch_answers, device, lr=args.lr,
            )
            all_metrics.append(metrics)

            print(f"  Step {step+1}/{args.max_steps}: "
                  f"loss={metrics['loss']:.4f}, reward={metrics['mean_reward']:.2f}, "
                  f"gen={metrics['gen_time']:.1f}s, train={metrics['train_time']:.1f}s, "
                  f"tok/s={metrics['gen_tok_s']:.0f}")

        # Save metrics
        with open("improvements/01_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics saved to improvements/01_metrics.json")

        avg_total = sum(m["total_time"] for m in all_metrics) / len(all_metrics)
        avg_gen = sum(m["gen_time"] for m in all_metrics) / len(all_metrics)
        avg_train = sum(m["train_time"] for m in all_metrics) / len(all_metrics)
        print(f"Average: {avg_total:.1f}s/step (gen={avg_gen:.1f}s, train={avg_train:.1f}s)")


if __name__ == "__main__":
    main()
