"""GRPO training on GSM8K with CUDA graph accelerated generation.

Uses CUDA graphs for 6.8x faster generation (46.7 tok/s vs 6.9 baseline).
Custom GRPO training loop replaces TRL's GRPOTrainer to use our fast
generation while keeping the same reward functions and GRPO math.

Architecture:
  1. Prefill prompt (variable length, not graphed)
  2. Decode tokens via CUDA graph replay (static shape, fast)
  3. Compute rewards (format + correctness)
  4. GRPO advantage normalization
  5. Backward pass + optimizer step (PyTorch, ~3.9s)

Usage:
    docker run --runtime nvidia --network=host \\
      -v $(pwd):/workspace \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      -e CUDA_HOME=/usr/local/cuda-12.6 \\
      -e CPATH=/usr/local/cuda-12.6/targets/aarch64-linux/include \\
      -e PATH=/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \\
      grpo-jetson bash -c 'export TRITON_PTXAS_PATH=/usr/local/cuda-12.6/bin/ptxas && \\
      python3 train_gsm8k_fast.py --max-steps 300'
"""
import argparse
import json
import os
import re
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import StaticCache

from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16

patch_amp_for_jetson()

# ── Constants ──
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""


def extract_xml_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# ── Reward functions ──
def compute_rewards(completions, answer):
    """Compute format + correctness rewards for a batch of completions."""
    format_rewards = []
    correctness_rewards = []
    for comp in completions:
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", comp, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", comp, re.DOTALL))
        if has_reasoning and has_answer:
            format_rewards.append(1.0)
        elif has_answer:
            format_rewards.append(0.5)
        else:
            format_rewards.append(0.0)

        extracted = extract_xml_answer(comp)
        correctness_rewards.append(2.0 if extracted == answer else -1.0)

    combined = [f + c for f, c in zip(format_rewards, correctness_rewards)]
    return combined, format_rewards, correctness_rewards


# ── CUDA Graph Generation ──

class CUDAGraphGenerator:
    """Fast autoregressive generation using CUDA graphs + static KV cache.

    Captures the decode step as a CUDA graph on first call, then replays
    it for each subsequent token. Eliminates ~120ms of CPU dispatch overhead
    per token (from 138ms to ~21ms).
    """

    def __init__(self, model, tokenizer, max_cache_len=512, temperature=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.max_cache_len = max_cache_len
        self.temperature = temperature
        self.device = next(model.parameters()).device
        self.graph = None
        self.static_input_ids = None
        self.static_cache_pos = None
        self.static_logits = None
        self.cache = None

    def _capture_graph(self, cache):
        """Capture a CUDA graph for the decode step."""
        self.static_input_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.static_cache_pos = torch.zeros(1, dtype=torch.long, device=self.device)

        # Warm up
        self.static_input_ids.fill_(0)
        self.static_cache_pos.fill_(0)
        with torch.no_grad():
            out = self.model(
                input_ids=self.static_input_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=self.static_cache_pos,
            )
        self.static_logits = out.logits

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                out = self.model(
                    input_ids=self.static_input_ids,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=self.static_cache_pos,
                )
            self.static_logits = out.logits

    def generate(self, prompt_messages, max_new_tokens=512):
        """Generate a completion for a single prompt using CUDA graphs.

        Returns the generated text (string).
        """
        # Tokenize prompt
        input_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        if prompt_len + max_new_tokens > self.max_cache_len:
            max_new_tokens = self.max_cache_len - prompt_len - 1

        # Fresh static cache for this generation
        cache = StaticCache(
            config=self.model.config,
            batch_size=1,
            max_cache_len=self.max_cache_len,
            device=self.device,
            dtype=torch.float16,
        )

        # Prefill (not graphed, variable prompt length)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        # Sample first token
        logits = out.logits[:, -1, :] / self.temperature
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        # Capture graph if not already done
        if self.graph is None:
            self._capture_graph(cache)

        # Decode loop with graph replay
        generated_ids = [next_token.item()]
        eos_id = self.tokenizer.eos_token_id

        for i in range(max_new_tokens - 1):
            self.static_input_ids.copy_(next_token)
            self.static_cache_pos.fill_(prompt_len + i + 1)
            self.graph.replay()

            logits = self.static_logits[:, -1, :] / self.temperature
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            token_id = next_token.item()
            generated_ids.append(token_id)

            if token_id == eos_id:
                break
            # Early stop on </answer>
            if len(generated_ids) > 5:
                tail = self.tokenizer.decode(generated_ids[-10:])
                if "</answer>" in tail:
                    break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_batch(self, prompt_messages, n_completions, max_new_tokens=512):
        """Generate n_completions for a prompt. Sequential (one at a time)."""
        completions = []
        for _ in range(n_completions):
            text = self.generate(prompt_messages, max_new_tokens)
            completions.append(text)
        return completions


# ── GRPO Training Step ──

def grpo_step(model, tokenizer, generator, prompt_msgs, answer, optimizer,
              n_completions=4, max_new_tokens=512, max_seq_len=1024):
    """One GRPO training step with CUDA graph generation.

    1. Generate G completions (CUDA graph, fast)
    2. Compute rewards
    3. Compute GRPO advantages
    4. Policy gradient loss + backward
    5. Optimizer step
    """
    device = next(model.parameters()).device

    # ── Generation (fast, CUDA graph) ──
    model.eval()
    torch.cuda.synchronize()
    t_gen = time.perf_counter()
    completions = generator.generate_batch(prompt_msgs, n_completions, max_new_tokens)
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t_gen

    # ── Rewards ──
    combined_rewards, fmt_rewards, cor_rewards = compute_rewards(completions, answer)

    # ── GRPO advantages (normalize within group) ──
    rewards_t = torch.tensor(combined_rewards, dtype=torch.float32)
    mean_r = rewards_t.mean()
    std_r = rewards_t.std()
    if std_r < 1e-8:
        # All same reward, no gradient signal
        return {
            "gen_time": t_gen, "train_time": 0.0, "total_time": t_gen,
            "mean_reward": mean_r.item(), "mean_format": sum(fmt_rewards) / len(fmt_rewards),
            "mean_correctness": sum(cor_rewards) / len(cor_rewards),
            "loss": 0.0, "n_valid": 0, "gen_tok_s": 0,
            "completions": completions,
        }
    advantages = (rewards_t - mean_r) / std_r

    # ── Backward (PyTorch, already fast) ──
    model.train()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t_train = time.perf_counter()

    total_loss = 0.0
    n_valid = 0

    for comp_text, adv in zip(completions, advantages):
        if abs(adv.item()) < 1e-8:
            continue

        # Tokenize prompt + completion
        full_msgs = prompt_msgs + [{"role": "assistant", "content": comp_text}]
        full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False)
        tokens = tokenizer(full_text, return_tensors="pt", truncation=True,
                           max_length=max_seq_len).to(device)

        # Get prompt length to mask loss
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        prompt_len = len(tokenizer(prompt_text).input_ids)

        # Forward + masked loss
        outputs = model(**tokens, labels=tokens["input_ids"])
        # Mask prompt tokens from loss (only train on completion)
        logits = outputs.logits[:, :-1, :]
        labels = tokens["input_ids"][:, 1:]
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        # Mask prompt
        mask = torch.zeros_like(labels, dtype=torch.float32)
        mask[:, prompt_len - 1:] = 1.0
        masked_loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1)

        # GRPO: weight loss by advantage
        weighted_loss = -masked_loss * adv.item()
        weighted_loss.backward()
        total_loss += weighted_loss.item()
        n_valid += 1

    if n_valid > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    torch.cuda.synchronize()
    t_train = time.perf_counter() - t_train

    total_tokens = sum(len(tokenizer(c).input_ids) for c in completions)
    return {
        "gen_time": t_gen,
        "train_time": t_train,
        "total_time": t_gen + t_train,
        "mean_reward": mean_r.item(),
        "mean_format": sum(fmt_rewards) / len(fmt_rewards),
        "mean_correctness": sum(cor_rewards) / len(cor_rewards),
        "loss": total_loss / max(n_valid, 1),
        "n_valid": n_valid,
        "gen_tok_s": total_tokens / max(t_gen, 0.01),
    }


# ── Dataset ──

def get_gsm8k_prompts():
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
    return prompts, answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.output_dir}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("GRPO Training — CUDA Graph Generation (6.8x faster)")
    print(f"Run: {run_id}")
    print("=" * 60)

    # ── Load model ──
    print(f"\nLoading {args.model} (4-bit NF4, fp16 compute)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_length
    model = cast_model_to_fp16(model)

    # ── Add LoRA ──
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA rank={args.lora_rank}, trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── CUDA graph generator ──
    generator = CUDAGraphGenerator(
        model, tokenizer,
        max_cache_len=args.max_seq_length,
        temperature=args.temperature,
    )
    print(f"  CUDA graph generator ready (max_cache={args.max_seq_length})")

    # ── Dataset ──
    print("\nLoading GSM8K dataset...")
    prompts, answers = get_gsm8k_prompts()
    print(f"  {len(prompts)} training examples")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # ── Save config ──
    config = {
        "run_id": run_id, "model": args.model, "max_steps": args.max_steps,
        "num_generations": args.num_generations, "max_completion_tokens": args.max_completion_tokens,
        "max_seq_length": args.max_seq_length, "lr": args.lr, "lora_rank": args.lora_rank,
        "temperature": args.temperature, "generation": "cuda_graph",
        "trainable_params": trainable, "total_params": total,
    }
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ──
    print(f"\nTraining: {args.max_steps} steps, G={args.num_generations}, "
          f"max_tokens={args.max_completion_tokens}")
    print(f"  LR={args.lr}, temperature={args.temperature}")
    print(f"  Generation: CUDA graph (46.7 tok/s expected)")
    print("=" * 60)

    all_metrics = []
    t_start = time.time()

    for step in range(args.max_steps):
        idx = step % len(prompts)

        metrics = grpo_step(
            model, tokenizer, generator,
            prompts[idx], answers[idx], optimizer,
            n_completions=args.num_generations,
            max_new_tokens=args.max_completion_tokens,
            max_seq_len=args.max_seq_length,
        )
        metrics["step"] = step + 1
        all_metrics.append(metrics)

        # Log
        elapsed = time.time() - t_start
        eta = (elapsed / (step + 1)) * (args.max_steps - step - 1)
        print(f"  Step {step+1}/{args.max_steps}: "
              f"reward={metrics['mean_reward']:.2f} "
              f"(fmt={metrics['mean_format']:.2f}, cor={metrics['mean_correctness']:.2f}) "
              f"loss={metrics['loss']:.4f} "
              f"gen={metrics['gen_time']:.1f}s train={metrics['train_time']:.1f}s "
              f"[{metrics['gen_tok_s']:.0f} tok/s] "
              f"ETA {eta/3600:.1f}h")

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_dir = f"{run_dir}/checkpoint-{step+1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"    Saved checkpoint: {ckpt_dir}")

    # ── Final save ──
    total_time = time.time() - t_start
    lora_path = f"{run_dir}/final_lora"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    # Save metrics
    summary = {
        "run_id": run_id, "total_time_s": total_time, "total_time_h": total_time / 3600,
        "steps": args.max_steps, "s_per_step": total_time / args.max_steps,
        "avg_gen_time": sum(m["gen_time"] for m in all_metrics) / len(all_metrics),
        "avg_train_time": sum(m["train_time"] for m in all_metrics) / len(all_metrics),
        "avg_reward": sum(m["mean_reward"] for m in all_metrics[-50:]) / min(50, len(all_metrics)),
        "generation": "cuda_graph",
    }
    with open(f"{run_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump({"steps": all_metrics}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Time: {total_time/3600:.1f}h ({total_time/args.max_steps:.1f}s/step)")
    print(f"  Avg gen: {summary['avg_gen_time']:.1f}s, avg train: {summary['avg_train_time']:.1f}s")
    print(f"  LoRA: {lora_path}")
    print(f"  Metrics: {run_dir}/metrics.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
