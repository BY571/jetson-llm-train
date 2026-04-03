"""GRPO/DAPO trainer: PPO-style clipped surrogate loss."""

import math

import torch
import torch.nn.functional as F

from triebwerk.trainers.base import BaseTrainer


class GRPOTrainer(BaseTrainer):
    """Group Relative Policy Optimization trainer.

    Supports two loss variants:
      - "grpo": PPO-style clipped surrogate (min of clipped and unclipped)
      - "dapo": Direct clipping without min, skips degenerate groups

    Args:
        loss_type: "grpo" or "dapo" (default "dapo").
        epsilon: Clipping range lower bound (default 0.2).
        epsilon_high: Clipping range upper bound (default same as epsilon).
        **kwargs: Passed to BaseTrainer.
    """

    def __init__(self, loss_type="dapo", **kwargs):
        if loss_type not in ("grpo", "dapo"):
            raise ValueError(f"GRPOTrainer supports loss_type 'grpo' or 'dapo', got '{loss_type}'")
        super().__init__(loss_type=loss_type, **kwargs)

    @property
    def needs_reference_logprobs(self):
        return True

    def compute_loss(self, model, optimizer, scaler, samples):
        """One GRPO/DAPO gradient step with batched forward pass.

        Uses flat token averaging across all completions, matching TRL's
        GRPOTrainer.compute_loss: loss = sum(token_losses) / total_tokens.
        Longer completions contribute more gradient (weighted by token count).
        """
        device = next(model.parameters()).device
        optimizer.zero_grad()

        # Pre-filter valid samples
        valid_samples = [s for s in samples
                         if s["mask_weight"] != 0.0
                         and len(s["completion_ids"]) > 0
                         and s["advantage"] != 0.0]
        if len(valid_samples) == 0:
            return 0.0

        total_tokens = sum(len(s["completion_ids"]) for s in valid_samples)
        if total_tokens == 0:
            return 0.0

        # Per-sequence forward + immediate backward to avoid OOM on long sequences.
        # Each sequence: forward → log-probs → loss → backward → free graph.
        # Gradients accumulate across sequences (no optimizer.zero_grad between them).
        total_loss_val = 0.0

        for i, s in enumerate(valid_samples):
            p_ids = s["prompt_ids"]
            c_ids = s["completion_ids"]
            old_lp = s["old_logprobs"]
            adv = s["advantage"]

            full_ids = p_ids + c_ids
            input_ids = torch.tensor([full_ids], device=device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids)
                logits = outputs.logits[0, :-1, :]
                targets = input_ids[0, 1:]

                # Chunked log-softmax to stay in fp16 memory
                new_lp = torch.zeros(logits.shape[0], device=device)
                chunk = 256
                for ci in range(0, logits.shape[0], chunk):
                    ei = min(ci + chunk, logits.shape[0])
                    lp_c = F.log_softmax(logits[ci:ei].float(), dim=-1)
                    new_lp[ci:ei] = lp_c.gather(1, targets[ci:ei].unsqueeze(1)).squeeze(1)
                    del lp_c

                new_comp_lp = new_lp[len(p_ids) - 1:]

                if self.loss_type == "grpo":
                    ratio = torch.exp(new_comp_lp - old_lp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon_high) * adv
                    per_token_loss = -torch.min(surr1, surr2)
                else:  # dapo
                    ratio = torch.exp(new_comp_lp - old_lp)
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon_high)
                    per_token_loss = -clipped_ratio * adv

                sample_loss = per_token_loss.sum() / total_tokens

            loss_val = sample_loss.item()
            if not (math.isnan(loss_val) or math.isinf(loss_val)):
                scaler.scale(sample_loss).backward()
                total_loss_val += per_token_loss.sum().item()

            del outputs, logits, new_lp, input_ids, sample_loss, per_token_loss, new_comp_lp

        if self.empty_cache:
            torch.cuda.empty_cache()

        # Gradient clipping + optimizer step
        if total_loss_val == 0.0:
            return 0.0
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )
        if math.isnan(grad_norm.item()) or math.isinf(grad_norm.item()):
            print(f"  WARNING: NaN/Inf gradient norm, skipping step")
            optimizer.zero_grad()
            scaler.update()
            return float('nan')
        scaler.step(optimizer)
        scaler.update()

        return total_loss_val / total_tokens
