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

        # Build padded batch for single forward pass
        seqs = [s["prompt_ids"] + s["completion_ids"] for s in valid_samples]
        max_len = max(len(s) for s in seqs)
        pad_id = 0
        padded = []
        masks = []
        for s in seqs:
            pad_len = max_len - len(s)
            padded.append([pad_id] * pad_len + s)
            masks.append([0] * pad_len + [1] * len(s))

        input_ids = torch.tensor(padded, device=device)
        attention_mask = torch.tensor(masks, device=device)

        # Single forward pass
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]  # (batch, seq_len-1, vocab)
            targets = input_ids[:, 1:]  # (batch, seq_len-1)

            # Chunked log-softmax to avoid fp32 OOM
            batch_size = logits.shape[0]
            new_lp = torch.zeros(batch_size, logits.shape[1], device=device)
            chunk = 256
            for ci in range(0, logits.shape[1], chunk):
                ei = min(ci + chunk, logits.shape[1])
                lp_c = F.log_softmax(logits[:, ci:ei].float(), dim=-1)
                new_lp[:, ci:ei] = lp_c.gather(2, targets[:, ci:ei].unsqueeze(2)).squeeze(2)
                del lp_c

        # Accumulate loss across all samples, then single backward
        total_loss = torch.tensor(0.0, device=device)
        total_loss_val = 0.0
        for i, s in enumerate(valid_samples):
            p_len = len(s["prompt_ids"])
            c_len = len(s["completion_ids"])
            old_lp = s["old_logprobs"]
            adv = s["advantage"]
            seq_len = p_len + c_len
            pad_len = max_len - seq_len
            new_comp_lp = new_lp[i, pad_len + p_len - 1 : pad_len + p_len - 1 + c_len]

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
            if not (math.isnan(sample_loss.item()) or math.isinf(sample_loss.item())):
                total_loss = total_loss + sample_loss
                total_loss_val += per_token_loss.sum().item()

        if total_loss.item() != 0.0:
            scaler.scale(total_loss).backward()

        del outputs, logits, new_lp, input_ids, attention_mask, total_loss

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
