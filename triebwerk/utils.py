"""Core GRPO utilities: advantage computation and log-prob computation."""

import torch
import torch.nn.functional as F


def compute_advantages(rewards, num_generations):
    """Per-group advantage normalization.

    Groups rewards by prompt (G completions each), normalizes within group.
    Returns zero advantages for groups with zero variance (DAPO dynamic sampling).
    """
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)

    # Zero-variance groups get zero advantage (no gradient signal)
    advantages = torch.where(
        std > 1e-8,
        (grouped - mean) / std,
        torch.zeros_like(grouped),
    )
    return advantages.view(-1)


def compute_token_logprobs(model, prompt_ids, completion_ids, device):
    """Forward pass to get per-token log-probs for the completion.

    Returns (completion_len,) tensor of log-probs.
    """
    if len(completion_ids) == 0:
        return torch.zeros(0, device=device)

    full_ids = prompt_ids + completion_ids
    input_tensor = torch.tensor([full_ids], device=device)

    # Use same AMP context as grpo_step to avoid ratio bias from precision mismatch
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(input_tensor)
        logits = outputs.logits[0, :-1, :]
    targets = input_tensor[0, 1:]

    # fp32 softmax for numerical stability
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Keep only completion portion
    comp_lp = token_lp[len(prompt_ids) - 1:]

    del outputs, logits, log_probs, token_lp, input_tensor
    return comp_lp


def compute_batch_token_logprobs(model, completions, device):
    """Batched forward pass for all G completions at once.

    Does a single forward pass with all sequences padded and batched,
    then extracts per-token log-probs for each completion.
    Returns list of (completion_len_i,) tensors.
    """
    # Build padded batch (left-pad so completion tokens align at the right)
    seqs = [c["prompt_ids"] + c["completion_ids"] for c in completions]
    max_len = max(len(s) for s in seqs)
    prompt_lens = [len(c["prompt_ids"]) for c in completions]
    comp_lens = [len(c["completion_ids"]) for c in completions]

    # Pad sequences and create attention mask
    pad_id = 0
    padded = []
    masks = []
    for s in seqs:
        pad_len = max_len - len(s)
        padded.append([pad_id] * pad_len + s)
        masks.append([0] * pad_len + [1] * len(s))

    input_ids = torch.tensor(padded, device=device)
    attention_mask = torch.tensor(masks, device=device)

    # Single forward pass for ALL sequences at once
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(input_ids, attention_mask=attention_mask)

    # Gather log-probs for target tokens only (1 float per position, not 151K)
    seq_logits = outputs.logits[:, :-1, :]  # (batch, seq_len-1, vocab)
    seq_targets = input_ids[:, 1:]  # (batch, seq_len-1)
    batch_size = seq_logits.shape[0]

    # Compute log-softmax in chunks to avoid fp32 OOM on full vocab
    tok_lp = torch.zeros(batch_size, seq_logits.shape[1], device=device)
    chunk = 256  # process 256 positions at a time
    for c in range(0, seq_logits.shape[1], chunk):
        end_c = min(c + chunk, seq_logits.shape[1])
        lp_chunk = F.log_softmax(seq_logits[:, c:end_c].float(), dim=-1)
        tok_lp[:, c:end_c] = lp_chunk.gather(2, seq_targets[:, c:end_c].unsqueeze(2)).squeeze(2)
        del lp_chunk

    del outputs, seq_logits

    # Extract completion portions for each sequence
    result = []
    for i in range(len(completions)):
        pad_len = max_len - len(seqs[i])
        start = pad_len + prompt_lens[i] - 1
        end = start + comp_lens[i]
        result.append(tok_lp[i, start:end].detach())

    del tok_lp, input_ids, attention_mask
    return result
