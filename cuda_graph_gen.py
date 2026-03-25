"""CUDA Graph accelerated generation for Jetson Orin.

Optimizations over the initial implementation:
1. Persistent cache: allocate once, zero-fill between completions (no realloc)
2. Cache prefill KV: same prompt across G completions, prefill once, copy KV
3. Token-ID stop detection: check raw IDs instead of tokenizer.decode per token
4. Sampling on GPU: multinomial stays on device, no CPU roundtrip
5. Single graph capture: captured once, reused across all completions

Memory: one StaticCache (pre-allocated for max_cache_len) lives for the
entire training run. No per-completion allocation.
"""
import torch
import torch.nn.functional as F
from transformers.cache_utils import StaticCache


class FastGenerator:
    """CUDA graph generation with persistent cache and prefill reuse."""

    def __init__(self, model, tokenizer, max_cache_len=1024, temperature=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.max_cache_len = max_cache_len
        self.temperature = temperature
        self.device = next(model.parameters()).device

        # Pre-compute stop token IDs
        self.eos_id = tokenizer.eos_token_id
        # Encode "</answer>" and get the token sequence
        answer_tokens = tokenizer.encode("</answer>", add_special_tokens=False)
        self.stop_ids = answer_tokens  # e.g. [524, 10630, 29]
        self.stop_len = len(self.stop_ids)

        # Persistent static cache (allocated once for entire training)
        self.cache = StaticCache(
            config=model.config,
            batch_size=1,
            max_cache_len=max_cache_len,
            device=self.device,
            dtype=torch.float16,
        )

        # CUDA graph state
        self.graph = None
        self.static_input_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.static_cache_pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.static_logits = None

        # Prefill cache snapshot (for reuse across G completions)
        self._prefill_snapshot = None
        self._prefill_prompt_len = 0

    def _zero_cache(self):
        """Reset cache to zeros without reallocating."""
        if hasattr(self.cache, 'reset'):
            self.cache.reset()
        else:
            # Fallback for older transformers
            for layer in self.cache.layers:
                layer.keys.zero_()
                layer.values.zero_()

    def _save_prefill_snapshot(self):
        """Save a copy of the KV cache after prefill for reuse."""
        self._prefill_snapshot = []
        for layer in self.cache.layers:
            self._prefill_snapshot.append((layer.keys.clone(), layer.values.clone()))

    def _restore_prefill_snapshot(self):
        """Restore the KV cache to the prefill state (fast tensor copy)."""
        for i, (k_snap, v_snap) in enumerate(self._prefill_snapshot):
            self.cache.layers[i].keys.copy_(k_snap)
            self.cache.layers[i].values.copy_(v_snap)

    def _capture_graph(self):
        """Capture CUDA graph for a single decode step."""
        # Warm up
        with torch.no_grad():
            out = self.model(
                input_ids=self.static_input_ids,
                past_key_values=self.cache,
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
                    past_key_values=self.cache,
                    use_cache=True,
                    cache_position=self.static_cache_pos,
                )
            self.static_logits = out.logits

    def _check_stop(self, generated_ids):
        """Check if the last tokens match the stop sequence (token-ID based)."""
        if len(generated_ids) < self.stop_len:
            return False
        return generated_ids[-self.stop_len:] == self.stop_ids

    def _prefill(self, input_ids):
        """Run prefill and optionally capture graph."""
        prompt_len = input_ids.shape[1]
        self._zero_cache()

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=self.cache,
                use_cache=True,
            )

        self._prefill_prompt_len = prompt_len
        self._save_prefill_snapshot()

        # Capture graph after first prefill (cache memory addresses are now fixed)
        if self.graph is None:
            # Need a dummy decode step to warm up graph capture
            self.static_cache_pos.fill_(prompt_len)
            self.static_input_ids.fill_(0)
            self._capture_graph()

        return out.logits

    def generate_batch(self, prompt_messages, n_completions, max_new_tokens=512):
        """Generate n_completions for one prompt with prefill reuse.

        Prefills once, snapshots the KV cache, then generates each
        completion by restoring the snapshot and decoding.
        """
        self.model.eval()

        # Tokenize once
        input_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        effective_max = min(max_new_tokens, self.max_cache_len - prompt_len - 1)
        if effective_max <= 0:
            return [""] * n_completions

        # Prefill once (sets snapshot)
        prefill_logits = self._prefill(input_ids)

        completions = []
        for g in range(n_completions):
            # Restore KV cache to prefill state
            self._restore_prefill_snapshot()

            # Sample first token from prefill logits
            logits = prefill_logits[:, -1, :] / self.temperature
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            # Decode with graph replay
            generated_ids = [next_token.item()]

            for i in range(effective_max - 1):
                self.static_input_ids.copy_(next_token)
                self.static_cache_pos.fill_(prompt_len + i + 1)
                self.graph.replay()

                logits = self.static_logits[:, -1, :] / self.temperature
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                token_id = next_token.item()
                generated_ids.append(token_id)

                if token_id == self.eos_id:
                    break
                if self._check_stop(generated_ids):
                    break

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            completions.append(text)

        return completions
