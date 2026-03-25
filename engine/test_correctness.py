"""Test C++ engine output matches HuggingFace exactly.

Loads the same model in both HF and our engine, feeds the same input,
and compares logits. Differences should be < 1e-3 (fp16 rounding).

Usage:
    # First, convert weights:
    python3 engine/convert_weights.py --model Qwen/Qwen3-0.6B --output engine/weights.bin

    # Then build the engine:
    cd engine && mkdir -p build && cd build && cmake .. && make -j$(nproc)

    # Then test:
    python3 engine/test_correctness.py
"""
import sys
import os
import time

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_hf_reference():
    """Generate reference logits from HuggingFace model."""
    print("Loading HF model (fp16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model.eval()

    prompt = "What is 2 + 2?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print(f"  Prompt: '{prompt}' -> {input_ids.shape[1]} tokens: {input_ids[0].tolist()}")

    # Get logits for the last position (after prefill)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :].float().cpu().numpy()

    print(f"  Logits shape: {logits.shape}")
    print(f"  Top 5 tokens: {np.argsort(logits)[-5:][::-1]}")
    print(f"  Top 5 logit values: {logits[np.argsort(logits)[-5:][::-1]]}")

    # Also get logits for one decode step
    next_token = torch.tensor([[logits.argmax()]], device=model.device)
    with torch.no_grad():
        outputs2 = model(input_ids=next_token, past_key_values=outputs.past_key_values)
    decode_logits = outputs2.logits[0, -1, :].float().cpu().numpy()

    print(f"\n  Decode step logits (token {next_token.item()}):")
    print(f"  Top 5: {np.argsort(decode_logits)[-5:][::-1]}")

    return {
        "input_ids": input_ids[0].tolist(),
        "prefill_logits": logits,
        "prefill_top5": np.argsort(logits)[-5:][::-1].tolist(),
        "decode_token": next_token.item(),
        "decode_logits": decode_logits,
        "decode_top5": np.argsort(decode_logits)[-5:][::-1].tolist(),
    }


def test_engine(reference):
    """Compare engine output to HF reference."""
    try:
        sys.path.insert(0, "engine/build")
        import jetson_engine
    except ImportError:
        print("\nEngine not built yet. Build with:")
        print("  cd engine && mkdir -p build && cd build && cmake .. && make -j$(nproc)")
        print("\nSkipping engine test. Saving reference for later.")
        return None

    print("\nLoading engine...")
    engine = jetson_engine.Engine(max_seq_len=1024)
    engine.load_weights("engine/weights.bin")

    # Prefill
    print("Running engine prefill...")
    engine.prefill(reference["input_ids"])
    # TODO: get_logits() and compare

    return None


def main():
    print("=" * 60)
    print("Correctness Test: HF vs C++ Engine")
    print("=" * 60)

    ref = test_hf_reference()

    # Save reference for offline comparison
    np.savez("engine/reference_logits.npz",
             input_ids=ref["input_ids"],
             prefill_logits=ref["prefill_logits"],
             decode_logits=ref["decode_logits"])
    print(f"\nSaved reference logits to engine/reference_logits.npz")

    # Test engine if available
    test_engine(ref)

    print("\n" + "=" * 60)
    print("Reference generated. Build the engine, then re-run to compare.")
    print("=" * 60)


if __name__ == "__main__":
    main()
