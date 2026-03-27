"""Compare TRL vs our engine — reads results and prints table."""
import json
import os

results_dir = os.path.join(os.path.dirname(__file__), "results")

trl_path = os.path.join(results_dir, "trl.json")
ours_path = os.path.join(results_dir, "ours.json")

if not os.path.exists(trl_path):
    print("Missing benchmark/results/trl.json — run bench_trl.py first")
if not os.path.exists(ours_path):
    print("Missing benchmark/results/ours.json — run bench_ours.py first")

if os.path.exists(trl_path) and os.path.exists(ours_path):
    trl = json.load(open(trl_path))
    ours = json.load(open(ours_path))

    speedup = trl["avg_step_time_s"] / ours["avg_step_time_s"]

    print("=" * 60)
    print("GRPO Training Benchmark Results")
    print(f"Model: {trl['config']['model']}")
    print(f"Config: G={trl['config']['G']}, tokens={trl['config']['max_tokens']}, "
          f"LoRA rank={trl['config']['lora_rank']}, loss={trl['config']['loss']}")
    print("=" * 60)
    print(f"{'':20s} {'TRL':>12s} {'Ours':>12s} {'Speedup':>10s}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*10}")
    print(f"{'Steps':20s} {trl['steps']:12d} {ours['steps']:12d}")
    print(f"{'Total time':20s} {trl['total_time_s']:11.0f}s {ours['total_time_s']:11.0f}s")
    print(f"{'Avg step time':20s} {trl['avg_step_time_s']:11.1f}s {ours['avg_step_time_s']:11.1f}s {speedup:9.1f}x")
    print(f"{'300-step estimate':20s} {trl['avg_step_time_s']*300/3600:10.1f}h {ours['avg_step_time_s']*300/3600:10.1f}h")
    print("=" * 60)
    print(f"Our engine is {speedup:.1f}x faster than TRL")
