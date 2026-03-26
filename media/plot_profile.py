"""Plot profiling results from improvement 01."""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("improvements/01_profile_results.json") as f:
    p = json.load(f)

gen = p["generation"]
bwd = p["backward"]
step = p["step_estimate"]

# ── Plot 1: Time breakdown pie chart ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Pie: generation vs backward
sizes = [step["generation_s"], step["backward_s"]]
labels = [f'Generation\n{step["generation_s"]:.0f}s ({step["generation_pct"]:.0f}%)',
          f'Backward+Opt\n{step["backward_s"]:.1f}s ({100-step["generation_pct"]:.0f}%)']
colors = ["#e74c3c", "#2ecc71"]
explode = (0.05, 0)
ax1.pie(sizes, labels=labels, colors=colors, explode=explode, startangle=90,
        textprops={"fontsize": 11, "fontweight": "bold"})
ax1.set_title("Time per GRPO Step", fontsize=13, fontweight="bold")

# Bar: backward breakdown
bwd_items = [
    ("Forward", bwd["forward_ms"], "#3498db"),
    ("Backward", bwd["backward_ms"], "#e67e22"),
    ("Optimizer", bwd["optimizer_ms"], "#9b59b6"),
]
names = [x[0] for x in bwd_items]
vals = [x[1] for x in bwd_items]
cols = [x[2] for x in bwd_items]
bars = ax2.barh(names, vals, color=cols, height=0.5)
for bar, v in zip(bars, vals):
    ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             f"{v:.0f}ms", va="center", fontsize=10)
ax2.set_xlabel("Time (ms)", fontsize=10)
ax2.set_title("Backward Pass Breakdown (per completion)", fontsize=13, fontweight="bold")
ax2.set_xlim(0, max(vals) * 1.3)
ax2.grid(True, alpha=0.15, axis="x")

plt.tight_layout()
plt.savefig("improvements/plot_profile_breakdown.png", dpi=150, bbox_inches="tight")
print("Saved: improvements/plot_profile_breakdown.png")
plt.close()

# ── Plot 2: Speedup potential ──
fig, ax = plt.subplots(figsize=(12, 5))

tok_speeds = [4.2, 10, 20, 40, 80, 160, 400]
avg_tokens = gen["avg_tokens_per_completion"]
n_completions = 4

step_times = []
for tps in tok_speeds:
    gen_time = (avg_tokens / tps) * n_completions
    total = gen_time + step["backward_s"]
    step_times.append(total)

bars = ax.bar(range(len(tok_speeds)), step_times,
              color=["#e74c3c"] + ["#3498db"] * (len(tok_speeds) - 1),
              edgecolor="#333", linewidth=0.5)

# Annotate
for i, (tps, st) in enumerate(zip(tok_speeds, step_times)):
    speedup = step_times[0] / st
    label = f"{st:.0f}s" if st >= 1 else f"{st:.1f}s"
    ax.text(i, st + 5, label, ha="center", fontsize=9, fontweight="bold")
    if i > 0:
        ax.text(i, st + 18, f"{speedup:.0f}x", ha="center", fontsize=8, color="#666")

ax.set_xticks(range(len(tok_speeds)))
ax.set_xticklabels([f"{t} tok/s" for t in tok_speeds], fontsize=9)
ax.set_ylabel("Step time (seconds)", fontsize=11)
ax.set_xlabel("Generation speed", fontsize=11)
ax.set_title("GRPO Step Time vs Generation Speed\n"
             f"(baseline: 4.2 tok/s, {avg_tokens:.0f} tokens avg, G=4, backward=3.9s fixed)",
             fontsize=12, fontweight="bold")
ax.axhline(step["backward_s"], color="#2ecc71", linewidth=1.5, linestyle="--",
           label=f'Theoretical min: {step["backward_s"]:.1f}s (backward only)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15, axis="y")

plt.savefig("improvements/plot_profile_speedup.png", dpi=150, bbox_inches="tight")
print("Saved: improvements/plot_profile_speedup.png")
plt.close()
