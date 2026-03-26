"""Profile per-operation decode time breakdown using CUDA events."""
import sys, os
sys.path.insert(0, os.environ.get("ENGINE_BUILD", "engine/build2"))
import jetson_engine

engine = jetson_engine.Engine(256)
engine.load_weights("engine/weights_nf4")

# Warmup
for i in range(5):
    engine.decode_token(i)
engine.sample(0.001, 1.0)
engine.reset()

# Prefill
prompt = [3838, 374, 220, 17, 488, 220, 17, 30]
for t in prompt:
    engine.decode_token(t)

# Profile 10 decode steps, average
all_profiles = []
for i in range(10):
    profile = engine.profile_decode(i % 100)
    all_profiles.append(profile)

# Average
avg = {}
for prof in all_profiles:
    for k, v in prof.items():
        avg[k] = avg.get(k, 0) + v
for k in avg:
    avg[k] /= len(all_profiles)

print("=== Per-token decode breakdown (avg of 10, CUDA events) ===")
total = 0
for name, us in sorted(avg.items(), key=lambda x: -x[1]):
    ms = us / 1000
    total += us
    print(f"  {name:25s}  {us:8.0f} us  ({ms:6.2f} ms)")
print(f"  {'TOTAL':25s}  {total:8.0f} us  ({total/1000:6.2f} ms)")
print(f"\n  Predicted tok/s: {1000/(total/1000):.1f}")
