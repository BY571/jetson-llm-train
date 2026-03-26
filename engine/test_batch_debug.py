"""Debug: compare single vs batch decode outputs."""
import sys
sys.path.insert(0, "engine/build2")
import jetson_engine

engine = jetson_engine.Engine(256)
engine.load_weights("engine/weights_q4l")

print("=== SINGLE decode token 3838 ===")
engine.decode_token(3838)

engine.reset()

print("=== BATCH decode token 3838 ===")
results = engine.generate_batch([[3838]], max_new_tokens=1, temperature=0.001, top_p=1.0, eos_token_id=-1)
print(f"Batch result tokens: {results}")
