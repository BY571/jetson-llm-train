"""Micro-benchmark: which part of the decode loop is slow?"""
import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import StaticCache
from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16

patch_amp_for_jetson()

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", quantization_config=bnb_config,
                                              device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
cast_model_to_fp16(model)
model.eval()
device = next(model.parameters()).device

input_ids = tokenizer("What is 25 * 4?", return_tensors="pt").input_ids.to(device)
prompt_len = input_ids.shape[1]

# Setup static cache + graph
cache = StaticCache(config=model.config, batch_size=1, max_cache_len=512,
                     device=device, dtype=torch.float16)
with torch.no_grad():
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)

static_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
static_pos = torch.zeros(1, dtype=torch.long, device=device)
static_ids.fill_(0)
static_pos.fill_(prompt_len)
with torch.no_grad():
    out = model(input_ids=static_ids, past_key_values=cache, use_cache=True,
                cache_position=static_pos)
static_logits = out.logits

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    with torch.no_grad():
        out = model(input_ids=static_ids, past_key_values=cache, use_cache=True,
                    cache_position=static_pos)
    static_logits = out.logits

N = 100
next_tok = torch.zeros(1, 1, dtype=torch.long, device=device)

print("Micro-benchmarks (100 iterations each):\n")

# 1. Graph replay only
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(N):
    static_pos.fill_(prompt_len + i + 1)
    graph.replay()
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / N * 1000
print(f"  Graph replay only:            {ms:6.2f}ms/tok  ({1000/ms:5.0f} tok/s)")

# 2. Graph + argmax (GPU only, no .item)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(N):
    static_ids.copy_(next_tok)
    static_pos.fill_(prompt_len + i + 1)
    graph.replay()
    next_tok = static_logits[:, -1, :].argmax(dim=-1, keepdim=True)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / N * 1000
print(f"  Graph + argmax (GPU):          {ms:6.2f}ms/tok  ({1000/ms:5.0f} tok/s)")

# 3. Graph + multinomial (GPU only)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(N):
    static_ids.copy_(next_tok)
    static_pos.fill_(prompt_len + i + 1)
    graph.replay()
    logits = static_logits[:, -1, :] / 1.0
    probs = F.softmax(logits, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / N * 1000
print(f"  Graph + multinomial (GPU):     {ms:6.2f}ms/tok  ({1000/ms:5.0f} tok/s)")

# 4. Graph + multinomial + .item() (CPU sync)
gen_ids = []
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(N):
    static_ids.copy_(next_tok)
    static_pos.fill_(prompt_len + i + 1)
    graph.replay()
    logits = static_logits[:, -1, :] / 1.0
    probs = F.softmax(logits, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1)
    tid = next_tok.item()  # CPU sync!
    gen_ids.append(tid)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / N * 1000
print(f"  Graph + sample + .item():      {ms:6.2f}ms/tok  ({1000/ms:5.0f} tok/s)")

# 5. Graph + sample + .item + stop check
stop_ids = tokenizer.encode("</answer>", add_special_tokens=False)
stop_len = len(stop_ids)
gen_ids = list(range(50))
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(N):
    static_ids.copy_(next_tok)
    static_pos.fill_(prompt_len + i + 1)
    graph.replay()
    logits = static_logits[:, -1, :] / 1.0
    probs = F.softmax(logits, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1)
    tid = next_tok.item()
    gen_ids.append(tid)
    _ = gen_ids[-stop_len:] == stop_ids
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / N * 1000
print(f"  Graph + sample + stop check:   {ms:6.2f}ms/tok  ({1000/ms:5.0f} tok/s)")

# 6. Full loop WITHOUT .item (accumulate on GPU, check every 10 tokens)
all_tokens = torch.zeros(N, dtype=torch.long, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(N):
    static_ids.copy_(next_tok)
    static_pos.fill_(prompt_len + i + 1)
    graph.replay()
    logits = static_logits[:, -1, :] / 1.0
    probs = F.softmax(logits, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1)
    all_tokens[i] = next_tok.squeeze()
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / N * 1000
print(f"  Graph + sample (no .item):     {ms:6.2f}ms/tok  ({1000/ms:5.0f} tok/s)")
