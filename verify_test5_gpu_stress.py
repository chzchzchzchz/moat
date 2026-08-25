import time
import sys
sys.path.insert(0, 'antigravity-engine/src')

from tokenizer import LlamaTokenizer
from native_bridge import NativeMetalEngine

print("=================================================================")
print("   TEST 5: CONTINUOUS METAL GPU STRESS TEST (8 PARALLEL CHANNELS)")
print("=================================================================")

tok = LlamaTokenizer('models/tinyllama/tokenizer.json')
engine = NativeMetalEngine(
    dylib_path='libantigravity_engine.dylib',
    model_path='models/tinyllama/model.safetensors',
    n_channels=8
)

prompt = "Analyze the physical thermodynamics of GPU compute tiles under heavy workloads."
prompt_ids = tok.encode(prompt)

t0 = time.perf_counter()
print("Launching sustained 8-channel decode across 128 tokens...")
tokens, logprobs, ttft, total = engine.generate(prompt_ids, max_new_tokens=128, temperature=0.7, top_p=0.9)
t1 = time.perf_counter()

total_toks = sum(len(c) for c in tokens)
elapsed_sec = t1 - t0
print(f"Generated {total_toks} tokens in {elapsed_sec:.2f} s ({total_toks / elapsed_sec:.2f} tok/s)")
print("Sustained GPU execution complete.")
