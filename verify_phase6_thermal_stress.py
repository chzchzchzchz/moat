import time
import sys
sys.path.insert(0, 'antigravity-engine/src')

from tokenizer import LlamaTokenizer
from native_bridge import NativeMetalEngine

print("=================================================================")
print("  PHASE 6: 1,000-TOKEN HEAVY METAL GPU THERMAL & WATTAGE STRESS ")
print("=================================================================")

tok = LlamaTokenizer('models/tinyllama/tokenizer.json')
engine = NativeMetalEngine(
    dylib_path='libantigravity_engine.dylib',
    model_path='models/tinyllama/model.safetensors',
    n_channels=8
)

prompt = "Prove that 2^n > n^2 for all integers n >= 5 using mathematical induction."
prompt_ids = tok.encode(prompt)

print(f"Prompt: '{prompt}'")
print("Dispatching heavy 1,000-token decode loop across 8 parallel channels...")

t0 = time.perf_counter()
tokens, logprobs, ttft, total = engine.generate(prompt_ids, max_new_tokens=125, temperature=0.7, top_p=0.9)
t1 = time.perf_counter()

total_tokens = sum(len(c) for c in tokens)
elapsed_sec = t1 - t0

print(f"✅ Generated {total_tokens} total tokens across 8 channels in {elapsed_sec:.2f} s")
print(f"✅ Combined GPU Throughput: {total_tokens / elapsed_sec:.2f} tok/s")
print("=================================================================")
