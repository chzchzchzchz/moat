import time
import mlx.core as mx
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("/Users/MohssineChazi2/moat/models/qwen3_5_4b_4bit", model_config={"trust_remote_code": True})

prompt = "Journal Entry: I am feeling overwhelmed and burned out. Insight:"
input_ids = tokenizer.encode(prompt)
prompts = [input_ids for _ in range(4)]

print("Starting batched generation for 4 paths (4B model)...")
start = time.time()
responses = batch_generate(model, tokenizer, prompts=prompts, max_tokens=100, sampler=make_sampler(temp=0.8))
elapsed = time.time() - start

print(f"Total time for 4 paths (max 100 tokens): {elapsed:.2f}s")
print(dir(responses))
