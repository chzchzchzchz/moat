import time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

print("Loading models...")
target_model, tokenizer = load("/Users/MohssineChazi2/moat/models/qwen3_5_4b_4bit", model_config={"trust_remote_code": True})
draft_model, _ = load("/Users/MohssineChazi2/moat/models/qwen3_5_0_8b_4bit", model_config={"trust_remote_code": True})

prompt = "Explain the history of the universe in deep detail, covering the Big Bang, inflation, nucleosynthesis, and recombination."
print("\n--- STANDARD GENERATION ---")
start = time.time()
generate(target_model, tokenizer, prompt, max_tokens=200, verbose=True)
standard_time = time.time() - start

print("\n--- SPECULATIVE GENERATION ---")
start = time.time()
generate(target_model, tokenizer, prompt, max_tokens=200, draft_model=draft_model, verbose=True)
speculative_time = time.time() - start

print(f"\nStandard Time: {standard_time:.2f}s")
print(f"Speculative Time: {speculative_time:.2f}s")
print(f"Speedup: {standard_time / speculative_time:.2f}x")
