import time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import numpy as np
import copy

print("Loading 4B Model for MCTS...")
model, tokenizer = load("/Users/MohssineChazi2/moat/models/qwen3_5_4b_4bit", model_config={"trust_remote_code": True})

def score_chunk(text: str) -> float:
    # Basic Process Reward heuristic for a chunk of text
    words = text.split()
    if not words: return -999.0
    diversity = len(set(words)) / len(words)
    return diversity * 5.0 + np.log1p(len(words))

prompt = "<|im_start|>system\nYou are an insightful private journaling assistant.<|im_end|>\n<|im_start|>user\nJournal Entry:\nI've been feeling overwhelmed and burned out. What should I do?<|im_end|>\n<|im_start|>assistant\nInsight:\n"
input_ids = tokenizer.encode(prompt)

# MCTS / Beam Search Configuration
CHUNK_TOKENS = 30
NUM_CHUNKS = 4
BRANCHES_PER_CHUNK = 3

print(f"\nStarting MCTS (Chunks: {NUM_CHUNKS}, Tokens/Chunk: {CHUNK_TOKENS}, Branches: {BRANCHES_PER_CHUNK})")

# We will maintain a list of active "nodes"
# Each node: {"tokens": [int], "cache": List[Any], "score": float}
# Unfortunately, mlx_lm `generate_step` handles caches in-place. 
# We'll use a hack to copy the caches, but MLX arrays share memory nicely until modified.
# Actually, the easiest way to do MCTS chunking in Python is to pass the text prefix back to `generate`
# because `generate` automatically recomputes KV caches very quickly, or we can use `batch_generate`.

start = time.time()
best_prefix = input_ids

for chunk in range(NUM_CHUNKS):
    print(f"\n--- Expanding Chunk {chunk+1}/{NUM_CHUNKS} ---")
    prompts = [best_prefix for _ in range(BRANCHES_PER_CHUNK)]
    
    # We use generate to extend the prefix for each branch
    # We'll use high temperature to get diverse branches
    branches_text = []
    branches_tokens = []
    
    for i in range(BRANCHES_PER_CHUNK):
        # Generate extension
        sampler = make_sampler(temp=0.9)
        ext = generate(model, tokenizer, best_prefix, max_tokens=CHUNK_TOKENS, sampler=sampler, verbose=False)
        # ext is just the new text generated. We need to append it.
        ext_tokens = tokenizer.encode(ext)
        branches_tokens.append(best_prefix + ext_tokens)
        branches_text.append(tokenizer.decode(best_prefix) + ext)
        
    # Evaluate branches
    best_branch_score = -9999
    best_branch_idx = 0
    for i in range(BRANCHES_PER_CHUNK):
        # We only score the NEW text chunk to see if it's a good direction
        new_text = tokenizer.decode(branches_tokens[i][len(best_prefix):])
        score = score_chunk(new_text)
        print(f"Branch {i} Score: {score:.2f} | Snippet: {new_text[:40].strip()}...")
        if score > best_branch_score:
            best_branch_score = score
            best_branch_idx = i
            
    # Prune and select best branch
    best_prefix = branches_tokens[best_branch_idx]
    print(f"-> Selected Branch {best_branch_idx}")

elapsed = time.time() - start
print(f"\nCompleted MCTS Generation in {elapsed:.2f}s")
print("\nFINAL MCTS OUTPUT:")
print(tokenizer.decode(best_prefix))
