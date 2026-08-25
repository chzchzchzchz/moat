import sys
sys.path.insert(0, 'antigravity-engine/src')

import torch
from transformer import TinyLlamaModel
from tokenizer import LlamaTokenizer
from native_bridge import NativeMetalEngine

tok = LlamaTokenizer('models/tinyllama/tokenizer.json')
prompt = "The capital of France is"
prompt_ids = tok.encode(prompt)

print(f"Prompt text: '{prompt}'")
print(f"Prompt token IDs: {prompt_ids}")

# 1. PyTorch MPS Model
print("\n--- 1. PyTorch MPS Forward Pass ---")
pt_model = TinyLlamaModel.from_safetensors('models/tinyllama/model.safetensors', device='cpu')
pt_model.eval()

p_tensor = torch.tensor([prompt_ids], dtype=torch.long)
with torch.no_grad():
    pt_logits, _ = pt_model(p_tensor)
    pt_last_logits = pt_logits[0, -1, :]
    pt_top_vals, pt_top_indices = torch.topk(pt_last_logits, 5)

print("PyTorch Top 5 next tokens:")
for val, idx in zip(pt_top_vals, pt_top_indices):
    print(f"  Token {idx.item():5d} ({tok.decode([idx.item()]):15s}): logit = {val.item():.4f}")

# 2. Native C++ Metal Engine
print("\n--- 2. Native C++ Metal Engine Forward Pass ---")
metal_engine = NativeMetalEngine(
    dylib_path='libantigravity_engine.dylib',
    model_path='models/tinyllama/model.safetensors',
    n_channels=8
)

tokens, logprobs, ttft, total = metal_engine.generate(prompt_ids, max_new_tokens=10, temperature=0.0, top_p=1.0)
print("Metal Engine Generated Tokens:", tokens[0])
print("Metal Engine Decoded Text:", tok.decode(tokens[0]))
