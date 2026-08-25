import sys
sys.path.insert(0, 'antigravity-engine/src')

import torch
from transformer import TinyLlamaModel
from tokenizer import LlamaTokenizer

tok = LlamaTokenizer('models/tinyllama/tokenizer.json')
prompt = "The capital of France is"
prompt_ids = tok.encode(prompt)

pt_model = TinyLlamaModel.from_safetensors('models/tinyllama/model.safetensors', device='cpu')
pt_model.eval()

p_tensor = torch.tensor([prompt_ids], dtype=torch.long)
h = pt_model.embed_tokens(p_tensor)

print("Prompt Token 0 ID:", prompt_ids[0])
print("Embedding slice (first 10 elements of token 0):")
print(h[0, 0, :10].detach().numpy())

norm_h = pt_model.layers[0].input_layernorm(h)
print("\nLayer 0 input_layernorm slice (first 10 elements):")
print(norm_h[0, 0, :10].detach().numpy())

q = pt_model.layers[0].self_attn.q_proj(norm_h)
k = pt_model.layers[0].self_attn.k_proj(norm_h)
v = pt_model.layers[0].self_attn.v_proj(norm_h)

print("\nLayer 0 Q proj slice (first 10 elements):")
print(q[0, 0, :10].detach().numpy())
print("Layer 0 K proj slice (first 10 elements):")
print(k[0, 0, :10].detach().numpy())
print("Layer 0 V proj slice (first 10 elements):")
print(v[0, 0, :10].detach().numpy())
