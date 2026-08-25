import sys
sys.path.insert(0, 'antigravity-engine/src')

import torch
import numpy as np
from transformer import TinyLlamaModel
from tokenizer import LlamaTokenizer

tok = LlamaTokenizer('models/tinyllama/tokenizer.json')
prompt = "The capital of France is"
prompt_ids = tok.encode(prompt)

pt_model = TinyLlamaModel.from_safetensors('models/tinyllama/model.safetensors', device='cpu')
pt_model.eval()

p_tensor = torch.tensor([prompt_ids], dtype=torch.long)
with torch.no_grad():
    h = pt_model.embed_tokens(p_tensor)
    print(f"Token 0 (BOS) Embed L2 Norm: {h[0, 0].norm().item():.4f}")
    print(f"Token 5 (is)  Embed L2 Norm: {h[0, 5].norm().item():.4f}")
    
    freqs_cis = pt_model.freqs_cis[:6].to(h.device)
    mask = torch.full((6, 6), float("-inf"), device=h.device)
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    
    for i, layer in enumerate(pt_model.layers):
        h, _ = layer(h, freqs_cis, mask)
        norm_t5 = h[0, 5].norm().item()
        print(f"PyTorch Layer {i:2d} (last token pos 5) L2 Norm: {norm_t5:.4f}")
