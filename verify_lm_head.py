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
with torch.no_grad():
    h = pt_model.embed_tokens(p_tensor)
    freqs_cis = pt_model.freqs_cis[:6].to(h.device)
    mask = torch.full((6, 6), float("-inf"), device=h.device)
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    
    for layer in pt_model.layers:
        h, _ = layer(h, freqs_cis, mask)
        
    final_h = pt_model.norm(h)
    last_h = final_h[0, -1, :] # last token position (is)
    
    print("PyTorch final_norm last token L2 norm:", last_h.norm().item())
    print("PyTorch final_norm last token (first 10 elements):", last_h[:10].numpy())
    
    lm_head_weight = pt_model.lm_head.weight # [32000, 2048]
    logits = torch.matmul(last_h, lm_head_weight.T)
    top_vals, top_idx = torch.topk(logits, 5)
    
    print("\nPyTorch Top 5 from last_h @ lm_head.T:")
    for val, idx in zip(top_vals, top_idx):
        print(f"  Token {idx.item():5d} ({tok.decode([idx.item()]):15s}): logit = {val.item():.4f}")
