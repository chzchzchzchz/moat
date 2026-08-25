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
    freqs_cis = pt_model.freqs_cis[:6].to(h.device)
    mask = torch.full((6, 6), float("-inf"), device=h.device)
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    
    print("PyTorch Layer 0 Token 0 Embedding L2 Norm:", h[0, 0].norm().item())
    
    h_layer0, _ = pt_model.layers[0](h, freqs_cis, mask)
    
    h_norm = pt_model.layers[0].input_layernorm(h)
    print("PyTorch Layer 0 input_layernorm Token 0 L2 Norm:", h_norm[0, 0].norm().item())
    
    # Manually compute Attention for Token 0
    attn = pt_model.layers[0].self_attn
    q = attn.q_proj(h_norm[:, 0:1])
    k = attn.k_proj(h_norm[:, 0:1])
    v = attn.v_proj(h_norm[:, 0:1])
    
    print("PyTorch Layer 0 Q before RoPE L2 Norm:", q[0, 0].norm().item())
    print("PyTorch Layer 0 K before RoPE L2 Norm:", k[0, 0].norm().item())
    print("PyTorch Layer 0 V before cache L2 Norm:", v[0, 0].norm().item())
    
    # Q and K after RoPE for seq_pos = 0
    from transformer import apply_rotary_emb
    q_out, k_out = apply_rotary_emb(
        q.view(1, 1, 32, 64).transpose(1, 2),
        k.view(1, 1, 4, 64).transpose(1, 2),
        freqs_cis[0:1]
    )
    print("PyTorch Layer 0 Q after RoPE L2 Norm:", q_out.norm().item())
    print("PyTorch Layer 0 K after RoPE L2 Norm:", k_out.norm().item())
    
    # Attn scores for seq_pos = 0 (only 1 element)
    import math
    scores = torch.matmul(q_out, k_out.repeat_interleave(8, dim=1).transpose(2, 3)) / math.sqrt(64)
    print("PyTorch Layer 0 Attn Scores L2 Norm:", scores.norm().item())
    
    probs = torch.nn.functional.softmax(scores, dim=-1)
    print("PyTorch Layer 0 Attn Softmax L2 Norm:", probs.norm().item())
    
    v_out = v.view(1, 1, 4, 64).transpose(1, 2)  # [1, 4, 1, 64]
    v_out = v_out.repeat_interleave(8, dim=1)    # [1, 32, 1, 64]
    
    attn_val = torch.matmul(probs, v_out)        # [1, 32, 1, 64]
    print("PyTorch Layer 0 Attn Value Output L2 Norm:", attn_val.norm().item())
    
    attn_out = attn.o_proj(attn_val.transpose(1, 2).reshape(1, 1, 2048))
    print("PyTorch Layer 0 O_proj Output L2 Norm:", attn_out.norm().item())
    
    residual = h[:, 0:1] + attn_out
    print("PyTorch Layer 0 Pre-MLP Residual L2 Norm:", residual.norm().item())
    
    mlp = pt_model.layers[0].mlp
    mlp_norm = pt_model.layers[0].post_attention_layernorm(residual)
    gate = mlp.gate_proj(mlp_norm)
    up = mlp.up_proj(mlp_norm)
    print("PyTorch Layer 0 Gate Proj L2 Norm:", gate.norm().item())
    print("PyTorch Layer 0 Up Proj L2 Norm:", up.norm().item())
    
    swiglu = torch.nn.functional.silu(gate) * up
    print("PyTorch Layer 0 SwiGLU Output L2 Norm:", swiglu.norm().item())
    
    mlp_out = mlp.down_proj(swiglu)
    print("PyTorch Layer 0 Down Proj Output L2 Norm:", mlp_out.norm().item())
    final_out = residual + mlp_out
    print("PyTorch Layer 0 Final Output L2 Norm:", final_out.norm().item())
    
    # Forward all layers
    # Forward all layers
    h = pt_model.embed_tokens(p_tensor)
    position_ids = torch.arange(0, h.size(1), dtype=torch.long, device=h.device).unsqueeze(0)
    for layer in pt_model.layers:
        h = layer(h, freqs_cis, mask)[0]
    
    print(f"PyTorch Final Hidden State L2 Norm: {h[0, -1].norm().item()}")
    final_norm = pt_model.norm(h)
    print(f"PyTorch Final RMSNorm Output L2 Norm: {final_norm[0, -1].norm().item()}")
    
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained('models/tinyllama', local_files_only=True)
    logits = hf_model.lm_head(final_norm)
    print(f"PyTorch Final Logits L2 Norm: {logits[0, -1].norm().item()}")


