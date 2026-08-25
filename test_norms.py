import sys
sys.path.insert(0, 'antigravity-engine/src')
import torch
from transformer import TinyLlamaModel
from tokenizer import LlamaTokenizer

tok = LlamaTokenizer('models/tinyllama/tokenizer.json')
prompt = "The capital of France is"
prompt_ids = tok.encode(prompt)
print(f"Prompt token IDs: {prompt_ids}")

pt_model = TinyLlamaModel.from_safetensors('models/tinyllama/model.safetensors', device='cpu')
pt_model.eval()

p_tensor = torch.tensor([[prompt_ids[0]]], dtype=torch.long)
with torch.no_grad():
    h = pt_model.embed_tokens(p_tensor)
    print(f"Layer 0 Input L2 Norm: {h.norm().item()}")
    
    layer0 = pt_model.layers[0]
    h_norm = layer0.input_layernorm(h)
    print(f"Layer 0 input_layernorm L2 Norm: {h_norm.norm().item()}")
    
    q = layer0.self_attn.q_proj(h_norm)
    k = layer0.self_attn.k_proj(h_norm)
    v = layer0.self_attn.v_proj(h_norm)
    print(f"Layer 0 Q before RoPE L2 Norm: {q.norm().item()}")
    print(f"Layer 0 Q before RoPE first 5 elements: {q[0, 0, :5].tolist()}")
    
    mlp = layer0.mlp
    freqs_cis = pt_model.freqs_cis[:1]
    attn_out = layer0.self_attn(h_norm, freqs_cis=freqs_cis)[0]
    h = h + attn_out
    print(f"Layer 0 Pre-MLP Residual first 5 elements: {h[0, 0, :5].tolist()}")
    
    gate = mlp.gate_proj(layer0.post_attention_layernorm(h))
    up = mlp.up_proj(layer0.post_attention_layernorm(h))
    print(f"Layer 0 Gate Proj L2 Norm: {gate.norm().item()}")
    print(f"Layer 0 Up Proj L2 Norm: {up.norm().item()}")
    
    swiglu = torch.nn.functional.silu(gate) * up
    print(f"Layer 0 SwiGLU Output L2 Norm: {swiglu.norm().item()}")
    print(f"Layer 0 SwiGLU first 5 elements: {swiglu[0, 0, :5].tolist()}")
    
    down = mlp.down_proj(swiglu)
    print(f"Layer 0 Down Proj Output L2 Norm: {down.norm().item()}")
    print(f"Layer 0 Down Proj first 5 elements: {down[0, 0, :5].tolist()}")
    
    h = h + down
    print(f"Layer 0 Final Output L2 Norm: {h.norm().item()}")
    print(f"Layer 0 Final Output first 5 elements: {h[0, 0, :5].tolist()}")
