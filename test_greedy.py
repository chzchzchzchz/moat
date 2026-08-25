import torch
from transformer import TinyLlamaModel
from tokenizer import LlamaTokenizer

model = TinyLlamaModel.from_safetensors("models/tinyllama/model.safetensors").to("mps")
tokenizer = LlamaTokenizer("models/tinyllama/tokenizer.json")
token_ids = tokenizer.encode("The capital of France is")

with torch.no_grad():
    for _ in range(10):
        inputs = torch.tensor([token_ids], dtype=torch.long, device="mps")
        logits, _ = model(inputs)
        next_token = logits[0, -1].argmax().item()
        token_ids.append(next_token)

print("PyTorch Output:", tokenizer.decode(token_ids))
