import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "models/qwen"
prompt = "<|im_start|>system\nYou are a math reasoning assistant. Solve the problem step-by-step and write the final answer as #### <number>.<|im_end|>\n<|im_start|>user\nJanet has 16 eggs. She eats 3 for breakfast and bakes 4 into muffins. How many eggs does she have left?<|im_end|>\n<|im_start|>assistant\n"

print("--- Testing PyTorch MPS / HuggingFace AutoModelForCausalLM ---")
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16).to("mps")

inputs = tok(prompt, return_tensors="pt").to("mps")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.95)

generated_text = tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
print("HuggingFace Output:")
print(generated_text)
