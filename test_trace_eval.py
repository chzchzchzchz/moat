import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "/Users/MohssineChazi2/moat/antigravity-engine/src")
from genprm_verifier import GenPRMVerifier

model_dir = "models/qwen"
prob = "Janet's ducks lay 20 eggs per day. She eats 3 for breakfast every morning and uses 4 for baking. She sells the remainder at the farmers' market for $3 per egg. How much money does she make in 5 days?"
gt = (20 - 3 - 4) * 3 * 5  # = 13 * 15 = 195

prompt = f"<|im_start|>system\nYou are a math reasoning assistant. Solve the problem step-by-step. To verify your work, you MUST write a Python program in a ```python ... ``` block that calculates the result and prints it. Finally, write the final answer as #### <number>.<|im_end|>\n<|im_start|>user\n{prob}<|im_end|>\n<|im_start|>assistant\n"

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16).to("mps")

inputs = tok(prompt, return_tensors="pt").to("mps")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95)

gen_seq = outputs[0][inputs.input_ids.shape[1]:]
trace_text = tok.decode(gen_seq, skip_special_tokens=False)

print("=== GENERATED TRACE ===")
print(trace_text)
print("=======================")

prm = GenPRMVerifier()
extracted_ans = prm.extract_stated_answer(trace_text)
rep = prm.verify_trace_with_code(trace_text)

print(f"Ground Truth Answer: {gt}")
print(f"Extracted Answer: {extracted_ans}")
print(f"GenPRM Report: {rep}")
