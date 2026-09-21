import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def load_gsm8k():
    data = []
    with open(os.path.join(MOAT_ROOT, 'gsm8k_test_set.jsonl'), 'r') as f:
        for i, line in enumerate(f):
            if i >= 1: break
            data.append(json.loads(line))
    return data

def main():
    print("Loading Qwen 3.5 4B Tokenizer...")
    model_id = os.path.join(MOAT_ROOT, "models/qwen3_5_4b")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    print("Loading Qwen 3.5 4B Model (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True)
    
    questions = load_gsm8k()
    q = questions[0]
    
    prompt = f"<|im_start|>system\nYou are a helpful math reasoning assistant.<|im_end|>\n<|im_start|>user\n{q['question']}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n--- PROMPT ---\n{prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    print("Generating Answer (Test for real)...")
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.01)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- EXACT RAW MODEL OUTPUT ---")
    print(result)

if __name__ == '__main__':
    main()
