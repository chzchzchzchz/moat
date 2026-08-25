import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_gsm8k():
    data = []
    with open('/Users/MohssineChazi2/moat/gsm8k_test_set.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 1: break
            data.append(json.loads(line))
    return data

def main():
    print("Loading Qwen 3.5 4B Model (via HF)...")
    model_id = "/Users/MohssineChazi2/moat/models/qwen3_5_4b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Load in FP16 to fit in memory
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="mps", torch_dtype=torch.float16, trust_remote_code=True)
    
    questions = load_gsm8k()
    q = questions[0]
    
    prompt = f"<|im_start|>system\nYou are a helpful math reasoning assistant.<|im_end|>\n<|im_start|>user\n{q['question']}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n--- PROMPT ---\n{prompt}\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    print("Generating Answer...")
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- RESULTS ---")
    print(result)

if __name__ == '__main__':
    main()
