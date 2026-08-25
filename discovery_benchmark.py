import json
import re
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def load_gsm8k(limit=3):
    data = []
    with open('/Users/MohssineChazi2/moat/gsm8k_test_set.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= limit: break
            data.append(json.loads(line))
    return data

def extract_answer(text):
    # Try to find the last number in the text
    matches = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if matches:
        return matches[-1].replace(',', '')
    return None

def main():
    path = "/Users/MohssineChazi2/moat/models/qwen3_5_2b_4bit"
    print(f"Loading {path} for Test-Time Compute Discovery...")
    model, tokenizer = load(path, model_config={"trust_remote_code": True})
    
    questions = load_gsm8k(5)
    
    pass_1_correct = 0
    pass_8_correct = 0
    
    for i, q in enumerate(questions):
        ground_truth = q['answer'].split('####')[-1].strip()
        prompt = f"<|im_start|>system\nYou are a helpful math reasoning assistant. Provide the final answer at the end.<|im_end|>\n<|im_start|>user\n{q['question']}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        
        print(f"\n--- Q{i+1}: {q['question']} (GT: {ground_truth}) ---")
        
        # Greedy (Pass@1)
        resp_greedy = generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=make_sampler(temp=0.0), verbose=False)
        ans_greedy = extract_answer(resp_greedy)
        is_greedy_correct = (ans_greedy == ground_truth)
        if is_greedy_correct:
            pass_1_correct += 1
        print(f"Greedy Answer: {ans_greedy} -> {'CORRECT' if is_greedy_correct else 'WRONG'}")
        
        # Best-of-8 (Pass@8)
        any_correct = is_greedy_correct
        print("Sampling 8 paths for Test-Time Search...")
        for j in range(8):
            resp_sample = generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=make_sampler(temp=0.8), verbose=False)
            ans_sample = extract_answer(resp_sample)
            if ans_sample == ground_truth:
                any_correct = True
                print(f"  Path {j+1}: {ans_sample} -> CORRECT! (Found breakthrough)")
                break # We found a valid reasoning path!
            else:
                print(f"  Path {j+1}: {ans_sample} -> WRONG")
                
        if any_correct:
            pass_8_correct += 1
            
    print(f"\n=== DISCOVERY RESULTS ===")
    print(f"Pass@1 (Standard Edge Inference): {pass_1_correct}/{len(questions)}")
    print(f"Pass@8 (Antigravity Test-Time Compute): {pass_8_correct}/{len(questions)}")

if __name__ == '__main__':
    main()
