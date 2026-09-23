import json
import re
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import os

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def load_gsm8k(limit=30):
    data = []
    with open(os.path.join(MOAT_ROOT, 'gsm8k_test_set.jsonl'), 'r') as f:
        for i, line in enumerate(f):
            if i >= limit: break
            data.append(json.loads(line))
    return data

def extract_answer(text):
    matches = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if matches:
        return matches[-1].replace(',', '')
    return None

def main():
    path = os.path.join(MOAT_ROOT, "models/qwen3_5_2b_4bit")
    print(f"Loading {path} for FULL Benchmark...")
    model, tokenizer = load(path, model_config={"trust_remote_code": True})
    
    questions = load_gsm8k(30)
    
    pass_1_correct = 0
    pass_8_correct = 0
    results = []
    
    for i, q in enumerate(questions):
        ground_truth = q['answer'].split('####')[-1].strip()
        prompt = f"<|im_start|>system\nYou are a helpful math reasoning assistant. Provide the final answer at the end.<|im_end|>\n<|im_start|>user\n{q['question']}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        
        print(f"\n--- Q{i+1}/{len(questions)} ---")
        
        resp_greedy = generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=make_sampler(temp=0.0), verbose=False)
        ans_greedy = extract_answer(resp_greedy)
        is_greedy_correct = (ans_greedy == ground_truth)
        if is_greedy_correct:
            pass_1_correct += 1
            
        any_correct = is_greedy_correct
        for j in range(8):
            resp_sample = generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=make_sampler(temp=0.8), verbose=False)
            ans_sample = extract_answer(resp_sample)
            if ans_sample == ground_truth:
                any_correct = True
                break
                
        if any_correct:
            pass_8_correct += 1
            
        print(f"Progress: Pass@1 = {pass_1_correct}/{i+1}, Pass@8 = {pass_8_correct}/{i+1}")
        results.append({
            "question": q['question'],
            "greedy_correct": is_greedy_correct,
            "pass_8_correct": any_correct
        })
            
    with open(os.path.join(MOAT_ROOT, 'full_benchmark_results.json'), 'w') as f:
        json.dump({
            "pass_1": pass_1_correct,
            "pass_8": pass_8_correct,
            "total": len(questions),
            "details": results
        }, f, indent=2)

if __name__ == '__main__':
    main()
