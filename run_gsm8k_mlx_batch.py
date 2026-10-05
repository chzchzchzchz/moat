import json
import time
from mlx_lm import load, generate
import os

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def load_gsm8k(limit=3):
    data = []
    with open(os.path.join(MOAT_ROOT, 'gsm8k_test_set.jsonl'), 'r') as f:
        for i, line in enumerate(f):
            if i >= limit: break
            data.append(json.loads(line))
    return data

def main():
    print("Loading MLX Model (Qwen3.5-4B)...")
    try:
        model, tokenizer = load(os.path.join(MOAT_ROOT, "models/qwen3_5_4b_4bit"), model_config={"trust_remote_code": True})
        questions = load_gsm8k(3)
        
        for i, q in enumerate(questions):
            prompt = f"<|im_start|>system\nYou are a helpful math reasoning assistant. Provide the final answer at the end.<|im_end|>\n<|im_start|>user\n{q['question']}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"
            
            print(f"\n====================================")
            print(f"QUESTION {i+1}: {q['question']}")
            print(f"====================================")
            
            t0 = time.time()
            response = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
            t1 = time.time()
            
            print(response)
            print(f"\n[Time: {t1-t0:.2f}s | Ground Truth: {q['answer'].split('####')[-1].strip()}]")
            
    except Exception as e:
        print("MLX Error:", e)

if __name__ == '__main__':
    main()
