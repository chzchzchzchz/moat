import json
import time
import os
from mlx_lm import load, generate

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

MODELS = [
    ("0.8B", os.path.join(MOAT_ROOT, "models/qwen3_5_0_8b_4bit")),
    ("2B", os.path.join(MOAT_ROOT, "models/qwen3_5_2b_4bit")),
    ("4B", os.path.join(MOAT_ROOT, "models/qwen3_5_4b_4bit")),
]

def main():
    questions = load_gsm8k(3)
    results = {}

    for name, path in MODELS:
        print(f"\n====================================")
        print(f"EVALUATING MODEL: {name}")
        print(f"====================================")
        
        if not os.path.exists(path):
            print(f"Path not found: {path}. Skipping.")
            continue
            
        try:
            model, tokenizer = load(path, model_config={"trust_remote_code": True})
            
            model_results = []
            for i, q in enumerate(questions):
                prompt = f"<|im_start|>system\nYou are a helpful math reasoning assistant. Provide the final answer at the end.<|im_end|>\n<|im_start|>user\n{q['question']}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"
                
                t0 = time.time()
                response = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
                t1 = time.time()
                
                model_results.append({
                    "question": i + 1,
                    "time": t1 - t0,
                    "response": response,
                    "ground_truth": q['answer'].split('####')[-1].strip()
                })
                print(f"Q{i+1} done in {t1-t0:.2f}s")
                
            results[name] = model_results
        except Exception as e:
            print(f"MLX Error for {name}:", e)
            
    with open(os.path.join(MOAT_ROOT, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print("Benchmark complete. Wrote benchmark_results.json")

if __name__ == '__main__':
    main()
