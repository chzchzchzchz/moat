import json
from mlx_lm import load, generate
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
    print("Loading MLX Model...")
    try:
        model, tokenizer = load(os.path.join(MOAT_ROOT, "models/qwen3_5_4b"))
        q = load_gsm8k()[0]
        prompt = f"<|im_start|>system\nYou are a helpful math reasoning assistant.<|im_end|>\n<|im_start|>user\n{q['question']}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"
        
        print("\nGenerating...")
        response = generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=True)
        print("\n--- RESULTS ---")
        print(response)
    except Exception as e:
        print("MLX Error:", e)

if __name__ == '__main__':
    main()
