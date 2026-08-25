import json
import subprocess
from tokenizers import Tokenizer

def load_gsm8k():
    data = []
    with open('/Users/MohssineChazi2/moat/gsm8k_test_set.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 1: break
            data.append(json.loads(line))
    return data

def run_c_engine(prompt_ids, model_path):
    prompt_str = ",".join(map(str, prompt_ids))
    cmd = ["/Users/MohssineChazi2/moat/test_qwen_runner", prompt_str, model_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def main():
    print("Loading Qwen Tokenizer...")
    tok = Tokenizer.from_file('/Users/MohssineChazi2/moat/models/qwen3_5_4b/tokenizer.json')
    
    questions = load_gsm8k()
    q = questions[0]
    
    prompt = f"<|im_start|>system\nYou are a helpful math reasoning assistant.<|im_end|>\n<|im_start|>user\n{q['question']}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n--- PROMPT ---\n{prompt}\n")
    
    prompt_ids = tok.encode(prompt).ids
    
    print("Running Native Hybrid Engine with Real Qwen 3.5 4B...")
    output = run_c_engine(prompt_ids, '/Users/MohssineChazi2/moat/models/qwen3_5_4b/model.safetensors')
    
    paths = []
    lines = output.split('\n')
    for line in lines:
        if line.startswith("PATH "):
            toks_str = line.split(": ")[1].strip().split(',')
            toks = [int(t) for t in toks_str if t]
            paths.append(toks)
        else:
            if "✅" in line or "Throughput:" in line or "Peak Memory RSS:" in line:
                print(line)
            
    print("\n--- RESULTS (N=8) ---")
    for i, path in enumerate(paths):
        decoded = tok.decode(path)
        print(f"\n[Path {i}] {decoded.strip()}")

if __name__ == '__main__':
    main()
