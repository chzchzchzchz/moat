import json
import subprocess
import sys
import re

sys.path.insert(0, '/Users/MohssineChazi2/moat/antigravity-engine/src')
from tokenizer import LlamaTokenizer

def load_gsm8k():
    data = []
    with open('/Users/MohssineChazi2/moat/data/gsm8k/test.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 1: break # Just run 1 question for the demo
            data.append(json.loads(line))
    return data

def run_c_engine(prompt_ids):
    prompt_str = ",".join(map(str, prompt_ids))
    cmd = ["/Users/MohssineChazi2/moat/test_qwen_runner", prompt_str]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def main():
    print("Loading Tokenizer...")
    tok = LlamaTokenizer('/Users/MohssineChazi2/moat/models/tinyllama/tokenizer.json')
    
    questions = load_gsm8k()
    q = questions[0]
    
    prompt = f"Question: {q['question']}\nLet's think step by step.\nAnswer:"
    print(f"\n--- PROMPT ---\n{prompt}\n")
    
    prompt_ids = tok.encode(prompt)
    
    print("Running Native Hybrid Engine...")
    output = run_c_engine(prompt_ids)
    
    # Parse generated tokens
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
