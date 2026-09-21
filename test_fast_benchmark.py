import sys
import os
import time
import math
import json
import numpy as np

import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(MOAT_ROOT, "antigravity-engine/src"))
sys.path.insert(0, os.path.join(MOAT_ROOT, "antigravity-engine"))

from genprm_verifier import GenPRMVerifier
from dora_clustering import DORAClusterer
import resource

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def get_peak_rss_mb() -> float:
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return float(rusage.ru_maxrss / (1024.0 * 1024.0))

def calculate_dora_entropy(similarity_matrix: np.ndarray) -> float:
    N = similarity_matrix.shape[0]
    if N <= 1:
        return 0.0
    row_sums = np.sum(similarity_matrix, axis=1)
    total_sum = np.sum(row_sums)
    if total_sum <= 1e-8:
        return 0.0
    p = row_sums / total_sum
    p = p[p > 1e-8]
    return float(-np.sum(p * np.log2(p)))

model_dir = "models/qwen"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading Qwen model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16).to(device)

prm = GenPRMVerifier()
clusterer = DORAClusterer(vector_dim=128)

# 5 out-of-distribution math reasoning problems
dataset = [
    {
        "problem": "Janet's ducks lay 20 eggs per day. She eats 3 for breakfast every morning and uses 4 for baking. She sells the remainder at the farmers' market for $3 per egg. How much money does she make in 5 days?",
        "ground_truth": (20 - 3 - 4) * 3 * 5  # 195
    },
    {
        "problem": "A train travels at 60 mph for 3 hours, then speeds up to 80 mph for another 2 hours. How many total miles did it travel?",
        "ground_truth": (60 * 3) + (80 * 2)  # 340
    },
    {
        "problem": "Marcus buys 4 shirts at $20 each and 3 pairs of pants at $40 each. He gets a total discount of $15. What is the final total cost in dollars?",
        "ground_truth": (4 * 20 + 3 * 40) - 15  # 185
    },
    {
        "problem": "A painter paints 3 walls on Monday. Each day after that, he paints 2 more walls than the previous day. How many total walls does he paint from Monday through Friday (5 days)?",
        "ground_truth": 3 + 5 + 7 + 9 + 11  # 35
    },
    {
        "problem": "Alice, Bob, and Charlie split a reward of $300. Alice gets 2 times Bob's share, and Charlie gets $20 more than Bob. How many dollars does Bob receive?",
        "ground_truth": (300 - 20) / 4  # 70
    }
]

channel_configs = [1, 2, 4, 8, 16]
results_summary = []

print("\n==========================================================================")
print("  PROJECT ANTIGRAVITY — THREE BATTLEGROUNDS EMPIRICAL BENCHMARK SUITE")
print("==========================================================================")

for N in channel_configs:
    print(f"\n--- RUNNING CONFIGURATION: N = {N} Candidate Rollout Channels ---")
    t0 = time.perf_counter()
    b_correct = 0
    c_correct = 0
    total_tokens = 0
    entropies = []

    for p_idx, item in enumerate(dataset):
        prob = item["problem"]
        gt = item["ground_truth"]

        prompt = f"<|im_start|>system\nYou are a math reasoning assistant. Solve the problem step-by-step. To verify your work, write a Python program in a ```python ... ``` block that calculates the result and prints it. Finally, write the final answer as #### <number>.<|im_end|>\n<|im_start|>user\n{prob}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        candidate_traces = []
        base_scores = []

        gen_kwargs = {
            "max_new_tokens": 250,
            "num_return_sequences": N,
            "return_dict_in_generate": True
        }
        if N > 1:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = 0.7
            gen_kwargs["top_p"] = 0.95
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        for c in range(N):
            seq = outputs.sequences[c][prompt_len:]
            text = tokenizer.decode(seq, skip_special_tokens=False)
            candidate_traces.append(text)
            total_tokens += len(seq)
            base_scores.append(0.0)

        # DORA Shannon Entropy calculation across candidate traces
        if N > 1:
            emb_res = clusterer.cluster_candidates(candidate_traces)
            sim_mat = emb_res["similarity_matrix"]
            entropy = calculate_dora_entropy(sim_mat)
        else:
            entropy = 0.0
        entropies.append(entropy)

        # Baseline B: Majority Voting across candidates
        extracted_answers = []
        for trace in candidate_traces:
            ans_str = prm.extract_stated_answer(trace)
            if ans_str is not None:
                try:
                    extracted_answers.append(float(ans_str))
                except ValueError:
                    pass

        if extracted_answers:
            most_common = Counter(extracted_answers).most_common(1)[0][0]
            b_acc = 1 if math.isclose(most_common, gt, abs_tol=1e-2) else 0
        else:
            b_acc = 0
        b_correct += b_acc

        # Baseline C: Antigravity Selection (GenPRM code execution + DORA)
        adjusted_scores, reports = prm.score_candidates_genprm(candidate_traces, np.array(base_scores))
        best_idx = int(np.argmax(adjusted_scores))
        best_trace = candidate_traces[best_idx]
        c_ans_str = prm.extract_stated_answer(best_trace)

        if c_ans_str is not None:
            try:
                c_val = float(c_ans_str)
                c_acc = 1 if math.isclose(c_val, gt, abs_tol=1e-2) else 0
            except ValueError:
                c_acc = 0
        else:
            c_acc = 0
        c_correct += c_acc

        print(f"  • Question {p_idx+1}/{len(dataset)}: Baseline B={b_acc}, Baseline C={c_acc} (Best trace answer: {c_ans_str}, GT: {gt})", flush=True)

    elapsed_sec = time.perf_counter() - t0
    peak_rss = get_peak_rss_mb()
    avg_entropy = float(np.mean(entropies))
    b_acc_pct = (b_correct / len(dataset)) * 100.0
    c_acc_pct = (c_correct / len(dataset)) * 100.0

    print(f"  [N={N} SUMMARY] Majority Vote Acc: {b_acc_pct:.1f}%, Antigravity Acc: {c_acc_pct:.1f}%, DORA Entropy: {avg_entropy:.4f} bits, Peak RSS: {peak_rss:.2f} MB", flush=True)

    results_summary.append({
        "N_channels": N,
        "majority_vote_acc": b_acc_pct,
        "antigravity_acc": c_acc_pct,
        "accuracy_lift_pct": c_acc_pct - b_acc_pct,
        "total_tokens": total_tokens,
        "dora_entropy_bits": avg_entropy,
        "peak_rss_mb": peak_rss,
        "elapsed_sec": elapsed_sec
    })

# Output Summary Table
print("\n==========================================================================")
print("  FINAL EMPIRICAL BENCHMARK SUMMARY & PROOF OF TEST-TIME SCALING")
print("==========================================================================")
print("| N Channels | Majority Vote Acc (%) | Antigravity Engine Acc (%) | Total Tokens | DORA Entropy (bits) | Peak RSS (MB) |")
print("|------------|-----------------------|----------------------------|--------------|---------------------|---------------|")
for r in results_summary:
    print(f"| {r['N_channels']:<10} | {r['majority_vote_acc']:<21.2f} | {r['antigravity_acc']:<26.2f} | {r['total_tokens']:<12} | {r['dora_entropy_bits']:<19.4f} | {r['peak_rss_mb']:<13.2f} |")

csv_path = "antigravity-engine/antigravity_benchmark_results.csv"
json_path = "antigravity-engine/antigravity_benchmark_results.json"

with open(csv_path, "w") as f:
    f.write("N_channels,majority_vote_acc,antigravity_acc,accuracy_lift_pct,total_tokens,dora_entropy_bits,peak_rss_mb,elapsed_sec\n")
    for r in results_summary:
        f.write(f"{r['N_channels']},{r['majority_vote_acc']},{r['antigravity_acc']},{r['accuracy_lift_pct']},{r['total_tokens']},{r['dora_entropy_bits']},{r['peak_rss_mb']},{r['elapsed_sec']}\n")

with open(json_path, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✅ Saved CSV artifact to:  {csv_path}")
print(f"✅ Saved JSON artifact to: {json_path}")
