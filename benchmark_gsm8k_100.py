import sys
import os
import re
import time
import math
import json
import gc
import resource
import numpy as np

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(MOAT_ROOT, "antigravity-engine/src"))
sys.path.insert(0, os.path.join(MOAT_ROOT, "antigravity-engine"))

from genprm_verifier import GenPRMVerifier
from dora_clustering import DORAClusterer


def get_peak_rss_mb() -> float:
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return float(rusage.ru_maxrss / (1024.0 * 1024.0))

def extract_answer_number(text: str):
    if not text:
        return None
    match = re.search(r'####\s*(-?\d[\d,]*)', text)
    if match:
        return int(match.group(1).replace(',', ''))
    match = re.search(r'\\boxed\{(-?\d[\d,]*)\}', text)
    if match:
        return int(match.group(1).replace(',', ''))
    match = re.search(r'(?:the answer is|equals|=)\s*(-?\d[\d,]*)', text, re.IGNORECASE)
    if match:
        return int(match.group(1).replace(',', ''))
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return int(numbers[-1])
    return None

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

def main():
    print("==========================================================================", flush=True)
    print("  PROJECT ANTIGRAVITY — 100-QUESTION GSM8K STATISTICAL ACCURACY BENCHMARK", flush=True)
    print("==========================================================================", flush=True)

    dataset_path = 'gsm8k_test_set.jsonl'
    checkpoint_path = 'gsm8k_100_checkpoint.json'
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found!", flush=True)
        return

    problems = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(problems) >= 100:
                break
            data = json.loads(line.strip())
            gt_num = extract_answer_number(data['answer'])
            if gt_num is not None:
                problems.append({
                    'id': len(problems) + 1,
                    'question': data['question'],
                    'answer_text': data['answer'],
                    'gt_number': gt_num
                })

    print(f"✅ Loaded {len(problems)} GSM8K test set problems.", flush=True)

    model_dir = "models/qwen"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Qwen2.5-Math-1.5B model on {device}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16).to(device)

    prm = GenPRMVerifier()
    clusterer = DORAClusterer(vector_dim=128)

    channel_configs = [1, 2, 4, 8, 16]

    # Resume checkpoint if exists
    checkpoint = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            print(f"🔄 Resumed from checkpoint: {len(checkpoint)} configurations/problems recorded.", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to read checkpoint: {e}", flush=True)
            checkpoint = {}

    summary_results = []

    for N in channel_configs:
        n_str = str(N)
        if n_str not in checkpoint:
            checkpoint[n_str] = {}

        print(f"\n-----------------------------------------------------------------", flush=True)
        print(f"  EVALUATING SEARCH BUDGET N = {N} ({'Greedy Baseline' if N==1 else f'Parallel {N}-Path Batched Rollouts'})", flush=True)
        print(f"-----------------------------------------------------------------", flush=True)

        t0 = time.perf_counter()
        b_correct = 0
        c_correct = 0
        total_tokens = 0
        entropies = []

        for p_idx, prob in enumerate(problems):
            p_key = str(prob['id'])
            if p_key in checkpoint[n_str]:
                res = checkpoint[n_str][p_key]
                b_correct += res['b_correct']
                c_correct += res['c_correct']
                total_tokens += res['tokens']
                entropies.append(res['entropy'])
                continue

            question = prob['question']
            gt = prob['gt_number']

            prompt = (
                f"<|im_start|>system\nYou are a math reasoning assistant. Solve the problem step-by-step. "
                f"To verify your work, write a Python program in a ```python ... ``` block that calculates the result and prints it. "
                f"Finally, write the final answer as #### <number>.<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]

            candidate_traces = []
            extracted_answers = []
            best_idx = 0
            best_score = -999.0
            
            # Generate rollouts one at a time
            for rollout_i in range(N):
                gen_kwargs = {
                    "max_new_tokens": 512,
                    "num_return_sequences": 1,
                    "return_dict_in_generate": True
                }
                if N > 1:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = 0.7
                    gen_kwargs["top_p"] = 0.95

                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)

                seq = outputs.sequences[0]
                gen_tokens = seq[prompt_len:]
                total_tokens += len(gen_tokens)
                trace_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                # Verify channel immediately
                prm_result = prm.verify_trace_with_code(trace_text)
                
                # Self-Heal Sequential Revision
                if prm_result['has_code'] and not prm_result['code_success']:
                    feedback = prm_result.get('feedback_prompt', '')
                    if feedback:
                        rev_prompt = prompt + trace_text + "\n" + feedback + "\n"
                        rev_inputs = tokenizer(rev_prompt, return_tensors="pt").to(device)
                        rev_prompt_len = rev_inputs.input_ids.shape[1]
                        
                        with torch.no_grad():
                            rev_outputs = model.generate(**rev_inputs, **gen_kwargs)
                            
                        rev_seq = rev_outputs.sequences[0]
                        rev_gen_tokens = rev_seq[rev_prompt_len:]
                        total_tokens += len(rev_gen_tokens)
                        rev_trace_text = tokenizer.decode(rev_gen_tokens, skip_special_tokens=True)
                        trace_text = trace_text + "\n" + feedback + "\n" + rev_trace_text
                        prm_result = prm.verify_trace_with_code(trace_text)
                
                candidate_traces.append(trace_text)
                extracted_answers.append(extract_answer_number(trace_text))
                
                # TOPS First-Finish Search Early Exit
                if prm_result['code_success'] and prm_result['is_consistent']:
                    # Early exit triggered!
                    # Fill the rest of the candidates with this successful trace so MajVote isn't broken
                    while len(candidate_traces) < N:
                        candidate_traces.append(trace_text)
                        extracted_answers.append(extracted_answers[-1])
                    best_idx = len(candidate_traces) - 1
                    break

            # Calculate DORA entropy
            if N > 1:
                # Use tokenizer to get token ids for DORA embedding
                tokenized_traces = [tokenizer.encode(t, add_special_tokens=False) for t in candidate_traces]
                dora_res = clusterer.cluster_candidates(tokenized_traces)
                sim_matrix = dora_res['similarity_matrix']
                entropy = calculate_dora_entropy(sim_matrix)
            else:
                dora_res = None
                entropy = 0.0
            entropies.append(entropy)

            # Baseline B: Standard Majority Vote
            valid_ans = [a for a in extracted_answers if a is not None]
            if valid_ans:
                counts = Counter(valid_ans)
                b_ans = counts.most_common(1)[0][0]
            else:
                b_ans = None
            b_acc = 1 if b_ans == gt else 0

            # Baseline C: Antigravity GenPRM + DORA Verified Selection
            # If early exit was triggered, best_idx is already the successful rollout
            # Otherwise we score them list-wise
            best_idx = 0
            best_score = -999.0
            for c_i, trace in enumerate(candidate_traces):
                prm_result = prm.verify_trace_with_code(trace)

                # Score purely from verifier signals — NO ground truth access
                base_score = float(prm_result['reward_modifier'])

                # DORA uniqueness weighting (prefer diverse, non-redundant traces)
                if N > 1 and dora_res is not None:
                    uniqueness = dora_res['uniqueness_weights'][c_i]
                    base_score += uniqueness * 0.3
                    
                # Massive bonus if consistent
                if prm_result['code_success'] and prm_result['is_consistent']:
                    base_score += 100.0

                if base_score > best_score:
                    best_score = base_score
                    best_idx = c_i

            c_ans = extracted_answers[best_idx]
            c_acc = 1 if c_ans == gt else 0

            b_correct += b_acc
            c_correct += c_acc

            checkpoint[n_str][p_key] = {
                'b_correct': b_acc,
                'c_correct': c_acc,
                'tokens': total_tokens,
                'entropy': entropy,
                'b_ans': b_ans,
                'c_ans': c_ans,
                'gt': gt
            }

            # Persist checkpoint on every single question so no progress is ever lost
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            b_mark = "✓" if b_acc else "✗"
            c_mark = "✓" if c_acc else "✗"
            print(f"  • [Q {p_idx+1:>2}/100] b_ans={str(b_ans):<6} [{b_mark}] | c_ans={str(c_ans):<6} [{c_mark}] | gt={str(gt):<6} | Running: MajVote={(b_correct/(p_idx+1))*100:.1f}%, Antigravity={(c_correct/(p_idx+1))*100:.1f}%", flush=True)

        elapsed = time.perf_counter() - t0
        peak_rss = get_peak_rss_mb()
        avg_entropy = float(np.mean(entropies))
        b_acc_pct = (b_correct / len(problems)) * 100.0
        c_acc_pct = (c_correct / len(problems)) * 100.0

        print(f"\n  [N={N} COMPLETE SUMMARY]", flush=True)
        print(f"  Majority Vote Acc (Baseline B) : {b_acc_pct:.2f}%", flush=True)
        print(f"  Antigravity Engine Acc (Baseline C): {c_acc_pct:.2f}%", flush=True)
        print(f"  Avg DORA Shannon Entropy       : {avg_entropy:.4f} bits", flush=True)
        print(f"  Peak RSS Memory Footprint     : {peak_rss:.2f} MB", flush=True)
        print(f"  Elapsed Time                   : {elapsed:.2f} seconds", flush=True)

        summary_results.append({
            "N": N,
            "majority_vote_acc": b_acc_pct,
            "antigravity_acc": c_acc_pct,
            "avg_dora_entropy_bits": avg_entropy,
            "peak_rss_mb": peak_rss,
            "elapsed_sec": elapsed
        })

    print("\n==========================================================================", flush=True)
    print("  FINAL 100-QUESTION GSM8K EMPIRICAL BENCHMARK SUMMARY & HARDWARE PROOF", flush=True)
    print("==========================================================================", flush=True)
    print("| N Channels | Majority Vote Acc (%) | Antigravity Engine Acc (%) | DORA Entropy (bits) | Peak RSS (MB) |", flush=True)
    print("|------------|-----------------------|----------------------------|---------------------|---------------|", flush=True)
    for res in summary_results:
        print(f"| {res['N']:<10} | {res['majority_vote_acc']:<21.2f} | {res['antigravity_acc']:<26.2f} | {res['avg_dora_entropy_bits']:<19.4f} | {res['peak_rss_mb']:<13.2f} |", flush=True)

    with open("antigravity_gsm8k_100_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    with open("antigravity_gsm8k_100_results.csv", "w") as f:
        f.write("N,majority_vote_acc,antigravity_acc,avg_dora_entropy_bits,peak_rss_mb,elapsed_sec\n")
        for r in summary_results:
            f.write(f"{r['N']},{r['majority_vote_acc']},{r['antigravity_acc']},{r['avg_dora_entropy_bits']},{r['peak_rss_mb']},{r['elapsed_sec']}\n")

    print("\n✅ Evaluation completed successfully! Results written to CSV & JSON artifacts.", flush=True)

if __name__ == "__main__":
    main()
