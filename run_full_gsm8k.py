import sys
import os
import re
import json
import time
import gc
import resource

sys.path.insert(0, os.path.abspath('antigravity-engine/src'))

from tokenizer import LlamaTokenizer
from native_bridge import NativeMetalEngine
import matplotlib.pyplot as plt

def get_peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024 * 1024)

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

def plot_scaling_curve(results_summary, output_file='gsm8k_scaling_curve.png'):
    n_list = [1, 2, 4, 8]
    acc_list = [results_summary[n]['accuracy'] for n in n_list]

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(n_list, acc_list, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Antigravity 1.1B Engine (On-Device Metal GPU)')
    plt.axhline(y=78.5, color='#d62728', linestyle='--', linewidth=2, label='70B Cloud Teacher Baseline (78.5%)')

    plt.title('GSM8K Full Dataset Test-Time Scaling Curve (On-Device Metal GPU)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Test-Time Compute Budget N (Parallel GPU Channels)', fontsize=11, fontweight='bold')
    plt.ylabel('GSM8K Accuracy (%)', fontsize=11, fontweight='bold')
    plt.xticks(n_list, [f'N={n}' for n in n_list], fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(max(0, min(acc_list) - 5), 100)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='lower right')

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def main():
    print("=================================================================")
    print("      RUNNING FULL GSM8K STATISTICAL BENCHMARK (1,319 PROBLEMS)  ")
    print("=================================================================")

    dataset_path = 'gsm8k_test_set.jsonl'
    checkpoint_path = 'gsm8k_full_checkpoint.json'
    
    problems = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            gt_num = extract_answer_number(data['answer'])
            if gt_num is not None:
                problems.append({
                    'question': data['question'],
                    'answer_text': data['answer'],
                    'gt_number': gt_num
                })

    num_problems = len(problems)
    print(f"✅ Loaded {num_problems} valid GSM8K test problems.")

    # Checkpoint loading for resilience & resume capability
    completed_data = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                completed_data = json.load(f)
            print(f"🔄 Resuming from checkpoint: {len(completed_data)} problems already evaluated.")
        except Exception as e:
            print(f"⚠️ Checkpoint read error, starting fresh: {e}")
            completed_data = {}

    tok = LlamaTokenizer('models/tinyllama/tokenizer.json')
    engine = NativeMetalEngine(
        dylib_path='libantigravity_engine.dylib',
        model_path='models/tinyllama/model.safetensors',
        n_channels=8
    )

    budget_n_values = [1, 2, 4, 8]
    start_time = time.perf_counter()

    for i, prob in enumerate(problems):
        prob_id = str(i)
        if prob_id in completed_data:
            continue

        prompt = prob['question']
        gt_num = prob['gt_number']
        prompt_ids = tok.encode(prompt)

        # Run 8-channel parallel rollout
        t0 = time.perf_counter()
        tokens, logprobs, ttft, total = engine.generate(
            prompt_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        t1 = time.perf_counter()

        # Evaluate candidate rollouts for budget N = 1, 2, 4, 8
        prob_results = {}
        total_tokens_generated = sum(len(tokens[c]) for c in range(8))

        for N in budget_n_values:
            channel_scores = [(c, logprobs[c]) for c in range(N)]
            channel_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx = channel_scores[0][0]

            gen_text = tok.decode(tokens[best_idx])
            pred_num = extract_answer_number(gen_text)

            # Verification logic for search budget N
            is_correct = (pred_num == gt_num)
            prob_results[str(N)] = {
                'pred_num': pred_num,
                'is_correct': is_correct,
                'text_snippet': gen_text[:80]
            }

        completed_data[prob_id] = {
            'gt_num': gt_num,
            'elapsed_sec': t1 - t0,
            'tokens_count': total_tokens_generated,
            'n_results': prob_results
        }

        # Save checkpoint atomically
        with open(checkpoint_path, 'w') as f:
            json.dump(completed_data, f, indent=2)

        evaluated_count = len(completed_data)

        # Periodic logging and plotting
        if evaluated_count % 10 == 0 or evaluated_count == num_problems:
            acc_n = {}
            for N in budget_n_values:
                correct = sum(1 for p in completed_data.values() if p['n_results'][str(N)]['is_correct'])
                acc_n[N] = (correct / evaluated_count) * 100.0

            tot_time = time.perf_counter() - start_time
            tot_toks = sum(p['tokens_count'] for p in completed_data.values())
            tok_per_sec = tot_toks / tot_time if tot_time > 0 else 0
            rss_mb = get_peak_memory_mb()

            print(f"Progress: [{evaluated_count}/{num_problems}] ({evaluated_count/num_problems*100:.1f}%) | "
                  f"Acc N=1: {acc_n[1]:.1f}% | Acc N=8: {acc_n[8]:.1f}% | "
                  f"Speed: {tok_per_sec:.1f} tok/s | Memory: {rss_mb:.1f} MB")

            # Update plot periodically
            results_summary = {N: {'accuracy': acc_n[N]} for N in budget_n_values}
            plot_scaling_curve(results_summary)

    del engine
    gc.collect()

    # Final summary calculations
    evaluated_count = len(completed_data)
    final_summary = {}
    print("\n=================================================================")
    print("                FULL GSM8K EVALUATION COMPLETE                  ")
    print("=================================================================")
    for N in budget_n_values:
        correct = sum(1 for p in completed_data.values() if p['n_results'][str(N)]['is_correct'])
        acc = (correct / evaluated_count) * 100.0 if evaluated_count > 0 else 0
        final_summary[N] = {'accuracy': acc, 'correct': correct, 'total': evaluated_count}
        print(f"  Search Budget N={N:d} | Accuracy: {acc:5.1f}% ({correct}/{evaluated_count})")

    plot_scaling_curve(final_summary)
    print("=================================================================")
    print("✅ Final scaling curve saved to gsm8k_scaling_curve.png")

if __name__ == '__main__':
    main()
