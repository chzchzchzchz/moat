import sys
import os

sys.path.insert(0, os.path.abspath("antigravity-engine/src"))
from orchestrator import AntigravityEngine
from genprm_verifier import GenPRMVerifier

engine = AntigravityEngine(n_channels=4)
prm = GenPRMVerifier()

prob = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and uses 4 for baking. She sells the remainder at the farmers' market for $2 per egg. How much money does she make in 7 days?"

prompt = f"<|im_start|>system\nYou are a math reasoning assistant. Solve the problem step-by-step. To verify your work, you MUST write a Python program in a ```python ... ``` block that calculates the result and prints it. Finally, write the final answer as #### <number>.<|im_end|>\n<|im_start|>user\n{prob}<|im_end|>\n<|im_start|>assistant\n"

res = engine.run_best_of_n_query(prompt, max_tokens=250, temperature=0.7, top_p=0.95)

print("\n--- Best Trace ---")
print(res['best_trace'])

for i, rep in enumerate(res['genprm_reports']):
    print(f"\nTrace {i} GenPRM Report:")
    print(f"Has code: {rep['has_code']}")
    if rep['has_code']:
        print(f"Consistent: {rep['is_consistent']}")
        print(f"Reward Mod: {rep['reward_modifier']}")
        
