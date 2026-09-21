from fastapi import FastAPI
from pydantic import BaseModel
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler
import numpy as np
import asyncio
import time
import os

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


app = FastAPI()

# Restoring the massive 4B model since we just unlocked True Batched Inference
path = os.path.join(MOAT_ROOT, "models/qwen3_5_4b_4bit")
print(f"Loading Journal Backend with {path}...")
model, tokenizer = load(path, model_config={"trust_remote_code": True})

generate_lock = asyncio.Lock()

class JournalEntry(BaseModel):
    entry: str

def score_reflection(text: str) -> float:
    words = text.split()
    if len(words) == 0: return -999.0
    unique_words = len(set(words))
    diversity_ratio = unique_words / len(words)
    length_score = np.log1p(len(words))
    penalty = 0.0
    lower_text = text.lower()
    if "as an ai" in lower_text or "i am an ai" in lower_text:
        penalty -= 10.0
    return (length_score * 0.4) + (diversity_ratio * 5.0) + penalty

@app.post("/reflect")
async def reflect_on_journal(entry: JournalEntry):
    print(f"Received Journal Entry: {entry.entry[:50]}...")
    prompt = f"<|im_start|>system\nYou are an empathetic, insightful private journaling assistant running entirely locally. Read the user's journal entry and provide a single, deep psychological or personal insight to help them reflect. Do not use generic AI disclaimers.<|im_end|>\n<|im_start|>user\nJournal Entry:\n{entry.entry}\n<|im_end|>\n<|im_start|>assistant\nHere is a deep reflection:\n"
    
    input_ids = tokenizer.encode(prompt)
    prompts = [input_ids for _ in range(4)]
    
    start_time = time.time()
    
    async with generate_lock:
        loop = asyncio.get_running_loop()
        # Batch Generate 4 paths simultaneously using shared memory bandwidth
        responses = await loop.run_in_executor(None, lambda: batch_generate(model, tokenizer, prompts=prompts, max_tokens=100, sampler=make_sampler(temp=0.8)))
    
    best_text = ""
    best_score = -9999.0
    
    for i, text in enumerate(responses.texts):
        score = score_reflection(text)
        print(f"Candidate {i} Score: {score:.2f}")
        if score > best_score:
            best_score = score
            best_text = text
            
    elapsed = time.time() - start_time
    print(f"Completed True Batched Test-Time Generation on 4B model in {elapsed:.2f}s")
            
    return {"reflection": best_text, "confidence_score": best_score}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
