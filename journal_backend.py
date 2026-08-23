from fastapi import FastAPI
from pydantic import BaseModel
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import numpy as np

app = FastAPI()

path = "/Users/MohssineChazi2/moat/models/qwen3_5_4b_4bit"
print(f"Loading Journal Backend with {path}...")
model, tokenizer = load(path, model_config={"trust_remote_code": True})

class JournalEntry(BaseModel):
    entry: str

def score_reflection(text: str) -> float:
    # Heuristic ListWise Verifier proxy for text quality without an actual PRM:
    # 1. Favor longer, detailed insights (up to a point)
    # 2. Penalize repetitive characters/words (low entropy = bad)
    words = text.split()
    if len(words) == 0: return -999.0
    
    unique_words = len(set(words))
    diversity_ratio = unique_words / len(words)
    
    length_score = np.log1p(len(words))
    
    # Penalize "As an AI" or generic phrases
    penalty = 0.0
    lower_text = text.lower()
    if "as an ai" in lower_text or "i am an ai" in lower_text:
        penalty -= 10.0
        
    return (length_score * 0.4) + (diversity_ratio * 5.0) + penalty

@app.post("/reflect")
def reflect_on_journal(entry: JournalEntry):
    print(f"Received Journal Entry: {entry.entry[:50]}...")
    
    prompt = f"<|im_start|>system\nYou are an empathetic, insightful private journaling assistant running entirely locally. Read the user's journal entry and provide a single, deep psychological or personal insight to help them reflect. Do not use generic AI disclaimers.<|im_end|>\n<|im_start|>user\nJournal Entry:\n{entry.entry}\n<|im_end|>\n<|im_start|>assistant\nHere is a deep reflection:\n"
    
    best_text = ""
    best_score = -9999.0
    
    # Simulate Edge Test-Time Compute (Best-of-N)
    for i in range(4):
        resp = generate(model, tokenizer, prompt=prompt, max_tokens=150, sampler=make_sampler(temp=0.8), verbose=False)
        score = score_reflection(resp)
        print(f"Candidate {i} Score: {score:.2f}")
        if score > best_score:
            best_score = score
            best_text = resp
            
    return {"reflection": best_text, "confidence_score": best_score}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
