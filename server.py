from fastapi import FastAPI
from pydantic import BaseModel
from mlx_lm import load, generate
import sys

app = FastAPI()

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/MohssineChazi2/moat/models/qwen3_5_4b_4bit"
print(f"Loading {MODEL_PATH}...")
model, tokenizer = load(MODEL_PATH, model_config={"trust_remote_code": True})

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

@app.post("/generate")
def generate_text(req: GenerateRequest):
    # MLX generate is blocking, but very fast on Apple Silicon
    response = generate(model, tokenizer, prompt=req.prompt, max_tokens=req.max_tokens, verbose=False)
    return {"text": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
