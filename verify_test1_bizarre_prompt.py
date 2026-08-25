import sys
sys.path.insert(0, 'antigravity-engine/src')

from tokenizer import LlamaTokenizer
from native_bridge import NativeMetalEngine

tok = LlamaTokenizer('models/tinyllama/tokenizer.json')
engine = NativeMetalEngine(
    dylib_path='libantigravity_engine.dylib',
    model_path='models/tinyllama/model.safetensors',
    n_channels=8
)

bizarre_prompt = "Write a three-word recipe for a shoe."
prompt_ids = tok.encode(bizarre_prompt)

print(f"Prompt: '{bizarre_prompt}'")
print(f"Prompt token IDs: {prompt_ids}")

tokens, logprobs, ttft, total = engine.generate(prompt_ids, max_new_tokens=20, temperature=0.7, top_p=0.9)

print("\nGenerated tokens per channel:")
for c in range(len(tokens)):
    decoded = tok.decode(tokens[c])
    print(f"  Channel {c}: '{decoded}'")
