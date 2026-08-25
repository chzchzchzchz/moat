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

bizarre_prompt = "Explain how a physical spoon feels when eating hot soup."
prompt_ids = tok.encode(bizarre_prompt)

print(f"Prompt: '{bizarre_prompt}'")
print(f"Prompt token IDs: {prompt_ids}\n")

tokens, logprobs, ttft, total = engine.generate(prompt_ids, max_new_tokens=30, temperature=0.7, top_p=0.9)

print("=================================================================")
print("  PHASE 4: 8-CHANNEL RAW DECODE OUTPUTS FOR BIZARRE SPOON PROMPT ")
print("=================================================================")
for c in range(len(tokens)):
    decoded = tok.decode(tokens[c])
    print(f"  Channel {c} (logprob={logprobs[c]:.2f}): '{decoded}'")
print("=================================================================")
