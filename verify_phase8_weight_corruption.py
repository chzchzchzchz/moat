import os
import shutil
import sys
sys.path.insert(0, 'antigravity-engine/src')

from tokenizer import LlamaTokenizer
from native_bridge import NativeMetalEngine

print("=================================================================")
print("  PHASE 8: MICROSCOPIC SINGLE-BYTE WEIGHT CORRUPTION AUDIT       ")
print("=================================================================")

orig_path = 'models/tinyllama/model.safetensors'
corrupt_path = 'models/tinyllama/corrupt_test.safetensors'

# 1. Create a copy of the model file
shutil.copyfile(orig_path, corrupt_path)

# 2. Corrupt a single byte in the weight section (at offset 10,000,000)
with open(corrupt_path, 'r+b') as f:
    f.seek(10_000_000)
    orig_byte = f.read(1)
    # Flip bits of that single byte
    new_byte = bytes([orig_byte[0] ^ 0xFF])
    f.seek(10_000_000)
    f.write(new_byte)
    print(f"✅ Modified byte at offset 10,000,000: 0x{orig_byte.hex()} -> 0x{new_byte.hex()}")

tok = LlamaTokenizer('models/tinyllama/tokenizer.json')
prompt = "The capital city of France is"
prompt_ids = tok.encode(prompt)

print(f"\nPrompt: '{prompt}'")

# Run generation on original weights
print("\n--- Running on ORIGINAL model.safetensors ---")
orig_engine = NativeMetalEngine(
    dylib_path='libantigravity_engine.dylib',
    model_path=orig_path,
    n_channels=1
)
orig_tokens, _, _, _ = orig_engine.generate(prompt_ids, max_new_tokens=15, temperature=0.0, top_p=1.0)
orig_text = tok.decode(orig_tokens[0])
print(f"Original Text Output: '{orig_text}'")
del orig_engine

# Run generation on single-byte corrupted weights
print("\n--- Running on SINGLE-BYTE CORRUPTED corrupt_test.safetensors ---")
corrupt_engine = NativeMetalEngine(
    dylib_path='libantigravity_engine.dylib',
    model_path=corrupt_path,
    n_channels=1
)
corrupt_tokens, _, _, _ = corrupt_engine.generate(prompt_ids, max_new_tokens=15, temperature=0.0, top_p=1.0)
corrupt_text = tok.decode(corrupt_tokens[0])
print(f"Corrupted Text Output: '{corrupt_text}'")
del corrupt_engine

# Clean up temporary corrupt model copy
if os.path.exists(corrupt_path):
    os.remove(corrupt_path)
    print("\n✅ Cleaned up temporary corrupt_test.safetensors")

print("=================================================================")
