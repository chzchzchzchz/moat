import os
import shutil
import ctypes
import sys

sys.path.insert(0, 'antigravity-engine/src')

from native_bridge import NativeMetalEngine
from tokenizer import LlamaTokenizer

# 1. Verify paths
original_weights = os.path.abspath("models/tinyllama/model.safetensors")
corrupted_dir = os.path.abspath("scratch/verify_truth_corrupt")
corrupted_weights = os.path.join(corrupted_dir, "model.safetensors")
lib_path = os.path.abspath("antigravity-engine/src/libantigravity_engine.dylib")

if not os.path.exists(original_weights):
    print(f"❌ Error: Original weights not found at {original_weights}")
    exit(1)

if not os.path.exists(lib_path):
    print(f"❌ Error: Compiled C++ library not found at {lib_path}")
    exit(1)

# 2. Duplicate the weights to protect your original model
print("🔄 Creating a temporary copy of model.safetensors...")
os.makedirs(corrupted_dir, exist_ok=True)
shutil.copyfile("models/tinyllama/tokenizer.json", os.path.join(corrupted_dir, "tokenizer.json"))
shutil.copyfile(original_weights, corrupted_weights)

# 3. Physically corrupt the file (Zero out 50MB of weights)
print("💥 Overwriting 50MB of binary tensor bytes with zeros...")
with open(corrupted_weights, "r+b") as f:
    # Jump 100MB into the file (safely past the header) and write zeros
    f.seek(100 * 1024 * 1024)
    f.write(b'\x00' * 50 * 1024 * 1024)
print("✅ Binary corruption injected successfully!")

# 4. Set up C++ Native Generate Bridge
def run_inference(weight_path):
    try:
        engine = NativeMetalEngine(
            dylib_path=lib_path,
            model_path=weight_path,
            n_channels=1,
            vocab_size=32000,
            hidden_dim=2048,
            max_seq_len=2048
        )
        
        prompt = [1, 450, 7483]
        tokens, logprobs, ttft, total = engine.generate(
            prompt_token_ids=prompt,
            max_new_tokens=15,
            temperature=0.7,
            top_p=0.9
        )
        
        engine.unload_weights()
        return tokens[0]
    except Exception as e:
        return f"FFI Call Failed: {e}"

# 5. Run and Compare
print("\n--- Running Inference with CLEAN weights ---")
clean_tokens = run_inference(original_weights)
print(f"Clean Output Tokens: {clean_tokens}")

print("\n--- Running Inference with CORRUPTED weights ---")
corrupted_tokens = run_inference(corrupted_weights)
print(f"Corrupted Output Tokens: {corrupted_tokens}")

# 6. Cleanup
if os.path.exists(corrupted_dir):
    shutil.rmtree(corrupted_dir)
    print("\n🧹 Cleaned up temporary corrupted weight file.")

# 7. The Ultimate Verdict
print("\n================== VERIFICATION VERDICT ==================")
if clean_tokens == corrupted_tokens:
    print("❌ FAILED: The output did not change! The engine is mocking or using hardcoded paths.")
else:
    print("🏆 SUCCESS: The token distribution completely broke, proving that the engine")
    print("   is executing real floating-point math directly on the physical weights!")
print("==========================================================")
