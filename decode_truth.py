import os
from transformers import AutoTokenizer

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


TOKENIZER_DIR = os.path.join(MOAT_ROOT, "models/tinyllama/")

clean_tokens = [17843, 6481, 9940, 2663, 823, 12117, 10538, 9139, 285, 29829, 491, 383, 22192, 4425, 297]
corrupted_tokens = [16915, 5952, 15643, 13, 6780, 16441, 10220, 9439, 4132, 23112, 1505, 10460, 17194, 1543, 24848]

print("🔄 Loading your local tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
except Exception as e:
    print(f"❌ Could not load tokenizer from {TOKENIZER_DIR}: {e}")
    exit(1)

clean_text = tokenizer.decode(clean_tokens)
corrupted_text = tokenizer.decode(corrupted_tokens)

print("\n=================== THE PHYSICAL TRUTH ===================")
print(f"CLEAN DECODED TEXT:\n👉 \"{clean_text}\"")
print("\n----------------------------------------------------------")
print(f"CORRUPTED DECODED TEXT:\n👉 \"{corrupted_text}\"")
print("==========================================================")
