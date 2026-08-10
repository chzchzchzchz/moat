#!/usr/bin/env python3
"""
Download Skywork-o1-Open-PRM-Qwen-2.5-1.5B Process Reward Model.

This downloads the real PRM model weights for candidate verification
in the Antigravity Best-of-N pipeline.
"""

import os
import sys
import json
import urllib.request
import hashlib

MODEL_REPO = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "skywork-prm-1.5b")
HF_API = "https://huggingface.co/api/models"

# Files we need for the PRM model
REQUIRED_FILES = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

def download_file(url: str, dest: str, desc: str = ""):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        print(f"  [SKIP] {desc or os.path.basename(dest)} already exists")
        return True
    
    print(f"  [DOWNLOADING] {desc or os.path.basename(dest)}...")
    print(f"    URL: {url}")
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_dest = dest + ".tmp"
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "antigravity-engine/1.0")
        
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8 * 1024 * 1024  # 8 MB chunks
            
            with open(tmp_dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = (downloaded / total) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(f"    {mb_done:.1f} / {mb_total:.1f} MB ({pct:.1f}%)", end="\r")
            
            print(f"    Downloaded {downloaded / (1024*1024):.1f} MB")
        
        os.rename(tmp_dest, dest)
        return True
        
    except Exception as e:
        print(f"    [ERROR] Failed to download: {e}")
        if os.path.exists(tmp_dest):
            os.remove(tmp_dest)
        return False


def main():
    print("=" * 70)
    print("  Downloading Skywork-o1-Open-PRM-Qwen-2.5-1.5B")
    print("  Process Reward Model for Best-of-N Verification")
    print("=" * 70)
    print(f"  Repository: {MODEL_REPO}")
    print(f"  Destination: {MODEL_DIR}")
    print()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    base_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
    
    success = True
    for filename in REQUIRED_FILES:
        url = f"{base_url}/{filename}"
        dest = os.path.join(MODEL_DIR, filename)
        if not download_file(url, dest, filename):
            success = False
    
    if success:
        print()
        print("=" * 70)
        print("  Download complete!")
        
        # Verify config.json
        config_path = os.path.join(MODEL_DIR, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            print(f"  Model type: {config.get('model_type', 'unknown')}")
            print(f"  Hidden size: {config.get('hidden_size', 'unknown')}")
            print(f"  Num layers: {config.get('num_hidden_layers', 'unknown')}")
            print(f"  Vocab size: {config.get('vocab_size', 'unknown')}")
        
        # Check total size
        total_bytes = sum(
            os.path.getsize(os.path.join(MODEL_DIR, f))
            for f in REQUIRED_FILES
            if os.path.exists(os.path.join(MODEL_DIR, f))
        )
        print(f"  Total size: {total_bytes / (1024**3):.2f} GB")
        print("=" * 70)
    else:
        print("\n  [WARNING] Some files failed to download. Retry later.")
        sys.exit(1)


if __name__ == "__main__":
    main()
