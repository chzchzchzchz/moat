#!/usr/bin/env python3
"""
Merge Qwen3-4B safetensors shards into a single model.safetensors file.

The Antigravity engine's parseSafetensors reads a single .safetensors file
(see engine_paths.resolve_weights). Qwen3-4B ships as 3 shards, so we merge
them into one checkpoint the engine can load.
"""
import os
import sys

from safetensors import safe_open

SHARD_DIR = "/Users/MohssineChazi2/moat/antigravity-engine/models/qwen3-4b"
SHARDS = [
    os.path.join(SHARD_DIR, f"model-0000{i}-of-00003.safetensors") for i in (1, 2, 3)
]
OUT = os.path.join(SHARD_DIR, "model.safetensors")


def main():
    tensors = {}
    metadata = {}
    for shard in SHARDS:
        print(f"Reading {shard} ...")
        with safe_open(shard, framework="pt", device="cpu") as f:
            if f.metadata():
                metadata = f.metadata()
            for key in f.keys():
                if key in tensors:
                    print(f"  WARNING: duplicate key {key}, skipping")
                    continue
                tensors[key] = f.get_tensor(key)
    print(f"Total tensors: {len(tensors)}")

    from safetensors.torch import save_file
    print(f"Writing {OUT} ...")
    save_file(tensors, OUT, metadata=metadata or {"format": "pt"})
    size = os.path.getsize(OUT)
    print(f"Written {OUT} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
