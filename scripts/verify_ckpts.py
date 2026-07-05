"""Quick integrity check of downloaded weights (opens safetensors headers only)."""
import json
import os

import torch
from safetensors import safe_open

BASE = "ckpts"


def check_st(path):
    with safe_open(path, framework="pt") as f:
        return len(f.keys())


def main():
    ok = True

    # HY-Motion ckpt
    ckpt = os.path.join(BASE, "tencent", "HY-Motion-1.0", "latest.ckpt")
    sz = os.path.getsize(ckpt) / 1e9
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    ntensors = sum(1 for _ in (sd.get("state_dict", sd)).items()) if isinstance(sd, dict) else -1
    print(f"[OK] HY-Motion latest.ckpt  {sz:.2f} GB, top-level keys={list(sd.keys())[:6] if isinstance(sd, dict) else type(sd)}")

    # CLIP
    clip = os.path.join(BASE, "clip-vit-large-patch14", "model.safetensors")
    print(f"[OK] CLIP model.safetensors  {check_st(clip)} tensors")

    # Qwen3 shards via index
    idx_path = os.path.join(BASE, "Qwen3-8B", "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    shards = sorted(set(idx["weight_map"].values()))
    for s in shards:
        p = os.path.join(BASE, "Qwen3-8B", s)
        if not os.path.exists(p):
            print(f"[MISSING] Qwen3 shard {s}")
            ok = False
            continue
        print(f"[OK] Qwen3 {s}  {check_st(p)} tensors")
    print(f"Qwen3 total mapped params: {len(idx['weight_map'])}, shards: {len(shards)}")

    print("\nRESULT:", "ALL GOOD" if ok else "PROBLEMS FOUND")


if __name__ == "__main__":
    main()
