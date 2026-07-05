"""Robust HF downloader that bypasses hf_xet (slow from CN) by enumerating repo
files via the HF metadata API and fetching each with curl -L (fast + resumable).

Usage:
    python scripts/curl_download.py <repo_id> <local_dir> [--allow PREFIX]

Examples:
    python scripts/curl_download.py openai/clip-vit-large-patch14 ckpts/clip-vit-large-patch14
    python scripts/curl_download.py Qwen/Qwen3-8B ckpts/Qwen3-8B
    python scripts/curl_download.py tencent/HY-Motion-1.0 ckpts/tencent --allow HY-Motion-1.0/
"""
import argparse
import os
import subprocess
import sys

from huggingface_hub import HfApi

ENDPOINT = "https://huggingface.co"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("local_dir")
    ap.add_argument("--allow", default="", help="only files starting with this prefix")
    ap.add_argument("--skip", default="", help="comma-separated substrings to exclude")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    skips = [s for s in args.skip.split(",") if s]

    api = HfApi(endpoint=ENDPOINT)
    files = api.list_repo_files(repo_id=args.repo_id, revision=args.revision)
    files = [f for f in files if f.startswith(args.allow)]
    if skips:
        files = [f for f in files if not any(s in f for s in skips)]
    print(f"[{args.repo_id}] {len(files)} files to fetch -> {args.local_dir}")

    for i, rel in enumerate(files, 1):
        url = f"{ENDPOINT}/{args.repo_id}/resolve/{args.revision}/{rel}"
        dest = os.path.join(args.local_dir, rel)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        print(f"\n=== ({i}/{len(files)}) {rel} ===", flush=True)
        cmd = [
            "curl.exe", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
            "-C", "-", "-o", dest, url,
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"!! curl failed ({rc}) for {rel}", file=sys.stderr)
            sys.exit(rc)

    print(f"\nAll {len(files)} files downloaded into {args.local_dir}")


if __name__ == "__main__":
    main()
