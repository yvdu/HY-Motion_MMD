#!/usr/bin/env bash
# 一键下载 HY-Motion / CLIP / Qwen3 模型权重到 ./ckpts/
# 需先安装 huggingface-cli:  pip install huggingface_hub
#
# 用法:
#   bash scripts/download_models.sh
#
# 可选环境变量:
#   HF_ENDPOINT=https://hf-mirror.com        # 国内镜像加速

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPTS="$ROOT/ckpts"
mkdir -p "$CKPTS"

echo "HY-Motion-MMD model downloader"
echo "  repo root : $ROOT"
echo "  ckpts dir : $CKPTS"
[ -n "$HF_ENDPOINT" ] && echo "  HF mirror : $HF_ENDPOINT"

# 1) HY-Motion-1.0 主模型（约 4GB）
echo
echo ">>> Downloading tencent/HY-Motion-1.0 -> $CKPTS/tencent"
huggingface-cli download tencent/HY-Motion-1.0 \
    --include "HY-Motion-1.0/*" \
    --local-dir "$CKPTS/tencent"

# 2) CLIP 文本编码器（约 1.7GB）
echo
echo ">>> Downloading openai/clip-vit-large-patch14 -> $CKPTS/clip-vit-large-patch14"
huggingface-cli download openai/clip-vit-large-patch14 \
    --local-dir "$CKPTS/clip-vit-large-patch14"

# 3) Qwen3-8B 文本编码器（约 16GB，CPU 推理需要 15GB+ 内存）
echo
echo ">>> Downloading Qwen/Qwen3-8B -> $CKPTS/Qwen3-8B"
huggingface-cli download Qwen/Qwen3-8B \
    --local-dir "$CKPTS/Qwen3-8B"

echo
echo ">>> All models downloaded into $CKPTS"
echo "    Remember to set USE_HF_MODELS=0 to load from local ckpts/."
