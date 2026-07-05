# -*- ps1 -*-
# 一键下载 HY-Motion / CLIP / Qwen3 模型权重到 ./ckpts/
# 需先安装 huggingface-cli:  pip install huggingface_hub
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1
#
# 可选环境变量:
#   $env:HF_ENDPOINT = "https://hf-mirror.com"   # 国内镜像加速

$ErrorActionPreference = "Stop"

$ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$CKPTS = Join-Path $ROOT "ckpts"
New-Item -ItemType Directory -Force -Path $CKPTS | Out-Null

function Invoke-HF {
    param([string]$Repo, [string]$Include, [string]$LocalDir)
    Write-Host ""
    Write-Host ">>> Downloading $Repo -> $LocalDir" -ForegroundColor Cyan
    $args = @("download", $Repo)
    if ($Include) { $args += @("--include", $Include) }
    $args += @("--local-dir", $LocalDir)
    & huggingface-cli @args
    if ($LASTEXITCODE -ne 0) {
        Write-Error "huggingface-cli failed for $Repo (exit=$LASTEXITCODE)"
    }
}

Write-Host "HY-Motion-MMD model downloader"
Write-Host "  repo root : $ROOT"
Write-Host "  ckpts dir : $CKPTS"
if ($env:HF_ENDPOINT) { Write-Host "  HF mirror : $($env:HF_ENDPOINT)" }

# 1) HY-Motion-1.0 主模型（约 4GB）
Invoke-HF -Repo "tencent/HY-Motion-1.0" -Include "HY-Motion-1.0/*" -LocalDir (Join-Path $CKPTS "tencent")

# 2) CLIP 文本编码器（约 1.7GB）
Invoke-HF -Repo "openai/clip-vit-large-patch14" -Include "" -LocalDir (Join-Path $CKPTS "clip-vit-large-patch14")

# 3) Qwen3-8B 文本编码器（约 16GB，CPU 推理需要 15GB+ 内存）
Invoke-HF -Repo "Qwen/Qwen3-8B" -Include "" -LocalDir (Join-Path $CKPTS "Qwen3-8B")

Write-Host ""
Write-Host ">>> All models downloaded into $CKPTS" -ForegroundColor Green
Write-Host "    Remember to set USE_HF_MODELS=0 to load from local ckpts/."
