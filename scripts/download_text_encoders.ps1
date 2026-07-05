# 通过 hf-mirror + curl 断点续传下载 HY-Motion 文本编码器
# 用法: powershell -ExecutionPolicy Bypass -File scripts/download_text_encoders.ps1

$ErrorActionPreference = "Stop"
$env:NO_PROXY = "*"
$env:no_proxy = "*"
$Base = "https://hf-mirror.com"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Download-File {
    param([string]$Url, [string]$OutPath)
    $dir = Split-Path $OutPath -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Write-Host ">>> Downloading: $OutPath"
    curl.exe -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 "$Url" -o "$OutPath"
    if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE): $Url" }
}

# CLIP (~1.7 GB)
Download-File `
    "$Base/openai/clip-vit-large-patch14/resolve/main/model.safetensors" `
    "$Root\ckpts\clip-vit-large-patch14\model.safetensors"

# Qwen3-8B (~16 GB, 5 shards)
$qwenDir = "$Root\ckpts\Qwen3-8B"
1..5 | ForEach-Object {
    $n = "{0:D5}" -f $_
    Download-File `
        "$Base/Qwen/Qwen3-8B/resolve/main/model-$n-of-00005.safetensors" `
        "$qwenDir\model-$n-of-00005.safetensors"
}

Write-Host ">>> 文本编码器下载完成。请设置: `$env:USE_HF_MODELS='0'"
