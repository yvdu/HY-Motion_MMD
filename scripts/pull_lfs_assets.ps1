# 拉取 Git LFS 资源（dump_wooden 骨骼网格 + stats 均值方差）
# 仓库非 git clone 时，Mean.npy / v_template.bin 等会是 LFS 指针，导致推理报错。
# 用法: powershell -ExecutionPolicy Bypass -File scripts/pull_lfs_assets.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Tmp = Join-Path $Root "_tmp_lfs"
$Repo = "https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git"

if (Test-Path $Tmp) { Remove-Item $Tmp -Recurse -Force }
New-Item -ItemType Directory -Path $Tmp | Out-Null

Push-Location $Tmp
git clone --filter=blob:none --sparse --depth=1 $Repo
Set-Location HY-Motion-1.0
git sparse-checkout set scripts/gradio/static/assets/dump_wooden stats
git lfs pull

Copy-Item "scripts\gradio\static\assets\dump_wooden\*.bin" (Join-Path $Root "scripts\gradio\static\assets\dump_wooden\") -Force
Copy-Item "scripts\gradio\static\assets\dump_wooden\joint_names.json" (Join-Path $Root "scripts\gradio\static\assets\dump_wooden\") -Force
Copy-Item "stats\Mean.npy","stats\Std.npy" (Join-Path $Root "stats\") -Force

Pop-Location
Remove-Item $Tmp -Recurse -Force
Write-Host ">>> LFS assets copied. v_template.bin size:" (Get-Item (Join-Path $Root "scripts\gradio\static\assets\dump_wooden\v_template.bin")).Length
