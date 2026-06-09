# 安装 GPT-SoVITS 到 engines/GPT-SoVITS（本地 import，非 HTTP API）
# 用法: powershell -ExecutionPolicy Bypass -File .\setup_sovits.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EngineDir = Join-Path $ProjectDir "engines\GPT-SoVITS"
$Repo = "https://github.com/RVC-Boss/GPT-SoVITS.git"

if (-not (Test-Path $EngineDir)) {
    Write-Host "Cloning GPT-SoVITS ..."
    git clone --depth 1 $Repo $EngineDir
} else {
    Write-Host "GPT-SoVITS already exists at $EngineDir"
}

Set-Location $EngineDir

Write-Host "Installing GPT-SoVITS dependencies and pretrained models ..."
Write-Host "This may take a while (several GB)."

# 优先 CU126；RTX 50 系若失败可改 CU128
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Device CU126 -Source HF-Mirror

Write-Host "Done. Pretrained models should be under GPT_SoVITS\pretrained_models\"
