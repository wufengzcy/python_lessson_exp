# GPT-SoVITS 安装（无需 conda，复用现有 .venv）
# 用法: cd voice_assistant
#       .\.venv\Scripts\Activate.ps1
#       powershell -ExecutionPolicy Bypass -File .\setup_sovits.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EngineDir = Join-Path $ProjectDir "engines\GPT-SoVITS"
$Repo = "https://github.com/RVC-Boss/GPT-SoVITS.git"
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$VenvPip = Join-Path $ProjectDir ".venv\Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Missing .venv. Run setup_tts_env.ps1 first."
    exit 1
}

if (-not (Test-Path $EngineDir)) {
    Write-Host "Cloning GPT-SoVITS ..."
    git clone --depth 1 $Repo $EngineDir
} else {
    Write-Host "GPT-SoVITS already exists at $EngineDir"
}

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Download-LargeFile {
    param(
        [string]$Url,
        [string]$OutFile
    )
    Write-Host "  -> $OutFile"
    if (Test-Path $OutFile) { Remove-Item $OutFile -Force }

    # hf-mirror 会 302/308 跳转，PowerShell 的 Invoke-WebRequest 常失败，优先 curl
    if (Test-Command curl.exe) {
        curl.exe -L --retry 5 --retry-delay 3 --progress-bar -o $OutFile $Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl download failed: $Url"
        }
        return
    }

    # 备用：Python huggingface_hub
    $pyScript = @"
import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download
repo = sys.argv[1]
fname = sys.argv[2]
out = sys.argv[3]
path = hf_hub_download(repo_id=repo, filename=fname, local_dir=os.path.dirname(out))
import shutil
shutil.move(path, out)
"@
    & $VenvPython -c $pyScript "XXXXRT/GPT-SoVITS-Pretrained" (Split-Path $OutFile -Leaf) (Resolve-Path (Split-Path $OutFile -Parent)).Path
}

Write-Host "Checking FFmpeg ..."
if (-not (Test-Command ffmpeg)) {
    Write-Host "FFmpeg not found. Installing imageio-ffmpeg (bundled ffmpeg for M4A/MP3 import) ..."
} else {
    Write-Host "FFmpeg OK: $(Get-Command ffmpeg | Select-Object -ExpandProperty Source)"
}

Set-Location $EngineDir
$env:HF_ENDPOINT = "https://hf-mirror.com"

$PretrainedZip = Join-Path $EngineDir "pretrained_models.zip"
$PretrainedUrl = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
$G2PWZip = Join-Path $EngineDir "G2PWModel.zip"
$G2PWUrl = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
$Marker = Join-Path $EngineDir "GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s2G2333k.pth"

if (-not (Test-Path $Marker)) {
    Write-Host "Downloading pretrained models (several GB, please wait) ..."
    Download-LargeFile -Url $PretrainedUrl -OutFile $PretrainedZip
    Write-Host "Extracting pretrained models ..."
    Expand-Archive -Path $PretrainedZip -DestinationPath (Join-Path $EngineDir "GPT_SoVITS") -Force
    Remove-Item $PretrainedZip -Force
    Write-Host "Pretrained models OK."
} else {
    Write-Host "Pretrained models already present."
}

$G2PWMarker = Join-Path $EngineDir "GPT_SoVITS\text\G2PWModel"
if (-not (Test-Path $G2PWMarker)) {
    Write-Host "Downloading G2PWModel ..."
    Download-LargeFile -Url $G2PWUrl -OutFile $G2PWZip
    Expand-Archive -Path $G2PWZip -DestinationPath (Join-Path $EngineDir "GPT_SoVITS\text") -Force
    Remove-Item $G2PWZip -Force
    Write-Host "G2PWModel OK."
} else {
    Write-Host "G2PWModel already present."
}

Write-Host "Installing GPT-SoVITS Python deps into project .venv ..."
Set-Location $ProjectDir
& $VenvPip install -q huggingface_hub
& $VenvPip install -q pyyaml peft librosa transformers modelscope sentencepiece jieba jieba_fast cn2an pypinyin g2p_en wordsegment pydub imageio-ffmpeg ffmpeg-python x-transformers tensorboard "pytorch-lightning>=2.4" opencc-python-reimplemented onnxruntime-gpu
& $VenvPip install -q -r (Join-Path $EngineDir "extra-req.txt") --no-deps
& $VenvPip install -q -r (Join-Path $EngineDir "requirements.txt") 2>$null

Write-Host "Verify ..."
& $VenvPython -c "import sovits_core; print(sovits_core.engine_status_message())"

if (-not (Test-Path $Marker)) {
    Write-Host "ERROR: pretrained models still missing. Check network and retry."
    exit 1
}

Write-Host "Done."
