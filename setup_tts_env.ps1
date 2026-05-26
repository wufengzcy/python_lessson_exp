# ChatTTS setup - keep caches on E:
# Run: cd e:\cursor_files\python_homeworks\voice_assistant
#      powershell -ExecutionPolicy Bypass -File .\setup_tts_env.ps1

$ErrorActionPreference = "Stop"

$PythonRoot = "E:\Python312"
$ProjectDir = "E:\cursor_files\python_homeworks\voice_assistant"
$CacheRoot  = "E:\python_cache"
$PipCache   = "$CacheRoot\pip"
$HfHome     = "$CacheRoot\huggingface"
$TmpDir     = "$CacheRoot\tmp"

New-Item -ItemType Directory -Force -Path $PipCache, $HfHome, $TmpDir | Out-Null

[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", $PipCache, "User")
[Environment]::SetEnvironmentVariable("HF_HOME", $HfHome, "User")
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$addPaths = @("$PythonRoot", "$PythonRoot\Scripts")
foreach ($p in $addPaths) {
    if ($userPath -notlike "*$p*") {
        $userPath = "$p;$userPath"
    }
}
[Environment]::SetEnvironmentVariable("Path", $userPath, "User")

$env:PIP_CACHE_DIR = $PipCache
$env:HF_HOME = $HfHome
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:TEMP = $TmpDir
$env:TMP = $TmpDir
$env:Path = "$PythonRoot;$PythonRoot\Scripts;" + $env:Path

if (-not (Test-Path "$PythonRoot\python.exe")) {
    Write-Host "Missing $PythonRoot\python.exe"
    exit 1
}

Set-Location $ProjectDir

if (Test-Path ".venv") {
    Write-Host "Removing old .venv ..."
    Remove-Item -Recurse -Force ".venv"
}

Write-Host "Creating venv ..."
& "$PythonRoot\python.exe" -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -U pip

Write-Host "Installing PyTorch cu130 ..."
& ".\.venv\Scripts\pip.exe" install torch torchaudio --index-url https://download.pytorch.org/whl/cu130

Write-Host "Installing ChatTTS ..."
& ".\.venv\Scripts\pip.exe" install ChatTTS numpy soundfile huggingface_hub requests

Write-Host "Verify GPU ..."
& ".\.venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ok', torch.cuda.is_available())"

Write-Host "Verify ChatTTS ..."
& ".\.venv\Scripts\python.exe" -c "import ChatTTS; c=ChatTTS.Chat(); c.load(compile=False); print('ChatTTS OK')"

Write-Host "Done."
