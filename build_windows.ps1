$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir = ".venv-gpu"
$IconPath = "assets\fennenote.ico"
$FasterWhisperAssets = "$VenvDir\Lib\site-packages\faster_whisper\assets"

if (-not (Test-Path "$VenvDir\Scripts\python.exe")) {
    py -3.10 -m venv $VenvDir
}

& .\$VenvDir\Scripts\python.exe -m pip install -r requirements-gpu-cu11.txt
& .\$VenvDir\Scripts\python.exe -m pip install faster-whisper==0.10.1 --no-deps
& .\$VenvDir\Scripts\python.exe -m pip install pyinstaller

if (-not (Test-Path "$FasterWhisperAssets\silero_vad.onnx")) {
    throw "Missing faster-whisper VAD asset: $FasterWhisperAssets\silero_vad.onnx"
}

& .\$VenvDir\Scripts\pyinstaller.exe `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name FenneNote `
    --icon $IconPath `
    --add-data "assets;assets" `
    --add-data "config.example.json;." `
    --add-data "$FasterWhisperAssets;faster_whisper/assets" `
    --hidden-import sounddevice `
    --hidden-import faster_whisper `
    --hidden-import ctranslate2 `
    gui.py

Write-Host ""
Write-Host "Build complete: $ScriptDir\dist\FenneNote\FenneNote.exe"
Write-Host "Run this exe if you want Task Manager to show the FenneNote icon."
