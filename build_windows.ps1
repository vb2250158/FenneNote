$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir = ".venv-gpu"
$IconPath = "assets\fennenote.ico"
$DistDir = Join-Path $ScriptDir "dist\FenneNote"
$DistConfigPath = Join-Path $DistDir "config.json"
$ConfigBackupDir = $null
$ConfigBackupPath = $null

if (Test-Path -LiteralPath $DistConfigPath) {
    $ConfigBackupDir = Join-Path ([System.IO.Path]::GetTempPath()) ("FenneNote-build-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $ConfigBackupDir | Out-Null
    $ConfigBackupPath = Join-Path $ConfigBackupDir "config.json"
    Copy-Item -LiteralPath $DistConfigPath -Destination $ConfigBackupPath -Force
    Write-Host "Preserved existing runtime config: $DistConfigPath"
}

if (-not (Test-Path "$VenvDir\Scripts\python.exe")) {
    py -3.10 -m venv $VenvDir
}

& .\$VenvDir\Scripts\python.exe -m pip install -r requirements-gpu-cu11.txt
& .\$VenvDir\Scripts\python.exe -m pip install faster-whisper==0.10.1 --no-deps
& .\$VenvDir\Scripts\python.exe -m pip install pyinstaller

try {
    & .\$VenvDir\Scripts\pyinstaller.exe `
        --noconfirm `
        --clean `
        --onedir `
        --contents-directory "." `
        --windowed `
        --name FenneNote `
        --icon $IconPath `
        --add-data "assets;assets" `
        --add-data "config.example.json;." `
        --hidden-import sounddevice `
        --hidden-import faster_whisper `
        --hidden-import ctranslate2 `
        --hidden-import PySide6 `
        qt_gui.py
}
finally {
    if ($ConfigBackupPath -and (Test-Path -LiteralPath $ConfigBackupPath)) {
        New-Item -ItemType Directory -Path $DistDir -Force | Out-Null
        Copy-Item -LiteralPath $ConfigBackupPath -Destination $DistConfigPath -Force
        Remove-Item -LiteralPath $ConfigBackupDir -Recurse -Force
        Write-Host "Restored runtime config: $DistConfigPath"
    }
}

Write-Host ""
Write-Host "Build complete: $ScriptDir\dist\FenneNote\FenneNote.exe"
Write-Host "Run this exe if you want Task Manager to show the FenneNote icon."
