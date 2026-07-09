$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir = ".venv-gpu"
$IconPath = "assets\fennenote.ico"
$DistDir = Join-Path $ScriptDir "dist\FenneNote"
$RuntimeBackupDir = $null
$RuntimeItems = @(
    "config.json",
    "transcripts",
    "cache",
    "reference-cache",
    "voice-references",
    "runs"
)

if (Test-Path -LiteralPath $DistDir) {
    $RuntimeBackupDir = Join-Path ([System.IO.Path]::GetTempPath()) ("FenneNote-build-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $RuntimeBackupDir | Out-Null
    $MovedDistDir = Join-Path $RuntimeBackupDir "FenneNote"
    Move-Item -LiteralPath $DistDir -Destination $MovedDistDir -Force
    foreach ($Item in $RuntimeItems) {
        $Source = Join-Path $MovedDistDir $Item
        if (Test-Path -LiteralPath $Source) {
            Write-Host "Preserved runtime data: $Source"
        }
    }
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
    if ($RuntimeBackupDir -and (Test-Path -LiteralPath $RuntimeBackupDir)) {
        New-Item -ItemType Directory -Path $DistDir -Force | Out-Null
        foreach ($Item in $RuntimeItems) {
            $Backup = Join-Path (Join-Path $RuntimeBackupDir "FenneNote") $Item
            if (Test-Path -LiteralPath $Backup) {
                Copy-Item -LiteralPath $Backup -Destination (Join-Path $DistDir $Item) -Recurse -Force
                Write-Host "Restored runtime data: $(Join-Path $DistDir $Item)"
            }
        }
        Remove-Item -LiteralPath $RuntimeBackupDir -Recurse -Force
    }
}

Write-Host ""
Write-Host "Build complete: $ScriptDir\dist\FenneNote\FenneNote.exe"
Write-Host "Run this exe if you want Task Manager to show the FenneNote icon."
