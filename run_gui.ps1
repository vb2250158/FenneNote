$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir = ".venv-gpu"
$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
$CanvasCudnn = "C:\Program Files\NVIDIA Corporation\NVIDIA Canvas"
if (Test-Path $CudaBin) { $env:PATH = "$CudaBin;$env:PATH" }
if (Test-Path $CanvasCudnn) { $env:PATH = "$CanvasCudnn;$env:PATH" }

if (-not (Test-Path "$VenvDir\Scripts\python.exe")) {
    py -3.10 -m venv $VenvDir
}

& .\$VenvDir\Scripts\python.exe -m pip install -r requirements-gpu-cu11.txt
& .\$VenvDir\Scripts\python.exe -m pip install faster-whisper==0.10.1 --no-deps
& .\$VenvDir\Scripts\python.exe .\gui.py
