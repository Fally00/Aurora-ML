$ErrorActionPreference = "Stop"

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw "Build Aurora.exe on Windows with a Windows Python environment."
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}

& $python -m PyInstaller --clean --noconfirm (Join-Path $PSScriptRoot "Aurora.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}
Write-Host "Built dist\Aurora.exe"
Write-Host "Run it from a console: .\dist\Aurora.exe [run|alerts|status]"
Write-Host "Windows Event Log clear monitoring requires running Aurora as administrator."
