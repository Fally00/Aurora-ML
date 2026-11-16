$ErrorActionPreference = "Stop"

if (-not $IsWindows) {
    throw "Build Aurora.exe on Windows with a Windows Python environment."
}

python -m PyInstaller --clean --noconfirm Aurora.spec
Write-Host "Built dist\Aurora.exe"
Write-Host "Run it from a console: .\dist\Aurora.exe [run|alerts|status]"
Write-Host "Windows Event Log clear monitoring requires running Aurora as administrator."
