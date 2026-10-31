$ErrorActionPreference = "Stop"

if (-not $IsWindows) {
    throw "Build Aurora.exe on Windows with a Windows Python environment."
}

python -m PyInstaller --clean --noconfirm Aurora.spec
Write-Host "Built dist\Aurora.exe"
Write-Host "Windows Event Log clear monitoring requires running Aurora as administrator."
