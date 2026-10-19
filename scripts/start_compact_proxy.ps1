$ErrorActionPreference = "Stop"

# Starts the /compact proxy in front of llama-server (see COMPACT.md).
# Open http://localhost:8081 instead of http://localhost:8080 for the chat UI.
# Extra arguments pass through to scripts\compact_proxy.py (e.g. --threshold 0.6).

$DemoDir = Split-Path $PSScriptRoot -Parent
Set-Location $DemoDir

$VenvPy = Join-Path $DemoDir ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPy) { $VenvPy } else { "python" }

& $Python (Join-Path $PSScriptRoot "compact_proxy.py") @args
exit $LASTEXITCODE
