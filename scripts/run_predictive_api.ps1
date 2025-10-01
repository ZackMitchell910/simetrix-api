# Activate venv
Set-Location "$PSScriptRoot\.."
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

# Load env vars
if (Test-Path ".env") {
    Get-Content .env | ForEach-Object {
        if ($_ -match "^\s*#") { return }
        if ($_ -match "^\s*$") { return }
        $parts = $_.Split("=",2)
        if ($parts.Count -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
        }
    }
}

# Run FastAPI service
python -m uvicorn backend.src.predictive_service:app --reload --host 0.0.0.0 --port 8081