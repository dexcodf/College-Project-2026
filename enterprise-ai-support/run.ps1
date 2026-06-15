# One-command launcher (Windows / PowerShell).
# Starts the FastAPI backend and the Streamlit frontend in separate windows.
#
#   powershell -ExecutionPolicy Bypass -File run.ps1
#
# Stop everything later with:  Get-Process python,streamlit -ErrorAction SilentlyContinue | Stop-Process

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$env:BACKEND_URL = "http://127.0.0.1:8000"

Write-Host "Starting backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Start-Process -FilePath "python" `
  -ArgumentList "-m","uvicorn","app.main:app","--host","127.0.0.1","--port","8000" `
  -WorkingDirectory $root

Start-Sleep -Seconds 3

Write-Host "Starting frontend on http://127.0.0.1:8501 ..." -ForegroundColor Cyan
Start-Process -FilePath "python" `
  -ArgumentList "-m","streamlit","run","frontend/streamlit_app.py","--server.port","8501","--server.address","127.0.0.1" `
  -WorkingDirectory $root

Write-Host ""
Write-Host "Backend : http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "Frontend: http://127.0.0.1:8501" -ForegroundColor Green
Write-Host "Login   : admin@example.com / admin12345" -ForegroundColor Yellow
