#!/usr/bin/env bash
# One-command launcher (macOS / Linux).
#   ./run.sh
# Stops both services on Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")"
export BACKEND_URL="http://127.0.0.1:8000"

echo "Starting backend on http://127.0.0.1:8000 ..."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

sleep 3

echo "Starting frontend on http://127.0.0.1:8501 ..."
python -m streamlit run frontend/streamlit_app.py \
  --server.port 8501 --server.address 127.0.0.1 &
FRONTEND_PID=$!

trap 'echo; echo "Stopping..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true' INT TERM

echo
echo "Backend : http://127.0.0.1:8000/docs"
echo "Frontend: http://127.0.0.1:8501"
echo "Login   : admin@example.com / admin12345"
wait
