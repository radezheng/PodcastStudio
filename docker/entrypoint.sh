#!/usr/bin/env bash
set -euo pipefail

# Run both services in one container:
# - Backend API: port 8001
# - TTS worker:  port 8002
# We use uvicorn's --app-dir so both can keep their module name as `app.*`
# in separate directories without import collisions.

uvicorn app.main:app --app-dir /app/backend --host 0.0.0.0 --port 8001 &
backend_pid=$!

uvicorn app.main:app --app-dir /app/tts_worker --host 0.0.0.0 --port 8002 &
worker_pid=$!

term_handler() {
  kill -TERM "$backend_pid" "$worker_pid" 2>/dev/null || true
  wait || true
}

trap term_handler SIGTERM SIGINT

# Exit if either process exits; propagate its exit code.
wait -n "$backend_pid" "$worker_pid"
exit_code=$?

term_handler
exit "$exit_code"
