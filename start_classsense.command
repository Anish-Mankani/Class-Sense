#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 server.py &
SERVER_PID=$!
cd frontend
npm run dev &
FRONT_PID=$!
sleep 5
open http://localhost:5173
trap "kill $SERVER_PID $FRONT_PID 2>/dev/null" EXIT
wait
