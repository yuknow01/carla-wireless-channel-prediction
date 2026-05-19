#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/dlghdbs200/anaconda3/envs/hoyun_312/bin/python}"
RUN_SCRIPT="$SCRIPT_DIR/run_16to4.py"
WAIT_PID="${WAIT_PID:?WAIT_PID is required}"
RUN_ID="${RUN_ID:-multimodal20_lwm_after_lstm_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$SCRIPT_DIR/outputs/logs"
PID_DIR="$SCRIPT_DIR/outputs/pids"
STATE_DIR="$SCRIPT_DIR/outputs/state"

mkdir -p "$LOG_DIR" "$PID_DIR" "$STATE_DIR"

STATE_FILE="$STATE_DIR/${RUN_ID}.state"
LOG_FILE="$LOG_DIR/${RUN_ID}_multimodal_lwm_gpu0.log"
PID_FILE="$PID_DIR/${RUN_ID}_multimodal_lwm_gpu0.pid"

{
  printf 'queued_at=%s\n' "$(date '+%F %T %Z')"
  printf 'watcher_pid=%s\n' "$$"
  printf 'wait_pid=%s\n' "$WAIT_PID"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'next=multimodal_lwm\n'
  printf 'gpu=CUDA_VISIBLE_DEVICES=0 device=cuda:0\n'
  printf 'log=%s\n' "$LOG_FILE"
} > "$STATE_FILE"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

{
  printf 'lstm_finished_at=%s\n' "$(date '+%F %T %Z')"
  printf 'stage=train_multimodal_lwm\n'
} >> "$STATE_FILE"

cd "$REPO_ROOT"
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 \
  "$PYTHON_BIN" "$RUN_SCRIPT" \
  --mode multimodal \
  --model lwm \
  --device cuda:0 \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --seed 42 \
  --log-every 10000 \
  --lwm-temporal-depth 4 \
  > "$LOG_FILE" 2>&1 &

child_pid=$!
echo "$child_pid" > "$PID_FILE"
printf 'launched_pid=%s\n' "$child_pid" >> "$STATE_FILE"

wait "$child_pid"
status=$?
printf 'completed_at=%s\n' "$(date '+%F %T %Z')" >> "$STATE_FILE"
printf 'exit_status=%s\n' "$status" >> "$STATE_FILE"
exit "$status"
