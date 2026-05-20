#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/dlghdbs200/anaconda3/envs/hoyun_312/bin/python}"
RUN_SCRIPT="$SCRIPT_DIR/run_16to4.py"
WAIT_PID="${WAIT_PID:?WAIT_PID is required}"
RUN_ID="${RUN_ID:-multimodal20_chiron_after_lwm_$(date +%Y%m%d_%H%M%S)}"
PREV_LOG="${PREV_LOG:?PREV_LOG is required}"
LOG_DIR="$SCRIPT_DIR/outputs/logs"
PID_DIR="$SCRIPT_DIR/outputs/pids"
STATE_DIR="$SCRIPT_DIR/outputs/state"

mkdir -p "$LOG_DIR" "$PID_DIR" "$STATE_DIR"

STATE_FILE="$STATE_DIR/${RUN_ID}.state"
LOG_FILE="$LOG_DIR/${RUN_ID}_multimodal_chiron_gpu0.log"
PID_FILE="$PID_DIR/${RUN_ID}_multimodal_chiron_gpu0.pid"

{
  printf 'queued_at=%s\n' "$(date '+%F %T %Z')"
  printf 'watcher_pid=%s\n' "$$"
  printf 'wait_pid=%s\n' "$WAIT_PID"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'previous_log=%s\n' "$PREV_LOG"
  printf 'next=multimodal_chiron\n'
  printf 'gpu=CUDA_VISIBLE_DEVICES=0 device=cuda:0\n'
  printf 'log=%s\n' "$LOG_FILE"
} > "$STATE_FILE"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

{
  printf 'lwm_finished_at=%s\n' "$(date '+%F %T %Z')"
  printf 'stage=check_lwm_completion\n'
} >> "$STATE_FILE"

for _ in {1..10}; do
  if grep -q 'Summary saved:' "$PREV_LOG"; then
    break
  fi
  sleep 30
done

if ! grep -q 'Summary saved:' "$PREV_LOG"; then
  printf 'abort_reason=lwm_summary_not_found\n' >> "$STATE_FILE"
  exit 1
fi

printf 'stage=train_multimodal_chiron\n' >> "$STATE_FILE"

cd "$REPO_ROOT"
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 \
  "$PYTHON_BIN" "$RUN_SCRIPT" \
  --mode multimodal \
  --model chiron \
  --device cuda:0 \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --seed 42 \
  --log-every 10000 \
  > "$LOG_FILE" 2>&1 &

child_pid=$!
echo "$child_pid" > "$PID_FILE"
printf 'launched_pid=%s\n' "$child_pid" >> "$STATE_FILE"

wait "$child_pid"
status=$?
printf 'completed_at=%s\n' "$(date '+%F %T %Z')" >> "$STATE_FILE"
printf 'exit_status=%s\n' "$status" >> "$STATE_FILE"
exit "$status"
