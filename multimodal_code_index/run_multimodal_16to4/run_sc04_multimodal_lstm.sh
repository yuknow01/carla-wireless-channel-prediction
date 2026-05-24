#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/dlghdbs200/anaconda3/envs/hoyun_312/bin/python}"
RUN_SCRIPT="$SCRIPT_DIR/run_16to4.py"
RUN_ID="${RUN_ID:-sc04_multimodal_lstm_$(date +%Y%m%d_%H%M%S)}"
SCENARIO="${SCENARIO:-sc04}"
GPU_ID="${GPU_ID:-0}"

LOG_DIR="$SCRIPT_DIR/outputs/logs"
PID_DIR="$SCRIPT_DIR/outputs/pids"
STATE_DIR="$SCRIPT_DIR/outputs/state"
STATS_DIR="$SCRIPT_DIR/outputs/stats"

mkdir -p "$LOG_DIR" "$PID_DIR" "$STATE_DIR" "$STATS_DIR"

STATE_FILE="$STATE_DIR/${RUN_ID}.state"
STATS_FILE="$STATS_DIR/channel_stats_${SCENARIO}_nsc64.npz"
LSTM_LOG="$LOG_DIR/${RUN_ID}_multimodal_lstm_gpu${GPU_ID}.log"

if [ ! -f "$STATS_FILE" ]; then
  echo "ERROR: stats file not found: $STATS_FILE" >&2
  exit 1
fi

{
  printf 'queued_at=%s\n' "$(date '+%F %T %Z')"
  printf 'watcher_pid=%s\n' "$$"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'scenario=%s\n' "$SCENARIO"
  printf 'mode=multimodal\n'
  printf 'models=lstm\n'
  printf 'settings=K16_P4_Nsc64_img8_epochs20_batch4_lr1e-4_seed42\n'
  printf 'stats_file=%s\n' "$STATS_FILE"
  printf 'lstm_gpu=CUDA_VISIBLE_DEVICES=%s device=cuda:0\n' "$GPU_ID"
  printf 'lstm_log=%s\n' "$LSTM_LOG"
} > "$STATE_FILE"

cd "$REPO_ROOT"

printf 'stage=train_multimodal_lstm\n' >> "$STATE_FILE"

env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "$PYTHON_BIN" "$RUN_SCRIPT" \
  --mode multimodal \
  --model lstm \
  --scenarios "$SCENARIO" \
  --device cuda:0 \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --seed 42 \
  --log-every 10000 \
  --lwm-temporal-depth 4 \
  --stats-file "$STATS_FILE" \
  > "$LSTM_LOG" 2>&1 &
lstm_pid=$!
echo "$lstm_pid" > "$PID_DIR/${RUN_ID}_multimodal_lstm_gpu${GPU_ID}.pid"
printf 'lstm_pid=%s\n' "$lstm_pid" >> "$STATE_FILE"

set +e
wait "$lstm_pid"
lstm_status=$?
set -e

printf 'completed_at=%s\n' "$(date '+%F %T %Z')" >> "$STATE_FILE"
printf 'lstm_exit_status=%s\n' "$lstm_status" >> "$STATE_FILE"

if [ "$lstm_status" -ne 0 ]; then
  exit 1
fi
