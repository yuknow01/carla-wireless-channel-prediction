#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/dlghdbs200/anaconda3/envs/hoyun_312/bin/python}"
RUN_SCRIPT="$SCRIPT_DIR/run_16to4.py"
RUN_ID="${RUN_ID:-sc04_channel_only20_lwm_lstm_$(date +%Y%m%d_%H%M%S)}"
SCENARIO="${SCENARIO:-sc04}"

LOG_DIR="$SCRIPT_DIR/outputs/logs"
PID_DIR="$SCRIPT_DIR/outputs/pids"
STATE_DIR="$SCRIPT_DIR/outputs/state"
STATS_DIR="$SCRIPT_DIR/outputs/stats"

mkdir -p "$LOG_DIR" "$PID_DIR" "$STATE_DIR" "$STATS_DIR"

STATE_FILE="$STATE_DIR/${RUN_ID}.state"
STATS_FILE="$STATS_DIR/channel_stats_${SCENARIO}_nsc64.npz"
STATS_LOG="$LOG_DIR/${RUN_ID}_stats_${SCENARIO}.log"
LWM_LOG="$LOG_DIR/${RUN_ID}_channel_only_lwm_gpu0.log"
LSTM_LOG="$LOG_DIR/${RUN_ID}_channel_only_lstm_gpu1.log"

{
  printf 'queued_at=%s\n' "$(date '+%F %T %Z')"
  printf 'watcher_pid=%s\n' "$$"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'scenario=%s\n' "$SCENARIO"
  printf 'mode=channel_only\n'
  printf 'models=lwm,lstm\n'
  printf 'settings=K16_P4_Nsc64_img8_epochs20_batch4_lr1e-4_seed42\n'
  printf 'stats_file=%s\n' "$STATS_FILE"
  printf 'lwm_gpu=CUDA_VISIBLE_DEVICES=0 device=cuda:0\n'
  printf 'lstm_gpu=CUDA_VISIBLE_DEVICES=1 device=cuda:0\n'
  printf 'stats_log=%s\n' "$STATS_LOG"
  printf 'lwm_log=%s\n' "$LWM_LOG"
  printf 'lstm_log=%s\n' "$LSTM_LOG"
} > "$STATE_FILE"

cd "$REPO_ROOT"

printf 'stage=prepare_stats\n' >> "$STATE_FILE"
env PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "$PYTHON_BIN" "$RUN_SCRIPT" \
  --mode channel_only \
  --model lstm \
  --scenarios "$SCENARIO" \
  --device cpu \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --seed 42 \
  --log-every 10000 \
  --lwm-temporal-depth 4 \
  --stats-file "$STATS_FILE" \
  --dry-run \
  > "$STATS_LOG" 2>&1
printf 'stats_ready_at=%s\n' "$(date '+%F %T %Z')" >> "$STATE_FILE"

printf 'stage=train_lwm_lstm\n' >> "$STATE_FILE"
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "$PYTHON_BIN" "$RUN_SCRIPT" \
  --mode channel_only \
  --model lwm \
  --scenarios "$SCENARIO" \
  --device cuda:0 \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --seed 42 \
  --log-every 10000 \
  --lwm-temporal-depth 4 \
  --stats-file "$STATS_FILE" \
  > "$LWM_LOG" 2>&1 &
lwm_pid=$!
echo "$lwm_pid" > "$PID_DIR/${RUN_ID}_channel_only_lwm_gpu0.pid"
printf 'lwm_pid=%s\n' "$lwm_pid" >> "$STATE_FILE"

env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "$PYTHON_BIN" "$RUN_SCRIPT" \
  --mode channel_only \
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
echo "$lstm_pid" > "$PID_DIR/${RUN_ID}_channel_only_lstm_gpu1.pid"
printf 'lstm_pid=%s\n' "$lstm_pid" >> "$STATE_FILE"

set +e
wait "$lwm_pid"
lwm_status=$?
wait "$lstm_pid"
lstm_status=$?
set -e

printf 'completed_at=%s\n' "$(date '+%F %T %Z')" >> "$STATE_FILE"
printf 'lwm_exit_status=%s\n' "$lwm_status" >> "$STATE_FILE"
printf 'lstm_exit_status=%s\n' "$lstm_status" >> "$STATE_FILE"

if [ "$lwm_status" -ne 0 ] || [ "$lstm_status" -ne 0 ]; then
  exit 1
fi
