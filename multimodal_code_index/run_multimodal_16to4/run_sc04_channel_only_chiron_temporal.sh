#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/dlghdbs200/anaconda3/envs/hoyun_312/bin/python}"
RUN_SCRIPT="$SCRIPT_DIR/run_16to4.py"
RUN_ID="${RUN_ID:-sc04_channel_only20_chiron_temporal_$(date +%Y%m%d_%H%M%S)}"
SCENARIO="${SCENARIO:-sc04}"

LOG_DIR="$SCRIPT_DIR/outputs/logs"
PID_DIR="$SCRIPT_DIR/outputs/pids"
STATE_DIR="$SCRIPT_DIR/outputs/state"
STATS_DIR="$SCRIPT_DIR/outputs/stats"

mkdir -p "$LOG_DIR" "$PID_DIR" "$STATE_DIR" "$STATS_DIR"

STATE_FILE="$STATE_DIR/${RUN_ID}.state"
STATS_FILE="$STATS_DIR/channel_stats_${SCENARIO}_nsc64.npz"
CHIRON_LOG="$LOG_DIR/${RUN_ID}_channel_only_chiron_gpu0.log"
TEMPORAL_LOG="$LOG_DIR/${RUN_ID}_channel_only_lwm_temporal_gpu1.log"

if [ ! -f "$STATS_FILE" ]; then
  echo "ERROR: stats file not found: $STATS_FILE" >&2
  exit 1
fi

{
  printf 'queued_at=%s\n' "$(date '+%F %T %Z')"
  printf 'watcher_pid=%s\n' "$$"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'scenario=%s\n' "$SCENARIO"
  printf 'mode=channel_only\n'
  printf 'models=chiron,lwm_temporal\n'
  printf 'settings=K16_P4_Nsc64_img8_epochs20_batch4_lr1e-4_seed42\n'
  printf 'stats_file=%s\n' "$STATS_FILE"
  printf 'chiron_gpu=CUDA_VISIBLE_DEVICES=0 device=cuda:0\n'
  printf 'lwm_temporal_gpu=CUDA_VISIBLE_DEVICES=1 device=cuda:0\n'
  printf 'chiron_log=%s\n' "$CHIRON_LOG"
  printf 'lwm_temporal_log=%s\n' "$TEMPORAL_LOG"
} > "$STATE_FILE"

cd "$REPO_ROOT"

printf 'stage=train_chiron_lwm_temporal\n' >> "$STATE_FILE"

env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "$PYTHON_BIN" "$RUN_SCRIPT" \
  --mode channel_only \
  --model chiron \
  --scenarios "$SCENARIO" \
  --device cuda:0 \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --seed 42 \
  --log-every 10000 \
  --lwm-temporal-depth 4 \
  --stats-file "$STATS_FILE" \
  > "$CHIRON_LOG" 2>&1 &
chiron_pid=$!
echo "$chiron_pid" > "$PID_DIR/${RUN_ID}_channel_only_chiron_gpu0.pid"
printf 'chiron_pid=%s\n' "$chiron_pid" >> "$STATE_FILE"

env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "$PYTHON_BIN" "$RUN_SCRIPT" \
  --mode channel_only \
  --model lwm_temporal \
  --scenarios "$SCENARIO" \
  --device cuda:0 \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --seed 42 \
  --log-every 10000 \
  --lwm-temporal-depth 4 \
  --stats-file "$STATS_FILE" \
  > "$TEMPORAL_LOG" 2>&1 &
lwm_temporal_pid=$!
echo "$lwm_temporal_pid" > "$PID_DIR/${RUN_ID}_channel_only_lwm_temporal_gpu1.pid"
printf 'lwm_temporal_pid=%s\n' "$lwm_temporal_pid" >> "$STATE_FILE"

set +e
wait "$chiron_pid"
chiron_status=$?
wait "$lwm_temporal_pid"
lwm_temporal_status=$?
set -e

printf 'completed_at=%s\n' "$(date '+%F %T %Z')" >> "$STATE_FILE"
printf 'chiron_exit_status=%s\n' "$chiron_status" >> "$STATE_FILE"
printf 'lwm_temporal_exit_status=%s\n' "$lwm_temporal_status" >> "$STATE_FILE"

if [ "$chiron_status" -ne 0 ] || [ "$lwm_temporal_status" -ne 0 ]; then
  exit 1
fi
