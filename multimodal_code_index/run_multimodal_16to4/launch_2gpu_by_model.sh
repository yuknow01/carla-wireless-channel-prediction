#!/usr/bin/env bash
set -euo pipefail

MODE_ARG="${1:-compare}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_16to4.py"

PYTHON_BIN="${PYTHON_BIN:-/home/dlghdbs200/anaconda3/envs/hoyun_312/bin/python}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-1000}"
LWM_TEMPORAL_DEPTH="${LWM_TEMPORAL_DEPTH:-4}"
STATS_MAX_SAMPLES="${STATS_MAX_SAMPLES:-0}"
USE_AMP="${USE_AMP:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

LOG_DIR="$SCRIPT_DIR/outputs/logs"
PID_DIR="$SCRIPT_DIR/outputs/pids"
STATE_DIR="$SCRIPT_DIR/outputs/state"
mkdir -p "$LOG_DIR" "$PID_DIR" "$STATE_DIR"
LAUNCHED_PID=""

cd "$REPO_ROOT"

case "$MODE_ARG" in
  channel_only)
    MODES=("channel_only")
    ;;
  multimodal)
    MODES=("multimodal")
    ;;
  compare)
    MODES=("channel_only" "multimodal")
    ;;
  *)
    echo "Usage: $0 [channel_only|multimodal|compare]" >&2
    exit 2
    ;;
esac

echo "run_id=$RUN_ID"
echo "modes=${MODES[*]}"
echo "epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR seed=$SEED amp=$USE_AMP lwm_temporal_depth=$LWM_TEMPORAL_DEPTH"
echo "logs=$LOG_DIR"
echo "pids=$PID_DIR"

echo "Preparing normalization stats once before parallel training..."
STATS_LOG="$(mktemp "${TMPDIR:-/tmp}/${RUN_ID}_stats.XXXXXX.log")"
set +e
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$RUN_SCRIPT" \
  --mode channel_only \
  --model lstm \
  --dry-run \
  --max-train-samples 4 \
  --max-val-samples 4 \
  --seed "$SEED" \
  --stats-max-samples "$STATS_MAX_SAMPLES" \
  > "$STATS_LOG" 2>&1
stats_status=$?
set -e
echo "Stats command exit=$stats_status"
if [[ "$stats_status" -ne 0 ]]; then
  echo "Stats check failed. Last stats log lines:"
  tail -n 80 "$STATS_LOG" || true
  echo "Full stats log: $STATS_LOG"
  exit "$stats_status"
fi
rm -f "$STATS_LOG"
echo "Stats check complete"

launch_model() {
  local mode="$1"
  local model="$2"
  local device="$3"
  local device_tag="${device//:/}"
  local log_file="$LOG_DIR/${RUN_ID}_${mode}_${model}_${device_tag}.log"
  local pid_file="$PID_DIR/${RUN_ID}_${mode}_${model}_${device_tag}.pid"
  local cmd=(
    env PYTHONUNBUFFERED=1 PYTHONHASHSEED="$SEED"
    "$PYTHON_BIN" "$RUN_SCRIPT"
    --mode "$mode"
    --model "$model"
    --device "$device"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --seed "$SEED"
    --log-every "$LOG_EVERY"
    --lwm-temporal-depth "$LWM_TEMPORAL_DEPTH"
  )

  if [[ "$USE_AMP" == "1" ]]; then
    cmd+=(--amp)
  fi

  echo "Launching mode=$mode model=$model device=$device log=$log_file" >&2
  "${cmd[@]}" > "$log_file" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_file"
  LAUNCHED_PID="$pid"
}

run_pair() {
  local mode="$1"
  local model_a="$2"
  local device_a="$3"
  local model_b="$4"
  local device_b="$5"
  local pid_a
  local pid_b
  local status_a
  local status_b

  launch_model "$mode" "$model_a" "$device_a"
  pid_a="$LAUNCHED_PID"
  launch_model "$mode" "$model_b" "$device_b"
  pid_b="$LAUNCHED_PID"

  set +e
  wait "$pid_a"
  status_a=$?
  wait "$pid_b"
  status_b=$?
  set -e

  if [[ "$status_a" -ne 0 || "$status_b" -ne 0 ]]; then
    echo "Failed pair: mode=$mode $model_a status=$status_a, $model_b status=$status_b" >&2
    exit 1
  fi
}

for mode in "${MODES[@]}"; do
  echo "Starting mode=$mode wave 1: cuda:0=chiron, cuda:1=lwm_temporal"
  run_pair "$mode" chiron cuda:0 lwm_temporal cuda:1

  echo "Starting mode=$mode wave 2: cuda:0=lstm, cuda:1=lwm"
  run_pair "$mode" lstm cuda:0 lwm cuda:1
done

printf 'completed_at=%s\n' "$(date '+%F %T %Z')" > "$STATE_DIR/${RUN_ID}.done"
echo "All requested 2-GPU experiments completed."
