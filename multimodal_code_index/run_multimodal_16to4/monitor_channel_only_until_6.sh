#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_2gpu_by_model.sh"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-7200}"
MONITOR_UNTIL="${MONITOR_UNTIL:-$(date +%F) 06:00:00}"
PYTHON_BIN="${PYTHON_BIN:-/home/dlghdbs200/anaconda3/envs/hoyun_312/bin/python}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
LOG_EVERY="${LOG_EVERY:-1000}"
LWM_TEMPORAL_DEPTH="${LWM_TEMPORAL_DEPTH:-4}"
USE_AMP="${USE_AMP:-0}"
STATS_MAX_SAMPLES="${STATS_MAX_SAMPLES:-0}"

LOG_DIR="$SCRIPT_DIR/outputs/logs"
PID_DIR="$SCRIPT_DIR/outputs/pids"
STATE_DIR="$SCRIPT_DIR/outputs/state"
CURRENT_RUN_FILE="$STATE_DIR/channel_only_current_run_id"
mkdir -p "$LOG_DIR" "$PID_DIR" "$STATE_DIR"

cd "$REPO_ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

is_pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

run_id_alive() {
  local run_id="$1"
  local pid_file
  for pid_file in "$PID_DIR/${run_id}"*.pid; do
    [[ -e "$pid_file" ]] || continue
    if is_pid_alive "$(cat "$pid_file")"; then
      return 0
    fi
  done
  return 1
}

run_id_completed() {
  local run_id="$1"
  [[ -f "$STATE_DIR/${run_id}.done" ]]
}

print_failure_context() {
  local run_id="$1"
  local log_file
  log "Recent failure context for run_id=$run_id"
  for log_file in "$LOG_DIR/${run_id}"*.log; do
    [[ -e "$log_file" ]] || continue
    if rg -n "Traceback|RuntimeError|CUDA out of memory|Failed pair|Error|FileNotFoundError" "$log_file" >/dev/null 2>&1; then
      log "Errors in $log_file"
      rg -n "Traceback|RuntimeError|CUDA out of memory|Failed pair|Error|FileNotFoundError" "$log_file" | tail -n 20 || true
    fi
  done
}

start_channel_only() {
  local run_id="channel_only_$(date +%Y%m%d_%H%M%S)"
  local scheduler_pid_file="$PID_DIR/${run_id}_scheduler.pid"

  log "Starting channel_only experiment run_id=$run_id"
  nohup env \
    RUN_ID="$run_id" \
    PYTHON_BIN="$PYTHON_BIN" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    LR="$LR" \
    LOG_EVERY="$LOG_EVERY" \
    LWM_TEMPORAL_DEPTH="$LWM_TEMPORAL_DEPTH" \
    USE_AMP="$USE_AMP" \
    STATS_MAX_SAMPLES="$STATS_MAX_SAMPLES" \
    "$LAUNCH_SCRIPT" channel_only \
    >/dev/null 2>&1 &
  echo $! > "$scheduler_pid_file"
  echo "$run_id" > "$CURRENT_RUN_FILE"
  log "Started scheduler pid=$(cat "$scheduler_pid_file")"
}

check_once() {
  local run_id=""
  if [[ -f "$CURRENT_RUN_FILE" ]]; then
    run_id="$(cat "$CURRENT_RUN_FILE")"
  fi

  if [[ -n "$run_id" ]]; then
    if run_id_completed "$run_id"; then
      log "channel_only experiment completed run_id=$run_id"
      return 2
    fi

    if run_id_alive "$run_id"; then
      log "channel_only experiment running run_id=$run_id"
      return 0
    fi

    log "channel_only experiment is not running run_id=$run_id"
    print_failure_context "$run_id"
  else
    log "No current channel_only run_id found."
  fi

  start_channel_only
  return 0
}

end_ts="$(date -d "$MONITOR_UNTIL" +%s)"
if [[ "$(date +%s)" -gt "$end_ts" ]]; then
  log "MONITOR_UNTIL already passed: $MONITOR_UNTIL"
  exit 1
fi

log "Monitoring channel_only until $MONITOR_UNTIL every ${INTERVAL_SECONDS}s"
status=0
check_once || status=$?
if [[ "$status" == "2" ]]; then
  exit 0
fi

sleep 60
status=0
check_once || status=$?
if [[ "$status" == "2" ]]; then
  exit 0
fi

while [[ "$(date +%s)" -lt "$end_ts" ]]; do
  sleep "$INTERVAL_SECONDS"
  status=0
  check_once || status=$?
  if [[ "$status" == "2" ]]; then
    exit 0
  fi
done

log "Monitor reached deadline: $MONITOR_UNTIL"
