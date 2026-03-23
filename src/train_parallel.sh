#!/usr/bin/env bash
# Run several training jobs in parallel; compare in one TensorBoard UI.
#
#   chmod +x train_parallel.sh
#   ./train_parallel.sh
#   tensorboard --logdir ./tensorboard_logs
#
# Uses BANANAGRAML_HEADLESS=1 so pygame does not open N windows.

set -euo pipefail
cd "$(dirname "$0")"

RUNS="${1:-3}"
TS="${2:-90000}"
TB_ROOT="${TB_ROOT:-./tensorboard_logs}"

# export BANANAGRAML_HEADLESS="${BANANAGRAML_HEADLESS:-1}"

pids=()
for i in $(seq 1 "$RUNS"); do
  name="parallel_${i}"
  seed=$((4200 + i))
  echo "Starting $name seed=$seed timesteps=$TS"
  python train.py --run-name "$name" --seed "$seed" --timesteps "$TS" --tb-dir "$TB_ROOT" &
  pids+=($!)
done

echo "Waiting on PIDs: ${pids[*]}"
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All runs finished. Compare runs:"
echo "  tensorboard --logdir $(pwd)/$TB_ROOT"
