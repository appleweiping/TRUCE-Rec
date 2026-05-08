#!/usr/bin/env bash
set -euo pipefail

cd "${TRUCE_REPO:-$HOME/projects/TRUCE-Rec}"

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: bash scripts/server/run_controlled_baseline_queue.sh [smoke|full]" >&2
  exit 2
fi

LOG_DIR="${TRUCE_SERVER_LOG_DIR:-outputs/server_logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

source "${QWEN_LORA_ENV:-$HOME/projects/TALLRec/.venv_tallrec/bin/activate}"

NAMES=(${TRUCE_CONTROLLED_BASELINES:-tallrec_qwen3_lora_amazon_beauty dealrec_qwen3_lora_amazon_beauty lc_rec_qwen3_lora_amazon_beauty})

for name in "${NAMES[@]}"; do
  manifest="outputs/server_training/controlled_baselines/$name/controlled_baseline_manifest.json"
  log="$LOG_DIR/${name}_${MODE}_${STAMP}.log"
  echo "===== $MODE $name ====="
  if [[ ! -f "$manifest" ]]; then
    echo "manifest not found: $manifest" >&2
    exit 1
  fi
  args=(python scripts/run_qwen_lora_controlled_baseline.py --manifest "$manifest" --trust-remote-code)
  if [[ "$MODE" == "smoke" ]]; then
    args+=(--max-train-examples 128 --max-steps 5 --max-score-rows 2)
  fi
  "${args[@]}" > "$log" 2>&1
  tail -n 20 "$log"
done

if [[ "${TRUCE_IMPORT_AFTER_BASELINES:-1}" == "1" ]]; then
  if [[ -f ".venv_truce/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv_truce/bin/activate
  fi
  for name in "${NAMES[@]}"; do
    python scripts/import_evaluate_controlled_baseline.py \
      --manifest "outputs/server_training/controlled_baselines/$name/controlled_baseline_manifest.json"
  done
  python scripts/summarize_controlled_baseline_suite.py
fi
