#!/usr/bin/env bash
set -euo pipefail

cd "${TRUCE_REPO:-$HOME/projects/TRUCE-Rec}"

SOURCE_ROOT="${WEEK8_SOURCE_ROOT:-$HOME/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks}"
OUTPUT_ROOT="${WEEK8_OUTPUT_ROOT:-data/processed/week8_same_candidate}"
DOMAINS="${WEEK8_DOMAINS:-beauty books electronics movies}"
SPLITS="${WEEK8_SPLITS:-valid test}"
LOG_DIR="${TRUCE_SERVER_LOG_DIR:-outputs/server_logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/week8_four_domain_${STAMP}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== TRUCE Week8 four-domain pipeline ====="
echo "repo=$(pwd)"
echo "source_root=$SOURCE_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "domains=$DOMAINS"
echo "splits=$SPLITS"
echo "log=$LOG_FILE"

if [[ -f ".venv_truce/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv_truce/bin/activate
fi

git pull --ff-only

python scripts/server/dry_run_week8_four_domain.py \
  --source-root "$SOURCE_ROOT" \
  --domains $DOMAINS \
  --splits $SPLITS

mapfile -t COMMANDS < <(python scripts/plan_four_domain_server_runs.py \
  --source-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --domains $DOMAINS \
  --splits $SPLITS \
  --include-ours-adapter-prep \
  --emit-json | python -c "import json,sys; [print(x) for x in json.load(sys.stdin)]")

for cmd in "${COMMANDS[@]}"; do
  echo "===== RUN: $cmd ====="
  eval "$cmd"
done

python scripts/validate_week8_same_candidate_processed.py \
  --root "$OUTPUT_ROOT" \
  --domains $DOMAINS \
  --splits $SPLITS \
  --expected-users "${WEEK8_EXPECTED_USERS:-10000}" \
  --expected-candidates "${WEEK8_EXPECTED_CANDIDATES:-101}" \
  --expected-negatives "${WEEK8_EXPECTED_NEGATIVES:-100}"

echo "===== completed week8 conversion + Ours adapter prep ====="
