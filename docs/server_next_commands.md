# Server Next Commands

This is the short handoff command sheet for continuing after the 2026-05-06
Main4 controlled-adapter smoke completion.

## Pull Latest

```bash
cd ~/projects/TRUCE-Rec
git pull --ff-only
```

## Recommended Server Entrypoints

For Week8 four-domain conversion plus Ours adapter prep:

```bash
cd ~/projects/TRUCE-Rec
bash scripts/server/run_week8_four_domain_pipeline.sh
```

For current Beauty controlled-adapter pilots with logs:

```bash
cd ~/projects/TRUCE-Rec
bash scripts/server/run_controlled_baseline_queue.sh smoke
bash scripts/server/run_controlled_baseline_queue.sh full
```

The full queue intentionally excludes slow OpenP5 for now.

## Full Run Fast Controlled Baselines

Use the Qwen/torch/peft environment:

```bash
cd ~/projects/TRUCE-Rec
source ~/projects/TALLRec/.venv_tallrec/bin/activate

for NAME in \
  tallrec_qwen3_lora_amazon_beauty \
  dealrec_qwen3_lora_amazon_beauty \
  lc_rec_qwen3_lora_amazon_beauty
do
  echo "===== FULL $NAME ====="
  python scripts/run_qwen_lora_controlled_baseline.py \
    --manifest outputs/server_training/controlled_baselines/$NAME/controlled_baseline_manifest.json \
    --trust-remote-code
done
```

Do not full-run `openp5_style_qwen3_lora_amazon_beauty` yet. Its smoke passed,
but current scoring is too slow for full evaluation.

## Import And Evaluate

After a full run completes:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

for NAME in \
  tallrec_qwen3_lora_amazon_beauty \
  dealrec_qwen3_lora_amazon_beauty \
  lc_rec_qwen3_lora_amazon_beauty
do
  case "$NAME" in
    tallrec_qwen3_lora_amazon_beauty) PROJECT=tallrec ;;
    dealrec_qwen3_lora_amazon_beauty) PROJECT=dealrec ;;
    lc_rec_qwen3_lora_amazon_beauty) PROJECT=lc_rec ;;
  esac
  BASE=outputs/server_training/controlled_baselines/$NAME
  RUN=outputs/runs/${NAME}_seed13
  mkdir -p "$RUN/artifacts"
  cp "$BASE/candidate_scores.csv" "$RUN/artifacts/candidate_scores.csv"
  python scripts/import_external_predictions.py \
    --scores "$RUN/artifacts/candidate_scores.csv" \
    --examples "outputs/server_packets/${PROJECT}_amazon_beauty/truce_examples.jsonl" \
    --output "$RUN/predictions.jsonl" \
    --method "$NAME" \
    --source-project "$PROJECT" \
    --model-name Qwen3-8B-LoRA \
    --seed 13 \
    --split test
  python scripts/evaluate_predictions.py \
    --predictions "$RUN/predictions.jsonl" \
    --output-dir "$RUN"
done
```

If shell prefix expansion is confusing, run each baseline manually with the
packet paths:

- TALLRec packet:
  `outputs/server_packets/tallrec_amazon_beauty/truce_examples.jsonl`
- DEALRec packet:
  `outputs/server_packets/dealrec_amazon_beauty/truce_examples.jsonl`
- LC-Rec packet:
  `outputs/server_packets/lc_rec_amazon_beauty/truce_examples.jsonl`

## Evidence Rule

Only imported `predictions.jsonl` and TRUCE `metrics.json`/`metrics.csv` are
paper-eligible. Smoke metrics and raw project-side logs are not paper results.

The current TRUCE-side Qwen3-LoRA adapter runs are controlled-adapter pilots
unless their fidelity to the official baseline repository has been audited.
Final main-table baselines should be official-native controlled runs: official
project implementation with shared TRUCE split/candidates/evaluator, Qwen3-8B
base-model substitution, LoRA adaptation, and official default or reported
optimal baseline hyperparameters. Ours may tune hyperparameters only on the
declared validation protocol.

## Status Summary

Summarize current controlled baseline status:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

python scripts/summarize_controlled_baseline_suite.py
```

This writes:

```text
outputs/summary/controlled_baselines/controlled_baseline_status.json
outputs/summary/controlled_baselines/controlled_baseline_status.csv
```

## Week8 Data Conversion

When the large same-candidate tasks are ready, convert one split/domain into a
TRUCE processed directory without resampling:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

python scripts/convert_week8_same_candidate_to_truce.py \
  --task-dir ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate \
  --output-dir data/processed/week8_same_candidate/books_large10000_100neg/test \
  --domain books \
  --split test \
  --strict-target-in-candidates
```

Repeat for `books`, `electronics`, and `movies`, and preserve the original
candidate/event alignment.

Generate all four-domain conversion commands:

```bash
python scripts/plan_four_domain_server_runs.py \
  --source-root ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  --output-root data/processed/week8_same_candidate \
  --domains beauty books electronics movies \
  --splits valid test
```

Validate converted artifacts:

```bash
python scripts/validate_week8_same_candidate_processed.py \
  --root data/processed/week8_same_candidate \
  --domains beauty books electronics movies \
  --splits valid test \
  --expected-users 10000 \
  --expected-candidates 101 \
  --expected-negatives 100
```
