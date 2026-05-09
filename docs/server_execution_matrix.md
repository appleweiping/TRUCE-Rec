# Server Execution Matrix

This document is the server-side checklist for moving from current pilots to a
complete four-domain experiment suite. It contains commands and acceptance
criteria, not results.

## Domains

The target full-scale domains are:

| Domain | Week8 source | Intended TRUCE output |
| --- | --- | --- |
| beauty | `beauty_supplementary_smallerN_100neg` | `data/processed/week8_same_candidate/beauty_supplementary_smallerN_100neg/{valid,test}` |
| books | Week8 external task | `data/processed/week8_same_candidate/books_large10000_100neg/{valid,test}` |
| electronics | Week8 external task | `data/processed/week8_same_candidate/electronics_large10000_100neg/{valid,test}` |
| movies | Week8 external task | `data/processed/week8_same_candidate/movies_large10000_100neg/{valid,test}` |

Do not resample users, negatives, histories, or candidates. Do not edit the
producer project's `candidate_items.csv` or `ranking_valid/test.jsonl`.

For the four-domain same-candidate artifact lane, score exports must use:

```text
source_event_id,user_id,item_id,score
```

and final cross-project import/evaluation must use
`main_import_same_candidate_baseline_scores.py`. TRUCE's internal
`example_id,user_id,item_id,score` importer remains for local adapters and
legacy packets only.

## Pull And Status

```bash
cd ~/projects/TRUCE-Rec
git pull --ff-only
source .venv_truce/bin/activate

python scripts/summarize_controlled_baseline_suite.py
```

## One-Command Week8 Preflight And Conversion

After the parallel data project finishes writing the Week8 task directories,
the preferred server entrypoint is:

```bash
cd ~/projects/TRUCE-Rec
bash scripts/server/run_week8_four_domain_pipeline.sh
```

The script pulls latest code, preflights the expected
`beauty/books/electronics/movies x valid/test` source directories, converts
them without resampling, validates the processed artifacts, prepares Ours
Qwen3-LoRA adapter data, and writes logs under `outputs/server_logs/`.

Default artifact slugs are:

```text
beauty=beauty_supplementary_smallerN_100neg
books=books_large10000_100neg
electronics=electronics_large10000_100neg
movies=movies_large10000_100neg
```

For a dry-run check only:

```bash
python scripts/server/dry_run_week8_four_domain.py \
  --source-root ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  --domains beauty books electronics movies \
  --splits valid test
```

Only set `WEEK8_EXPECTED_USERS=10000` for validation when every selected
artifact, including Beauty, is expected to contain exactly 10,000 users.

## Week8 File Check

```bash
find ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  -path "*large10000_100neg*" -type f | sort
```

Expected per split/domain:

- `ranking_valid.jsonl` or `ranking_test.jsonl`;
- `candidate_items.csv`;
- `train_interactions.csv`;
- `item_metadata.csv`;
- `selected_users.csv`;
- `metadata.json`.

## Convert Week8 Tasks

Preferred: generate commands locally or on the server:

```bash
python scripts/plan_four_domain_server_runs.py \
  --source-root ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  --output-root data/processed/week8_same_candidate \
  --domains beauty books electronics movies \
  --splits valid test \
  --include-ours-adapter-prep
```

Then run the printed commands when the corresponding task directories exist.

Manual example:

```bash
python scripts/convert_week8_same_candidate_to_truce.py \
  --task-dir ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate \
  --output-dir data/processed/week8_same_candidate/books_large10000_100neg/test \
  --domain books \
  --split test
```

Acceptance:

- `preprocess_manifest.json` exists;
- `example_count > 0`;
- each event has target included in candidates;
- `candidate_row_count` equals the sum of all candidate rows;
- event/source IDs are preserved in example metadata.

## Generate External Packets

For the current Beauty packet lane:

```bash
python scripts/prepare_controlled_baseline_suite.py
```

For the full six-family official baseline pool:

```bash
python scripts/prepare_controlled_baseline_suite.py \
  --suite-name qwen3_base_adapter_main6_amazon_beauty \
  --include-added-official-candidates
```

For Week8 domains, add domain-specific packet configs before running official
baselines. Do not reuse Beauty packet configs for books/electronics/movies.

## Run Baselines

Priority order:

1. Cheap traditional/retrieval/sequential baselines.
2. Current controlled-adapter pilots only as pipeline diagnostics.
3. Official-native controlled baselines after fidelity audit.
4. Ours full.
5. Ours ablations.

Every method must end with:

```text
candidate_scores.csv -> predictions.jsonl -> metrics.json + metrics.csv
```

For current Beauty controlled-adapter pilots, use the logging queue:

```bash
cd ~/projects/TRUCE-Rec
bash scripts/server/run_controlled_baseline_queue.sh smoke
bash scripts/server/run_controlled_baseline_queue.sh full
```

By default the full queue runs TALLRec, DEALRec, and LC-Rec. OpenP5 remains out
of the full queue until scoring is optimized. These are still
`controlled_adapter_pilot` unless an official-fidelity audit promotes them.

## Build Week8 Observation Inputs

After Week8 conversion, build Qwen3 observation inputs for each domain/split.
This creates an observation-compatible catalog with train-popularity buckets
and a prompt JSONL accepted by `scripts/server/run_qwen3_observation.py`.

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

for SPEC in \
  beauty:beauty_supplementary_smallerN_100neg \
  books:books_large10000_100neg \
  electronics:electronics_large10000_100neg \
  movies:movies_large10000_100neg
do
  DOMAIN=${SPEC%%:*}
  SLUG=${SPEC#*:}
  for SPLIT in valid test; do
    python scripts/build_week8_observation_inputs.py \
      --processed-dir data/processed/week8_same_candidate/${SLUG}/${SPLIT} \
      --dataset ${SLUG} \
      --domain ${DOMAIN} \
      --split ${SPLIT} \
      --prompt-template forced_json
  done
done
```

For a small server check, add `--max-examples 20`. Do not use small checks as
paper evidence.

## Run Base Qwen3 Observation Sweep

Run base Qwen3-8B observation on the generated inputs. Use `nohup` or keep the
terminal open if `tmux` is unavailable.

```bash
cd ~/projects/TRUCE-Rec
source ~/projects/TALLRec/.venv_tallrec/bin/activate
mkdir -p outputs/logs

for DOMAIN in beauty books electronics movies; do
  case "$DOMAIN" in
    beauty) SLUG=beauty_supplementary_smallerN_100neg ;;
    books) SLUG=books_large10000_100neg ;;
    electronics) SLUG=electronics_large10000_100neg ;;
    movies) SLUG=movies_large10000_100neg ;;
  esac
  INPUT=outputs/observation_inputs/week8_same_candidate/${SLUG}/test_forced_json.jsonl
  OUT=outputs/server_observations/qwen3_8b/week8_same_candidate/${SLUG}/test_forced_json
  nohup python scripts/server/run_qwen3_observation.py \
    --input-jsonl "$INPUT" \
    --output-dir "$OUT" \
    --run-label qwen3_base_${DOMAIN}_week8_test \
    --run-stage observation \
    --execute-server \
    > outputs/logs/qwen3_base_observation_${DOMAIN}.log 2>&1 &
done
```

Then analyze completed runs:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

for DOMAIN in beauty books electronics movies; do
  case "$DOMAIN" in
    beauty) SLUG=beauty_supplementary_smallerN_100neg ;;
    books) SLUG=books_large10000_100neg ;;
    electronics) SLUG=electronics_large10000_100neg ;;
    movies) SLUG=movies_large10000_100neg ;;
  esac
  python scripts/analyze_observation.py \
    --run-dir outputs/server_observations/qwen3_8b/week8_same_candidate/${SLUG}/test_forced_json \
    --input-jsonl outputs/observation_inputs/week8_same_candidate/${SLUG}/test_forced_json.jsonl \
    --source-label qwen3_base_${DOMAIN}_week8_test
done
```

## Observation Gate Across Four Senior Baselines

Observation claims are incomplete until the same phenomenon checks are run for
base Qwen3-8B and at least the four senior-recommended Qwen3-8B-LoRA baselines:
TALLRec, OpenP5, DEALRec, and LC-Rec. Use their formal/pilot ranking outputs
only with the correct evidence label, then convert/import through the shared
observation report path. The required question is:

```text
Does the uncertainty/grounding/popularity/echo phenomenon persist under
stronger baseline systems, or was it only a base-Qwen3 behavior?
```

## Prepare Ours Qwen Adapter Data

After converting a domain, prepare Ours/TRUCE adapter data:

```bash
python scripts/prepare_ours_qwen_adapter_training.py \
  --processed-dir data/processed/week8_same_candidate/books_large10000_100neg/test \
  --output-dir outputs/server_training/ours_qwen_adapters/books_large10000_100neg \
  --domain books \
  --seed 13
```

This writes `train_sft.jsonl`, `valid_sft.jsonl`, `test_score_plan.jsonl`,
`ours_adapter_manifest.json`, and a server command plan. The score schema stays
`example_id,user_id,item_id,score` so Ours can be imported and evaluated by the
same TRUCE evaluator as official baselines.

After server-side Qwen scoring writes `candidate_scores.csv`, import/evaluate:

```bash
python scripts/import_evaluate_ours_adapter.py \
  --manifest outputs/server_training/ours_qwen_adapters/books_large10000_100neg/ours_adapter_manifest.json \
  --split test
```

Ours adapter training data is not a generic prompt baseline. The training rows
encode pairwise acceptance, listwise target-first supervision, train-popularity
buckets, deterministic grounding-risk targets, popularity-bias risk,
history-repetition/echo risk, and contrast roles for head/tail/history-probe
negatives. Ablations must later disable these components rather than comparing
only against a single monolithic Ours score.

The current Ours v2 adapter objective is
`truce_observation_residual_policy_sft_v2`. It adds candidate-normalized
utility, popularity-residual utility, harm risk, abstain risk, and conservative
`promote/suppress/defer_to_fallback` policy actions to train/valid
supervision. Test scoring estimates the likelihood of
`{"policy_action": "promote"}` for each candidate while keeping the score
schema unchanged. Formal paper claims still require four-domain training,
import/evaluation, and ablations for uncertainty, grounding, candidate
normalization, popularity residuals, and echo/history guard.

## Import And Evaluate

For completed controlled baseline scores:

```bash
python scripts/import_evaluate_controlled_baseline.py \
  --manifest outputs/server_training/controlled_baselines/<name>/controlled_baseline_manifest.json
```

Then:

```bash
python scripts/summarize_controlled_baseline_suite.py
```

Paper eligibility requires:

- `implementation_fidelity=official_native_controlled` for official main
  baselines;
- `official_algorithm_reused=true`;
- TRUCE `metrics.json`;
- raw score artifacts and logs retained.

## Observation And Framework Analysis

After predictions are available:

```bash
python scripts/analyze_observation.py --run-dir <observation_or_baseline_run_dir>
```

For CURE/TRUCE feature and reranking analysis, follow:

- `scripts/build_confidence_features.py`;
- `scripts/calibrate_confidence_features.py`;
- `scripts/residualize_confidence_features.py`;
- `scripts/rerank_confidence_features.py`;
- `scripts/triage_confidence_features.py`.

Only use train/valid fit splits for learned calibration/residualization. Same
split fitting is diagnostic only.

## Final Artifact Gate

Before paper table export, each run needs:

- resolved config or manifest;
- environment and git info;
- stdout/stderr logs;
- raw scores or raw LLM responses where applicable;
- `predictions.jsonl`;
- `metrics.json`;
- `metrics.csv`;
- cost/latency or runtime summary;
- official-fidelity audit for official baselines.
