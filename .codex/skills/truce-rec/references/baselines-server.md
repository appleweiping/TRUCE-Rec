# Baselines And Server Workflow

Use this reference for official baselines, same-candidate artifacts, server
handoffs, and import/evaluation gates.

## Main Baseline Contract

Main LLM baseline comparisons should use:

```text
official project implementation where claimed
  + Qwen3-8B base model
  + LoRA adaptation
  + official/default or reported-optimal hyperparameters
  + same TRUCE splits and fixed candidate rows
  + same TRUCE evaluator
  + source_event_id/user_id/item_id/score preservation
```

Ours may tune only through the declared validation protocol. Baselines must not
be tuned on TRUCE test outcomes.

Do not silently mix full fine-tuning, QLoRA-only runs, original project
checkpoints, different backbones, or different adapter budgets into the main
Qwen3-8B-LoRA table. If such runs are useful, separate and label them as a
reference or appendix protocol.

## Baseline Families

Main official LLM4Rec families:

- TALLRec
- OpenP5
- DEALRec
- LC-Rec
- LLaRA
- LLM-ESR

TRUCE-side Qwen3-LoRA adapter pilots are useful diagnostics, but they are not
official-native baselines unless an official-fidelity audit promotes them.

## Same-Candidate Data Lane

Target four-domain protocol:

- `beauty_supplementary_smallerN_100neg`
- `books_large10000_100neg`
- `electronics_large10000_100neg`
- `movies_large10000_100neg`

Do not resample users, negatives, histories, or candidates inside TRUCE. Do not
edit producer-side `candidate_items.csv` or `ranking_valid/test.jsonl`.

Cross-project score exports must use:

```text
source_event_id,user_id,item_id,score
```

Local adapter imports may use `example_id,user_id,item_id,score` only where the
existing TRUCE importer expects it.

## Server Operating Model

Local Codex usually cannot see server state. Use command-and-log discipline:

1. Update local repo and push if needed.
2. Give exact server commands.
3. Wait for the user to paste logs, status, errors, or artifact listings.
4. Diagnose and patch based on evidence.
5. Do not claim completion without artifacts.

Preferred Week8 entrypoint:

```bash
cd ~/projects/TRUCE-Rec
git pull --ff-only
bash scripts/server/run_week8_four_domain_pipeline.sh
```

Current Beauty controlled-adapter pilot queue:

```bash
cd ~/projects/TRUCE-Rec
bash scripts/server/run_controlled_baseline_queue.sh smoke
bash scripts/server/run_controlled_baseline_queue.sh full
```

Do not full-run slow OpenP5 unless docs or the user indicate scoring has been
optimized.

## Artifact Gate

Each paper-eligible run needs:

- resolved config or manifest;
- environment and git info;
- stdout/stderr logs;
- raw scores or raw LLM responses when applicable;
- `predictions.jsonl`;
- `metrics.json`;
- `metrics.csv`;
- cost/latency or runtime summary;
- official-fidelity audit for official baselines.

Every comparable method must flow through:

```text
candidate_scores.csv -> predictions.jsonl -> metrics.json + metrics.csv
```
