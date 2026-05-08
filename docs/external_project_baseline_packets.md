# External Project Baseline Packets

This document tracks official external LLM4Rec project integrations. It is a
packet and execution-status matrix, not a paper result table.

For the main fair-comparison lane that controls the small LLM base model and
TRUCE protocol, see `docs/qwen3_lora_controlled_baselines.md`. The main lane
uses Qwen3-8B plus LoRA for every compared LLM baseline, official source/default
or reported-optimal hyperparameters where feasible, and the shared TRUCE score
schema. The first adapter-pilot suite is TALLRec, OpenP5-style, DEALRec, and
LC-Rec with Qwen3-8B-LoRA.

## Status Matrix

| Project | Family | Official repo | Packet configs | Qwen3-LoRA compliance | Official implementation reused | Hyperparameter source | TRUCE score schema | Paper-table eligibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenP5 | Generative LLM recommender | <https://github.com/agiresearch/OpenP5> | MovieLens, Amazon Beauty | pilot only | pending audit | pending official audit | ready | Main table only after official-native Qwen3-LoRA run and TRUCE import |
| TALLRec | Instruction tuning for recommendation | <https://github.com/SAI990323/TALLRec> | MovieLens, Amazon Beauty | pilot only | pending audit | pending official audit | ready | Main table only after official-native Qwen3-LoRA instruction tuning and TRUCE import |
| BIGRec | Data-efficient LLM4Rec | <https://github.com/Linxyhaha/DEALRec> | MovieLens, Amazon Beauty | not selected for main six | pending audit | pending official audit | ready | Follow-up/appendix after controlled run |
| DEALRec | Data-efficient LLM4Rec | <https://github.com/Linxyhaha/DEALRec> | MovieLens, Amazon Beauty | pilot only | pending audit | pending official audit | ready | Main table only after official-native Qwen3-LoRA run and TRUCE import |
| LC-Rec | LLM plus collaborative signal | <https://github.com/RUCAIBox/LC-Rec/> | MovieLens, Amazon Beauty | pilot only | pending audit | pending official audit | ready | Main table only after official-native Qwen3-LoRA collaborative run and TRUCE import |
| LLaRA | LLM plus recommendation-signal alignment | <https://github.com/ljy0ustc/LLaRA> | MovieLens, Amazon Beauty | config added | pending audit | pending official audit | ready | Main table only after official-native Qwen3-LoRA run and TRUCE import |
| CoLLM | LLM plus collaborative signal | <https://github.com/zyang1580/CoLLM> | MovieLens, Amazon Beauty | follow-up | pending audit | pending official audit | ready | Follow-up/appendix after controlled run |
| LLM-ESR | Long-tail/sequential LLM4Rec | <https://github.com/Applied-Machine-Learning-Lab/LLM-ESR> | MovieLens, Amazon Beauty | config added | pending audit | pending official audit | ready | Main/robustness table after official-native Qwen3-LoRA run and TRUCE import |
| SLMRec | Small sequential LLM4Rec | <https://github.com/WujiangXu/SLMRec> | MovieLens, Amazon Beauty | follow-up | pending audit | pending official audit | ready | Sequential/efficiency appendix after run |

The added official baseline candidates are LLaRA and LLM-ESR. LLaRA is chosen
because the SIGIR 2024 project/paper page points to the official
`ljy0ustc/LLaRA` repository and its recommendation-signal alignment is
methodologically distinct from TALLRec/OpenP5/DEALRec/LC-Rec. LLM-ESR is
chosen because the Applied-Machine-Learning-Lab repository and NeurIPS 2024
page identify it as a long-tail sequential recommendation method, making it a
strong robustness baseline rather than another generic reranker.

## Packet Generation

OpenP5 should be run first. Generate its packets with:

```powershell
py -3 scripts/prepare_project_baseline_packet.py --config configs/server/project_baselines/openp5_movielens_packet.yaml
py -3 scripts/prepare_project_baseline_packet.py --config configs/server/project_baselines/openp5_amazon_beauty_packet.yaml
```

Generate all packet contracts when preparing the server queue:

```powershell
Get-ChildItem configs/server/project_baselines/*_packet.yaml |
  ForEach-Object { py -3 scripts/prepare_project_baseline_packet.py --config $_.FullName }
```

Each packet contains canonical TRUCE artifacts plus project-facing files:

- `truce_examples.jsonl`
- `truce_candidate_sets.jsonl`
- `item_catalog.jsonl`
- `<project>/*_project_tasks.jsonl` for generic projects
- `openp5/*_sequential_tasks.jsonl` for OpenP5
- `tallrec/*.json` and row maps for TALLRec
- `candidate_scores_template.csv`
- `project_baseline_manifest.json`

External projects must return:

```text
example_id,user_id,item_id,score
```

Then import and evaluate with:

```powershell
py -3 scripts/import_external_predictions.py `
  --scores <candidate_scores.csv> `
  --examples <packet>/truce_examples.jsonl `
  --output <run_dir>/predictions.jsonl `
  --method <method> `
  --source-project <project> `
  --model-name <model> `
  --seed 13 `
  --split test

py -3 scripts/evaluate_predictions.py --predictions <run_dir>/predictions.jsonl --output-dir <run_dir>
```

## OpenP5 Priority

OpenP5 is the first external project to execute because it is a recognized P5
family baseline and already has a project-specific sequential-task packet. The
server adapter smoke has already adapted `openp5/*_sequential_tasks.jsonl` into
a T5-style bridge template and proved that a candidate score CSV can be imported
and evaluated by TRUCE.

OpenP5 outputs may be item tokens, generated item strings, or scores. Any
generated output must be grounded back to canonical TRUCE `item_id`; invalid or
ungrounded outputs are not paper-successes and must remain auditable.

The next OpenP5 server task is to replace the deterministic smoke scores with a
real OpenP5/T5 scorer or official training/evaluation run, then import with
`--split test`.

For the main controlled comparison, prefer `OpenP5-style-Qwen3-LoRA` over an
uncontrolled OpenP5 T5 checkpoint unless both are clearly separated in tables.

## Generic Project Contract

BIGRec, DEALRec, LC-Rec, LLaRA, CoLLM, LLM-ESR, and SLMRec currently share the
generic packet contract:

- train/valid/test task JSONL with history, target, candidates, and item text;
- item mapping CSV;
- test candidate map;
- candidate score template.

This is enough to start real upstream integration without changing TRUCE splits,
negative sampling, or evaluator definitions. A project becomes an official
baseline only after its upstream code is cloned, environment is recorded, a real
run is completed, and its scores are imported/evaluated by TRUCE.

## Diagnostic Rows

The Amazon Beauty TALLRec-style Qwen3 zero-shot scorer is recorded only in
`outputs/tables/paper_project_baseline_diagnostics.*`. It is not an official
trained TALLRec result and must not appear in the main strong-baseline table.

The OpenP5 Amazon Beauty adapter smoke is an even weaker diagnostic: it is only
a bridge/import/evaluator smoke test and should stay out of both main and
appendix result tables unless a future diagnostic table explicitly lists
plumbing checks.
