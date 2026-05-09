# TRUCE-Rec

This repository is TRUCE-Rec unless the user states otherwise. It is a
research-grade LLM4Rec codebase for uncertainty-aware generative
recommendation, with a shared data, baseline, OursMethod, evaluation, export,
and reproducibility scaffold.

Current repository identity:

- GitHub: `https://github.com/appleweiping/TRUCE-Rec.git`
- Historical remote alias in this checkout: `https://github.com/appleweiping/uncertainty-llm4rec.git`
- Local path: `D:\Research\TRUCE-Rec`
- Active branch: `main`
- Current stage: Gate R1 server-first four-domain controlled experiment
  buildout. No paper-result claim is allowed until full official baseline/Ours
  runs are imported and evaluated by TRUCE.

## Evidence Labels

Use these labels consistently:

- Smoke/mock: fixture-data or MockLLM runs that verify code paths only.
- Pilot: small approved real-data run used to debug the protocol.
- Diagnostic: prompt, grounding, candidate, or artifact QA; not a paper
  conclusion.
- Controlled adapter pilot: TRUCE-side implementation that uses the shared
  protocol and Qwen3-8B base model but has not yet passed official baseline
  fidelity audit.
- Official-native controlled baseline: official project algorithm with shared
  TRUCE protocol, Qwen3-8B base-model substitution, LoRA adaptation, and the
  baseline's official/default or reported-optimal hyperparameters where
  feasible.
- Paper result: approved real experiment with tracked code, saved config, logs,
  raw outputs where applicable, predictions, metrics, and artifact checklist.

Strict rules:

- Smoke outputs are not paper evidence.
- MockLLM outputs are not paper evidence.
- Pilot/API diagnostic outputs are not paper conclusions.
- Controlled adapter pilots are not final official baseline results.
- Ignored local diagnostics under `outputs/` or `data/processed/` are not
  paper evidence unless explicitly promoted by a later approved protocol.
- Formal paper results must come from approved real experiment configs, tracked
  code, saved configs, logs, raw outputs, predictions, and metrics.

## Current Status

Implemented and smoke-tested:

- dataset preprocessing and tiny fixture data;
- random, popularity, BM25, MF, sequential Markov, and LLM mock baselines;
- MockLLM, OpenAI-compatible provider interface, HF provider scaffold, and
  LoRA dry-run scaffold;
- unified prediction schema and shared evaluator;
- ranking, validity, confidence, calibration, coverage, diversity, novelty,
  long-tail, efficiency, slicing, aggregation, and table export support;
- Phase 6 OursMethod:
  `Calibrated Uncertainty-Guided Generative Recommendation`;
- Phase 7 paper support, real experiment templates, reproduction docs, and
  safe preflight helpers.
- Main4 controlled-adapter server suite is prepared for TALLRec, OpenP5-style,
  DEALRec, and LC-Rec. These runs validate the protocol, but they remain
  `controlled_adapter_pilot` until official-native fidelity is audited.
- External baseline packets exist for TALLRec, OpenP5, BIGRec/DEALRec,
  LC-Rec, LLaRA, CoLLM, LLM-ESR, and SLMRec.
- The current project route is now organized as:
  `observation -> CURE/TRUCE framework -> official baselines -> four-domain
  same-candidate recommendation system`.
- Ours/TRUCE Qwen adapter preparation and import/evaluation scaffolds exist for
  server-side training: `scripts/prepare_ours_qwen_adapter_training.py` and
  `scripts/import_evaluate_ours_adapter.py`.

Not yet completed:

- no approved paper-result experiment suite;
- no claim that OursMethod is effective;
- no HF model download;
- no completed Gate R1 four-domain paper-scale run;
- no final official-native controlled baseline table;
- no completed official-native fidelity audit for the external baselines;
- no final paper conclusions.

## Method Lineage

The active research line is:

```text
RESEARCH_IDEA
  -> title-generation observation and grounding
  -> CU-GR / CURE-TRUCE uncertainty features
  -> CU-GR v2 candidate-normalized preference fusion
  -> full TRUCE recommendation system with official baselines
```

The active package is `src/llm4rec/`. Historical `src/storyflow/` references
are legacy scaffolding and should be treated as background unless a current
document explicitly maps them to `llm4rec` modules.

## Durable Memory

Future agents must read `docs/PROJECT_MEMORY.md` before nontrivial work. It
records the current big direction, senior baseline advice, server workflow,
multi-agent expectation, evidence labels, four-domain plan, and update
discipline. If a task changes the roadmap, baseline policy, server commands,
or current status, update `docs/PROJECT_MEMORY.md` in the same commit.

## Official Baseline Contract

The paper-grade external baseline lane is:

```text
official project algorithm
  + shared TRUCE split/candidates/evaluator
  + shared Qwen3-8B base model
  + LoRA adaptation under the baseline's official training logic
  + official default or reported-optimal baseline hyperparameters
  + candidate_scores.csv -> predictions.jsonl -> metrics.json
```

This is the main academic comparison protocol recommended for the project:
every compared LLM baseline uses Qwen3-8B and LoRA, while official source code,
project modules, prompts/objectives, and default or reported-optimal
hyperparameters are retained as much as possible. Baseline hyperparameters are
not tuned on TRUCE test outcomes. Compatibility changes, official commits, and
hyperparameter sources must be recorded in provenance. Ours may tune
hyperparameters only through the declared validation protocol.

Runs that keep an original non-Qwen backbone, use full fine-tuning instead of
LoRA, or rely on an official checkpoint belong in a separate reference/appendix
protocol unless the table explicitly separates that comparison. All methods in
the main lane must export the same score schema:

```text
example_id,user_id,item_id,score
```

The current recommended official baseline families are:

| Role | Family | Current repository status |
| --- | --- | --- |
| Main | TALLRec | controlled adapter pilot; official-native audit required |
| Main | OpenP5 | controlled adapter pilot; scoring optimization and official-native audit required |
| Main | DEALRec | controlled adapter pilot; official-native audit required |
| Main | LC-Rec | controlled adapter pilot; official-native audit required |
| Main | LLaRA | packet/config added as Qwen3-LoRA; official-native implementation required |
| Main | LLM-ESR | packet/config added as Qwen3-LoRA; official-native implementation required |

CoLLM and SLMRec remain useful follow-up candidates, especially for
collaborative-signal and efficiency appendices.

## Key Commands

Run smoke baselines:

```powershell
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_all_baselines.yaml
```

Run Phase 6 smoke suite:

```powershell
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_phase6_all.yaml
```

Run all tests:

```powershell
.\.venv\bin\python.exe -m pytest
```

Prepare the server controlled-adapter suite:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate
python scripts/prepare_controlled_baseline_suite.py
```

Summarize controlled baseline status:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate
python scripts/summarize_controlled_baseline_suite.py
```

Validate a real experiment template without running it:

```powershell
.\.venv\bin\python.exe scripts\validate_experiment_ready.py --config configs/experiments/real_ours_method_template.yaml
```

List required artifacts for a planned run:

```powershell
.\.venv\bin\python.exe scripts\list_required_artifacts.py --config configs/experiments/real_ours_method_template.yaml
```

## Important Docs

- `AGENTS.md`: engineering and research governance rules.
- `docs/PROJECT_MEMORY.md`: durable future-agent memory and current project
  direction.
- `docs/RESEARCH_IDEA.md`: core research direction.
- `docs/experiment_protocol.md`: split, candidate, prompt, LLM, metric, and
  leakage protocol.
- `docs/real_experiment_matrix.md`: planned real experiment groups.
- `docs/reproduction.md`: local reproduction commands.
- `docs/pre_experiment_checklist.md`: real-run readiness checklist.
- `docs/result_artifact_checklist.md`: artifact contract.
- `docs/baselines.md`: baseline readiness and limitations.
- `docs/ours_method_plan.md`: Phase 6 method plan and boundaries.
- `docs/ablation_protocol.md`: OursMethod ablation protocol.
- `docs/leakage_fairness_checklist.md`: leakage/fairness safeguards.
- `docs/server_runbook.md`: API/HF/server/LoRA safety runbook.
- `docs/submission_roadmap.md`: milestone roadmap from observation to
  four-domain submission system.
- `docs/server_execution_matrix.md`: server-first command and artifact matrix.
- `docs/top_conference_review_plan.md`: internal reviewer/literature-agent
  checklist before paper writing.
- `docs/qwen3_lora_controlled_baselines.md`: controlled external baseline
  protocol and status.
- `docs/controlled_baseline_fidelity_audit.md`: official-native fidelity rule
  and promotion checklist.
- `docs/server_next_commands.md`: current server continuation commands.
- `docs/external_project_baseline_packets.md`: external project packet matrix.
- `docs/week8_large_same_candidate_protocol.md`: larger same-candidate
  books/electronics/movies protocol.

## Package Layout

The active implementation package is `src/llm4rec/`. Older `storyflow`
references in historical local notes are not the active Phase 6/7 package
contract.

## Real Experiment Rule

Before any real pilot, fill a `configs/experiments/real_*_template.yaml`,
validate it with `scripts/validate_experiment_ready.py`, confirm datasets and
resources, and keep the safety flags blocking API calls, downloads, and
training until the user explicitly approves the run.
