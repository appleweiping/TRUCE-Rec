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
- Current stage: **Method locked (CALM-Rec) + 8-domain performance buildout.** The eight official
  baselines are complete and frozen in `data/official_baselines/` (8 domains × 8 baselines, 64
  pairs). TRUCE-Rec's own method — after the prior uncertainty-gate route lost to fallback in R3 —
  is **CALM-Rec** (Calibrated trust over Attribute-anchored Latent Multi-intent), locked via a
  ≥20-iteration tri-agent upgrade (ARIS 9.0/10); scoring core implemented + tested. No paper-result
  claim is allowed until CALM-Rec runs, ablations, and audits are evaluated under the same protocol.

> **Read first for setting/metrics/baselines:**
> [`docs/experimental_setting_and_baselines.md`](docs/experimental_setting_and_baselines.md).

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
- Reference papers and official projects may be read carefully for
  reproduction fidelity and inspiration, but the TRUCE/CURE method must not be
  a stitched, copied, or renamed version of those systems.

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
- Pony/Uncertainty official-qwen3base same-candidate baseline evidence is now
  the paper-facing external baseline source. TRUCE imports/copies the evidence
  packages and tracks eligibility in
  `configs/baselines/pony_official_external_baselines.yaml`.
- The old TRUCE-side controlled-adapter server suite for TALLRec/OpenP5-style/
  DEALRec/LC-Rec remains legacy pilot infrastructure, not the current main
  baseline route.
- The current project route is now organized as:
  `observation -> CURE/TRUCE framework -> official baselines -> four-domain
  same-candidate recommendation system`.
- Ours/TRUCE Qwen adapter preparation and import/evaluation scaffolds exist for
  server-side training: `scripts/prepare_ours_qwen_adapter_training.py` and
  `scripts/import_evaluate_ours_adapter.py`.
- Ours adapter supervision has been upgraded to
  `truce_observation_residual_policy_sft_v2`: train/valid rows include
  candidate-normalized utility, popularity-residual utility, harm/abstain risk,
  and conservative promote/suppress/fallback policy targets; test scoring keeps
  the same `candidate_scores.csv` schema by scoring promote-action likelihood.

Not yet completed:

- no approved paper-result experiment suite;
- no claim that OursMethod is effective;
- no HF model download;
- no completed Gate R1 TRUCE Ours/ablation paper-scale run;
- no final TRUCE Ours/ablation table under the reused Pony candidate protocol;
- no completed TRUCE-side observation sweep over Ours plus reused strong
  baselines;
- no final paper conclusions.
- no completed four-domain observation sweep that checks base Qwen3-8B and the
  four senior-recommended Qwen3-8B-LoRA baselines side by side;
- no final learned TRUCE policy/adapter ablation suite proving that Ours is
  deeper than heuristic prompting or conservative rules.

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

Complex tasks should use multi-agent collaboration by default when available:
implementation/exploration plus a reviewer/fairness pass. Each completed task
should end with the next concrete plan and a stage verdict: still open, blocked
by specific gates, or ready to move toward paper writing after top-conference
review.

## Official Baseline Contract

The paper-facing external comparison is the **eight official LLM4Rec baselines**, evaluated under
TRUCE-Rec's shared same-candidate protocol on the **Qwen3-8B** backbone. The complete, frozen,
lightweight evidence lives in [`data/official_baselines/`](data/official_baselines/): 64 pairs
(8 domains × 8 baselines), master table `baseline_comparison_8domains.csv`, integrity manifest
`IMPORT_MANIFEST.json`, and per-domain frontier `frontier_best_baseline_per_domain.csv`.

```text
official baseline run (official code @ pinned commit)
  + 8-domain same-candidate task (101 candidates: 1 pos + 100 neg)
  + Qwen3-8B backbone (LoRA where the method fine-tunes)
  + official/default (or reported-optimal) baseline hyper-parameters
  + declared, audited adaptation to the shared candidate schema
  + score schema: source_event_id,user_id,item_id,score
```

These baseline numbers are **frozen and reused as a shared reference, never re-run** (same baseline
+ same data + same protocol = same scores). TRUCE-Rec's own method is implemented, trained, and
evaluated **fully independently** and is the only thing measured against this set. Ours may tune
hyper-parameters only through the declared validation protocol; never on test.

The eight official baselines:

| key | method | family |
| --- | --- | --- |
| `elmrec`  | ELMRec  | graph-enhanced LLM4Rec |
| `irllrec` | IRLLRec | intent-aware LLM4Rec |
| `llm2rec` | LLM2Rec | sequential (SASRec-style) LLM4Rec |
| `llmemb`  | LLMEmb  | embedding-alignment LLM4Rec |
| `llmesr`  | LLM-ESR | LLM-enhanced sequential rec |
| `proex`   | ProEx   | profile / explanation LLM4Rec |
| `promax`  | ProMax  | profile-based LLM4Rec |
| `rlmrec`  | RLMRec  | graph-contrastive representation LLM4Rec |

Domains: beauty (973 users, supplementary smaller-N), books / electronics / movies / sports / toys
/ home / tools (10,000 users each).

See `data/official_baselines/README.md` and `docs/experimental_setting_and_baselines.md`.

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

Import Pony/Uncertainty official baseline evidence:

```powershell
py -3 scripts\import_pony_official_baselines.py `
  --pony-root D:\Research\Uncertainty `
  --output-root outputs\pony_official_baselines `
  --manifest configs\baselines\pony_official_external_baselines.yaml
```

Build the Pony baseline comparison/status tables:

```powershell
py -3 scripts\build_pony_baseline_comparison.py `
  --manifest-json outputs\pony_official_baselines\manifest.json `
  --output-root outputs\pony_official_baselines\tables `
  --output-name pony_official_baseline_comparison
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

- `docs/experimental_setting_and_baselines.md`: **authoritative** setting, metrics, the 8 official
  baselines, frozen evidence location, and the per-domain SOTA bar. Read this first.
- `docs/method_calm_rec_spec.md`: **the live Ours method — CALM-Rec** (Calibrated trust over
  Attribute-anchored Latent Multi-intent), locked via a ≥20-iteration tri-agent upgrade (ARIS 9.0/10).
- `docs/method_redesign_decision.md`: SCALR (superseded by CALM-Rec) — design history + the shared
  falsifiability/leakage discipline CALM-Rec inherits.
- `docs/followup_experiment_plan.md`: the three required follow-up experiments (observation,
  ablation, hyper-parameter analysis) + overview figure, scheduled after the performance table.
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
- `docs/pony_official_baseline_reuse.md`: current paper-facing reused Pony
  official baseline policy and commands.
- `docs/ours_method_plan.md`: Phase 6 method plan and boundaries.
- `docs/ablation_protocol.md`: OursMethod ablation protocol.
- `docs/leakage_fairness_checklist.md`: leakage/fairness safeguards.
- `docs/server_runbook.md`: API/HF/server/LoRA safety runbook.
- `docs/submission_roadmap.md`: milestone roadmap from observation to
  four-domain submission system.
- `docs/server_execution_matrix.md`: server-first command and artifact matrix.
- `docs/top_conference_review_plan.md`: internal reviewer/literature-agent
  checklist before paper writing.
- `docs/server_execution_matrix.md`: includes the base/baseline observation
  gate and formal Ours/baseline server command ladder.
- `docs/qwen3_lora_controlled_baselines.md`: legacy controlled-adapter pilot
  protocol.
- `docs/controlled_baseline_fidelity_audit.md`: legacy fidelity checklist.
- `docs/server_next_commands.md`: current server continuation commands.
- `docs/external_project_baseline_packets.md`: external project packet matrix.
- `docs/week8_large_same_candidate_protocol.md`: larger same-candidate
  books/electronics/movies protocol.

Current four-domain same-candidate artifact slugs are
`beauty_supplementary_smallerN_100neg`, `books_large10000_100neg`,
`electronics_large10000_100neg`, and `movies_large10000_100neg`. This artifact
lane is not model weights or a paper result by itself; do not edit its
`candidate_items.csv` or `ranking_valid/test.jsonl`, and export cross-project
scores as `source_event_id,user_id,item_id,score`.

## Package Layout

The active implementation package is `src/llm4rec/`. Older `storyflow`
references in historical local notes are not the active Phase 6/7 package
contract.

## Real Experiment Rule

Before any real pilot, fill a `configs/experiments/real_*_template.yaml`,
validate it with `scripts/validate_experiment_ready.py`, confirm datasets and
resources, and keep the safety flags blocking API calls, downloads, and
training until the user explicitly approves the run.
