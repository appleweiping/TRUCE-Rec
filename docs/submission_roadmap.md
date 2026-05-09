# Submission Roadmap

This roadmap keeps the project centered on a publishable TRUCE-Rec system. It
is a planning and execution contract, not a result table.

## North Star

TRUCE-Rec should become a complete recommendation framework that starts from a
recommendation-specific observation and ends with a reproducible four-domain
experimental system:

```text
LLM generative recommendation observation
  -> grounded uncertainty and popularity/long-tail diagnostics
  -> original CURE/TRUCE framework
  -> official-native controlled baselines
  -> unified same-candidate evaluator
  -> four-domain paper-scale results
```

The core original contribution remains uncertainty-aware generative
recommendation: generated title grounding, uncertainty reliability, popularity
confounding, long-tail under-confidence, echo/history inertia risk, and
exposure-aware routing/reranking.

## Milestones

### M0. Governance And Evidence Boundaries

Status: mostly complete.

Keep:

- `AGENTS.md`, `docs/RESEARCH_IDEA.md`, and `docs/experiment_protocol.md` as
  the source of truth for non-toy, non-fabricated research practice.
- `docs/PROJECT_MEMORY.md` as the durable future-agent memory for big
  direction, senior baseline advice, server workflow, update discipline, and
  current next moves.
- Evidence labels: smoke/mock, diagnostic, controlled adapter pilot,
  official-native controlled baseline, paper result.
- One shared prediction/evaluation schema for every method.

Exit criteria:

- README and phase handoff point to the current roadmap.
- README, AGENTS, and phase handoff point to `docs/PROJECT_MEMORY.md`.
- Any generated result is machine-labeled with its evidence scope.

### M1. Observation Layer

Status: implemented as schema/scaffold plus prior pilots; needs four-domain
reruns.

Purpose:

- Show the phenomenon, not just method performance.
- Quantify confidence vs correctness, grounding, hallucination, popularity,
  long-tail, history similarity, diversity, and candidate adherence.

Required outputs:

- grounded observation rows;
- reliability and selective-risk analysis;
- head/mid/tail slices;
- wrong-high-confidence and correct-low-confidence cases;
- candidate adherence and grounding diagnostics.

Exit criteria:

- Observation analysis runs on Beauty plus Week8 books/electronics/movies.
- Claims are limited to completed runs and paired source artifacts.

### M2. Original CURE/TRUCE Framework

Status: scaffold exists; TRUCE-native Qwen adapter data preparation now exists;
paper-grade training/inference still needs server execution.

Required components:

- generated title and catalog grounding;
- multiple uncertainty signals where available;
- popularity residual/deconfounded confidence;
- echo/history-inertia risk;
- exposure-aware candidate routing/reranking;
- TRUCE-native Qwen adapter data that combines pairwise acceptance and listwise
  target-first supervision while preserving uncertainty/risk metadata;
- ablation switches for each component.

Current implementation anchors:

- `src/llm4rec/methods/ours_framework.py`: builds Ours pairwise/listwise
  training and scoring prompts with grounding, popularity bucket, and
  history-risk evidence.
- `scripts/prepare_ours_qwen_adapter_training.py`: writes
  `train_sft.jsonl`, `valid_sft.jsonl`, `test_score_plan.jsonl`, and
  `ours_adapter_manifest.json`.
- `scripts/import_evaluate_ours_adapter.py`: imports Ours `candidate_scores.csv`
  through the same TRUCE evaluator path as official baselines.

Required ablations:

- Ours full;
- no uncertainty policy;
- no grounding;
- no candidate-normalized confidence;
- no popularity residual/adjustment;
- no echo/history guard;
- fallback-only;
- LLM generative only;
- LLM rerank only.

Exit criteria:

- Ours emits the same prediction schema as every baseline.
- Every component has an ablation and a matching observation motivation.
- No policy decision uses target correctness.
- Ours adapter scores import through `candidate_scores.csv` with event/source
  IDs preserved for paired comparison.

### M3. Baseline System

Status: traditional/text/sequential interfaces exist; external official-native
baseline lane is now explicit but not complete.

Baseline tiers:

- Traditional and retrieval: Random, Popularity, BM25, MF/BPR where feasible.
- Sequential/graph: SASRec, BERT4Rec, GRU4Rec, LightGCN or RecBole-compatible
  wrappers.
- LLM task baselines: zero-shot/few-shot generative, candidate rerank,
  constrained candidate rerank.
- Official LLM4Rec families:
  - TALLRec;
  - OpenP5;
  - DEALRec;
  - LC-Rec;
  - LLaRA;
  - LLM-ESR.

All six official LLM4Rec families are part of the main baseline pool. They can
have different analysis slices, such as long-tail/sequential slices for
LLM-ESR, while keeping equal baseline status.

Official baseline contract:

```text
official project algorithm
  + shared TRUCE split/candidates/evaluator
  + shared Qwen3-8B base model
  + LoRA adaptation under baseline-specific official training logic
  + official default or reported-optimal baseline hyperparameters
  + candidate_scores.csv -> predictions.jsonl -> metrics.json
```

Exit criteria:

- Each official baseline has an official-fidelity audit.
- Each official baseline records official commit, official hyperparameter
  source, and any Qwen3-8B-LoRA compatibility changes.
- Controlled adapter pilots are not mixed into final official baseline tables.
- All baselines preserve the score schema:
  `example_id,user_id,item_id,score`.

### M4. Data Ladder

Status: Beauty early pipeline exists; Week8 converter exists for large
same-candidate tasks.

Dataset ladder:

- tiny fixtures: tests only;
- MovieLens: sanity and low-cost pilot;
- Amazon Beauty: first full-domain pipeline and debugging domain;
- Week8 four-domain benchmark:
  - `beauty`;
  - `books`;
  - `electronics`;
  - `movies`.

Week8 target protocol:

- up to 10,000 users per domain;
- 1 positive + 100 popularity-sampled negatives per event;
- same-candidate setting for all methods;
- test history mode: `train_plus_valid`;
- preserve `event_id`, `source_event_id`, `user_id`, `item_id`, and `split`.

Exit criteria:

- No user/negative/candidate resampling in TRUCE.
- Every method scores the same candidate rows.
- Outputs support paired tests and rank-fusion/oracle analysis.

### M5. Four-Domain Experiment Suite

Status: planned.

For each domain:

1. Convert Week8 task directories into TRUCE processed artifacts.
2. Generate project packets.
3. Run traditional/retrieval/sequential baselines.
4. Run official-native controlled LLM4Rec baselines.
5. Run Ours full and ablations.
6. Import all scores through TRUCE.
7. Evaluate ranking, validity, coverage, diversity, novelty, long-tail,
   efficiency, and paired significance.
8. Run observation diagnostics on Ours and strong baselines.

Exit criteria:

- Each domain has complete `predictions.jsonl`, `metrics.json`, `metrics.csv`,
  manifests, environment, git info, logs, and raw score artifacts.
- Missing runs are explicitly listed; no empty table cells are silently filled.

### M6. Top-Conference Defense

Status: planned and partially documented.

Reviewer-proof requirements:

- demonstrate originality beyond generic LLM reranking or RAG;
- use official-native baselines, not weak prompt lookalikes;
- report multiple domains and long-tail/coverage/validity metrics;
- include ablations tied to the observation findings;
- show paired statistical tests;
- include cost/latency/throughput;
- preserve failure cases and limitations.

Exit criteria:

- A reviewer checklist exists before paper writing.
- Paper tables are generated only from completed metrics artifacts.
- Limitations include grounding ambiguity, implicit-feedback incompleteness,
  candidate-set scope, and compute/API budget constraints.

## Server-First Execution Order

1. Finish any running adapter-pilot process and import/evaluate it only as
   `controlled_adapter_pilot`.
2. Pull latest TRUCE-Rec on the server.
3. Verify Week8 task availability for beauty/books/electronics/movies.
4. Convert Week8 valid/test splits into TRUCE processed artifacts.
5. Generate packets for all selected official baselines.
6. Run cheap traditional/retrieval/sequential baselines first.
7. Upgrade official-native LLM4Rec baselines one at a time, starting with the
   easiest official code path.
8. Run Ours and ablations.
9. Run import/evaluation/analysis/export.
10. Freeze artifacts for paper tables.

## Non-Negotiable Stop Conditions

- If a candidate set changes, stop and regenerate all affected comparable
  methods.
- If an official baseline falls back to a generic TRUCE prompt adapter, label
  it as a pilot.
- If a run lacks `predictions.jsonl` and TRUCE `metrics.json`, do not use it in
  a result table.
- If a method uses a different backbone or official checkpoint, move it to
  reference/appendix unless the table explicitly separates protocols.
- If a completed stage changes direction, commands, baseline policy, or status
  but `docs/PROJECT_MEMORY.md` is not updated, stop and update the memory before
  handing off.
