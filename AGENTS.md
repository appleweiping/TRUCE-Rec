# AGENTS.md

This repository is a research-grade LLM4Rec codebase. Codex must treat it as a publishable research system, not a toy demo.

## 0. Non-Negotiable Rules

- Do not implement toy demos, mocks, pseudo-code, or notebook-only experiments unless explicitly requested.
- Do not fabricate experimental results, tables, logs, metrics, or paper claims.
- Do not write paper conclusions before actual experiments are run.
- Do not silently skip failed tests, missing dependencies, missing data, or broken commands.
- Do not make different baselines incomparable through different splits, metrics, negative sampling, candidate construction, or evaluation code.
- Do not hard-code dataset paths, model names, seeds, metrics, or prompt templates inside source files.
- Do not design only for one dataset or one model.
- Do not implement only our method; strong baselines must be supported with the same evaluator.
- Do not overwrite existing user work without first identifying the files and explaining the intended change.
- Do not perform large refactors before producing an implementation plan.
- When consulting senior-recommended papers/projects or other top-conference
  implementations, study them carefully as inspiration and fidelity guidance
  only. The TRUCE/CURE method must not become a stitched-together clone,
  copied objective, copied pipeline, or renamed combination of existing
  baselines.
- **实时更新硬规则**：每完成一个阶段、一个 step、一次错误排除、一次贡献，
  必须立即更新共享 memory（`D:\research\Vipin's Knowledgebase\memory\`）和
  项目文档（PROJECT_MEMORY.md, PHASE_HANDOFF.md 等）。不攒着，不跳过。
  违反此规则等于工作没做。
- **项目独立性**：TRUCE-Rec 与 Pony/TGL-Rec 共享 8 个外部 baseline 和数据
  setting，但方法/framework 完全独立。Baseline 分数可复用，方法代码绝不混用。

## 1. Research Goal

The repository must support a submission-level LLM4Rec experimental pipeline.

The final system should support:

- multi-domain recommendation datasets;
- traditional recommendation baselines;
- sequential recommendation baselines;
- text retrieval baselines;
- LLM zero-shot / few-shot / reranking baselines;
- API-based LLM experiments;
- local small-LLM LoRA / QLoRA fine-tuning;
- our original LLM4Rec method;
- ablation studies;
- cold-start analysis;
- long-tail analysis;
- hallucination and validity evaluation;
- diversity, coverage, novelty metrics;
- latency, token, and cost statistics;
- reproducible result export for paper tables and plots.

The implementation must make it possible to start real experiments after the core modules are completed.

Before implementing or changing OursMethod, Codex must read
`docs/RESEARCH_IDEA.md` and `docs/PROJECT_MEMORY.md`. `AGENTS.md` defines
engineering standards, not the core research idea. Codex must not replace
OursMethod with a generic LLM reranker, generic RAG recommender,
prompt-engineering baseline, or known-paper clone.

## 1A. Durable Project Memory

Future agents must preserve the project direction recorded in
`docs/PROJECT_MEMORY.md`. For any nontrivial task, read that file before
planning and update it when the task changes the roadmap, baseline policy,
server commands, evidence boundaries, or current status.

For every complex task, default to multi-agent collaboration when tools permit
it. Use at least one implementation/explorer pass and one reviewer/fairness
pass. The reviewer pass must compare TRUCE-Rec against the rigor, originality,
technical depth, experimental completeness, and artifact standards expected in
recent top-conference recommender/LLM4Rec papers. When literature context is
needed, search broadly across multiple relevant SIGIR/WWW/KDD/RecSys/NeurIPS/
ICLR-style papers and official projects, not only one or two convenient
references. If subagents are unavailable, perform these passes explicitly in
the main agent and say so.

At the start of a nontrivial task, read the default startup packet unless the
task is clearly unrelated:

- `AGENTS.md`
- `docs/PROJECT_MEMORY.md`
- `docs/RESEARCH_IDEA.md`
- `docs/submission_roadmap.md`
- `docs/top_conference_review_plan.md`
- `docs/qwen3_lora_controlled_baselines.md` for baseline/protocol work
- `docs/server_execution_matrix.md` and `docs/server_next_commands.md` for
  server-facing work
- relevant source, config, and test files before editing

At the end of a nontrivial task, report what was completed, commands/tests run,
remaining risks, and the next concrete plan. Also state whether the current
project/experiment stage is still open or can be considered complete. Do not
create an endless "next step" loop: the project can move from experiments to
paper writing only after four-domain runs, official/fair baselines, Ours
ablations, statistical tests, efficiency/cost artifacts, failure cases, and a
top-conference-style reviewer pass show no fatal gaps.

The current strategic spine is:

```text
generative recommendation observation
  -> Beauty full-domain plus books/electronics/movies 10k-user observation
  -> base Qwen3-8B and senior-recommended Qwen3-8B-LoRA baseline observation
  -> catalog grounding and uncertainty/popularity/long-tail/echo diagnostics
  -> original, non-stitched CURE/TRUCE framework
  -> Qwen3-8B-LoRA Ours adapter and ablations
  -> official Qwen3-8B-LoRA baseline families
  -> shared same-candidate evaluator
  -> four-domain paper-scale experiments
```

Main LLM baseline policy follows the senior collaborator's recommended
academic setup: obtain official source implementations, use Qwen3-8B as the
shared backbone, train all compared LLM methods with LoRA, use official default
or reported-optimal baseline hyperparameters, do not tune baselines on TRUCE
test outcomes, and evaluate all methods through the same TRUCE candidate rows
and evaluator. Ours may tune hyperparameters only under the declared validation
protocol.

Use multi-agent collaboration for nontrivial tasks when the environment
permits it. At minimum, separate implementation from protocol/fairness review
and top-conference-style critique. If subagents are unavailable, perform those
passes explicitly yourself.

Server `pony-rec-gpu` is now directly accessible via SSH: `ssh pony-rec-gpu "<command>"`.
Host: `125.71.97.70:15302`, User: `ajifang`, GPU: RTX 4090 (49GB).
Server project path: `~/projects/pony-rec-rescue-shadow-v6`.
Do not guess server state — run a command to verify. After substantial local
work, commit and push to `origin/main` unless the user explicitly asks not to.

Do not let docs become stale. After a completed stage, update the artifact
truth, narrative truth, and comparability truth: manifests/metrics/logs,
roadmap/status/limitations, and candidate/evaluator/baseline status.

## 2. Expected Repository Structure

Prefer the following structure unless the existing repo already has a better compatible layout:

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt / pyproject.toml
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── retrievers/
│   ├── baselines/
│   ├── llm/
│   ├── training/
│   ├── evaluation/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── tiny/
│   └── README.md
├── src/
│   └── llm4rec/
│       ├── __init__.py
│       ├── data/
│       ├── prompts/
│       ├── retrievers/
│       ├── rankers/
│       ├── generators/
│       ├── llm/
│       ├── models/
│       ├── trainers/
│       ├── evaluation/
│       ├── metrics/
│       ├── experiments/
│       ├── utils/
│       └── io/
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── rerank.py
│   ├── run_experiment.py
│   ├── run_all.py
│   └── export_tables.py
├── tests/
│   ├── unit/
│   ├── smoke/
│   └── fixtures/
├── outputs/
│   ├── logs/
│   ├── metrics/
│   ├── checkpoints/
│   ├── predictions/
│   ├── tables/
│   └── runs/
└── docs/
    ├── experiment_protocol.md
    ├── data_format.md
    ├── baselines.md
    └── paper_plan.md
```

If the current repo uses a different layout, preserve compatibility where reasonable and document deviations.

3. Required Core Interfaces

All implementations must use unified interfaces. Do not create one-off scripts that bypass these interfaces.

3.1 Dataset / DataModule

Required responsibilities:

load raw interactions;
map raw user/item IDs to internal IDs;
load or build item metadata;
build user histories;
create train/valid/test splits;
support temporal split and leave-one-out split;
support sampled ranking and full ranking;
construct candidate sets;
expose item text for LLM prompts and retrieval;
save processed artifacts.

Expected records:

Interaction = {
    "user_id": str,
    "item_id": str,
    "timestamp": int | float | None,
    "rating": float | None,
    "domain": str | None,
}

ItemRecord = {
    "item_id": str,
    "title": str,
    "description": str | None,
    "category": str | None,
    "brand": str | None,
    "domain": str | None,
    "raw_text": str | None,
}

UserExample = {
    "user_id": str,
    "history": list[str],
    "target": str,
    "candidates": list[str] | None,
    "domain": str | None,
}

Required files or equivalent:

src/llm4rec/data/base.py
src/llm4rec/data/registry.py
src/llm4rec/data/preprocess.py
src/llm4rec/data/splits.py
src/llm4rec/data/candidates.py
src/llm4rec/data/text_fields.py
3.2 Retriever

A retriever returns candidate items.

Required retrievers:

PopularityRetriever
BM25Retriever
DenseRetriever interface
SequentialModelRetriever interface
HybridRetriever

Required output:

RetrievalResult = {
    "user_id": str,
    "items": list[str],
    "scores": list[float],
    "metadata": dict,
}

Required files or equivalent:

src/llm4rec/retrievers/base.py
src/llm4rec/retrievers/popularity.py
src/llm4rec/retrievers/bm25.py
src/llm4rec/retrievers/dense.py
src/llm4rec/retrievers/hybrid.py
3.3 Ranker

A ranker scores or reorders a candidate set.

Required rankers:

RandomRanker
PopularityRanker
BM25Ranker
MatrixFactorization/BPR ranker if feasible
Sequential ranker interface
LLMReranker

Required output:

RankingResult = {
    "user_id": str,
    "items": list[str],
    "scores": list[float],
    "raw_output": str | None,
    "metadata": dict,
}

Required files or equivalent:

src/llm4rec/rankers/base.py
src/llm4rec/rankers/popularity.py
src/llm4rec/rankers/bm25.py
src/llm4rec/rankers/mf.py
src/llm4rec/rankers/sequential.py
src/llm4rec/rankers/llm_reranker.py
3.4 Generator

A generator produces recommendation outputs, item IDs, explanations, or structured responses.

Required responsibilities:

constrained generation over candidate item IDs;
parse generated recommendation lists;
detect invalid items;
optionally generate explanations;
preserve raw LLM output for auditing.

Required files or equivalent:

src/llm4rec/generators/base.py
src/llm4rec/generators/constrained.py
src/llm4rec/generators/parser.py
src/llm4rec/generators/explanation.py
3.5 LLM Provider

Required providers:

Mock provider only for tests, not experiments;
OpenAI-compatible API provider;
local Hugging Face causal LM provider;
local LoRA / QLoRA inference provider.

Required features:

temperature, top_p, max_tokens, seed if supported;
retry and timeout handling;
token usage logging when available;
latency logging;
raw request/response persistence;
deterministic test mode.

Required files or equivalent:

src/llm4rec/llm/base.py
src/llm4rec/llm/openai_provider.py
src/llm4rec/llm/hf_provider.py
src/llm4rec/llm/mock_provider.py
src/llm4rec/llm/cost_tracker.py
src/llm4rec/llm/response_cache.py
3.6 Prompt Builder

Required prompt types:

zero-shot recommendation;
few-shot recommendation;
candidate reranking;
constrained item-ID generation;
explanation generation;
preference summarization;
self-verification / grounding check.

Required files or equivalent:

src/llm4rec/prompts/base.py
src/llm4rec/prompts/templates.py
src/llm4rec/prompts/builder.py
src/llm4rec/prompts/formatters.py
src/llm4rec/prompts/parsers.py
3.7 Trainer

Required training support:

traditional baseline training;
sequential baseline training;
local small-LLM LoRA / QLoRA training interface;
checkpoint save/load;
resume;
eval-only mode;
gradient accumulation;
mixed precision where applicable;
seed control.

Required files or equivalent:

src/llm4rec/trainers/base.py
src/llm4rec/trainers/traditional.py
src/llm4rec/trainers/sequential.py
src/llm4rec/trainers/lora.py
src/llm4rec/trainers/checkpointing.py
3.8 Evaluator

One evaluator must be shared by all methods.

Required responsibilities:

load predictions;
validate prediction schema;
compute ranking metrics;
compute validity / hallucination metrics;
compute coverage / diversity / novelty / long-tail metrics;
compute latency / token / cost metrics;
support per-domain and aggregate metrics;
export JSON, CSV, and LaTeX-ready tables.

Required files or equivalent:

src/llm4rec/evaluation/evaluator.py
src/llm4rec/evaluation/prediction_schema.py
src/llm4rec/evaluation/export.py
src/llm4rec/evaluation/significance.py
3.9 Metrics

Required ranking metrics:

Recall@K
NDCG@K
HitRate@K
MRR@K
MAP@K if feasible

Required LLM/grounding metrics:

validity rate
hallucination rate
parse success rate
candidate adherence rate
explanation groundedness placeholder or interface

Required beyond-accuracy metrics:

item coverage
catalog coverage
intra-list diversity
novelty
long-tail ratio
popularity-stratified metrics

Required efficiency metrics:

latency mean / p50 / p95
token count
API cost if applicable
GPU memory if available
throughput

Required files or equivalent:

src/llm4rec/metrics/ranking.py
src/llm4rec/metrics/validity.py
src/llm4rec/metrics/diversity.py
src/llm4rec/metrics/long_tail.py
src/llm4rec/metrics/efficiency.py
3.10 Experiment Runner

Required responsibilities:

read YAML config;
instantiate dataset, retriever, ranker/generator, evaluator;
save resolved config;
save logs;
save predictions;
save metrics;
support dry-run;
support smoke-run;
support multi-seed runs;
support multi-domain runs.

Required files or equivalent:

src/llm4rec/experiments/config.py
src/llm4rec/experiments/runner.py
src/llm4rec/experiments/registry.py
src/llm4rec/experiments/seeding.py
src/llm4rec/experiments/logging.py
4. Required Baselines

The codebase must support at least the following baseline families.

4.1 Non-personalized
Random
Popularity
4.2 Traditional Collaborative Filtering
Matrix Factorization or BPR-MF
LightGCN or compatible graph recommender if feasible
4.3 Sequential Recommendation

At least one implemented or wrapped:

GRU4Rec
SASRec
BERT4Rec
4.4 Text Retrieval / Text Ranking
BM25
dense retrieval interface
sentence-transformer retrieval if dependency is acceptable
4.5 LLM Baselines
zero-shot direct recommendation;
few-shot direct recommendation;
candidate reranking;
constrained candidate reranking;
explanation generation baseline.

All baselines must output the same prediction schema and must be evaluated by the same evaluator.

5. Required Data Support

The system should be dataset-agnostic.

At minimum:

tiny fixture dataset for smoke tests;
one MovieLens-style dataset adapter;
one Amazon Reviews-style multi-domain adapter or interface;
generic CSV/JSONL adapter.

Required split strategies:

leave-one-out;
temporal split;
user-stratified split if applicable.

Required candidate protocols:

full ranking when feasible;
sampled negatives with fixed seed;
candidate set loaded from file;
retriever-generated candidate set.

All protocols must be recorded in config and saved with outputs.

6. Required Configs

Use YAML configs. Do not rely on manual source edits.

Expected config categories:

configs/datasets/tiny.yaml
configs/datasets/movielens.yaml
configs/datasets/amazon_books.yaml
configs/retrievers/popularity.yaml
configs/retrievers/bm25.yaml
configs/baselines/popularity.yaml
configs/baselines/bpr.yaml
configs/baselines/sasrec.yaml
configs/baselines/llm_rerank.yaml
configs/llm/openai_compatible.yaml
configs/llm/hf_local.yaml
configs/training/lora.yaml
configs/evaluation/default.yaml
configs/experiments/smoke.yaml
configs/experiments/main_movielens.yaml
configs/experiments/main_multidomain.yaml
configs/experiments/ablation.yaml
configs/experiments/cold_start.yaml
configs/experiments/long_tail.yaml

Each experiment config must specify:

dataset;
split strategy;
candidate strategy;
model or baseline;
prompt template if applicable;
metrics;
seeds;
output directory;
logging level;
device;
run mode.
7. Required Scripts

Scripts must be thin wrappers around source modules.

Required commands:

python scripts/preprocess.py --config configs/datasets/tiny.yaml
python scripts/train.py --config configs/experiments/smoke.yaml
python scripts/evaluate.py --config configs/experiments/smoke.yaml
python scripts/run_experiment.py --config configs/experiments/smoke.yaml
python scripts/run_all.py --config configs/experiments/smoke.yaml
python scripts/export_tables.py --input outputs/metrics --output outputs/tables
pytest tests/

If commands differ, document the actual commands in README and in the audit report.

8. Required Output Artifacts

Each run must save:

outputs/runs/<run_id>/
├── resolved_config.yaml
├── environment.json
├── git_info.json
├── logs.txt
├── predictions.jsonl
├── metrics.json
├── metrics.csv
├── cost_latency.json
├── checkpoints/
└── artifacts/

Prediction JSONL schema:

{
  "user_id": "u1",
  "target_item": "i9",
  "candidate_items": ["i1", "i2", "i9"],
  "predicted_items": ["i9", "i2", "i1"],
  "scores": [0.9, 0.5, 0.1],
  "method": "llm_rerank",
  "domain": "movies",
  "raw_output": null,
  "metadata": {}
}
9. Testing Requirements

Minimum tests:

dataset loading smoke test;
split determinism test;
candidate generation test;
ranking metrics correctness test;
validity/hallucination metric test;
prompt formatting test;
LLM response parser test;
experiment runner smoke test;
export table smoke test.

Tests must use tiny fixture data and must not require API keys or large downloads.

10. Done Criteria

A stage is not complete unless:

source code exists;
config exists;
script entrypoint exists if needed;
smoke test or unit test exists;
command has been run;
result is reported;
remaining risks are documented.

For every stage, report:

Files changed:
Commands run:
Tests run:
Observed results:
Design rationale:
Remaining risks:
Next step:
11. Paper-Writing Rule

Paper writing is allowed only after the experiment pipeline can produce real metrics.

The paper draft must be grounded in:

actual implemented modules;
actual experiment protocol;
actual metrics files;
actual ablation outputs;
actual failure cases or limitations.

Do not invent results.
