---
name: truce-rec
description: Operate safely inside the TRUCE-Rec research codebase. Use when working on TRUCE-Rec code, configs, docs, experiments, Ours/CURE/TRUCE method work, baseline implementation or audits, server handoffs, artifact validation, paper-readiness review, or any nontrivial LLM4Rec recommendation task that must preserve same-candidate fairness, Qwen3-8B-LoRA baseline policy, evidence boundaries, and publishable-research standards.
---

# TRUCE-Rec

## Core Contract

Treat TRUCE-Rec as a publishable LLM4Rec research system. Do not add toy demos,
mock-only claims, pseudo-results, paper conclusions, or shortcuts that make
baselines incomparable.

Preserve the research spine:

```text
generative recommendation observation
  -> catalog grounding and uncertainty/popularity/long-tail/echo diagnostics
  -> original non-stitched CURE/TRUCE framework
  -> Qwen3-8B-LoRA Ours and official/fair baselines
  -> shared same-candidate evaluator
  -> four-domain paper-scale experiments
```

## Startup Workflow

For every nontrivial task, read the project startup packet before planning:

- `AGENTS.md`
- `docs/PROJECT_MEMORY.md`
- `docs/RESEARCH_IDEA.md`
- `docs/submission_roadmap.md`
- `docs/top_conference_review_plan.md`
- task-specific source, config, and test files before editing

Read additional docs by task type:

- Baselines/protocols: `docs/qwen3_lora_controlled_baselines.md` and
  `docs/controlled_baseline_fidelity_audit.md` when present.
- Server work: `docs/server_execution_matrix.md`,
  `docs/server_next_commands.md`, and the exact scripts being changed.
- Ours/method work: `docs/ours_method_plan.md`,
  `docs/cure_truce_framework.md`, `docs/ablation_protocol.md`, and current
  method source/tests when present.

For the full decision table, read `references/task-startup.md`.

## How To Work

1. Inspect first with `rg`, `rg --files`, and targeted file reads.
2. Plan before large refactors or research-method changes.
3. Keep source changes config-driven and compatible with existing interfaces.
4. Use the shared candidate rows, score schema, prediction schema, and evaluator
   for every comparable method.
5. Add or update focused tests/smoke checks when behavior changes.
6. Run local commands that are feasible; for server-only jobs, provide exact
   commands and wait for pasted logs/artifacts.
7. Update stale docs when status, commands, evidence boundaries, or baseline
   policy changes.
8. After substantial local code/doc/config work, commit and push unless the user
   explicitly says not to.

When the environment permits and the task is complex, use separate
implementation/explorer and reviewer/fairness passes. If subagents are
unavailable, perform those passes explicitly in the main agent.

## Research Guardrails

Before changing Ours/CURE/TRUCE, read `docs/RESEARCH_IDEA.md` and
`docs/PROJECT_MEMORY.md`.

Do not turn Ours into:

- a generic LLM reranker;
- a generic RAG recommender;
- prompt engineering as the contribution;
- a renamed or stitched clone of TALLRec, OpenP5, DEALRec, LC-Rec, LLaRA,
  LLM-ESR, or another reference project.

The contribution must stay recommendation-specific: generated title grounding,
catalog validity, hallucination, popularity-confounded confidence, long-tail
under-confidence, history/echo risk, and conservative fallback-aware routing or
policy learning.

For evidence labels, paper-result rules, and method no-go checks, read
`references/research-evidence.md`.

## Baseline And Evaluation Policy

Main LLM baseline comparisons must follow the shared policy:

```text
official implementation where claimed
  + Qwen3-8B shared backbone
  + LoRA adaptation
  + official/default or reported-optimal baseline hyperparameters
  + same TRUCE splits/candidates/evaluator
  + preserved event/source IDs
```

Never tune baselines on TRUCE test outcomes. Ours may tune only through the
declared validation protocol.

Do not call TRUCE-side adapter pilots official-native baselines. Do not mix
methods with different candidates, splits, negative sampling, backbones, or
evaluators in one main comparison table.

For baseline families, score schemas, Week8 domains, and server handoffs, read
`references/baselines-server.md`.

## Server And Evidence Discipline

Assume local Codex cannot see the server unless artifacts are local or the user
pastes logs. Provide concrete server commands, then wait for evidence before
claiming completion.

Use these evidence labels consistently:

- `smoke/mock`
- `diagnostic`
- `controlled_adapter_pilot`
- `official_native_controlled`
- `paper_result`

Paper writing is allowed only after real metrics, ablations, failure cases,
statistical tests, efficiency/cost artifacts, and a top-conference-style review
show no fatal gaps.

For reviewer prompts and writing-ready gates, read
`references/review-checklist.md`.
