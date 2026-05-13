# Task Startup

Use this reference when deciding what to read and how much process a TRUCE-Rec
task needs.

## Always Read For Nontrivial Work

- `AGENTS.md`: engineering rules and done criteria.
- `docs/PROJECT_MEMORY.md`: durable status, policies, and current next moves.
- `docs/RESEARCH_IDEA.md`: core research idea and Ours no-go rules.
- `docs/submission_roadmap.md`: milestone ladder and stage gates.
- `docs/top_conference_review_plan.md`: reviewer standard and writing gate.
- Relevant source, config, tests, and docs before editing.

## Task-Specific Reads

- Baseline/fairness work:
  `docs/qwen3_lora_controlled_baselines.md`,
  `docs/controlled_baseline_fidelity_audit.md` if present, baseline configs,
  packet builders, importers, and evaluator code.
- Server work:
  `docs/server_execution_matrix.md`, `docs/server_next_commands.md`, and the
  exact server scripts or manifests being changed.
- Ours/CURE/TRUCE method work:
  `docs/ours_method_plan.md`, `docs/cure_truce_framework.md`,
  `docs/ablation_protocol.md`, current method modules, adapter preparation
  scripts, import/evaluation scripts, and tests.
- Paper/export work:
  metric manifests, `outputs/runs/**/metrics.json`, raw predictions, artifact
  checklists, limitations, and reviewer gates. Do not write conclusions from
  missing artifacts.
- Literature or official-repo work:
  search broadly and prefer official papers, repositories, docs, or release
  artifacts. Use them for fidelity guidance only.

## Work Pattern

1. Restate the current artifact/evidence boundary before changing it.
2. Inspect the repo with `rg` and targeted reads.
3. Identify whether the task is source, config, docs, server handoff, baseline
   audit, method design, experiment execution, or paper export.
4. Make scoped edits through existing interfaces and configs.
5. Validate with tests, smoke runs, dry runs, or explicit server commands.
6. Update docs that would otherwise be stale.
7. Report files changed, commands/tests run, observed results, risks, and the
   next concrete step.

Use multi-pass review for complex work: an implementation/explorer pass plus a
fairness/top-conference reviewer pass. If subagents are not available, run
those passes explicitly in the main response.
