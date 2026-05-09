# TRUCE-Rec Project Memory

This document is the durable memory for future Codex/agent sessions. Read it
before planning nontrivial work. Keep it current after each completed stage so
new agents do not have to reconstruct the project from stale fragments.

Last major update: 2026-05-09.

## One-Sentence Direction

TRUCE-Rec is a publishable LLM4Rec research system, not a toy demo: it starts
from recommendation-specific uncertainty observations, builds an original
CURE/TRUCE method, compares against official Qwen3-8B-LoRA baseline families
under one same-candidate evaluator, and scales to Beauty/Books/Electronics/
Movies with 10k-user, 1-positive+100-negative protocols.

## Required Startup Reading

Before a nontrivial task, read at least:

- `AGENTS.md`: engineering and done criteria.
- `docs/PROJECT_MEMORY.md`: current project memory and workflow contract.
- `docs/RESEARCH_IDEA.md`: core research idea; do not replace it with generic
  LLM reranking, RAG, or prompt engineering.
- `docs/submission_roadmap.md`: milestone ladder.
- `docs/qwen3_lora_controlled_baselines.md`: baseline fairness protocol.
- `docs/server_execution_matrix.md` and `docs/server_next_commands.md` when
  the task touches server execution.
- The relevant source/config/test files before editing.

Do not work from memory alone. If a paper/repo/API/detail is current or
uncertain, look it up and prefer official sources.

## User Workflow Assumptions

- The user has the server; Codex usually cannot directly inspect server files
  unless the user pastes logs/results or a local mount exists.
- Give concrete server commands for the user to run. The user will paste
  outputs or errors back into chat.
- Do not claim a server result unless the user provides logs/artifacts or the
  repo contains tracked evidence.
- After substantial local code/doc/config work, commit and push to
  `origin/main` unless the user explicitly says not to.
- Keep responses and handoffs in Chinese when reporting to the user, unless a
  generated artifact has an existing English style.

## Multi-Agent Collaboration Rule

The user explicitly wants multi-agent collaboration for nontrivial tasks.

Use multiple agents when the active tool policy permits and the task is not a
small one-command/simple-answer job. Typical roles:

- Explorer for codebase/protocol audit.
- Explorer for literature/official-repo/fairness review.
- Worker for bounded implementation slices with disjoint files.
- Reviewer for top-conference-style critique, novelty risk, and baseline
  fairness.
- Main agent integrates, verifies, commits, pushes, and gives the final server
  commands.

If subagents are unavailable in a future environment, simulate the same checks
explicitly as separate audit passes. Do not skip the review/audit layer just
because the task feels familiar.

## Non-Toy Standard

Never add toy demos, pseudo-results, mock-only claims, or notebook-only paths
unless the user explicitly asks for a small test fixture.

Real progress means:

- source code or executable scripts exist;
- configs/manifests exist;
- tests or smoke checks exist;
- commands have been run locally when possible;
- server-only commands are documented for the user;
- evidence boundaries are labeled;
- stale docs are updated or deleted;
- the change is committed and pushed.

Documentation alone is acceptable only for governance/planning tasks. For
framework or experiment tasks, prefer runnable code, validators, importers,
or orchestration scripts plus docs.

## Research Spine

The project must stay on this spine:

```text
LLM generative recommendation observation
  -> catalog grounding and uncertainty/popularity/long-tail/echo diagnostics
  -> original CURE/TRUCE framework
  -> Qwen3-8B-LoRA Ours adapter and ablations
  -> official Qwen3-8B-LoRA baseline families
  -> shared same-candidate evaluator
  -> four-domain paper-scale experiments
  -> top-conference review and artifact export
```

The contribution should not be stated as "we ask an LLM for confidence." The
contribution is recommendation-specific uncertainty: generated title grounding,
catalog validity, hallucination, popularity-confounded confidence, long-tail
under-confidence, history/echo risk, and uncertainty-aware routing/reranking or
training.

## Ours Framework Memory

The stronger Ours direction is:

```text
observation signals
  + catalog grounding
  + candidate-normalized diagnostic panels
  + popularity residual/deconfounding
  + history/echo risk
  + learned or structured improve/harm/risk targets
  + conservative promotion/fusion over a shared fallback ranking
```

Current anchors:

- `src/llm4rec/methods/ours_framework.py`
- `scripts/prepare_ours_qwen_adapter_training.py`
- `scripts/import_evaluate_ours_adapter.py`
- `src/llm4rec/methods/cu_gr.py`
- `src/llm4rec/methods/preference_fusion.py`
- `src/llm4rec/methods/override_calibrator.py`

Do not pitch the older smoke `OursMethodRanker` as the final research method.
It is infrastructure. The paper-grade path should emphasize CURE/TRUCE,
structured uncertainty targets, ablations, and observation-motivated design.

Ours may tune hyperparameters only with a declared validation protocol. Never
tune on test.

## Senior Baseline Advice To Preserve

The user's senior colleague gave the following practical academic advice:

1. Fair baseline comparison has several accepted modes:
   - run original code with original backbone/hyperparameters and only adapt the
     input dataset;
   - reuse a prior work's dataset and its reported baselines;
   - adapt every LLM baseline to the same backbone, such as Qwen3-8B;
   - tune both baselines and Ours, though this is expensive.
2. The recommended route for this project is:
   - use official source code;
   - use Qwen3-8B as the shared backbone;
   - use LoRA for all compared LLM methods;
   - use each baseline's official default or reported optimal hyperparameters;
   - do not spend time exhaustively tuning every baseline;
   - Ours may tune hyperparameters under the validation protocol.
3. In the experimental setting, write clearly:
   "We obtain source code from official implementations, use Qwen3-8B as the
   shared backbone, train with LoRA, use official/default hyperparameters for
   baselines, and evaluate all methods on the same TRUCE candidate protocol."
4. Full fine-tuning versus LoRA must not be mixed silently. If full fine-tuning
   is used, it needs a separated protocol/table.
5. Reviewers may still challenge fairness, but this Qwen3-8B-LoRA controlled
   setup is a common academic compromise.

This advice is now the main LLM baseline policy unless the user explicitly
changes it.

## Baseline Contract

Main compared LLM baselines:

```text
official project implementation
  + Qwen3-8B base model
  + LoRA adaptation
  + official default or reported-optimal baseline hyperparameters
  + shared TRUCE split/candidates/evaluator
  + example_id,user_id,item_id,score
```

Main official baseline families:

- TALLRec
- OpenP5
- DEALRec
- LC-Rec
- LLaRA
- LLM-ESR

Current TRUCE-side controlled adapters are pilots unless promoted by an
official-fidelity audit. Do not call them final official baselines. CoLLM,
SLMRec, BIGRec, and other methods may be follow-up or appendix candidates.

Every official baseline must record:

- official repo and commit;
- official config/hyperparameter source;
- official modules/objective reused;
- Qwen3-8B-LoRA compatibility changes;
- score export shim;
- TRUCE import/evaluation artifacts;
- evidence label.

## Data And Experiment Scale

Small fixture data is for tests only. Amazon Beauty can debug the pipeline, but
the intended paper-scale evaluation is the four-domain same-candidate protocol:

- domains: `beauty`, `books`, `electronics`, `movies`;
- up to 10,000 users per domain;
- each event has 1 positive + 100 popularity-sampled negatives;
- same-candidate setting for every method;
- negative sampling: popularity;
- test history mode: train_plus_valid;
- preserve `event_id`, `source_event_id`, `user_id`, `item_id`, and `split`.

Server source root expected from the parallel data project:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/
```

Do not resample users, negatives, or candidates inside TRUCE. If candidate rows
change, all comparable methods must be regenerated.

## Server Operating Model

Local Codex does not automatically see server state. Treat server work as a
command-and-log loop:

1. Update/push repo locally.
2. Give user exact server commands.
3. User runs commands on `~/projects/TRUCE-Rec`.
4. User pastes logs/status/errors.
5. Agent diagnoses and updates code/docs if needed.
6. Commit/push fixes.

Preferred server entrypoints:

```bash
cd ~/projects/TRUCE-Rec
git pull --ff-only
bash scripts/server/run_week8_four_domain_pipeline.sh
```

For current Beauty controlled-adapter pilots:

```bash
cd ~/projects/TRUCE-Rec
git pull --ff-only
bash scripts/server/run_controlled_baseline_queue.sh smoke
bash scripts/server/run_controlled_baseline_queue.sh full
```

If long GPU jobs are run without `tmux`, use logging scripts or `nohup` with
explicit log files and PID/status checks. Do not assume a job survived a
disconnect; verify with `ps`, `nvidia-smi`, summaries, candidate scores, and
metrics artifacts.

## Evidence Boundaries

Use these labels consistently:

- `smoke/mock`: code path only.
- `diagnostic`: useful QA or observation, not paper evidence.
- `controlled_adapter_pilot`: TRUCE-side adapter under shared protocol, not yet
  official-native.
- `official_native_controlled`: official implementation adapted to Qwen3-8B-
  LoRA and TRUCE protocol, eligible only after full train/score/import/eval.
- `paper_result`: completed approved real run with tracked code, manifests,
  logs, raw scores/responses where applicable, predictions, metrics, and
  artifact checklist.

Never fabricate metrics, tables, paper conclusions, or server status. Paper
writing comes after real metrics and ablations exist.

## Update Discipline

After any completed stage, update every file whose status would otherwise be
stale. At minimum consider:

- `docs/PROJECT_MEMORY.md`: big direction, policies, current status, next
  commands.
- `docs/PHASE_HANDOFF.md`: latest completed stage and server status.
- `README.md`: high-level status, important docs, commands.
- `docs/submission_roadmap.md`: milestone status and exit criteria.
- `docs/server_next_commands.md`: exact server commands after code changes.
- `docs/server_execution_matrix.md`: artifact gates and new entrypoints.
- baseline docs/configs when fairness/provenance changes.
- tests when behavior changes.

Prefer updating or deleting stale statements over appending contradictory new
sections. The next agent should not have to guess which paragraph is current.

## CRUD And Review Expectations

Future agents should perform real maintenance:

- Create missing modules, scripts, configs, tests, and docs.
- Read and review relevant existing files before editing.
- Update stale policies/status and command sheets.
- Delete or rename obsolete paths when a name misleads future work.
- Validate with tests or clear dry-run commands.
- Search official docs/repos/papers when baseline details are uncertain.
- Run a reviewer-style critique for novelty, fairness, leakage, and toy-risk
  before declaring a stage complete.

## Current Next Moves

1. On the server, pull latest `main`.
2. Check whether Week8 `beauty/books/electronics/movies` task directories exist.
3. Run `scripts/server/run_week8_four_domain_pipeline.sh` when ready.
4. Import/evaluate any completed Beauty controlled-adapter pilot only as
   `controlled_adapter_pilot`.
5. Continue official-native baseline audits for all six main families.
6. Train/score/evaluate Ours Qwen3-LoRA adapter and ablations on the same
   candidate rows.
7. Run observation diagnostics on Ours and strong baselines, not only base or
   weak models.
8. Use top-conference reviewer/literature checks before paper claims.
