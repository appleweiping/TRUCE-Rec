# TRUCE-Rec Project Memory

This document is the durable memory for future Codex/agent sessions. Read it
before planning nontrivial work. Keep it current after each completed stage so
new agents do not have to reconstruct the project from stale fragments.

Last major update: 2026-05-13.

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

Project-local skill: `.codex/skills/truce-rec/SKILL.md` now summarizes the
startup workflow, research guardrails, baseline/server discipline, evidence
labels, and reviewer gates for future Codex sessions. Use it as the compact
operating entrypoint, then load the canonical docs above as needed.

Do not work from memory alone. If a paper/repo/API/detail is current or
uncertain, look it up and prefer official sources.

Default startup packet by task type:

- Any roadmap/method/baseline/server task: `AGENTS.md`,
  `docs/PROJECT_MEMORY.md`, `docs/RESEARCH_IDEA.md`,
  `docs/submission_roadmap.md`, and `docs/top_conference_review_plan.md`.
- Baseline or fairness task: also read
  `docs/qwen3_lora_controlled_baselines.md`,
  `docs/controlled_baseline_fidelity_audit.md`, and relevant official packet
  docs/configs.
- Server task: also read `docs/server_execution_matrix.md`,
  `docs/server_next_commands.md`, and the exact server scripts being changed.
- Ours/method task: also read `docs/ours_method_plan.md`,
  `docs/cure_truce_framework.md`, `docs/ablation_protocol.md`, and current
  method source/tests.

If the task touches literature, novelty, or baseline selection, search broadly
across multiple recent top-conference papers and official repositories instead
of relying on a few convenient examples. Prefer official sources for factual
claims.

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

Use multiple agents by default when the active tool policy permits and the task
is not a small one-command/simple-answer job. Typical roles:

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

For method-building tasks, the expected loop is:

```text
implementation proposal
  -> top-conference reviewer critique
  -> fairness/protocol audit
  -> implementation revision
  -> runnable server command/update
```

Record the reviewer verdict honestly. If the current Ours design is still a
heuristic scaffold, say so and improve it rather than writing paper-ready
claims.

The reviewer pass should compare TRUCE-Rec against recent top-conference
LLM4Rec/recommender work on:

- rigor of experimental protocol;
- originality rather than stitched components;
- technical depth and model/algorithm complexity;
- strength and officialness of baselines;
- data scale and multi-domain coverage;
- ablation completeness;
- leakage and fairness controls;
- statistical testing, efficiency, and reproducibility.

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
  -> Beauty full-domain and books/electronics/movies 10k-user observations
  -> base Qwen3-8B plus four senior-recommended Qwen3-8B-LoRA baseline observations
  -> catalog grounding and uncertainty/popularity/long-tail/echo diagnostics
  -> original non-stitched CURE/TRUCE framework
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

When reference papers or official repos are needed, future agents may carefully
read senior-recommended or top-conference projects to understand task
formulation, official training flow, and fair reproduction details. Those works
are inspiration and fidelity guidance only. The actual TRUCE/CURE method must
not be stitched, copied, or renamed from their objectives, prompts, or system
pipelines.

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

Reviewer audit as of 2026-05-09: the current Ours scaffold is promising but not
yet enough for a strong submission. Its weak point is that pairwise/listwise SFT
prompts and conservative gates can look like hand-written heuristics. The next
method upgrade must add a learned observation-to-target layer, candidate-
normalized uncertainty, popularity residual/deconfounding, echo/history guard,
learned improve/harm/abstain policy, fallback-preserving fusion, and ablations
tied directly to the observation findings.

Implementation update as of 2026-05-09: `src/llm4rec/methods/ours_framework.py`
now adds an observation-residual policy target layer. Ours adapter supervision
includes candidate-normalized utility, popularity-residual utility, harm risk,
abstain risk, and conservative `promote/suppress/defer_to_fallback` policy
actions. Server scoring now estimates the likelihood of
`{"policy_action": "promote"}` for each candidate, while preserving the same
`candidate_scores.csv` schema. This is still not a paper result; it is a
stronger trainable objective for the next server runs and ablations.

Core method milestones:

- M2a: derive structured train/valid targets from observation rows without
  using test correctness. Current v2 scaffold derives deterministic
  observation-residual policy targets from train/catalog evidence.
- M2b: train a TRUCE adapter/policy that predicts improve/harm/abstain or
  calibrated candidate preference from diagnostic evidence. Current v2 scoring
  target is promote-action likelihood.
- M2c: combine the learned policy with conservative fallback ranking so bad
  LLM generations can be blocked rather than blindly promoted.
- M2d: run ablations for grounding, uncertainty, candidate normalization,
  popularity residuals, echo/history guard, and fallback-only routing.
- M2e: pass a reviewer novelty check confirming the method is not a generic
  LLM reranker, prompt-engineering baseline, RAG wrapper, or stitched clone of
  the reference projects.

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
- Beauty supplementary smaller-N plus up to 10,000 users for books,
  electronics, and movies;
- each event has 1 positive + 100 popularity-sampled negatives;
- same-candidate setting for every method;
- negative sampling: popularity;
- test history mode: train_plus_valid;
- preserve `event_id`, `source_event_id`, `user_id`, `item_id`, and `split`.

Server source root expected from the parallel data project:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/
```

Current reusable artifact slugs:

- `beauty_supplementary_smallerN_100neg`
- `books_large10000_100neg`
- `electronics_large10000_100neg`
- `movies_large10000_100neg`

Do not resample users, negatives, or candidates inside TRUCE. If candidate rows
change, all comparable methods must be regenerated.

Do not modify `candidate_items.csv` or `ranking_valid/test.jsonl` in the
producer artifact directories. For the cross-project same-candidate lane, all
methods must export:

```text
source_event_id,user_id,item_id,score
```

and must be imported/evaluated with `main_import_same_candidate_baseline_scores.py`.
TRUCE internal `example_id` imports may be used for local scaffolding, but the
source-event schema is the authoritative four-domain artifact contract. Never
use `test` for hyperparameter selection. If reusing LLM2Rec official results,
reuse only scores/provenance/audit; do not make intermediate checkpoints or
embeddings required long-term artifacts.

Observation scale must match the formal training/evaluation scale whenever
budget allows. The intended observation set is Beauty full-domain plus
books/electronics/movies 10k-user same-candidate tasks. Observation is not only
for base Qwen3-8B: at minimum, run the same observation analysis for base
Qwen3-8B and the four senior-recommended Qwen3-8B-LoRA baselines
TALLRec/OpenP5/DEALRec/LC-Rec, then check whether the base-model phenomena
also appear under stronger baseline systems.

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

## Endgame And Stop Rule

Agents must know when the project/experiment phase can end. Do not keep
inventing vague next steps after the required evidence exists.

Experiment phase can be considered basically complete only when:

- Beauty/books/electronics/movies same-candidate runs are complete at the
  declared scale.
- Base Qwen3-8B and the four senior-recommended Qwen3-8B-LoRA baselines have
  observation analyses, or missing runs are explicitly justified.
- Official-native or clearly labeled controlled baselines have complete
  score/import/evaluation artifacts.
- Ours full and required ablations have complete artifacts under the same
  protocol.
- Metrics include ranking, validity/hallucination/candidate adherence,
  coverage/diversity/novelty, long-tail/popularity slices, efficiency/cost, and
  paired significance where applicable.
- Failure cases and limitations are documented.
- A top-conference-style reviewer pass finds no fatal gaps in novelty,
  fairness, scale, leakage, ablations, or reproducibility.

When these are satisfied, tell the user that the project has reached the
writing-ready stage and the next phase is paper writing/export/positioning,
not more open-ended experimentation. If any item is missing, state exactly
which gate remains and the shortest concrete command or implementation step to
close it.
