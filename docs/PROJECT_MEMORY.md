# TRUCE-Rec Project Memory

This document is the durable memory for future Codex/agent sessions. Read it
before planning nontrivial work. Keep it current after each completed stage so
new agents do not have to reconstruct the project from stale fragments.

Last major update: 2026-06-07.

> **Authoritative setting reference:** `docs/experimental_setting_and_baselines.md`
> (8 domains, 101-candidate protocol, Qwen3-8B, the 8 official baselines, frozen evidence in
> `data/official_baselines/`, per-domain SOTA bar). Read it before any experiment/method/paper work.
>
> **Standing procedure for any future agent:** `docs/CALM_REC_RUNBOOK.md` — how to run experiments and
> write the paper, end to end. The method is designed + implemented; follow the runbook, don't
> re-derive rules.

## Status Snapshot (2026-06-07)

- **Setting:** 8 domains (beauty 973 + books/electronics/movies/sports/toys/home/tools 10k each),
  same-candidate 101 (1 pos + 100 neg), Qwen3-8B backbone, schema
  `source_event_id,user_id,item_id,score`.
- **Official baselines: DONE & FROZEN.** 8 official LLM4Rec methods × 8 domains = 64 pairs,
  lightweight evidence committed at `data/official_baselines/` (master table + per-pair
  metrics/coverage/provenance + sha256 manifest). These are reused as a shared reference and are
  **never re-run**. Beauty SOTA bar = **ProEx NDCG@10 0.1506 / HR@10 0.2528 / MRR 0.1429**;
  the other 7 domains are led by LLMEmb.
- **Ours method: CALM-Rec (LOCKED, ARIS 9.0/10).** The prior `ours_uncertainty_guided` /
  conservative-gate route did **not** beat fallback-only in the R3 formal run
  (`docs/r3_ours_error_decomposition.md`) — retired. A first tri-agent redesign produced SCALR
  (`docs/method_redesign_decision.md`); a follow-up **≥20-iteration** tri-agent upgrade found SCALR
  was only rank-1 personalization (matches but can't beat baselines) and evolved it into **CALM-Rec**
  (Calibrated trust over Attribute-anchored Latent Multi-intent) — see **`docs/method_calm_rec_spec.md`**.
  Headline = endogenous per-user-item *calibrated trust* between an LLM multi-intent personalized
  score and a history-free prior (multi-intent is the vehicle). Scoring core implemented + tested
  (no GPU); real Qwen3-8B encoders + trainer are the remaining no-GPU TODO.
- **Server:** pony-rec-gpu `~/projects/TRUCE-Rec`. **BUSY with another project — run NO experiments
  until the user gives the go-ahead.** First run when unblocked = beauty-first formal Ours run.
- **Follow-up (after the 8-domain performance table):** observation, ablation, hyper-parameter
  analysis + overview figure — planned in `docs/followup_experiment_plan.md`.

## One-Sentence Direction

TRUCE-Rec is a fully independent publishable LLM4Rec research system: it starts
from recommendation-specific uncertainty observations, builds an original
CURE/TRUCE method with its own data pipeline, baselines, and evaluation, and
scales to multiple Amazon domains with same-candidate protocols.

## CRITICAL: Project Independence

TRUCE-Rec 和 Pony/TGL-Rec 的关系：
- 共同点：都是推荐系统论文，共享 8 个外部 baseline（LLM2Rec, LLM-ESR,
  LLMEmb, RLMRec, IRLLRec, ELMRec, ProEx, ProMax），数据 setting 一样
  （Amazon 四域、same-candidate 协议）
- 不同点：方法/framework 完全不同。TRUCE 做 uncertainty-aware generative
  recommendation（title grounding + confidence decomposition + exposure-aware
  calibration），Pony 做 task-grounded uncertainty，TGL-Rec 做 temporal graph

因为 baseline 和数据 setting 相同，baseline 分数可以复用（同一个 baseline 在
同一个数据上跑出来的分数是一样的）。但 TRUCE 的方法代码、framework、实验
pipeline 必须完全独立。

TRUCE-Rec 在服务器上应该有自己的独立部署，不依赖 Pony 的目录结构来生成数据。
数据可以从 Amazon Reviews 2023 原始数据独立准备，或者如果 Pony 已经生成了
相同 setting 的 same-candidate tasks，可以直接复制使用（因为数据 setting 完全
相同，不是"依赖"而是"共享公共资源"）。

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

- **本地是主开发环境**：所有代码编辑、文档更新、git commit/push 都在本地完成。
- **服务器只作为实验场所**：只做 git pull → 跑 GPU 实验 → 产出结果。
  服务器上不改代码、不 commit、不 push。
- **GitHub 更新从本地提交**：永远不从服务器 push。

## Local ↔ Server Alignment Discipline (HARD RULE, 2026-06-07)

1. **本地和服务器必须对齐。** 代码 / configs / docs 两边一致。服务器产出结果后，把**轻量重要
   证据打包回本地**（metrics CSV/JSON、provenance、summaries、manifests —— 不是多 GB 的
   scores/checkpoints/predictions，那些留在服务器）。参照 `data/official_baselines/` 的
   import 模式。
2. **实验在服务器跑，commit/push 从本地。** 绝不从服务器 `git push`；服务器只 git pull + 执行。
   更新 GitHub：先把轻量证据同步回本地，本地 commit，本地 push。
3. **不要停。** 自主跑完整个任务，只有遇到真正的 blocker 才停：**服务器磁盘满**、
   **API/endpoint 故障**、无法修复的 review 驳回，或**用户让你暂停**（如"GPU 被别人占着先别跑"
   —— 当前正是此状态）。否则步骤之间不要停下来问。
4. **周期性持久化。** 每完成一段有意义的工作 → 写/更新 agentmemory、更新本地重要文档、更新
   README。边做边写，不要攒到最后。
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
  -> reused Pony official-qwen3base baseline evidence
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

## Baseline Policy

TRUCE-Rec compares against **8 official LLM4Rec baselines** under a fixed same-candidate setting it
shares with sibling recommendation projects in the group. Because the same official baseline, run
on the same data under the same protocol and backbone, yields identical scores regardless of which
paper consumes them, the baseline numbers are **imported once as a shared frozen reference and
never re-run**. TRUCE-Rec's Ours method is implemented and trained **independently** — no other
project's method code is reused.

- Frozen evidence: `data/official_baselines/` (64 pairs, master table, sha256 manifest, frontier).
- Ours may tune hyper-parameters only via the declared validation protocol; never on test.

The 8 official baselines:
ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLM-ESR, ProEx, ProMax, RLMRec.

Data setting (8-domain same-candidate):
- Domains: beauty (973), books, electronics, movies, sports, toys, home, tools (10k each).
- 1 positive + 100 popularity-sampled negatives per event (101 candidates).
- Same-candidate evaluation and the same evaluator for all methods.
- Qwen3-8B + LoRA as the shared backbone.

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

This advice explains why the Pony/Uncertainty reused official-qwen3base lane is
acceptable. It is no longer an instruction to rerun the local TRUCE controlled
adapter suite; the active implementation policy is the Pony evidence reuse
policy above.

## Legacy Baseline Contract

Historical TRUCE-side compared LLM baselines:

```text
official project implementation
  + Qwen3-8B base model
  + LoRA adaptation
  + official default or reported-optimal baseline hyperparameters
  + shared TRUCE split/candidates/evaluator
  + example_id,user_id,item_id,score
```

Legacy official baseline families:

- TALLRec
- OpenP5
- DEALRec
- LC-Rec
- LLaRA
- LLM-ESR

Current TRUCE-side controlled adapters are legacy pilots. Do not call them the
current paper-facing baseline source. The active source is Pony/Uncertainty
official-qwen3base evidence described above. CoLLM, SLMRec, BIGRec, and other
methods may be follow-up or appendix candidates only if the user explicitly
reopens that lane.

If the legacy lane is explicitly reopened, every local official baseline must
record:

- official repo and commit;
- official config/hyperparameter source;
- official modules/objective reused;
- Qwen3-8B-LoRA compatibility changes;
- score export shim;
- TRUCE import/evaluation artifacts;
- evidence label.

## Data And Experiment Scale

TRUCE-Rec 使用与 Pony 相同的数据 setting（因为是同一个研究组的论文）。

Already prepared datasets (local):
- MovieLens 1M (sanity, pilot, full)
- Amazon Reviews 2023: Beauty (sample_5k, full, cu_gr_v2), Digital Music,
  Handmade, Health, Video Games

Target paper-scale evaluation (四域 same-candidate):
- Domains: Beauty, Books, Electronics, Movies
- Protocol: same-candidate, 1 positive + 100 popularity-sampled negatives
- Scale: up to 10,000 users per domain
- Test history mode: train_plus_valid

数据来源：如果 Pony 项目已经生成了四域 same-candidate tasks（因为 setting
完全相同），可以直接复制使用。否则 TRUCE 用自己的 preprocess 脚本从 Amazon
Reviews 2023 原始数据生成。

Score export schema for all methods:
```text
source_event_id,user_id,item_id,score
```

Never use `test` split for hyperparameter selection.

## Server Operating Model

Server: pony-rec-gpu (125.71.97.70:15302), user ajifang, GPU RTX 4090.
Server repo path: `~/projects/TRUCE-Rec` (to be deployed).

TRUCE-Rec deploys independently on the server. It does NOT share environments,
data directories, or scripts with any other project.

Deployment steps:
1. `git clone git@github.com:appleweiping/TRUCE-Rec.git ~/projects/TRUCE-Rec`
2. Create venv: `python3 -m venv .venv_truce && source .venv_truce/bin/activate`
3. Install: `pip install -e .`
4. For GPU work (observation, LoRA training): use TALLRec's venv which has
   torch/transformers/peft, or install them into .venv_truce

Local Codex does not automatically see server state. Treat server work as a
command-and-log loop:
1. Update/push repo locally.
2. Give user exact server commands.
3. User runs commands on `~/projects/TRUCE-Rec`.
4. User pastes logs/status/errors.
5. Agent diagnoses and updates code/docs if needed.

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

1. **CALM-Rec implementation (mostly DONE, no GPU):** scoring core + ranker + encoders (hashed CPU +
   gated Qwen) + weak-labels + 3-stage trainer scaffold + stage-2.5 gate + runner wiring + configs +
   `scripts/run_calm_rec.py` + `scripts/build_calm_weak_labels.py` + 27 tests — all green, runs
   end-to-end through the official runner/evaluator on CPU. **Only remaining work:** the real Qwen3-8B
   forward in `QwenItemEncoder` + the Qwen+LoRA Stage-B gradient loop (the contract is already
   exercised by the hashed encoders). Procedure: `docs/CALM_REC_RUNBOOK.md` §2.2-2.3.
2. **Beauty-first formal run** (BLOCKED on server availability — user will say when): run CALM-Rec on
   beauty under the frozen protocol, target SOTA (beat ProEx NDCG@10 0.1506). **Priority-1 experiment
   = real ρ vs variance-matched placebo** + the stage-2.5 reliability AUC gate. If not SOTA, re-run
   the tri-agent discussion and iterate on beauty until it is.
3. **Roll out** the validated method to the other 7 domains (same protocol).
4. **Follow-up experiments** (`docs/followup_experiment_plan.md`): observation, ablation,
   hyper-parameter analysis; draw the overview figure.
5. **Paper:** fill the 8-domain main table, run the top-conference reviewer gate, prep submission.

Do NOT re-run any official baseline, re-prepare baseline data, or re-deploy — those are done. The
only open experimental work is TRUCE-Rec's own method.

## Literature Status (updated 2026-05-21)

Novelty confirmed safe. Key competitors and differentiation:
- UGR (2602.11719): uncertainty-weighted reward for Semantic-ID GenRec. TRUCE
  does title grounding + confidence decomposition + exposure-aware calibration.
- Echoes in the Loop (2602.07442): diagnoses feedback-loop risks in LLM RecSys.
  Pure diagnostic, no method. TRUCE provides the solution.
- Uncertainty & Fairness (2602.02582): entropy-based uncertainty + fairness for
  zero-shot LLM rec. Small scale, no title grounding.
- ConfTuner (NeurIPS 2025): tokenized Brier score for LLM verbal confidence.
  Generic QA, not recommendation. TRUCE adapts this idea as RecBrier.

TRUCE's unique position remains:
1. Title-level generative recommendation + catalog grounding
2. Confidence-popularity causal disentanglement
3. Exposure-counterfactual confidence target
4. Uncertainty-guided data triage (not naive pruning)

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

## 2026-06-12 — Qwen encoders + Stage-B implemented; frozen panels converted; queued on GPU

- Branch `feat/calm-qwen-stage-b`: `methods/calm_qwen.py` (frozen Qwen3-8B item encoder
  with fp16 npz cache; intent encoder with K attribute-anchored soft slots, z = c_k +
  clip(W_r u), pi = softmax(g + lambda*E_hist); D_uk deferred gamma=0 recorded in
  artifacts; `torch_calm_scores` differentiable scorer at 1e-9 parity with calm_rec.py).
  `scripts/train_calm_stage_b.py` = full CALMLossSpec loop (L_rank on fixed-rho-0.1 mixed
  score + attr/bal/orth/use/tau; tau annealed 1.5->4; popularity-matched negatives;
  leakage-clean Stage-A stats; artifacts lora/ + extras + anchors + item cache + meta).
- Formal evaluator `scripts/eval_calm_beauty.py`: per-user signals computed ONCE
  (vectorized), Stage-C rho grid + stage-2.5 AUC gate + ladder (raw/full/K1/placebo) +
  paired bootstrap all derived from the cache. (Naive grid = 81x model reruns; pure-python
  scoring at d=4096 was 30h+ CPU — both fixed.) Parity tests green (40 CALM tests total).
- Frozen-protocol data: `convert_frozen_task.py` ran on server ->
  `data/processed/frozen_week8_beauty` (973 test + 973 valid 101-cand panels from the pony
  external_tasks exports + canonical item_metadata.csv/train_interactions.csv; 3578 train
  transitions; 1184 items). Weak labels rebuilt: outputs/calm/beauty_frozen (672/1184
  dominant facets). CRITICAL: uncertainty-llm4rec panels share users/positives but NOT
  candidate sets (verified 0/973) — never evaluate on them.
- Server copy converted from tarball to git checkout (data/ preserved). Python env reused:
  `~/miniconda3/envs/tglrec-lora/bin/python` (runtime only; method code fully independent).
  Sync = git bundle over scp (server cannot reach GitHub).
- QUEUED on GPU (after TGL-Rec's zero-shot pair; `~/projects/gpu_queue_20260612.sh`):
  Stage-B smoke (--max-train 32) -> Stage-B full -> then run eval_calm_beauty.py
  (--sota-ndcg10 0.1506). Release needs all four falsifiability checks; stage-2.5
  AUC <= ~0.55 -> drop the trust headline automatically (spec section 9).
- agentmemory MCP unavailable in this session; durable state recorded here + CONTEXT.md +
  local auto-memory. Store a digest into agentmemory in the next session that has it.

## 2026-06-20 — HEADLINE PIVOT to RankCRC + ARIS citation-audit (conformal core)

- **Headline correction (the "Status Snapshot (2026-06-07)" above is now PARTIALLY STALE):** the
  paper headline is **RankCRC** — distribution-free **risk-controlled selective reliability** for
  recommendation (marginal-expectation CRC bound `μ(λ) ≤ E_B2[α(B2)]` on the served-slice listwise
  rank-risk `E[1−NDCG@k | served]`; frozen-λ B1/B2 split; label-free-at-inference rank-geometry
  selector `g`; falsifiable gain-law). **CALM-Rec is now strictly subordinate** (calibrated-trust
  ablation/supplementary lane), NOT the headline. The contribution is the *reliability* lane, not raw
  ranking SOTA — orthogonal to pony. Canonical RankCRC docs: `refine-logs/RESEARCH_REFINE_TRUCE_*.md`
  (gate PASSED Codex 7/7), `refine-logs/EXPERIMENT_PLAN_TRUCE.md` + `rankcrc_formalization_verified.md`
  (frozen-λ theorem adversarially verified), `paper/{method_rankcrc,related_positioning_rankcrc,
  notation_rankcrc,introduction}.md` (method+theorem dual-reviewed ≥8: Codex 8 + Opus 8).
  beauty CPU validation: guarantee HOLDS 5/5 (`scripts/rankcrc_validate.py`, A/B1/B2/C 4-fold).
- **ARIS citation-audit (conformal core) RESOLVED.** `paper/refs/conformal.bib` = 8 bibliographically-
  VERIFIED entries (Conformal Risk Control ICLR'24; RCPS JACM'21; Learn-then-Test'21; distribution-free
  recsys reliability COPA'23; two-stage risk control for ranked retrieval IJCAI'25; SelectiveNet ICML'19;
  Vovk-Gammerman-Shafer'05; Angelopoulos&Bates gentle intro). Each author/year/venue web-verified — none
  fabricated. `related_positioning_rankcrc.md` keyed to \citep + explicit incrementality defense vs the
  two closest neighbors (Angelopoulos2023 set-coverage recsys; Xu2025 stage-wise ranked-retrieval).
  Committed+pushed branch `design/rankcrc` @433d4d0. PENDING (results phase, after GPU): 8 official-baseline
  + RankCRC empirical-claim cites — verify each, don't fabricate.
- **GPU status:** TRUCE GPU runs (beauty Stage-B → Stage-2.5 gate → 8-domain) remain QUEUED behind
  pony's priority 3-backbone sweep on the shared RTX 4090. All CPU-frontloadable RankCRC work
  (theory, validation, method/related/intro sections, citations) is done.
