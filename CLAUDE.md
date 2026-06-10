# CLAUDE.md — TRUCE-Rec

You are working on TRUCE-Rec: Uncertainty-Aware Generative Recommendation with Trustworthy Calibration.

## Mandatory Read Order
1. `AGENTS.md` — authoritative engineering contract (666 lines)
2. `README.md` — project documentation and current gate
3. `docs/RESEARCH_IDEA.md` — core research direction
4. `docs/PROJECT_MEMORY.md` — durable agent memory
5. `docs/submission_roadmap.md` — milestone ladder
6. This file

## Quick Context
- GitHub: https://github.com/appleweiping/TRUCE-Rec
- Stage: Gate R1 — server-first four-domain buildout
- Core code: `src/llm4rec/` (active), `src/storyflow/` (legacy)
- Configs: `configs/` (datasets, baselines, experiments, methods, evaluation, training, llm)
- Tests: `tests/unit/` (~70) + `tests/smoke/` (~6)
- Paper draft: `paper/` (introduction, method, notation, related work)

## Critical Rules
1. Never fabricate experiment results or claim unverified improvements
2. Evidence labeling is mandatory: smoke/mock → pilot → diagnostic → controlled → official → paper-result
3. No "paper-result" label without full controlled experiment + significance test
4. TRUCE-Rec 与 Pony/TGL-Rec 共享 8 个外部 baseline 和数据 setting，但方法/framework 完全独立
5. Four domains: Beauty, Books, Electronics, Movies
6. MockLLM for development; real LLM (API/HF) for official runs only
7. Follow gate system: no advancement without gate criteria met
8. **实时更新硬规则**：每完成一个阶段、一个 step、一次错误排除、一次贡献，必须立即更新 memory（agentmemory MCP）和项目文档（PROJECT_MEMORY.md, experimental_setting_and_baselines.md 等）。不攒着，不跳过。违反等于工作没做。
9. **本地↔服务器对齐硬规则**：代码/configs/docs 两边一致；实验在服务器跑，commit/push 只从本地；服务器产出后把轻量证据（metrics/provenance/manifests，非多 GB 文件）打包回本地。除非服务器磁盘满 / API 故障 / 用户喊停，否则不要停。详见 `docs/PROJECT_MEMORY.md` 的 "Local ↔ Server Alignment Discipline"。
10. **8 域 8 baseline 已冻结**：官方 baseline 证据在 `data/official_baselines/`（64 对，read-only，永不重跑）。setting/metrics/baselines/SOTA bar 一律以 `docs/experimental_setting_and_baselines.md` 为准。

## Research Direction
Uncertainty-aware generative recommendation:
- LLMs generate recommendations but lack calibrated confidence
- TRUCE adds uncertainty quantification + trustworthy calibration
- Key components: CU-GR framework, uncertainty policy, preference fusion, override calibrator
- Ablation: each component must show independent contribution

## Current Gate (method redesign → beauty-first SOTA)
- Infrastructure: COMPLETE (evaluator, metrics, baselines, configs, tests)
- Official baselines: **COMPLETE & FROZEN** — 8 official methods × 8 domains (64 pairs) in
  `data/official_baselines/`, reused as shared reference, never re-run.
- Setting: 8 domains (beauty 973 + 7 domains @10k), 101-candidate, Qwen3-8B. SOTA bar per domain in
  `docs/experimental_setting_and_baselines.md` (beauty bar = ProEx NDCG@10 0.1506).
- Ours method: **CALM-Rec (LOCKED, ARIS 9.0/10), implemented (CPU) + tested.** Calibrated trust over
  Attribute-anchored Latent Multi-intent. Headline = endogenous per-user-item calibrated trust between
  an LLM multi-intent score and a history-free prior. Spec: `docs/method_calm_rec_spec.md`. Code:
  `src/llm4rec/methods/calm_{rec,encoders,weak_labels,trainer}.py` (+27 tests, runs through the
  official runner). Only remaining: real Qwen3-8B encoder forward + Stage-B LoRA loop.
  **Any agent: follow `docs/CALM_REC_RUNBOOK.md` for experiments + paper — don't re-derive rules.**
- Server: pony-rec-gpu, **BUSY with another project — no runs until user says go**. First run =
  beauty-first formal Ours run.
- Follow-up after performance table: observation / ablation / hyper-parameter analysis + overview
  figure (`docs/followup_experiment_plan.md`).

## Server Access

Remote GPU server `pony-rec-gpu`:
- **SSH command**: `ssh pony-rec-gpu` (or `ssh -p 15302 ajifang@125.71.97.70`)
- **GPU**: NVIDIA RTX 4090 (49GB VRAM)
- **TRUCE-Rec server path**: `~/projects/TRUCE-Rec` (待部署)
- **Local project path**: `D:\Research\TRUCE-Rec`

TRUCE-Rec 在服务器上独立部署，不依赖其他项目的目录结构。

## Agent Roles
- **Codex**: Primary execution engine, server commands, parallel experiment runs
- **Claude/Opus**: Architecture review, paper writing, complex reasoning, claim verification
- **OpenCode**: Implementation, testing, doc updates
