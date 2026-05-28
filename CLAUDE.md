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
8. **实时更新硬规则**：每完成一个阶段、一个 step、一次错误排除、一次贡献，必须立即更新 memory（`D:\research\Vipin's Knowledgebase\memory\`）和项目文档（PROJECT_MEMORY.md, PHASE_HANDOFF.md 等）。不攒着，不跳过。违反等于工作没做。

## Research Direction
Uncertainty-aware generative recommendation:
- LLMs generate recommendations but lack calibrated confidence
- TRUCE adds uncertainty quantification + trustworthy calibration
- Key components: CU-GR framework, uncertainty policy, preference fusion, override calibrator
- Ablation: each component must show independent contribution

## Current Gate (R1 → R2 transition)
- Infrastructure: COMPLETE (evaluator, metrics, baselines, configs, tests)
- Ours method (CURE/TRUCE): IMPLEMENTED, smoke-tested
- Official baselines: 共享 8 个外部 baseline (LLM2Rec, LLM-ESR, LLMEmb, RLMRec, IRLLRec, ELMRec, ProEx, ProMax)，分数可复用
- Six-domain experiments: IN PROGRESS (beauty/books/electronics/movies converted, sports/toys preprocessing)
- Server deployment: COMPLETE (~/projects/TRUCE-Rec on pony-rec-gpu)
- Novelty: CONFIRMED safe (2026-05-21 literature check)
- Paper sections: DRAFT (intro, method, notation, related)
- Qwen3-8B observation: RUNNING on beauty domain (973 examples)

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
