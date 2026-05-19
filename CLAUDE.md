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
4. Pony official baselines are REUSED (shared same-candidate protocol)
5. Four domains: Beauty, Books, Electronics, Movies
6. MockLLM for development; real LLM (API/HF) for official runs only
7. Follow gate system: no advancement without gate criteria met

## Research Direction
Uncertainty-aware generative recommendation:
- LLMs generate recommendations but lack calibrated confidence
- TRUCE adds uncertainty quantification + trustworthy calibration
- Key components: CU-GR framework, uncertainty policy, preference fusion, override calibrator
- Ablation: each component must show independent contribution

## Current Gate (R1)
- Infrastructure: COMPLETE (evaluator, metrics, baselines, configs, tests)
- Ours method (CURE/TRUCE): IMPLEMENTED, smoke-tested
- Official baselines: REUSING from Pony/Uncertainty project
- Four-domain experiments: NOT YET RUN at paper scale
- Paper sections: DRAFT (intro, method, notation, related)

## Agent Roles
- **Codex**: Primary execution engine, server commands, parallel experiment runs
- **Claude/Opus**: Architecture review, paper writing, complex reasoning, claim verification
- **OpenCode**: Implementation, testing, doc updates
