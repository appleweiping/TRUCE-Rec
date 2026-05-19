# OPENCODE.md — TRUCE-Rec

Read `AGENTS.md` first. That is the authoritative contract.

## Context
TRUCE-Rec: uncertainty-aware generative recommendation with trustworthy calibration.
The system adds calibrated confidence to LLM-generated recommendations.

## Your Role
Implementation, testing, documentation updates.

## Quick Commands
- Run unit tests: `python -m pytest tests/unit/ -x`
- Run smoke tests: `python -m pytest tests/smoke/ -x`
- Lint: `ruff check src/ scripts/`
- Run Ours smoke: `python scripts/run_ours_smoke.py`
- Evaluate: `python scripts/evaluate.py --config configs/evaluation/default.yaml`

## Key Paths
- Active code: `src/llm4rec/` (data, methods, metrics, evaluation, prompts, trainers, analysis)
- Legacy: `src/storyflow/` (observation, providers, server)
- Scripts: `scripts/` (90+ files)
- Configs: `configs/`
- Paper: `paper/`
- Docs: `docs/` (70+ files)

## Current State
- Gate R1: server-first four-domain buildout
- Ours method: implemented + smoke-tested
- Official baselines: reusing from Pony
- Paper-scale experiments: NOT YET RUN
- No approved paper-result evidence yet
