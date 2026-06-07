# Project Context — TRUCE-Rec

## Current State (as of 2026-06-07)

| Metric | Value |
|--------|-------|
| GitHub | https://github.com/appleweiping/TRUCE-Rec |
| Stage | Method redesign (prior formal method did not beat fallback; see below) |
| Official baselines | 8 official LLM4Rec methods, frozen evidence in `data/official_baselines/` |
| Setting | 8 domains, 101-candidate (1 pos + 100 neg), Qwen3-8B backbone |
| Domains | beauty(973), books, electronics, movies, sports, toys, home, tools (10k each) |
| Method | UNDER REDESIGN — prior CU-GR/uncertainty-gate route lost to fallback (R3) |
| LLM providers | Qwen3-8B local (server), MockLLM (dev) |
| Python | >=3.10 |
| Server | pony-rec-gpu (`~/projects/TRUCE-Rec`); BUSY with another project — no runs yet |

> **Read first:** [`docs/experimental_setting_and_baselines.md`](docs/experimental_setting_and_baselines.md)
> — authoritative setting, metrics, the 8 official baselines, frozen evidence location, and the
> per-domain SOTA bar (beauty bar = ProEx NDCG@10 0.1506).

## Research Thesis
LLM-based recommenders generate plausible suggestions but lack calibrated confidence.
TRUCE adds:
1. Uncertainty quantification for each recommendation
2. Trustworthy calibration (override calibrator)
3. Preference fusion with uncertainty-weighted signals
4. CU-GR framework for confidence-aware generation

## Evidence Levels (Enforced)
- L0: smoke/mock (development only)
- L1: pilot (small-scale real data)
- L2: diagnostic (targeted analysis)
- L3: controlled adapter pilot
- L4: official-native controlled baseline
- L5: paper-result (full protocol + significance)

## Key Decisions
- Same-candidate protocol shared with TGL-Rec and Uncertainty
- Pony baselines reused (not re-implemented)
- MockLLM for all development; real LLM only for official runs
- No paper claims without L5 evidence
- Four-domain generalization required (not single-domain)

## What's Next

The performance phase = **8 domains × (Ours + 8 official baselines)** under the frozen setting.
Baselines are done (frozen evidence); the gap is **TRUCE-Rec's own method**.

- [ ] **Method redesign** (active): prior uncertainty-gate route lost to fallback-only in the R3
      formal run. Tri-agent ARIS discussion to design a non-stitched, SOTA-targeting method aligned
      to the 8-baseline setting + Qwen3-8B. See `docs/method_redesign_decision.md`.
- [ ] **Beauty-first formal run** (BLOCKED — server busy with another project; run only on user
      go-ahead): bring Ours to SOTA on beauty (beat ProEx NDCG@10 0.1506). If it can't, re-run the
      tri-agent discussion and iterate until beauty is SOTA.
- [ ] Roll the validated method out to the other seven domains.
- [ ] Then the three required follow-up experiments + overview figure (see
      `docs/followup_experiment_plan.md`): observation, ablation, hyper-parameter analysis.
- [ ] Fill paper tables; internal top-conference review gate; submission prep.
