# Project Context — TRUCE-Rec

## Current State (as of 2026-05-19)

| Metric | Value |
|--------|-------|
| GitHub | https://github.com/appleweiping/TRUCE-Rec |
| Commits | 115 |
| Stage | Gate R1 — four-domain buildout |
| Official baselines | Reusing from Pony/Uncertainty (same-candidate protocol) |
| Datasets | 4 domains (Beauty, Books, Electronics, Movies) |
| Method | CURE/TRUCE (uncertainty-aware generative recommendation) |
| LLM providers | MockLLM (dev), OpenAI-compatible (prod), HF local |
| Python | >=3.10 |
| Test count | ~96 (70 unit + 6 smoke + 20 misc) |

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
- [ ] Run four-domain experiments at paper scale
- [ ] Collect L5 evidence for main claims
- [ ] Ablation study (each component independently)
- [ ] Statistical significance testing
- [ ] Fill paper tables
- [ ] Internal review gate
- [ ] Submission preparation
