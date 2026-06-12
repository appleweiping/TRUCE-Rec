# Project Context — TRUCE-Rec

## Current State (as of 2026-06-07)

| Metric | Value |
|--------|-------|
| GitHub | https://github.com/appleweiping/TRUCE-Rec |
| Stage | Method locked (CALM-Rec); 8-domain performance buildout |
| Official baselines | 8 official LLM4Rec methods, frozen evidence in `data/official_baselines/` |
| Setting | 8 domains, 101-candidate (1 pos + 100 neg), Qwen3-8B backbone |
| Domains | beauty(973), books, electronics, movies, sports, toys, home, tools (10k each) |
| Method | CALM-Rec (Calibrated trust over Attribute-anchored Latent Multi-intent); implemented (CPU) + tested, wired into runner |
| LLM providers | Qwen3-8B local (server), MockLLM (dev) |
| Python | >=3.10 |
| Server | pony-rec-gpu (`~/projects/TRUCE-Rec`); BUSY with another project — no runs yet |

> **Read first:** [`docs/experimental_setting_and_baselines.md`](docs/experimental_setting_and_baselines.md)
> — authoritative setting, metrics, the 8 official baselines, frozen evidence location, and the
> per-domain SOTA bar (beauty bar = ProEx NDCG@10 0.1506).
>
> **Future agents — run experiments + write the paper:** [`docs/CALM_REC_RUNBOOK.md`](docs/CALM_REC_RUNBOOK.md).
> Method (CALM-Rec) is designed + implemented; follow the runbook, don't re-derive rules.

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

- [x] **Method locked + implemented: CALM-Rec** — evolved from SCALR over a ≥20-iteration tri-agent
      upgrade (ARIS 9.0/10). Spec `docs/method_calm_rec_spec.md`; procedure `docs/CALM_REC_RUNBOOK.md`.
- [x] **Real Qwen3-8B encoders + Stage-B loop implemented** (2026-06-12, branch
      `feat/calm-qwen-stage-b`): `calm_qwen.py` runtime (frozen item encoder w/ fp16 cache; intent
      encoder w/ K anchored soft slots; differentiable scorer at 1e-9 parity vs the python core);
      `scripts/train_calm_stage_b.py` (full CALMLossSpec gradient loop);
      `scripts/eval_calm_beauty.py` (cached-signal formal evaluator: Stage-C grid + 2.5 gate +
      ladder + placebo + paired bootstrap without per-grid-point model reruns).
      D_uk salience deferred (γ=0, recorded in artifacts). 40 CALM tests green.
- [x] **Frozen-protocol data converted** (server `data/processed/frozen_week8_beauty`): 973 test +
      973 valid 101-cand panels from the pony external_tasks exports (the exact frozen sets the
      baselines scored), 3578 train transitions, 1184 items; weak labels rebuilt (672/1184 dominant
      facets). WARNING: uncertainty-project panels share positives but NOT candidate sets — never
      evaluate on those.
- [ ] **Beauty formal run** — queued on server (`~/projects/gpu_queue_20260612.sh`): after TGL's
      zero-shot pair → Stage-B smoke → Stage-B full → `eval_calm_beauty.py`. Beat ProEx NDCG@10
      0.1506 with falsifiability checks passing; if not SOTA → tri-agent redesign loop.
- [ ] Roll the validated method out to the other seven domains.
- [ ] Then the three required follow-up experiments + overview figure (see
      `docs/followup_experiment_plan.md`): observation, ablation, hyper-parameter analysis.
- [ ] Fill paper tables; internal top-conference review gate; submission prep.
