# TRUCE-Rec ARIS research-refine — 2026-06-18

**Status:** ARIS research-refine of the EXISTING (designed + beauty-CPU-validated) TRUCE method, run through the formal ARIS gate (Codex GPT-5.5 xhigh: novelty + feasibility each ≥7). This is NOT a from-scratch redesign (unlike TGL/PaRC) — the method is designed and beauty CPU validation is done; this gate stress-tests the framing + the honest pivot contingency before the GPU rollout.

## Core direction (docs/RESEARCH_IDEA.md)
Uncertainty-aware generative recommendation: when an LLM scores/generates a recommendation, does its uncertainty signal explain *when* the recommendation is reliable — and can we turn that into a **certified reliability guarantee** rather than just better ranking? This is a recommendation-specific reliability problem, not generic QA calibration, and explicitly NOT a generic LLM reranker (that lane is pony's).

## The contribution — honest dual, headline = RankCRC
TRUCE's contribution is **certified selective reliability**, not raw-ranking SOTA. (Raw same-candidate NDCG@10 on beauty is 0.1143 < the ProEx bar 0.1506 — TRUCE adds a *guarantee on the served slice*, not raw ranking power. We state this inequality openly.)

**(A) RankCRC (headline, beauty-CPU-VALIDATED) — Distribution-Free Risk-Controlled Selective Recommendation.** Given a base ranker + a per-user **label-free** confidence estimator `g`, abstain on low-confidence users so that the *served* slice satisfies a finite-sample, distribution-free guarantee `E[1 − NDCG@k | served] ≤ α` via conformal risk control. Four novel pieces:
1. The controlled quantity is **listwise rank-risk** (`1 − NDCG@k`), not pointwise loss (vs standard conformal prediction).
2. The guarantee is on the **served slice after abstention** (selective recommendation), not the full population.
3. `g` is **label-free** — a logistic model over 24 rank-geometry features (score margins, softmax peakiness, responsibility-entropy order-stats, MC-dropout variance, popularity) with `H_target` FORBIDDEN ⇒ fully deployable (no test-label peek). Beauty: test AUC 0.627 > H_top baseline 0.519.
4. **Selective-Gain Decomposition (Prop 3):** closed-form `ΔNDCG@k ≈ (2·AUC − 1)·Gini·κ(c)` predicting the served-slice lift from the confidence-AUC, score-dispersion Gini, and coverage.
Beauty validation (CPU, paper-labeled): the CRC guarantee **holds in expectation across all 5 target coverages over 200 resplits**; the (risk,coverage) Pareto has 18 non-crossing points; one finite-sample edge point (60% coverage) is within bootstrap CI. Reported honestly as "holds in expectation; finite-sample edge noise at small served-n", NOT "always holds".

**(B) CALM-Rec (contingent) — Calibrated trust over Attribute-anchored Latent Multi-intent.** Endogenous per-user-item calibrated trust ρ between an LLM multi-intent relevance score and a history-free prior, using intent-mixture geometry (responsibility entropy + ensemble variance) as the reliability signal. **Risk flagged honestly:** on the K=2 CPU proxy the reliability signal is WEAK (responsibility-entropy AUC 0.351 ≈ random; variance AUC 0.576). **Pre-registered contingency:** the GPU beauty Stage-B (real Qwen3-8B encoders, K=4) tests whether the reliability signal is real (Stage-2.5 gate AUC>0.6); if NOT, ρ is demoted to diagnostic and the headline is **RankCRC-only** (selective reliability), keeping the K-intent core with a fixed blend.

## Novelty vs closest work (differentiation, not A+B stitching)
- **Conformal prediction / risk control (Angelopoulos, Bates, et al.):** pointwise loss, full-population coverage. RankCRC controls *listwise rank-risk on the served slice* with a *label-free* confidence — a new controlled quantity + selective framing for recommendation.
- **SelectiveNet / selective classification:** no distribution-free finite-sample guarantee; RankCRC adds the CRC certificate + the closed-form gain law.
- **LLM-uncertainty calibration (verbalized confidence, ECE):** TRUCE makes NO ECE/reliability-diagram claim; the object is a *served-slice risk certificate*, a different and stronger operational guarantee.

## Feasibility
Beauty CPU validation DONE (scripts/rankcrc_validate.py; artifacts committed @design/rankcrc bfc9c94). Remaining = GPU beauty Stage-B (Qwen3-8B + LoRA K=4) + 8-domain rollout + ablations + observation, all full-scale (8 domains, 10k users/beauty 973, 101 cand, 8 official baselines frozen, paired Holm-bootstrap). GPU-queued behind pony (priority) + tgl. Compute: per the CALM_REC_RUNBOOK.

## Pre-registered decision gates / pivots (honest)
- Beauty Stage-2.5: reliability AUC>0.6 AND CALM full ≥ ProEx 0.1506 ⇒ CALM-Rec headline + 8-domain. ELSE ⇒ **RankCRC-only** headline (selective risk-control), CALM core kept with fixed blend.
- RankCRC: if the finite-sample guarantee fails on more domains beyond expectation, report as expectation-guarantee + widen the calibration fold; do NOT claim always-holds.

## Kill-argument (strongest reviewer objection) + answerability
"RankCRC is conformal risk control (known) applied to NDCG — incremental; and the raw ranker is below SOTA, so why care?" **Answer:** (i) the *combination* is new — listwise rank-risk + served-slice selective guarantee + a **label-free, deployable** confidence over rank-geometry (no test peek) + a closed-form gain law that *predicts* the lift; (ii) the value is orthogonal to raw SOTA: a certified "serve only when E[1−NDCG@k]≤α" is operationally valuable even when raw ranking is mid-pack — it is a *reliability* contribution, and we position it as such, not as a ranking-SOTA claim. If a reviewer still reads it as incremental conformal, the fallback is to strengthen the decomposition-law theory + add the cross-domain certificate as the empirical novelty.

## Next ARIS step
On gate PASS (both ≥7): experiment-plan (M-ladder: GPU beauty Stage-B → Stage-2.5 reliability gate → decision → 8-domain RankCRC certificate + CALM ablations + observation; ≥20-seed paper rows; compute). GPU runs queued behind pony.
