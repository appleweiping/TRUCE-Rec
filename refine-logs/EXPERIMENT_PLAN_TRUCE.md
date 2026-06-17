# TRUCE-Rec ARIS experiment-plan — RankCRC (Risk-Controlled Selective Recommendation) — 2026-06-18

**Gate context:** research-refine PASSED (Codex GPT-5.5 xhigh v2 = NOVELTY 7 / FEASIBILITY 7, both ≥7; artifact `RESEARCH_REFINE_TRUCE_2026-06-18.md` @design/rankcrc 86b78fb). Codex's residual TOP_BLOCKER was **empirical, not framing**: "demonstrate non-trivial cross-domain served coverage at useful α while preserving the formally stated frozen-selector marginal served-slice RankCRC guarantee" — plus four concrete carry-overs folded into this plan: (a) write the served-slice theorem carefully (RANDOM served-calibration size + frozen-selector conditioning); (b) strict no-leakage audit of the 3-fold split; (c) non-trivial coverage at useful α across 8 domains = the MAIN empirical gate; (d) label-free confidence transfer / cross-ranker table. This plan is written to clear the experiment-plan gate (Codex ≥6 on EVIDENCE / RIGOR / GATES / FEASIBILITY / PAPER_POTENTIAL) and pre-empts the test-leakage GATES failure mode (all proceed/kill + α/τ/λ decisions on calibration/validation folds; official TEST scored once, no post-selection).

## Headline
**RankCRC — distribution-free risk-controlled selective recommendation.** Wrap ANY base ranker with a per-user **label-free** confidence `g` and abstain on low-confidence users so the *served* slice satisfies the **Conformal-Risk-Control marginal-expectation guarantee** `E[ 1 − NDCG@k | served ] ≤ α` (Angelopoulos–Bates 2024), where the expectation is over the exchangeable calibration+test draw conditional on a frozen selector. Contribution = **certified-class reliability** (a guarantee + a deployable label-free selector + a falsifiable gain law), NOT raw-ranking SOTA (raw beauty NDCG@10 0.1143 < ProEx bar 0.1506 — stated openly). CALM-Rec is a strictly subordinate, gated base-ranker upgrade.

## The served-slice guarantee — precise statement (Codex point a)
Let users be exchangeable. Fix a frozen selector `(g, τ)`: user `u` is *served* iff `g(x_u) ≥ τ`, where `x_u` are label-free rank-geometry features (NO H_target). On the **served** sub-population, the per-user risk `R_u = 1 − NDCG@k(u) ∈ [0,1]` is bounded+monotone in the CRC tuning parameter. CRC tunes `λ̂` on the served calibration users so that
  `E[ R(λ̂) | served ] ≤ α` ,
the expectation taken over the **joint** draw of (served calibration, served test). Two subtleties we state explicitly and handle:
1. **The served calibration size `n_s` is RANDOM** (it is the number of calibration users with `g ≥ τ`). CRC's finite-sample bound is applied with the *realized* `n_s` (the `(n_s+1)/n_s · α` style correction), and the guarantee is **marginal over the joint randomness of which users are served**, not conditional on a fixed `n_s`. We do NOT claim a per-`n_s` conditional bound.
2. **Conditioning on the frozen selector.** Because `(g, τ)` are frozen on a *separate* fold (below), "served" is a deterministic function of frozen features; served calibration and served test users remain exchangeable, so CRC validity transfers to the served sub-population. The claim is **marginal-expectation, conditional on the frozen selector** — explicitly NOT PAC/high-probability (RCPS) and NOT per-user conditional.

## 3-fold split discipline + no-leakage audit (Codex point b)
Per domain, partition users into **three disjoint folds** (committed manifest, seed 20260506):
- **Fold A — confidence-fit:** fit `g` (logistic over the 24 rank-geometry features) and choose the abstention threshold `τ` for each target coverage `c`. Uses Fold-A labels ONLY to train `g`'s logistic target (a label-free *feature* set; the training *signal* is whether the served set's risk is low — see weak-label note). `g` never sees Fold-B/C.
- **Fold B — CRC calibration:** on Fold-B served users (`g ≥ τ`), compute the CRC `λ̂` controlling `E[1−NDCG@k|served] ≤ α`.
- **Fold C — TEST:** apply frozen `(g, τ, λ̂)`; report served-slice risk + coverage. Scored ONCE.
- **No-leakage automated audit (committed as a test):** (i) `g`'s feature matrix asserted to exclude any H_target-derived / test-label column (the `H_target FORBIDDEN` invariant from the design, enforced by an explicit column-allowlist check); (ii) fold index sets asserted pairwise-disjoint; (iii) `τ, λ̂` provenance asserted to come only from Folds A/B (no test users in their fit); (iv) adaptive re-fitting of `g` on Fold B is forbidden (would break exchangeability) — asserted by a frozen-hash check on `g`'s coefficients between calibration and test.

## Frozen protocol (shared with pony/tgl — non-negotiable, full-scale)
8 Amazon domains, 10k users (beauty 973), 101 same candidates/event (1 pos + 100 popularity-matched neg), Qwen3-8B base, metrics HR@5/10/20 + NDCG@5/10/20 + MRR, paired Holm-corrected bootstrap. 8 official baselines frozen in `data/official_baselines/`. RankCRC wraps a base ranker; the base ranker is CALM-Rec **iff** Stage-2.5 passes, else the fixed-blend/pony scorer (the headline does not depend on which).

## Experiment blocks

### Block 0 — M0: beauty GPU Stage-B + Stage-2.5 reliability gate (run FIRST; GPU; smallest domain)
- **Stage-B:** real Qwen3-8B encoders + LoRA, K=4 intents (per `docs/CALM_REC_RUNBOOK.md`) → produce CALM base-ranker scores + the reliability signal (responsibility entropy + ensemble variance) on beauty.
- **Stage-2.5 reliability gate (decides base ranker, NOT the headline):** reliability AUC > 0.6 (the CPU K=2 proxy was weak: resp-entropy 0.351, variance 0.576 — Stage-B with real encoders + K=4 is the honest test). PASS ⇒ CALM is the base ranker + a co-headline; FAIL ⇒ CALM demoted to diagnostic/ablation, base ranker = fixed-blend/pony. **RankCRC proceeds either way.**
- **RankCRC headline kill-gate (validation/Fold-B only, official beauty TEST untouched here):** on beauty, with the label-free `g`, the marginal-expectation bound must hold (mean served-risk over resplits ≤ α at the pre-registered α grid) AND served coverage at ≥1 useful α must be non-trivial (def. below). If the bound is violated in expectation OR coverage is ~0 at every useful α with the GPU base ranker → the selective framing fails on its easiest domain → **escalate to a fresh research-refine round** (do NOT scale to 8 domains). (CPU validation already showed the bound holds + g AUC 0.627, so this is a confirmation gate, not an expected kill.)

### Block 1 — RankCRC core on beauty (GPU re-confirmation of the CPU result)
Reproduce the CPU-validated result with the real Qwen3-8B base ranker: (risk, coverage) Pareto (target: monotone, non-crossing); mean served-risk ≤ α across the α grid over ≥20 resplits; `g` test AUC vs the H_top score-margin baseline (0.519). Evidence label: controlled → official.

### Block 2 — 8-domain feasibility triad (GPU) — THE MAIN EMPIRICAL GATE (Codex point c)
For all 8 domains, with the 3-fold discipline:
1. **Cross-domain risk control:** mean served-risk over resplits ≤ α at every target coverage. Reported per domain; the *guarantee* claim requires it to hold on the clear majority (pre-registered: ≥6/8 in expectation; domains where it fails reported honestly with the realized gap).
2. **Useful operating points (the gate):** pre-registered **"useful α"** = an α at which served NDCG@k exceeds the serve-all population NDCG@k by a pre-set absolute margin (≥ +0.01 NDCG@10) AND served coverage ≥ a non-triviality floor (pre-registered **≥30% of users**). GATE = a useful operating point exists on **≥6/8 domains**. Coverage–risk Pareto reported for every domain; any domain forcing near-zero coverage at useful α is flagged, not hidden.
3. **Label-free confidence transfer (Codex point d):** an **AUC transfer matrix** — fit `g` on domain X, apply to domain Y (8×8 cross-domain) — plus **cross-ranker transfer** (fit `g` on base-ranker R1's geometry, apply to R2). Claim = "risk control is maintained wherever `g` ≥ chance"; where `g` collapses to chance, RankCRC degrades gracefully to serve-all at the population risk (stated honestly). This is the robustness answer to the modest beauty AUC 0.627.

### Block 3 — Gain-law (Prop 3) falsification (Codex point 4 carry-over)
`ΔNDCG@k ≈ (2·AUC − 1)·Gini·κ(c)` is pre-registered as a **predictive law**: AUC, Gini, κ computed on Fold-A/B (held-out), the predicted ΔNDCG compared to the observed Fold-C ΔNDCG across all 8 domains × all coverage levels. Pre-set falsification bar: out-of-sample **R² ≥ 0.5** ⇒ reported as a predictive law; below ⇒ demoted to a heuristic. **No coefficients fit to test risk.** The law is a *secondary* contribution; the headline does not rest on it.

### Block 4 — Selective-recommendation baselines (the right comparison class)
RankCRC is NOT a ranking-SOTA play, so the comparison is against selective alternatives at matched coverage: (1) **serve-all** (no abstention, population risk); (2) **random abstention** (coverage-matched); (3) **score-margin-only confidence** (H_top, AUC 0.519); (4) **SelectiveNet-style learned abstention WITHOUT the CRC guarantee** (shows the guarantee is the contribution, not just "abstain on hard users"); (5) **oracle label-aware confidence** (upper bound on what any selector could achieve). Claim: RankCRC alone delivers the risk guarantee with a *label-free* selector that (3)/(4) cannot certify, at coverage competitive with the oracle. Paired Holm-bootstrap.

### Block 5 — Robustness / reproducibility
**What is seeded (≥20 seeds for paper-result rows):** the CRC guarantee is MARGINAL over the calibration+test draw, so the ≥20 seeds = **≥20 resplits of users into the 3 folds** (the exact randomness the guarantee is over) — report mean served-risk ± bootstrap CI over resplits. The LLM forward is deterministic (vLLM greedy, temp→0): NO model-randomness averaging. Also seeded: `g`-training bootstrap (AUC CIs), the popularity-matched negative draw. Sensitivity: α-grid density, coverage grid, K (CALM intents) where applicable.

## Baselines (≥8 satisfied for the rollout context)
8 official (ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLM-ESR, ProEx, ProMax, RLMRec) provide the population NDCG context (RankCRC's serve-all point sits among them); the *method* comparison is the 5 selective baselines in Block 4. Total reported methods ≥ 13.

## Milestones + decision gates
- **M0** (beauty Stage-B + Stage-2.5 + RankCRC confirmation kill-gate): decides base ranker (CALM vs fixed-blend) + confirms the selective framing holds on the easiest domain. ~1–2 GPU-days (Qwen3-8B encoders + LoRA on 973 beauty users + CPU-cheap CRC/g).
- **M1** (8-domain feasibility triad): GATE = useful operating point on ≥6/8 + risk control in expectation on ≥6/8 + transfer ≥ chance on the majority. ~3–5 GPU-days (base-ranker scores per domain; CRC/g are CPU-cheap once scores exist).
- **M2** (gain-law falsification + selective baselines + robustness, ≥20 resplits): paper-ready, all evidence-labeled.
- **M3** (paper-write → auto-review-loop ≥8 → citation-audit → paper-claim-audit).

## Compute & timeline (1×RTX4090, GPU-queued behind pony's 3-backbone run, then tgl M0)
The GPU cost is the **base-ranker scoring** (Qwen3-8B over 8×10k×101); CRC + the label-free `g` are CPU-cheap (logistic + sort over precomputed scores), so once base scores exist the entire selective layer + 8-domain triad + ≥20 resplits run on CPU in hours. M0 ~1–2 GPU-days; full ~3–5 GPU-days. The reuse opportunity: if RankCRC's base ranker = pony/fixed-blend, the base scores may reuse the **existing pony Qwen pointwise scores** (the α-anchor), collapsing most GPU cost — confirm during experiment-bridge. CPU/design (this plan, the 3-fold harness, the no-leakage audit, the gain-law fitter) proceeds now in parallel.

## Evidence discipline
Labels smoke→pilot→diagnostic→controlled→official→paper-result; only paper-result rows enter the paper; significance required for every claim; configs+seeds+fold-manifests committed; large artifacts server-side with manifests; light evidence to git.

## Pre-registered KILL / pivot (repeat)
- Stage-2.5 reliability AUC ≤ 0.6 ⇒ CALM demoted, RankCRC-only headline (NOT a project kill — pre-registered honest pivot).
- RankCRC headline kill ⇒ ONLY if the marginal-expectation bound is violated in expectation OR no useful operating point exists on the majority of domains ⇒ escalate to a fresh research-refine round; the characterized finding (label-free rank-geometry confidence does/doesn't carry selective signal across domains) is a separate, honestly-labeled diagnostic contribution.
