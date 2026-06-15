# RankCRC — Method Spec (TRUCE-Rec converged design)

**RankCRC = Distribution-Free Risk-Controlled Selective Recommendation via listwise rank-risk
Conformal Risk Control over a label-free confidence estimator.**

**Date:** 2026-06-15 · **Status:** DESIGN CONVERGED · beauty milestone validated on CPU (this doc
+ `scripts/rankcrc_validate.py`). **Supersedes** the CALM-Rec "calibrated personalization-trust"
headline (`docs/method_calm_rec_spec.md`): the ρ trust-gate is *inert* for same-candidate ranking
(full+gate NDCG@10 0.1004 < raw 0.1143), so trust-mixing is demoted to a diagnostic. The deployable,
defensible contribution is **a finite-sample risk certificate for selective recommendation**, built
on the SASRec-over-frozen-Qwen scorer (raw, no trust-mix) and a label-free confidence estimator.

The base scorer and the calibrator's features are *shared infrastructure* with CALM-Rec; what is new
here is (a) the conformal **risk-control layer** that turns a confidence score into a served-slice
guarantee, (b) the **joint (risk, coverage) Pareto** view, and (c) a **Selective-Gain Decomposition**
that explains *why* selection helps in closed form. The method is **model-agnostic**: any scorer that
emits per-user candidate scores can be certified, so baselines can be certified on the same axis.

---

## 0. Headline contribution (one sentence)

> RankCRC issues a **distribution-free, finite-sample guarantee** that the recommendations *actually
> served* (after abstaining on low-confidence users) have listwise rank-risk at most a user-chosen
> level α — `E[1 − NDCG@k | served] ≤ α` up to an `O(1/n)` conformal correction — using a *label-free*
> confidence estimator and **no distributional assumptions** on the scorer, the data, or the
> confidence model.

Selective recommendation, conformal prediction, risk-coverage curves, SelectiveNet, and Conformal
Risk Control (CRC) all pre-exist. RankCRC's novelty is the **synthesis pointed at recommendation**:
(i) the controlled quantity is a *listwise ranking* risk `1 − NDCG@k`, not a pointwise 0/1 loss, and
the guarantee is on the **served slice** (the deployed subpopulation after abstention) rather than the
full population; (ii) the confidence estimator that defines the served slice is **label-free** and
trained on calibration data only; (iii) we give a closed-form **decomposition** that predicts the
selective NDCG gain from the confidence AUC, the rank-risk Gini, and a coverage factor. The honest
framing is: this is **selective reliability + a risk certificate**, NOT a new full-coverage SOTA scorer.

---

## 1. Setup and notation

Same-candidate protocol (frozen week-8 setting): per user `u` a panel of `C = 101` candidates
(1 held-out positive + 100 popularity-sampled negatives). The base scorer emits a score vector
`s_u ∈ R^C`; ranking is `argsort(−s_u)`. Let `r_u ∈ {0,…,C−1}` be the 0-based rank of the positive.

**Listwise rank-risk (the controlled loss).** For cutoff `k` (default `k=10`),

```
L_u = 1 − NDCG@k(u) = 1 − 1{r_u < k} / log2(r_u + 2)  ∈ [0, 1].
```

`L_u = 0` iff the positive is ranked first; `L_u = 1` iff the positive falls outside the top-k.
This is a *bounded* loss (`B = 1`), which is exactly what CRC requires.

**Splits.** Train (3578 panels) fits the scorer; a held-out **calibration** split `D_cal` of `n` users
and a disjoint **test** split of `m` users are used for the certificate. In the beauty milestone the
existing artifacts give `n = 973` (val→calibration) and `m = 973` (test), all label-bearing only on
the calibration side at fit time.

**Exchangeability.** The certificate requires only that `(features_u, L_u)` for `u ∈ D_cal ∪ {test
user}` are *exchangeable* (i.i.d. is sufficient). No parametric assumption on the scorer, the loss
distribution, or the confidence model. Users are independent draws from the population, so this holds.

---

## 2. The label-free confidence estimator g (frozen at calibration)

`g(u) ∈ [0,1]` estimates the probability that user `u`'s top-1 recommendation is correct, from
**rank-geometry features of the score vector only** — never the label, never `H_target`:

- **score margins**: `s_top1`, `s_top1 − s_top2`, `s_top1 − s_top3`, score mean/std/spread;
- **softmax peakiness**: `softmax(s)` mass on the winner `sm_p1`, winner-vs-runner-up gap, softmax
  entropy over the 101 candidates;
- **responsibility-entropy order-statistics**: `H_top` (entropy at the winner), `H_min/max/mean/std`,
  and `H_rank_of_top` (how many candidates are more confident than the winner) — H is the
  intent-responsibility entropy from the scorer's own mixture geometry;
- **MC-dropout variance**: `Var_top`, `Var_mean`, `Var_max` of the score under M user-side dropout
  passes;
- **argmax log-popularity**: `logpop_top1`, `n_i_top1`, panel mean log-pop, `top1_is_most_popular`,
  `pop_rank_of_top1`.

`g` is a logistic regression (or GBT) **fit on the calibration split only**, predicting the binary
`1{top-1 correct}`, then **isotonic-recalibrated** on the calibration split so `g` is a monotone,
probability-valid confidence score. **`H_target` (entropy at the positive's slot) is FORBIDDEN as a
feature** — it peeks at the label; it is reported only as a leaking AUC ceiling. On beauty, label-free
`g` reaches test AUC ≈ 0.69 (LogReg) / 0.73 (GBT) vs the `H_top`-only baseline 0.52, with VAL 5-fold
OOF ≈ test (genuine generalization, not calibration overfit).

`g` is the *abstention signal*: at serve time we keep users with high `g` and abstain (defer to a
fallback channel) on low `g`. **`H_target` is FORBIDDEN throughout RankCRC** — the entire pipeline is
deployable.

---

## 3. RankCRC — the risk-control layer (the core method)

We want: choose how many users to serve so the **served-slice** rank-risk is provably ≤ α. We control
this by a single threshold `λ` on the confidence `g`: serve `u` iff `g(u) ≥ λ`. Larger `λ` ⇒ fewer,
more-confident users served ⇒ lower risk but lower coverage.

### 3.1 Calibration objective (fit λ on D_cal only)

Define the empirical served-slice risk at threshold `λ` on calibration:

```
R̂_n(λ) = ( Σ_{u∈D_cal} 1{g(u) ≥ λ} · L_u ) / ( Σ_{u∈D_cal} 1{g(u) ≥ λ} ).
```

For a target risk α, the **conformal threshold** is the *least selective* (smallest, i.e. highest
coverage) λ whose finite-sample-corrected served risk is still ≤ α:

```
λ̂(α) = inf { λ :  R̂_n(λ) · n_served/(n_served+1)  +  B/(n_served+1)  ≤ α },
```

where `n_served = Σ 1{g(u) ≥ λ}` is the calibration served count and `B = 1` is the loss bound. The
`n/(n+1)` shrink + `B/(n+1)` slack is the **CRC finite-sample correction** (Angelopoulos et al.,
Conformal Risk Control): it inflates the empirical risk just enough that the guarantee holds in
expectation over a fresh exchangeable test point, not merely in-sample. We scan λ over the calibration
`g`-quantiles (each distinct served-set boundary) and take the smallest qualifying λ.

A target *coverage* (e.g. 50%) is the dual knob: fix the served fraction and *report* the certified
risk, or fix α and *report* the achieved coverage. The milestone reports both.

### 3.2 Served-slice guarantee (Theorem 1)

**Theorem 1 (served-slice risk certificate).** Let `(g(u), L_u)` for `u ∈ D_cal ∪ {U*}` be
exchangeable, `L_u ∈ [0, B]`. Fix the calibration-selected threshold `λ̂(α)` as in §3.1. Then for a
fresh test user `U*` drawn from the same distribution, conditioned on being served
(`g(U*) ≥ λ̂(α)`),

```
E[ L_{U*} | g(U*) ≥ λ̂(α) ]  ≤  α .
```

Equivalently `E[1 − NDCG@k | served] ≤ α`, i.e. **served NDCG@k ≥ 1 − α in expectation**,
distribution-free and finite-sample.

*Proof sketch.* Restrict attention to the served subpopulation `S_λ = {u : g(u) ≥ λ}`. Within `S_λ`
the losses `{L_u}` are exchangeable (selection by `g`, a deterministic function of the label-free
features, preserves exchangeability of the conditional loss). CRC's monotone-risk control theorem
applied to the loss `L` restricted to `S_λ`, with the `n_served/(n_served+1) + B/(n_served+1)`
correction, yields `E[L_{U*} | U* ∈ S_λ] ≤ α`. The risk `R̂_n(λ)` is monotone non-increasing in λ
(more selective ⇒ keep higher-`g`, lower-risk users — empirically and by construction of `g`), so the
`inf` is well-defined and the chosen λ̂ is the highest-coverage point meeting the bound. ∎

**Reading.** The guarantee is on **what is served**, which is exactly the quantity a deployed system
is accountable for. It is *not* a full-population claim (abstained users are routed elsewhere). The
correction is `O(1/n_served)`; with `n_served` in the hundreds it is a sub-0.01 inflation of α.

### 3.3 Honest caveats baked into the method

- The guarantee is **marginal over the served slice**, not conditional per user; it does not claim a
  per-user PAC bound.
- It is an **expectation** bound (CRC controls `E[L]`). A high-probability PAC variant (UCB / `RCPS`)
  is a drop-in extension (Bates et al.) and is noted as future work; the milestone reports the CRC
  expectation guarantee and *checks it empirically held on test*.
- λ̂ is fit on calibration; the test certificate is the out-of-sample check. If test served-risk
  exceeds α beyond noise, the guarantee is reported as **violated** (it should not, under
  exchangeability, beyond the `O(1/n)` slack + sampling noise — we attach paired-bootstrap CIs).

---

## 4. Joint (risk, coverage) Pareto (Theorem 2)

Sweeping α (equivalently λ) traces a **Pareto frontier** in (coverage, served-risk) space. RankCRC
emits, for each α, the pair `(coverage(α), certified_risk(α), achieved_test_risk(α))`.

**Theorem 2 (Pareto monotonicity & dominance).** The map `α ↦ (coverage(λ̂(α)),
served_risk(λ̂(α)))` is monotone: decreasing α (stricter) weakly decreases coverage and weakly
decreases served risk. Consequently RankCRC's frontier is a *non-crossing* curve, and a scorer A
**Pareto-dominates** scorer B if A's served-risk is ≤ B's at every coverage with strict inequality
somewhere. This gives a single, assumption-light axis on which to compare *any* two scorers (including
official baselines) for selective deployment — not just full-coverage NDCG.

*Proof sketch.* Monotonicity of `R̂_n(λ)` in λ (§3.2) plus monotonicity of coverage in λ (serving a
superset as λ drops) gives the coordinatewise monotone map; dominance is then the pointwise order on
the resulting curves. ∎

This reframes the contribution as **comparative selective reliability**: even a scorer that loses on
full-coverage NDCG can win on the certified-risk-at-coverage axis, which is the deployable question.

---

## 5. Selective-Gain Decomposition (Proposition 3)

Why does abstaining on low-`g` users raise served NDCG@k? RankCRC gives a closed-form first-order
account.

**Proposition 3 (selective-gain decomposition).** Let coverage fraction be `c`, full-population mean
NDCG be `μ`, and let the per-user gain from selection be approximated to first order by how strongly
`g` ranks users by their realized NDCG. Then the selective NDCG lift over random abstention satisfies

```
ΔNDCG(c)  ≈  (2·AUC − 1) · G · κ(c),
```

where:
- `AUC` = AUC of `g` predicting the binary correctness target (the confidence discriminativeness);
- `G` = Gini-style dispersion of the per-user NDCG distribution (`G = Δ̄ / (2μ)` with `Δ̄` the mean
  absolute pairwise NDCG difference) — the *headroom* selection can exploit;
- `κ(c)` = a coverage factor, the expected NDCG-rank advantage of the served slice under a confidence
  signal with rank-correlation `2·AUC−1`, increasing as `c → 0` (≈ `(1−c)` to first order for the
  mean-over-served lift; the milestone reports the exact realized `κ`).

`2·AUC−1` is the **Somers' D / rank-correlation** of `g` with correctness; multiplying by the NDCG
dispersion `G` and the coverage factor `κ` recovers the realized selective lift. The milestone reports
**predicted ΔNDCG (from AUC, G, κ) vs realized ΔNDCG at 50% coverage** and the relative error; a tight
match validates that the gain is *explained* by confidence discriminativeness × headroom × coverage,
not an artifact.

*Derivation sketch.* Selection by `g` reorders users; the expected NDCG of the top-`c` fraction under
a signal with rank-correlation `ρ_s = 2·AUC−1` to realized NDCG equals `μ + ρ_s · (dispersion term) ·
(coverage term)` to first order (standard concomitant-of-order-statistics expansion). Identifying the
dispersion term with the Gini `G·μ` and the coverage term with `κ(c)` gives the stated form. The
approximation is first-order in `ρ_s` and exact when NDCG and `g` are jointly Gaussian-copula. ∎

---

## 6. Model-agnostic certification

RankCRC's only input is a per-user score vector `s_u` (to compute `L_u`) plus a confidence `g(u)`.
Therefore **any scorer can be certified on the same protocol**: re-fit `g` on that scorer's
calibration scores, fit `λ̂(α)`, report the (risk, coverage) Pareto. Official baselines (ELMRec,
IRLLRec, LLM2Rec, LLMEmb, LLM-ESR, ProEx, ProMax, RLMRec) become *certifiable* on the selective-
reliability axis, which is the comparison the paper makes (8-domain rollout, GPU, later). The base
scorer here is SASRec-over-frozen-Qwen with the **raw** score (no ρ trust-mix); the trust-mix is a
demoted diagnostic, not part of the served pipeline.

---

## 7. Honest framing & falsifiability

- **What RankCRC IS:** a deployable, distribution-free *risk certificate* for selective recommendation
  + a comparative selective-reliability axis + a closed-form explanation of the selective gain.
- **What RankCRC is NOT:** a new full-coverage SOTA scorer. Full-coverage NDCG@10 on beauty is 0.1143
  (raw scorer), below the ProEx bar 0.1506; RankCRC does not and is not claimed to beat that on full
  coverage. The claim is on the *served slice* and the *certificate*.
- **Falsifiable predictions (report ALL):**
  1. The Theorem-1 guarantee **holds on test**: at each α the achieved test served-risk ≤ α (within the
     `O(1/n)` slack + paired-bootstrap CI). If test risk exceeds α systematically → guarantee fails.
  2. The (risk, coverage) frontier is **monotone & non-crossing** (Thm 2). A crossing → bug or
     non-exchangeability.
  3. The decomposition (Prop 3) **predicts realized ΔNDCG at 50% coverage** within a stated tolerance.
     A large mismatch → the gain is not explained by `(2·AUC−1)·G·κ` and the proposition is retracted.
  4. `g` must beat the deployable `H_top` baseline (AUC and selective gain), else the calibrator adds
     nothing over the trivial signal.

---

## 8. The two Codex-found mismatches (reconciled here)

See `docs/rankcrc_mismatch_reconciliation.md` for the full write-up; summary:

1. **D_uk panel-discriminativeness deferred at γ=0** (`src/llm4rec/methods/calm_qwen.py`,
   `gamma_D_uk: 0.0` in `stage_b/calm_stage_b_meta.json`). **Resolution: intended, not a bug, and
   orthogonal to RankCRC.** The IntentEncoder contract is *panel-free* (`encode_intents` sees only
   history); a non-zero `D_uk` would require feeding the 101-candidate panel into inference, which
   breaks same-candidate consistency. `γ=0` is recorded for honest reporting. RankCRC operates on the
   final score vector and does not depend on `D_uk`; the panel-discriminativeness term is a deferred
   GPU experiment (8-domain rollout), not a blocker for the beauty certificate.

2. **Stage-B artifact `n_intents=2` vs spec `K=4`.** **Resolution: the frozen beauty signals were
   produced by the SASRec scorer (`train_calm_sasrec.py`, default `--n-intents 2`), not the Qwen
   Stage-B GPU trainer (`train_calm_stage_b.py`, default `--n-intents 4` = spec K=4).** The frozen
   `signals_*_sasrec.npz` therefore carry K=2 responsibility geometry (confirmed: H ∈ [0, ln 2 ≈
   0.693]). This **does not affect the RankCRC milestone**: K only changes the dimensionality of the
   responsibility-entropy *feature* fed to `g`, which is still computed and present; the certificate is
   over the score vector and `g`. The K=4 spec value is the **GPU Stage-B target** for the 8-domain
   rollout; the CPU SASRec proxy at K=2 is the validated local scorer. We document the discrepancy and
   defer the K=4 Qwen re-run to GPU. No silent edit of the spec or the artifact.

---

## 9. Beauty milestone (CPU, validated locally)

Script `scripts/rankcrc_validate.py`; artifacts `outputs/calm/beauty_frozen_v2/rankcrc/`. Reports:
served-slice certificate at target risks giving ≈90/80/70/60/50% coverage (achieved coverage, served
NDCG@10/HR@10/MRR, guarantee-held flag), the (risk, coverage) Pareto, the Prop-3 decomposition
predicted-vs-realized ΔNDCG at 50% coverage, all with paired-bootstrap CIs; plus a journal-style
matplotlib risk-coverage figure and a CSV + markdown certificate table. See the milestone REPORT and
`outputs/calm/beauty_frozen_v2/rankcrc/RANKCRC_VERDICT.md`.
