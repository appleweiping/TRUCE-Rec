# Method — RankCRC (Risk-Controlled Selective Recommendation)

> **Status / provenance.** This is the CURRENT headline method, replacing the stale CU-GR v2 draft in `method.md` (pre-pivot, May 4 — kept for history). The served-slice guarantee below is adversarially verified (`refine-logs/rankcrc_formalization_verified.md`, verdict `breaks=false`) and matches the implemented validation harness (`scripts/rankcrc_validate.py`, beauty re-validated end-to-end). Experiment-independent: the theorem and procedure are LOCKED; only the empirical tables (8-domain rollout) remain GPU-gated. Frozen protocol: 8 Amazon domains, 10k users (beauty 973), 101 same-candidates/event (1 positive + 100 popularity-matched negatives), Qwen3-8B base, metrics HR/NDCG@{5,10,20} + MRR, paired Holm-corrected bootstrap.

## 1. Problem setup — selective recommendation

For each user `u`, a **frozen** base ranker scores a fixed candidate set and emits a top-`k` list; recommendation quality is the listwise rank-risk
```
ρ_u = 1 − NDCG@k(u) ∈ [0, 1],
```
fixed once the base ranker and `k` are fixed. Standard recommenders serve *every* user. We instead allow **abstention**: a deployable system may decline to serve a user (route to a fallback channel) when it cannot vouch for the recommendation. The object of study is the **served-slice risk** — the expected rank-risk *among the users we choose to serve* — and the goal is a distribution-free guarantee that this served-slice risk stays below a target `α`, using only signals available at deployment time (no peek at the held-out relevance).

This is a *reliability* contribution, deliberately orthogonal to raw-ranking SOTA: on beauty the base raw NDCG@10 (0.1143) is below the strongest baseline bar (ProEx 0.1506). RankCRC does not claim to out-rank; it aims to *bound, in marginal expectation, the rank-risk on the users it chooses to serve*. The lane of raw-ranking improvement belongs to the sibling pointwise-posterior method (pony); TRUCE owns calibrated served-slice reliability.

**Contribution framing (honest).** We do not claim a new conformal *theorem* — once the selector is frozen, the served-slice control is standard holdout/CRC-style risk estimation on the accepted sub-population. The contribution is a *protocol + application*: (i) the served-slice, recommendation-specific **listwise** rank-risk formulation; (ii) a **leakage-safe, label-free selector** over rank-geometry with the frozen-`λ` B1/B2 discipline that makes the conditional-mean control valid; (iii) a falsifiable served-slice gain law; and (iv) the empirical feasibility across 8 domains (the GPU-gated result). A skeptic's "this is CRC on accepted examples" is *correct* and is exactly the foundation we build the deployable selective-recommendation protocol on.

## 2. The label-free confidence `g`

We score each user's recommendation reliability with a confidence `g(x_u) ∈ ℝ` that is **label-free at inference**: a logistic model over 24 **rank-geometry** features computed entirely from the base ranker's candidate scores and side information — score margins (top-1 vs top-2, top-1 vs median), softmax peakiness/temperature, responsibility-entropy order statistics, MC-dropout score variance, and candidate-popularity statistics. The held-out relevance `H_target` is **forbidden as a feature** (enforced by a column-allowlist audit, §6), so at deployment `g` reads no test-fold label and the feature map is fully deployable. (To be precise about training: `g`'s logistic *target* on Fold A is the per-user rank-risk `ρ_u = 1−NDCG@k`, which is derived from the relevance `y_u`; so `g` is *trained* on relevance labels on the disjoint Fold A, but is label-free as a *feature map at inference* — the only property validity requires, since A is disjoint from B1/B2/C.) On beauty, `g` attains test AUC 0.627 for separating high- from low-risk users, versus 0.519 for the raw top-margin baseline `H_top`. On beauty, `g` attains test AUC 0.627 for separating high- from low-risk users, versus 0.519 for the raw top-margin baseline `H_top`.

`g` is the *only* learned component, and it is a pure scoring function — it ranks users by reliability but sets no threshold. Thresholding and certification are handled by the conformal layer (§3), which is where the distribution-free guarantee enters.

## 3. The RankCRC procedure (A / B1 / B2 / C fold discipline)

The single decision RankCRC makes is **whom to serve**: serve `u` iff `g(x_u) ≥ λ`, abstain otherwise. The acceptance sets are nested (`S(λ′) ⊆ S(λ)` for `λ′ > λ`), so raising `λ` serves fewer, higher-confidence users. Because the base ranker is frozen and `k` is fixed, `λ` changes *only which users are served*, never the contents of any list — so for a fixed-`k` NDCG objective the abstention threshold `λ` **is** the coverage threshold (there is no within-slice risk knob; we write `λ ≡ τ`). A genuinely distinct `λ ≠ τ` arises only under a *changed* objective (a charged abstention price, or numerator-control with a coverage floor; §4).

Partition users into four disjoint folds (committed manifest, seed 20260506):

- **Fold A — fit `g`** (freeze #1). Fit the 24-feature logistic `g`; `H_target` forbidden. `g` is frozen and never refit.
- **Fold B = B1 ∪ B2 — calibrate** (freeze #2, the load-bearing discipline). On a *pre-registered* coverage grid, set the threshold on **B1 only**: `λ := Quantile_{1−c}(g over B1)`. Then on the **disjoint B2**, compute the served count `m = |{u ∈ B2 : g(x_u) ≥ λ}|`, the served mean `R̂_m = (1/m) Σ_{u∈B2: g≥λ} ρ_u`, and the conformal-risk-control certificate
  ```
  α(c) = R̂_m · m/(m+1) + 1/(m+1).
  ```
  Crucially, **`λ` is chosen on B1 and never optimized against the B2 losses it averages.** The coverage grid, the deployed operating point, and the minimum-served-count `m_min` are **pre-registered before B2 is inspected**: an operating point whose B2 served count falls below `m_min` is declared invalid by the pre-registered rule, *not* dropped by looking at its B2 risk. (Selecting the operating point by inspecting B2 certificates/risks/valid-`m` across the grid would void the theorem unless that selection rule is itself pre-registered and loss-independent, or upgraded to LTT/RCPS — §4.) The deployed operating point and grid are frozen here, before any test data is touched.
- **Fold C — test** (scored once). Apply the frozen `(g, λ)`; report achieved coverage `|S_C|/|C|`, served NDCG@k, and the served-risk alongside the certificate. A single Fold-C split is one draw of an expectation guarantee, so the *headline* validation is the mean over ≥200 resplits (mean test served-risk ≤ mean `α`), not a single-split pass/fail. Fold C performs no selection or tuning.

**Why the split is load-bearing.** The naive served mean `R̂(λ) = (1/m) Σ_{g≥λ} ρ` is a *self-normalized ratio* with a random denominator `m`; its per-example contribution is not monotone in `λ`, so plain Conformal Risk Control (CRC, Angelopoulos & Bates 2024) does not attach to it *if `λ` is chosen on the same fold whose losses are averaged* (a winner's-curse / post-selection bias on the ratio). Choosing `λ` on a disjoint B1 dissolves this: conditional on the B1-frozen `λ`, the served-B2 users are i.i.d. draws from the fixed conditional law `P(·|g≥λ)`, so the served-B2 mean is an ordinary bounded i.i.d. mean and the CRC correction applies. (The split is a *provably-never-worse* safeguard: under the coverage-quantile parameterization the served count `m≈c·n` is near-deterministic, so the same-fold estimator merely *appears* benign for a continuous `g`; the split makes validity unconditional rather than data-dependent.)

## 3.1 Theorem (served-slice selective-risk certificate — frozen-selector marginal expectation)

*Let users be exchangeable with label-free features `x_u` and bounded rank-risk `ρ_u = 1 − NDCG@k(u) ∈ [0,1]` (fixed `k`, frozen base scorer). Let `g` be fit and frozen on Fold A, and assume `g` is continuous at the operating quantile (no atom at `λ`; otherwise apply `U(0,ε)` tie-break jitter). Set `λ := Quantile_{1−c}(g)` on B1, and on the disjoint B2 form the served count `m = |{u∈B2: g≥λ}|`, the served mean `R̂_m`, and the per-sample value `α(B2) := R̂_m·m/(m+1) + 1/(m+1)` when `m ≥ 1` (and `α(B2) := 1` when `m = 0`; we additionally require a pre-registered minimum served count `m_min`, with invalidity declared by the `m_min` rule **before any B2 risk is inspected**, never by reading B2 losses). Write `μ(λ) := E[1 − NDCG@k(U*) | g(x_{U*}) ≥ λ]` for the true served-slice risk of a fresh exchangeable user `U*`. Then, conditional on the frozen selector `(g, λ)`,*
```
   μ(λ)  ≤  E_{B2}[ α(B2) | g, λ ],
```
*i.e. the EXPECTATION of the certificate over B2 draws upper-bounds the true served-slice risk (distribution-free, finite-sample).*

**Realized vs expected (the precise reading — do not overclaim).** The bound is on `E_{B2}[α]`, NOT on any single realized `α(B2)`: since `E[α(B2) | m] = (m·μ + 1)/(m+1) ≥ μ`, the certificate is upward-biased in expectation, but a *single* B2 can yield `α(B2) < μ`. So `α(B2)` reported from one calibration draw is an estimate of an upper bound, not a realized guarantee. The headline empirical claim is the **expectation check over ≥200 resplits** (mean test served-risk ≤ mean `α`), never a single split.

**Guarantee class (no overclaim).** Marginal-expectation, conditional on the frozen selector `(g, λ)`: (a) marginal over the served sub-population, **not** per-user; (b) an expectation bound — **not** PAC/high-probability (a single deployment may exceed `μ`); (c) conditional on this frozen selector, **not** integrated over the randomness of fitting `(g,λ)`; (d) **not** a fixed-served-size bound (`m` random, integrated in expectation). Choosing `λ` on the same fold whose losses are averaged voids even this expectation bound.

**Proof sketch.** Conditional on `(g, λ)` frozen on A∪B1, the indicator `1{g(x_u) ≥ λ}` and the loss `ρ_u` are deterministic transforms of the exchangeable draw, so — *conditional on the served count `m`* — the accepted B2 observations are an exchangeable sample from the fixed accepted law `P(·|g≥λ)` and hence `E[R̂_m | m] = μ` **for every `m`**. Thus `E[α(B2) | m] = (mμ+1)/(m+1) = μ + (1−μ)/(m+1) ≥ μ` for every `m`. Because *each* mixture component already exceeds `μ`, any pre-registered, loss-independent reweighting over `m` — in particular the `m ≥ m_min` exclusion (and the `m=0 ⇒ α=1` convention) — preserves the inequality: `E[α | m ≥ m_min] = Σ_m P(m | m ≥ m_min)·E[α|m] ≥ μ`. The load-bearing condition is therefore **conditional-mean-unbiasedness given `m`** (`E[R̂_m|m]=μ`), which holds under exchangeability; it is *not* merely "i.i.d." in passing. **Where this can break:** if there is *population-level* dependence that couples `m` with the served risk (the popularity-coupling / temporal-drift threat in §6 / the limitations), `E[R̂_m|m] ≠ μ` and a pre-registered `m`-reweighting can select adversarially — so the exclusion's safety is exactly as strong as the exchangeability assumption, no stronger. (After freezing `λ`, monotonicity in `λ` plays no role — this is bounded-loss risk estimation on a fixed accepted distribution, with `1/(m+1)` the finite-sample slack; monotonicity matters only when searching over thresholds, the LTT route of §4.) ∎

## 4. Optional upgrades (named, not headline)

- **`(1−δ)` high-probability variant.** Replace the coverage-fixed CRC step by **fixed-sequence Learn-then-Test** (FWER-controlled one-sided Hoeffding–Bentkus or betting p-values) over the pre-registered coverage grid → `P(E[1−NDCG | g≥τ̂] ≤ α) ≥ 1−δ`, valid for a *data-chosen* operating point and robust to ties, at a coverage cost. (The same machinery is the only valid route if one insists on selecting `λ` to hit a *target* `α`; selecting `λ` by minimizing the B2 certificate would re-introduce the post-selection bias.)
- **Exact belt-and-suspenders surrogate.** CRC-control the *unconditional* numerator `N(λ) = E[ρ·1{g≥λ}] ≤ β` (a genuine monotone per-example CRC family) together with a certified coverage floor `C(λ) ≥ c₀` ⟹ `E[1−NDCG | served] ≤ β/c₀` (cost `1/c₀`).

## 5. Selective-gain decomposition (Prop 3) — a falsifiable predictive law

Under stated assumptions (per-user NDCG and `g` coupled only through a monotone link; served-slice lift factorizing into ranking-quality-separation × score-dispersion × coverage), the served-slice lift admits the closed form
```
   ΔNDCG@k(c) ≈ (2·AUC − 1) · Gini · κ(c),
```
predicting the lift *a priori* from the confidence AUC, the score-dispersion Gini, and a coverage factor `κ` — all measurable on the calibration folds. It is pre-registered with a **falsification test**: predicted vs observed ΔNDCG across all domains × coverage levels; if out-of-sample R² is below a pre-set bar, the law is demoted to a heuristic. No coefficients are fit to test risk. This is a *secondary* contribution; the headline guarantee does not rest on it.

## 6. No-leakage audit

Validity rests on three assumptions, each enforced in code: (i) `H_target` forbidden as a `g` feature — a column-allowlist audit that we verify *has teeth* via a negative control (injecting any of {`H_target`, `target_rank`, `target_ndcg`, `is_correct`, `rank_risk`} must raise); (ii) the fold index sets A/B1/B2/C are asserted pairwise-disjoint, with `λ` provenance restricted to B1 and loss-averaging to B2; (iii) `g` is asserted (near-)continuous at the operating quantile, else `U(0,ε)` jitter is applied. The achieved B2 served count `m` is reported per operating point (the B1/B2 split costs sample — a power, not a validity, cost).

## 7. Relationship to CALM-Rec (strictly subordinate)

CALM-Rec — calibrated trust over attribute-anchored latent multi-intent — is an *optional base-ranker upgrade*, not a load-bearing claim. RankCRC wraps **any** base ranker; the headline guarantee holds whether the base ranker is CALM, the fixed-blend scorer, or the pony pointwise posterior. CALM is elevated to a co-headline only if the pre-registered Stage-2.5 reliability gate passes (real Qwen3-8B encoders, K=4 intents, reliability AUC > 0.6 — the K=2 CPU proxy was weak); otherwise CALM is demoted to a diagnostic/ablation and the base ranker reverts to the fixed-blend scorer. RankCRC is the standalone headline either way.
