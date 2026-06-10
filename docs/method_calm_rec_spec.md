# CALM-Rec — Method Spec (TRUCE-Rec Ours v2)

**CALM-Rec = Calibrated trust over Attribute-anchored Latent Multi-intent for Recommendation.**

**Date:** 2026-06-07 · **Status:** DESIGN LOCKED (tri-agent ARIS, ≥20 iterations, both cross-seats +
lead endorse) · implementation of the scoring core DONE + tested · formal run BLOCKED (server busy).
**Supersedes** SCALR as the main method (SCALR was a single debiased listwise scorer + two additive
penalties — only enough to *match*, not *beat*, strong baselines). SCALR's pure functions remain in
the repo as an ablation/diagnostic reference. **Replaces** the retired generate→ground→gate route.

Full iteration trail: `outputs/method_redesign_discussion/iterations/` (STATE_v0 + round1-5 syntheses
+ per-seat notes; gitignored). This doc is the tracked decision.

---

## 0. Headline contribution (the paper's one sentence)

> CALM-Rec learns, **per user–item, how much to trust the LLM's personalized multi-intent judgment
> versus a history-free item prior**, using the geometry of its own intent mixture (responsibility
> entropy + ensemble variance) as an **endogenous reliability signal**. Multi-intent is the *vehicle*
> that makes the reliability signal computable; **calibrated personalization-trust is the
> contribution** — selective abstention *from personalization* (not from prediction).

No LLM4Rec baseline performs signal-driven, per-candidate fallback to a prior using a reliability
signal *internal to the model's own representation* (vs an external confidence head).

## 1. Scoring (maps user history + 101-candidate panel → 101 scores)

For user `u` (K attribute-anchored intents) and candidate `i`:

```
e_uik  = ⟨W z_uk, h_i⟩ − δ_k · log(1 + n_i)              # per-intent energy, popularity-residualized
s_pers = (1/τ) · logsumexp_k( log π_uk + τ · e_uik )      # soft-OR mixture over K intents
r_uik  = softmax_k( log π_uk + τ · e_uik )                # intent responsibilities
H_ui   = −Σ_k r_uik log r_uik                             # responsibility entropy (reliability signal)
ρ_ui   = σ( α0 + α1·H_ui + α2·Var_m[s_pers] − α3·log(1+n_i) )   # trust gate, α1,α2,α3 ≥ 0
s_ui   = (1 − ρ'_ui)·s_pers + ρ'_ui · s_prior_i           # calibrated trust mixing, ρ' capped by ρ-floor
```

Rank the 101 candidates by `s_ui` (descending). Schema `source_event_id,user_id,item_id,score`.

- `z_uk = c_k + r_uk`: **attribute-anchored** intent (anchor `c_k` = attribute-prototype centroid;
  only the residual `r_uk` is personalized → overfit control + the firewall vs ComiRec).
- `h_i`: **precomputed** item vector from a cached item encoder (frozen Qwen3-8B over item text). The
  101 candidates are **never** in the LLM prompt (position noise breaks same-candidate consistency).
- `n_i`: train-only item support (all held-out targets removed). `s_prior_i = b_i + β^T x_i` is a
  history-free item-text/global prior (the calibrated fallback).
- `Var_m`: variance of `s_pers` over M MC-dropout passes of the user-side forward (items deterministic).
- `ρ-floor` (≈0.15): personalization weight never vanishes, so CALM never collapses to the prior.

## 2. Why this beats SOTA (not just matches — the SCALR fix)

The binding bottleneck of SCALR was **rank-1 personalization**: the whole history squeezed into one
scalar lift per candidate, while baselines rank at rank-d (LLMEmb's d-term embedding match, ProEx's
multi-attribute profile). CALM restores **rank-K** capacity via the intent mixture, keeps debiasing
by moving the popularity residual *into each intent dimension*, and adds the thing no baseline has: a
**per-user-item trust decision** between personalization and a calibrated prior. Against beauty's
ProEx specifically: `z_uk` is a *latent* profile in the LLM's semantic space (same world knowledge
ProEx taps) but continuous (no text bottleneck), panel-conditioned, and — crucially —
**calibrated-trust** (shrinks toward the prior exactly when a sparse user's posterior is ambiguous,
whereas ProEx fully commits to a possibly-hallucinated profile).

## 3. Panel-conditioned intent salience (no popularity-sampler confound)

```
π_uk = softmax_k[ g_uk + λ·E_uk + γ·D_uk ]
E_uk = (1/|H_u|) Σ_{j∈H_u} exp(e_ukj) / q(j)            # importance-weighted history evidence (q = train exposure prob)
D_uk = stop_grad( Var_{i∈C}( e_uik − s_prior_i ) )       # prior-residualized panel discriminativeness
```

`E_uk` reads which intents the user's history supports (exposure-debiased). `D_uk` upweights intents
that separate the slate *beyond popularity*; the `−s_prior_i` residual + `stop_grad` (+ per-query
leave-target-out) prevent the model from gaming the popularity-sampled negatives. `g_uk` = panel-free
prior read from history.

## 4. Training (3 stages; test split never used)

- **Stage A (CPU, offline, train-only, frozen):** item-text embeddings `x_i` + cached `h_i`; item
  support `n_i` and exposure `q(j)` (ALL held-out targets removed); popularity `b_i`; attribute
  weak-labels + anchor centroids `c_k`; fit `s_prior = b_i + β^T x_i` by train-LOO regression.
- **Stage B (end-to-end, 1×4090):** train LoRA + `W` + residuals `r_uk` + salience `{g,λ,γ}` + `δ_k`
  + temperature `θ_τ` on
  `L = L_rank + 0.3·L_attr + 0.1·L_bal + 0.05·L_orth + 0.05·L_use + 0.01·L_τ`.
  `L_rank` = listwise sampled-softmax CE over the 101 (only label-bearing term). `τ` annealed UP from
  ~1.5 (bounded [1,8]). **ρ held FIXED mild (≈0.1)** so the panel must learn to rank without leaning
  on the prior crutch.
- **Stage C (CPU, post-hoc, LoRA frozen):** fit ONLY ρ's 4 `α` (constrain α1,α2,α3 ≥ 0) + final τ on a
  **validation** holdout (carved from train, never test) minimizing val listwise NLL (+ECE/Brier).
- **Stage 2.5 reliability gate (CRITICAL, before trusting ρ):** measure AUC of `[H(r), Var_m]`
  predicting whether `s_pers` ranked the positive above the negatives on validation. **AUC > ~0.6** →
  the reliability signal is real, proceed with ρ. **AUC ≈ 0.5** → drop the entropy/variance term (or
  add a learned reliability head), OR ship the K-intent core with a *fixed* blend and **drop the
  trust headline**. (`reliability_auc()` is implemented in `calm_rec.py`.)

## 5. Anti-collapse / identifiability (K=4 on 973 users)

- Attribute-ANCHORED intents (`c_k` from weak-labeled attribute centroids; hard-init each from a
  DIFFERENT attribute field); only `r_uk` personalized (‖r_uk‖≤ε).
- `L_orth` on the **projected** directions `(WZ)(WZ)^T` off-diagonal (orthogonalize `WZ`, since `e_ik`
  only sees `WZ`); `L_bal` load-balancing on mean responsibilities; `L_use` usage floor.
- K=4 for sparse domains (beauty); raise toward 6 only where a domain has >2k users. K chosen by
  attribute count, NOT validation-tuned. Cross-domain shared init of `W` + anchor angular spread on
  dense domains (books/electronics), fine-tune magnitude on the target domain.

## 6. Weak-labels (reproducible, no paid model, no leak)

Beauty: versioned lexicon YAML + regex over item title/brand/category/description/ingredients →
{skin concern, finish/effect, shade/color, brand-ingredient}; soft multi-label = softmax cosine of a
FREE local item-text embedding (Qwen3 last-hidden or bge-small) to facet-seed centroids; `c_k` = mean
embedding of top-assigned items (fixed seed); low-confidence → unknown. 7 non-beauty domains: same
pipeline, K-means(K=4, fixed seed) on item-text embeddings + TF-IDF auto-naming, or clean Amazon
taxonomy facets; downweight `L_attr` where clusters are noisy. Commit lexicon YAML + embed-model hash
+ kmeans seed + TF-IDF vocab for exact reproduction. Labels derive ONLY from item text (never
interactions / never reviews postdating the target).

## 7. Leakage controls (3 concrete leaks found in review + fixed)

- `n_i`, `b_i`, `q(j)`: **train-only, ALL held-out targets removed** (else the positive's own held-out
  edge inflates its support asymmetrically vs the negatives).
- `E_uk`: sums only over history items (never candidates/target). `D_uk`: stop-grad + per-query
  leave-target-out.
- `s_prior`: global TRAIN interactions + item text only; cold-start item → text prior.
- Attribute labels: item metadata/text TRAIN snapshot only; ban review-derived labels postdating the
  target. Negatives share the same train-only popularity as positives.
- ρ's α and τ: validation only; appendix includes a deliberately LEAKY-prior red-flag control.

## 8. Falsifiability contract (report ALL; a win on the full model alone is NOT credited)

Beauty NDCG@10, paired user-bootstrap, ProEx bar = 0.1506:
1. **Full CALM ≥ ~0.156–0.158** (beat ProEx by ≥ +0.005) — else the method fails outright.
2. **ρ vs variance-matched placebo** (same marginal/variance, dependence on H/Var/n destroyed):
   real ρ − placebo ≥ +0.004 NDCG@10 (p<0.05) AND lower ECE; reliability Spearman(ρ, realized error)
   ≥ 0.15, monotone. Else ρ is ornamental → drop shrinkage.
3. **Intent ablation** K∈{1,2,4,8}: K=4 − K=1 ≥ +0.005 (p<0.05); K=4 ≥ K=8; + responsibility-shuffle
   placebo ≥ +0.004. If K=1 ties K=4 → collapse to K=1 (drop orth/bal/use). If K=2≈K=4 → report "low
   effective intent", don't oversell.
4. **No-attribute-anchor** must clearly degrade (the firewall vs ComiRec); report `p_ψ` attribute
   accuracy as evidence the anchor binds.

## 9. Release gates (hard) + honesty fallbacks

RELEASE requires: (1) beat ProEx on beauty; (2) real ρ beats placebo; (3) no-anchor clearly degrades;
(4) cross-domain replication — all 7 LLMEmb domains under SHARED hyperparameters, wins in ≥5/7 with
paired-bootstrap CIs. A method winning only on 973 beauty users is RETRACTED. Honesty fallbacks:
ρ can't beat placebo → remove shrinkage; K=2≈K=4 → "low effective intent"; multi-intent doesn't move
→ demote from title, sell as "uncertainty-calibrated semantic personalization with
attribute-constrained decomposition".

## 10. Feasibility (one RTX 4090)

Cached item encoder (offline, minutes for beauty); short user-side prompt (history ≤20 + K intent
queries; beauty sparsity → ~0.5–1.5k tokens); candidate scoring is a K×101 matmul in vector space
(microseconds); M=4 dropout passes (M=8 audit) on the short user forward only. 8B + (Q)LoRA r16-32 +
grad-checkpoint fits 24 GB. Beauty trains (3 stages) + infers in **hours**.

## 11. CALM-specific failure modes (detector → fix)

| Failure | Detector | Fix |
|---|---|---|
| ρ → 0 globally (no trust) | std(ρ)<0.05 / α1,α2≈0 | standardize H,Var; sign-constrain α1,α2>0; center α0 to mean-ρ≈0.3 |
| ρ → 1 globally (ships prior/popularity) | placebo-ρ≈real-ρ; Kendall-τ(full,prior)→1 | cap mean ρ; ρ-floor; s_prior uses item-TEXT not bare popularity |
| anchors → popularity proxy | regress anchor activation on log n_i (high R²) | per-intent δ_k + stop-grad prior-residual D; raise L_orth; re-seed c_k from attribute text |
| intents collapse to one axis | high cos(c_k,c_l); K=4≈K=1 | L_bal/L_orth/L_use; hard-init each c_k from a different attribute field; anchor repulsion |
| Var_m is noise | not monotone w/ error; real ρ ≤ placebo | drop Var_m (keep entropy); if still nothing, drop shrinkage headline |

## 12. Implementation status & next steps

- **DONE (no GPU), tested:**
  - Scoring core + `CALMRecRanker` + encoder abstractions + deterministic mocks —
    `src/llm4rec/methods/calm_rec.py`.
  - Encoders: `calm_encoders.py` — `HashedItemEncoder`/`LexiconIntentEncoder` (CPU, runs anywhere) +
    gated `QwenItemEncoder`/`QwenIntentEncoder` (server path) + `build_encoders` factory.
  - Weak-labels: `calm_weak_labels.py` (attribute lexicon, deterministic soft-labeling) +
    `scripts/build_calm_weak_labels.py` (verified: 254/479 real beauty items get a dominant facet).
  - 3-stage trainer scaffold + leakage-clean Stage-A stats + Stage-C ρ calibration + the stage-2.5
    reliability gate — `calm_trainer.py`.
  - Runner wiring (method `calm_rec`), `configs/methods/calm_rec.yaml`,
    `configs/experiments/smoke_calm_rec.yaml`; runs end-to-end through the official runner + evaluator.
  - Falsifiability ladder + verdict — `scripts/run_calm_rec.py`.
  - Tests: `tests/unit/test_calm_rec.py` (13) + `tests/unit/test_calm_encoders.py` (11) +
    `tests/smoke/test_calm_rec_pipeline.py` (2) + `tests/smoke/test_calm_rec_runner.py` (1) — all pass,
    numpy-free, no regressions.
- **TODO (the only remaining work):** implement the real Qwen3-8B forward inside
  `QwenItemEncoder.encode` (cached) and the Qwen3-8B+LoRA Stage-B gradient loop behind
  `QwenIntentEncoder` (the contract it must satisfy is already exercised by the hashed encoders).
  Everything that *consumes* the encoders/trainer exists. See `docs/CALM_REC_RUNBOOK.md` §2.2-2.3.
- **Server run only on the user's go-ahead** (server busy). First experiment priority: **real ρ vs
  placebo** + beat ProEx 0.1506 on beauty. **Any future agent: follow `docs/CALM_REC_RUNBOOK.md`.**

## 13. ARIS design-gate score (lead ruling, post-20-iteration)

| Axis | /10 | Note |
|---|---|---|
| Novelty | 9 | Headline = endogenous calibrated personalization-trust (multi-intent as vehicle); differentiated hard vs MIND/ComiRec, MoE, LLMEmb/ProEx, UGR/ConfTuner. Firewall (attribute anchor) identified + evidenced. |
| Falsifiability | 9 | Placebo-ρ + intent ablation + no-anchor + stage-2.5 AUC gate + cross-domain ≥5/7; pre-registered negatives; honesty fallbacks. |
| Feasibility | 8 | One 4090, cached encoder + vector-space scoring; hours on beauty. Main risk (reliability-signal correlation) has a cheap one-pass gate. |
| Depth / capacity | 9 | rank-1 → rank-K mixture + per-intent debiasing + representation-level uncertainty + calibrated trust; genuine mechanism depth, non-stitched. |
| Respects R3 negatives | 10 | No generation, no verbalized confidence, no accept/abstain gate; uncertainty mixes (never collapses to fallback). |
| **Overall** | **9.0** | **Clears the ≥8/10 gate. Locked for implementation.** |

## 14. Participants & record

Tri-agent ARIS, 5 rounds / 20 iterations: Opus 4.8 (lead, synthesis+ruling) + Opus 4.8 #2 (Agent
tool) + GPT-5.5 xhigh (relay `OPENAI_BASE_URL`, per `multi-agent-discussion-rule`). Iteration trail in
`outputs/method_redesign_discussion/iterations/round{1..5}_synthesis.md` + per-seat notes. Strong
independent convergence each round; both cross-seats explicitly endorsed locking CALM-Rec.
