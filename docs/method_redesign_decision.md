# Method Redesign Decision — TRUCE-Rec "Ours"

**Date:** 2026-06-07
**Process:** tri-agent ARIS discussion (Opus 4.8 lead + Opus 4.8 #2 + GPT-5.5 xhigh), 5 steps.
**Status:** DESIGN LOCKED, passed the ARIS ≥8/10 design gate (see §6). Implementation pending;
formal run BLOCKED until the user frees the server (currently busy with another project).
**Replaces:** the `ours_uncertainty_guided` / conservative-gate route, which lost to fallback-only
in the R3 formal run (`docs/r3_ours_error_decomposition.md`). That route is retired for the main
method (kept only as legacy/diagnostic).

Raw discussion (brief, 3 independent proposals, cross-critiques, fork-resolution) is preserved in
`outputs/method_redesign_discussion/` (gitignored). This doc is the tracked decision.

---

## 1. The method: SCALR — Set-aware Calibrated Lift Ranking

> **One line.** Score each of the 101 candidates with a single Qwen3-8B+LoRA **cross-encoder that
> sees the whole panel at once**, by the **popularity-residual lift** of seeing the user's history
> (vs not), then apply **one additive, per-candidate uncertainty penalty** (panel-instability) and
> **one additive echo penalty** (history-near-duplicate risk). Uncertainty *reorders* candidates;
> it never gates, accepts, or abstains — so it cannot collapse to fallback the way R3 did.

This is the convergent synthesis of three independent proposals (CARP, CLR, PCUR). It keeps each
proposal's strongest, identifiable element and deletes every component the cross-critique proved to
be a ranking no-op or non-identifiable.

### Final scoring formula

For user `u` with history `H` and the fixed panel `C = {c_1..c_101}`:

```
Δ_i  = f_θ(H,  C, i)  −  f_θ(∅, C, i)        # popularity-residual "lift" (per-candidate, set-aware)
u_i  = std_r [ f_θ(H, C^(r), i) ]            # panel-instability uncertainty, R perturbations
echo_i = EchoHead(c_i, NN_H(c_i))            # near-duplicate-of-history risk in [0,1]

score_i = g(Δ_i)  −  λ · u_i  −  β · echo_i   # ADDITIVE; rank the 101 score_i, descending
```

- `f_θ` = the panel-conditioned cross-encoder readout (one Qwen3-8B+LoRA, §3).
- `∅` = a **null/history-masked** context: the same prompt with the user history removed (the
  candidate panel and instruction stay). Subtracting `f_θ(∅,C,i)` removes "this item is generically
  attractive / popular in this panel," isolating the user-specific signal. This counterfactual is
  per-candidate, so it genuinely reorders.
- `g(·)` = a single **global** monotone recalibration of the lift (one isotonic map fit on the
  calibration split), for probability honesty only. It is deliberately order-preserving on the raw
  lift, so it is NOT relied on to reorder — its value is in keeping `Δ` on a sane scale for the
  additive terms. **Ablatable** (see §5); the method must not depend on it for ranking gains.
- `u_i`, `echo_i` are **per-candidate**, so both reorder *within* a popularity bucket — the property
  the cross-critique proved is mandatory (NDCG/HR depend only on within-panel order; any term
  constant across a user's candidates is a no-op).
- `λ, β ≥ 0` selected on the **validation** split by NDCG@10. Two scalars only.

---

## 2. The ruling principle that shaped every design choice

> **Within-panel ranking is invariant to any score transform that is constant across a user's 101
> candidates, or globally monotone in the base score.** Therefore every term in `score_i` must
> EITHER reorder within a popularity bucket (be per-candidate) OR re-level across buckets. Pure
> no-ops are deleted.

All three seats independently arrived at this principle in the cross-critique. It is what kills the
weaker components below, and it is the litmus test for any future tweak to the method.

## 3. The cross-encoder readout (the one real feasibility risk, and its fix)

A naive "read the hidden state at the `[c_i]` index tag" is fragile — the tag token is a positional
list marker, not the item's semantics, and a per-candidate logit inside a 101-item context is
position-confounded (GPT-5.5 and Opus#2 both flagged this).

**Fix (locked):**
- Pool the readout over the **candidate's title token span**, not the index-tag token
  (mean/attention pool over the item's own tokens).
- Estimate `u_i` from **R ≥ 12** perturbations (random candidate orderings + history truncations) —
  R=4 is too few to marginalize 101 position slots. This TTA *is* the uncertainty source, so the
  cost is dual-purpose.
- If 101 items is too long for a clean readout on Qwen3-8B context, chunk into sub-panels (~25
  candidates incl. the positive's popularity-mates) and merge by lift; adds forward passes, keeps
  the readout clean. Decide empirically on beauty.

## 4. What was KEPT, DROPPED, and WHY (audit trail)

| Element | Origin | Verdict | Reason |
|---|---|---|---|
| Single 101-panel **cross-encoder** (set-aware) | CARP | **KEEP** | Pointwise/dual-encoder scoring (CLR) is panel-blind; the task is explicitly a 101-way competition where relative/substitution signal is the edge over LLMEmb/ProEx. |
| Candidate **title-span pooling** (not tag token) | GPT-5.5 critique | **KEEP** | Tag-token readout is a position artifact; span pooling is a stable item representation. |
| **Popularity-residual lift** `Δ_i = f(H,C,i) − f(∅,C,i)` | CARP info-gain × CLR lift | **KEEP** | Per-candidate, model-internal popularity de-confounding; directly attacks the dominant error (popular negative beats true positive) since negatives are popularity-sampled. Cleaner than CLR's external per-domain null. |
| **One** per-candidate **panel-instability** `u_i` | CARP | **KEEP** | Reorders within bucket; doubles as the TTA that fixes position confound. |
| **One** **echo head** `echo_i` (history-neighbor hard negatives) | PCUR | **KEEP** | Orthogonal, identifiable, per-candidate; targets a real rec failure (recommending near-duplicates of history) the others ignore. |
| **Additive** penalty form `−λu_i − β·echo_i` | CARP/Opus#2 | **KEEP** | See §4.1 — multiplicative shrinkage has a sign-flip bug and near-no-op risk. |
| Global monotone recalibration `g(Δ)` | CLR/CARP isotonic, de-scoped | **KEEP (honesty only, ablatable)** | A single global isotonic map is order-preserving on raw lift → not a reordering mechanism; kept only for probability honesty, never credited with ranking lift. |
| Split-**conformal** slack `τ_b` | CARP | **DROP** | `τ_b` is constant within a popularity bucket ⇒ within-bucket no-op; across buckets it's redundant with isotonic; marginal coverage ≠ ranking guarantee. (Opus#2 + lead proved it; GPT-5.5 concurred.) |
| **Per-bucket** isotonic as a reordering step | CARP | **DROP** | If most of a user's 101 candidates share a bucket, a per-bucket monotone map is within-panel order-preserving ⇒ near-no-op. Popularity is already removed per-candidate by the null contrast, so no per-bucket reordering calibration is needed. |
| Multiplicative James–Stein shrinkage `ρ_i Δ_i` | CLR | **DROP** | Sign-flip: when `Δ_i<0`, `ρ_i∈(0,1)` pulls it toward 0 = UP in rank (backwards); and within a bucket `ρ_i≈const` ⇒ near-no-op. |
| `σ_pop` / `σ_tail` heads | PCUR | **DROP** | Popularity and tail-ness are the same axis, opposite sign ⇒ coefficients non-identifiable; `L_pop` head just relearns popularity (debiasing in disguise). The null contrast already does the debiasing, once. |
| MC-dropout epistemic variance head | CLR/PCUR | **DROP** | Redundant with panel-instability `u_i` as the uncertainty source; avoids a second, weakly-calibrated variance estimator. |
| `δ·logit(cal_i)` additive term | PCUR | **DROP** | Monotone in `μ_i` ⇒ within-panel near-no-op. |
| Free-text generation + grounding + accept/abstain gate | R3 method | **DROP (retired)** | Lost to fallback; head-biased grounding; verbalized confidence ECE 0.85. The whole gating paradigm is abandoned. |

### 4.1 Why additive, not multiplicative (locked, both cross-seats agreed)

Multiplicative `score_i = ρ_i·g(Δ_i)`, `ρ_i=ω²/(ω²+τ_i²)` has two disqualifying problems: (1) when
`g(Δ_i)<0` (lift below the null), multiplying by `ρ_i∈(0,1)` moves the candidate *toward zero, i.e.
UP* in rank — backwards; (2) within a popularity bucket, `g(Δ_i)` and `τ_i` are similar in magnitude,
so `ρ_i≈const` and the term barely reorders. Additive `−λu_i − β·echo_i` reorders monotonically in
`u_i`/`echo_i` regardless of the lift's sign, with displacement set directly by `λ,β` (tunable on
val to a magnitude that actually moves NDCG). **Additive is locked.**

---

## 5. Training, leakage control, and rollout

**Backbone / adapter.** One Qwen3-8B, QLoRA (4-bit, rank 16) on attention+MLP projections, matching
the 8 baselines' controlled Qwen3-8B+LoRA setting. Item titles/metadata cacheable.

**Stage A — GPU (the only thing trained).** Listwise softmax cross-entropy on the lift over the 101
panel:

```
L_rank = − log [ exp(Δ_pos) / Σ_j exp(Δ_j) ]
L_echo = contrastive term teaching EchoHead to score history-near-duplicate hard negatives as high-echo
L = L_rank + γ · L_echo            # two losses, not five
```

History-neighbor hard negatives for `L_echo` are mined from items semantically close to the user's
history (the echo failure mode). `f_θ(∅,·)` is the same network with history masked — trained
jointly so the lift is well-defined.

**Stage B — CPU, no GPU.** Fit the single global monotone recalibration `g` on the disjoint
**calibration** split (honesty only). Select `λ, β` (and R) on the **validation** split by NDCG@10.

**Leakage control (hard).** Three disjoint splits: train (Stage A) / calibration (Stage B `g`) /
validation (`λ,β,R`). **Test split never touched for any selection.** Popularity counts, item cache,
echo-neighbor index, and the null context are built from **train interactions only**. The null
contrast masks the *same user's* history → no future leak.

**Rollout.** Beauty first (973 users, cheapest, fastest to falsify). If SOTA on beauty, roll to the
other 7 domains by re-running Stage A + re-fitting `g, λ, β` per domain (same protocol). If NOT SOTA
on beauty, re-convene the tri-agent discussion and iterate on beauty until it is — do not scale a
losing method.

## 6. Falsifiability — the beauty kill test (run FIRST, before any rollout)

From **one** Stage-A checkpoint on beauty, re-score the **validation** split (no GPU retrain) along a
nested ladder, plus a placebo and a diagnostic:

1. raw panel score `f_θ(H,C,i)`
2. + popularity-residual lift `Δ_i`
3. + global recalibration `g(Δ_i)`
4. **full** `g(Δ_i) − λu_i − β·echo_i`
5. **placebo:** replace `(λu_i + β·echo_i)` with **variance-matched per-candidate random noise**
6. **diagnostic:** bin candidates by `u_i` within each popularity bucket; check realized ranking
   error increases with `u_i`.

**The method is KILLED if any of:**
- (2) ≤ (1): the popularity-residual lift adds nothing → core mechanism dead.
- (4) ≤ (3): the uncertainty/echo terms (the TRUCE thesis) add nothing → not a TRUCE paper.
- (4) does **not** beat (5) the variance-matched-noise placebo beyond a paired permutation test:
  the "lift" is generic within-bucket tie-breaking, not signal → uncertainty machinery is ornamental.
- realized error is **flat across `u_i` bins** (diagnostic): `u_i` is uninformative → drop it.
- **the hard bar:** variant (4) must clear **ProEx NDCG@10 = 0.1506** on beauty (HR@10 0.2528 / MRR
  0.1429 as secondary) — winning the internal ablation is not enough; it must beat the SOTA baseline.

This is an afternoon of re-scoring on one 4090, and it independently tests every load-bearing claim.

## 7. ARIS design-gate scoring (lead ruling)

| Axis | Score /10 | Justification |
|---|---|---|
| **Novelty** | 8 | Set-aware panel cross-encoder + per-candidate popularity-residual *lift* + additive structured uncertainty (panel-instability + echo) is not any of the 8 baselines, nor UGR (RL reward weighting) / ConfTuner (verbalized Brier) / entropy-fairness (zero-shot token entropy). The combination is a coherent estimator, not a stitch. Not a 9-10 because each ingredient has individual precedent; the novelty is the rec-specific composition + the "uncertainty-reorders-not-gates" framing. |
| **Falsifiability** | 9 | Nested ladder + variance-matched-noise placebo + within-bucket error-vs-`u` diagnostic + hard ProEx bar. Each load-bearing claim has an independent, cheap (no-retrain) kill condition. |
| **Feasibility** | 8 | One Qwen3-8B+QLoRA on one 4090; beauty trains in hours; R≥12 TTA + dual (H/∅) passes are the main inference cost, bounded and dual-purpose. The one risk (101-item readout) has a concrete fallback (span pooling + sub-panel chunking). |
| **Respects R3 negatives** | 10 | No generation, no grounding, no verbalized confidence, no accept/abstain gate; uncertainty only reorders, so the worst case is "a competent panel scorer," never a fallback collapse. |
| **Overall** | **8.25** | **Clears the ≥8/10 gate. Cleared for implementation.** |

## 8. Open items before the formal run (no GPU needed)

1. ~~Implement SCALR in `src/llm4rec/methods/`~~ **DONE** — `src/llm4rec/methods/scalr.py`
   (pure scoring core + `SCALRRanker` + `PanelScorer` abstraction + `MockPanelScorer`).
2. ~~Mock/smoke test the scoring contract~~ **DONE** — `tests/unit/test_scalr.py` (13) +
   `tests/smoke/test_scalr_pipeline.py` (2, incl. end-to-end through the real evaluator). 15/15 pass,
   no regressions. The scoring contract (101 scores, schema `source_event_id,user_id,item_id,score`)
   is validated; uncertainty terms reorder; no gating/generation.
3. **TODO (no GPU):** decide the 101-item readout (full panel vs sub-panel chunking) on a fixture.
4. **TODO (no GPU):** implement the real `PanelScorer` (Qwen3-8B+LoRA, title-span pooling) behind the
   same interface; wire SCALR into the experiment runner/registry + a `configs/methods/scalr.yaml` and
   `configs/experiments/` entry; wire the beauty kill-test ladder (variants 1-6) as a re-scoring script.
5. **Server run only on the user's go-ahead** (server busy). First target: beat ProEx 0.1506 on beauty.

### Implementation notes (2026-06-07)

- The Qwen3-8B+LoRA cross-encoder `f_θ` is abstracted as `PanelScorer.score_panel(PanelScoreRequest)
  -> {candidate_id: score}`; SCALR's scoring/ranking logic has **zero GPU/model dependency** and is
  fully unit-tested with `MockPanelScorer`. The real scorer drops in behind this interface server-side.
- Pure, separately-tested functions: `popularity_residual_lift`, `panel_instability`, `echo_risk`,
  `global_recalibrate` (+ `MonotoneCalibration` PAV isotonic), `combine_scores` (additive form),
  `perturbed_panels` (R reorderings). `SCALRRanker` implements the `BaseRanker` `fit`/`rank` contract
  and builds train-only popularity + category-neighbor (echo) indices; `fit_calibration` fits the
  global monotone map on a disjoint split; `set_hyperparams` applies validation-selected `λ, β`.
- The echo neighbor index is currently a leakage-free category-overlap proxy; the trained adapter
  will supply a finer echo signal behind the same lookup.

## 9. Participants & record

- Brief: `outputs/method_redesign_discussion/00_brief.md`
- Proposals: `01_proposal_opus_lead.md` (CARP), `02_proposal_gpt55.md` (PCUR), `03_proposal_opus2.md` (CLR)
- Critiques: `04_critique_gpt55.md`, `05_critique_opus_lead.md`, Opus#2 critique (SCALR synthesis)
- Fork resolution: `06_gpt55_resolve.md` + Opus#2 resolution (additive; global-only calibration)
- GPT-5.5 reached via the relay (`OPENAI_BASE_URL`), `reasoning_effort=xhigh`, per
  `multi-agent-discussion-rule`. All three seats genuinely independent at proposal time.
