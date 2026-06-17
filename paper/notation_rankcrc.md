# Notation — RankCRC

> Current notation for the RankCRC headline (replaces stale CU-GR `notation.md`, kept for history). Locked / experiment-independent.

## Users, items, base ranker
| Symbol | Meaning |
|---|---|
| `u`, `U`, `U*` | a user; a random user; a fresh test user |
| `H_u` | user `u`'s chronological interaction history |
| `C_u` | candidate set for `u` (101 = 1 positive + 100 popularity-matched negatives) |
| `y_u` | held-out target item (`∈ C_u`); used only for offline evaluation / label construction, never shown to the model |
| `k` | fixed ranking cutoff (metrics at `k ∈ {5,10,20}`; headline `k=10`) |
| `π_u` | top-`k` list emitted by the **frozen** base ranker |
| `H_target` | the held-out relevance signal — **FORBIDDEN** as a feature of `g` (no test peek) |

## Risk and confidence
| Symbol | Meaning |
|---|---|
| `ρ_u = 1 − NDCG@k(u) ∈ [0,1]` | per-user listwise rank-risk (fixed once base ranker + `k` are fixed) |
| `x_u` | 24-dim label-free rank-geometry feature vector (score margins, softmax peakiness, responsibility-entropy order-stats, MC-dropout variance, popularity) |
| `g(x_u) ∈ ℝ` | confidence (logistic over `x_u`), **label-free at inference**; higher ⇒ predicted-more-reliable. Fold-A training target is `ρ_u=1−NDCG@k` (derived from `y_u`); `g` reads no test-fold label at deployment |
| AUC | discrimination of `g` for high- vs low-risk users (beauty: 0.627 vs `H_top` 0.519) |

## Selection and certification
| Symbol | Meaning |
|---|---|
| `λ` (`≡ τ`) | abstention/coverage threshold: serve `u` iff `g(x_u) ≥ λ`. For fixed-`k` NDCG there is no within-slice knob, so `λ` equals the coverage threshold `τ` |
| `c` | target coverage (pre-registered grid {0.9,0.8,0.7,0.6,0.5}; fine grid 0.95…0.10 for the Pareto) |
| `S(λ) = {u : g(x_u) ≥ λ}` | served set (nested: `S(λ′) ⊆ S(λ)` for `λ′ > λ`) |
| `R(λ) = E[ρ_U · 1{g≥λ}]/E[1{g≥λ}] = E[1−NDCG@k | g≥λ]` | the (desired) **conditional served-slice risk** |
| `C(λ) = E[1{g≥λ}]` | coverage (the only exactly-monotone, nested member) |
| `N(λ) = E[ρ_U · 1{g≥λ}]` | unconditional numerator (a genuine monotone per-example CRC family; `R = N/C`) |
| `m` | realized **served-B2 count** `= |{u ∈ B2 : g(x_u) ≥ λ}|` (random) |
| `R̂_m` | served-B2 sample mean `= (1/m) Σ_{u∈B2: g≥λ} ρ_u` |
| `α` | CRC certificate `= R̂_m·m/(m+1) + 1/(m+1)` (`B=1`); an EXPECTATION-level random certificate per operating point |
| `δ` | (optional LTT upgrade) failure probability for a `(1−δ)` high-probability guarantee |
| `β`, `c₀` | (optional surrogate) numerator bound and coverage floor: `N(λ)≤β`, `C(λ)≥c₀` ⟹ `R ≤ β/c₀` |

## Fold discipline
| Symbol | Meaning |
|---|---|
| Fold **A** | fit + freeze `g` (`H_target` forbidden) |
| Fold **B1** | set `λ := Quantile_{1−c}(g)` (threshold-selection only) |
| Fold **B2** | compute `m`, `R̂_m`, `α` (loss-averaging only) — disjoint from B1 |
| Fold **C** | test: apply frozen `(g, λ)`, scored once |
| seed 20260506 | committed fold-manifest seed |

## Gain law (Prop 3, falsifiable)
| Symbol | Meaning |
|---|---|
| `ΔNDCG@k(c)` | served-slice NDCG lift at coverage `c` |
| `Gini` | score-dispersion Gini of the base ranker |
| `κ(c)` | coverage factor |
| Law | `ΔNDCG@k(c) ≈ (2·AUC − 1) · Gini · κ(c)`, pre-registered with an out-of-sample R² falsification bar |
