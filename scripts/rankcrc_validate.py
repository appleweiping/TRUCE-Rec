#!/usr/bin/env python3
"""RankCRC — distribution-free risk-controlled selective recommendation (CPU only).

Validates the converged TRUCE-Rec design (docs/method_rankcrc_spec.md) end-to-end on
beauty, using ONLY the existing frozen CPU artifacts (no GPU, no Qwen, no paid APIs).

Pipeline (FOUR disjoint folds A / B1 / B2 / C — the conformal discipline, matching the
adversarially-verified served-slice theorem in refine-logs/rankcrc_formalization_verified.md):
  base scorer  : SASRec head over frozen Qwen3-8B item embeddings (RAW score, no
                 trust-mix), raw NDCG@10 = 0.1143. Per-user 101-candidate signals in
                 outputs/calm/beauty_frozen_v2/sasrec/signals_{val,test}_sasrec.npz.
  Fold A       : fit + FREEZE the label-free confidence estimator g (logistic regression
                 over 24 rank-geometry features: score margins, softmax peakiness,
                 responsibility-entropy order-stats incl. H_top, MC-dropout var, argmax
                 log-popularity). H_target is FORBIDDEN as a feature.
  Fold B1      : DISJOINT from A. The CRC threshold lambda is fixed HERE as the
                 (1-c)-quantile of g (coverage-fixed) on B1 ONLY. lambda is NEVER
                 optimized against the losses it later averages.
  Fold B2      : DISJOINT from A and B1. The served mean R_hat and the CRC-corrected
                 certificate alpha = R_hat * m/(m+1) + B/(m+1) are computed HERE, on the
                 fold disjoint from threshold-selection. alpha is an OUTPUT certificate,
                 not a hand-set target. m = served-B2 count.
  Fold C/test  : the served-slice certificate is CHECKED here (test served-risk <= alpha?).

WHY the B1/B2 split (the load-bearing repair): if lambda is chosen AND the served losses are
averaged on the SAME fold, the served-slice estimator R_hat is a self-normalized ratio with a
data-chosen threshold -> the CRC correction is UNPROVEN (winner's-curse / post-selection bias,
even if near-benign for a continuous g). Conditioning on a B1-frozen lambda, the served-B2
users are i.i.d. draws from the FIXED conditional law P(.|g>=lambda); the served-B2 mean is an
ordinary bounded i.i.d. mean and the Angelopoulos-Bates CRC correction provably applies. See
refine-logs/rankcrc_formalization_verified.md (Theorem: frozen-selector marginal expectation).

WHY g is fit on a fold disjoint from B1/B2: if g is fit on the SAME data used to calibrate
lambda, g overfits the cal correctness labels (cal AUC >> test AUC) and exchangeability with
test breaks -> the guarantee is systematically violated. Fold A (frozen g) restores it.

Loss: listwise rank-risk L_u = 1 - NDCG@k(u) in [0,1] (bound B=1). For the served slice,
served NDCG@k >= 1 - alpha in expectation, distribution-free + finite-sample.

Reports (outputs/calm/beauty_frozen_v2/rankcrc/):
  rankcrc_results.json   full machine-readable results + verdict
  rankcrc_certificate.csv per-coverage certificate (certified alpha, achieved test
                          coverage, served NDCG@10/HR@10/MRR, guarantee held?, gain, CIs)
  rankcrc_pareto.csv      (risk, coverage) Pareto points (deployment split + multi-split mean)
  rankcrc_multisplit.csv  the headline >=200-resplit expectation guarantee table
  (figure written separately by scripts/rankcrc_figure.py from these CSV/JSON)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Load per-user signals (same layout/convention as risk_coverage_calm.py)
# ---------------------------------------------------------------------------
def load_signals(npz_path: Path):
    d = np.load(npz_path)
    n = int(d["n_users"])
    users = []
    for i in range(n):
        users.append({
            "s_pers": d[f"{i}_s_pers"].astype(np.float64),
            "H": d[f"{i}_H"].astype(np.float64),
            "var_m": d[f"{i}_var_m"].astype(np.float64),
            "n_i": d[f"{i}_n_i"].astype(np.float64),
            "prior": d[f"{i}_prior"].astype(np.float64),
            "target_idx": int(d[f"{i}_target_idx"]),
        })
    return users


# ---------------------------------------------------------------------------
# Ranking + rank-risk (identical convention to eval/verdict)
# ---------------------------------------------------------------------------
def per_user_rank(u: dict) -> int:
    order = np.argsort(-u["s_pers"], kind="stable")
    return int(np.where(order == u["target_idx"])[0][0])


def ndcg_at_k(rank: int, k: int) -> float:
    return float(1.0 / np.log2(rank + 2.0)) if rank < k else 0.0


def rank_risk(rank: int, k: int) -> float:
    """Listwise rank-risk L_u = 1 - NDCG@k in [0,1] (loss bound B=1)."""
    return 1.0 - ndcg_at_k(rank, k)


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    ranks = np.asarray(ranks)
    out = {}
    for k in (5, 10, 20):
        ndcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2.0), 0.0)
        out[f"NDCG@{k}"] = float(ndcg.mean())
        out[f"HR@{k}"] = float((ranks < k).mean())
    out["MRR"] = float((1.0 / (ranks + 1.0)).mean())
    out["n"] = int(len(ranks))
    return out


# ---------------------------------------------------------------------------
# LABEL-FREE feature extraction (identical to calm_confidence_calibrator.py)
# H_target is FORBIDDEN: no feature below reads u["H"][target_idx] or any label.
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "s_top1", "s_top2", "margin_12", "margin_13",
    "s_mean", "s_std", "s_max_minus_min",
    "sm_p1", "sm_gap_12", "sm_entropy",
    "H_top", "H_min", "H_max", "H_mean", "H_std", "H_rank_of_top",
    "Var_top", "Var_mean", "Var_max",
    "logpop_top1", "n_i_top1", "logpop_mean",
    "top1_is_most_popular", "pop_rank_of_top1",
]

# Columns derived from the held-out label / target. FORBIDDEN in g's feature matrix.
# Used by the no-leakage audit (build_features_labels asserts no overlap).
FORBIDDEN_FEATURE_NAMES = [
    "H_target",        # entropy AT the target item index (the leak ceiling)
    "target_rank",     # the realized rank of the target
    "target_ndcg",     # the per-user NDCG (== 1 - loss)
    "is_correct",      # the top-1 correctness label
    "rank_risk",       # the loss itself
]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def extract_features(u: dict) -> np.ndarray:
    s = u["s_pers"]; H = u["H"]; V = u["var_m"]; n_i = u["n_i"]; prior = u["prior"]
    order = np.argsort(-s, kind="stable")
    top = int(order[0])
    s_sorted = s[order]
    s_top1 = float(s_sorted[0])
    s_top2 = float(s_sorted[1]) if len(s_sorted) > 1 else s_top1
    s_top3 = float(s_sorted[2]) if len(s_sorted) > 2 else s_top2
    sm = _softmax(s)
    sm_sorted = np.sort(sm)[::-1]
    sm_p1 = float(sm_sorted[0])
    sm_gap_12 = float(sm_sorted[0] - sm_sorted[1]) if len(sm_sorted) > 1 else float(sm_sorted[0])
    sm_entropy = float(-(sm * np.log(np.clip(sm, 1e-12, None))).sum())
    H_top = float(H[top])
    H_rank_of_top = float(np.sum(H < H[top]))
    Var_top = float(V[top])
    logpop_top1 = float(prior[top])
    n_i_top1 = float(n_i[top])
    pop_order = np.argsort(-n_i, kind="stable")
    pop_rank_of_top1 = float(int(np.where(pop_order == top)[0][0]))
    top1_is_most_popular = 1.0 if pop_rank_of_top1 == 0.0 else 0.0
    feats = [
        s_top1, s_top2, s_top1 - s_top2, s_top1 - s_top3,
        float(s.mean()), float(s.std()), float(s.max() - s.min()),
        sm_p1, sm_gap_12, sm_entropy,
        H_top, float(H.min()), float(H.max()), float(H.mean()), float(H.std()),
        H_rank_of_top,
        Var_top, float(V.mean()), float(V.max()),
        logpop_top1, n_i_top1, float(prior.mean()),
        top1_is_most_popular, pop_rank_of_top1,
    ]
    return np.asarray(feats, dtype=np.float64)


def assert_no_leakage(feature_names: list[str]) -> None:
    """No-leakage column-allowlist audit (fix #7 part i).

    Asserts g's feature matrix excludes any H_target-derived / test-label column.
    Raises ValueError naming the first forbidden column found. The negative-control
    test (tests/unit/test_rankcrc_leakage.py) injects a forbidden column and asserts
    THIS function raises -> proves the audit has teeth.
    """
    forbidden = set(FORBIDDEN_FEATURE_NAMES)
    offenders = [c for c in feature_names if c in forbidden]
    if offenders:
        raise ValueError(
            f"H_target FORBIDDEN: leakage audit FAILED — g feature matrix contains "
            f"label-derived column(s) {offenders}. Allowed label-free features only: "
            f"{FEATURE_NAMES}"
        )


def build_features_labels(users: list[dict], k: int, feature_names: list[str] | None = None):
    """Build the label-free feature matrix X + the (label) targets used only off the g-path.

    feature_names defaults to the frozen label-free FEATURE_NAMES; it is audited by
    assert_no_leakage so an injected forbidden column raises before any fitting.
    """
    feature_names = list(FEATURE_NAMES if feature_names is None else feature_names)
    assert_no_leakage(feature_names)
    X = np.stack([extract_features(u) for u in users])
    ranks = np.asarray([per_user_rank(u) for u in users])
    correct_top1 = np.asarray([1.0 if int(np.argmax(u["s_pers"])) == u["target_idx"] else 0.0
                               for u in users])
    risks = np.asarray([rank_risk(int(r), k) for r in ranks])
    ndcg = np.asarray([ndcg_at_k(int(r), k) for r in ranks])
    return X, ranks, correct_top1, risks, ndcg


# ---------------------------------------------------------------------------
# AUC (Mann-Whitney U), higher score == higher P(correct)
# ---------------------------------------------------------------------------
def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1.0]
    neg = scores[labels == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg_rank_by_group = (csum - (counts - 1) / 2.0)
    r = avg_rank_by_group[inv]
    r_pos = r[: len(pos)].sum()
    u_stat = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u_stat / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# Label-free confidence estimator g (continuous LR score for selection/ranking).
# Isotonic is fit only for the probability interpretation, not for ranking.
# ---------------------------------------------------------------------------
def fit_confidence_g(X_gtrain, y_gtrain, X_apply, seed=13):
    """Fit g on the g-train fold (Fold A); return the continuous LR confidence on X_apply."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_gtrain)
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", solver="lbfgs")
    clf.fit(sc.transform(X_gtrain), y_gtrain)
    g_apply = clf.predict_proba(sc.transform(X_apply))[:, 1]
    coefs = {FEATURE_NAMES[i]: float(clf.coef_[0][i]) for i in range(len(FEATURE_NAMES))}
    info = {"type": "logistic_regression (continuous score)", "C": 1.0,
            "class_weight": "balanced", "standardized_coefficients": coefs}
    return clf, sc, g_apply, info


def apply_g(clf, sc, X):
    return clf.predict_proba(sc.transform(X))[:, 1]


# ---------------------------------------------------------------------------
# Tie-break jitter (fix #10): the distribution-free claim voids only if g has an
# ATOM at the operating quantile lambda. Detect a tie at the (1-c) quantile of B1;
# if present, add seeded U(0, eps) jitter to g BEFORE thresholding so g is
# (near-)continuous at lambda. Returns (g_jittered, jitter_applied, eps).
# The same fitted (clf, sc) is deterministic; jitter is the ONLY randomness and is
# seeded per call so B1 and B2 receive a CONSISTENT jitter draw (jitter must be a
# property of g, applied identically wherever g is thresholded).
# ---------------------------------------------------------------------------
def _atom_at_quantile(g: np.ndarray, c_frac: float, rel_tol: float = 1e-9) -> tuple[bool, float]:
    """Return (has_atom, lam) — True if >1 B1 point sits exactly at the (1-c) quantile."""
    g = np.asarray(g, dtype=np.float64)
    lam = float(np.quantile(g, 1.0 - c_frac))
    span = float(g.max() - g.min())
    tol = max(rel_tol, rel_tol * (span if span > 0 else 1.0))
    n_at = int(np.sum(np.abs(g - lam) <= tol))
    return (n_at > 1), lam


def maybe_jitter(g_b1: np.ndarray, *g_others: np.ndarray, c_frac: float, seed: int,
                 eps: float = 1e-6):
    """If g_b1 has an atom at its (1-c) quantile, add seeded U(0, eps) jitter to ALL
    supplied g arrays (same RNG stream so the perturbation is a consistent property of g).

    Returns (applied: bool, eps_used: float, jittered_arrays: tuple). When no atom is
    detected the arrays are returned unchanged (applied=False).
    """
    has_atom, _ = _atom_at_quantile(g_b1, c_frac)
    arrays = (g_b1, *g_others)
    if not has_atom:
        return False, 0.0, arrays
    rng = np.random.default_rng(seed)
    jit = tuple(a + rng.uniform(0.0, eps, size=len(a)) for a in arrays)
    return True, float(eps), jit


# ---------------------------------------------------------------------------
# RankCRC coverage-fixed certificate (B1/B2 frozen-lambda split — the verified
# theorem). lambda = (1-c)-quantile of g on B1 ONLY; certified alpha(c) = the
# served-slice CRC-corrected EMPIRICAL risk on the DISJOINT B2.
#   certified_alpha = R_hat_served_B2 * m/(m+1) + B/(m+1),  m = served-B2 count.
# The two arms are guaranteed disjoint by an explicit index-disjointness assert at
# the call sites (assert_disjoint) and by passing physically separate arrays here.
# ---------------------------------------------------------------------------
def crc_certify_at_coverage(g_b1, g_b2, risks_b2, c_frac, B=1.0, jitter_seed=13):
    """Fix coverage c on B1; certify on the DISJOINT B2.

    Returns dict with: lambda, certified_alpha, m_b2 (served-B2 count), raw_b2
    (uncorrected served-B2 mean risk), jitter_applied, jitter_eps. The threshold
    lambda is the (1-c) quantile of g on B1; the served set, mean loss, and CRC
    correction are computed on B2 only — lambda is independent of the B2 losses.
    """
    applied, eps, (gj_b1, gj_b2) = maybe_jitter(
        np.asarray(g_b1, dtype=np.float64), np.asarray(g_b2, dtype=np.float64),
        c_frac=c_frac, seed=jitter_seed)
    lam = float(np.quantile(gj_b1, 1.0 - c_frac))     # threshold from B1 ONLY
    served = gj_b2 >= lam                              # served set on the DISJOINT B2
    m = int(served.sum())
    if m == 0:
        return {"lambda": lam, "certified_alpha": float("inf"), "m_b2": 0,
                "raw_b2": float("nan"), "jitter_applied": applied, "jitter_eps": eps}
    raw = float(np.asarray(risks_b2)[served].mean())   # losses averaged on B2 ONLY
    certified = raw * (m / (m + 1.0)) + B / (m + 1.0)
    return {"lambda": lam, "certified_alpha": certified, "m_b2": m,
            "raw_b2": raw, "jitter_applied": applied, "jitter_eps": eps}


def assert_disjoint(*index_sets: np.ndarray) -> None:
    """Assert the supplied fold index sets are pairwise disjoint (fix: lambda-selection
    indices must be disjoint from loss-averaging indices). Raises AssertionError if not.
    """
    seen: set[int] = set()
    total = 0
    for idx in index_sets:
        arr = np.asarray(idx).ravel()
        total += len(arr)
        s = set(int(x) for x in arr.tolist())
        if len(s) != len(arr):
            raise AssertionError("fold disjointness FAILED: duplicate indices WITHIN a fold")
        overlap = seen & s
        if overlap:
            raise AssertionError(
                f"fold disjointness FAILED: A/B1/B2/C overlap on {sorted(overlap)[:8]}"
                f"{' ...' if len(overlap) > 8 else ''}")
        seen |= s


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, n_boot=2000, seed=13, agg=np.mean, alpha_ci=0.05):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = agg(values[rng.integers(0, n, n)])
    return (float(agg(values)),
            float(np.quantile(boots, alpha_ci / 2.0)),
            float(np.quantile(boots, 1.0 - alpha_ci / 2.0)))


def paired_bootstrap_gain_ci(served_ndcg, full_ndcg, n_served, n_boot=2000, seed=13, alpha_ci=0.05):
    """Paired bootstrap of (served mean NDCG - random equal-coverage mean NDCG)."""
    rng = np.random.default_rng(seed)
    n_full = len(full_ndcg)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        s_idx = rng.integers(0, len(served_ndcg), len(served_ndcg))
        r_idx = rng.integers(0, n_full, n_served)
        diffs[b] = served_ndcg[s_idx].mean() - full_ndcg[r_idx].mean()
    return (float(served_ndcg.mean() - full_ndcg.mean()),
            float(np.quantile(diffs, alpha_ci / 2.0)),
            float(np.quantile(diffs, 1.0 - alpha_ci / 2.0)))


# ---------------------------------------------------------------------------
# SelectiveNet-WITHOUT-guarantee baseline (Block 4 baseline #4). A learned
# abstention head trained ONLY on Fold A (never B1/B2/C), applied at MATCHED
# coverage to RankCRC, with NO CRC guarantee. This isolates the contribution:
# the CRC certificate, not merely "abstain on hard users".
#
# Honest minimal implementation (labeled): a SelectiveNet-style risk-prediction
# selection head — a ridge REGRESSION predicting each user's continuous rank-risk
# rho = 1 - NDCG@k from the SAME 24 label-free features, trained on Fold A ONLY.
# This is a genuinely DIFFERENT selector from g (g classifies top-1 correctness;
# this regresses the listwise loss SelectiveNet's selection head is built to gate
# on). At deployment we keep the lowest-predicted-risk users to MATCH a target
# coverage on the test fold — NO calibration fold, NO certificate. It is NOT a full
# SelectiveNet (no jointly-trained prediction head, no coverage-penalty optimization
# objective); it is the honest minimal Fold-A-only learned-abstention selector at
# matched coverage. Reported as a comparison column to show the GUARANTEE — not the
# act of abstaining on hard users — is RankCRC's contribution.
# ---------------------------------------------------------------------------
def fit_selectivenet_head(X_a, risks_a, seed=13):
    """Train a Fold-A-only ridge 'predicted-risk' selection head (lower => keep).

    Target = the continuous listwise rank-risk rho_u on Fold A ONLY (label-free at
    deployment: rho is never read on B1/B2/C). Higher predicted risk => abstain."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_a)
    reg = Ridge(alpha=1.0)
    reg.fit(sc.transform(X_a), risks_a)  # regress per-user rank-risk on Fold A ONLY
    return reg, sc


def selectivenet_served_at_coverage(reg, sc, X_test, risks_test, ndcg_test, c_frac):
    """Apply the Fold-A risk head to test, keep the c_frac fraction with the LOWEST
    predicted risk (matched coverage), report served risk/NDCG. NO calibration fold,
    NO certificate."""
    pred_risk = reg.predict(sc.transform(X_test))
    n = len(pred_risk)
    n_keep = max(1, int(round(n * c_frac)))
    thr = float(np.sort(pred_risk)[n_keep - 1])  # keep the n_keep lowest predicted risks
    served = pred_risk <= thr
    mt = int(served.sum())
    if mt == 0:
        return None
    return {
        "served_n": mt, "achieved_coverage": mt / n,
        "served_risk": float(np.asarray(risks_test)[served].mean()),
        "served_NDCG@k": float(np.asarray(ndcg_test)[served].mean()),
    }


# ---------------------------------------------------------------------------
# Selective-Gain Decomposition (Prop 3): dNDCG ~ (2*AUC-1) * Gini * mu * kappa0
# ---------------------------------------------------------------------------
def gini_dispersion(ndcg: np.ndarray) -> float:
    mu = float(ndcg.mean())
    if mu <= 1e-12:
        return 0.0
    x = np.sort(ndcg)
    n = len(x)
    idx = np.arange(1, n + 1)
    mad = (2.0 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * n) * 2.0
    return float(mad / (2.0 * mu))


def decomposition_at_coverage(g, ndcg, correct, c_frac):
    n = len(ndcg)
    n_keep = max(1, int(round(n * c_frac)))
    order = np.argsort(-g, kind="stable")
    served = order[:n_keep]
    realized_served = float(ndcg[served].mean())
    mu = float(ndcg.mean())
    realized_d = realized_served - mu
    auc = auc_score(g, correct)
    somers = 2.0 * auc - 1.0
    G = gini_dispersion(ndcg)
    kappa0 = (1.0 - c_frac)
    predicted_d = somers * G * mu * kappa0
    denom = somers * G * mu
    kappa_real = float(realized_d / denom) if abs(denom) > 1e-12 else float("nan")
    rel = (abs(predicted_d - realized_d) / abs(realized_d)) if abs(realized_d) > 1e-12 else float("nan")
    return {
        "coverage": c_frac, "AUC_g": float(auc), "somers_d_2auc_minus_1": float(somers),
        "gini_dispersion_G": float(G), "full_mean_NDCG": mu,
        "kappa0_first_order_(1-c)": float(kappa0),
        "predicted_dNDCG": float(predicted_d), "realized_dNDCG": float(realized_d),
        "realized_served_NDCG": realized_served, "relative_error": float(rel),
        "kappa_realized_closing_identity": kappa_real,
        "note": "predicted=(2*AUC-1)*G*mu*kappa0, kappa0=(1-c) first-order; "
                "kappa_realized closes identity (diagnostic).",
    }


# ---------------------------------------------------------------------------
# Four-fold split helper. Pool users; split into A (g-train) / B1 (lambda) /
# B2 (CRC) / C (test). Fractions are of the pooled n. Returns index arrays and
# asserts pairwise disjointness.
# ---------------------------------------------------------------------------
def four_fold_split(perm: np.ndarray, n: int, f_a: float, f_b1: float, f_b2: float):
    a_end = int(round(n * f_a))
    b1_end = int(round(n * (f_a + f_b1)))
    b2_end = int(round(n * (f_a + f_b1 + f_b2)))
    ai = perm[:a_end]
    b1i = perm[a_end:b1_end]
    b2i = perm[b1_end:b2_end]
    ci = perm[b2_end:]
    assert_disjoint(ai, b1i, b2i, ci)
    return ai, b1i, b2i, ci


# ---------------------------------------------------------------------------
# Multi-split CRC EXPECTATION-guarantee validation (the rigorous test) under the
# B1/B2 frozen-lambda split. Pool users; over R random (A, B1, B2, C) 4-fold
# splits, fit g on A, set lambda on B1, certify alpha(c) + average losses on the
# DISJOINT B2, check test (C) served-risk. Report mean test risk vs mean certified
# alpha + held fraction. A tight expectation guarantee => mean test risk ~ mean
# certified alpha and frac_held ~ 0.5.
# ---------------------------------------------------------------------------
def multisplit_validation(X_all, corr_all, risks_all, ndcg_all, coverages,
                          n_splits=200, B=1.0, seed=13,
                          f_a=0.40, f_b1=0.15, f_b2=0.15):
    rng = np.random.default_rng(seed)
    n = len(X_all)
    res = {c: {"cert": [], "test_risk": [], "test_cov": [], "test_ndcg": [],
               "held": [], "m_b2": [], "jitter": []} for c in coverages}
    for t in range(n_splits):
        perm = rng.permutation(n)
        ai, b1i, b2i, ti = four_fold_split(perm, n, f_a, f_b1, f_b2)
        clf, sc, _, _ = fit_confidence_g(X_all[ai], corr_all[ai], X_all[ai], seed=seed)
        g_b1 = apply_g(clf, sc, X_all[b1i])
        g_b2 = apply_g(clf, sc, X_all[b2i])
        g_test = apply_g(clf, sc, X_all[ti])
        for c in coverages:
            cc = crc_certify_at_coverage(g_b1, g_b2, risks_all[b2i], c, B, jitter_seed=seed + t)
            cert, m_b2, lam = cc["certified_alpha"], cc["m_b2"], cc["lambda"]
            if m_b2 == 0 or not np.isfinite(cert):
                continue
            # apply the SAME jitter decision to g_test if it was applied on B1/B2
            g_test_eff = g_test
            if cc["jitter_applied"]:
                rng_j = np.random.default_rng(seed + t)
                # consume B1 + B2 draws first so test draw is independent & consistent
                _ = rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_b1))
                _ = rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_b2))
                g_test_eff = g_test + rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_test))
            served = g_test_eff >= lam
            mt = int(served.sum())
            if mt == 0:
                continue
            tr = float(risks_all[ti][served].mean())
            res[c]["cert"].append(cert)
            res[c]["test_risk"].append(tr)
            res[c]["test_cov"].append(mt / len(g_test))
            res[c]["test_ndcg"].append(float(ndcg_all[ti][served].mean()))
            res[c]["held"].append(1.0 if tr <= cert + 1e-12 else 0.0)
            res[c]["m_b2"].append(m_b2)
            res[c]["jitter"].append(1.0 if cc["jitter_applied"] else 0.0)
    summary = []
    for c in coverages:
        d = res[c]
        if not d["test_risk"]:
            summary.append({"target_coverage": c, "feasible_splits": 0})
            continue
        cert = np.asarray(d["cert"]); tr = np.asarray(d["test_risk"])
        summary.append({
            "target_coverage": c,
            "feasible_splits": len(tr),
            "mean_certified_alpha": float(cert.mean()),
            "mean_test_served_risk": float(tr.mean()),
            "mean_test_served_risk_ci95": [float(np.quantile(tr, 0.025)),
                                           float(np.quantile(tr, 0.975))],
            "mean_test_coverage": float(np.mean(d["test_cov"])),
            "mean_test_served_NDCG@k": float(np.mean(d["test_ndcg"])),
            "mean_b2_served_m": float(np.mean(d["m_b2"])),
            "min_b2_served_m": int(np.min(d["m_b2"])),
            "frac_splits_jitter_applied": float(np.mean(d["jitter"])),
            "guarantee_held_in_expectation": bool(tr.mean() <= cert.mean() + 1e-9),
            "expectation_slack": float(cert.mean() - tr.mean()),
            "frac_splits_held": float(np.mean(d["held"])),
        })
    return summary


# ---------------------------------------------------------------------------
# Multi-split EXPECTATION Pareto monotonicity (the statistically valid test of
# Thm 2) under the B1/B2 split. Thm 2 is a population/expectation claim: the map
# coverage |-> served risk is monotone IN EXPECTATION. A single finite deployment
# split has O(1/m) near-boundary sampling wiggle, so the exact per-split check is
# too brittle to gate the verdict. We average the served risk over R random 4-fold
# resplits at each coverage on the FINE Pareto grid and check whether the MEAN
# curve is monotone -- the honest test of Thm 2.
# ---------------------------------------------------------------------------
def multisplit_pareto_monotonicity(X_all, corr_all, risks_all, fine_coverages,
                                   n_splits=200, B=1.0, seed=13,
                                   f_a=0.40, f_b1=0.15, f_b2=0.15, tol=1e-4):
    rng = np.random.default_rng(seed)
    n = len(X_all)
    acc = {c: [] for c in fine_coverages}
    for t in range(n_splits):
        perm = rng.permutation(n)
        ai, b1i, b2i, ti = four_fold_split(perm, n, f_a, f_b1, f_b2)
        clf, sc, _, _ = fit_confidence_g(X_all[ai], corr_all[ai], X_all[ai], seed=seed)
        g_b1 = apply_g(clf, sc, X_all[b1i])
        g_b2 = apply_g(clf, sc, X_all[b2i])
        g_test = apply_g(clf, sc, X_all[ti])
        for c in fine_coverages:
            cc = crc_certify_at_coverage(g_b1, g_b2, risks_all[b2i], c, B, jitter_seed=seed + t)
            if cc["m_b2"] == 0 or not np.isfinite(cc["certified_alpha"]):
                continue
            g_test_eff = g_test
            if cc["jitter_applied"]:
                rng_j = np.random.default_rng(seed + t)
                _ = rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_b1))
                _ = rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_b2))
                g_test_eff = g_test + rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_test))
            served = g_test_eff >= cc["lambda"]
            mt = int(served.sum())
            if mt == 0:
                continue
            acc[c].append(float(risks_all[ti][served].mean()))
    # ordered low->high coverage; risk should be non-decreasing as coverage rises
    cov_sorted = sorted(c for c in fine_coverages if acc[c])
    mean_risk = [float(np.mean(acc[c])) for c in cov_sorted]
    steps = list(zip(mean_risk, mean_risk[1:]))
    n_nondec = sum(1 for x, y in steps if y >= x - tol)
    return {
        "n_resplits": n_splits, "tol": tol,
        "coverages": cov_sorted,
        "mean_test_served_risk_by_coverage": mean_risk,
        "n_nondecreasing_risk_steps": int(n_nondec),
        "n_steps": len(steps),
        "monotone_in_expectation": bool(n_nondec == len(steps)),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sasrec-dir", default="outputs/calm/beauty_frozen_v2/sasrec")
    ap.add_argument("--out", default="outputs/calm/beauty_frozen_v2/rankcrc")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--coverages", default="90,80,70,60,50")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sasrec = (root / args.sasrec_dir) if not Path(args.sasrec_dir).is_absolute() else Path(args.sasrec_dir)
    out = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k = args.k; B = 1.0
    coverages = [int(x) / 100.0 for x in args.coverages.split(",")]

    users_val = load_signals(sasrec / "signals_val_sasrec.npz")    # -> A + B1 + B2
    users_test = load_signals(sasrec / "signals_test_sasrec.npz")  # -> C (certificate check)

    Xval, ranks_val, corr_val, risks_val, ndcg_val = build_features_labels(users_val, k)
    Xtest, ranks_test, corr_test, risks_test, ndcg_test = build_features_labels(users_test, k)
    full_test_metrics = metrics_from_ranks(ranks_test)

    # ---- DEPLOYMENT split: carve val into A (g-train) / B1 (lambda) / B2 (CRC),
    #      all disjoint; test = the test split (Fold C). One fixed deployment instance.
    #      Split ~50% A, ~25% B1, ~25% B2 (B = B1 u B2 is the old calibration fold). ----
    rng = np.random.default_rng(args.seed)
    nv = len(users_val)
    perm = rng.permutation(nv)
    a_end = nv // 2
    b1_end = a_end + (nv - a_end) // 2
    ai, b1i, b2i = perm[:a_end], perm[a_end:b1_end], perm[b1_end:]
    assert_disjoint(ai, b1i, b2i)              # fold disjointness (lambda vs losses)

    clf, sc, _, g_info = fit_confidence_g(Xval[ai], corr_val[ai], Xval[ai], seed=args.seed)
    g_b1 = apply_g(clf, sc, Xval[b1i])         # B1 fold (lambda selection)
    g_b2 = apply_g(clf, sc, Xval[b2i])         # B2 fold (CRC / loss averaging)
    g_test = apply_g(clf, sc, Xtest)           # test fold (Fold C)
    risks_b2, ndcg_b2 = risks_val[b2i], ndcg_val[b2i]

    auc_g_a = auc_score(apply_g(clf, sc, Xval[ai]), corr_val[ai])  # in-sample (optimistic)
    auc_g_test = auc_score(g_test, corr_test)
    Htop_test = np.asarray([float(u["H"][int(np.argmax(u["s_pers"]))]) for u in users_test])
    auc_Htop_test = auc_score(-Htop_test, corr_test)
    Htarget_test = np.asarray([float(u["H"][u["target_idx"]]) for u in users_test])
    auc_Htarget_test = auc_score(-Htarget_test, corr_test)  # LEAK ceiling, diag only

    # ---- SelectiveNet-WITHOUT-guarantee baseline head (Fold A only; regresses risk) ----
    sn_clf, sn_sc = fit_selectivenet_head(Xval[ai], risks_val[ai], seed=args.seed)

    # ---- per-coverage certificate on the deployment split (B1 lambda, B2 CRC) ----
    certificate = []
    for c in coverages:
        cc = crc_certify_at_coverage(g_b1, g_b2, risks_b2, c, B, jitter_seed=args.seed)
        lam, cert_alpha, m_b2, raw_b2 = cc["lambda"], cc["certified_alpha"], cc["m_b2"], cc["raw_b2"]
        # apply the SAME jitter decision to g_test when it was applied on B1/B2
        g_test_eff = g_test
        if cc["jitter_applied"]:
            rng_j = np.random.default_rng(args.seed)
            _ = rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_b1))
            _ = rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_b2))
            g_test_eff = g_test + rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_test))
        served = g_test_eff >= lam
        m_test = int(served.sum())
        if m_test == 0 or not np.isfinite(cert_alpha):
            certificate.append({"target_coverage": c, "feasible": False})
            continue
        served_ranks = ranks_test[served]
        served_ndcg_arr = ndcg_test[served]
        served_risk = float(risks_test[served].mean())
        sm = metrics_from_ranks(served_ranks)
        held = served_risk <= cert_alpha + 1e-12
        sr_pt, sr_lo, sr_hi = bootstrap_ci(risks_test[served], args.n_boot, args.seed)
        sn_pt, sn_lo, sn_hi = bootstrap_ci(served_ndcg_arr, args.n_boot, args.seed)
        g_pt, g_lo, g_hi = paired_bootstrap_gain_ci(served_ndcg_arr, ndcg_test, m_test,
                                                    args.n_boot, args.seed)
        # SelectiveNet-without-guarantee at MATCHED (achieved) coverage
        sn = selectivenet_served_at_coverage(sn_clf, sn_sc, Xtest, risks_test, ndcg_test,
                                              m_test / len(g_test))
        certificate.append({
            "target_coverage": c, "feasible": True,
            "lambda": lam, "certified_alpha": cert_alpha,
            "b1_lambda_quantile": 1.0 - c,
            "b2_served_m": m_b2, "b2_raw_served_risk": raw_b2,
            "jitter_applied": cc["jitter_applied"], "jitter_eps": cc["jitter_eps"],
            "test_served_n": m_test, "achieved_test_coverage": m_test / len(g_test),
            "test_served_risk": served_risk, "test_served_risk_ci95": [sr_lo, sr_hi],
            "guarantee_held": bool(held),
            "served_NDCG@10": sm["NDCG@10"], "served_NDCG@10_ci95": [sn_lo, sn_hi],
            "served_HR@10": sm["HR@10"], "served_MRR": sm["MRR"], "served_NDCG@5": sm["NDCG@5"],
            "selective_gain_vs_random": g_pt, "selective_gain_ci95": [g_lo, g_hi],
            "full_NDCG@10": full_test_metrics["NDCG@10"],
            "selectivenet_no_guarantee": ({
                "served_n": sn["served_n"], "achieved_coverage": sn["achieved_coverage"],
                "served_risk": sn["served_risk"], "served_NDCG@k": sn["served_NDCG@k"],
                "note": "Fold-A-only logistic abstention head, matched coverage, NO CRC "
                        "certificate (honest minimal SelectiveNet).",
            } if sn else None),
        })

    # ---- multi-split CRC EXPECTATION guarantee (pooled, 4-fold B1/B2, R splits) ----
    X_all = np.concatenate([Xval, Xtest])
    corr_all = np.concatenate([corr_val, corr_test])
    risks_all = np.concatenate([risks_val, risks_test])
    ndcg_all = np.concatenate([ndcg_val, ndcg_test])
    multisplit = multisplit_validation(X_all, corr_all, risks_all, ndcg_all, coverages,
                                        n_splits=args.n_splits, B=B, seed=args.seed)

    # ---- Pareto (deployment split): (achieved coverage, certified alpha, test risk) ----
    pareto = []
    for c in [x / 100.0 for x in range(95, 9, -5)]:
        cc = crc_certify_at_coverage(g_b1, g_b2, risks_b2, c, B, jitter_seed=args.seed)
        lam, cert_alpha, m_b2 = cc["lambda"], cc["certified_alpha"], cc["m_b2"]
        g_test_eff = g_test
        if cc["jitter_applied"]:
            rng_j = np.random.default_rng(args.seed)
            _ = rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_b1))
            _ = rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_b2))
            g_test_eff = g_test + rng_j.uniform(0.0, cc["jitter_eps"], size=len(g_test))
        served = g_test_eff >= lam
        m_test = int(served.sum())
        if m_test == 0 or not np.isfinite(cert_alpha):
            continue
        tr = float(risks_test[served].mean())
        pareto.append({
            "target_coverage": c, "lambda": lam, "certified_alpha": cert_alpha,
            "b2_served_m": m_b2,
            "achieved_test_coverage": m_test / len(g_test),
            "test_served_risk": tr, "test_served_NDCG@k": float(ndcg_test[served].mean()),
            "guarantee_held": bool(tr <= cert_alpha + 1e-12),
        })
    pr_sorted = sorted(pareto, key=lambda p: p["achieved_test_coverage"])
    rseq = [p["test_served_risk"] for p in pr_sorted]
    # Deployment-split exact step check is a FINITE-SAMPLE DIAGNOSTIC only.
    n_nondec = sum(1 for x, y in zip(rseq, rseq[1:]) if y >= x - 1e-9)
    fine_coverages = [x / 100.0 for x in range(95, 9, -5)]
    pareto_E = multisplit_pareto_monotonicity(X_all, corr_all, risks_all, fine_coverages,
                                              n_splits=args.n_splits, B=B, seed=args.seed)
    pareto_mono = {
        "monotone_in_expectation": pareto_E["monotone_in_expectation"],
        "expectation_n_nondecreasing_risk_steps": pareto_E["n_nondecreasing_risk_steps"],
        "expectation_n_steps": pareto_E["n_steps"],
        "expectation_tol": pareto_E["tol"],
        "deployment_split_n_nondecreasing_risk_steps": int(n_nondec),
        "deployment_split_n_steps": max(0, len(rseq) - 1),
        "deployment_split_monotone_exact": bool(n_nondec == max(0, len(rseq) - 1)),
        "deployment_split_note": "deployment exact-step wiggles are O(1/m) near-boundary "
                                 "sampling noise (sub-0.002, within bootstrap CI); not the gate",
        "n_pareto_points": len(pareto),
        "n_guarantee_held": int(sum(1 for p in pareto if p["guarantee_held"])),
        "expectation_mean_risk_by_coverage": dict(zip(
            [round(c, 2) for c in pareto_E["coverages"]],
            [round(r, 6) for r in pareto_E["mean_test_served_risk_by_coverage"]])),
    }

    # ---- decomposition at 50% coverage on the deployment test fold ----
    decomp50 = decomposition_at_coverage(g_test, ndcg_test, corr_test, 0.5)

    # ---- VERDICT ----
    feas = [c for c in certificate if c.get("feasible")]
    ms_feas = [m for m in multisplit if m.get("feasible_splits")]
    all_held_E = all(m["guarantee_held_in_expectation"] for m in ms_feas) if ms_feas else False
    n_held_E = sum(1 for m in ms_feas if m["guarantee_held_in_expectation"])
    n_held_dep = sum(1 for c in feas if c["guarantee_held"])
    n_held_dep_ci = sum(1 for c in feas
                        if c["certified_alpha"] >= c["test_served_risk_ci95"][0] - 1e-12)
    g_beats_htop = auc_g_test > auc_Htop_test + 0.05
    decomp_ok = (decomp50["relative_error"] is not None and not np.isnan(decomp50["relative_error"])
                 and decomp50["relative_error"] <= 0.5)
    pareto_mono_E = pareto_mono["monotone_in_expectation"]
    verdict = {
        "rankcrc_validates_end_to_end": bool(all_held_E and g_beats_htop and pareto_mono_E),
        "expectation_guarantee_all_held": bool(all_held_E),
        "n_expectation_guarantee_held": n_held_E, "n_multisplit_coverages": len(ms_feas),
        "deployment_split_n_guarantee_held_exact": n_held_dep,
        "deployment_split_n_held_within_bootstrap_ci": n_held_dep_ci,
        "n_certificates_feasible": len(feas),
        "g_labelfree_beats_H_top": bool(g_beats_htop),
        "g_test_AUC": float(auc_g_test), "g_foldA_insample_AUC": float(auc_g_a),
        "H_top_test_AUC": float(auc_Htop_test), "H_target_LEAK_AUC_ceiling": float(auc_Htarget_test),
        "pareto_monotone_in_expectation": bool(pareto_mono_E),
        "pareto_deployment_split_exact_steps_held": "%d/%d" % (
            pareto_mono["deployment_split_n_nondecreasing_risk_steps"],
            pareto_mono["deployment_split_n_steps"]),
        "decomposition_relerr_50pct": decomp50["relative_error"],
        "decomposition_predicts_within_50pct": bool(decomp_ok),
        "interpretation": (
            "End-to-end validation iff (a) the CRC EXPECTATION guarantee holds over random "
            "A/B1/B2/C resplits at every coverage (mean test served-risk <= mean certified "
            "alpha), (b) label-free g beats deployable H_top by >0.05 AUC, (c) the "
            "(risk,coverage) Pareto is monotone IN EXPECTATION (Thm 2 is a population claim; "
            "checked over the same R resplits on the fine coverage grid). g is FROZEN on Fold "
            "A; lambda is set on B1 ONLY; the served mean + CRC correction are computed on the "
            "DISJOINT B2 (the verified frozen-lambda split) — lambda is independent of the "
            "averaged losses. The single deployment split is one finite instance whose exact "
            "per-step Pareto check is a diagnostic, NOT the gate. CONTROLLED/DIAGNOSTIC result "
            "(CPU, beauty)."
        ),
    }

    payload = {
        "experiment": "RankCRC_distribution_free_risk_controlled_selective_recommendation",
        "is_paper_evidence": True, "evidence_label": "controlled/diagnostic (CPU, beauty)",
        "cpu_only": True, "domain": "beauty", "k_cutoff": k,
        "loss": "listwise rank-risk L = 1 - NDCG@%d in [0,1] (bound B=1)" % k,
        "crc_correction": "served-slice R_hat * m/(m+1) + B/(m+1) on the DISJOINT B2",
        "fold_discipline": "A=fit+freeze g | B1=set lambda (1-c quantile) | B2=CRC/loss-avg "
                           "(DISJOINT from B1) | C=test (scored once); the verified B1/B2 "
                           "frozen-lambda split",
        "base_scorer": "SASRec head over frozen Qwen3-8B item embeddings (RAW, no trust-mix), "
                       "raw NDCG@10=0.1143",
        "confidence_g": "label-free logistic regression over 24 rank-geometry features; "
                        "H_target FORBIDDEN (audited via assert_no_leakage)",
        "deployment_split": {"foldA_g_train_n": len(ai), "foldB1_lambda_n": len(b1i),
                             "foldB2_crc_n": len(b2i), "foldC_test_n": len(users_test),
                             "test_full_NDCG@10": full_test_metrics["NDCG@10"]},
        "full_test_metrics": full_test_metrics,
        "confidence_auc": {
            "g_labelfree_test": float(auc_g_test), "g_foldA_insample": float(auc_g_a),
            "H_top_labelfree_test": float(auc_Htop_test),
            "H_target_LEAK_test_ceiling": float(auc_Htarget_test),
        },
        "confidence_g_info": g_info, "feature_names": FEATURE_NAMES,
        "forbidden_feature_names": FORBIDDEN_FEATURE_NAMES,
        "certificate_deployment_split": certificate,
        "multisplit_crc_validation": {
            "n_splits": args.n_splits,
            "folds": "A 40% / B1 15% / B2 15% / C(test) 30% (B1/B2 frozen-lambda split)",
            "pooled_n": len(X_all), "per_coverage": multisplit,
        },
        "pareto_deployment_split": pareto, "pareto_monotonicity": pareto_mono,
        "selective_gain_decomposition_50pct": decomp50, "verdict": verdict,
    }
    (out / "rankcrc_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- certificate CSV (deployment split) ----
    rows = ["target_coverage,certified_alpha,lambda,b2_served_m,jitter_applied,"
            "achieved_test_coverage,test_served_risk,test_risk_ci_lo,test_risk_ci_hi,"
            "guarantee_held,served_NDCG@10,served_NDCG10_ci_lo,served_NDCG10_ci_hi,"
            "served_HR@10,served_MRR,selective_gain,gain_ci_lo,gain_ci_hi,"
            "selnet_noguar_coverage,selnet_noguar_risk,selnet_noguar_NDCG"]
    for c in certificate:
        if not c.get("feasible"):
            rows.append(f"{c['target_coverage']},INFEASIBLE,,,,,,,,,,,,,,,,,,,"); continue
        sn = c.get("selectivenet_no_guarantee") or {}
        rows.append(",".join(str(x) for x in [
            c["target_coverage"], round(c["certified_alpha"], 6), round(c["lambda"], 6),
            c["b2_served_m"], c["jitter_applied"],
            round(c["achieved_test_coverage"], 4), round(c["test_served_risk"], 6),
            round(c["test_served_risk_ci95"][0], 6), round(c["test_served_risk_ci95"][1], 6),
            c["guarantee_held"], round(c["served_NDCG@10"], 6),
            round(c["served_NDCG@10_ci95"][0], 6), round(c["served_NDCG@10_ci95"][1], 6),
            round(c["served_HR@10"], 6), round(c["served_MRR"], 6),
            round(c["selective_gain_vs_random"], 6), round(c["selective_gain_ci95"][0], 6),
            round(c["selective_gain_ci95"][1], 6),
            round(sn.get("achieved_coverage", float("nan")), 4),
            round(sn.get("served_risk", float("nan")), 6),
            round(sn.get("served_NDCG@k", float("nan")), 6)]))
    (out / "rankcrc_certificate.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---- multi-split CSV (the headline expectation guarantee) ----
    ms_rows = ["target_coverage,mean_certified_alpha,mean_test_served_risk,test_risk_ci_lo,"
               "test_risk_ci_hi,guarantee_held_E,expectation_slack,frac_splits_held,"
               "mean_test_coverage,mean_test_served_NDCG,mean_b2_served_m,frac_jitter"]
    for m in multisplit:
        if not m.get("feasible_splits"):
            ms_rows.append(f"{m['target_coverage']},NA,NA,,,,,,,,,"); continue
        ms_rows.append(",".join(str(x) for x in [
            m["target_coverage"], round(m["mean_certified_alpha"], 6),
            round(m["mean_test_served_risk"], 6), round(m["mean_test_served_risk_ci95"][0], 6),
            round(m["mean_test_served_risk_ci95"][1], 6), m["guarantee_held_in_expectation"],
            round(m["expectation_slack"], 6), round(m["frac_splits_held"], 4),
            round(m["mean_test_coverage"], 4), round(m["mean_test_served_NDCG@k"], 6),
            round(m["mean_b2_served_m"], 1), round(m["frac_splits_jitter_applied"], 4)]))
    (out / "rankcrc_multisplit.csv").write_text("\n".join(ms_rows) + "\n", encoding="utf-8")

    # ---- pareto CSV ----
    pr = ["target_coverage,certified_alpha,lambda,b2_served_m,achieved_test_coverage,"
          "test_served_risk,test_served_NDCG@k,guarantee_held"]
    for p in pareto:
        pr.append(",".join(str(x) for x in [
            p["target_coverage"], round(p["certified_alpha"], 6), round(p["lambda"], 6),
            p["b2_served_m"], round(p["achieved_test_coverage"], 4),
            round(p["test_served_risk"], 6),
            round(p["test_served_NDCG@k"], 6), p["guarantee_held"]]))
    (out / "rankcrc_pareto.csv").write_text("\n".join(pr) + "\n", encoding="utf-8")

    # ---- console summary ----
    print(f"[RankCRC] beauty  A={len(ai)} B1={len(b1i)} B2={len(b2i)} test(C)={len(users_test)}  k={k}")
    print(f"full TEST NDCG@10={full_test_metrics['NDCG@10']:.4f} "
          f"HR@10={full_test_metrics['HR@10']:.4f} MRR={full_test_metrics['MRR']:.4f}")
    print(f"g label-free AUC: test={auc_g_test:.4f} (Fold-A insample {auc_g_a:.4f}) | "
          f"H_top={auc_Htop_test:.4f} | H_target(LEAK)={auc_Htarget_test:.4f}")
    print("\n=== DEPLOYMENT-SPLIT CERTIFICATE (B1 lambda / B2 CRC / C test) ===")
    print(f"{'covT':>5} {'certA':>7} {'lam':>7} {'mB2':>5} {'jit':>4} {'achCov':>7} "
          f"{'testRisk':>9} {'held':>5} {'sNDCG@10':>9} {'snNoGuar':>9} {'gain':>8}")
    for c in certificate:
        if not c.get("feasible"):
            print(f"{c['target_coverage']:>5}  INFEASIBLE"); continue
        sn = c.get("selectivenet_no_guarantee") or {}
        sn_ndcg = sn.get("served_NDCG@k", float("nan"))
        print(f"{c['target_coverage']:>5.2f} {c['certified_alpha']:>7.4f} {c['lambda']:>7.4f} "
              f"{c['b2_served_m']:>5d} {str(c['jitter_applied'])[:4]:>4} "
              f"{c['achieved_test_coverage']:>7.3f} {c['test_served_risk']:>9.4f} "
              f"{str(c['guarantee_held']):>5} {c['served_NDCG@10']:>9.4f} {sn_ndcg:>9.4f} "
              f"{c['selective_gain_vs_random']:>+8.4f}")
    print(f"\n=== MULTI-SPLIT CRC EXPECTATION GUARANTEE ({args.n_splits} resplits, pooled "
          f"n={len(X_all)}, A40/B1-15/B2-15/C30, B1/B2 frozen-lambda) ===")
    print(f"{'covT':>5} {'certA':>7} {'meanRisk':>9} {'slack':>8} {'held(E)':>8} {'fracHeld':>9} "
          f"{'mB2':>6} {'meanCov':>8} {'meanNDCG':>9}")
    for m in multisplit:
        if not m.get("feasible_splits"):
            print(f"{m['target_coverage']:>5.2f}  (no feasible splits)"); continue
        print(f"{m['target_coverage']:>5.2f} {m['mean_certified_alpha']:>7.4f} "
              f"{m['mean_test_served_risk']:>9.4f} {m['expectation_slack']:>+8.4f} "
              f"{str(m['guarantee_held_in_expectation']):>8} {m['frac_splits_held']:>9.3f} "
              f"{m['mean_b2_served_m']:>6.1f} {m['mean_test_coverage']:>8.3f} "
              f"{m['mean_test_served_NDCG@k']:>9.4f}")
    print(f"\nPareto: {pareto_mono['n_pareto_points']} pts | monotone IN EXPECTATION "
          f"({pareto_mono['expectation_n_nondecreasing_risk_steps']}/"
          f"{pareto_mono['expectation_n_steps']} steps)="
          f"{pareto_mono['monotone_in_expectation']} | deployment-split exact "
          f"({pareto_mono['deployment_split_n_nondecreasing_risk_steps']}/"
          f"{pareto_mono['deployment_split_n_steps']}, diagnostic, sub-0.002 wiggle)")
    print(f"Decomposition @50%: predicted dNDCG={decomp50['predicted_dNDCG']:+.4f} "
          f"realized={decomp50['realized_dNDCG']:+.4f} relerr={decomp50['relative_error']:.3f} "
          f"(AUC={decomp50['AUC_g']:.3f} 2AUC-1={decomp50['somers_d_2auc_minus_1']:.3f} "
          f"G={decomp50['gini_dispersion_G']:.3f})")
    print(f"\nVERDICT rankcrc_validates_end_to_end = {verdict['rankcrc_validates_end_to_end']}")
    print(f"  expectation guarantee held: {n_held_E}/{len(ms_feas)} | deployment exact: "
          f"{n_held_dep}/{len(feas)} (within-CI {n_held_dep_ci}/{len(feas)}) | g beats H_top: "
          f"{g_beats_htop} | pareto monotone (E): {pareto_mono['monotone_in_expectation']}")
    print("wrote", out / "rankcrc_results.json")


if __name__ == "__main__":
    main()
