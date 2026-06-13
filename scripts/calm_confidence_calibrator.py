#!/usr/bin/env python3
"""Label-free confidence calibrator for CALM-Rec selective recommendation (CPU only).

GOAL (TRUCE-Rec pivot, Stage-B): recover the selective-recommendation headroom that
the risk-coverage analysis proved exists. The deployable label-free signal H_top
(responsibility entropy at the model's TOP-ranked candidate) has test AUC only ~0.519
and a small selective gain; but the ORACLE (retain by realized correctness) reaches
NDCG@10 0.2288 at 50% coverage and the label-LEAKING H_target (AUC 0.785) reaches
0.2012 -- so reliability IS predictable; the gap is a deployable ESTIMATOR.

This script trains a LABEL-FREE confidence model that predicts, per user, whether the
model's TOP-ranked candidate is actually the held-out positive (binary target
"is top-1 correct", available on the VALIDATION set), using ONLY label-free test-time
features of the scoring (top-1 score, top1-top2 margin, entropy stats, MC-dropout
variance, responsibility-distribution stats, log-popularity of the top-1 item, ...).

Train the calibrator (logistic regression + gradient-boosted tree, compared) on the
973 VALIDATION users, APPLY to the 973 TEST users, then redo the risk-coverage curve.

Leakage discipline:
  * The binary target uses the label, but ONLY on VAL (training). It is NEVER used to
    build a feature, and is NEVER computed/used on TEST during calibration.
  * On TEST we only apply the frozen calibrator to label-free features and rank users
    by predicted confidence -- exactly what a deployed system can do.
  * H_target (entropy at the positive's slot) is NOT a feature -- it peeks at the label.

Inputs (CPU only, NO GPU/Qwen): signals_{val,test}_sasrec.npz from the VALIDATED
Stage-2 SASRec scorer (raw NDCG@10=0.1143, auc_entropy=0.785). Each user has
101-candidate arrays s_pers, H, var_m, n_i, prior + scalar target_idx.

Outputs: confidence_calibration_results.json with (1) learned-confidence TEST AUC
(label-free) vs H_top 0.519, (2) full risk-coverage table for learned vs H_top vs
random(N seeds) vs oracle (NDCG@10/HR@10/MRR), (3) selective gain per coverage + AURC,
(4) feature importances + val->test generalization, (5) a machine VERDICT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load regenerated per-user signals (same layout as risk_coverage_calm.py)
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
# Ranking helpers (identical convention to eval/verdict and risk_coverage_calm.py)
# ---------------------------------------------------------------------------
def per_user_rank(u: dict) -> int:
    order = np.argsort(-u["s_pers"], kind="stable")
    return int(np.where(order == u["target_idx"])[0][0])


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
# LABEL-FREE feature extraction (per user, from the 101-candidate scoring)
# ---------------------------------------------------------------------------
# Convention reminder (from train_calm_sasrec.compute_signals):
#   s_pers : personalized mixture score per candidate (HIGHER = better; top-1 = argmax)
#   H      : responsibility entropy of intent-assignment per candidate (HIGHER = ambiguous)
#   var_m  : MC-dropout std of s_pers per candidate                  (HIGHER = uncertain)
#   n_i    : train support count (popularity) per candidate
#   prior  : log-popularity prior per candidate
# NONE of these condition on the held-out positive -> all label-free / deployable.
FEATURE_NAMES = [
    "s_top1",            # top-1 personalized score
    "s_top2",            # runner-up score
    "margin_12",         # s_top1 - s_top2  (decisiveness of the winner)
    "margin_13",         # s_top1 - s_top3
    "s_mean", "s_std", "s_max_minus_min",
    "sm_p1",             # softmax(s_pers) prob mass on the winner (peakiness)
    "sm_gap_12",         # softmax prob gap winner vs runner-up
    "sm_entropy",        # entropy of softmax(s_pers) over the 101 candidates (nats)
    "H_top",             # responsibility entropy at the winner  (== the old H_top signal)
    "H_min", "H_max", "H_mean", "H_std",
    "H_rank_of_top",     # popularity-independent: H_top's rank among all H (0..100), low=conf
    "Var_top",           # MC-dropout std at the winner          (== the old Var_top signal)
    "Var_mean", "Var_max",
    "logpop_top1",       # log-popularity prior of the winner
    "n_i_top1",          # raw support count of the winner
    "logpop_mean",       # mean prior over candidates (panel popularity baseline)
    "top1_is_most_popular",  # is the winner also the most popular candidate? (1/0)
    "pop_rank_of_top1",  # popularity rank of the winner among candidates (0=most popular)
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
    H_rank_of_top = float(np.sum(H < H[top]))  # how many candidates are MORE confident (lower H)
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


def build_xy(users: list[dict]):
    X = np.stack([extract_features(u) for u in users])
    # binary correctness target: is the model's TOP-ranked candidate the positive?
    y = np.asarray([1.0 if int(np.argmax(u["s_pers"])) == u["target_idx"] else 0.0
                    for u in users])
    return X, y


# ---------------------------------------------------------------------------
# AUC (Mann-Whitney U); higher score == higher predicted P(correct)
# ---------------------------------------------------------------------------
def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1.0]
    neg = scores[labels == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    r = np.empty(len(order)); r[order] = np.arange(1, len(order) + 1)
    # average ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg_rank_by_group = (csum - (counts - 1) / 2.0)
    r = avg_rank_by_group[inv]
    r_pos = r[: len(pos)].sum()
    u_stat = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u_stat / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# Risk-coverage curves (same definitions as risk_coverage_calm.py)
# ---------------------------------------------------------------------------
def coverage_counts(n: int, coverages: list[int]) -> dict[int, int]:
    return {c: max(1, int(round(n * c / 100.0))) for c in coverages}


def confident_curve(ranks: np.ndarray, conf: np.ndarray, coverages: list[int]) -> dict:
    """Retain the k MOST-confident users (HIGHEST conf) at each coverage."""
    n = len(ranks)
    order = np.argsort(-conf, kind="stable")  # descending confidence
    counts = coverage_counts(n, coverages)
    return {c: metrics_from_ranks(ranks[order[: counts[c]]]) for c in coverages}


def uncertainty_curve(ranks: np.ndarray, unc: np.ndarray, coverages: list[int]) -> dict:
    """Retain the k LOWEST-uncertainty users (for H_top: lower H = more confident)."""
    n = len(ranks)
    order = np.argsort(unc, kind="stable")  # ascending uncertainty
    counts = coverage_counts(n, coverages)
    return {c: metrics_from_ranks(ranks[order[: counts[c]]]) for c in coverages}


def random_curve(ranks: np.ndarray, coverages: list[int], n_seeds: int, base_seed: int) -> dict:
    n = len(ranks)
    counts = coverage_counts(n, coverages)
    keys = ("NDCG@10", "HR@10", "MRR", "NDCG@5", "HR@5", "NDCG@20", "HR@20")
    acc = {c: {m: [] for m in keys} for c in coverages}
    rng = np.random.default_rng(base_seed)
    for _ in range(n_seeds):
        perm = rng.permutation(n)
        for c in coverages:
            m = metrics_from_ranks(ranks[perm[: counts[c]]])
            for k in keys:
                acc[c][k].append(m[k])
    res = {}
    for c in coverages:
        res[c] = {"n": counts[c]}
        for k in keys:
            arr = np.asarray(acc[c][k])
            res[c][k] = float(arr.mean())
            res[c][f"{k}_std"] = float(arr.std())
    return res


def oracle_curve(ranks: np.ndarray, coverages: list[int]) -> dict:
    n = len(ranks)
    order = np.argsort(ranks, kind="stable")  # best (smallest) rank first
    counts = coverage_counts(n, coverages)
    return {c: metrics_from_ranks(ranks[order[: counts[c]]]) for c in coverages}


def aurc_area(curve: dict, coverages: list[int], metric: str) -> float:
    cs = sorted(coverages)
    xs = [c / 100.0 for c in cs]
    ys = [curve[c][metric] for c in cs]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(ys, xs))


def selective_gain(conf: dict, rand: dict, coverages: list[int], metric="NDCG@10") -> dict:
    per = {}
    diffs = []
    for c in coverages:
        d = conf[c][metric] - rand[c][metric]
        per[c] = float(d)
        if c < 100:
            diffs.append(d)
    return {
        "per_coverage_delta": per,
        "mean_operating_delta": float(np.mean(diffs)) if diffs else 0.0,
        "min_operating_delta": float(np.min(diffs)) if diffs else 0.0,
        "max_operating_delta": float(np.max(diffs)) if diffs else 0.0,
    }


def monotone_rise(conf: dict, coverages: list[int], metric="NDCG@10") -> dict:
    cs = sorted(coverages, reverse=True)
    vals = [conf[c][metric] for c in cs]
    rises = sum(1 for a, b in zip(vals, vals[1:]) if b >= a - 1e-9)
    return {
        "values_high_to_low_coverage": {c: conf[c][metric] for c in cs},
        "n_nondecreasing_steps": int(rises),
        "n_steps": len(vals) - 1,
        "endpoint_gain_full_to_min_coverage": float(vals[-1] - vals[0]),
        "strictly_rising": all(b > a - 1e-9 for a, b in zip(vals, vals[1:])),
    }


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------
def fit_logreg(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced",
                             solver="lbfgs")
    clf.fit(Xtr_s, ytr)
    p_tr = clf.predict_proba(Xtr_s)[:, 1]
    p_te = clf.predict_proba(Xte_s)[:, 1]
    coefs = {FEATURE_NAMES[i]: float(clf.coef_[0][i]) for i in range(len(FEATURE_NAMES))}
    return p_tr, p_te, {"type": "logistic_regression",
                        "C": 1.0, "class_weight": "balanced",
                        "standardized_coefficients": coefs}


def fit_gbt(Xtr, ytr, Xte, seed=13):
    import xgboost as xgb
    pos = float(ytr.sum()); neg = float(len(ytr) - ytr.sum())
    spw = (neg / max(pos, 1.0))
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5.0,
        reg_lambda=1.0, scale_pos_weight=spw, random_state=seed,
        eval_metric="logloss", n_jobs=4, tree_method="hist",
    )
    clf.fit(Xtr, ytr)
    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]
    imp = clf.feature_importances_
    importances = {FEATURE_NAMES[i]: float(imp[i]) for i in range(len(FEATURE_NAMES))}
    return p_tr, p_te, {"type": "xgboost_gbt",
                        "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
                        "scale_pos_weight": spw,
                        "gain_feature_importances": importances}


def kfold_oof_auc(X, y, kind, k=5, seed=13):
    """Honest in-sample estimate: K-fold out-of-fold AUC on VAL only.

    Guards against the calibrator overfitting the small VAL set: if OOF AUC on VAL
    is close to the val->test AUC, the model generalizes; a big drop = overfit.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        if kind == "logreg":
            _, p_va, _ = fit_logreg(X[tr_idx], y[tr_idx], X[va_idx])
        else:
            _, p_va, _ = fit_gbt(X[tr_idx], y[tr_idx], X[va_idx], seed=seed)
        oof[va_idx] = p_va
    return auc_score(oof, y), oof


# ---------------------------------------------------------------------------
def coverage_table(curve, coverages, keyfn=lambda c: c):
    return {str(c): {m: curve[c][m] for m in ("NDCG@10", "HR@10", "MRR", "NDCG@5", "n")}
            for c in coverages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sasrec-dir", default="outputs/calm/beauty_frozen_v2/sasrec")
    ap.add_argument("--out", default="outputs/calm/beauty_frozen_v2/confidence_calibrator")
    ap.add_argument("--coverages", default="100,90,80,70,60,50")
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sasrec = (root / args.sasrec_dir) if not Path(args.sasrec_dir).is_absolute() else Path(args.sasrec_dir)
    out = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    coverages = [int(x) for x in args.coverages.split(",")]

    users_val = load_signals(sasrec / "signals_val_sasrec.npz")
    users_test = load_signals(sasrec / "signals_test_sasrec.npz")

    Xtr, ytr = build_xy(users_val)    # VAL: features + correctness target (training)
    Xte, yte = build_xy(users_test)   # TEST: features (+ correctness only for AUC reporting)

    ranks_test = np.asarray([per_user_rank(u) for u in users_test])
    full_metrics = metrics_from_ranks(ranks_test)

    # ---- label-free baseline signals on TEST (for the head-to-head AUC + curve) ----
    H_top_test = np.asarray([float(u["H"][int(np.argmax(u["s_pers"]))]) for u in users_test])
    Var_top_test = np.asarray([float(u["var_m"][int(np.argmax(u["s_pers"]))]) for u in users_test])
    H_target_test = np.asarray([float(u["H"][u["target_idx"]]) for u in users_test])  # LEAK, diag only

    auc_Htop = auc_score(-H_top_test, yte)        # low H = high conf -> negate
    auc_Vtop = auc_score(-Var_top_test, yte)
    auc_Htarget = auc_score(-H_target_test, yte)  # label-leaking ceiling reference

    # ---- train calibrators on VAL, apply to TEST ----
    p_tr_lr, p_te_lr, info_lr = fit_logreg(Xtr, ytr, Xte)
    p_tr_gb, p_te_gb, info_gb = fit_gbt(Xtr, ytr, Xte, seed=args.seed)

    auc_lr_train = auc_score(p_tr_lr, ytr)   # in-sample (optimistic)
    auc_gb_train = auc_score(p_tr_gb, ytr)
    auc_lr_test = auc_score(p_te_lr, yte)    # val->test generalization (the headline number)
    auc_gb_test = auc_score(p_te_gb, yte)
    oof_lr_auc, _ = kfold_oof_auc(Xtr, ytr, "logreg", k=5, seed=args.seed)
    oof_gb_auc, _ = kfold_oof_auc(Xtr, ytr, "gbt", k=5, seed=args.seed)

    # ---- risk-coverage curves on TEST ----
    rand = random_curve(ranks_test, coverages, args.n_seeds, args.seed)
    oracle = oracle_curve(ranks_test, coverages)
    curve_Htop = uncertainty_curve(ranks_test, H_top_test, coverages)   # lower H = retain
    curve_lr = confident_curve(ranks_test, p_te_lr, coverages)          # higher conf = retain
    curve_gb = confident_curve(ranks_test, p_te_gb, coverages)

    def pack_signal(curve):
        return {
            "confident": coverage_table(curve, coverages),
            "selective_gain_NDCG@10": selective_gain(curve, rand, coverages, "NDCG@10"),
            "selective_gain_HR@10": selective_gain(curve, rand, coverages, "HR@10"),
            "selective_gain_MRR": selective_gain(curve, rand, coverages, "MRR"),
            "monotone_NDCG@10": monotone_rise(curve, coverages, "NDCG@10"),
            "aurc_NDCG@10": aurc_area(curve, coverages, "NDCG@10"),
        }

    aurc = {
        "random": aurc_area(rand, coverages, "NDCG@10"),
        "oracle": aurc_area(oracle, coverages, "NDCG@10"),
        "H_top": aurc_area(curve_Htop, coverages, "NDCG@10"),
        "learned_logreg": aurc_area(curve_lr, coverages, "NDCG@10"),
        "learned_gbt": aurc_area(curve_gb, coverages, "NDCG@10"),
    }

    # how much of the H_top->oracle AURC gap does the learned model close?
    gap = aurc["oracle"] - aurc["H_top"]
    def frac_closed(name):
        if gap <= 1e-12:
            return float("nan")
        return float((aurc[name] - aurc["H_top"]) / gap)

    # selective-gain fraction-of-ceiling at 50% coverage (NDCG@10)
    base50 = rand[50]["NDCG@10"]
    ceil50 = oracle[50]["NDCG@10"] - base50
    def frac_ceiling_50(curve):
        if ceil50 <= 1e-12:
            return float("nan")
        return float((curve[50]["NDCG@10"] - base50) / ceil50)

    best_auc_test = max(auc_lr_test, auc_gb_test)
    best_name = "logreg" if auc_lr_test >= auc_gb_test else "gbt"
    best_curve = curve_lr if best_name == "logreg" else curve_gb
    gain50_learned = best_curve[50]["NDCG@10"] - rand[50]["NDCG@10"]
    gain50_htop = curve_Htop[50]["NDCG@10"] - rand[50]["NDCG@10"]

    # AUC ceiling-gap: how much of the (H_target_leak - H_top) AUC gap the learned
    # label-free model recovers. The leaking H_target AUC is the upper bound a perfect
    # *top-1-correctness* classifier could reach.
    auc_gap = auc_Htarget - auc_Htop
    auc_frac_recovered = float((best_auc_test - auc_Htop) / auc_gap) if auc_gap > 1e-9 else float("nan")

    # Two SEPARATE, non-conflated questions (honest verdict):
    # (Q1) Does the learned LABEL-FREE estimator substantially beat the deployable H_top
    #      baseline? -> AUC lift and selective-gain multiple over H_top.
    beats_H_top = (best_auc_test > auc_Htop + 0.05) and (gain50_learned >= 1.5 * max(gain50_htop, 1e-9))
    # (Q2) Does it CLOSE the gap to the ORACLE (realized-rank upper bound)? -> AURC fraction.
    closes_oracle_gap = frac_closed("learned_" + best_name) >= 0.33

    verdict = {
        "best_calibrator": best_name,
        "best_test_auc_labelfree": float(best_auc_test),
        "H_top_test_auc": float(auc_Htop),
        "H_target_LEAK_auc_ceiling": float(auc_Htarget),
        "auc_lift_over_H_top": float(best_auc_test - auc_Htop),
        "auc_frac_of_H_top_to_leak_ceiling_recovered": auc_frac_recovered,
        "selective_gain50_learned_NDCG@10": float(gain50_learned),
        "selective_gain50_H_top_NDCG@10": float(gain50_htop),
        "selective_gain50_multiple_vs_H_top": float(gain50_learned / max(gain50_htop, 1e-9)),
        "frac_of_AURC_gap_to_oracle_closed": frac_closed("learned_" + best_name),
        "frac_of_50pct_ceiling_reached_learned": frac_ceiling_50(best_curve),
        "frac_of_50pct_ceiling_reached_H_top": frac_ceiling_50(curve_Htop),
        # ---- the two headline booleans ----
        "Q1_beats_deployable_H_top": bool(beats_H_top),
        "Q2_closes_gap_to_oracle": bool(closes_oracle_gap),
        "interpretation": (
            "Q1 (beats_deployable_H_top): test AUC lift > +0.05 AND >= 1.5x H_top's "
            "50%-coverage NDCG@10 selective gain. Q2 (closes_gap_to_oracle): learned "
            "model recovers >= 33% of the (oracle - H_top) AURC gap. The ORACLE sorts by "
            "REALIZED rank-of-positive (not just top-1 correctness), so it is a loose "
            "upper bound that even a near-perfect correctness classifier cannot reach -- "
            "Q1 is the deployable, decision-relevant question; Q2 measures distance to a "
            "rank-aware ceiling."
        ),
    }

    payload = {
        "experiment": "labelfree_confidence_calibrator_for_selective_recommendation",
        "is_paper_evidence": True,
        "cpu_only": True,
        "scorer": "sasrec_head_over_cached_llm_embeddings (validated repaired scorer, "
                  "raw NDCG@10=0.1143, auc_entropy=0.785 leaking)",
        "leakage_discipline": (
            "calibrator TRAINED on VAL only (973 users); binary 'is top-1 correct' target "
            "uses the label only on VAL and is never a feature; on TEST only label-free "
            "features feed the frozen calibrator. H_target excluded from features."
        ),
        "n_val_users": len(users_val),
        "n_test_users": len(users_test),
        "val_base_rate_top1_correct": float(ytr.mean()),
        "test_base_rate_top1_correct": float(yte.mean()),
        "feature_names": FEATURE_NAMES,
        "coverages_percent": coverages,
        "n_random_seeds": args.n_seeds,
        "full_coverage_metrics": full_metrics,
        "auc_summary": {
            "H_top_labelfree": float(auc_Htop),
            "Var_top_labelfree": float(auc_Vtop),
            "H_target_LEAKING_ceiling": float(auc_Htarget),
            "learned_logreg_train_insample": float(auc_lr_train),
            "learned_logreg_val_oof_5fold": float(oof_lr_auc),
            "learned_logreg_test": float(auc_lr_test),
            "learned_gbt_train_insample": float(auc_gb_train),
            "learned_gbt_val_oof_5fold": float(oof_gb_auc),
            "learned_gbt_test": float(auc_gb_test),
            "note": "val_oof_5fold ~ test indicates generalization (no VAL overfit). "
                    "All learned AUCs are LABEL-FREE; H_target is the label-leaking ceiling.",
        },
        "calibrator_logreg": info_lr,
        "calibrator_gbt": info_gb,
        "risk_coverage": {
            "full_coverage_metrics": full_metrics,
            "random_abstention": coverage_table(rand, coverages),
            "oracle_abstention": coverage_table(oracle, coverages),
            "H_top": pack_signal(curve_Htop),
            "learned_logreg": pack_signal(curve_lr),
            "learned_gbt": pack_signal(curve_gb),
        },
        "aurc_NDCG@10": aurc,
        "aurc_gap_closed_vs_oracle": {
            "H_top": 0.0,
            "learned_logreg": frac_closed("learned_logreg"),
            "learned_gbt": frac_closed("learned_gbt"),
            "note": "fraction of the (oracle - H_top) AURC gap recovered by the learned model.",
        },
        "verdict": verdict,
    }

    (out / "confidence_calibration_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    # ---------------- console summary ----------------
    print(f"VAL base-rate(top1 correct)={ytr.mean():.4f}  TEST base-rate={yte.mean():.4f}")
    print(f"full TEST NDCG@10={full_metrics['NDCG@10']:.4f} HR@10={full_metrics['HR@10']:.4f} "
          f"MRR={full_metrics['MRR']:.4f}")
    print("\n=== LABEL-FREE confidence AUC on TEST (predict top-1 correctness) ===")
    print(f"  H_top (baseline)        : {auc_Htop:.4f}")
    print(f"  Var_top (baseline)      : {auc_Vtop:.4f}")
    print(f"  learned LogReg  test    : {auc_lr_test:.4f}  (val 5fold-OOF {oof_lr_auc:.4f}, "
          f"train {auc_lr_train:.4f})")
    print(f"  learned GBT     test    : {auc_gb_test:.4f}  (val 5fold-OOF {oof_gb_auc:.4f}, "
          f"train {auc_gb_train:.4f})")
    print(f"  [ceiling] H_target LEAK : {auc_Htarget:.4f}  (label-leaking, not deployable)")
    print("\n=== Risk-coverage NDCG@10 (retain most-confident) ===")
    hdr = "  cov  " + "  ".join(f"{c:>6}%" for c in coverages)
    print(hdr)
    for nm, cv in (("H_top", curve_Htop), ("learn_LR", curve_lr), ("learn_GBT", curve_gb)):
        row = "  ".join(f"{cv[c]['NDCG@10']:.4f}" for c in coverages)
        print(f"  {nm:>9} {row}")
    print(f"  {'random':>9} " + "  ".join(f"{rand[c]['NDCG@10']:.4f}" for c in coverages))
    print(f"  {'oracle':>9} " + "  ".join(f"{oracle[c]['NDCG@10']:.4f}" for c in coverages))
    print("\n=== AURC NDCG@10 (coverage 0.5..1.0) ===")
    for k in ("random", "H_top", "learned_logreg", "learned_gbt", "oracle"):
        print(f"  {k:>16}: {aurc[k]:.5f}")
    print(f"  gap closed vs oracle: LR={frac_closed('learned_logreg'):.2%}  "
          f"GBT={frac_closed('learned_gbt'):.2%}")
    print(f"\nVERDICT  Q1 beats deployable H_top = {verdict['Q1_beats_deployable_H_top']}  "
          f"(best={best_name} AUC {best_auc_test:.4f} vs H_top {auc_Htop:.4f}; "
          f"50%-cov gain {gain50_learned:+.4f} = {verdict['selective_gain50_multiple_vs_H_top']:.1f}x H_top)")
    print(f"VERDICT  Q2 closes gap to ORACLE = {verdict['Q2_closes_gap_to_oracle']}  "
          f"(AURC gap closed {frac_closed('learned_' + best_name):.1%}; "
          f"AUC recovered {auc_frac_recovered:.1%} of H_top->leak ceiling)")
    print("wrote", out / "confidence_calibration_results.json")


if __name__ == "__main__":
    main()
