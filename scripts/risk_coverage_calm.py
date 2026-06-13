#!/usr/bin/env python3
"""Risk-coverage / selective-prediction analysis for CALM-Rec (TRUCE-Rec pivot test).

Tests the PIVOT HYPOTHESIS: the validated calibrated uncertainty (responsibility
entropy H, MC-dropout variance Var_m) enables UNCERTAINTY-AWARE SELECTIVE
RECOMMENDATION -- abstaining/deferring on the most-uncertain TEST users raises
ranking quality (NDCG@10/HR@10/MRR) among the RETAINED (covered) users, beating
random abstention. If true, TRUCE has a distinct, working contribution (selective
recommendation), even though the rho trust-GATE is inert for same-candidate ranking
(full+gate 0.1004 < raw 0.1143) and full does not clear the 0.1506 SOTA bar.

Inputs (CPU only, NO GPU/Qwen): the per-user signals regenerated from the VALIDATED
Stage-2 SASRec scorer (raw NDCG@10=0.1143, auc_entropy=0.785), saved as
signals_test_sasrec.npz / signals_val_sasrec.npz. Each user i has 101-candidate
arrays s_pers, H, var_m, n_i, prior + scalar target_idx.

Uncertainty scores (LOWER = more confident):
  - H_top    : responsibility entropy at the model's top-ranked candidate (argmax s_pers).
               LABEL-FREE, deployable. The natural per-user confidence signal.
  - Var_top  : MC-dropout std at the top-ranked candidate. LABEL-FREE, deployable.
  - H_target : entropy at the held-out positive's slot. Peeks at the label -> NOT
               deployable; reported only because it is the exact signal whose AUC=0.785
               was validated. Flagged non_deployable in the JSON.

Baselines per coverage:
  - random  : retain a random c% of users; mean +/- std over N_SEEDS seeds.
  - oracle  : retain the c% with the BEST realized rank-of-positive (upper bound).

Summary statistic: selective gain = mean over operating coverages (<100%) of
(confident_NDCG@10 - random_NDCG@10). Also an AURC-style area (trapezoid over
coverage in [0.5,1.0]) for confident, random, oracle; and a monotonic-rise flag.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Load regenerated per-user signals
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
# Per-user ranking metrics (raw personalized s_pers; same as eval/verdict)
# ---------------------------------------------------------------------------
def per_user_rank(u: dict) -> int:
    """0-based rank of the positive under raw s_pers (stable argsort, matches eval)."""
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
# Per-user uncertainty scores (lower = more confident)
# ---------------------------------------------------------------------------
def uncertainty_scores(users: list[dict]) -> dict[str, np.ndarray]:
    H_top, Var_top, H_target = [], [], []
    for u in users:
        top = int(np.argmax(u["s_pers"]))
        H_top.append(float(u["H"][top]))
        Var_top.append(float(u["var_m"][top]))
        H_target.append(float(u["H"][u["target_idx"]]))
    return {
        "H_top": np.asarray(H_top),
        "Var_top": np.asarray(Var_top),
        "H_target": np.asarray(H_target),
    }


# ---------------------------------------------------------------------------
# Risk-coverage curves
# ---------------------------------------------------------------------------
def coverage_counts(n: int, coverages: list[int]) -> dict[int, int]:
    return {c: max(1, int(round(n * c / 100.0))) for c in coverages}


def confident_curve(ranks: np.ndarray, unc: np.ndarray, coverages: list[int]) -> dict:
    """Retain the k most-confident (lowest-uncertainty) users at each coverage.

    Ties broken deterministically by uncertainty then user index (stable sort).
    """
    n = len(ranks)
    order = np.argsort(unc, kind="stable")  # ascending uncertainty
    counts = coverage_counts(n, coverages)
    res = {}
    for c in coverages:
        k = counts[c]
        keep = order[:k]
        res[c] = metrics_from_ranks(ranks[keep])
    return res


def random_curve(ranks: np.ndarray, coverages: list[int], n_seeds: int, base_seed: int) -> dict:
    n = len(ranks)
    counts = coverage_counts(n, coverages)
    acc = {c: {m: [] for m in ("NDCG@10", "HR@10", "MRR", "NDCG@5", "HR@5", "NDCG@20", "HR@20")}
           for c in coverages}
    rng = np.random.default_rng(base_seed)
    for _ in range(n_seeds):
        perm = rng.permutation(n)
        for c in coverages:
            keep = perm[: counts[c]]
            m = metrics_from_ranks(ranks[keep])
            for key in acc[c]:
                acc[c][key].append(m[key])
    res = {}
    for c in coverages:
        res[c] = {"n": counts[c]}
        for key in acc[c]:
            arr = np.asarray(acc[c][key])
            res[c][key] = float(arr.mean())
            res[c][f"{key}_std"] = float(arr.std())
    return res


def oracle_curve(ranks: np.ndarray, coverages: list[int]) -> dict:
    """Upper bound: retain the users with the BEST realized rank-of-positive."""
    n = len(ranks)
    order = np.argsort(ranks, kind="stable")  # smallest rank (best) first
    counts = coverage_counts(n, coverages)
    res = {}
    for c in coverages:
        keep = order[: counts[c]]
        res[c] = metrics_from_ranks(ranks[keep])
    return res


def aurc_area(curve: dict, coverages: list[int], metric: str, use_mean_key=False) -> float:
    """Trapezoidal area of metric vs coverage-fraction over the operating range.

    Higher = better selective behaviour. Coverages sorted ascending in fraction.
    """
    cs = sorted(coverages)
    xs = [c / 100.0 for c in cs]
    ys = [curve[c][metric] for c in cs]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz
    return float(trapezoid(ys, xs))


def selective_gain(conf: dict, rand: dict, coverages: list[int], metric="NDCG@10") -> dict:
    """Per-coverage and aggregate confident-minus-random advantage (operating coverages<100)."""
    per = {}
    op = [c for c in coverages if c < 100]
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
        "operating_coverages": op,
    }


def monotone_rise(conf: dict, coverages: list[int], metric="NDCG@10") -> dict:
    """Does the metric rise as coverage drops (confidence increases)?"""
    cs = sorted(coverages, reverse=True)  # 100 -> 50
    vals = [conf[c][metric] for c in cs]
    rises = sum(1 for a, b in zip(vals, vals[1:]) if b >= a - 1e-9)
    return {
        "values_high_to_low_coverage": {c: conf[c][metric] for c in cs},
        "n_nondecreasing_steps": int(rises),
        "n_steps": len(vals) - 1,
        "endpoint_gain_full_to_min_coverage": float(vals[-1] - vals[0]),
        "strictly_rising": all(b > a - 1e-9 for a, b in zip(vals, vals[1:])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sasrec-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coverages", default="100,90,80,70,60,50")
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    sasrec = Path(args.sasrec_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    coverages = [int(x) for x in args.coverages.split(",")]

    users_test = load_signals(sasrec / "signals_test_sasrec.npz")
    users_val = load_signals(sasrec / "signals_val_sasrec.npz")
    verdict = json.loads((sasrec / "calm_rec_verdict.json").read_text(encoding="utf-8"))

    ranks = np.asarray([per_user_rank(u) for u in users_test])
    full_metrics = metrics_from_ranks(ranks)  # coverage=100, all users

    # AUC sanity (label-aware H_target, exactly the validated signal definition)
    unc = uncertainty_scores(users_test)
    correct_top1 = np.asarray([1.0 if int(np.argmax(u["s_pers"])) == u["target_idx"] else 0.0
                               for u in users_test])

    def auc(scores, labels):
        # AUC of P(correct) vs LOW uncertainty: use -score so higher=more confident.
        from itertools import product as _p  # noqa
        pos = (-scores)[labels == 1.0]
        neg = (-scores)[labels == 0.0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        # Mann-Whitney U / (n_pos*n_neg)
        order = np.argsort(np.concatenate([pos, neg]), kind="stable")
        ranks_all = np.empty(len(order)); ranks_all[order] = np.arange(1, len(order) + 1)
        r_pos = ranks_all[: len(pos)].sum()
        u_stat = r_pos - len(pos) * (len(pos) + 1) / 2.0
        return float(u_stat / (len(pos) * len(neg)))

    auc_check = {
        "auc_H_target_predicts_top1": auc(unc["H_target"], correct_top1),
        "auc_H_top_predicts_top1": auc(unc["H_top"], correct_top1),
        "auc_Var_top_predicts_top1": auc(unc["Var_top"], correct_top1),
        "note": "H_target is the validated label-aware signal (expect ~0.785); "
                "H_top/Var_top are the deployable label-free signals.",
    }

    signals_spec = {
        "H_top": {"deployable": True,
                  "desc": "responsibility entropy at the model's top-ranked candidate (label-free)"},
        "Var_top": {"deployable": True,
                    "desc": "MC-dropout std at the top-ranked candidate (label-free)"},
        "H_target": {"deployable": False,
                     "desc": "entropy at the held-out positive (peeks at label; matches the "
                             "validated AUC=0.785 definition; diagnostic only)"},
    }

    rand = random_curve(ranks, coverages, args.n_seeds, args.seed)
    oracle = oracle_curve(ranks, coverages)

    results = {}
    for sig_name, scores in unc.items():
        conf = confident_curve(ranks, scores, coverages)
        gain = selective_gain(conf, rand, coverages, "NDCG@10")
        gain_hr = selective_gain(conf, rand, coverages, "HR@10")
        gain_mrr = selective_gain(conf, rand, coverages, "MRR")
        results[sig_name] = {
            "deployable": signals_spec[sig_name]["deployable"],
            "desc": signals_spec[sig_name]["desc"],
            "confident": conf,
            "selective_gain_NDCG@10": gain,
            "selective_gain_HR@10": gain_hr,
            "selective_gain_MRR": gain_mrr,
            "monotone_NDCG@10": monotone_rise(conf, coverages, "NDCG@10"),
            "aurc_NDCG@10_confident": aurc_area(conf, coverages, "NDCG@10"),
        }

    payload = {
        "experiment": "risk_coverage_selective_recommendation",
        "is_paper_evidence": True,
        "scorer": "sasrec_head_over_cached_llm_embeddings (validated repaired scorer)",
        "scorer_raw_NDCG@10": full_metrics["NDCG@10"],
        "sota_ndcg10_bar": verdict.get("sota_ndcg10_bar", 0.1506),
        "validated_gate_auc_entropy_from_verdict": verdict["stage_2p5_reliability_gate"]["auc_entropy"],
        "n_test_users": len(users_test),
        "coverages_percent": coverages,
        "n_random_seeds": args.n_seeds,
        "full_coverage_metrics": full_metrics,
        "auc_sanity_check": auc_check,
        "signals": signals_spec,
        "random_abstention": rand,
        "oracle_abstention": oracle,
        "aurc_NDCG@10": {
            "random": aurc_area(rand, coverages, "NDCG@10"),
            "oracle": aurc_area(oracle, coverages, "NDCG@10"),
            **{f"confident_{k}": results[k]["aurc_NDCG@10_confident"] for k in results},
        },
        "results_by_uncertainty_signal": results,
    }

    (out / "risk_coverage_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    # ---- console summary ----
    print(f"n_test_users={len(users_test)}  full NDCG@10={full_metrics['NDCG@10']:.4f} "
          f"HR@10={full_metrics['HR@10']:.4f} MRR={full_metrics['MRR']:.4f}")
    print(f"AUC(H_target->top1)={auc_check['auc_H_target_predicts_top1']:.4f} (validate ~0.785) | "
          f"AUC(H_top)={auc_check['auc_H_top_predicts_top1']:.4f} | "
          f"AUC(Var_top)={auc_check['auc_Var_top_predicts_top1']:.4f}")
    print("\nNDCG@10 by coverage (confident-retention vs random vs oracle):")
    hdr = "  cov  " + "  ".join(f"{c:>6}%" for c in coverages)
    print(hdr)
    for name in ("H_top", "Var_top", "H_target"):
        row = "  ".join(f"{results[name]['confident'][c]['NDCG@10']:.4f}" for c in coverages)
        tag = "" if signals_spec[name]["deployable"] else " (label-aware/diag)"
        print(f"  conf[{name:>8}] {row}{tag}")
    rrow = "  ".join(f"{rand[c]['NDCG@10']:.4f}" for c in coverages)
    orow = "  ".join(f"{oracle[c]['NDCG@10']:.4f}" for c in coverages)
    print(f"  {'random':>13}  {rrow}")
    print(f"  {'oracle':>13}  {orow}")
    print("\nselective gain (confident - random) NDCG@10, mean over operating coverages<100:")
    for name in ("H_top", "Var_top", "H_target"):
        g = results[name]["selective_gain_NDCG@10"]
        print(f"  {name:>8}: mean_delta={g['mean_operating_delta']:+.4f} "
              f"min={g['min_operating_delta']:+.4f} max={g['max_operating_delta']:+.4f}")
    print("\nwrote", out / "risk_coverage_results.json")


if __name__ == "__main__":
    main()
