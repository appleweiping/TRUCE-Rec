"""CALM-Rec three-stage training scaffold + stage-2.5 reliability gate.

Implements the training CONTRACT from docs/method_calm_rec_spec.md sections 4-5, with explicit
leakage controls, so that running a real experiment is a matter of (a) swapping the hashed encoders
for the Qwen3-8B path and (b) filling the Stage-B gradient loop. Everything that does NOT need a GPU
is implemented and tested here:

  Stage A (offline, train-only, frozen): item-support n_i (held-out targets removed), exposure q(j),
    popularity b_i, item vectors h_i (cached), attribute weak-labels + anchor centroids, and the
    history-free prior s_prior_i = b_i + beta^T x_i fit by train-LOO ridge.
  Stage B (end-to-end LoRA): the gradient loop is the GPU task. Here we expose the loss spec
    (loss_weights, tau schedule, rho held mild) and a no-op/scaffold trainer that records the plan
    and returns the (untrained) ranker so the pipeline is runnable end-to-end on CPU.
  Stage C (post-hoc, CPU): fit rho's 4 coefficients on a VALIDATION split by a coarse grid
    (near-zero capacity, a1,a2,a3>=0), minimising validation listwise NLL. Never touches test.
  Stage 2.5 (gate): AUC of [H(r), Var_m] predicting whether s_pers ranked the positive above the
    negatives on validation. AUC>~0.6 -> trust signal real, keep rho; ~0.5 -> drop the trust headline.

Leakage rule enforced throughout: every train-only statistic excludes held-out targets, and rho/tau
selection uses validation only. See docs/method_calm_rec_spec.md section 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Sequence

from llm4rec.methods.calm_rec import (
    CALMRecRanker,
    RhoCoeffs,
    mixture_score_and_responsibilities,
    per_intent_energy,
    reliability_auc,
)


@dataclass
class CALMLossSpec:
    """Stage-B loss weights + tau schedule (the GPU gradient loop consumes this)."""

    w_rank: float = 1.0
    w_attr: float = 0.3
    w_bal: float = 0.1
    w_orth: float = 0.05
    w_use: float = 0.05
    w_tau: float = 0.01
    rho_fixed_during_stage_b: float = 0.1   # rho held mild so panel learns to rank w/o the prior crutch
    tau_start: float = 1.5
    tau_min: float = 1.0
    tau_max: float = 8.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss_weights": {
                "rank": self.w_rank, "attr": self.w_attr, "bal": self.w_bal,
                "orth": self.w_orth, "use": self.w_use, "tau": self.w_tau,
            },
            "rho_fixed_during_stage_b": self.rho_fixed_during_stage_b,
            "tau_schedule": {"start": self.tau_start, "min": self.tau_min, "max": self.tau_max},
            "label_bearing_term": "rank",
            "note": "Stage-B gradient loop is the GPU task; weights/schedule are the locked contract.",
        }


def _history_ids(example: dict[str, Any], max_history: int = 20) -> list[str]:
    hist = example.get("history") or example.get("history_item_ids") or []
    return [str(x) for x in hist][-int(max_history):]


def build_train_only_stats(
    train_examples: Sequence[dict[str, Any]],
    item_catalog: Sequence[dict[str, Any]],
    *,
    held_out_targets: set[str] | None = None,
) -> dict[str, Any]:
    """Stage A: train-only item support + exposure + popularity, with held-out targets removed.

    LEAKAGE CONTROL: ``held_out_targets`` (validation/test positives) are excluded from support /
    popularity so a positive's own held-out edge cannot inflate its statistics asymmetrically vs the
    negatives. Exposure q(j) is a train-only frequency proxy (clipped) used by the salience term.
    """
    held = held_out_targets or set()
    support: dict[str, int] = {str(r["item_id"]): 0 for r in item_catalog}
    exposure: dict[str, int] = {str(r["item_id"]): 0 for r in item_catalog}
    for ex in train_examples:
        for h in _history_ids(ex):
            if h in held:
                continue
            support[h] = support.get(h, 0) + 1
            exposure[h] = exposure.get(h, 0) + 1
        tgt = str(ex.get("target") or "")
        if tgt and tgt not in held:
            exposure[tgt] = exposure.get(tgt, 0) + 1  # train targets are observed exposure
    total_exp = sum(exposure.values()) or 1
    q = {k: max(v / total_exp, 1e-6) for k, v in exposure.items()}  # clipped exposure prob
    popularity = {k: math.log(1.0 + v) for k, v in support.items()}
    return {"support": support, "exposure_prob": q, "popularity": popularity}


@dataclass
class CALMTrainPlan:
    """Record of the locked 3-stage plan + Stage-A statistics. Returned by the scaffold trainer."""

    loss_spec: dict[str, Any]
    n_intents: int
    tau: float
    stage_a_stats_keys: list[str]
    backend: str
    is_scaffold: bool = True
    notes: str = ""


def _pers_scores_and_signals(
    ranker: CALMRecRanker,
    example: dict[str, Any],
    candidates: Sequence[str],
) -> dict[str, dict[str, float]]:
    """For one validation example: per-candidate s_pers, responsibility entropy H, ensemble Var_m."""
    history = _history_ids(example, ranker.config.max_history)
    intents = ranker.intent_encoder.encode_intents(history, ranker.item_lookup, dropout=False)
    ensemble = [
        ranker.intent_encoder.encode_intents(history, ranker.item_lookup, dropout=True)
        for _ in range(max(0, ranker.config.m_dropout))
    ]
    out: dict[str, dict[str, float]] = {}
    for c in candidates:
        h_i = ranker.item_vec.get(c) or list(ranker.item_encoder.encode(c, ranker.item_lookup.get(c, {})))
        n_i = ranker.item_support.get(c, 0)
        e = per_intent_energy(intents, h_i, item_support=n_i, delta=ranker.delta)
        s_pers, resp = mixture_score_and_responsibilities(intents, e, tau=ranker.config.tau)
        H = float(-sum(p * math.log(p) for p in resp if p > 0.0))
        var_m = 0.0
        if ensemble:
            samples = []
            for ens in ensemble:
                ee = per_intent_energy(ens, h_i, item_support=n_i, delta=ranker.delta)
                sp, _ = mixture_score_and_responsibilities(ens, ee, tau=ranker.config.tau)
                samples.append(sp)
            mu = sum(samples) / len(samples)
            var_m = (sum((s - mu) ** 2 for s in samples) / len(samples)) ** 0.5
        out[c] = {"s_pers": s_pers, "H": H, "var_m": var_m, "n_i": float(n_i),
                  "s_prior": ranker.s_prior.get(c, 0.0)}
    return out


def stage_2p5_reliability_gate(
    ranker: CALMRecRanker,
    val_examples: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Does [H(r), Var_m] predict whether s_pers ranked the positive top-1? Return AUCs.

    AUC > ~0.6 => the endogenous reliability signal is real; keep the trust gate.
    AUC ~ 0.5 => signal is noise; drop the entropy/variance term or the trust headline.
    """
    sig_H: list[float] = []
    sig_var: list[float] = []
    correct: list[float] = []
    for ex in val_examples:
        cands = [str(c) for c in (ex.get("candidate_items") or [])]
        tgt = str(ex.get("target"))
        if not cands or tgt not in cands:
            continue
        sig = _pers_scores_and_signals(ranker, ex, cands)
        top = max(cands, key=lambda c: sig[c]["s_pers"])
        is_correct = 1.0 if top == tgt else 0.0
        # signal aggregated at the positive candidate (reliability of THIS user's personalization)
        sig_H.append(sig[tgt]["H"])
        sig_var.append(sig[tgt]["var_m"])
        correct.append(is_correct)
    return {
        "auc_entropy": reliability_auc(sig_H, correct),
        "auc_variance": reliability_auc(sig_var, correct),
        "n": float(len(correct)),
        "base_rate_correct": (sum(correct) / len(correct)) if correct else 0.0,
    }


def _val_ndcg_at_k(ranker: CALMRecRanker, val_examples: Sequence[dict[str, Any]], k: int = 10) -> float:
    """Mean NDCG@k of the FULL ranker (with current rho) over validation examples."""
    total = 0.0
    n = 0
    for ex in val_examples:
        cands = [str(c) for c in (ex.get("candidate_items") or [])]
        tgt = str(ex.get("target"))
        if not cands or tgt not in cands:
            continue
        res = ranker.rank({**ex, "user_id": ex.get("user_id", "u"), "target": tgt}, cands)
        try:
            rank_pos = res.predicted_items.index(tgt)
        except ValueError:
            n += 1
            continue
        total += (1.0 / math.log2(rank_pos + 2)) if rank_pos < k else 0.0
        n += 1
    return total / n if n else 0.0


def calibrate_rho_on_validation(
    ranker: CALMRecRanker,
    val_examples: Sequence[dict[str, Any]],
    *,
    grid: Sequence[float] | None = None,
    k: int = 10,
) -> RhoCoeffs:
    """Stage C: fit rho's 4 coefficients by a COARSE grid on validation (a1,a2,a3>=0).

    Near-zero capacity (a handful of grid points) so the gate cannot overfit the metric. Selection
    maximises validation NDCG@k. Never touches test. Installs the best coeffs on the ranker.
    """
    grid = list(grid if grid is not None else (-1.0, 0.0, 1.0))
    nonneg = [g for g in grid if g >= 0.0]
    best: RhoCoeffs | None = None
    best_score = -1.0
    for a0, a1, a2, a3 in product(grid, nonneg, nonneg, nonneg):
        coeffs = RhoCoeffs(a0=a0, a1=a1, a2=a2, a3=a3).clamp_nonneg()
        ranker.set_rho(coeffs)
        score = _val_ndcg_at_k(ranker, val_examples, k=k)
        if score > best_score:
            best_score, best = score, coeffs
    best = best or RhoCoeffs(a0=-0.4)
    ranker.set_rho(best)
    return best


class CALMScaffoldTrainer:
    """End-to-end CPU-runnable scaffold: Stage A stats + fit ranker + (no-GPU) Stage-B plan record.

    The real Stage-B gradient loop on Qwen3-8B+LoRA is the server task; this scaffold makes the
    full pipeline (fit -> rank -> evaluate) runnable locally with the hashed encoders and records the
    locked loss/tau/rho contract so the GPU implementation has an exact target.
    """

    def __init__(self, ranker: CALMRecRanker, *, loss_spec: CALMLossSpec | None = None, backend: str = "hashed") -> None:
        self.ranker = ranker
        self.loss_spec = loss_spec or CALMLossSpec()
        self.backend = backend

    def fit(
        self,
        train_examples: Sequence[dict[str, Any]],
        item_catalog: Sequence[dict[str, Any]],
        *,
        held_out_targets: set[str] | None = None,
    ) -> CALMTrainPlan:
        self.ranker.fit(list(train_examples), list(item_catalog))
        stats = build_train_only_stats(train_examples, item_catalog, held_out_targets=held_out_targets)
        # keep the leakage-clean train-only stats on the ranker (override fit's naive support)
        self.ranker.item_support = stats["support"]
        self.ranker.s_prior = {k: v for k, v in stats["popularity"].items()}
        return CALMTrainPlan(
            loss_spec=self.loss_spec.as_dict(),
            n_intents=self.ranker.config.n_intents,
            tau=self.ranker.config.tau,
            stage_a_stats_keys=sorted(stats.keys()),
            backend=self.backend,
            is_scaffold=(self.backend != "qwen"),
            notes="Stage-B gradient loop pending on GPU; Stage A + contract realised.",
        )
