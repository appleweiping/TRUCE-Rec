"""SCALR — Set-aware Calibrated Lift Ranking (TRUCE-Rec Ours method).

Implements the locked method from ``docs/method_redesign_decision.md``. SCALR scores
each candidate in a fixed same-candidate panel by

    score_i = g(Delta_i) - lambda * u_i - beta * echo_i

where

    Delta_i = f(history, panel, i) - f(no-history, panel, i)   # popularity-residual lift
    u_i     = std_r f(history, panel^(r), i)                   # panel-instability uncertainty
    echo_i  = history-near-duplicate risk for candidate i in [0, 1]
    g(.)    = a single GLOBAL monotone recalibration of the lift (honesty only; ablatable)

Design invariants (see the decision doc):
- Uncertainty *reorders* candidates; it never gates / accepts / abstains. The worst case is
  "a competent panel scorer ranked by lift", never a fallback collapse (the R3 failure).
- Every term is per-candidate (reorders within a popularity bucket) or a deliberate honesty-only
  no-op. Nothing constant across a user's panel is credited with ranking lift.
- The Qwen3-8B+LoRA cross-encoder is abstracted behind ``PanelScorer`` so this module is fully
  unit/smoke testable with a deterministic mock and contains NO GPU dependency. The real scorer is
  wired server-side later.

This module is the inference/scoring contract. It does not train the adapter.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from llm4rec.rankers.base import CheckpointNotImplementedMixin, RankingResult


# --------------------------------------------------------------------------------------
# Scorer abstraction (f_theta). Real impl = Qwen3-8B+LoRA panel cross-encoder (server-side).
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelScoreRequest:
    """One panel-conditioned scoring request.

    ``history`` is the ordered list of history item ids (empty list = the no-history /
    null counterfactual context). ``panel`` is the candidate ordering presented to the
    scorer for this pass (perturbations reorder it). The scorer must return one raw
    score per candidate id, keyed by candidate id (order-independent on the way out).
    """

    history: list[str]
    panel: list[str]
    item_lookup: dict[str, dict[str, Any]]


class PanelScorer(Protocol):
    """Abstraction of the panel-conditioned cross-encoder f_theta.

    A real implementation reads a per-candidate score from the candidate's title-token
    span (NOT an index-tag token) after one Qwen3-8B+LoRA forward over the whole panel.
    """

    def score_panel(self, request: PanelScoreRequest) -> dict[str, float]:
        """Return {candidate_id: raw_score} for every candidate in ``request.panel``."""
        ...


# --------------------------------------------------------------------------------------
# Pure scoring core (the testable heart of SCALR). No model, no GPU, no I/O.
# --------------------------------------------------------------------------------------
def popularity_residual_lift(
    history_scores: dict[str, float],
    null_scores: dict[str, float],
    candidates: Sequence[str],
) -> dict[str, float]:
    """Delta_i = f(history, panel, i) - f(no-history, panel, i), per candidate.

    Subtracting the history-masked (null) panel pass removes generic / popularity-driven
    attractiveness, isolating the user-specific signal. Per-candidate => reorders.
    """
    return {
        c: float(history_scores.get(c, 0.0)) - float(null_scores.get(c, 0.0))
        for c in candidates
    }


def panel_instability(
    perturbed_history_scores: Sequence[dict[str, float]],
    candidates: Sequence[str],
) -> dict[str, float]:
    """u_i = std over R perturbed (reordered / history-truncated) passes of f(history, panel, i).

    High when a candidate's score is an artifact of panel position / context rather than
    stable user evidence. With R < 2 passes uncertainty is undefined => returns 0.0
    (the term then has no effect, which is the correct degenerate behaviour).
    """
    out: dict[str, float] = {}
    for c in candidates:
        vals = [float(p.get(c, 0.0)) for p in perturbed_history_scores]
        out[c] = float(statistics.pstdev(vals)) if len(vals) >= 2 else 0.0
    return out


def echo_risk(
    candidate: str,
    history_item_ids: Sequence[str],
    neighbor_index: dict[str, set[str]],
) -> float:
    """history-near-duplicate risk for one candidate in [0, 1].

    1.0 if the candidate IS a history item (pure repeat); else the fraction of history
    items for which the candidate is a precomputed (train-only) semantic neighbor. The
    next item is typically novel, so a high-echo candidate should be penalised, not
    promoted. Train-only neighbor index => no leakage.
    """
    hist = [str(h) for h in history_item_ids]
    if not hist:
        return 0.0
    if candidate in set(hist):
        return 1.0
    hits = sum(1 for h in hist if candidate in neighbor_index.get(h, set()))
    return float(hits) / float(len(hist))


def global_recalibrate(lift: dict[str, float], calibration: "MonotoneCalibration | None") -> dict[str, float]:
    """Apply a single GLOBAL monotone recalibration g(.) to every lift value.

    This is intentionally order-preserving on the raw lift (honesty / scale only). It
    is NOT a reordering mechanism and is ablatable; if ``calibration`` is None it is the
    identity.
    """
    if calibration is None:
        return dict(lift)
    return {c: calibration.transform(v) for c, v in lift.items()}


def combine_scores(
    calibrated_lift: dict[str, float],
    instability: dict[str, float],
    echo: dict[str, float],
    candidates: Sequence[str],
    *,
    lam: float,
    beta: float,
) -> dict[str, float]:
    """score_i = g(Delta_i) - lambda * u_i - beta * echo_i  (additive; locked form).

    Additive (not multiplicative shrinkage) because multiplicative rho_i*g(Delta_i) flips
    sign for negative lift and degenerates to a no-op within a bucket (decision doc 4.1).
    """
    return {
        c: float(calibrated_lift.get(c, 0.0))
        - float(lam) * float(instability.get(c, 0.0))
        - float(beta) * float(echo.get(c, 0.0))
        for c in candidates
    }


@dataclass
class MonotoneCalibration:
    """A single global monotone (isotonic-style) map fit on the calibration split.

    Fit with Pool-Adjacent-Violators on (lift -> hit) pairs; ``transform`` linearly
    interpolates within the fitted, sorted, monotone-nondecreasing knots. Because the
    map is monotone and global, it preserves the order of raw lift values and is kept
    only for probability honesty / scale. It is never relied on to reorder a panel.
    """

    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, lift_values: Sequence[float], hits: Sequence[float]) -> "MonotoneCalibration":
        pairs = sorted(zip((float(x) for x in lift_values), (float(y) for y in hits)), key=lambda p: p[0])
        if not pairs:
            return cls()
        # Pool-Adjacent-Violators for isotonic (nondecreasing) regression.
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        weights = [1.0] * len(ys)
        i = 0
        while i < len(ys) - 1:
            if ys[i] > ys[i + 1] + 1e-12:
                # pool i and i+1
                total_w = weights[i] + weights[i + 1]
                pooled = (ys[i] * weights[i] + ys[i + 1] * weights[i + 1]) / total_w
                ys[i] = pooled
                weights[i] = total_w
                del ys[i + 1]
                del weights[i + 1]
                del xs[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1
        return cls(xs=xs, ys=ys)

    def transform(self, x: float) -> float:
        if not self.xs:
            return float(x)
        if x <= self.xs[0]:
            return float(self.ys[0])
        if x >= self.xs[-1]:
            return float(self.ys[-1])
        for j in range(len(self.xs) - 1):
            if self.xs[j] <= x <= self.xs[j + 1]:
                x0, x1, y0, y1 = self.xs[j], self.xs[j + 1], self.ys[j], self.ys[j + 1]
                if x1 == x0:
                    return float(y0)
                t = (x - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        return float(self.ys[-1])


@dataclass
class SCALRConfig:
    """SCALR hyper-parameters. lam/beta are selected on validation by NDCG@10."""

    lam: float = 1.0           # weight on panel-instability penalty u_i
    beta: float = 1.0          # weight on echo penalty echo_i
    n_perturbations: int = 12  # R >= 12 (decision doc 3); marginalises panel position
    max_history: int = 50
    use_calibration: bool = True
    seed: int = 0

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "SCALRConfig":
        p = dict(params or {})
        return cls(
            lam=float(p.get("lam", p.get("lambda", 1.0))),
            beta=float(p.get("beta", 1.0)),
            n_perturbations=int(p.get("n_perturbations", p.get("R", 12))),
            max_history=int(p.get("max_history", 50)),
            use_calibration=bool(p.get("use_calibration", True)),
            seed=int(p.get("seed", 0)),
        )


def perturbed_panels(candidates: Sequence[str], *, n: int, seed: int) -> list[list[str]]:
    """Deterministic candidate reorderings for the R panel-instability passes.

    The first pass is the identity order; the rest are seeded shuffles. Returns ``n``
    orderings (n >= 1). History-truncation perturbations are applied by the ranker.
    """
    import random as _random

    base = [str(c) for c in candidates]
    out = [list(base)]
    rng = _random.Random(int(seed))
    for _ in range(max(0, int(n) - 1)):
        shuffled = list(base)
        rng.shuffle(shuffled)
        out.append(shuffled)
    return out


def _history_ids(example: dict[str, Any], max_history: int) -> list[str]:
    hist = example.get("history") or example.get("history_item_ids") or []
    return [str(x) for x in hist][-int(max_history):]


class SCALRRanker(CheckpointNotImplementedMixin):
    """Set-aware Calibrated Lift Ranking (TRUCE-Rec Ours).

    Scores a fixed candidate panel via a popularity-residual lift plus additive
    per-candidate uncertainty (panel-instability) and echo penalties. The cross-encoder
    is injected as ``scorer`` (PanelScorer); pass a deterministic mock for smoke tests.
    """

    method_name = "scalr"

    def __init__(
        self,
        *,
        scorer: PanelScorer,
        config: SCALRConfig | None = None,
        method_config: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.scorer = scorer
        mc = dict(method_config or {})
        self.method_name = str(mc.get("name") or self.method_name)
        if config is not None:
            self.config = config
        else:
            self.config = SCALRConfig.from_params(dict(mc.get("params") or {}))
            if not (mc.get("params") or {}).get("seed"):
                self.config.seed = int(mc.get("seed") or seed)
        self.item_lookup: dict[str, dict[str, Any]] = {}
        self.train_popularity: Counter[str] = Counter()
        self.neighbor_index: dict[str, set[str]] = {}
        self.calibration: MonotoneCalibration | None = None

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.item_lookup = {str(r["item_id"]): r for r in item_catalog}
        # Train-only popularity (used for popularity-bucket diagnostics downstream).
        pop: Counter[str] = Counter()
        for ex in train_examples:
            for h in _history_ids(ex, self.config.max_history):
                pop[h] += 1
            tgt = ex.get("target")
            if tgt is not None:
                pop[str(tgt)] += 1
        self.train_popularity = pop
        self.neighbor_index = self._build_neighbor_index(item_catalog)

    def _build_neighbor_index(self, item_catalog: list[dict[str, Any]]) -> dict[str, set[str]]:
        """Train-only category-based neighbor index for the echo penalty.

        Two items are neighbors if they share a (normalised) category. This is a cheap,
        leakage-free proxy for "near-duplicate of history"; the real adapter learns a
        finer echo head, but the inference contract only needs a neighbor lookup.
        """
        by_cat: dict[str, list[str]] = {}
        item_cat: dict[str, str] = {}
        for row in item_catalog:
            iid = str(row["item_id"])
            cat = str(row.get("category") or row.get("genres") or "").strip().lower()
            item_cat[iid] = cat
            if cat:
                by_cat.setdefault(cat, []).append(iid)
        index: dict[str, set[str]] = {}
        for iid, cat in item_cat.items():
            if not cat:
                index[iid] = set()
                continue
            index[iid] = {other for other in by_cat.get(cat, []) if other != iid}
        return index

    def fit_calibration(self, calibration_examples: list[dict[str, Any]]) -> None:
        """Fit the global monotone recalibration g(.) on a disjoint calibration split.

        For each calibration event, compute the candidate lifts and label each candidate
        by whether it is the target (hit=1) or not (hit=0); fit one global isotonic map.
        Order-preserving on raw lift => honesty/scale only, never reorders.
        """
        lifts: list[float] = []
        hits: list[float] = []
        for ex in calibration_examples:
            cands = [str(c) for c in (ex.get("candidate_items") or [])]
            if not cands:
                continue
            lift = self._candidate_lift(ex, cands)
            tgt = str(ex.get("target"))
            for c in cands:
                lifts.append(lift.get(c, 0.0))
                hits.append(1.0 if c == tgt else 0.0)
        if lifts:
            self.calibration = MonotoneCalibration.fit(lifts, hits)

    def set_hyperparams(self, *, lam: float | None = None, beta: float | None = None) -> None:
        """Set lam/beta after validation selection (never selected on test)."""
        if lam is not None:
            self.config.lam = float(lam)
        if beta is not None:
            self.config.beta = float(beta)

    def _candidate_lift(self, example: dict[str, Any], candidates: list[str]) -> dict[str, float]:
        history = _history_ids(example, self.config.max_history)
        hist_scores = self.scorer.score_panel(
            PanelScoreRequest(history=history, panel=candidates, item_lookup=self.item_lookup)
        )
        null_scores = self.scorer.score_panel(
            PanelScoreRequest(history=[], panel=candidates, item_lookup=self.item_lookup)
        )
        return popularity_residual_lift(hist_scores, null_scores, candidates)

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        candidates = [str(c) for c in candidate_items]
        history = _history_ids(example, self.config.max_history)

        # Popularity-residual lift (history pass - null pass), per candidate.
        lift = self._candidate_lift(example, candidates)

        # Panel-instability u_i over R reordered passes (the position-confound fix + the
        # uncertainty source). Scores are re-keyed by candidate id, so reordering is safe.
        orders = perturbed_panels(candidates, n=self.config.n_perturbations, seed=self.config.seed)
        perturbed = [
            self.scorer.score_panel(
                PanelScoreRequest(history=history, panel=order, item_lookup=self.item_lookup)
            )
            for order in orders
        ]
        instability = panel_instability(perturbed, candidates)

        # Echo risk from train-only neighbor index.
        echo = {
            c: echo_risk(c, history, self.neighbor_index)
            for c in candidates
        }

        calibrated = global_recalibrate(lift, self.calibration if self.config.use_calibration else None)
        scores = combine_scores(
            calibrated, instability, echo, candidates,
            lam=self.config.lam, beta=self.config.beta,
        )

        ordered = sorted(candidates, key=lambda c: (-scores.get(c, 0.0), c))
        return RankingResult(
            user_id=str(example["user_id"]),
            target_item=str(example["target"]),
            candidate_items=candidates,
            predicted_items=ordered,
            scores=[float(scores.get(c, 0.0)) for c in ordered],
            method=self.method_name,
            domain=str(example.get("domain") or "tiny"),
            raw_output=None,
            metadata={
                "example_id": example.get("example_id"),
                "split": example.get("split"),
                "scalr": {
                    "lam": self.config.lam,
                    "beta": self.config.beta,
                    "n_perturbations": self.config.n_perturbations,
                    "calibrated": bool(self.config.use_calibration and self.calibration is not None),
                    "uses_gate": False,
                    "uses_generation": False,
                },
            },
        )


# --------------------------------------------------------------------------------------
# Deterministic mock scorer for smoke / unit tests. NOT a model; NOT paper evidence.
# --------------------------------------------------------------------------------------
class MockPanelScorer:
    """A deterministic, dependency-free stand-in for the Qwen3-8B panel cross-encoder.

    It fabricates a plausible signal purely from text overlap so the SCALR scoring
    contract can be exercised without a GPU:
    - a history-conditioned component: token overlap between candidate title and history
      titles (this disappears when history is empty -> drives a real, non-zero lift);
    - a popularity-like component present in BOTH history and null passes (cancels in the
      lift, exactly what the popularity-residual is meant to remove);
    - a tiny position-dependent jitter so reordered passes differ -> non-zero instability.
    Outputs are keyed by candidate id (order-independent), matching the real contract.
    """

    def __init__(self, *, position_jitter: float = 0.01) -> None:
        self.position_jitter = float(position_jitter)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in str(text).lower().replace("|", " ").split() if t}

    def _title(self, item_id: str, lookup: dict[str, dict[str, Any]]) -> str:
        row = lookup.get(str(item_id), {})
        return str(row.get("title") or row.get("raw_text") or item_id)

    def score_panel(self, request: PanelScoreRequest) -> dict[str, float]:
        hist_tokens: set[str] = set()
        for h in request.history:
            hist_tokens |= self._tokens(self._title(h, request.item_lookup))
        scores: dict[str, float] = {}
        for pos, cid in enumerate(request.panel):
            ctoks = self._tokens(self._title(cid, request.item_lookup))
            # popularity-like component (length proxy) present in both passes -> cancels in lift
            pop_component = 0.1 * len(ctoks)
            # history-conditioned overlap -> only present when history is non-empty
            overlap = len(ctoks & hist_tokens) if hist_tokens else 0
            hist_component = float(overlap)
            jitter = self.position_jitter * ((pos % 3) - 1)  # -j, 0, +j cycling
            scores[str(cid)] = pop_component + hist_component + jitter
        return scores
