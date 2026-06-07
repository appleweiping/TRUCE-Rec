"""Unit tests for SCALR (Set-aware Calibrated Lift Ranking) — the TRUCE-Rec Ours method.

These exercise the pure scoring core and the ranker contract with a deterministic mock
panel scorer (no GPU, no model). They encode the load-bearing invariants from
docs/method_redesign_decision.md.
"""

from __future__ import annotations

from llm4rec.methods.scalr import (
    MockPanelScorer,
    MonotoneCalibration,
    SCALRConfig,
    SCALRRanker,
    combine_scores,
    echo_risk,
    panel_instability,
    popularity_residual_lift,
)


def _catalog() -> list[dict[str, str]]:
    return [
        {"item_id": "i1", "title": "red lipstick matte", "category": "makeup"},
        {"item_id": "i2", "title": "red lip gloss shiny", "category": "makeup"},
        {"item_id": "i3", "title": "garden hose long", "category": "garden"},
        {"item_id": "i4", "title": "blue eyeshadow palette makeup", "category": "makeup"},
    ]


def _train() -> list[dict[str, object]]:
    return [
        {"example_id": "u1:1", "user_id": "u1", "history": ["i1"], "target": "i2", "split": "train", "domain": "tiny"},
        {"example_id": "u2:1", "user_id": "u2", "history": ["i3"], "target": "i3", "split": "train", "domain": "tiny"},
    ]


# ---- pure functions -------------------------------------------------------------------
def test_popularity_residual_lift_subtracts_null() -> None:
    lift = popularity_residual_lift({"a": 3.0, "b": 1.0}, {"a": 2.0, "b": 2.0}, ["a", "b"])
    assert lift == {"a": 1.0, "b": -1.0}


def test_panel_instability_zero_when_constant_and_single_pass() -> None:
    assert panel_instability([{"a": 1.0}, {"a": 1.0}], ["a"]) == {"a": 0.0}
    assert panel_instability([{"a": 1.0}], ["a"]) == {"a": 0.0}  # R<2 -> undefined -> 0


def test_panel_instability_positive_when_varies() -> None:
    u = panel_instability([{"a": 0.0}, {"a": 2.0}], ["a"])
    assert u["a"] > 0.0


def test_echo_risk_repeat_and_neighbor() -> None:
    idx = {"i1": {"i2"}, "i2": {"i1"}}
    assert echo_risk("i1", ["i1"], idx) == 1.0          # pure repeat
    assert echo_risk("i2", ["i1"], idx) == 1.0          # neighbor of all history
    assert echo_risk("i9", ["i1"], idx) == 0.0          # unrelated
    assert echo_risk("i1", [], idx) == 0.0              # no history


def test_combine_is_additive_and_penalises() -> None:
    cal = {"a": 1.0, "b": 1.0}
    inst = {"a": 0.0, "b": 1.0}
    echo = {"a": 0.0, "b": 0.0}
    s = combine_scores(cal, inst, echo, ["a", "b"], lam=2.0, beta=1.0)
    assert s["a"] == 1.0
    assert s["b"] == 1.0 - 2.0  # instability penalty applied additively
    assert s["a"] > s["b"]      # the unstable candidate is pushed down


def test_monotone_calibration_is_order_preserving() -> None:
    # isotonic on noisy-but-trending data must not invert order of inputs
    cal = MonotoneCalibration.fit([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.5, 1.0])
    vals = [cal.transform(x) for x in [0.0, 1.0, 2.0, 3.0]]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_calibration_identity_when_empty() -> None:
    cal = MonotoneCalibration()
    assert cal.transform(0.7) == 0.7


# ---- ranker contract ------------------------------------------------------------------
def _ranker(**cfg) -> SCALRRanker:
    r = SCALRRanker(scorer=MockPanelScorer(), config=SCALRConfig(**cfg))
    r.fit(_train(), _catalog())
    return r


def test_rank_outputs_prediction_schema_shape() -> None:
    r = _ranker(n_perturbations=4, seed=1)
    example = {"example_id": "u1:2", "user_id": "u1", "history": ["i1"], "target": "i2",
               "split": "test", "domain": "tiny"}
    cands = ["i1", "i2", "i3", "i4"]
    res = r.rank(example, cands)
    assert res.method == "scalr"
    assert set(res.candidate_items) == set(cands)
    assert len(res.predicted_items) == len(cands) == len(res.scores)
    assert sorted(res.predicted_items) == sorted(cands)  # permutation of candidates
    rec = res.to_prediction_record()
    assert set(rec) >= {"user_id", "target_item", "candidate_items", "predicted_items", "scores", "method"}


def test_never_gates_or_generates() -> None:
    r = _ranker(n_perturbations=4)
    res = r.rank({"user_id": "u1", "target": "i2", "history": ["i1"], "domain": "tiny"},
                 ["i1", "i2", "i3", "i4"])
    meta = res.metadata["scalr"]
    assert meta["uses_gate"] is False
    assert meta["uses_generation"] is False


def test_deterministic_for_same_seed() -> None:
    a = _ranker(n_perturbations=8, seed=7)
    b = _ranker(n_perturbations=8, seed=7)
    ex = {"user_id": "u1", "target": "i2", "history": ["i1"], "domain": "tiny"}
    assert a.rank(ex, ["i1", "i2", "i3", "i4"]).scores == b.rank(ex, ["i1", "i2", "i3", "i4"]).scores


def test_lift_uses_history_to_separate_relevant_from_offtopic() -> None:
    # History is "red lipstick matte". The mock lift is driven by token overlap with
    # history, so candidates sharing tokens (i1, i2) must rank strictly above candidates
    # with zero history overlap (i3 garden, i4 non-overlapping). lam=beta=0 isolates lift.
    r = _ranker(n_perturbations=6, lam=0.0, beta=0.0, use_calibration=False)
    ex = {"user_id": "u1", "target": "i2", "history": ["i1"], "domain": "tiny"}
    res = r.rank(ex, ["i1", "i2", "i3", "i4"])
    rank_pos = {item: i for i, item in enumerate(res.predicted_items)}
    # the two history-overlapping items outrank both zero-overlap items
    assert max(rank_pos["i1"], rank_pos["i2"]) < min(rank_pos["i3"], rank_pos["i4"])


def test_echo_penalty_demotes_history_repeat() -> None:
    ex = {"user_id": "u1", "target": "i2", "history": ["i1"], "domain": "tiny"}
    # with a large echo weight, the history item i1 (echo=1.0) must not rank first
    r = _ranker(n_perturbations=4, lam=0.0, beta=10.0, use_calibration=False)
    res = r.rank(ex, ["i1", "i2", "i3", "i4"])
    assert res.predicted_items[0] != "i1"


def test_fit_calibration_does_not_crash_and_sets_map() -> None:
    r = _ranker(n_perturbations=4)
    cal_examples = [
        {"user_id": "u1", "target": "i2", "history": ["i1"], "candidate_items": ["i1", "i2", "i3", "i4"]},
        {"user_id": "u2", "target": "i4", "history": ["i1"], "candidate_items": ["i1", "i2", "i3", "i4"]},
    ]
    r.fit_calibration(cal_examples)
    assert r.calibration is not None
