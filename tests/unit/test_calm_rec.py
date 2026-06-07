"""Unit tests for CALM-Rec (Calibrated trust over Attribute-anchored Latent Multi-intent).

Exercises the pure scoring core and ranker contract with deterministic mock encoders (no GPU).
Encodes the locked invariants from docs/method_calm_rec_spec.md.
"""

from __future__ import annotations

import math

from llm4rec.methods.calm_rec import (
    CALMConfig,
    CALMRecRanker,
    IntentSet,
    MockIntentEncoder,
    MockItemEncoder,
    RhoCoeffs,
    calibrated_mix,
    mixture_score_and_responsibilities,
    per_intent_energy,
    reliability_auc,
    trust_gate,
)

VOCAB = ["matte", "dewy", "red", "rose", "retinol", "spf", "hose", "rake", "drill", "steel"]


def _catalog() -> list[dict[str, str]]:
    return [
        {"item_id": "m1", "title": "matte red", "category": "makeup"},
        {"item_id": "m2", "title": "dewy rose", "category": "makeup"},
        {"item_id": "s1", "title": "retinol spf", "category": "skincare"},
        {"item_id": "g1", "title": "hose rake", "category": "garden"},
        {"item_id": "t1", "title": "drill steel", "category": "tools"},
    ]


def _train() -> list[dict[str, object]]:
    return [
        {"user_id": "u1", "history": ["m1"], "target": "m2", "domain": "tiny"},
        {"user_id": "u2", "history": ["g1"], "target": "t1", "domain": "tiny"},
        {"user_id": "u1", "history": ["m1", "m2"], "target": "s1", "domain": "tiny"},
    ]


# ---- pure functions -------------------------------------------------------------------
def test_per_intent_energy_subtracts_popularity() -> None:
    intents = IntentSet(z=[[1.0, 0.0], [0.0, 1.0]], pi=[0.5, 0.5])
    e = per_intent_energy(intents, [2.0, 3.0], item_support=0, delta=[0.0, 0.0])
    assert e == [2.0, 3.0]
    e2 = per_intent_energy(intents, [2.0, 3.0], item_support=99, delta=[1.0, 1.0])
    assert e2[0] < e[0] and e2[1] < e[1]  # popularity penalty applied


def test_mixture_is_soft_or_between_mean_and_max() -> None:
    intents = IntentSet(z=[[1.0], [1.0]], pi=[0.5, 0.5])
    # energies 0 and 4; soft-OR score must lie between mean(2) and max(4)
    s, resp = mixture_score_and_responsibilities(intents, [0.0, 4.0], tau=1.0)
    assert 2.0 <= s <= 4.0
    assert abs(sum(resp) - 1.0) < 1e-9
    assert resp[1] > resp[0]  # higher-energy intent gets more responsibility


def test_trust_gate_rises_with_entropy_and_falls_with_support() -> None:
    c = RhoCoeffs(a0=0.0, a1=2.0, a2=0.0, a3=1.0)
    low_H = trust_gate(resp_entropy=0.1, ensemble_var=0.0, item_support=0, coeffs=c)
    high_H = trust_gate(resp_entropy=1.0, ensemble_var=0.0, item_support=0, coeffs=c)
    assert high_H > low_H  # ambiguous intent -> more trust in prior
    more_support = trust_gate(resp_entropy=1.0, ensemble_var=0.0, item_support=500, coeffs=c)
    assert more_support < high_H  # well-supported item -> less shrinkage to prior


def test_calibrated_mix_respects_rho_floor() -> None:
    # even at rho=1.0, personalization keeps at least rho_floor weight
    out = calibrated_mix(s_pers=10.0, s_prior=0.0, rho=1.0, rho_floor=0.2)
    assert math.isclose(out, 0.2 * 10.0, rel_tol=1e-9)


def test_reliability_auc_detects_signal_and_noise() -> None:
    # signal high => wrong (correct=0); low => right (correct=1) -> strong AUC
    sig = [0.9, 0.8, 0.1, 0.2]
    correct = [0.0, 0.0, 1.0, 1.0]
    assert reliability_auc(sig, correct) > 0.9
    # constant signal => uninformative ~0.5
    assert abs(reliability_auc([0.5, 0.5, 0.5, 0.5], correct) - 0.5) < 1e-9


# ---- ranker contract ------------------------------------------------------------------
def _ranker(**cfg) -> CALMRecRanker:
    ie = MockItemEncoder(VOCAB)
    qe = MockIntentEncoder(ie, n_intents=cfg.get("n_intents", 4))
    r = CALMRecRanker(item_encoder=ie, intent_encoder=qe, config=CALMConfig(**cfg))
    r.fit(_train(), _catalog())
    return r


def test_rank_schema_shape_and_no_gate_decision() -> None:
    r = _ranker(n_intents=4, tau=2.0, m_dropout=4, seed=1)
    ex = {"user_id": "u1", "history": ["m1", "m2"], "target": "s1", "domain": "tiny"}
    cands = ["m2", "s1", "g1", "t1"]
    res = r.rank(ex, cands)
    assert res.method == "calm_rec"
    assert sorted(res.predicted_items) == sorted(cands)
    assert len(res.scores) == len(cands)
    meta = res.metadata["calm_rec"]
    assert meta["uses_gate_decision"] is False  # mixes, never accepts/abstains
    assert meta["uses_generation"] is False
    assert meta["n_intents"] == 4


def test_deterministic_same_seed() -> None:
    a = _ranker(n_intents=4, tau=2.0, m_dropout=4, seed=5)
    b = _ranker(n_intents=4, tau=2.0, m_dropout=4, seed=5)
    ex = {"user_id": "u1", "history": ["m1"], "target": "m2", "domain": "tiny"}
    assert a.rank(ex, ["m2", "s1", "g1", "t1"]).scores == b.rank(ex, ["m2", "s1", "g1", "t1"]).scores


def test_multi_intent_separates_relevant_category() -> None:
    # history = makeup; makeup/skincare candidates should outrank garden/tools
    r = _ranker(n_intents=4, tau=3.0, use_trust_gate=False)  # isolate the intent mixture
    ex = {"user_id": "u1", "history": ["m1", "m2"], "target": "s1", "domain": "tiny"}
    res = r.rank(ex, ["m2", "s1", "g1", "t1"])
    pos = {it: i for i, it in enumerate(res.predicted_items)}
    assert pos["m2"] < pos["g1"] and pos["m2"] < pos["t1"]


def test_trust_gate_off_equals_pure_personalized_metadata() -> None:
    r = _ranker(n_intents=4, use_trust_gate=False)
    ex = {"user_id": "u1", "history": ["m1"], "target": "m2", "domain": "tiny"}
    res = r.rank(ex, ["m2", "s1", "g1", "t1"])
    assert res.metadata["calm_rec"]["uses_trust_gate"] is False
    assert res.metadata["calm_rec"]["mean_rho"] == 0.0


def test_cold_start_empty_history_does_not_crash() -> None:
    r = _ranker(n_intents=4, tau=2.0)
    ex = {"user_id": "u9", "history": [], "target": "m2", "domain": "tiny"}
    res = r.rank(ex, ["m2", "s1", "g1", "t1"])
    assert len(res.predicted_items) == 4  # graceful: ranks by anchors/prior, no crash


def test_set_rho_clamps_negative_coeffs() -> None:
    r = _ranker(n_intents=4)
    r.set_rho(RhoCoeffs(a0=0.0, a1=-5.0, a2=-3.0, a3=-1.0))
    assert r.rho.a1 == 0.0 and r.rho.a2 == 0.0 and r.rho.a3 == 0.0  # a1,a2,a3 >= 0 enforced
