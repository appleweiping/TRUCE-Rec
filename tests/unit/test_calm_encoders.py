"""Unit tests for CALM-Rec encoders, weak-labels, and the 3-stage trainer scaffold (no GPU)."""

from __future__ import annotations

from llm4rec.methods.calm_encoders import (
    HashedItemEncoder,
    LexiconIntentEncoder,
    QwenItemEncoder,
    build_encoders,
)
from llm4rec.methods.calm_rec import CALMConfig, CALMRecRanker
from llm4rec.methods.calm_trainer import (
    CALMLossSpec,
    CALMScaffoldTrainer,
    build_train_only_stats,
    calibrate_rho_on_validation,
    stage_2p5_reliability_gate,
)
from llm4rec.methods.calm_weak_labels import AttributeLexicon, default_beauty_lexicon


def _catalog() -> list[dict[str, str]]:
    return [
        {"item_id": "m1", "title": "matte red lipstick", "category": "makeup", "brand": "x"},
        {"item_id": "s1", "title": "retinol spf serum", "category": "skincare", "brand": "y"},
        {"item_id": "g1", "title": "garden hose", "category": "garden", "brand": "z"},
    ]


def _train() -> list[dict[str, object]]:
    return [
        {"user_id": "u1", "history": ["m1"], "target": "s1", "split": "train",
         "candidate_items": ["s1", "g1", "m1"], "domain": "beauty"},
        {"user_id": "u2", "history": ["g1"], "target": "m1", "split": "train",
         "candidate_items": ["m1", "s1", "g1"], "domain": "beauty"},
    ]


# ---- weak labels ----------------------------------------------------------------------
def test_lexicon_soft_label_is_distribution() -> None:
    lex = default_beauty_lexicon()
    soft = lex.soft_label({"title": "hydrating retinol serum spf", "category": "skincare"})
    assert abs(sum(soft.values()) - 1.0) < 1e-9
    assert set(soft.keys()) == set(lex.facet_names)


def test_lexicon_detects_dominant_facet() -> None:
    lex = default_beauty_lexicon()
    dom = lex.dominant_facet({"title": "retinol hyaluronic niacinamide vitamin c spf serum"})
    assert dom == "brand_ingredient"
    assert lex.dominant_facet({"title": "plain wooden box"}) is None


# ---- encoders -------------------------------------------------------------------------
def test_hashed_item_encoder_is_deterministic_and_normalised() -> None:
    lex = default_beauty_lexicon()
    enc = HashedItemEncoder(dim=32, lexicon=lex)
    v1 = enc.encode("m1", _catalog()[0])
    v2 = enc.encode("m1", _catalog()[0])
    assert v1 == v2
    assert enc.out_dim == 32 + lex.n_facets
    assert len(v1) == enc.out_dim


def test_intent_encoder_returns_k_intents_summing_to_one() -> None:
    lex = default_beauty_lexicon()
    ie = HashedItemEncoder(dim=32, lexicon=lex)
    qe = LexiconIntentEncoder(ie, n_intents=4, lexicon=lex, seed=1)
    iset = qe.encode_intents(["m1"], {r["item_id"]: r for r in _catalog()})
    assert len(iset.z) == 4 and len(iset.pi) == 4
    assert abs(sum(iset.pi) - 1.0) < 1e-9


def test_intent_encoder_dropout_changes_vectors() -> None:
    lex = default_beauty_lexicon()
    ie = HashedItemEncoder(dim=32, lexicon=lex)
    qe = LexiconIntentEncoder(ie, n_intents=4, lexicon=lex, seed=1)
    lk = {r["item_id"]: r for r in _catalog()}
    a = qe.encode_intents(["m1"], lk, dropout=True)
    b = qe.encode_intents(["m1"], lk, dropout=True)
    assert a.z != b.z  # seeded perturbation differs across draws -> non-zero ensemble variance


def test_build_encoders_qwen_requires_path() -> None:
    import pytest
    with pytest.raises(ValueError):
        build_encoders(backend="qwen")  # no model path
    enc = QwenItemEncoder(model_path="/nonexistent")
    with pytest.raises(NotImplementedError):
        enc.encode("x", {})  # gated, never silently runs


# ---- trainer scaffold + leakage controls ----------------------------------------------
def test_train_only_stats_excludes_held_out_targets() -> None:
    # 's1' and 'm1' are targets; if held out they must not be counted in support
    stats_all = build_train_only_stats(_train(), _catalog(), held_out_targets=set())
    stats_held = build_train_only_stats(_train(), _catalog(), held_out_targets={"m1"})
    # m1 appears in u1's history; holding it out removes that support
    assert stats_held["support"]["m1"] <= stats_all["support"]["m1"]
    assert all(v >= 1e-6 for v in stats_held["exposure_prob"].values())  # clipped


def test_scaffold_trainer_fits_and_records_contract() -> None:
    ie, qe = build_encoders(backend="hashed", n_intents=4)
    ranker = CALMRecRanker(item_encoder=ie, intent_encoder=qe, config=CALMConfig(n_intents=4))
    trainer = CALMScaffoldTrainer(ranker, loss_spec=CALMLossSpec(), backend="hashed")
    plan = trainer.fit(_train(), _catalog(), held_out_targets=set())
    assert plan.is_scaffold is True
    assert plan.loss_spec["label_bearing_term"] == "rank"
    assert plan.loss_spec["rho_fixed_during_stage_b"] == 0.1  # rho held mild during Stage B
    assert ranker.item_vec  # fit populated the item cache


def test_rho_calibration_runs_and_installs_nonneg_coeffs() -> None:
    ie, qe = build_encoders(backend="hashed", n_intents=4)
    ranker = CALMRecRanker(item_encoder=ie, intent_encoder=qe, config=CALMConfig(n_intents=4))
    CALMScaffoldTrainer(ranker, backend="hashed").fit(_train(), _catalog())
    coeffs = calibrate_rho_on_validation(ranker, _train(), grid=(-1.0, 0.0, 1.0))
    assert coeffs.a1 >= 0.0 and coeffs.a2 >= 0.0 and coeffs.a3 >= 0.0


def test_reliability_gate_returns_aucs() -> None:
    ie, qe = build_encoders(backend="hashed", n_intents=4)
    ranker = CALMRecRanker(item_encoder=ie, intent_encoder=qe, config=CALMConfig(n_intents=4))
    CALMScaffoldTrainer(ranker, backend="hashed").fit(_train(), _catalog())
    gate = stage_2p5_reliability_gate(ranker, _train())
    assert "auc_entropy" in gate and "auc_variance" in gate
    assert 0.0 <= gate["auc_entropy"] <= 1.0
