"""Smoke test: CALM-Rec end-to-end scoring -> prediction schema -> real evaluator.

Runs the TRUCE-Rec Ours v2 method (CALM-Rec) over a tiny multi-user same-candidate dataset with
deterministic mock encoders (no GPU, no model, no network), validates every record against the
prediction schema, and runs the real evaluator to compute ranking metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.evaluation.prediction_schema import validate_prediction
from llm4rec.methods.calm_rec import CALMConfig, CALMRecRanker, MockIntentEncoder, MockItemEncoder

VOCAB = ["matte", "dewy", "red", "rose", "retinol", "spf", "serum", "hose", "rake", "drill", "steel", "wrench"]


def _catalog() -> list[dict[str, str]]:
    return [
        {"item_id": "m1", "title": "matte red lipstick", "category": "makeup"},
        {"item_id": "m2", "title": "dewy rose blush", "category": "makeup"},
        {"item_id": "s1", "title": "retinol spf serum", "category": "skincare"},
        {"item_id": "s2", "title": "dewy serum spf", "category": "skincare"},
        {"item_id": "g1", "title": "hose rake garden", "category": "garden"},
        {"item_id": "t1", "title": "drill steel wrench", "category": "tools"},
    ]


def _train() -> list[dict[str, object]]:
    return [
        {"user_id": "u1", "history": ["m1"], "target": "m2", "split": "train", "domain": "beauty"},
        {"user_id": "u2", "history": ["g1"], "target": "t1", "split": "train", "domain": "beauty"},
        {"user_id": "u3", "history": ["s1"], "target": "s2", "split": "train", "domain": "beauty"},
    ]


def _test_examples() -> list[dict[str, object]]:
    return [
        {"user_id": "u1", "history": ["m1", "m2"], "target": "s1", "split": "test", "domain": "beauty",
         "candidate_items": ["s1", "g1", "t1", "m1"]},
        {"user_id": "u3", "history": ["s1", "s2"], "target": "m2", "split": "test", "domain": "beauty",
         "candidate_items": ["m2", "g1", "t1", "s1"]},
    ]


def _ranker() -> CALMRecRanker:
    ie = MockItemEncoder(VOCAB)
    qe = MockIntentEncoder(ie, n_intents=4)
    r = CALMRecRanker(item_encoder=ie, intent_encoder=qe, config=CALMConfig(n_intents=4, tau=2.5, m_dropout=4, seed=13))
    r.fit(_train(), _catalog())
    return r


def test_calm_rec_end_to_end_schema_and_evaluation(tmp_path: Path) -> None:
    ranker = _ranker()
    predictions = [
        ranker.rank(ex, list(ex["candidate_items"])).to_prediction_record()
        for ex in _test_examples()
    ]
    for rec in predictions:
        v = validate_prediction(rec)
        assert v["method"] == "calm_rec"
        assert len(v["scores"]) == len(v["predicted_items"])
        assert set(v["predicted_items"]) == set(v["candidate_items"])
        assert v["metadata"]["calm_rec"]["uses_gate_decision"] is False

    pred_path = tmp_path / "predictions.jsonl"
    pred_path.write_text("\n".join(json.dumps(r) for r in predictions) + "\n", encoding="utf-8")

    result = evaluate_predictions(
        predictions_jsonl=pred_path,
        output_dir=tmp_path / "eval",
        top_k=[5, 10],
        item_catalog=_catalog(),
        train_examples=_train(),
        all_examples=_train() + _test_examples(),
    )
    assert isinstance(result, dict)
    assert (tmp_path / "eval" / "metrics.json").exists()


def test_calm_rec_uses_intent_signal_not_constant(tmp_path: Path) -> None:
    # With trust gate off, the skincare target s1 (history is makeup-adjacent) should rank above the
    # clearly off-domain garden/tools candidates -> the intent mixture is doing real work.
    ie = MockItemEncoder(VOCAB)
    qe = MockIntentEncoder(ie, n_intents=4)
    r = CALMRecRanker(item_encoder=ie, intent_encoder=qe, config=CALMConfig(n_intents=4, tau=3.0, use_trust_gate=False))
    r.fit(_train(), _catalog())
    ex = {"user_id": "u1", "history": ["m1", "m2"], "target": "s1", "domain": "beauty"}
    res = r.rank(ex, ["s1", "g1", "t1", "m1"])
    pos = {it: i for i, it in enumerate(res.predicted_items)}
    assert pos["m1"] < pos["g1"] and pos["m1"] < pos["t1"]
