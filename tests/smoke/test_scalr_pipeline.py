"""Smoke test: SCALR end-to-end scoring -> prediction schema -> evaluator.

Runs the TRUCE-Rec Ours method (SCALR) over a tiny multi-user same-candidate dataset
with the deterministic mock panel scorer, writes predictions.jsonl, validates every
record against the prediction schema, and runs the real evaluator to compute ranking
metrics. No GPU, no model, no network. This proves the full scoring -> evaluation
contract holds before any server run.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.evaluation.prediction_schema import validate_prediction
from llm4rec.methods.scalr import MockPanelScorer, SCALRConfig, SCALRRanker


def _catalog() -> list[dict[str, str]]:
    return [
        {"item_id": "m1", "title": "red lipstick matte", "category": "makeup"},
        {"item_id": "m2", "title": "red lip gloss shiny", "category": "makeup"},
        {"item_id": "m3", "title": "matte lipstick rose", "category": "makeup"},
        {"item_id": "g1", "title": "garden hose long", "category": "garden"},
        {"item_id": "g2", "title": "steel garden rake", "category": "garden"},
        {"item_id": "t1", "title": "cordless power drill", "category": "tools"},
    ]


def _train() -> list[dict[str, object]]:
    return [
        {"example_id": "u1:1", "user_id": "u1", "history": ["m1"], "target": "m2", "split": "train", "domain": "beauty"},
        {"example_id": "u2:1", "user_id": "u2", "history": ["g1"], "target": "g2", "split": "train", "domain": "beauty"},
        {"example_id": "u3:1", "user_id": "u3", "history": ["t1"], "target": "m3", "split": "train", "domain": "beauty"},
    ]


def _test_examples() -> list[dict[str, object]]:
    # each test user: history in one category, target is a same-category item in the panel
    return [
        {"example_id": "u1:2", "user_id": "u1", "history": ["m1", "m2"], "target": "m3",
         "split": "test", "domain": "beauty", "candidate_items": ["m3", "g1", "g2", "t1"]},
        {"example_id": "u2:2", "user_id": "u2", "history": ["g1", "g2"], "target": "g1",
         "split": "test", "domain": "beauty", "candidate_items": ["g1", "m1", "m2", "t1"]},
    ]


def test_scalr_end_to_end_schema_and_evaluation(tmp_path: Path) -> None:
    ranker = SCALRRanker(scorer=MockPanelScorer(), config=SCALRConfig(n_perturbations=8, lam=0.5, beta=0.5, seed=13))
    ranker.fit(_train(), _catalog())

    predictions = [
        ranker.rank(ex, list(ex["candidate_items"])).to_prediction_record()
        for ex in _test_examples()
    ]

    # every record is schema-valid (source-of-truth validator used by the evaluator)
    for rec in predictions:
        validated = validate_prediction(rec)
        assert validated["method"] == "scalr"
        assert len(validated["scores"]) == len(validated["predicted_items"])
        assert set(validated["predicted_items"]) == set(validated["candidate_items"])
        assert validated["metadata"]["scalr"]["uses_gate"] is False

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
    metrics_file = tmp_path / "eval" / "metrics.json"
    assert metrics_file.exists()


def test_scalr_beats_random_on_signal(tmp_path: Path) -> None:
    # With history-aligned mock signal, the same-category target should be ranked at or
    # near the top — i.e. SCALR is not producing a degenerate / constant ordering.
    ranker = SCALRRanker(scorer=MockPanelScorer(), config=SCALRConfig(n_perturbations=8, lam=0.3, beta=0.3, seed=1))
    ranker.fit(_train(), _catalog())
    ex = {"user_id": "u1", "target": "m3", "history": ["m1", "m2"], "domain": "beauty"}
    res = ranker.rank(ex, ["m3", "g1", "g2", "t1"])
    # the makeup target outranks every off-category candidate
    pos = {item: i for i, item in enumerate(res.predicted_items)}
    assert pos["m3"] < pos["g1"] and pos["m3"] < pos["g2"] and pos["m3"] < pos["t1"]
