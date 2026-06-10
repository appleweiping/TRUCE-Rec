"""Smoke test: CALM-Rec runs end-to-end through the official experiment runner + evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from llm4rec.experiments.runner import run_all


def test_calm_rec_runs_through_runner() -> None:
    result = run_all("configs/experiments/smoke_calm_rec.yaml")
    run_dir = Path(result["run_dir"])
    assert run_dir.name == "smoke_calm_rec_seed13"
    preds = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert preds, "no predictions produced"
    assert preds[0]["method"] == "calm_rec"
    meta = preds[0]["metadata"]["calm_rec"]
    assert meta["uses_gate_decision"] is False  # mixes, never accepts/abstains
    assert meta["uses_generation"] is False
    assert meta["n_intents"] == 4
    # every record is a valid permutation of its candidate set with aligned scores
    for rec in preds:
        assert sorted(rec["predicted_items"]) == sorted(rec["candidate_items"])
        assert len(rec["scores"]) == len(rec["predicted_items"])
    assert (run_dir / "metrics.json").exists()
