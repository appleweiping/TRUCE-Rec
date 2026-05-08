import csv
import json
from pathlib import Path

from scripts.prepare_ours_qwen_adapter_training import prepare


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_prepare_ours_adapter_training_contract(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    rows = [
        {
            "example_id": "e_train",
            "user_id": "u1",
            "history": ["i1"],
            "target": "i2",
            "candidates": ["i2", "i3"],
            "split": "train",
            "domain": "tiny",
            "metadata": {"event_id": "e_train", "source_event_id": "s_train"},
        },
        {
            "example_id": "e_test",
            "user_id": "u1",
            "history": ["i1", "i2"],
            "target": "i3",
            "candidates": ["i2", "i3"],
            "split": "test",
            "domain": "tiny",
            "metadata": {"event_id": "e_test", "source_event_id": "s_test"},
        },
    ]
    _write_jsonl(processed / "examples.jsonl", rows)
    with (processed / "items.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "title", "category", "brand", "raw_text"])
        writer.writeheader()
        writer.writerow({"item_id": "i1", "title": "One", "category": "cat", "brand": "", "raw_text": "One"})
        writer.writerow({"item_id": "i2", "title": "Two", "category": "cat", "brand": "", "raw_text": "Two"})
        writer.writerow({"item_id": "i3", "title": "Three", "category": "cat", "brand": "", "raw_text": "Three"})
    with (processed / "interactions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "item_id"])
        writer.writeheader()
        writer.writerow({"user_id": "u1", "item_id": "i1"})
        writer.writerow({"user_id": "u1", "item_id": "i2"})
        out = tmp_path / "ours_adapter"
    manifest = prepare(
        processed_dir=processed,
        processed_root=None,
        output_dir=out,
        base_model="/models/qwen",
        domain="tiny",
        seed=13,
        negatives_per_example=1,
        max_history=5,
    )
    assert manifest["counts"]["train_sft_rows"] == 3
    assert manifest["counts"]["score_rows"] == 2
    assert Path(manifest["files"]["test_examples"]).exists() or manifest["files"]["test_examples"].endswith("examples.jsonl")
    train_first = json.loads((out / "train_sft.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "TRUCE uncertainty-aware recommendation task" in train_first["messages"][0]["content"]
    answer = json.loads(train_first["messages"][1]["content"])
    assert "grounding_risk" in answer
    assert "popularity_bias_risk" in answer
    assert "history_repetition_risk" in answer
    assert "feature_targets" in train_first["metadata"]
    score_first = json.loads((out / "test_score_plan.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert score_first["metadata"]["source_event_id"] == "s_test"
    assert (out / "ours_adapter_manifest.json").exists()
    assert manifest["objective_family"] == "truce_uncertainty_structured_sft"
    assert manifest["negative_sampling_policy"] == "stratified_head_tail_echo_stable_negatives"


def test_prepare_ours_adapter_training_from_week8_root(tmp_path: Path) -> None:
    root = tmp_path / "week8" / "books_large10000_100neg"
    valid = root / "valid"
    test = root / "test"
    valid.mkdir(parents=True)
    test.mkdir(parents=True)
    valid_rows = [
        {
            "example_id": "e_valid",
            "user_id": "u1",
            "history": ["i1"],
            "target": "i2",
            "candidates": ["i2", "i3"],
            "split": "valid",
            "domain": "books",
        }
    ]
    test_rows = [
        {
            "example_id": "e_test",
            "user_id": "u2",
            "history": ["i1"],
            "target": "i3",
            "candidates": ["i2", "i3"],
            "split": "test",
            "domain": "books",
        }
    ]
    _write_jsonl(valid / "examples.jsonl", valid_rows)
    _write_jsonl(test / "examples.jsonl", test_rows)
    for split_dir in (valid, test):
        with (split_dir / "items.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["item_id", "title", "category", "brand", "raw_text"])
            writer.writeheader()
            writer.writerow({"item_id": "i1", "title": "One", "category": "cat", "brand": "", "raw_text": "One"})
            writer.writerow({"item_id": "i2", "title": "Two", "category": "cat", "brand": "", "raw_text": "Two"})
            writer.writerow({"item_id": "i3", "title": "Three", "category": "cat", "brand": "", "raw_text": "Three"})
        with (split_dir / "interactions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["user_id", "item_id"])
            writer.writeheader()
            writer.writerow({"user_id": "u1", "item_id": "i1"})
    out = tmp_path / "ours_week8"
    manifest = prepare(
        processed_dir=None,
        processed_root=root,
        output_dir=out,
        base_model="/models/qwen",
        domain="books",
        seed=13,
        negatives_per_example=1,
        max_history=5,
    )
    assert manifest["counts"]["train_examples"] == 1
    assert manifest["counts"]["test_examples"] == 1
    assert manifest["counts"]["score_rows"] == 2
