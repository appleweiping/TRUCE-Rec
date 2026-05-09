import json

from scripts.validate_week8_same_candidate_processed import validate_split


def test_week8_validator_accepts_aligned_processed_dir(tmp_path):
    base = tmp_path / "books_large10000_100neg" / "test"
    base.mkdir(parents=True)
    example = {
        "example_id": "e1",
        "user_id": "u1",
        "target": "i1",
        "candidates": ["i1", "i2", "i3"],
        "split": "test",
        "metadata": {
            "event_id": "e1",
            "source_event_id": "src1",
            "target_inserted_by_converter": False,
        },
    }
    (base / "examples.jsonl").write_text(json.dumps(example) + "\n", encoding="utf-8")
    (base / "preprocess_manifest.json").write_text("{}", encoding="utf-8")
    result = validate_split(
        root=tmp_path,
        domain="books",
        split="test",
        expected_users=1,
        expected_candidates=3,
        expected_negatives=2,
    )
    assert result["status"] == "passed"
    assert result["examples"] == 1


def test_week8_validator_accepts_beauty_supplementary_slug(tmp_path):
    base = tmp_path / "beauty_supplementary_smallerN_100neg" / "test"
    base.mkdir(parents=True)
    example = {
        "example_id": "e1",
        "user_id": "u1",
        "target": "i1",
        "candidates": ["i1", "i2", "i3"],
        "split": "test",
        "metadata": {
            "event_id": "e1",
            "source_event_id": "src1",
            "target_inserted_by_converter": False,
        },
    }
    (base / "examples.jsonl").write_text(json.dumps(example) + "\n", encoding="utf-8")
    (base / "preprocess_manifest.json").write_text("{}", encoding="utf-8")
    result = validate_split(
        root=tmp_path,
        domain="beauty",
        split="test",
        expected_users=1,
        expected_candidates=3,
        expected_negatives=2,
    )
    assert result["task_slug"] == "beauty_supplementary_smallerN_100neg"
    assert result["status"] == "passed"
