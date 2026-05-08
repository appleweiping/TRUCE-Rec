from pathlib import Path

from scripts.server.dry_run_week8_four_domain import build_report


def test_week8_server_dry_run_reports_missing_required(tmp_path: Path) -> None:
    report = build_report(
        source_root=tmp_path,
        domains=["books"],
        splits=["test"],
        expected_users=10000,
        expected_candidates=101,
    )
    assert report["missing_task_dirs"]
    assert report["missing_required"]
    assert report["estimated_candidate_rows_if_complete"] == 1010000


def test_week8_server_dry_run_accepts_ready_task(tmp_path: Path) -> None:
    task = tmp_path / "books_large10000_100neg_test_same_candidate"
    task.mkdir()
    for name in [
        "ranking_test.jsonl",
        "candidate_items.csv",
        "train_interactions.csv",
        "item_metadata.csv",
        "selected_users.csv",
        "metadata.json",
    ]:
        (task / name).write_text("", encoding="utf-8")
    report = build_report(
        source_root=tmp_path,
        domains=["books"],
        splits=["test"],
        expected_users=10000,
        expected_candidates=101,
    )
    assert report["missing_task_dirs"] == []
    assert report["missing_required"] == []
    assert report["checks"][0]["status"] == "ready"
