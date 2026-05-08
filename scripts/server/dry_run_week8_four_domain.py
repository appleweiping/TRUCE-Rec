#!/usr/bin/env python3
"""Preflight Week8 four-domain source tasks before server conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DOMAINS = ["beauty", "books", "electronics", "movies"]
DEFAULT_SPLITS = ["valid", "test"]
REQUIRED_FILES = ["candidate_items.csv", "train_interactions.csv", "item_metadata.csv", "selected_users.csv", "metadata.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks").expanduser(),
    )
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--expected-users", type=int, default=10000)
    parser.add_argument("--expected-candidates", type=int, default=101)
    parser.add_argument("--allow-missing-beauty", action="store_true")
    args = parser.parse_args()
    report = build_report(
        source_root=args.source_root,
        domains=args.domains,
        splits=args.splits,
        expected_users=args.expected_users,
        expected_candidates=args.expected_candidates,
        allow_missing_beauty=args.allow_missing_beauty,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if report["missing_required"] or report["missing_task_dirs"]:
        return 1
    return 0


def build_report(
    *,
    source_root: Path,
    domains: list[str],
    splits: list[str],
    expected_users: int,
    expected_candidates: int,
    allow_missing_beauty: bool = False,
) -> dict[str, Any]:
    checks = []
    missing_dirs = []
    missing_files = []
    for domain in domains:
        for split in splits:
            task_dir = source_root / f"{domain}_large10000_100neg_{split}_same_candidate"
            ranking_file = task_dir / f"ranking_{split}.jsonl"
            files = [ranking_file, *[task_dir / name for name in REQUIRED_FILES]]
            absent = [str(path) for path in files if not path.exists()]
            if absent:
                if domain == "beauty" and allow_missing_beauty:
                    status = "missing_allowed"
                else:
                    status = "missing"
                    if not task_dir.exists():
                        missing_dirs.append(str(task_dir))
                    missing_files.extend(absent)
            else:
                status = "ready"
            checks.append({
                "domain": domain,
                "split": split,
                "task_dir": str(task_dir),
                "ranking_file": str(ranking_file),
                "status": status,
                "missing_files": absent,
            })
    estimated_candidate_rows = len(domains) * len(splits) * expected_users * expected_candidates
    return {
        "source_root": str(source_root),
        "domains": domains,
        "splits": splits,
        "expected_users": expected_users,
        "expected_candidates": expected_candidates,
        "estimated_candidate_rows_if_complete": estimated_candidate_rows,
        "checks": checks,
        "missing_task_dirs": sorted(set(missing_dirs)),
        "missing_required": sorted(set(missing_files)),
        "next_command": (
            "python scripts/plan_four_domain_server_runs.py --source-root "
            f"{source_root} --output-root data/processed/week8_same_candidate "
            "--domains "
            + " ".join(domains)
            + " --splits "
            + " ".join(splits)
            + " --include-ours-adapter-prep"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
