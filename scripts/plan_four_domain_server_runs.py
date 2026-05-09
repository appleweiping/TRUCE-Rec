#!/usr/bin/env python3
"""Print server commands for the four-domain Week8 TRUCE pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DOMAINS = ["beauty", "books", "electronics", "movies"]
DEFAULT_TASK_SLUGS = {
    "beauty": "beauty_supplementary_smallerN_100neg",
    "books": "books_large10000_100neg",
    "electronics": "electronics_large10000_100neg",
    "movies": "movies_large10000_100neg",
}
DEFAULT_SPLITS = ["valid", "test"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks",
        help="Root containing {artifact_slug}_{split}_same_candidate directories.",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/week8_same_candidate",
        help="TRUCE output root for converted processed artifacts.",
    )
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument(
        "--task-slugs",
        nargs="*",
        default=[],
        help="Optional domain=artifact_slug overrides, e.g. beauty=beauty_supplementary_smallerN_100neg.",
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--allow-target-insertion", action="store_true")
    parser.add_argument("--include-ours-adapter-prep", action="store_true")
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()

    commands = build_commands(
        source_root=args.source_root,
        output_root=args.output_root,
        domains=args.domains,
        task_slugs=_parse_task_slugs(args.task_slugs),
        splits=args.splits,
        strict_target_in_candidates=not args.allow_target_insertion,
        include_ours_adapter_prep=args.include_ours_adapter_prep,
    )
    if args.emit_json:
        print(json.dumps(commands, indent=2, sort_keys=True))
    else:
        print("# Run from ~/projects/TRUCE-Rec after activating .venv_truce")
        for command in commands:
            print(command)
    return 0


def build_commands(
    *,
    source_root: str,
    output_root: str,
    domains: list[str],
    splits: list[str],
    task_slugs: dict[str, str] | None = None,
    strict_target_in_candidates: bool = True,
    include_ours_adapter_prep: bool = False,
) -> list[str]:
    task_slugs = {**DEFAULT_TASK_SLUGS, **(task_slugs or {})}
    commands = []
    for domain in domains:
        slug = _task_slug(domain, task_slugs)
        for split in splits:
            task_dir = f"{source_root.rstrip('/')}/{slug}_{split}_same_candidate"
            out_dir = str(Path(output_root) / slug / split).replace("\\", "/")
            commands.append(
                "python scripts/convert_week8_same_candidate_to_truce.py "
                f"--task-dir {task_dir} "
                f"--output-dir {out_dir} "
                f"--domain {domain} "
                f"--split {split}"
                + (" --strict-target-in-candidates" if strict_target_in_candidates else "")
            )
        if include_ours_adapter_prep and "test" in splits:
            processed_root = str(Path(output_root) / slug).replace("\\", "/")
            output_dir = f"outputs/server_training/ours_qwen_adapters/{slug}"
            commands.append(
                "python scripts/prepare_ours_qwen_adapter_training.py "
                f"--processed-root {processed_root} "
                f"--output-dir {output_dir} "
                f"--domain {domain} "
                "--seed 13"
            )
    return commands


def _parse_task_slugs(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--task-slugs entries must be domain=slug, got: {value}")
        domain, slug = value.split("=", 1)
        if not domain or not slug:
            raise SystemExit(f"--task-slugs entries must be domain=slug, got: {value}")
        parsed[domain] = slug
    return parsed


def _task_slug(domain: str, task_slugs: dict[str, str]) -> str:
    return task_slugs.get(domain, f"{domain}_large10000_100neg")


if __name__ == "__main__":
    raise SystemExit(main())
