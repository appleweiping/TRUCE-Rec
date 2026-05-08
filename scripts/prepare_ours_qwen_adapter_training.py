#!/usr/bin/env python3
"""Prepare TRUCE-native Ours Qwen3 adapter training/scoring artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.methods.ours_framework import build_score_row, build_training_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, help="Single processed directory containing examples.jsonl.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        help="Root containing valid/ and test/ processed directories; preferred for Week8.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="/home/ajifang/models/Qwen/Qwen3-8B")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--negatives-per-example", type=int, default=15)
    parser.add_argument("--max-history", type=int, default=50)
    args = parser.parse_args()
    if args.processed_dir is None and args.processed_root is None:
        raise SystemExit("--processed-dir or --processed-root is required")
    manifest = prepare(
        processed_dir=args.processed_dir,
        processed_root=args.processed_root,
        output_dir=args.output_dir,
        base_model=args.base_model,
        domain=args.domain,
        seed=args.seed,
        negatives_per_example=args.negatives_per_example,
        max_history=args.max_history,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def prepare(
    *,
    processed_dir: Path | None,
    processed_root: Path | None,
    output_dir: Path,
    base_model: str,
    domain: str,
    seed: int,
    negatives_per_example: int,
    max_history: int,
) -> dict[str, Any]:
    split_dirs = _resolve_split_dirs(processed_dir=processed_dir, processed_root=processed_root)
    examples_by_split = {split: _read_jsonl(path / "examples.jsonl") if path else [] for split, path in split_dirs.items()}
    examples = [row for rows in examples_by_split.values() for row in rows]
    item_lookup = _read_items(_first_existing([path / "items.csv" for path in split_dirs.values() if path is not None]))
    train_pop = _train_popularity(
        _first_existing_dir([path for path in [split_dirs.get("train"), split_dirs.get("valid"), split_dirs.get("test")] if path is not None]),
        examples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = {
        "train": list(examples_by_split.get("train") or []),
        "valid": list(examples_by_split.get("valid") or []),
        "test": list(examples_by_split.get("test") or []),
    }
    if processed_dir is not None:
        by_split = {"train": [], "valid": [], "test": []}
        for ex in examples:
            split = _normalize_split(ex.get("split"))
            if split in by_split:
                by_split[split].append(ex)
    if not by_split["train"] and by_split["valid"]:
        # Week8 task directories expose valid/test events plus train_interactions.
        # Use valid as adapter fitting data only when no explicit train events exist.
        by_split["train"] = list(by_split["valid"])
    train_rows = [
        row
        for ex in by_split["train"]
        for row in build_training_rows(
            ex,
            item_lookup=item_lookup,
            train_popularity=train_pop,
            negatives_per_example=negatives_per_example,
            max_history=max_history,
        )
    ]
    valid_rows = [
        row
        for ex in by_split["valid"]
        for row in build_training_rows(
            ex,
            item_lookup=item_lookup,
            train_popularity=train_pop,
            negatives_per_example=negatives_per_example,
            include_listwise=False,
            max_history=max_history,
        )
    ]
    score_rows = [
        build_score_row(
            ex,
            candidate_item_id=str(item_id),
            item_lookup=item_lookup,
            train_popularity=train_pop,
            max_history=max_history,
        )
        for ex in by_split["test"]
        for item_id in ex.get("candidates", [])
    ]
    train_path = output_dir / "train_sft.jsonl"
    valid_path = output_dir / "valid_sft.jsonl"
    score_path = output_dir / "test_score_plan.jsonl"
    _write_jsonl(train_path, [row_to_json(row) for row in train_rows])
    _write_jsonl(valid_path, [row_to_json(row) for row in valid_rows])
    _write_jsonl(score_path, score_rows)
    manifest = {
        "schema_version": "truce_ours_qwen_adapter_training_v1",
        "project": "ours_truce",
        "controlled_baseline_name": f"ours_truce_qwen_adapter_{domain}",
        "method": "ours_truce_qwen_adapter",
        "definition": "TRUCE-native uncertainty-aware generative/reranking adapter data; not an experiment result.",
        "processed_dir": str(processed_dir or ""),
        "processed_root": str(processed_root or ""),
        "output_dir": str(output_dir),
        "base_model": base_model,
        "base_model_policy": "shared_qwen3_8b_base_model",
        "adapter_training_policy": "ours_uncertainty_aware_pairwise_and_listwise_adapter",
        "objective_family": "truce_uncertainty_structured_sft",
        "uses_grounding_risk_targets": True,
        "uses_popularity_residual_targets": True,
        "uses_echo_risk_targets": True,
        "negative_sampling_policy": "stratified_head_tail_echo_stable_negatives",
        "target_label_policy": "deterministic_train_evidence_targets_no_test_outcome_tuning",
        "feature_sources": [
            "train_interactions_popularity",
            "catalog_title_category_brand",
            "user_history_overlap",
            "fixed_same_candidate_panel",
        ],
        "hyperparameter_policy": "ours_validation_tunable_no_test_tuning",
        "domain": domain,
        "seed": seed,
        "files": {
            "train_sft": str(train_path),
            "valid_sft": str(valid_path),
            "test_score_plan": str(score_path),
            "test_examples": str(_test_examples_path(processed_dir=processed_dir, split_dirs=split_dirs)),
        },
        "training": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 0.0002,
            "warmup_ratio": 0.03,
            "weight_decay": 0.0,
            "bf16": True,
            "gradient_checkpointing": True,
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "scoring": {
            "type": "ours_pairwise_acceptance_likelihood",
            "output": "candidate_scores.csv",
        },
        "counts": {
            "examples": len(examples),
            "train_examples": len(by_split["train"]),
            "valid_examples": len(by_split["valid"]),
            "test_examples": len(by_split["test"]),
            "train_sft_rows": len(train_rows),
            "valid_sft_rows": len(valid_rows),
            "score_rows": len(score_rows),
            "item_count": len(item_lookup),
        },
        "required_next_steps": [
            "train Qwen3 adapter on train_sft.jsonl",
            "score test_score_plan.jsonl into candidate_scores.csv",
            "import candidate_scores.csv with TRUCE import_external_predictions.py",
            "evaluate with TRUCE evaluator",
        ],
        "claim_boundary": (
            "Prepared adapter data is not a result. Ours may tune hyperparameters on the declared validation "
            "protocol, but paper claims require completed candidate scoring, TRUCE import/evaluation, and ablations."
        ),
        "is_experiment_result": False,
        "is_paper_result": False,
    }
    (output_dir / "ours_adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_server_plan(output_dir / "server_command_plan.md", manifest=manifest)
    return manifest


def _resolve_split_dirs(*, processed_dir: Path | None, processed_root: Path | None) -> dict[str, Path | None]:
    if processed_root is not None:
        return {
            "train": processed_root / "train" if (processed_root / "train").exists() else None,
            "valid": processed_root / "valid" if (processed_root / "valid").exists() else None,
            "test": processed_root / "test" if (processed_root / "test").exists() else None,
        }
    assert processed_dir is not None
    split = _infer_single_dir_split(processed_dir)
    return {"train": processed_dir if split == "train" else None, "valid": processed_dir if split == "valid" else None, "test": processed_dir if split == "test" else None}


def _test_examples_path(*, processed_dir: Path | None, split_dirs: dict[str, Path | None]) -> Path:
    test_dir = split_dirs.get("test")
    if test_dir is not None:
        return test_dir / "examples.jsonl"
    if processed_dir is not None:
        return processed_dir / "examples.jsonl"
    raise SystemExit("test examples path not found")


def _infer_single_dir_split(processed_dir: Path) -> str:
    examples_path = processed_dir / "examples.jsonl"
    if examples_path.exists():
        for row in _read_jsonl(examples_path):
            split = _normalize_split(row.get("split"))
            if split in {"train", "valid", "test"}:
                return split
    return "test"


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise SystemExit("required file not found in processed split directories")


def _first_existing_dir(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise SystemExit("processed split directory not found")


def row_to_json(row: Any) -> dict[str, Any]:
    return {"messages": row.messages, "metadata": row.metadata}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_items(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row["item_id"]): dict(row) for row in csv.DictReader(handle)}


def _train_popularity(processed_dir: Path, examples: list[dict[str, Any]]) -> dict[str, int]:
    interactions_path = processed_dir / "interactions.csv"
    counts: Counter[str] = Counter()
    if interactions_path.exists():
        with interactions_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                item = str(row.get("item_id") or "")
                if item:
                    counts[item] += 1
    if not counts:
        for ex in examples:
            if _normalize_split(ex.get("split")) == "train":
                target = str(ex.get("target") or "")
                if target:
                    counts[target] += 1
    return dict(counts)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _normalize_split(value: Any) -> str:
    split = str(value or "").lower()
    if split in {"val", "validation"}:
        return "valid"
    return split


def _write_server_plan(path: Path, *, manifest: dict[str, Any]) -> None:
    out = manifest["output_dir"]
    name = manifest["controlled_baseline_name"]
    test_examples = manifest["files"]["test_examples"]
    text = f"""# Ours Qwen Adapter Server Plan

This plan prepares Ours/TRUCE adapter data only. The objective is
`truce_uncertainty_structured_sft`: pairwise/listwise recommendation
supervision with deterministic grounding, popularity, and history-repetition
targets derived from train/catalog evidence. Training and scoring must run on
the server and then be imported through the TRUCE evaluator.

```bash
cd ~/projects/TRUCE-Rec
source ~/projects/TALLRec/.venv_tallrec/bin/activate

python scripts/run_qwen_lora_controlled_baseline.py \\
  --manifest {out}/ours_adapter_manifest.json \\
  --trust-remote-code
```

Import and evaluate:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

python scripts/import_evaluate_ours_adapter.py \\
  --manifest {out}/ours_adapter_manifest.json \\
  --split test
```

The generic Qwen runner expects the same train/score manifest fields as a
controlled baseline. If the Ours manifest is extended with a specialized
trainer, keep the output score schema unchanged:

```text
example_id,user_id,item_id,score
```
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
