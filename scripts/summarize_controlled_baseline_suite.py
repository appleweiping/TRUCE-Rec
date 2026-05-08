#!/usr/bin/env python3
"""Summarize controlled-baseline training, scoring, import, and metric status."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NAMES = [
    "tallrec_qwen3_lora_amazon_beauty",
    "openp5_style_qwen3_lora_amazon_beauty",
    "dealrec_qwen3_lora_amazon_beauty",
    "lc_rec_qwen3_lora_amazon_beauty",
    "llara_qwen3_lora_amazon_beauty",
    "llmesr_qwen3_lora_amazon_beauty",
]

CONFIG_BY_NAME = {
    "tallrec_qwen3_lora_amazon_beauty": ROOT / "configs/server/controlled_baselines/tallrec_qwen3_lora_amazon_beauty.yaml",
    "openp5_style_qwen3_lora_amazon_beauty": ROOT / "configs/server/controlled_baselines/openp5_style_qwen3_lora_amazon_beauty.yaml",
    "dealrec_qwen3_lora_amazon_beauty": ROOT / "configs/server/controlled_baselines/dealrec_qwen3_lora_amazon_beauty.yaml",
    "lc_rec_qwen3_lora_amazon_beauty": ROOT / "configs/server/controlled_baselines/lc_rec_qwen3_lora_amazon_beauty.yaml",
    "llara_qwen3_lora_amazon_beauty": ROOT / "configs/server/controlled_baselines/llara_qwen3_lora_amazon_beauty.yaml",
    "llmesr_qwen3_lora_amazon_beauty": ROOT / "configs/server/controlled_baselines/llmesr_qwen3_lora_amazon_beauty.yaml",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "summary" / "controlled_baselines")
    args = parser.parse_args()
    rows = [summarize_name(name) for name in args.names]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "controlled_baseline_status.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with (args.output_dir / "controlled_baseline_status.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "name",
            "project",
            "implementation_fidelity",
            "official_native_controlled",
            "official_fidelity_audit_required",
            "base_model_policy",
            "adaptation",
            "lora_required",
            "adapter_training_policy",
            "hyperparameter_policy",
            "baseline_hparam_tuning_allowed",
            "official_repo",
            "official_algorithm_reused",
            "official_hyperparameters_reused",
            "adapter_dir",
            "training_status",
            "score_rows",
            "candidate_score_lines",
            "adapter_exists",
            "run_exists",
            "prediction_lines",
            "metric_count",
            "recall@10",
            "ndcg@10",
            "mrr@10",
            "next_action",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def summarize_name(name: str) -> dict[str, Any]:
    base = ROOT / "outputs" / "server_training" / "controlled_baselines" / name
    run = ROOT / "outputs" / "runs" / f"{name}_seed13"
    manifest = _read_json(base / "controlled_baseline_manifest.json")
    if not manifest:
        manifest = _read_yaml(CONFIG_BY_NAME.get(name, Path("")))
    summary = _read_json(base / "training_scoring_summary.json")
    metrics = _read_json(run / "metrics.json")
    score_path = base / "candidate_scores.csv"
    prediction_path = run / "predictions.jsonl"
    metric_agg = metrics.get("aggregate", {}) if isinstance(metrics, dict) else {}
    provenance = manifest.get("provenance", {}) if isinstance(manifest.get("provenance"), dict) else {}
    row = {
        "name": name,
        "project": manifest.get("project", ""),
        "implementation_fidelity": _fidelity(manifest, summary),
        "official_native_controlled": bool(
            manifest.get("official_native_controlled", summary.get("official_native_controlled", False))
        ),
        "official_fidelity_audit_required": bool(
            manifest.get("official_fidelity_audit_required", summary.get("official_fidelity_audit_required", True))
        ),
        "base_model_policy": manifest.get("base_model_policy", summary.get("base_model_policy", "shared_qwen3_8b_base_model")),
        "adaptation": manifest.get("adaptation", provenance.get("adaptation", "lora")),
        "lora_required": manifest.get("lora_required", provenance.get("lora_required", True)),
        "adapter_training_policy": manifest.get(
            "adapter_training_policy",
            summary.get("adapter_training_policy", "baseline_official_lora_with_project_modules"),
        ),
        "hyperparameter_policy": manifest.get(
            "hyperparameter_policy",
            provenance.get("hyperparameter_policy", "official_default_or_reported_optimal_for_baselines"),
        ),
        "baseline_hparam_tuning_allowed": manifest.get(
            "baseline_hparam_tuning_allowed",
            provenance.get("baseline_hparam_tuning_allowed", False),
        ),
        "official_repo": provenance.get("official_repo", manifest.get("official_repo", "")),
        "official_algorithm_reused": provenance.get("official_algorithm_reused", ""),
        "official_hyperparameters_reused": provenance.get("official_hyperparameters_reused", ""),
        "adapter_dir": summary.get("adapter_dir", str(base / "adapter")),
        "training_status": summary.get("status", "missing"),
        "score_rows": summary.get("score_rows", 0),
        "candidate_score_lines": _line_count(score_path),
        "adapter_exists": (base / "adapter").exists(),
        "run_exists": run.exists(),
        "prediction_lines": _line_count(prediction_path),
        "metric_count": metrics.get("count", 0) if isinstance(metrics, dict) else 0,
        "recall@10": metric_agg.get("recall@10", ""),
        "ndcg@10": metric_agg.get("ndcg@10", ""),
        "mrr@10": metric_agg.get("mrr@10", ""),
        "next_action": _next_action(name, summary=summary, metrics=metrics),
    }
    return row


def _next_action(name: str, *, summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    if not summary:
        return "run_training_scoring"
    if name.startswith("openp5_style") and int(summary.get("score_rows") or 0) <= 2:
        return "optimize_scoring_before_full_run"
    if not metrics:
        return "import_and_evaluate"
    return "complete_or_ready_for_analysis"


def _fidelity(manifest: dict[str, Any], summary: dict[str, Any]) -> str:
    value = manifest.get("implementation_fidelity") or summary.get("implementation_fidelity")
    return str(value or "controlled_adapter_pilot")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


if __name__ == "__main__":
    raise SystemExit(main())
