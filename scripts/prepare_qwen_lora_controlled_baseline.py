#!/usr/bin/env python3
"""Prepare Qwen3-8B LoRA controlled-baseline inputs for project adapters.

This script creates an execution contract for server-side training. It does not
train a model locally and does not produce experiment results.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.config import load_config  # noqa: E402


SUPPORTED_PROJECTS = {"tallrec", "openp5", "dealrec", "lc_rec", "llara", "llmesr"}
DEFAULT_IMPLEMENTATION_FIDELITY = "controlled_adapter_pilot"
FINAL_FIDELITY = "official_native_controlled"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = prepare_controlled_baseline(config=config, config_path=args.config)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def prepare_controlled_baseline(*, config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    project = _normalize_project(config.get("project"))
    if project not in SUPPORTED_PROJECTS:
        raise SystemExit(f"unsupported controlled baseline project: {project}")
    packet_dir = ROOT / str(config.get("packet_dir") or "")
    if not packet_dir.exists():
        raise SystemExit(f"packet_dir not found: {packet_dir}")
    output_dir = ROOT / str(config.get("output_dir") or f"outputs/server_training/{project}_qwen3_lora")
    output_dir.mkdir(parents=True, exist_ok=True)
    if project == "tallrec":
        files = _prepare_tallrec(config=config, packet_dir=packet_dir, output_dir=output_dir)
    elif project == "openp5":
        files = _prepare_openp5(config=config, packet_dir=packet_dir, output_dir=output_dir)
    else:
        files = _prepare_generic_project(config=config, packet_dir=packet_dir, output_dir=output_dir, project=project)
    manifest = _manifest(config=config, config_path=config_path, output_dir=output_dir, files=files)
    (output_dir / "controlled_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_command_plan(output_dir / "server_command_plan.md", manifest=manifest)
    return manifest


def _prepare_tallrec(*, config: dict[str, Any], packet_dir: Path, output_dir: Path) -> dict[str, str]:
    project_dir = packet_dir / "tallrec"
    train_rows = json.loads((project_dir / "train.json").read_text(encoding="utf-8"))
    valid_rows = json.loads((project_dir / "valid.json").read_text(encoding="utf-8"))
    test_rows = json.loads((project_dir / "test.json").read_text(encoding="utf-8"))
    test_row_map = _read_jsonl(project_dir / "test_row_map.jsonl")
    test_meta_by_row_id = {str(row.get("row_id") or ""): row for row in test_row_map}
    max_train = _optional_int(config, "max_train_examples")
    max_valid = _optional_int(config, "max_valid_examples")
    train_sft = output_dir / "train_sft.jsonl"
    valid_sft = output_dir / "valid_sft.jsonl"
    _write_jsonl(train_sft, [_tallrec_sft_row(row) for row in train_rows[:max_train]])
    _write_jsonl(valid_sft, [_tallrec_sft_row(row) for row in valid_rows[:max_valid]])
    score_plan = output_dir / "test_score_plan.jsonl"
    _write_jsonl(score_plan, [_tallrec_score_row(row, test_meta_by_row_id=test_meta_by_row_id) for row in test_rows])
    return {
        "train_sft": str(train_sft),
        "valid_sft": str(valid_sft),
        "test_score_plan": str(score_plan),
        "candidate_scores_template": str(project_dir / "candidate_scores_template.csv"),
        "test_row_map": str(project_dir / "test_row_map.jsonl"),
    }


def _prepare_openp5(*, config: dict[str, Any], packet_dir: Path, output_dir: Path) -> dict[str, str]:
    project_dir = packet_dir / "openp5"
    item_map = _read_item_map(project_dir / "item_id_mapping.csv")
    max_train = _optional_int(config, "max_train_examples")
    max_valid = _optional_int(config, "max_valid_examples")
    train_tasks = _read_jsonl(project_dir / "train_sequential_tasks.jsonl")
    valid_tasks = _read_jsonl(project_dir / "valid_sequential_tasks.jsonl")
    test_tasks = _read_jsonl(project_dir / "test_sequential_tasks.jsonl")
    train_sft = output_dir / "train_sft.jsonl"
    valid_sft = output_dir / "valid_sft.jsonl"
    test_score_plan = output_dir / "test_score_plan.jsonl"
    _write_jsonl(train_sft, [_openp5_sft_row(row, item_map=item_map) for row in train_tasks[:max_train]])
    _write_jsonl(valid_sft, [_openp5_sft_row(row, item_map=item_map) for row in valid_tasks[:max_valid]])
    _write_jsonl(test_score_plan, [_openp5_score_row(row, item_map=item_map) for row in test_tasks])
    return {
        "train_sft": str(train_sft),
        "valid_sft": str(valid_sft),
        "test_score_plan": str(test_score_plan),
        "candidate_scores_template": str(project_dir / "candidate_scores_template.csv"),
        "item_id_mapping": str(project_dir / "item_id_mapping.csv"),
    }


def _prepare_generic_project(
    *,
    config: dict[str, Any],
    packet_dir: Path,
    output_dir: Path,
    project: str,
) -> dict[str, str]:
    project_dir = packet_dir / project
    max_train = _optional_int(config, "max_train_examples")
    max_valid = _optional_int(config, "max_valid_examples")
    negatives_per_train = int(config.get("negatives_per_train_example") or 15)
    rng = random.Random(int(config.get("seed") or 13))
    train_tasks = _read_jsonl(project_dir / "train_project_tasks.jsonl")
    valid_tasks = _read_jsonl(project_dir / "valid_project_tasks.jsonl")
    test_tasks = _read_jsonl(project_dir / "test_project_tasks.jsonl")
    train_sft = output_dir / "train_sft.jsonl"
    valid_sft = output_dir / "valid_sft.jsonl"
    test_score_plan = output_dir / "test_score_plan.jsonl"
    train_rows = [
        sft_row
        for row in train_tasks
        for sft_row in _generic_pairwise_sft_rows(
            row,
            project=project,
            negatives_per_train=negatives_per_train,
            rng=rng,
        )
    ]
    valid_rows = [
        sft_row
        for row in valid_tasks
        for sft_row in _generic_pairwise_sft_rows(
            row,
            project=project,
            negatives_per_train=negatives_per_train,
            rng=rng,
        )
    ]
    score_rows = [score_row for row in test_tasks for score_row in _generic_pairwise_score_rows(row, project=project)]
    _write_jsonl(train_sft, train_rows[:max_train])
    _write_jsonl(valid_sft, valid_rows[:max_valid])
    _write_jsonl(test_score_plan, score_rows)
    return {
        "train_sft": str(train_sft),
        "valid_sft": str(valid_sft),
        "test_score_plan": str(test_score_plan),
        "candidate_scores_template": str(project_dir / "candidate_scores_template.csv"),
        "item_id_mapping": str(project_dir / "item_id_mapping.csv"),
    }


def _tallrec_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = f"{row.get('instruction', '')}\n\n{row.get('input', '')}".strip()
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(row.get("output") or "")},
        ],
        "metadata": {
            "source_project": "TALLRec",
            "truce_row_id": str(row.get("truce_row_id") or ""),
        },
    }


def _tallrec_score_row(row: dict[str, Any], *, test_meta_by_row_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prompt = f"{row.get('instruction', '')}\n\n{row.get('input', '')}".strip()
    row_id = str(row.get("truce_row_id") or "")
    meta = test_meta_by_row_id.get(row_id, {})
    return {
        "example_id": str(meta.get("example_id") or ""),
        "user_id": str(meta.get("user_id") or ""),
        "prompt": prompt,
        "candidate_item_ids": [str(meta.get("item_id") or "")],
        "candidate_outputs": ["Yes."],
        "metadata": {
            "source_project": "TALLRec",
            "truce_row_id": row_id,
        },
        "scoring_contract": "Compute causal-LM log-likelihood of Yes. for this pairwise prompt, then write one candidate score.",
    }


def _openp5_sft_row(row: dict[str, Any], *, item_map: dict[str, str]) -> dict[str, Any]:
    target = str(row.get("target_item_id") or "")
    prompt = _openp5_prompt(row, item_map=item_map)
    response = item_map.get(target, target)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "source_project": "OpenP5",
            "example_id": str(row.get("example_id") or ""),
            "target_item_id": target,
        },
    }


def _openp5_score_row(row: dict[str, Any], *, item_map: dict[str, str]) -> dict[str, Any]:
    candidates = [str(item) for item in row.get("candidate_item_ids") or []]
    return {
        "example_id": str(row.get("example_id") or ""),
        "user_id": str(row.get("user_id") or ""),
        "prompt": _openp5_prompt(row, item_map=item_map),
        "candidate_item_ids": candidates,
        "candidate_outputs": [item_map.get(item, item) for item in candidates],
        "scoring_contract": "Compute causal-LM log-likelihood for each candidate_output given prompt, then write candidate_scores.csv.",
    }


def _openp5_prompt(row: dict[str, Any], *, item_map: dict[str, str]) -> str:
    history_ids = [str(item) for item in row.get("history_item_ids") or []]
    candidate_ids = [str(item) for item in row.get("candidate_item_ids") or []]
    history_tokens = [item_map.get(item, item) for item in history_ids]
    candidate_tokens = [item_map.get(item, item) for item in candidate_ids]
    return (
        "Sequential recommendation task.\n"
        f"User history item tokens: {' '.join(history_tokens)}\n"
        f"Candidate item tokens: {' '.join(candidate_tokens)}\n"
        "Return exactly one candidate item token as the next recommendation."
    )


def _generic_pairwise_sft_rows(
    row: dict[str, Any],
    *,
    project: str,
    negatives_per_train: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    target = str(row.get("target_item_id") or "")
    negatives = [str(item) for item in row.get("candidate_item_ids") or [] if str(item) != target]
    negatives = rng.sample(negatives, k=min(negatives_per_train, len(negatives)))
    candidate_ids = ([target] if target else []) + negatives
    rng.shuffle(candidate_ids)
    return [
        {
            "messages": [
                {"role": "user", "content": _generic_pairwise_prompt(row, project=project, candidate_item_id=item_id)},
                {"role": "assistant", "content": "Yes." if item_id == target else "No."},
            ],
            "metadata": {
                "source_project": _project_display(project),
                "example_id": str(row.get("example_id") or ""),
                "target_item_id": target,
                "candidate_item_id": item_id,
            },
        }
        for item_id in candidate_ids
    ]


def _generic_pairwise_score_rows(row: dict[str, Any], *, project: str) -> list[dict[str, Any]]:
    return [
        {
            "example_id": str(row.get("example_id") or ""),
            "user_id": str(row.get("user_id") or ""),
            "prompt": _generic_pairwise_prompt(row, project=project, candidate_item_id=str(item_id)),
            "candidate_item_ids": [str(item_id)],
            "candidate_outputs": ["Yes."],
            "scoring_contract": "Compute causal-LM log-likelihood of Yes. for this candidate prompt, then write one candidate score.",
        }
        for item_id in row.get("candidate_item_ids") or []
    ]


def _generic_pairwise_prompt(row: dict[str, Any], *, project: str, candidate_item_id: str) -> str:
    candidate_text = candidate_item_id
    for candidate in row.get("candidate_texts") or []:
        if isinstance(candidate, dict) and str(candidate.get("item_id") or "") == candidate_item_id:
            candidate_text = str(candidate.get("text") or candidate_item_id)
            break
    family_hint = {
        "dealrec": "Use a data-efficient recommendation adaptation.",
        "lc_rec": "Use collaborative preference evidence from the user history and item text.",
    }.get(project, "Use the project-style recommendation adaptation.")
    return (
        f"{_project_display(project)} controlled recommendation task.\n"
        f"{family_hint}\n"
        f"User history: {row.get('history_text') or ''}\n"
        f"Candidate item: {candidate_text}\n"
        f"Candidate item id: {candidate_item_id}\n"
        "Answer exactly Yes. or No."
    )


def _manifest(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    files: dict[str, str],
) -> dict[str, Any]:
    project = _normalize_project(config.get("project"))
    implementation_fidelity = str(config.get("implementation_fidelity") or DEFAULT_IMPLEMENTATION_FIDELITY)
    provenance = _provenance(config, implementation_fidelity=implementation_fidelity)
    _validate_fidelity(config=config, implementation_fidelity=implementation_fidelity, provenance=provenance)
    official_native = implementation_fidelity == FINAL_FIDELITY
    return {
        "schema_version": "truce_qwen_lora_controlled_baseline_v1",
        "project": project,
        "controlled_baseline_name": str(config.get("controlled_baseline_name") or f"{project}_qwen3_lora"),
        "definition": "Controlled-backbone project adaptation. Paper-grade main baselines require official-native fidelity unless explicitly labeled otherwise.",
        "implementation_fidelity": implementation_fidelity,
        "official_native_controlled": official_native,
        "official_fidelity_audit_required": not official_native,
        "base_model_policy": str(config.get("base_model_policy") or "shared_qwen3_8b_base_model"),
        "adaptation": str(config.get("adaptation") or "lora"),
        "lora_required": bool(config.get("lora_required", True)),
        "adapter_training_policy": str(config.get("adapter_training_policy") or "baseline_official_lora_with_project_modules"),
        "hyperparameter_policy": str(config.get("hyperparameter_policy") or "official_default_or_reported_optimal_for_baselines"),
        "baseline_hparam_tuning_allowed": bool(config.get("baseline_hparam_tuning_allowed", False)),
        "official_hparam_source": str(config.get("official_hparam_source") or ""),
        "shared_protocol": {
            "dataset_split": True,
            "candidate_rows": True,
            "evaluator": True,
            "score_schema": "candidate_scores_csv_v1",
        },
        "provenance": provenance,
        "base_model": str(config.get("base_model") or "/home/ajifang/models/Qwen/Qwen3-8B"),
        "packet_dir": str(config.get("packet_dir") or ""),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "seed": int(config.get("seed") or 13),
        "lora": _as_dict(config.get("lora")),
        "training": _as_dict(config.get("training")),
        "scoring": _as_dict(config.get("scoring")),
        "files": files,
        "is_experiment_result": False,
        "is_paper_result": False,
        "paper_table_policy": _paper_table_policy(implementation_fidelity),
        "required_return_artifacts": [
            "adapter checkpoint or checkpoint reference",
            "candidate_scores.csv",
            "predictions.jsonl imported by TRUCE",
            "metrics.json and metrics.csv from TRUCE evaluator",
            "environment.json, git_info.json, stdout/stderr logs",
        ],
    }


def _paper_table_policy(implementation_fidelity: str) -> str:
    if implementation_fidelity == FINAL_FIDELITY:
        return (
            "Eligible for controlled main comparison only after real Qwen3-8B LoRA training, scoring, "
            "TRUCE import, evaluator metrics, and recorded official-fidelity audit are complete."
        )
    return (
        "Controlled adapter pilot only: useful for pipeline validation and diagnostics, but not eligible "
        "as a final main-table official baseline until promoted by an official-fidelity audit."
    )


def _provenance(config: dict[str, Any], *, implementation_fidelity: str) -> dict[str, Any]:
    raw = config.get("provenance")
    provenance = raw.copy() if isinstance(raw, dict) else {}
    provenance.setdefault("official_repo", str(config.get("official_repo") or ""))
    provenance.setdefault("official_algorithm_reused", implementation_fidelity == FINAL_FIDELITY)
    provenance.setdefault("official_hyperparameters_reused", implementation_fidelity == FINAL_FIDELITY)
    provenance.setdefault("adapter_scope", "per_baseline_independent")
    provenance.setdefault("adapter_retention_required", True)
    provenance.setdefault("backbone_control", "shared_across_controlled_baselines")
    provenance.setdefault("adaptation", str(config.get("adaptation") or "lora"))
    provenance.setdefault("lora_required", bool(config.get("lora_required", True)))
    provenance.setdefault("baseline_hparam_tuning_allowed", bool(config.get("baseline_hparam_tuning_allowed", False)))
    provenance.setdefault("hyperparameter_policy", str(config.get("hyperparameter_policy") or "official_default_or_reported_optimal_for_baselines"))
    provenance.setdefault("official_hparam_source", str(config.get("official_hparam_source") or ""))
    provenance.setdefault("truce_adapter_scope", [
        "dataset_split_candidate_ingestion",
        "qwen3_8b_base_model_substitution",
        "lora_adaptation_under_official_baseline_algorithm",
        "score_schema_export",
        "truce_import_and_evaluation",
    ])
    provenance.setdefault("score_schema", "candidate_scores_csv_v1")
    provenance.setdefault("claim_label", implementation_fidelity)
    return provenance


def _validate_fidelity(
    *,
    config: dict[str, Any],
    implementation_fidelity: str,
    provenance: dict[str, Any],
) -> None:
    if implementation_fidelity != FINAL_FIDELITY:
        return
    official_repo = str(config.get("official_repo") or provenance.get("official_repo") or "").strip()
    if not official_repo:
        raise SystemExit("official_native_controlled requires official_repo")
    if provenance.get("official_algorithm_reused") is not True:
        raise SystemExit("official_native_controlled requires provenance.official_algorithm_reused=true")
    if provenance.get("official_hyperparameters_reused") is not True:
        raise SystemExit("official_native_controlled requires provenance.official_hyperparameters_reused=true")
    if str(config.get("adaptation") or provenance.get("adaptation") or "lora") != "lora":
        raise SystemExit("official_native_controlled main lane requires adaptation=lora")
    required = ["official_training_entrypoint", "official_scoring_contract"]
    missing = [key for key in required if not str(provenance.get(key) or "").strip()]
    if missing:
        raise SystemExit(f"official_native_controlled requires provenance fields: {', '.join(missing)}")


def _write_command_plan(path: Path, *, manifest: dict[str, Any]) -> None:
    name = manifest["controlled_baseline_name"]
    packet_dir = manifest["packet_dir"]
    output_dir = manifest["output_dir"]
    text = f"""# Server Command Plan: {name}

This is a controlled-backbone Qwen3-8B LoRA baseline plan. It is not complete
until real LoRA training and TRUCE evaluation have run.

## Inputs

- Packet: `{packet_dir}`
- SFT train: `{manifest['files']['train_sft']}`
- SFT valid: `{manifest['files']['valid_sft']}`
- Test score plan: `{manifest['files']['test_score_plan']}`
- Base model: `{manifest['base_model']}`

## Required Output

Write `candidate_scores.csv` with columns:

```text
example_id,user_id,item_id,score
```

Then import/evaluate with:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate
RUN=outputs/runs/{name}_seed{manifest['seed']}
mkdir -p "$RUN/artifacts"
cp {output_dir}/candidate_scores.csv "$RUN/artifacts/candidate_scores.csv"
python scripts/import_external_predictions.py \\
  --scores "$RUN/artifacts/candidate_scores.csv" \\
  --examples {packet_dir}/truce_examples.jsonl \\
  --output "$RUN/predictions.jsonl" \\
  --method {name} \\
  --source-project {manifest['project']} \\
  --model-name Qwen3-8B-LoRA \\
  --seed {manifest['seed']} \\
  --split test
python scripts/evaluate_predictions.py --predictions "$RUN/predictions.jsonl" --output-dir "$RUN"
```
"""
    path.write_text(text, encoding="utf-8")


def _read_item_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row["item_id"]): str(row["openp5_token"]) for row in csv.DictReader(handle)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _optional_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value in (None, "", 0):
        return None
    return int(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_project(value: Any) -> str:
    project = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "lcrec": "lc_rec",
        "lc_rec": "lc_rec",
    }
    return aliases.get(project, project)


def _project_display(project: str) -> str:
    return {
        "tallrec": "TALLRec",
        "openp5": "OpenP5",
        "dealrec": "DEALRec",
        "lc_rec": "LC-Rec",
    }.get(project, project)


if __name__ == "__main__":
    raise SystemExit(main())
