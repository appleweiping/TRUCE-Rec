"""Qwen3 server observation planning and guarded execution.

The default path is plan-only and never loads a model. The guarded execution
path is intended for a server after the user approves the run and supplies the
hardware/model environment. Outputs intentionally mirror the API observation
layers so downstream parsing, grounding, metrics, and analysis stay shared.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from storyflow.grounding import TitleGrounder
from storyflow.observation import (
    catalog_records,
    compute_observation_metrics,
    load_catalog_rows,
    observation_metrics_markdown,
    read_jsonl,
    utc_now_iso,
    write_jsonl,
)
from storyflow.observation_parsing import parse_observation_response
from storyflow.utils.config import load_simple_yaml


QWEN_SERVER_OUTPUT_FILES = {
    "request_records": "request_records.jsonl",
    "raw_responses": "raw_responses.jsonl",
    "parsed_predictions": "parsed_predictions.jsonl",
    "failed_cases": "failed_cases.jsonl",
    "grounded_predictions": "grounded_predictions.jsonl",
    "metrics": "metrics.json",
    "report": "report.md",
    "manifest": "manifest.json",
}

REQUIRED_INPUT_FIELDS = {
    "input_id",
    "example_id",
    "user_id",
    "split",
    "prompt",
    "prompt_hash",
    "prompt_template",
    "target_item_id",
    "target_title",
    "target_popularity",
    "target_popularity_bucket",
    "source",
}


def _repo_relative(path: str | Path, *, root: str | Path = ".") -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def _nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def load_qwen_server_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a Qwen3 server observation config."""

    config = load_simple_yaml(config_path)
    if str(config.get("backend")) != "qwen_server":
        raise ValueError("Qwen server config must set backend: qwen_server")
    if not str(config.get("model_name") or "").strip():
        raise ValueError("Qwen server config must set model_name")
    if not str(config.get("model_alias") or "").strip():
        raise ValueError("Qwen server config must set model_alias")
    output_contract = _nested(config, "output_contract", default={})
    for key, filename in QWEN_SERVER_OUTPUT_FILES.items():
        if str(output_contract.get(key) or filename) != filename:
            raise ValueError(f"output_contract.{key} must be {filename}")
    guards = _nested(config, "guards", default={})
    if guards.get("server_execution_required") is not True:
        raise ValueError("guards.server_execution_required must be true")
    return config


def _generation_config(config: dict[str, Any]) -> dict[str, Any]:
    generation = dict(config.get("generation") or {})
    return {
        "max_new_tokens": int(generation.get("max_new_tokens") or 256),
        "temperature": float(generation.get("temperature") or 0.0),
        "top_p": float(generation.get("top_p") or 1.0),
        "do_sample": bool(generation.get("do_sample", False)),
        "repetition_penalty": float(generation.get("repetition_penalty") or 1.0),
    }


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config.get("model") or {})
    return {
        "source": str(model.get("source") or config.get("model_name")),
        "revision": str(model.get("revision") or "main"),
        "torch_dtype": str(model.get("torch_dtype") or "bfloat16"),
        "device_map": str(model.get("device_map") or "auto"),
        "trust_remote_code": bool(model.get("trust_remote_code", True)),
        "cache_dir": model.get("cache_dir"),
        "local_files_only": bool(model.get("local_files_only", False)),
    }


def _check_input_records(inputs: list[dict[str, Any]]) -> None:
    if not inputs:
        raise ValueError("input_jsonl contains no records")
    for index, row in enumerate(inputs):
        missing = sorted(REQUIRED_INPUT_FIELDS - set(row))
        if missing:
            raise ValueError(f"input row {index} is missing fields: {missing}")
        source = row.get("source")
        if not isinstance(source, dict) or not source.get("catalog_csv"):
            raise ValueError(f"input row {index} is missing source.catalog_csv")


def default_qwen_server_output_dir(
    *,
    input_jsonl: str | Path,
    model_alias: str = "qwen3_8b",
    root: str | Path = ".",
) -> Path:
    input_path = Path(input_jsonl)
    parts = input_path.parts
    dataset = parts[-3] if len(parts) >= 3 else "dataset"
    processed_suffix = parts[-2] if len(parts) >= 2 else "processed"
    return (
        Path(root)
        / "outputs"
        / "server_observations"
        / model_alias
        / dataset
        / processed_suffix
        / input_path.stem
    )


def _expected_paths(output_dir: Path) -> dict[str, str]:
    return {
        key: str(output_dir / filename)
        for key, filename in QWEN_SERVER_OUTPUT_FILES.items()
    }


def _request_records(
    inputs: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any],
    run_label: str | None,
    run_stage: str,
    execution_mode: str,
) -> list[dict[str, Any]]:
    model_alias = str(config["model_alias"])
    model_name = str(config["model_name"])
    generation_config = _generation_config(config)
    records: list[dict[str, Any]] = []
    for row in inputs:
        records.append(
            {
                "request_id": f"{model_alias}:{row['input_id']}",
                "input_id": str(row["input_id"]),
                "provider": "server_local_model",
                "model": model_name,
                "model_alias": model_alias,
                "prompt_template": str(row["prompt_template"]),
                "prompt_hash": str(row["prompt_hash"]),
                "prompt": str(row["prompt"]),
                "generation_config": generation_config,
                "metadata": {
                    "example_id": row.get("example_id"),
                    "dataset": row.get("dataset"),
                    "processed_suffix": row.get("processed_suffix"),
                    "split": row.get("split"),
                    "run_label": run_label,
                    "run_stage": run_stage,
                    "execution_mode": execution_mode,
                    "server_execution_required": True,
                },
            }
        )
    return records


def _output_contract(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "schema_family": "api_observation_compatible",
        "model_alias": str(config["model_alias"]),
        "paths": _expected_paths(output_dir),
        "required_layers": [
            "request_records",
            "raw_responses",
            "parsed_predictions",
            "failed_cases",
            "grounded_predictions",
            "metrics",
            "report",
            "manifest",
        ],
        "grounding_required_before_correctness": True,
        "raw_outputs_gitignored": True,
        "is_experiment_result": False,
        "note": (
            "Contract only until --execute-server is run on approved server "
            "hardware. Grounded outputs must be analyzed with the same "
            "observation tools as API outputs."
        ),
    }


def _command_plan(
    *,
    config_path: Path,
    input_jsonl: Path,
    output_dir: Path,
    max_examples: int | None,
    run_label: str | None,
    run_stage: str,
) -> str:
    parts = [
        "python scripts/server/run_qwen3_observation.py",
        f"--config {config_path.as_posix()}",
        f"--input-jsonl {input_jsonl.as_posix()}",
        f"--output-dir {output_dir.as_posix()}",
        "--execute-server",
        f"--run-stage {run_stage}",
    ]
    if max_examples is not None:
        parts.append(f"--max-examples {max_examples}")
    if run_label:
        parts.append(f"--run-label {run_label}")
    lines = [
        "# Qwen3-8B Server Observation Command Plan",
        "",
        "This file is a command plan. It was not executed by Codex.",
        "Run only on approved server hardware after recording the git commit,",
        "environment, model source, dataset paths, and output policy.",
        "",
        "```powershell",
        " ".join(parts),
        "```",
        "",
        "Expected outputs use the API observation-compatible schema and remain",
        "under ignored outputs/runs paths.",
    ]
    return "\n".join(lines) + "\n"


def build_qwen_server_observation_plan(
    *,
    config_path: str | Path,
    input_jsonl: str | Path,
    output_dir: str | Path | None = None,
    max_examples: int | None = None,
    run_label: str | None = None,
    run_stage: str = "planned",
    root: str | Path = ".",
) -> dict[str, Any]:
    """Write a plan-only server observation manifest and request records."""

    resolved_config = _repo_relative(config_path, root=root)
    resolved_input = _repo_relative(input_jsonl, root=root)
    config = load_qwen_server_config(resolved_config)
    output_path = (
        _repo_relative(output_dir, root=root)
        if output_dir
        else default_qwen_server_output_dir(
            input_jsonl=resolved_input,
            model_alias=str(config["model_alias"]),
            root=root,
        )
    )
    output_path.mkdir(parents=True, exist_ok=True)
    inputs = read_jsonl(resolved_input)
    if max_examples is not None:
        inputs = inputs[:max_examples]
    _check_input_records(inputs)
    records = _request_records(
        inputs,
        config=config,
        run_label=run_label,
        run_stage=run_stage,
        execution_mode="server_plan_only",
    )
    paths = _expected_paths(output_path)
    write_jsonl(paths["request_records"], records)
    contract = _output_contract(config, output_path)
    contract_path = output_path / "expected_output_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    command_plan_path = output_path / "server_command_plan.md"
    command_plan_path.write_text(
        _command_plan(
            config_path=resolved_config,
            input_jsonl=resolved_input,
            output_dir=output_path,
            max_examples=max_examples,
            run_label=run_label,
            run_stage=run_stage,
        ),
        encoding="utf-8",
    )
    manifest = {
        "created_at_utc": utc_now_iso(),
        "backend": "qwen_server",
        "provider": "server_local_model",
        "model": str(config["model_name"]),
        "model_alias": str(config["model_alias"]),
        "config_path": str(resolved_config),
        "input_jsonl": str(resolved_input),
        "output_dir": str(output_path),
        "requested_input_count": len(inputs),
        "max_examples": max_examples,
        "run_label": run_label,
        "run_stage": run_stage,
        "execution_mode": "server_plan_only",
        "request_records": paths["request_records"],
        "raw_responses": paths["raw_responses"],
        "parsed_predictions": paths["parsed_predictions"],
        "failed_cases": paths["failed_cases"],
        "grounded_predictions": paths["grounded_predictions"],
        "metrics": paths["metrics"],
        "report": paths["report"],
        "manifest": paths["manifest"],
        "expected_output_contract": str(contract_path),
        "server_command_plan": str(command_plan_path),
        "output_schema_matches_api_observation": True,
        "grounding_required_before_correctness": True,
        "api_called": False,
        "server_executed": False,
        "model_inference_run": False,
        "model_training": False,
        "is_experiment_result": False,
        "note": (
            "Plan-only Qwen3 server observation scaffold. No model inference, "
            "API call, training, or paper result was produced."
        ),
    }
    Path(paths["manifest"]).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _completed_input_ids(*paths: Path) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if "input_id" in row:
                completed.add(str(row["input_id"]))
    return completed


def _model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model = _model_config(config)
    kwargs: dict[str, Any] = {
        "revision": model["revision"],
        "device_map": model["device_map"],
        "trust_remote_code": model["trust_remote_code"],
    }
    if model["cache_dir"]:
        kwargs["cache_dir"] = str(model["cache_dir"])
    if model.get("local_files_only"):
        kwargs["local_files_only"] = True
    dtype_name = str(model["torch_dtype"]).lower()
    try:
        import torch

        if dtype_name in {"bf16", "bfloat16"}:
            kwargs["torch_dtype"] = torch.bfloat16
        elif dtype_name in {"fp16", "float16"}:
            kwargs["torch_dtype"] = torch.float16
        elif dtype_name in {"fp32", "float32"}:
            kwargs["torch_dtype"] = torch.float32
    except ImportError:
        pass
    return kwargs


def _tokenizer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model = _model_config(config)
    kwargs: dict[str, Any] = {
        "revision": model["revision"],
        "trust_remote_code": model["trust_remote_code"],
    }
    if model["cache_dir"]:
        kwargs["cache_dir"] = str(model["cache_dir"])
    if model.get("local_files_only"):
        kwargs["local_files_only"] = True
    return kwargs


def _format_model_prompt(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


def _generate_text(tokenizer: Any, model: Any, prompt: str, generation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("server execution requires torch") from exc

    formatted = _format_model_prompt(tokenizer, prompt)
    encoded = tokenizer(formatted, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    generate_kwargs = {
        "max_new_tokens": generation["max_new_tokens"],
        "do_sample": generation["do_sample"],
        "repetition_penalty": generation["repetition_penalty"],
    }
    if generation["do_sample"]:
        generate_kwargs["temperature"] = generation["temperature"]
        generate_kwargs["top_p"] = generation["top_p"]
    with torch.no_grad():
        output_ids = model.generate(**encoded, **generate_kwargs)
    prompt_len = int(encoded["input_ids"].shape[-1])
    generated_ids = output_ids[0][prompt_len:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    usage = {
        "prompt_tokens": prompt_len,
        "completion_tokens": int(generated_ids.shape[-1]),
        "total_tokens": int(output_ids[0].shape[-1]),
    }
    return raw_text, usage


def _candidate_dicts(candidates: Iterable[Any]) -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in candidates]


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def run_qwen_server_observation(
    *,
    config_path: str | Path,
    input_jsonl: str | Path,
    output_dir: str | Path | None = None,
    max_examples: int | None = None,
    execute_server: bool = False,
    resume: bool = True,
    run_label: str | None = None,
    run_stage: str = "server",
    root: str | Path = ".",
) -> dict[str, Any]:
    """Run plan-only by default, or guarded Qwen3 inference on a server."""

    if not execute_server:
        return build_qwen_server_observation_plan(
            config_path=config_path,
            input_jsonl=input_jsonl,
            output_dir=output_dir,
            max_examples=max_examples,
            run_label=run_label,
            run_stage=run_stage,
            root=root,
        )

    resolved_config = _repo_relative(config_path, root=root)
    resolved_input = _repo_relative(input_jsonl, root=root)
    config = load_qwen_server_config(resolved_config)
    output_path = (
        _repo_relative(output_dir, root=root)
        if output_dir
        else default_qwen_server_output_dir(
            input_jsonl=resolved_input,
            model_alias=str(config["model_alias"]),
            root=root,
        )
    )
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: Path(path) for key, path in _expected_paths(output_path).items()}
    inputs = read_jsonl(resolved_input)
    if max_examples is not None:
        inputs = inputs[:max_examples]
    _check_input_records(inputs)

    if not resume:
        for key in (
            "request_records",
            "raw_responses",
            "parsed_predictions",
            "failed_cases",
            "grounded_predictions",
        ):
            paths[key].write_text("", encoding="utf-8")

    completed = _completed_input_ids(
        paths["grounded_predictions"],
        paths["failed_cases"],
    ) if resume else set()
    pending_inputs = [row for row in inputs if str(row["input_id"]) not in completed]
    request_records = _request_records(
        pending_inputs,
        config=config,
        run_label=run_label,
        run_stage=run_stage,
        execution_mode="execute_server",
    )
    if request_records:
        write_jsonl(paths["request_records"], request_records, append=resume and paths["request_records"].exists())

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("server execution requires transformers") from exc

    model_config = _model_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["source"],
        **_tokenizer_kwargs(config),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["source"],
        **_model_kwargs(config),
    )
    if hasattr(model, "eval"):
        model.eval()

    catalog_csv = _repo_relative(inputs[0]["source"]["catalog_csv"], root=root)
    catalog_rows = load_catalog_rows(catalog_csv)
    grounder = TitleGrounder(catalog_records(catalog_rows))
    generation = _generation_config(config)
    newly_processed = 0
    failed_count = 0
    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    grounded_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    request_by_input = {row["input_id"]: row for row in request_records}

    for input_record in pending_inputs:
        input_id = str(input_record["input_id"])
        request = request_by_input[input_id]
        request_id = str(request["request_id"])
        try:
            raw_text, usage = _generate_text(
                tokenizer,
                model,
                str(input_record["prompt"]),
                generation,
            )
            raw_rows.append(
                {
                    "request_id": request_id,
                    "input_id": input_id,
                    "provider": "server_local_model",
                    "model": str(config["model_name"]),
                    "model_alias": str(config["model_alias"]),
                    "raw_text": raw_text,
                    "status": "ok",
                    "cache_hit": False,
                    "server_executed": True,
                    "dry_run": False,
                    "usage": usage,
                    "error": None,
                    "created_at_utc": utc_now_iso(),
                }
            )
        except Exception as exc:
            failed_count += 1
            failed_rows.append(
                {
                    "request_id": request_id,
                    "input_id": input_id,
                    "provider": "server_local_model",
                    "model": str(config["model_name"]),
                    "error": str(exc),
                    "failure_stage": "generation",
                    "server_executed": True,
                    "created_at_utc": utc_now_iso(),
                }
            )
            continue

        parsed = parse_observation_response(raw_text)
        parsed_row = {
            "input_id": input_id,
            "request_id": request_id,
            "provider": "server_local_model",
            "model": str(config["model_name"]),
            "model_alias": str(config["model_alias"]),
            "parse": asdict(parsed),
            "server_executed": True,
            "created_at_utc": utc_now_iso(),
        }
        parsed_rows.append(parsed_row)
        if not parsed.success:
            failed_count += 1
            failed_rows.append(
                {
                    **parsed_row,
                    "error": parsed.error,
                    "failure_stage": "parse",
                }
            )
            continue

        grounded = grounder.ground(
            parsed.generated_title or "",
            prediction_id=request_id,
        )
        correctness = int(
            grounded.is_grounded
            and grounded.item_id == input_record["target_item_id"]
        )
        newly_processed += 1
        grounded_rows.append(
            {
                "input_id": input_id,
                "request_id": request_id,
                "example_id": input_record["example_id"],
                "user_id": input_record["user_id"],
                "split": input_record["split"],
                "provider": "server_local_model",
                "model": str(config["model_name"]),
                "model_alias": str(config["model_alias"]),
                "generated_title": parsed.generated_title,
                "confidence": parsed.confidence,
                "is_likely_correct": parsed.is_likely_correct,
                "parse_strategy": parsed.parse_strategy,
                "target_item_id": input_record["target_item_id"],
                "target_title": input_record["target_title"],
                "target_popularity": input_record["target_popularity"],
                "target_popularity_bucket": input_record["target_popularity_bucket"],
                "target_in_history": bool(input_record.get("target_in_history", False)),
                "target_history_occurrence_count": int(
                    input_record.get("target_history_occurrence_count") or 0
                ),
                "target_same_timestamp_as_history": bool(
                    input_record.get("target_same_timestamp_as_history", False)
                ),
                "history_duplicate_item_count": int(
                    input_record.get("history_duplicate_item_count") or 0
                ),
                "history_unique_item_count": int(
                    input_record.get("history_unique_item_count") or 0
                ),
                "grounded_item_id": grounded.item_id,
                "grounding_status": grounded.status.value,
                "grounding_score": grounded.score,
                "grounding_ambiguity": grounded.ambiguity,
                "grounding_second_score": grounded.second_score,
                "grounding_candidates": _candidate_dicts(grounded.candidates),
                "correctness": correctness,
                "usage": usage,
                "api_called": False,
                "server_executed": True,
                "is_experiment_result": False,
            }
        )

    append_outputs = resume
    if raw_rows:
        write_jsonl(paths["raw_responses"], raw_rows, append=append_outputs and paths["raw_responses"].exists())
    if parsed_rows:
        write_jsonl(paths["parsed_predictions"], parsed_rows, append=append_outputs and paths["parsed_predictions"].exists())
    if failed_rows:
        write_jsonl(paths["failed_cases"], failed_rows, append=append_outputs and paths["failed_cases"].exists())
    if grounded_rows:
        write_jsonl(paths["grounded_predictions"], grounded_rows, append=append_outputs and paths["grounded_predictions"].exists())

    input_ids = {str(row["input_id"]) for row in inputs}
    all_grounded = [
        row
        for row in read_jsonl(paths["grounded_predictions"])
        if str(row.get("input_id")) in input_ids
    ] if paths["grounded_predictions"].exists() else []
    metrics: dict[str, Any] | None = None
    if all_grounded:
        metrics = compute_observation_metrics(all_grounded)
        metrics.update(
            {
                "provider": "server_local_model",
                "model": str(config["model_name"]),
                "model_alias": str(config["model_alias"]),
                "api_called": False,
                "server_executed": True,
                "run_stage": run_stage,
                "tail_underconfidence_gap": _finite_or_none(
                    metrics.get("tail_underconfidence_gap")
                ),
                "is_experiment_result": False,
                "note": (
                    "Server-side Qwen observation artifact. Treat as evidence "
                    "only with the corresponding server logs, config snapshot, "
                    "manifest, and user-provided execution artifact."
                ),
            }
        )
        paths["metrics"].write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        paths["report"].write_text(
            observation_metrics_markdown(metrics, title="Qwen3 Server Observation Report"),
            encoding="utf-8",
        )

    manifest = {
        "created_at_utc": utc_now_iso(),
        "backend": "qwen_server",
        "provider": "server_local_model",
        "model": str(config["model_name"]),
        "model_alias": str(config["model_alias"]),
        "config_path": str(resolved_config),
        "input_jsonl": str(resolved_input),
        "output_dir": str(output_path),
        "requested_input_count": len(inputs),
        "newly_processed_count": newly_processed,
        "failed_count": failed_count,
        "total_grounded_count": len(all_grounded),
        "run_label": run_label,
        "run_stage": run_stage,
        "execution_mode": "execute_server",
        "resume": resume,
        "request_records": str(paths["request_records"]),
        "raw_responses": str(paths["raw_responses"]),
        "parsed_predictions": str(paths["parsed_predictions"]),
        "failed_cases": str(paths["failed_cases"]),
        "grounded_predictions": str(paths["grounded_predictions"]),
        "metrics": str(paths["metrics"]) if metrics else None,
        "report": str(paths["report"]) if metrics else None,
        "output_schema_matches_api_observation": True,
        "grounding_required_before_correctness": True,
        "api_called": False,
        "server_executed": True,
        "model_inference_run": True,
        "model_training": False,
        "is_experiment_result": False,
        "note": (
            "Qwen3 server observation execution. Codex must not claim this "
            "was run unless this manifest and server logs are provided."
        ),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
