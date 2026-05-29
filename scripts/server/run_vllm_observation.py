"""Qwen3-8B observation using vLLM for fast batch inference.

Replaces the slow transformers batch_size=1 approach with vLLM offline inference.
Produces the same output schema as the transformers-based observation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="vLLM-based Qwen3 observation")
    parser.add_argument("--config", default="configs/server/qwen3_8b_observation.yaml")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    config = load_simple_yaml(Path(args.config))
    model_source = config.get("model", {}).get("source", "Qwen/Qwen3-8B")
    max_new_tokens = config.get("generation", {}).get("max_new_tokens", 512)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = read_jsonl(Path(args.input_jsonl))
    if args.max_examples:
        inputs = inputs[:args.max_examples]

    # Resume support
    completed_ids = set()
    grounded_path = output_dir / "grounded_predictions.jsonl"
    failed_path = output_dir / "failed_cases.jsonl"
    if args.resume:
        if grounded_path.exists():
            for row in read_jsonl(grounded_path):
                completed_ids.add(str(row.get("input_id", "")))
        if failed_path.exists():
            for row in read_jsonl(failed_path):
                completed_ids.add(str(row.get("input_id", "")))

    pending = [r for r in inputs if str(r["input_id"]) not in completed_ids]
    if not pending:
        print(f"All {len(inputs)} examples already processed. Nothing to do.")
        return

    print(f"Total: {len(inputs)}, Already done: {len(completed_ids)}, Pending: {len(pending)}")

    # Load vLLM
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_source,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        dtype="bfloat16",
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts
    prompts = []
    for record in pending:
        prompt_text = str(record["prompt"])
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(formatted)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    # Run batch inference
    print(f"Running vLLM inference on {len(prompts)} prompts...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time
    print(f"Inference complete: {elapsed:.1f}s ({elapsed/len(prompts):.2f}s/example)")

    # Load catalog for grounding
    catalog_csv_path = Path(pending[0]["source"]["catalog_csv"])
    if not catalog_csv_path.is_absolute():
        catalog_csv_path = ROOT / catalog_csv_path
    catalog_rows = load_catalog_rows(catalog_csv_path)
    grounder = TitleGrounder(catalog_records(catalog_rows))

    # Process outputs
    raw_rows = []
    parsed_rows = []
    grounded_rows = []
    failed_rows = []

    for i, (record, output) in enumerate(zip(pending, outputs)):
        input_id = str(record["input_id"])
        request_id = f"qwen3_8b:{input_id}"
        raw_text = output.outputs[0].text

        raw_rows.append({
            "request_id": request_id,
            "input_id": input_id,
            "provider": "server_vllm",
            "model": model_source,
            "model_alias": "qwen3_8b",
            "raw_text": raw_text,
            "status": "ok",
            "cache_hit": False,
            "server_executed": True,
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
            },
            "created_at_utc": utc_now_iso(),
        })

        parsed = parse_observation_response(raw_text)
        parsed_row = {
            "input_id": input_id,
            "request_id": request_id,
            "provider": "server_vllm",
            "model": model_source,
            "model_alias": "qwen3_8b",
            "parse": {
                "success": parsed.success,
                "generated_title": parsed.generated_title,
                "confidence": parsed.confidence,
                "error": parsed.error,
            },
            "server_executed": True,
            "created_at_utc": utc_now_iso(),
        }
        parsed_rows.append(parsed_row)

        if not parsed.success:
            failed_rows.append({
                **parsed_row,
                "error": parsed.error,
                "failure_stage": "parse",
            })
            continue

        grounded = grounder.ground(
            parsed.generated_title or "",
            prediction_id=request_id,
        )
        is_correct = (
            grounded.is_grounded
            and str(grounded.grounded_item_id) == str(record.get("target_item_id", ""))
        )

        grounded_rows.append({
            "input_id": input_id,
            "request_id": request_id,
            "example_id": str(record.get("example_id", input_id)),
            "user_id": str(record.get("user_id", "")),
            "split": str(record.get("split", "test")),
            "provider": "server_vllm",
            "model": model_source,
            "model_alias": "qwen3_8b",
            "generated_title": parsed.generated_title,
            "confidence": parsed.confidence,
            "grounded_item_id": grounded.grounded_item_id,
            "grounding_score": grounded.score,
            "grounding_second_score": grounded.second_score,
            "grounding_status": grounded.status,
            "grounding_ambiguity": grounded.ambiguity,
            "grounding_candidates": grounded.candidates[:3] if grounded.candidates else [],
            "correctness": "correct" if is_correct else "incorrect",
            "is_likely_correct": is_correct,
            "target_item_id": str(record.get("target_item_id", "")),
            "target_title": str(record.get("target_title", "")),
            "target_popularity": record.get("target_popularity"),
            "target_popularity_bucket": str(record.get("target_popularity_bucket", "")),
            "target_in_history": record.get("target_in_history", False),
            "target_history_occurrence_count": record.get("target_history_occurrence_count", 0),
            "target_same_timestamp_as_history": record.get("target_same_timestamp_as_history", False),
            "history_unique_item_count": record.get("history_unique_item_count", 0),
            "history_duplicate_item_count": record.get("history_duplicate_item_count", 0),
            "parse_strategy": "forced_json",
            "server_executed": True,
            "api_called": False,
            "is_experiment_result": False,
            "usage": raw_rows[-1]["usage"],
        })

    # Write outputs (append if resuming)
    append = args.resume and len(completed_ids) > 0
    write_jsonl(output_dir / "raw_responses.jsonl", raw_rows, append=append)
    write_jsonl(output_dir / "parsed_predictions.jsonl", parsed_rows, append=append)
    write_jsonl(output_dir / "grounded_predictions.jsonl", grounded_rows, append=append)
    write_jsonl(output_dir / "failed_cases.jsonl", failed_rows, append=append)

    # Write manifest
    manifest = {
        "backend": "vllm",
        "model": model_source,
        "model_alias": "qwen3_8b",
        "provider": "server_vllm",
        "requested_input_count": len(inputs),
        "newly_processed_count": len(pending),
        "total_grounded_count": len(grounded_rows) + len(completed_ids),
        "failed_count": len(failed_rows),
        "inference_time_seconds": elapsed,
        "seconds_per_example": elapsed / max(len(pending), 1),
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "resume": args.resume,
        "server_executed": True,
        "created_at_utc": utc_now_iso(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(f"\nResults: {len(grounded_rows)} grounded, {len(failed_rows)} failed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
