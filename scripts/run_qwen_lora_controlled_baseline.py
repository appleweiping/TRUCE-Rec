#!/usr/bin/env python3
"""Train and score a Qwen LoRA controlled baseline on the server.

This is intended for the GPU server. It trains a LoRA adapter from
`train_sft.jsonl`, scores candidates from `test_score_plan.jsonl`, and writes
`candidate_scores.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-score-rows", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = run(manifest, args=args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def run(manifest: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(str(manifest["output_dir"]))
    adapter_dir = output_dir / "adapter"
    logs_dir = output_dir / "logs"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = _environment()
    (output_dir / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "git_info.json").write_text(json.dumps(_git_info(), indent=2, sort_keys=True), encoding="utf-8")

    base_model = str(manifest["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not args.skip_train:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if manifest["training"].get("bf16") else torch.float16,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        if bool(manifest["training"].get("gradient_checkpointing")):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        lora = manifest["lora"]
        lora_config = LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=[str(x) for x in lora["target_modules"]],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        train_rows = _read_jsonl(Path(manifest["files"]["train_sft"]))
        if args.max_train_examples:
            train_rows = train_rows[: args.max_train_examples]
        train_seconds = _train(model, tokenizer, train_rows, manifest=manifest, max_steps=args.max_steps)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        score_model = model
    else:
        train_seconds = 0.0
        score_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if manifest["training"].get("bf16") else torch.float16,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        score_model = PeftModel.from_pretrained(score_model, adapter_dir)
    score_model.eval()
    score_rows = _read_jsonl(Path(manifest["files"]["test_score_plan"]))
    if args.max_score_rows:
        score_rows = score_rows[: args.max_score_rows]
    start = time.time()
    scores_path = output_dir / "candidate_scores.csv"
    _score(score_model, tokenizer, score_rows, scores_path=scores_path)
    scoring_seconds = time.time() - start
    summary = {
        "status": "completed",
        "controlled_baseline_name": manifest["controlled_baseline_name"],
        "implementation_fidelity": manifest.get("implementation_fidelity", "controlled_adapter_pilot"),
        "official_native_controlled": bool(manifest.get("official_native_controlled", False)),
        "official_fidelity_audit_required": bool(manifest.get("official_fidelity_audit_required", True)),
        "base_model_policy": manifest.get("base_model_policy", "shared_qwen3_8b_base_model"),
        "adapter_training_policy": manifest.get("adapter_training_policy", "baseline_official_lora_with_project_modules"),
        "provenance": manifest.get("provenance", {}),
        "train_seconds": train_seconds,
        "scoring_seconds": scoring_seconds,
        "score_rows": len(score_rows),
        "candidate_scores": str(scores_path),
        "adapter_dir": str(adapter_dir),
        "is_experiment_result": True,
        "is_paper_result": False,
        "paper_table_policy": manifest.get(
            "paper_table_policy",
            "Controlled adapter pilot unless an official-fidelity audit promotes it.",
        ),
        "next_step": "Import candidate_scores.csv with TRUCE import_external_predictions.py --split test and evaluate_predictions.py.",
    }
    (output_dir / "training_scoring_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _train(model: Any, tokenizer: Any, rows: list[dict[str, Any]], *, manifest: dict[str, Any], max_steps: int | None) -> float:
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    training = manifest["training"]
    batch_size = int(training.get("per_device_train_batch_size") or 1)
    grad_accum = int(training.get("gradient_accumulation_steps") or 1)
    epochs = int(training.get("num_train_epochs") or 1)
    loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: _collate(tokenizer, batch))
    optimizer = AdamW(model.parameters(), lr=float(training.get("learning_rate") or 2e-4), weight_decay=float(training.get("weight_decay") or 0.0))
    total_steps = max_steps or max(1, math.ceil(len(loader) * epochs / grad_accum))
    warmup_steps = int(total_steps * float(training.get("warmup_ratio") or 0.0))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    start = time.time()
    step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(epochs):
        for batch_index, batch in enumerate(loader, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            if batch_index % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                print(json.dumps({"train_step": step, "loss": float(loss.detach().cpu()) * grad_accum}), flush=True)
                if max_steps and step >= max_steps:
                    return time.time() - start
    return time.time() - start


def _collate(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    input_ids = []
    labels = []
    for row in rows:
        messages = row["messages"]
        prompt = messages[0]["content"]
        answer = messages[1]["content"]
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        answer_ids = tokenizer(answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
        ids = prompt_ids + answer_ids
        label = [-100] * len(prompt_ids) + answer_ids
        input_ids.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor(label, dtype=torch.long))
    padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (padded != tokenizer.pad_token_id).long()
    return {"input_ids": padded, "attention_mask": attention_mask, "labels": padded_labels}


def _score(model: Any, tokenizer: Any, rows: list[dict[str, Any]], *, scores_path: Path) -> None:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "user_id", "item_id", "score"])
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            prompt = str(row["prompt"])
            candidate_items = [str(x) for x in row.get("candidate_item_ids") or []]
            candidate_outputs = [str(x) for x in row.get("candidate_outputs") or []]
            for item_id, output in zip(candidate_items, candidate_outputs):
                score = _conditional_logprob(model, tokenizer, prompt, output)
                writer.writerow({
                    "example_id": row.get("example_id") or "",
                    "user_id": row.get("user_id") or "",
                    "item_id": item_id,
                    "score": score,
                })
            if index % 25 == 0:
                print(json.dumps({"scored_rows": index}), flush=True)


def _conditional_logprob(model: Any, tokenizer: Any, prompt: str, output: str) -> float:
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    output_ids = tokenizer(output, add_special_tokens=False).input_ids
    if not output_ids:
        return -1e9
    input_ids = torch.tensor([prompt_ids + output_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids)
    total = 0.0
    for offset, token_id in enumerate(output_ids):
        pos = start + offset - 1
        total += float(log_probs[0, pos, token_id].detach().cpu())
    return total / len(output_ids)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }


def _git_info() -> dict[str, str]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "status_short": run_git(["status", "--short"]),
        "remote": run_git(["remote", "get-url", "origin"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
