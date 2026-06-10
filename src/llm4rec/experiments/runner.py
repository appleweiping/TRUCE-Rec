"""Deterministic smoke experiment runner for skeleton and minimal baselines."""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm4rec.data.preprocess import preprocess_dataset
from llm4rec.evaluation.aggregation import aggregate_run_metrics
from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.evaluation.table_export import export_phase5_tables
from llm4rec.experiments.config import load_config, save_resolved_config
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.seeding import set_seed
from llm4rec.io.artifacts import create_run_dir, read_jsonl, write_environment, write_json, write_jsonl
from llm4rec.llm.base import LLMRequest, LLMResponse
from llm4rec.llm.hf_provider import HFLocalProvider
from llm4rec.llm.mock_provider import MockLLMProvider
from llm4rec.llm.openai_provider import OpenAICompatibleProvider
from llm4rec.llm.response_cache import ResponseCache
from llm4rec.methods.ours_method import OursMethodRanker
from llm4rec.methods.calm_rec import CALMConfig, CALMRecRanker
from llm4rec.methods.calm_encoders import build_encoders
from llm4rec.rankers.base import BaseRanker
from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.rankers.llm_generative import LLMConfidenceObservationRanker, LLMGenerativeRanker
from llm4rec.rankers.llm_reranker import LLMReranker
from llm4rec.rankers.mf import MatrixFactorizationRanker
from llm4rec.rankers.popularity import PopularityRanker
from llm4rec.rankers.random import RandomRanker
from llm4rec.rankers.sequential import MarkovSequentialRanker, SasrecInterfaceRanker, SequentialLastItemRanker
from llm4rec.trainers.lora import LoraTrainer
from llm4rec.trainers.sequential import SequentialTrainer
from llm4rec.trainers.traditional import TraditionalRankerTrainer


SEQUENTIAL_METHODS = {"sequential_last_item", "sequential_markov", "sasrec_interface", "sequential_interface"}


class ControlledLLMProvider:
    """Runtime guard for approved API runs: cache, cap requests, and estimate cost."""

    def __init__(
        self,
        provider: Any,
        *,
        llm_config: dict[str, Any],
        safety_config: dict[str, Any],
    ) -> None:
        self.provider = provider
        self.provider_name = provider.provider_name
        self.model_name = provider.model_name
        self.supports_logprobs = provider.supports_logprobs
        self.supports_seed = provider.supports_seed
        self.llm_config = llm_config
        self.safety_config = safety_config
        request_limits = dict(llm_config.get("request_limits") or {})
        cache_config = dict(llm_config.get("cache") or {})
        self.max_requests = _optional_int(safety_config.get("max_requests") or request_limits.get("max_requests"))
        self.cost_limit_usd = _optional_float(safety_config.get("cost_limit_usd"))
        self.cache_enabled = bool(cache_config.get("enabled", False))
        self.cache_require_hit = bool(cache_config.get("require_hit", False))
        cache_dir = str(cache_config.get("cache_dir") or "outputs/api_cache/llm4rec")
        self.cache = ResponseCache(cache_dir) if self.cache_enabled else None
        if self.cache_require_hit and self.cache is None:
            raise ValueError("llm.cache.require_hit requires llm.cache.enabled=true")
        pricing = dict(llm_config.get("pricing") or {})
        self.input_per_1m_tokens = float(pricing.get("input_per_1m_tokens") or 0.0)
        self.output_per_1m_tokens = float(pricing.get("output_per_1m_tokens") or 0.0)
        self.real_request_count = 0
        self.cache_hit_count = 0
        self.estimated_cost_total = 0.0
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def generate(self, request: LLMRequest) -> LLMResponse:
        cache_key = self._cache_key(request)
        if self.cache is not None:
            replay_start = time.perf_counter()
            with self._lock:
                cached = self.cache.get(cache_key)
            replay_latency_seconds = time.perf_counter() - replay_start
            if cached is not None:
                response = _response_from_cache(
                    cached,
                    cache_key=cache_key,
                    replay_latency_seconds=replay_latency_seconds,
                )
                with self._lock:
                    self.cache_hit_count += 1
                    self.records.append(_provider_record(response, replay_estimated_cost=0.0))
                return response
            if self.cache_require_hit:
                raise RuntimeError(
                    "LLM cache-only replay missing response cache entry; "
                    f"cache_key={cache_key}"
                )

        with self._lock:
            if self.max_requests is not None and self.real_request_count >= self.max_requests:
                raise RuntimeError(f"LLM max_requests exceeded: {self.max_requests}")

        response = self.provider.generate(request)
        estimated_cost = self._estimated_cost(response.usage)
        with self._lock:
            self.real_request_count += 1
            self.estimated_cost_total += estimated_cost
            real_request_index = self.real_request_count
            estimated_cost_total = self.estimated_cost_total
        metadata = dict(response.metadata)
        metadata.update(
            {
                "cache_key": cache_key,
                "real_request_index": real_request_index,
                "estimated_cost": estimated_cost,
                "estimated_cost_total": estimated_cost_total,
            }
        )
        guarded = LLMResponse(
            text=response.text,
            raw_response=response.raw_response,
            usage=response.usage,
            latency_seconds=response.latency_seconds,
            model=response.model,
            provider=response.provider,
            cache_hit=False,
            metadata=metadata,
        )
        with self._lock:
            self.records.append(_provider_record(guarded, live_estimated_cost=estimated_cost))
        if self.cost_limit_usd is not None and estimated_cost_total > self.cost_limit_usd:
            raise RuntimeError(
                f"LLM cost limit exceeded: estimated {estimated_cost_total:.6f} > {self.cost_limit_usd:.6f}"
            )
        if self.cache is not None:
            with self._lock:
                self.cache.set(cache_key, _response_cache_payload(guarded))
        return guarded

    def _cache_key(self, request: LLMRequest) -> str:
        if self.cache is None:
            return ""
        return self.cache.key_for(
            provider=self.provider_name,
            model=self.model_name,
            prompt=request.prompt,
            params={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "seed": request.seed,
                "extra_body": self.llm_config.get("extra_body") or {},
            },
        )

    def _estimated_cost(self, usage: dict[str, int]) -> float:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        return (
            prompt_tokens / 1_000_000.0 * self.input_per_1m_tokens
            + completion_tokens / 1_000_000.0 * self.output_per_1m_tokens
        )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = [dict(row) for row in self.records]
            real_request_count = self.real_request_count
            cache_hit_count = self.cache_hit_count
        live_records = [row for row in records if not row.get("cache_hit")]
        cache_records = [row for row in records if row.get("cache_hit")]
        current_latencies = [float(row.get("latency_seconds") or 0.0) for row in records]
        original_live_latencies = [
            float(row.get("original_latency_seconds") or row.get("latency_seconds") or 0.0)
            for row in records
        ]
        live_cost_usd = sum(float(row.get("live_cost_usd") or row.get("estimated_cost") or 0.0) for row in live_records)
        replay_cost_usd = sum(float(row.get("replay_cost_usd") or 0.0) for row in cache_records)
        original_cached_cost_usd = sum(float(row.get("original_cached_cost_usd") or 0.0) for row in cache_records)
        effective_cost_usd = live_cost_usd + original_cached_cost_usd
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "request_count": len(records),
            "total_requests": len(records),
            "real_request_count": real_request_count,
            "live_provider_requests": real_request_count,
            "cache_hit_count": cache_hit_count,
            "cache_hit_requests": cache_hit_count,
            "cache_hit_rate": cache_hit_count / len(records) if records else 0.0,
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in records),
            "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in records),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in records),
            "live_prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in live_records),
            "live_completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in live_records),
            "live_total_tokens": sum(int(row.get("total_tokens") or 0) for row in live_records),
            "original_cached_prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in cache_records),
            "original_cached_completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in cache_records),
            "original_cached_total_tokens": sum(int(row.get("total_tokens") or 0) for row in cache_records),
            "live_cost_usd": live_cost_usd,
            "replay_cost_usd": replay_cost_usd,
            "original_cached_cost_usd": original_cached_cost_usd,
            "effective_cost_usd": effective_cost_usd,
            "estimated_cost": effective_cost_usd,
            "live_latency_seconds_sum": sum(float(row.get("latency_seconds") or 0.0) for row in live_records),
            "replay_latency_seconds_sum": sum(float(row.get("replay_latency_seconds") or 0.0) for row in cache_records),
            "original_cached_latency_seconds_sum": sum(float(row.get("original_latency_seconds") or 0.0) for row in cache_records),
            "latency_p50_seconds": _percentile(current_latencies, 0.50),
            "latency_p95_seconds": _percentile(current_latencies, 0.95),
            "original_live_latency_p50_seconds": _percentile(original_live_latencies, 0.50),
            "original_live_latency_p95_seconds": _percentile(original_live_latencies, 0.95),
            "latency_p50": _percentile(current_latencies, 0.50),
            "latency_p95": _percentile(current_latencies, 0.95),
            "max_requests": self.max_requests,
            "cost_limit_usd": self.cost_limit_usd,
            "cache_require_hit": self.cache_require_hit,
        }


class CacheOnlyLLMProviderStub:
    """Configured provider identity for cache-only replay without network setup."""

    supports_logprobs = False
    supports_seed = True

    def __init__(self, *, provider_name: str, model_name: str) -> None:
        self.provider_name = provider_name
        self.model_name = model_name

    def generate(self, _request: LLMRequest) -> LLMResponse:
        raise RuntimeError("cache-only replay forbids live provider calls")


def run_preprocess_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return preprocess_dataset(config)


def run_experiment(config_path: str | Path, *, preprocess: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    return run_experiment_config(config, preprocess=preprocess)


def run_experiment_config(config: dict[str, Any], *, preprocess: bool = False) -> dict[str, Any]:
    config = _resolve_runtime_config(config)
    seed = set_seed(int(config.get("seed") or 0))
    method_name = _method_name(config)
    run_name = _run_name(config, method_name)
    run_id = f"{run_name}_seed{seed}"
    run_dir = create_run_dir(_output_dir(config), run_id)
    logger = RunLogger(run_dir / "logs.txt")
    logger.log(f"start run_id={run_id} method={method_name}")
    if preprocess:
        dataset_config = load_config(_required_path(config, "dataset", "config_path"))
        preprocess_manifest = preprocess_dataset(dataset_config)
        logger.log(f"preprocessed dataset to {preprocess_manifest['processed_dir']}")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_environment(run_dir)
    predictions = _build_predictions(config, run_dir=run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    _write_raw_llm_outputs(run_dir, predictions)
    logger.log(f"wrote predictions count={len(predictions)}")
    eval_context = _evaluation_context_for_config(config)
    metrics = evaluate_predictions(
        predictions_jsonl=predictions_path,
        output_dir=run_dir,
        top_k=[int(k) for k in config.get("top_k", [1, 5, 10])],
        item_catalog=eval_context["item_catalog"],
        train_examples=eval_context["train_examples"],
        all_examples=eval_context["all_examples"],
        evaluation_config=config,
    )
    logger.log("wrote metrics.json and metrics.csv")
    cost_latency = _cost_latency_for_run(metrics, run_dir)
    write_json(run_dir / "cost_latency.json", cost_latency)
    logger.log("wrote cost_latency.json")
    logger.log("finish")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "predictions": str(predictions_path),
        "metrics": metrics,
    }


def run_all(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    if str(config.get("experiment_kind") or "") == "cu_gr_v2_preference_subgate":
        from llm4rec.experiments.cu_gr_v2_preference import run_cu_gr_v2_preference_subgate

        return run_cu_gr_v2_preference_subgate(config_path)
    baselines = config.get("baselines")
    if not baselines:
        return run_experiment_config(config, preprocess=True)
    if not isinstance(baselines, list) or not all(isinstance(item, str) for item in baselines):
        raise ValueError("config.baselines must be an inline list of baseline names")
    runs = []
    preprocess_manifests = []
    seeds = _seeds_for_config(config)
    candidate_sizes = _candidate_sizes_for_config(config)
    preprocess_enabled = not (
        isinstance(config.get("preprocess"), dict)
        and config.get("preprocess", {}).get("enabled") is False
    )
    for candidate_size in candidate_sizes:
        sized_config = _config_for_candidate_size(config, candidate_size)
        if preprocess_enabled:
            dataset_config = _dataset_config_for_experiment(sized_config)
            preprocess_manifest = preprocess_dataset(dataset_config)
        else:
            processed_dir = Path(_required_path(sized_config, "dataset", "processed_dir"))
            manifest_path = processed_dir / "preprocess_manifest.json"
            preprocess_manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists()
                else {"processed_dir": str(processed_dir), "preprocess_skipped": True}
            )
        preprocess_manifests.append(preprocess_manifest)
        for seed in seeds:
            for baseline in baselines:
                child_config = _config_for_baseline(sized_config, baseline, seed=seed)
                result = _completed_run_result(child_config)
                if result is None:
                    result = run_experiment_config(child_config, preprocess=False)
                result["preprocess_manifest"] = preprocess_manifest
                if candidate_size is not None:
                    result["candidate_size"] = candidate_size
                runs.append(result)
    postprocess = _postprocess_run_all(config, runs)
    return {
        "run_count": len(runs),
        "baseline_methods": baselines,
        "seeds": seeds,
        "candidate_sizes": [size for size in candidate_sizes if size is not None],
        "preprocess_manifest": preprocess_manifests[0] if len(preprocess_manifests) == 1 else preprocess_manifests,
        "runs": runs,
        "postprocess": postprocess,
    }


def _build_predictions(config: dict[str, Any], *, run_dir: Path) -> list[dict[str, Any]]:
    method = _method_name(config)
    if method == "skeleton":
        return _build_skeleton_predictions(config)
    if method == "lora_dry_run":
        return _build_lora_dry_run(config, run_dir=run_dir)
    return _build_ranker_predictions(config, run_dir=run_dir)


def _build_skeleton_predictions(config: dict[str, Any]) -> list[dict[str, Any]]:
    method = _method_name(config)
    if method != "skeleton":
        raise ValueError("Phase 1 runner only supports method: skeleton")
    processed_dir = Path(_required_path(config, "dataset", "processed_dir"))
    split = str(config.get("split") or "test")
    examples = [
        row for row in read_jsonl(processed_dir / "examples.jsonl")
        if str(row.get("split")) == split
    ]
    if not examples:
        raise ValueError(f"no examples found for split={split} in {processed_dir}")
    predictions = []
    for row in examples:
        target = str(row["target"])
        candidates = [str(item) for item in row.get("candidates", [])]
        ordered = [target] + sorted(item for item in candidates if item != target)
        scores = [1.0 / (rank + 1) for rank in range(len(ordered))]
        predictions.append(
            {
                "user_id": str(row["user_id"]),
                "target_item": target,
                "candidate_items": candidates,
                "predicted_items": ordered,
                "scores": scores,
                "method": "skeleton",
                "domain": str(row.get("domain") or config.get("domain") or "tiny"),
                "raw_output": None,
                "metadata": {
                    "example_id": row.get("example_id"),
                    "split": row.get("split"),
                    "phase": "phase1_reproducible_skeleton",
                    "not_a_formal_baseline": True,
                },
            }
        )
    return predictions


def _build_ranker_predictions(config: dict[str, Any], *, run_dir: Path) -> list[dict[str, Any]]:
    processed_dir = Path(_required_path(config, "dataset", "processed_dir"))
    split = str(config.get("split") or "test")
    examples = read_jsonl(processed_dir / "examples.jsonl")
    item_catalog = _read_csv(processed_dir / "items.csv")
    interactions = _read_csv(processed_dir / "interactions.csv")
    train_examples = [row for row in examples if str(row.get("split")) == "train"]
    eval_examples = _limit_eval_examples(config, [row for row in examples if str(row.get("split")) == split])
    if not eval_examples:
        raise ValueError(f"no examples found for split={split} in {processed_dir}")
    ranker = _build_ranker(config, run_dir=run_dir)
    if _method_name(config) in SEQUENTIAL_METHODS:
        training = _training_config(config)
        checkpoint_dir = Path(str(training.get("checkpoint_dir") or (run_dir / "checkpoints")))
        trainer = SequentialTrainer(
            ranker,
            train_examples=train_examples,
            item_catalog=item_catalog,
            interactions=interactions,
            checkpoint_dir=checkpoint_dir,
            config=config,
            seed=int(config.get("seed") or 0),
            eval_only=bool(training.get("eval_only", False)),
        )
        trainer.train()
        return trainer.predict(eval_examples)
    trainer = TraditionalRankerTrainer(
        ranker,
        train_examples=train_examples,
        item_catalog=item_catalog,
        interactions=interactions,
    )
    trainer_result = trainer.train()
    predictions = _rank_eval_examples(
        config,
        ranker=ranker,
        eval_examples=eval_examples,
        trainer_metadata=trainer_result.metadata,
        run_dir=run_dir,
    )
    _write_llm_provider_artifacts(run_dir, ranker)
    return predictions


def _build_lora_dry_run(config: dict[str, Any], *, run_dir: Path) -> list[dict[str, Any]]:
    training = _training_config(config)
    output_dir = str(training.get("output_dir") or (run_dir / "artifacts" / "lora_dry_run"))
    trainer = LoraTrainer({**training, "output_dir": output_dir}, dry_run=bool(training.get("dry_run", True)))
    result = trainer.train()
    write_json(run_dir / "artifacts" / "lora_dry_run_manifest.json", result.metadata)
    return []


def _rank_eval_examples(
    config: dict[str, Any],
    *,
    ranker: BaseRanker,
    eval_examples: list[dict[str, Any]],
    trainer_metadata: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    concurrency_levels = _concurrency_levels(config)
    if len(concurrency_levels) == 1 and concurrency_levels[0] == 1:
        return [
            _rank_one_example(ranker, example, trainer_metadata=trainer_metadata)
            for example in eval_examples
        ]

    predictions_by_index: dict[int, dict[str, Any]] = {}
    remaining = list(enumerate(eval_examples))
    failure_rows: list[dict[str, Any]] = []
    for concurrency in concurrency_levels:
        current_failures: list[tuple[int, dict[str, Any], str]] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_rank_one_example, ranker, example, trainer_metadata=trainer_metadata): (index, example)
                for index, example in remaining
            }
            for future in as_completed(futures):
                index, example = futures[future]
                try:
                    predictions_by_index[index] = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve partial API artifacts for audit.
                    current_failures.append((index, example, str(exc)))
        failure_rows.extend(
            {
                "example_index": index,
                "example_id": example.get("example_id"),
                "user_id": example.get("user_id"),
                "method": _method_name(config),
                "concurrency": concurrency,
                "error": error,
            }
            for index, example, error in current_failures
        )
        if not current_failures:
            remaining = []
            break
        remaining = [(index, example) for index, example, _ in current_failures]

    if remaining:
        method = _method_name(config)
        for index, example in remaining:
            predictions_by_index[index] = _api_failure_prediction(config, example, method=method)
    if failure_rows:
        write_jsonl(run_dir / "artifacts" / "api_failures.jsonl", failure_rows)
    return [predictions_by_index[index] for index in sorted(predictions_by_index)]


def _rank_one_example(
    ranker: BaseRanker,
    example: dict[str, Any],
    *,
    trainer_metadata: dict[str, Any],
) -> dict[str, Any]:
    candidates = [str(item_id) for item_id in example.get("candidates", [])]
    result = ranker.rank(example, candidates)
    record = result.to_prediction_record()
    record["metadata"].update(
        {
            "trainer": trainer_metadata,
            "not_ours_method": True,
        }
    )
    record["metadata"].setdefault("phase", "phase2_minimal_baseline")
    return record


def _concurrency_levels(config: dict[str, Any]) -> list[int]:
    if not _method_uses_llm(config):
        return [1]
    safety = _safety_config(config)
    start = _optional_int(safety.get("concurrency")) or 1
    start = max(1, start)
    if start == 1:
        return [1]
    levels = []
    value = start
    while value >= 2:
        levels.append(value)
        value //= 2
    return _unique_ints(levels)


def _method_uses_llm(config: dict[str, Any]) -> bool:
    method = _method_name(config)
    method_config = _method_config(config)
    provider = ""
    llm = config.get("llm")
    if isinstance(llm, dict):
        provider = str(llm.get("provider") or llm.get("type") or "")
    return bool(provider == "openai_compatible" and (method.startswith("llm_") or _is_ours_method(method, method_config)))


def _api_failure_prediction(config: dict[str, Any], example: dict[str, Any], *, method: str) -> dict[str, Any]:
    return {
        "user_id": str(example.get("user_id") or ""),
        "target_item": str(example.get("target") or ""),
        "candidate_items": [str(item_id) for item_id in example.get("candidates", [])],
        "predicted_items": [],
        "scores": [],
        "method": method,
        "domain": str(example.get("domain") or config.get("domain") or "unknown"),
        "raw_output": None,
        "metadata": {
            "example_id": example.get("example_id"),
            "split": example.get("split"),
            "api_failure": True,
            "parse_success": False,
            "parse_error": "api_failure_after_concurrency_retries",
            "is_catalog_valid": False,
            "is_hallucinated": False,
            "candidate_adherent": False,
            "confidence": None,
            "provider": (_llm_config(config).get("provider") if isinstance(config.get("llm"), dict) else None),
            "model": (_llm_config(config).get("model") if isinstance(config.get("llm"), dict) else None),
        },
    }


def _build_ranker(config: dict[str, Any], *, run_dir: Path | None = None) -> BaseRanker:
    method = _method_name(config)
    method_config = _method_config(config)
    params = dict(method_config.get("params") or {})
    seed = int(method_config.get("seed") or config.get("seed") or 0)
    if method == "random":
        return RandomRanker(seed=seed)
    if method == "popularity":
        return PopularityRanker()
    if method == "bm25":
        return BM25Ranker(
            text_policy=str(params.get("text_policy") or "title"),
            k1=float(params.get("k1") or 1.5),
            b=float(params.get("b") or 0.75),
        )
    if method in {"mf", "matrix_factorization"}:
        return MatrixFactorizationRanker(
            seed=seed,
            factors=int(params.get("factors") or 8),
            epochs=int(params.get("epochs") or 25),
            learning_rate=float(params.get("learning_rate") or 0.05),
            regularization=float(params.get("regularization") or 0.001),
        )
    if method == "sequential_last_item":
        return SequentialLastItemRanker(max_history_length=int(params.get("max_history_length") or 50))
    if method in {"sequential_markov", "sequential_interface"}:
        return MarkovSequentialRanker(max_history_length=int(params.get("max_history_length") or 50))
    if method == "sasrec_interface":
        return SasrecInterfaceRanker(max_history_length=int(params.get("max_history_length") or 50))
    if method in {"llm_generative", "llm_generative_mock"}:
        return LLMGenerativeRanker(
            provider=_build_llm_provider(config, run_dir=run_dir),
            method_name=str(params.get("prediction_method") or "llm_generative_mock"),
            text_policy=str(params.get("text_policy") or "title"),
        )
    if method == "llm_rerank":
        return LLMReranker(
            provider=_build_llm_provider(config, run_dir=run_dir),
            method_name=str(params.get("prediction_method") or "llm_rerank_mock"),
            text_policy=str(params.get("text_policy") or "title"),
        )
    if method == "llm_confidence_observation":
        return LLMConfidenceObservationRanker(
            provider=_build_llm_provider(config, run_dir=run_dir),
            method_name=str(params.get("prediction_method") or "llm_confidence_observation_mock"),
            text_policy=str(params.get("text_policy") or "title"),
        )
    if method in {"calm_rec", "calm"}:
        backend = str(params.get("backend") or "hashed")
        n_intents = int(params.get("n_intents") or params.get("K") or 4)
        item_encoder, intent_encoder = build_encoders(
            backend=backend,
            n_intents=n_intents,
            dim=int(params.get("dim") or 64),
            qwen_model_path=params.get("qwen_model_path"),
            qwen_lora_path=params.get("qwen_lora_path"),
            seed=seed,
        )
        return CALMRecRanker(
            item_encoder=item_encoder,
            intent_encoder=intent_encoder,
            config=CALMConfig.from_params(params),
            method_config=method_config,
            seed=seed,
        )
    if _is_ours_method(method, method_config):
        return OursMethodRanker(
            provider=_build_llm_provider(config, run_dir=run_dir),
            method_config=method_config,
            seed=seed,
        )
    raise ValueError(f"unknown experiment method: {method}")


def _build_llm_provider(config: dict[str, Any], *, run_dir: Path | None = None) -> Any:
    llm_config = _llm_config(config)
    provider_type = str(llm_config.get("provider") or llm_config.get("type") or "")
    if provider_type == "mock":
        return MockLLMProvider(
            response_mode=str(llm_config.get("response_mode") or "generative_correct"),
            seed=int(llm_config.get("seed") or config.get("seed") or 0),
        )
    if provider_type == "openai_compatible":
        if _cache_require_hit(llm_config):
            provider = CacheOnlyLLMProviderStub(
                provider_name="openai_compatible",
                model_name=_required_llm_text(llm_config, "model"),
            )
            if run_dir is not None:
                return ControlledLLMProvider(provider, llm_config=llm_config, safety_config=_safety_config(config))
            return provider
        if run_dir is not None:
            _assert_api_execution_approved(config, llm_config)
        provider = OpenAICompatibleProvider(
            model_name=_required_llm_text(llm_config, "model"),
            api_key_env=_required_llm_text(llm_config, "api_key_env"),
            base_url=_required_llm_text(llm_config, "base_url"),
            timeout_seconds=float(llm_config.get("timeout_seconds") or _safety_config(config).get("request_timeout_seconds") or 60.0),
            max_retries=int(llm_config.get("max_retries") or _safety_config(config).get("max_retries") or 2),
            backoff_seconds=float(llm_config.get("backoff_seconds") or _safety_config(config).get("backoff_seconds") or 0.25),
            extra_body=dict(llm_config.get("extra_body") or {}),
        )
        if run_dir is not None:
            return ControlledLLMProvider(provider, llm_config=llm_config, safety_config=_safety_config(config))
        return provider
    if provider_type == "hf_local":
        return HFLocalProvider(
            model_name_or_path=_required_llm_text(llm_config, "model_name_or_path"),
            device=str(llm_config.get("device") or "auto"),
        )
    raise ValueError(f"unknown llm provider type: {provider_type}")


def _assert_api_execution_approved(config: dict[str, Any], llm_config: dict[str, Any]) -> None:
    safety = _safety_config(config)
    if bool(config.get("dry_run", False)) or bool(safety.get("dry_run", False)):
        raise RuntimeError("real API execution blocked: dry_run must be false")
    if bool(config.get("requires_confirm", False)) or bool(safety.get("requires_confirm", False)):
        raise RuntimeError("real API execution blocked: requires_confirm must be false")
    if safety.get("allow_api_calls") is not True:
        raise RuntimeError("real API execution blocked: safety.allow_api_calls must be true")
    if safety.get("acknowledged_expensive_run") is not True:
        raise RuntimeError("real API execution blocked: safety.acknowledged_expensive_run must be true")
    model = _required_llm_text(llm_config, "model")
    base_url = _required_llm_text(llm_config, "base_url")
    api_key_env = _required_llm_text(llm_config, "api_key_env")
    if _is_tbd_text(model) or _is_tbd_text(base_url) or _is_tbd_text(api_key_env):
        raise RuntimeError("real API execution blocked: provider model/base_url/api_key_env must not be TBD")
    if not os.environ.get(api_key_env):
        raise RuntimeError(f"real API execution blocked: missing API key environment variable {api_key_env}")


def _safety_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("safety") or {}) if isinstance(config.get("safety"), dict) else {}


def _cache_require_hit(llm_config: dict[str, Any]) -> bool:
    cache = llm_config.get("cache")
    return isinstance(cache, dict) and bool(cache.get("require_hit", False))


def _limit_eval_examples(config: dict[str, Any], eval_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset = dict(config.get("dataset") or {}) if isinstance(config.get("dataset"), dict) else {}
    safety = _safety_config(config)
    limits = [
        _optional_int(dataset.get("subset_size")),
        _optional_int(safety.get("subset_size")),
        _optional_int(safety.get("max_examples")),
    ]
    active = [value for value in limits if value is not None]
    if not active:
        return eval_examples
    limit = min(active)
    if limit <= 0:
        raise ValueError("subset_size/max_examples must be positive when configured")
    return eval_examples[:limit]


def _write_raw_llm_outputs(run_dir: Path, predictions: list[dict[str, Any]]) -> None:
    rows = []
    for index, row in enumerate(predictions):
        metadata = dict(row.get("metadata") or {})
        raw_fields = {
            "raw_output": row.get("raw_output"),
            "candidate_normalized_raw_output": metadata.get("candidate_normalized_raw_output"),
            "verification_raw_output": metadata.get("verification_raw_output"),
        }
        if not any(value for value in raw_fields.values()):
            continue
        rows.append(
            {
                "row_index": index,
                "user_id": row.get("user_id"),
                "method": row.get("method"),
                "provider": metadata.get("provider"),
                "model": metadata.get("model"),
                "prompt_template_id": metadata.get("prompt_template_id"),
                "prompt_hash": metadata.get("prompt_hash"),
                "cache_hit": metadata.get("cache_hit"),
                "token_usage": metadata.get("token_usage"),
                "latency_seconds": metadata.get("latency_seconds"),
                **raw_fields,
            }
        )
    if rows:
        write_jsonl(run_dir / "raw_llm_outputs.jsonl", rows)


def _write_llm_provider_artifacts(run_dir: Path, ranker: BaseRanker) -> None:
    provider = getattr(ranker, "provider", None)
    if not isinstance(provider, ControlledLLMProvider):
        return
    write_json(run_dir / "artifacts" / "llm_provider_summary.json", provider.summary())
    with provider._lock:
        records = [dict(row) for row in provider.records]
    if records:
        write_jsonl(run_dir / "artifacts" / "llm_request_log.jsonl", records)


def _cost_latency_for_run(metrics: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    cost_latency = dict(metrics.get("aggregate", {}).get("efficiency", {}))
    provider_summary_path = run_dir / "artifacts" / "llm_provider_summary.json"
    if not provider_summary_path.exists():
        return cost_latency
    provider_summary = json.loads(provider_summary_path.read_text(encoding="utf-8"))
    cost_latency.update(
        {
            "provider_request_count": provider_summary.get("request_count", 0),
            "provider_real_request_count": provider_summary.get("real_request_count", 0),
            "provider_cache_hit_count": provider_summary.get("cache_hit_count", 0),
            "total_requests": provider_summary.get("total_requests", provider_summary.get("request_count", 0)),
            "live_provider_requests": provider_summary.get(
                "live_provider_requests", provider_summary.get("real_request_count", 0)
            ),
            "cache_hit_requests": provider_summary.get("cache_hit_requests", provider_summary.get("cache_hit_count", 0)),
            "cache_hit_rate": provider_summary.get("cache_hit_rate", cost_latency.get("cache_hit_rate", 0.0)),
            "prompt_tokens": provider_summary.get("prompt_tokens", cost_latency.get("prompt_tokens", 0)),
            "completion_tokens": provider_summary.get("completion_tokens", cost_latency.get("completion_tokens", 0)),
            "total_tokens": provider_summary.get("total_tokens", cost_latency.get("total_tokens", 0)),
            "estimated_cost": provider_summary.get("estimated_cost", cost_latency.get("estimated_cost", 0.0)),
            "live_prompt_tokens": provider_summary.get("live_prompt_tokens", cost_latency.get("live_prompt_tokens", 0)),
            "live_completion_tokens": provider_summary.get(
                "live_completion_tokens", cost_latency.get("live_completion_tokens", 0)
            ),
            "live_total_tokens": provider_summary.get("live_total_tokens", cost_latency.get("live_total_tokens", 0)),
            "original_cached_prompt_tokens": provider_summary.get(
                "original_cached_prompt_tokens", cost_latency.get("original_cached_prompt_tokens", 0)
            ),
            "original_cached_completion_tokens": provider_summary.get(
                "original_cached_completion_tokens", cost_latency.get("original_cached_completion_tokens", 0)
            ),
            "original_cached_total_tokens": provider_summary.get(
                "original_cached_total_tokens", cost_latency.get("original_cached_total_tokens", 0)
            ),
            "live_cost_usd": provider_summary.get("live_cost_usd", cost_latency.get("live_cost_usd", 0.0)),
            "replay_cost_usd": provider_summary.get("replay_cost_usd", cost_latency.get("replay_cost_usd", 0.0)),
            "original_cached_cost_usd": provider_summary.get(
                "original_cached_cost_usd", cost_latency.get("original_cached_cost_usd", 0.0)
            ),
            "effective_cost_usd": provider_summary.get("effective_cost_usd", cost_latency.get("effective_cost_usd", 0.0)),
            "live_latency_seconds_sum": provider_summary.get(
                "live_latency_seconds_sum", cost_latency.get("live_latency_seconds_sum", 0.0)
            ),
            "replay_latency_seconds_sum": provider_summary.get(
                "replay_latency_seconds_sum", cost_latency.get("replay_latency_seconds_sum", 0.0)
            ),
            "original_cached_latency_seconds_sum": provider_summary.get(
                "original_cached_latency_seconds_sum",
                cost_latency.get("original_cached_latency_seconds_sum", 0.0),
            ),
            "latency_p50_seconds": provider_summary.get(
                "latency_p50_seconds", cost_latency.get("latency_p50_seconds", cost_latency.get("latency_p50", 0.0))
            ),
            "latency_p95_seconds": provider_summary.get(
                "latency_p95_seconds", cost_latency.get("latency_p95_seconds", cost_latency.get("latency_p95", 0.0))
            ),
            "original_live_latency_p50_seconds": provider_summary.get(
                "original_live_latency_p50_seconds",
                cost_latency.get("original_live_latency_p50_seconds", 0.0),
            ),
            "original_live_latency_p95_seconds": provider_summary.get(
                "original_live_latency_p95_seconds",
                cost_latency.get("original_live_latency_p95_seconds", 0.0),
            ),
            "latency_p50": provider_summary.get("latency_p50", cost_latency.get("latency_p50", 0.0)),
            "latency_p95": provider_summary.get("latency_p95", cost_latency.get("latency_p95", 0.0)),
        }
    )
    return cost_latency


def _provider_record(
    response: LLMResponse,
    *,
    live_estimated_cost: float = 0.0,
    replay_estimated_cost: float = 0.0,
) -> dict[str, Any]:
    usage = dict(response.usage)
    metadata = dict(response.metadata)
    original_latency_seconds = float(metadata.get("original_latency_seconds") or response.latency_seconds or 0.0)
    replay_latency_seconds = float(metadata.get("replay_latency_seconds") or (response.latency_seconds if response.cache_hit else 0.0) or 0.0)
    original_cached_cost_usd = (
        float(metadata.get("original_estimated_cost_usd") or 0.0)
        if response.cache_hit
        else 0.0
    )
    return {
        "provider": response.provider,
        "model": response.model,
        "cache_hit": response.cache_hit,
        "latency_seconds": response.latency_seconds,
        "original_latency_seconds": original_latency_seconds,
        "replay_latency_seconds": replay_latency_seconds,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "prompt_cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens") or 0),
        "prompt_cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens") or 0),
        "estimated_cost": live_estimated_cost if not response.cache_hit else replay_estimated_cost,
        "live_cost_usd": live_estimated_cost if not response.cache_hit else 0.0,
        "replay_cost_usd": replay_estimated_cost if response.cache_hit else 0.0,
        "original_cached_cost_usd": original_cached_cost_usd,
        "effective_cost_usd": live_estimated_cost + original_cached_cost_usd,
        "source": metadata.get("source") or metadata.get("cache_source") or ("cache" if response.cache_hit else "live"),
    }


def _response_cache_payload(response: LLMResponse) -> dict[str, Any]:
    return {
        "text": response.text,
        "raw_response": response.raw_response,
        "usage": dict(response.usage),
        "latency_seconds": response.latency_seconds,
        "model": response.model,
        "provider": response.provider,
        "metadata": dict(response.metadata),
    }


def _response_from_cache(
    payload: dict[str, Any],
    *,
    cache_key: str,
    replay_latency_seconds: float = 0.0,
) -> LLMResponse:
    usage = {
        key: int(value)
        for key, value in dict(payload.get("usage") or {}).items()
        if isinstance(value, int)
    }
    metadata = dict(payload.get("metadata") or {})
    original_estimated_cost = float(
        metadata.get("original_estimated_cost_usd")
        or metadata.get("estimated_cost")
        or 0.0
    )
    original_latency_seconds = float(
        metadata.get("original_latency_seconds")
        or payload.get("latency_seconds")
        or 0.0
    )
    metadata.update(
        {
            "cache_key": cache_key,
            "source": "cache",
            "cache_source": "response_cache",
            "estimated_cost": 0.0,
            "replay_estimated_cost_usd": 0.0,
            "original_estimated_cost_usd": original_estimated_cost,
            "replay_latency_seconds": float(replay_latency_seconds or 0.0),
            "original_latency_seconds": original_latency_seconds,
            "original_usage": dict(usage),
            "prompt_cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens") or 0),
            "prompt_cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens") or 0),
        }
    )
    return LLMResponse(
        text=str(payload.get("text") or ""),
        raw_response=dict(payload.get("raw_response") or {}),
        usage=usage,
        latency_seconds=float(replay_latency_seconds or 0.0),
        model=str(payload.get("model") or ""),
        provider=str(payload.get("provider") or ""),
        cache_hit=True,
        metadata=metadata,
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _unique_ints(values: list[int]) -> list[int]:
    output = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _is_tbd_text(value: Any) -> bool:
    text = str(value or "").strip()
    return not text or text.upper() == "TBD" or "TBD" in text


def _llm_config(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("config.llm is required for LLM methods")
    if llm.get("config_path"):
        loaded = load_config(str(llm["config_path"]))
        merged = dict(loaded)
        merged.update({key: value for key, value in llm.items() if key != "config_path"})
        return merged
    return llm


def _resolve_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    method = resolved.get("method")
    if isinstance(method, dict) and method.get("config_path"):
        source_path = str(method["config_path"])
        merged = load_config(source_path)
        merged.update({key: value for key, value in method.items() if key != "config_path"})
        merged["config_path"] = source_path
        resolved["method"] = merged
    llm = resolved.get("llm")
    if isinstance(llm, dict) and llm.get("config_path"):
        source_path = str(llm["config_path"])
        merged = load_config(source_path)
        merged.update({key: value for key, value in llm.items() if key != "config_path"})
        merged["config_path"] = source_path
        resolved["llm"] = merged
    training = resolved.get("training")
    if isinstance(training, dict) and training.get("config_path"):
        source_path = str(training["config_path"])
        merged = load_config(source_path)
        merged.update({key: value for key, value in training.items() if key != "config_path"})
        merged["config_path"] = source_path
        resolved["training"] = merged
    return resolved


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training")
    if not isinstance(training, dict):
        return {}
    if training.get("config_path"):
        loaded = load_config(str(training["config_path"]))
        loaded.update({key: value for key, value in training.items() if key != "config_path"})
        return loaded
    return training


def _required_llm_text(llm_config: dict[str, Any], key: str) -> str:
    value = llm_config.get(key)
    text = str(value or "").strip()
    if not text or text.casefold() in {"none", "null"}:
        raise ValueError(f"config.llm.{key} is required for provider={llm_config.get('provider')}")
    return text


def _method_config(config: dict[str, Any]) -> dict[str, Any]:
    method = config.get("method")
    if isinstance(method, dict) and method.get("config_path"):
        loaded = load_config(str(method["config_path"]))
        loaded.update({key: value for key, value in method.items() if key != "config_path"})
        loaded["config_path"] = str(method["config_path"])
        return loaded
    return method if isinstance(method, dict) else {"name": method}


def _method_name(config: dict[str, Any]) -> str:
    method = config.get("method")
    if isinstance(method, dict):
        return str(method.get("name") or method.get("type") or "")
    return str(method or "")


def _run_name(config: dict[str, Any], method_name: str) -> str:
    output = config.get("output") if isinstance(config.get("output"), dict) else {}
    if output.get("run_name"):
        return str(output["run_name"])
    if config.get("run_name"):
        return str(config["run_name"])
    return f"smoke_{method_name}"


def _output_dir(config: dict[str, Any]) -> str:
    output = config.get("output") if isinstance(config.get("output"), dict) else {}
    return str(output.get("output_dir") or config.get("output_dir") or "outputs/runs")


def _config_for_baseline(config: dict[str, Any], baseline: str, *, seed: int | None = None) -> dict[str, Any]:
    child = json.loads(json.dumps(config))
    child.pop("baselines", None)
    child.pop("seeds", None)
    resolved_seed = int(seed if seed is not None else config.get("seed") or 0)
    child["seed"] = resolved_seed
    method_config = _named_method_config(baseline)
    if method_config is None:
        method_config = {
            "name": baseline,
            "type": _default_method_type_for(baseline),
            "seed": resolved_seed,
            "params": _default_params_for(baseline),
        }
    method_config["seed"] = resolved_seed
    child["method"] = method_config
    output = child.get("output") if isinstance(child.get("output"), dict) else {}
    output["run_name"] = _run_name_for_child(config, baseline)
    output["output_dir"] = str(output.get("output_dir") or child.get("output_dir") or "outputs/runs")
    child["output"] = output
    child["run_name"] = output["run_name"]
    child["output_dir"] = output["output_dir"]
    return child


def _default_params_for(method: str) -> dict[str, Any]:
    if method == "bm25":
        return {"text_policy": "title", "k1": 1.5, "b": 0.75}
    if method == "mf":
        return {"factors": 4, "epochs": 20, "learning_rate": 0.05, "regularization": 0.001}
    if method in {"sequential_last_item", "sequential_markov", "sasrec_interface", "sequential_interface"}:
        return {"max_history_length": 5}
    if method in {"llm_generative", "llm_generative_mock"}:
        return {"prediction_method": "llm_generative_mock", "text_policy": "title"}
    if method == "llm_rerank":
        return {"prediction_method": "llm_rerank_mock", "text_policy": "title"}
    if method == "llm_confidence_observation":
        return {"prediction_method": "llm_confidence_observation_mock", "text_policy": "title"}
    return {}


def _default_method_type_for(method: str) -> str:
    if method.startswith("ours_"):
        return "ours_method"
    if method.startswith("llm_"):
        return "llm"
    if method in SEQUENTIAL_METHODS:
        return "sequential"
    if method == "lora_dry_run":
        return "training"
    return "ranker"


def _default_run_name_for(method: str) -> str:
    if method == "sequential_last_item":
        return "smoke_sequential_last_item"
    if method == "sequential_markov":
        return "smoke_sequential_markov"
    if method == "sasrec_interface":
        return "smoke_sasrec_interface"
    if method == "lora_dry_run":
        return "smoke_lora_dry_run"
    if method in {"llm_generative", "llm_generative_mock"}:
        return "smoke_llm_generative"
    if method == "llm_rerank":
        return "smoke_llm_rerank"
    if method == "llm_confidence_observation":
        return "smoke_llm_confidence"
    return f"smoke_all_{method}"


def _is_ours_method(method: str, method_config: dict[str, Any]) -> bool:
    return (
        str(method_config.get("type") or "") == "ours_method"
        or str(method_config.get("type") or "") == "ours_method_policy_refinement"
        or method == "ours_uncertainty_guided"
        or method == "ours_conservative_uncertainty_gate"
        or method.startswith("ours_ablation_")
        or method == "ours_fallback_only"
    )


def _named_method_config(name: str) -> dict[str, Any] | None:
    path = Path("configs") / "methods" / f"{name}.yaml"
    if not path.exists():
        return None
    return load_config(path)


def _run_name_for_child(config: dict[str, Any], method: str) -> str:
    parent_name = str(
        ((config.get("output") or {}) if isinstance(config.get("output"), dict) else {}).get("run_name")
        or config.get("run_name")
        or ""
    )
    suffix = _child_run_suffix(method)
    if parent_name == "smoke_ours_ablation":
        return f"smoke_ours_ablation_{suffix}"
    if parent_name == "smoke_phase6_all":
        return f"smoke_phase6_{suffix}"
    if parent_name.startswith(("pilot_", "real_", "r2_", "r3_", "r3b_")):
        return f"{parent_name}_{suffix}"
    return _default_run_name_for(method)


def _child_run_suffix(method: str) -> str:
    mapping = {
        "ours_uncertainty_guided": "ours_full",
        "ours_ablation_no_uncertainty": "ours_no_uncertainty",
        "ours_ablation_no_grounding": "ours_no_grounding",
        "ours_ablation_no_candidate_normalized_confidence": "ours_no_candidate_normalized_confidence",
        "ours_ablation_no_popularity_adjustment": "ours_no_popularity_adjustment",
        "ours_ablation_no_echo_guard": "ours_no_echo_guard",
        "ours_fallback_only": "ours_fallback_only",
        "llm_generative": "llm_generative",
        "llm_rerank": "llm_rerank",
        "bm25": "bm25",
        "popularity": "popularity",
        "mf": "mf",
        "sequential_markov": "sequential_markov",
    }
    return mapping.get(method, method)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_decode_csv_row(row) for row in csv.DictReader(handle)]


def _decode_csv_row(row: dict[str, str]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            decoded[key] = value
            continue
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                decoded[key] = json.loads(stripped)
                continue
            except json.JSONDecodeError:
                pass
        decoded[key] = value
    return decoded


def _evaluation_context_for_config(config: dict[str, Any]) -> dict[str, Any]:
    processed_dir = Path(_required_path(config, "dataset", "processed_dir"))
    examples = read_jsonl(processed_dir / "examples.jsonl")
    return {
        "item_catalog": _read_csv(processed_dir / "items.csv"),
        "all_examples": examples,
        "train_examples": [row for row in examples if str(row.get("split")) == "train"],
    }


def _seeds_for_config(config: dict[str, Any]) -> list[int]:
    seeds = config.get("seeds")
    if isinstance(seeds, list) and seeds:
        return [int(seed) for seed in seeds]
    return [int(config.get("seed") or 0)]


def _candidate_sizes_for_config(config: dict[str, Any]) -> list[int | None]:
    sizes = config.get("candidate_sizes")
    if isinstance(sizes, list) and sizes:
        return [int(size) for size in sizes]
    return [None]


def _config_for_candidate_size(config: dict[str, Any], candidate_size: int | None) -> dict[str, Any]:
    child = json.loads(json.dumps(config))
    if candidate_size is None:
        return child
    child["active_candidate_size"] = candidate_size
    candidate = child.get("candidate") if isinstance(child.get("candidate"), dict) else {}
    candidate["size"] = candidate_size
    candidate["k"] = candidate_size
    candidate["sample_size"] = candidate_size
    child["candidate"] = candidate

    dataset = child.get("dataset") if isinstance(child.get("dataset"), dict) else {}
    processed_dir = _candidate_processed_dir(child, candidate_size)
    dataset["processed_dir"] = processed_dir
    dataset["candidate_size"] = candidate_size
    child["dataset"] = dataset
    candidate["set_path"] = str(Path(processed_dir) / "candidate_sets.jsonl")

    data_protocol = child.get("data_protocol") if isinstance(child.get("data_protocol"), dict) else {}
    data_protocol["candidate_set_saved_path"] = candidate["set_path"]
    child["data_protocol"] = data_protocol

    base_run_name = str(
        ((config.get("output") or {}) if isinstance(config.get("output"), dict) else {}).get("run_name")
        or config.get("run_name")
        or "candidate_sensitivity"
    )
    output = child.get("output") if isinstance(child.get("output"), dict) else {}
    output["run_name"] = f"{base_run_name}_candidate{candidate_size}"
    output["output_dir"] = str(output.get("output_dir") or child.get("output_dir") or "outputs/runs")
    child["output"] = output
    child["run_name"] = output["run_name"]
    child["output_dir"] = output["output_dir"]
    return child


def _candidate_processed_dir(config: dict[str, Any], candidate_size: int) -> str:
    sensitivity = config.get("candidate_sensitivity") if isinstance(config.get("candidate_sensitivity"), dict) else {}
    template = str(
        sensitivity.get("processed_dir_template")
        or "data/processed/movielens_1m/r2_candidate_sensitivity/candidate_{candidate_size}"
    )
    return template.format(candidate_size=candidate_size)


def _dataset_config_for_experiment(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    dataset_config = load_config(_required_path(config, "dataset", "config_path"))
    dataset_config.update({key: value for key, value in dataset.items() if key != "config_path"})
    candidate = config.get("candidate") if isinstance(config.get("candidate"), dict) else {}
    if candidate:
        dataset_candidate = dict(dataset_config.get("candidate") or {})
        if candidate.get("protocol") is not None:
            dataset_candidate["protocol"] = candidate.get("protocol")
        sample_size = candidate.get("sample_size") or candidate.get("size") or candidate.get("k")
        if sample_size not in (None, ""):
            dataset_candidate["sample_size"] = int(sample_size)
        dataset_config["candidate"] = dataset_candidate
    dataset_config["seed"] = int(config.get("seed") or dataset_config.get("seed") or 0)
    return dataset_config


def _postprocess_run_all(config: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    del runs
    output = config.get("output") if isinstance(config.get("output"), dict) else {}
    runs_dir = Path(str(output.get("output_dir") or config.get("output_dir") or "outputs/runs"))
    tables_config = config.get("tables") if isinstance(config.get("tables"), dict) else {}
    aggregate_config = config.get("aggregate") if isinstance(config.get("aggregate"), dict) else {}
    table_output_dir = Path(str(tables_config.get("output_dir") or aggregate_config.get("output_dir") or "outputs/tables"))
    postprocess: dict[str, Any] = {}
    if bool(aggregate_config.get("enabled", False)):
        postprocess["aggregation"] = aggregate_run_metrics(runs_dir, output_dir=table_output_dir)
    if bool(tables_config.get("enabled", False)):
        postprocess["tables"] = export_phase5_tables(runs_dir, output_dir=table_output_dir)
    return postprocess


def _completed_run_result(config: dict[str, Any]) -> dict[str, Any] | None:
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    resume = llm.get("resume") if isinstance(llm.get("resume"), dict) else {}
    if resume.get("enabled") is not True:
        return None
    seed = int(config.get("seed") or 0)
    run_name = _run_name(config, _method_name(config))
    run_id = f"{run_name}_seed{seed}"
    run_dir = Path(_output_dir(config)) / run_id
    required = [
        "resolved_config.yaml",
        "environment.json",
        "logs.txt",
        "predictions.jsonl",
        "metrics.json",
        "metrics.csv",
        "cost_latency.json",
    ]
    if not all((run_dir / name).exists() for name in required):
        return None
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "predictions": str(run_dir / "predictions.jsonl"),
        "metrics": metrics,
        "resumed_from_complete_artifacts": True,
    }


def _required_path(config: dict[str, Any], section: str, key: str) -> str:
    value = (config.get(section) or {}).get(key)
    if not value:
        raise ValueError(f"config.{section}.{key} is required")
    return str(value)
