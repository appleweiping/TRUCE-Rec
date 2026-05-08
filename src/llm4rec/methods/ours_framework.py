"""TRUCE-native training data builders for the original Ours framework.

These helpers turn same-candidate recommendation examples into supervision for
an uncertainty-aware generative/reranking adapter. They do not train a model;
they create auditable SFT and scoring contracts for server-side Qwen3 adapter
training.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OursTrainingRow:
    messages: list[dict[str, str]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateEvidence:
    preference_evidence: float
    confidence: float
    grounding_risk: float
    popularity_bias_risk: float
    history_repetition_risk: float
    popularity_bucket: str
    contrast_role: str
    uncertainty_reason: str


def item_text(item_id: str, item_lookup: dict[str, dict[str, Any]]) -> str:
    row = item_lookup.get(str(item_id), {})
    title = str(row.get("title") or row.get("raw_text") or item_id)
    category = str(row.get("category") or "")
    brand = str(row.get("brand") or "")
    bits = [title]
    if category:
        bits.append(f"category={category}")
    if brand:
        bits.append(f"brand={brand}")
    return " | ".join(bits)


def history_text(example: dict[str, Any], item_lookup: dict[str, dict[str, Any]], *, max_history: int = 50) -> str:
    history = [str(x) for x in example.get("history") or example.get("history_item_ids") or []][-max_history:]
    if not history:
        return "(empty)"
    return " ; ".join(item_text(item_id, item_lookup) for item_id in history)


def popularity_bucket(item_id: str, train_popularity: dict[str, int]) -> str:
    if not train_popularity:
        return "unknown"
    counts = sorted(train_popularity.values())
    value = int(train_popularity.get(str(item_id), 0))
    if not counts or value <= 0:
        return "tail"
    q80 = counts[max(0, int(0.8 * (len(counts) - 1)))]
    q50 = counts[max(0, int(0.5 * (len(counts) - 1)))]
    if value >= q80:
        return "head"
    if value >= q50:
        return "mid"
    return "tail"


def build_pairwise_prompt(
    example: dict[str, Any],
    *,
    candidate_item_id: str,
    item_lookup: dict[str, dict[str, Any]],
    train_popularity: dict[str, int],
    max_history: int = 50,
) -> str:
    """Prompt for candidate-level acceptance with uncertainty-aware evidence."""

    candidate = item_text(candidate_item_id, item_lookup)
    bucket = popularity_bucket(candidate_item_id, train_popularity)
    return (
        "TRUCE uncertainty-aware recommendation task.\n"
        "Estimate whether the candidate should be accepted for the user's next interaction.\n"
        "Use user preference evidence, catalog grounding, popularity/long-tail risk, and history repetition risk.\n"
        f"User history: {history_text(example, item_lookup, max_history=max_history)}\n"
        f"Candidate item: {candidate}\n"
        f"Candidate item id: {candidate_item_id}\n"
        f"Train-popularity bucket: {bucket}\n"
        "Answer exactly in JSON with keys: accept, confidence, preference_evidence, "
        "grounding_risk, popularity_bias_risk, history_repetition_risk, uncertainty_reason."
    )


def build_pairwise_answer(*, is_positive: bool, evidence: CandidateEvidence) -> str:
    import json

    accept = "true" if is_positive else "false"
    payload = {
        "accept": accept == "true",
        "confidence": evidence.confidence,
        "preference_evidence": evidence.preference_evidence,
        "grounding_risk": evidence.grounding_risk,
        "popularity_bias_risk": evidence.popularity_bias_risk,
        "history_repetition_risk": evidence.history_repetition_risk,
        "uncertainty_reason": evidence.uncertainty_reason,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_listwise_prompt(
    example: dict[str, Any],
    *,
    candidate_item_ids: list[str],
    item_lookup: dict[str, dict[str, Any]],
    train_popularity: dict[str, int],
    max_history: int = 50,
) -> str:
    lines = [
        "TRUCE listwise reranking task.",
        "Rank the candidate IDs for the user's next interaction.",
        "Prefer grounded preference evidence, but avoid blindly over-trusting popular or history-repetitive items.",
        f"User history: {history_text(example, item_lookup, max_history=max_history)}",
        "Candidates:",
    ]
    for item_id in candidate_item_ids:
        lines.append(
            f"- {item_id}: {item_text(item_id, item_lookup)} | train-popularity={popularity_bucket(item_id, train_popularity)}"
        )
    lines.append("Return JSON with keys: ranked_item_ids, confidence_by_item, risk_notes.")
    return "\n".join(lines)


def build_listwise_answer(
    *,
    target_item_id: str,
    candidate_item_ids: list[str],
    item_lookup: dict[str, dict[str, Any]] | None = None,
    train_popularity: dict[str, int] | None = None,
    example: dict[str, Any] | None = None,
) -> str:
    ranked = [target_item_id] + [item for item in candidate_item_ids if item != target_item_id]
    confidence: dict[str, float] = {}
    risk_notes: dict[str, str] = {}
    for item in ranked:
        evidence = candidate_evidence(
            example or {},
            candidate_item_id=item,
            target_item_id=target_item_id,
            item_lookup=item_lookup or {},
            train_popularity=train_popularity or {},
        )
        confidence[item] = evidence.confidence
        risk_notes[item] = evidence.contrast_role
    return (
        '{"ranked_item_ids": '
        + _json_list(ranked)
        + ', "confidence_by_item": '
        + _json_float_map(confidence)
        + ', "risk_notes": '
        + _json_string_map(risk_notes)
        + "}"
    )


def build_training_rows(
    example: dict[str, Any],
    *,
    item_lookup: dict[str, dict[str, Any]],
    train_popularity: dict[str, int],
    negatives_per_example: int = 15,
    include_listwise: bool = True,
    max_history: int = 50,
) -> list[OursTrainingRow]:
    target = str(example.get("target") or example.get("target_item") or "")
    candidates = [str(item) for item in example.get("candidates") or example.get("candidate_items") or []]
    negatives = _stratified_negatives(
        example,
        candidates=[item for item in candidates if item != target],
        item_lookup=item_lookup,
        train_popularity=train_popularity,
        k=negatives_per_example,
        key=str(example.get("example_id") or ""),
    )
    selected = ([target] if target else []) + negatives
    rows: list[OursTrainingRow] = []
    for item_id in selected:
        is_positive = item_id == target
        evidence = candidate_evidence(
            example,
            candidate_item_id=item_id,
            target_item_id=target,
            item_lookup=item_lookup,
            train_popularity=train_popularity,
        )
        rows.append(
            OursTrainingRow(
                messages=[
                    {
                        "role": "user",
                        "content": build_pairwise_prompt(
                            example,
                            candidate_item_id=item_id,
                            item_lookup=item_lookup,
                            train_popularity=train_popularity,
                            max_history=max_history,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": build_pairwise_answer(
                            is_positive=is_positive,
                            evidence=evidence,
                        ),
                    },
                ],
                metadata={
                    "example_id": str(example.get("example_id") or ""),
                    "target_item_id": target,
                    "candidate_item_id": item_id,
                    "supervision_type": "pairwise_acceptance",
                    "is_positive": is_positive,
                    "popularity_bucket": evidence.popularity_bucket,
                    "contrast_role": evidence.contrast_role,
                    "feature_targets": {
                        "preference_evidence": evidence.preference_evidence,
                        "confidence": evidence.confidence,
                        "grounding_risk": evidence.grounding_risk,
                        "popularity_bias_risk": evidence.popularity_bias_risk,
                        "history_repetition_risk": evidence.history_repetition_risk,
                    },
                },
            )
        )
    if include_listwise and target and candidates:
        panel = _listwise_panel(candidates, target=target, train_popularity=train_popularity, key=str(example.get("example_id") or ""))
        rows.append(
            OursTrainingRow(
                messages=[
                    {
                        "role": "user",
                        "content": build_listwise_prompt(
                            example,
                            candidate_item_ids=panel,
                            item_lookup=item_lookup,
                            train_popularity=train_popularity,
                            max_history=max_history,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": build_listwise_answer(
                            target_item_id=target,
                            candidate_item_ids=panel,
                            item_lookup=item_lookup,
                            train_popularity=train_popularity,
                            example=example,
                        ),
                    },
                ],
                metadata={
                    "example_id": str(example.get("example_id") or ""),
                    "target_item_id": target,
                    "candidate_item_ids": panel,
                    "supervision_type": "listwise_target_first",
                },
            )
        )
    return rows


def build_score_row(
    example: dict[str, Any],
    *,
    candidate_item_id: str,
    item_lookup: dict[str, dict[str, Any]],
    train_popularity: dict[str, int],
    max_history: int = 50,
) -> dict[str, Any]:
    return {
        "example_id": str(example.get("example_id") or ""),
        "user_id": str(example.get("user_id") or ""),
        "prompt": build_pairwise_prompt(
            example,
            candidate_item_id=candidate_item_id,
            item_lookup=item_lookup,
            train_popularity=train_popularity,
            max_history=max_history,
        ),
        "candidate_item_ids": [str(candidate_item_id)],
        "candidate_outputs": ['{"accept": true'],
        "metadata": {
            "event_id": _metadata_value(example, "event_id", str(example.get("example_id") or "")),
            "source_event_id": _metadata_value(example, "source_event_id", str(example.get("example_id") or "")),
            "supervision_type": "pairwise_acceptance_score",
        },
        "scoring_contract": "Score the likelihood that the adapter accepts this candidate under the TRUCE uncertainty-aware prompt.",
    }


def candidate_evidence(
    example: dict[str, Any],
    *,
    candidate_item_id: str,
    target_item_id: str,
    item_lookup: dict[str, dict[str, Any]],
    train_popularity: dict[str, int],
) -> CandidateEvidence:
    """Compute deterministic uncertainty targets from catalog/train evidence."""

    candidate_item_id = str(candidate_item_id)
    target_item_id = str(target_item_id)
    is_positive = candidate_item_id == target_item_id
    bucket = popularity_bucket(candidate_item_id, train_popularity)
    grounding_risk = _grounding_risk(candidate_item_id, item_lookup)
    history_risk = _history_repetition_risk(example, candidate_item_id, item_lookup)
    popularity_risk = {"head": 0.70, "mid": 0.35, "tail": 0.12, "unknown": 0.25}.get(bucket, 0.25)
    if is_positive:
        preference = _clip(0.66 + 0.20 * (1.0 - grounding_risk) + 0.14 * history_risk - 0.06 * popularity_risk)
        confidence = _clip(preference - 0.08 * grounding_risk - (0.04 if bucket == "tail" else 0.0), low=0.50, high=0.95)
        role = "positive_next_item"
        reason = f"positive target with {bucket} popularity and history_repetition={history_risk:.2f}"
    else:
        preference = _clip(0.08 + 0.24 * history_risk + 0.18 * popularity_risk + 0.05 * (1.0 - grounding_risk), high=0.75)
        confidence = _clip(preference, low=0.03, high=0.70)
        role = _negative_role(bucket=bucket, history_risk=history_risk)
        reason = f"{role} with {bucket} popularity and history_repetition={history_risk:.2f}"
    return CandidateEvidence(
        preference_evidence=round(preference, 4),
        confidence=round(confidence, 4),
        grounding_risk=round(grounding_risk, 4),
        popularity_bias_risk=round(popularity_risk, 4),
        history_repetition_risk=round(history_risk, 4),
        popularity_bucket=bucket,
        contrast_role=role,
        uncertainty_reason=reason,
    )


def _metadata_value(row: dict[str, Any], key: str, default: str = "") -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get(key) or meta.get(key) or default)


def _stable_sample(items: list[str], *, k: int, key: str) -> list[str]:
    if k < 0 or len(items) <= k:
        return list(items)
    ranked = sorted(items, key=lambda item: hashlib.sha256(f"{key}:{item}".encode()).hexdigest())
    return ranked[:k]


def _stratified_negatives(
    example: dict[str, Any],
    *,
    candidates: list[str],
    item_lookup: dict[str, dict[str, Any]],
    train_popularity: dict[str, int],
    k: int,
    key: str,
) -> list[str]:
    if k < 0 or len(candidates) <= k:
        return list(candidates)
    buckets: dict[str, list[str]] = {
        "history_echo_probe": [],
        "head_bias_probe": [],
        "tail_underconfidence_probe": [],
        "mid_popularity_probe": [],
        "stable_negative": [],
    }
    for item in candidates:
        evidence = candidate_evidence(
            example,
            candidate_item_id=item,
            target_item_id=str(example.get("target") or example.get("target_item") or ""),
            item_lookup=item_lookup,
            train_popularity=train_popularity,
        )
        buckets.setdefault(evidence.contrast_role, []).append(item)
    chosen: list[str] = []
    for role in ["history_echo_probe", "head_bias_probe", "tail_underconfidence_probe", "mid_popularity_probe", "stable_negative"]:
        for item in _stable_sample(buckets.get(role, []), k=1, key=f"{key}:{role}"):
            if item not in chosen:
                chosen.append(item)
        if len(chosen) >= k:
            return chosen[:k]
    for item in _stable_sample(candidates, k=len(candidates), key=f"{key}:fill"):
        if item not in chosen:
            chosen.append(item)
        if len(chosen) >= k:
            break
    return chosen[:k]


def _listwise_panel(items: list[str], *, target: str, train_popularity: dict[str, int], key: str, size: int = 12) -> list[str]:
    chosen = [target] if target in items else []
    rest = [item for item in items if item != target]
    rest.sort(key=lambda item: (-int(train_popularity.get(item, 0)), hashlib.sha256(f"{key}:head:{item}".encode()).hexdigest()))
    chosen.extend(rest[: max(0, size // 2 - len(chosen))])
    tail = sorted(rest, key=lambda item: (int(train_popularity.get(item, 0)), hashlib.sha256(f"{key}:tail:{item}".encode()).hexdigest()))
    for item in tail:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) >= size:
            break
    return chosen[:size]


def _json_list(values: list[str]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False)


def _json_float_map(values: dict[str, float]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def _json_string_map(values: dict[str, str]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def _grounding_risk(item_id: str, item_lookup: dict[str, dict[str, Any]]) -> float:
    row = item_lookup.get(str(item_id))
    if not row:
        return 0.85
    title = str(row.get("title") or row.get("raw_text") or "").strip()
    if not title or title == str(item_id):
        return 0.35
    return 0.05


def _history_repetition_risk(example: dict[str, Any], item_id: str, item_lookup: dict[str, dict[str, Any]]) -> float:
    history = [str(x) for x in example.get("history") or example.get("history_item_ids") or []]
    if not history:
        return 0.0
    if str(item_id) in history:
        return 1.0
    candidate = item_lookup.get(str(item_id), {})
    candidate_category = str(candidate.get("category") or "").lower()
    candidate_brand = str(candidate.get("brand") or "").lower()
    candidate_tokens = _title_tokens(candidate)
    score = 0.0
    for hist_item in history[-50:]:
        hist = item_lookup.get(hist_item, {})
        if candidate_category and candidate_category == str(hist.get("category") or "").lower():
            score += 0.55
        if candidate_brand and candidate_brand == str(hist.get("brand") or "").lower():
            score += 0.25
        overlap = candidate_tokens & _title_tokens(hist)
        if overlap:
            score += min(0.20, 0.04 * len(overlap))
    return _clip(score / max(1, len(history[-50:])))


def _title_tokens(row: dict[str, Any]) -> set[str]:
    raw = str(row.get("title") or row.get("raw_text") or "")
    return {token for token in raw.lower().replace("|", " ").replace("/", " ").split() if len(token) > 2}


def _negative_role(*, bucket: str, history_risk: float) -> str:
    if history_risk >= 0.45:
        return "history_echo_probe"
    if bucket == "head":
        return "head_bias_probe"
    if bucket == "tail":
        return "tail_underconfidence_probe"
    if bucket == "mid":
        return "mid_popularity_probe"
    return "stable_negative"


def _clip(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
