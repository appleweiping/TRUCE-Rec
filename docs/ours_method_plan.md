# Calibrated Uncertainty-Guided Generative Recommendation

> **⚠️ SUPERSEDED for the main method (2026-06-07).** The generate-title → ground → uncertainty-gate
> route described below **lost to fallback-only** in the R3 formal run
> (`docs/r3_ours_error_decomposition.md`) and is **retired** as the paper's method. The current main
> method is **SCALR** (Set-aware Calibrated Lift Ranking) — see
> **`docs/method_redesign_decision.md`** (tri-agent ARIS design, passed the ≥8/10 gate). This file
> is kept for the design history, the shared fairness/leakage rules (§16), the metrics list (§19),
> and the required-ablations scaffold (§17), which still apply. Read the decision doc for the live
> method.

## 1. Method name

Calibrated Uncertainty-Guided Generative Recommendation, abbreviated in configs as
`ours_uncertainty_guided`.

This is a Phase 6 integration method. It is not yet validated as a paper result.

## 2. Problem setting

Given a user's observed history before the evaluation target, the method asks an
LLM to generate one recommendation title, grounds that title to a catalog item,
and uses uncertainty and recommendation-specific risk signals to decide whether
to accept the grounded item, abstain, rerank, or fall back to a comparable
candidate ranker.

The task remains next-item recommendation over the same prediction schema and
shared evaluator used by baselines.

## 3. Inputs

- User ID.
- Training-only item catalog and item metadata.
- User history available in the example.
- Candidate item IDs supplied by the experiment protocol.
- Train-split popularity statistics.
- Configured MockLLM provider in Phase 6 smoke mode.
- Configured fallback ranker: `bm25`, `popularity`, or `sequential_markov`.
- Configured uncertainty thresholds and ablation flags.

## 4. Outputs

The method emits the unified prediction JSONL schema:

- `user_id`
- `target_item`
- `candidate_items`
- `predicted_items`
- `scores`
- `method`
- `domain`
- `raw_output`
- `metadata`

Metadata must record generated title, confidence, grounding result, uncertainty
decision, fallback method, echo risk, popularity bucket, history similarity,
ablation variant, disabled components, parse status, prompt template ID, and
prompt hash.

## 5. Allowed signals

- History titles and item IDs from the example history only.
- Visible candidate titles after removing the target item from the prompt.
- Generated title and parsed confidence from the LLM response.
- Catalog-only grounding score and grounded item ID.
- Candidate-normalized confidence computed over visible non-target alternatives.
- Train-split item popularity and popularity bucket.
- Similarity between generated or grounded item title and history titles.
- Fallback ranker scores over the same candidate set.
- Provider metadata such as latency and token usage from the mock provider.

## 6. Forbidden signals

- Target title in any prompt.
- Target item ID in any prompt.
- Future interactions beyond the example history.
- Test or validation popularity when computing popularity buckets.
- Ground-truth target label when computing confidence, policy decisions, or
  fallback ordering.
- External API responses in Phase 6 smoke runs.
- Downloaded HF models or real LoRA/QLoRA training outputs.
- Any post-hoc decision that depends on whether the prediction hit the target.

## 7. Inference pipeline

1. Build a generative title prompt from user history and visible candidates,
   excluding the target item from the prompt context.
2. Call the configured MockLLM provider.
3. Parse the generated title and confidence.
4. Ground the generated title to the catalog.
5. Optionally compute candidate-normalized confidence using a second mock prompt.
6. Compute train-popularity bucket and history-similarity / echo-risk signals.
7. Run the uncertainty policy using only allowed signals.
8. Accept the grounded item when confidence and grounding are sufficient and no
   enabled guard redirects the decision.
9. Fall back to the configured ranker, rerank, or abstain when policy requires.
10. Emit one unified prediction record and full audit metadata.

## 8. Uncertainty signals used

Phase 6 used a minimal subset:

- Parsed generative confidence.
- Grounding success and grounding score.
- Candidate-normalized confidence when enabled.
- Confidence adjusted by popularity risk when enabled.
- History similarity and echo-risk flag when enabled.

These are infrastructure signals, not evidence of empirical effectiveness.

The current Ours adapter path has been upgraded to
`truce_observation_residual_policy_sft_v2`. It still starts from the same
observation object, but it now supervises a policy target rather than only an
acceptance target:

- candidate-normalized utility within the fixed candidate panel;
- popularity-residual utility to reduce head-item familiarity bias;
- harm risk from popularity bias, grounding risk, and echo/history repetition;
- abstain risk for ambiguous or poorly grounded candidates;
- conservative `promote`, `suppress`, or `defer_to_fallback` action labels.

At score time the server estimates each candidate's
`{"policy_action": "promote"}` likelihood and writes the usual
`candidate_scores.csv`. This preserves the evaluator contract while making the
method closer to an observation-driven policy learner than a prompt-only
reranker.

## 9. Grounding strategy

Generated titles are grounded against the catalog only. The grounding strategy is:

- case-insensitive exact title match;
- normalized title match;
- token-overlap match above a configured minimum score;
- failure if no catalog title satisfies the threshold.

Grounding does not use target labels. Grounding failure triggers fallback or
abstention according to config.

## 10. Confidence calibration strategy

Phase 6 implemented config-driven decision calibration rather than learned
calibration:

- absolute confidence threshold for accepting a grounded item;
- minimum grounding score threshold;
- optional candidate-normalized confidence threshold;
- optional popularity adjustment for high-confidence head items;
- optional echo-risk guard for high-history-similarity items.

Thresholds are provisional and smoke-only. No calibrated-performance claim is
allowed until real experiments produce metrics.

For the Qwen3 adapter lane, the v2 objective introduces learned calibration
targets through the adapter data. These are still deterministic train/valid
targets until real observation diagnostics and validation tuning are available;
they are not evidence of effectiveness by themselves.

## 11. Popularity-bias handling

The method computes popularity from train examples only and buckets items into
`head`, `mid`, `tail`, or `unknown`. When enabled, high-confidence grounded head
items receive a risk flag. The smoke policy conservatively falls back or reranks
when the overconfidence flag is active, depending on config.

This is a bias-handling mechanism to test, not a proven mitigation.

## 12. Echo-chamber risk handling

History similarity is computed between the generated or grounded title and
history titles. When confidence is high and similarity exceeds the configured
threshold, the method marks `echo_risk=true`. If the echo guard is enabled, the
policy can redirect to fallback or reranking to avoid accepting a high-confidence
low-diversity recommendation by default.

This is a proxy guard, not a measured echo-chamber reduction claim.

## 13. Fallback / abstention policy

The policy can return:

- `accept`: accept the grounded generated item.
- `fallback`: use a configured baseline ranker over the same candidate set.
- `rerank`: run conservative fallback ranking while recording rerank intent.
- `abstain`: emit an empty recommendation list when abstention is allowed and
  fallback is disabled.

Fallback methods must be documented and evaluated with the same candidate set.

## 14. Relation to Phase 3 observation hooks

Phase 3 added generation, confidence parsing, candidate-normalized confidence,
grounding, popularity metadata, and echo-risk observation hooks. Phase 6 uses
those observation signals as inputs to a minimal policy-driven method. It does
not replace the observation framework and still records raw outputs for audit.

## 15. How this differs from existing baselines

### LLM reranking

LLM reranking orders a supplied candidate set directly. Ours first generates a
title, grounds it, estimates uncertainty and risk, then decides whether to accept
or route to fallback.

### Generic RAG

Generic RAG retrieves evidence and generates from it. Ours focuses on generated
title grounding, recommendation-specific confidence, popularity confounding, and
echo-risk handling under a shared recommender evaluator.

### Direct verbalized confidence

Direct verbalized confidence asks for a confidence score and reports it. Ours
uses that confidence only as one signal alongside grounding, candidate-normalized
confidence, popularity bucket, and history similarity.

### Popularity baseline

Popularity ranks by train-split item frequency. Ours may use popularity as a
risk signal or fallback, but its primary observation unit is the generated title
and grounded catalog item.

### BM25

BM25 scores candidate text similarity to history text. Ours can fall back to
BM25, but the method itself is a generative-title policy with uncertainty and
grounding decisions.

## 16. Fair comparison requirements

- Same train/valid/test split for all comparable methods.
- Same candidate set where the protocol compares candidate-based methods.
- Same evaluator and prediction schema.
- Same top-k metrics and beyond-accuracy metrics.
- Train-only popularity for both features and metrics.
- No target leakage in prompts or confidence policy.
- Ablations must disable only the intended component where feasible.
- Mock smoke outputs must be labeled as non-paper evidence.

## 17. Required ablations

- Ours full.
- Ours without uncertainty policy.
- Ours without grounding.
- Ours without candidate-normalized confidence.
- Ours without popularity adjustment.
- Ours without history-similarity / echo-risk guard.
- Ours fallback-only.
- LLM generative baseline.
- LLM rerank baseline.
- BM25.
- Popularity.
- MF.
- Sequential Markov or comparable sequential baseline.

## 18. Failure modes

- Generated title cannot be parsed.
- Generated title does not ground to any catalog item.
- Grounded item is not in the candidate set.
- High-confidence recommendation is a head item with weak user evidence.
- High-confidence recommendation repeats history too closely.
- Fallback ranker is misconfigured.
- Ablation disables a safety component and creates incomparable behavior.
- Mock provider behavior hides real API failure cases.

## 19. Metrics to report

- Recall@K, NDCG@K, HitRate@K, MRR@K, MAP@K where available.
- Validity rate, hallucination rate, parse success rate, candidate adherence.
- Confidence metrics, calibration metrics, risk-coverage data.
- Coverage, catalog coverage, diversity, novelty, long-tail metrics.
- Popularity-stratified metrics and confidence by popularity bucket.
- Latency, token usage, and cost/latency summaries when available.
- Ablation summary by method and disabled component.

## 20. Evidence after R3 (offline refinement)

MovieLens 1M, candidate-500 (target included), real-LLM R3 artifacts (see
`docs/r3_ours_error_decomposition.md`):

- **Observation-first** uncertainty / calibration / grounding / fallback-impact
  analysis is supported.
- **OursMethod ranking improvement** over fallback-only is **not** supported:
  Ours full is worse; accepted overrides **hurt**; conservative gate behavior is
  primarily **fallback-safe**, not a lift claim.

R3b (`configs/experiments/r3b_movielens_1m_conservative_gate_cache_replay.yaml`)
replays the conservative gate from **cache only** across seeds; use
`outputs/tables/r3b_*.csv` for updated numbers. Do not change prompts, evaluator,
split, or candidate protocol for R3b.

## 21. What must not be claimed yet

- Do not claim the method improves recommendation quality.
- Do not claim calibrated uncertainty has been achieved.
- Do not claim popularity bias or echo-chamber risk is mitigated.
- Do not claim superiority over baselines.
- Do not report mock smoke metrics as paper results.
- Do not write paper conclusions from Phase 6 outputs.
