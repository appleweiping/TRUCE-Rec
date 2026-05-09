# RESEARCH_IDEA.md

## 1. Purpose

This document is the source of truth for the project's research idea.

AGENTS.md defines engineering rules, reproducibility requirements, interfaces, baselines, metrics, and validation standards. It does not define the core research idea.

This document locks the research direction so that future implementation phases do not accidentally turn the project into a generic LLM reranker, generic RAG recommender, or prompt-engineering baseline.

## 2. Core Research Direction

The project studies:

```text
Uncertainty-aware Generative Recommendation
```

The final task is generative recommendation:

```text
user history -> LLM generates recommended item title -> generated title is grounded to catalog item -> evaluate correctness and uncertainty
```

The central question is:

```text
When an LLM generates a recommendation title, does its uncertainty signal explain when the recommendation is correct, hallucinated, popularity-biased, long-tail, or echo-chamber reinforcing?
```

This is a recommendation-specific uncertainty problem, not generic QA calibration.

## 3. Main Motivation

LLMs may generate plausible recommendation titles, but plausibility is not reliability.

We care about the following questions:

- The LLM may generate the correct recommendation, but is it actually confident?
- If the generated recommendation is wrong, is it wrong because the model is uncertain, or because it is confidently wrong?
- Are high-confidence generated recommendations mostly popular items?
- Are non-popular or long-tail items assigned lower confidence even when they are correct?
- Does high confidence reinforce the user's past preferences too strongly, causing echo-chamber or filter-bubble effects?
- Can uncertainty identify noisy generated pseudo-labels or unreliable training samples?
- Can uncertainty decide when to abstain, retrieve more evidence, sample more generations, or rerank conservatively?
- If yes/no self-verification confidence correlates with correctness but does not improve final recommendation quality, can we design a richer generative recommendation observation around title generation and grounding?

## 4. Why Direct Confidence Is Not Enough

There is already substantial work on LLM confidence calibration.

Known issues:

- verbalized confidence can be miscalibrated;
- LLMs may be overconfident;
- confidence depends on prompt design;
- confidence may measure fluency or familiarity rather than correctness;
- confidence can be biased toward frequent or familiar entities;
- yes/no self-verification may correlate with correctness but still fail to improve recommendation quality.

Therefore, this project should not claim novelty from simply asking:

```text
Are you confident? yes/no
```

or

```text
Give a confidence score from 0 to 1.
```

Instead, the project should study uncertainty in the structure of generative recommendation.

## 5. Why Recommendation Is Different From Generic Calibration

Generic LLM calibration usually evaluates whether confidence matches answer correctness.

Generative recommendation has additional structure:

- The output is a generated title, not a fixed class label.
- The generated title must be grounded to a catalog item.
- Exact title string match may fail even when the semantic item match is correct.
- The ground truth next item is implicit and incomplete.
- Multiple recommendations may be plausible.
- Popularity is a major confounder.
- Long-tail items may be correct but under-confident.
- User history creates preference inertia.
- High-confidence recommendations may reduce diversity and reinforce echo chambers.
- Candidate-set adherence and catalog validity are separate from correctness.

Therefore, the observation target is not only:

```text
confidence vs correctness
```

but also:

```text
confidence vs grounding
confidence vs hallucination
confidence vs popularity
confidence vs long-tail
confidence vs history similarity
confidence vs diversity
confidence vs echo-chamber risk
```

## 6. Core Observation Unit

Each observation example should contain:

```json
{
  "user_id": "u1",
  "history_items": ["..."],
  "history_titles": ["..."],
  "generated_title": "...",
  "generated_confidence": 0.0,
  "uncertainty_signal_type": "verbalized | yes_no_verify | sample_consistency | candidate_normalized | token_probability | evidence_based",
  "grounded_item_id": "...",
  "grounded_title": "...",
  "grounding_success": true,
  "grounding_score": 0.0,
  "target_item_id": "...",
  "target_title": "...",
  "is_exact_hit": false,
  "is_grounded_hit": false,
  "is_catalog_valid": true,
  "is_hallucinated": false,
  "item_popularity": 0,
  "popularity_bucket": "head | mid | tail",
  "history_similarity": 0.0,
  "category_repetition": 0.0,
  "novelty": 0.0,
  "diversity_context": {},
  "raw_output": "...",
  "prompt_template_id": "...",
  "prompt_hash": "..."
}
```

This observation schema is more important than directly optimizing a model at this stage.

## 7. Uncertainty Signals To Compare

The project should compare several uncertainty signals, because direct verbalized confidence alone may be weak.

### 7.1 Verbalized Confidence

Prompt the LLM to generate a title and confidence:

```json
{
  "recommendation": "generated item title",
  "confidence": 0.0,
  "reason": "...",
  "uncertainty_reason": "..."
}
```

This is easy to implement and works with API models, but it may be miscalibrated or overconfident.

### 7.2 Yes/No Self-Verification

After generating a title, ask:

```text
Given the user's history and the generated recommendation, is this recommendation likely to match the user's next interaction? Answer yes/no and provide confidence.
```

This should be treated as an observation signal, not a final method.

### 7.3 Multi-Sample Consistency

Generate multiple recommendation titles with controlled temperature and measure:

- exact title agreement;
- grounded item agreement;
- category agreement;
- entropy over grounded item IDs;
- confidence variance.

If many samples ground to the same item, the model may be more certain.

### 7.4 Candidate-Normalized Confidence

Build a small set of plausible alternatives:

- generated title;
- BM25 candidate title;
- popularity candidate title;
- random plausible distractor;
- tail candidate title.

Ask the LLM to assign confidence over all options.

This is inspired by confidence calibration work using distractors or normalized confidence. The recommendation-specific version should normalize confidence over plausible item alternatives rather than independent absolute confidence.

### 7.5 Token / Sequence Probability

For local HF models, optionally compute likelihood of the generated title.

This is optional because API models may not expose logits.

### 7.6 Evidence-Based Confidence

Estimate confidence from non-LLM evidence:

- semantic similarity between generated item and user history;
- category continuity;
- brand continuity if available;
- item popularity;
- retrieval score;
- collaborative score.

This helps separate:

```text
confident because user evidence supports it
```

from:

```text
confident because the item is popular/familiar
```

## 8. Main Observation Questions

The first observation paper or section should answer:

### Q1. Is LLM confidence calibrated for generative recommendation?

Metrics:

- ECE
- Brier score
- reliability diagram data
- confidence bucket accuracy
- risk-coverage curve

Correctness targets:

- exact generated title hit;
- grounded item hit;
- catalog validity;
- candidate adherence.

### Q2. Are wrong recommendations low-confidence or high-confidence?

Report four groups:

- high confidence + correct
- low confidence + correct
- low confidence + wrong
- high confidence + wrong

The most important failure case is:

```text
high confidence + wrong / hallucinated
```

### Q3. Does confidence correlate with popularity?

Measure:

- correlation(confidence, log item popularity);
- confidence by popularity bucket;
- accuracy by popularity bucket;
- ECE by popularity bucket;
- overconfidence gap for head/mid/tail items.

### Q4. Are correct long-tail recommendations under-confident?

Measure:

- confidence on correct head items;
- confidence on correct tail items;
- under-confidence rate for correct tail items;
- abstention rate by popularity bucket.

### Q5. Does high confidence amplify echo-chamber effects?

Use proxies:

- similarity between generated item and user history;
- category repetition;
- brand repetition;
- popularity reinforcement;
- confidence-weighted novelty;
- confidence-weighted diversity.

Observation:

```text
high confidence may correspond to low novelty and high history similarity
```

### Q6. Can uncertainty prune noisy generated labels?

Possible later experiment:

- train on all generated pseudo-labels;
- train after removing low-confidence generated labels;
- train after removing high-confidence hallucinated labels;
- train with confidence weighting.

This is not Phase 3 yet, but Phase 3 should log the necessary metadata.

## 9. Expected Contribution Shape

The expected contribution is not:

```text
We ask LLM for confidence.
```

The expected contribution is:

```text
We introduce a generative recommendation uncertainty observation framework that grounds generated titles to a catalog and analyzes how uncertainty interacts with correctness, hallucination, popularity bias, long-tail recommendation, and echo-chamber risk.
```

A later method may use these findings to design uncertainty-aware training, filtering, abstention, or reranking.

## 10. OursMethod Status

Phase 6 implemented a minimal smoke-capable OursMethod integration after the
user confirmed the mechanism. The implemented method direction is:

```text
Calibrated Uncertainty-Guided Generative Recommendation
```

Implemented Phase 6 mechanisms include:

- uncertainty-aware abstention;
- confidence-normalized candidate comparison;
- popularity-bias-adjusted confidence;
- high-confidence hallucination filtering;
- confidence-diversity tradeoff reranking.

This integration is not a paper result. It is a config-driven smoke pipeline
for real-experiment planning. No effectiveness, calibration, popularity-bias
mitigation, or echo-risk mitigation claim is allowed until approved real
experiments produce metrics.

## 11. Phase 3 Implication

Phase 3 should implement LLM baseline infrastructure and uncertainty observation hooks.

It should include:

- prompt templates for generative title recommendation;
- prompt templates for verbalized confidence;
- prompt templates for yes/no self-verification;
- prompt templates for candidate-normalized confidence;
- parser for generated title + confidence;
- grounding from generated title to catalog item;
- confidence metrics;
- calibration metrics;
- popularity metadata hooks;
- echo-chamber proxy hooks;
- mock provider tests;
- API provider interface;
- HF provider interface.

Phase 3 must not implement OursMethod. Phase 6 is the first phase where the
user-approved OursMethod integration exists.

## 12. Literature Inspiration

This project should be informed by existing LLM confidence calibration work, especially Bryan Hooi group's work:

- "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs"
  - This work studies black-box confidence elicitation for LLMs.
  - It decomposes confidence elicitation into prompting strategies, sampling multiple responses, and aggregation/consistency methods.
  - It evaluates both confidence calibration and failure prediction.
  - It finds that LLMs often show overconfidence when verbalizing confidence.
  - It also suggests that human-inspired prompts, multi-sample consistency, and aggregation strategies can mitigate overconfidence, but no method is universally reliable.

Our project should not simply copy verbalized confidence. Instead, we adapt the motivation to generative recommendation:

- QA answer confidence -> generated item title confidence
- answer correctness -> generated title grounding + next-item hit
- sampling consistency -> multi-sample grounded-item consistency
- failure prediction -> hallucination / invalid title / wrong grounded item prediction
- distractor-normalized confidence -> candidate-normalized item confidence
- overconfidence -> high-confidence wrong recommendation / high-confidence hallucination
- task difficulty -> user-history ambiguity and long-tail uncertainty

Other related directions to keep in mind:

- LLM confidence estimation and calibration surveys
- verbalized confidence scores
- self-consistency / sample consistency
- self-generated distractors and normalized confidence
- calibration for generative structured prediction
- uncertainty decomposition
- generative recommendation studies

These are inspiration only. The novelty must come from recommendation-specific generative uncertainty observation.

Important: Bryan Hooi group's confidence calibration work is literature inspiration only.

Do not copy their problem setting, experiments, code, writing, claims, or method as our contribution.

Use it only to understand useful uncertainty-estimation ideas such as:

- verbalized confidence;
- prompting strategies for eliciting confidence;
- multi-sample consistency;
- aggregation-based confidence;
- calibration metrics;
- failure prediction.

Our project must adapt these ideas to a different task:

```text
generative recommendation -> generated item title -> catalog grounding -> recommendation-specific uncertainty observation
```

The novelty must come from recommendation-specific structure:

- generated title grounding;
- catalog validity;
- hallucination of non-existing items;
- popularity-confounded confidence;
- long-tail under-confidence;
- confidence-driven echo-chamber risk;
- uncertainty-aware filtering or abstention in recommendation.

Do not present Bryan Hooi group's confidence elicitation framework as our own method.

Do not claim novelty for generic confidence elicitation.

Do not reuse paper text.

Do not implement their method as OursMethod.

## 13. Non-Negotiable Safeguards

Codex must not:

- treat LLM reranking alone as our contribution;
- treat verbalized confidence alone as sufficient novelty;
- claim calibration improvement before metrics exist;
- claim echo-chamber mitigation before measuring it;
- use target item title in the prompt;
- use future interactions as evidence;
- evaluate only exact title match and ignore catalog grounding;
- ignore item popularity as a confounder;
- compare OursMethod against baselines using different candidate sets;
- write paper claims before real metrics exist.

## 14. Open Questions For Later

Before paper-grade Ours finalization and large-scale runs, the user and agents
must decide:

- Which uncertainty signal is primary?
- Which correctness target is primary: exact title, grounded item hit, candidate adherence, or semantic match?
- Whether the final method uses abstention, reranking, filtering, or training-time pruning.
- How to control popularity bias.
- How to define echo-chamber risk.
- Which datasets are appropriate for generative title recommendation.
- Whether observation alone is enough for a paper, or whether a method is required.
