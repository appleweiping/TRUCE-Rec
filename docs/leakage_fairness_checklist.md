# Phase 6 Leakage and Fairness Checklist

This checklist applies to OursMethod, fallback variants, ablations, and
baseline comparisons that share the TRUCE evaluator.

## Prompt leakage

- [x] Target title never appears in the generative prompt.
- [x] Target item ID never appears in the generative prompt.
- [x] Prompt builders exclude the target item from visible candidates.
- [x] User history is filtered so the target item is not repeated in prompt
  history if it appears there due to fixture construction.

## Temporal and split leakage

- [x] Future interactions are not used by OursMethod policy.
- [x] Fallback rankers are fit with train examples only.
- [x] Train popularity only is used for OursMethod popularity buckets.
- [x] Evaluator popularity context is train-only.

## Candidate and evaluator fairness

- [x] Fallback ranker uses the same candidate set as the OursMethod example.
- [x] Comparable baselines use the same processed examples and candidate
  protocol in smoke configs.
- [x] All methods emit the unified prediction schema.
- [x] All methods are evaluated by the same evaluator.
- [x] Baseline candidate scoring may include the target item as an unlabeled
  candidate, but prompts and training rows must not reveal that it is the
  target.
- [x] Main compared LLM baselines use Qwen3-8B-LoRA, shared TRUCE candidate
  rows, and official default or reported-optimal baseline hyperparameters
  unless explicitly separated into a reference protocol.

## Confidence and grounding safeguards

- [x] Confidence is parsed from MockLLM output or candidate-normalized prompt
  output; it is not computed using the target label.
- [x] Grounding uses catalog titles only.
- [x] Grounding success, grounding score, and grounded item ID are recorded for
  audit.
- [x] Policy decisions are based on confidence, grounding, popularity, history
  similarity, and config flags, not target correctness.

## Fallback transparency

- [x] Fallback method is recorded in prediction metadata.
- [x] Fallback decisions record reasons.
- [x] Fallback-only ablation is explicitly labeled.
- [x] Fallback ranker metadata is preserved under OursMethod metadata.

## Ablation fairness

- [x] Ablations use config flags, not separate ad hoc code paths.
- [x] Disabled components are recorded in metadata.
- [x] Each ablation disables exactly one component where possible.
- [x] Fallback-only is documented as a multi-component routing ablation.

## Phase 6 execution limits

- [x] MockLLM only for smoke runs.
- [x] No real external API calls.
- [x] No HF model downloads.
- [x] No real LoRA/QLoRA training.
- [x] No Phase 7 paper claims or conclusions.
