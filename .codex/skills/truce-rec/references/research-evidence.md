# Research And Evidence

Use this reference for Ours/CURE/TRUCE, uncertainty observations, claims, and
paper-readiness decisions.

## Research Spine

TRUCE-Rec studies uncertainty-aware generative recommendation:

```text
user history -> LLM generates item title -> title is grounded to catalog item
  -> correctness, uncertainty, hallucination, popularity, long-tail, and echo
     risk are analyzed
```

The novelty is not direct confidence prompting. It is recommendation-specific
uncertainty around generated titles, catalog grounding, validity, popularity
confounding, long-tail under-confidence, history/echo risk, and conservative
fallback-aware routing or learning.

## Ours No-Go Rules

Do not present Ours as:

- generic LLM reranking;
- generic RAG;
- verbalized confidence alone;
- prompt engineering alone;
- a copied objective, prompt shape, module, or pipeline from a reference paper;
- a stitched combination of senior-recommended baselines.

Reference projects are allowed as inspiration for task formulation and
reproduction fidelity. They must not define the TRUCE/CURE method.

## Evidence Labels

- `smoke/mock`: code path or fixture-only validation.
- `diagnostic`: useful QA or observation, not paper evidence.
- `controlled_adapter_pilot`: TRUCE-side adapter under shared protocol but not
  official-native.
- `official_native_controlled`: official implementation adapted to Qwen3-8B,
  LoRA, and TRUCE same-candidate evaluation with fidelity audit.
- `paper_result`: approved real run with manifests, logs, raw scores/responses,
  predictions, metrics, environment/git info, and artifact checklist.

Never promote one label to another without artifacts.

## Claim Rules

Do not fabricate metrics, tables, logs, conclusions, or server completion. Do
not write paper claims from planned commands, empty directories, smoke output,
or partial logs.

Do not use target item titles or IDs in prompts, future interactions as
evidence, test popularity/test correctness in policy targets, test split
outcomes for tuning, or different candidate sets for comparable methods.

Observation claims should include grounding success, hallucination, candidate
adherence, calibration metrics, selective risk, high-confidence wrong cases,
correct-low-confidence tail cases, popularity buckets, diversity/coverage/
novelty, and echo/history proxies.

## Ours Ablation Expectations

Required ablations include:

- full Ours;
- no uncertainty policy;
- no grounding;
- no candidate-normalized confidence;
- no popularity residual/adjustment;
- no echo/history guard;
- fallback-only;
- LLM generative only;
- LLM rerank only.

If these are missing, mark the method stage open.
