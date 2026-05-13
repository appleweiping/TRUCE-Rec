# Review Checklist

Use this reference before declaring a stage complete or preparing paper-facing
claims.

## Fatal-Risk Review

Check for these failure modes:

- Ours looks like generic LLM reranking, generic RAG, or prompt engineering.
- Ours looks stitched from TALLRec, OpenP5, DEALRec, LC-Rec, LLaRA, LLM-ESR, or
  another recent project.
- Baselines are not official-native where claimed.
- Methods use different splits, candidates, negative sampling, backbones, or
  evaluators.
- Tiny/MovieLens/Beauty-only results are overclaimed as paper-scale evidence.
- Test split outcomes were used for tuning or design selection.
- Candidate rows changed for only one method.
- Uncertainty analysis lacks grounding, hallucination, calibration,
  popularity, long-tail, diversity, or echo/history slices.
- Ours lacks component ablations.
- Cost, latency, throughput, raw artifacts, or logs are missing.
- Stale docs contradict the current status.

## Writing-Ready Gate

The experiment phase can move to paper writing only when all are true:

- Beauty, books, electronics, and movies same-candidate runs are complete at the
  declared scale.
- Base Qwen3-8B and at least the senior-recommended Qwen3-8B-LoRA baselines
  have observation analyses, or omissions are justified.
- Official-native or clearly labeled controlled baselines have complete score,
  import, prediction, metric, manifest, and log artifacts.
- Ours full and required ablations have complete artifacts under the same
  protocol.
- Metrics include ranking, validity/hallucination/candidate adherence,
  coverage/diversity/novelty, long-tail/popularity slices, efficiency/cost, and
  paired significance where applicable.
- Failure cases and limitations are documented.
- A top-conference-style reviewer pass finds no fatal gap in novelty,
  fairness, scale, leakage, ablations, or reproducibility.

If any gate is missing, state the blocker and the shortest concrete command or
implementation step to close it.

## Reviewer Prompt

Use this when running an explicit review pass:

```text
Act as a SIGIR/WWW/RecSys/NeurIPS reviewer. Given the current metrics,
artifacts, configs, and docs, identify fatal flaws in originality, baseline
fairness, data scale, leakage, ablations, statistical testing, efficiency,
long-tail analysis, and reproducibility. Do not praise the work unless the
artifact evidence supports it. List missing experiments before paper claims.
Also check whether the proposed method is a genuinely original TRUCE/CURE
framework or only a stitched combination of senior-recommended papers.
```
