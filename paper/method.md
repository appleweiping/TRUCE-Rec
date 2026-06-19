# Method

> **⚠️ SUPERSEDED — do NOT assemble into the paper.** This is the stale **CU-GR v2** method
> (candidate-panel LLM scorer + conservative fusion + harmful-swap control, a raw-ranking-improvement
> goal). The current TRUCE headline is **RankCRC** (distribution-free risk-controlled *selective*
> reliability) — see **`method_rankcrc.md`** (+ `notation_rankcrc.md`, `related_positioning_rankcrc.md`,
> `introduction.md`). This file is kept for history only, mirroring the superseded `related_positioning.md`.
> Reviewer scores and the gated theorem apply to `method_rankcrc.md`, not this file.

## 1. Problem Setup

Let `I` denote the item catalog and let `H_u = (i_1, ..., i_t)` denote the chronological interaction history for user `u`. For each offline evaluation example, we are given a candidate set `C_u subset I`, a held-out target item `y_u in C_u`, and a fallback ranker `f` that scores or ranks all items in `C_u`. The target `y_u` is used only for offline evaluation and for constructing training or validation labels; it is never shown to the LLM as the target item in prompts.

We use a sampled candidate protocol with target inclusion. Thus, the evaluation candidate set contains the held-out item, but it is not identified to the model. A candidate-local LLM scorer `g` is queried only on a panel `P_u subset C_u`, where `|P_u|` is much smaller than `|C_u|`. The method returns a final ranking `pi_u` over candidate items. The goal is to improve top-`K` ranking metrics, especially NDCG@10, while preserving catalog validity and controlling harmful swaps relative to the fallback ranking.

In the main experiments, MovieLens uses `|C_u| = 500`; Amazon Beauty uses `|C_u| = 479` because the local catalog contains 479 items. Both use `|P_u| = 15` and subset_size=200 examples per seed for real LLM calls.

## 2. Why Free-Form Generation Fails

CU-GR v2 is motivated by negative evidence from the earlier free-form and confidence-based pipeline. In that setting, the LLM generated item titles directly and reported verbalized confidence. This interface has several failure modes. A generated title may not correspond to any catalog item. A grounded title may correspond to a catalog item that is not in the evaluation candidate set. Even when the response is parseable, the model's verbalized confidence is not necessarily calibrated to ranking correctness. In the retained MovieLens artifacts, high-confidence wrong outputs were common, and promoting generated items to rank 1 did not yield positive override labels for CU-GR v1. Conservative gating reduced harm primarily by falling back to the baseline ranker, not by producing a reliable LLM-driven improvement.

These failures motivate a methodological change: the LLM should not be asked to invent the final item. Instead, it should be constrained to compare valid candidate items, and its signal should be calibrated before it can alter the fallback ranking.

## 3. Candidate Panel Construction

For each user `u`, CU-GR v2 constructs a deterministic candidate panel `P_u subset C_u`. The panel is designed to expose the LLM to plausible alternatives and contrastive candidates while preserving candidate adherence. In the main experiments, `|P_u| = 15`.

The panel includes:

- fallback ranks 1-5;
- fallback ranks 10, 20, and 50 when available;
- popularity contrast items;
- tail contrast items;
- a sequential candidate when an existing sequential reference is available;
- a candidate-adherent generated item when such an artifact is available;
- deterministic fill items from the candidate set until the target panel size is reached.

All panel items must belong to `C_u`. The held-out target may appear in the panel because it is part of the evaluation candidate set, but it is never marked as the target. The construction is deterministic for a fixed example, seed, candidate set, and panel size.

## 4. Candidate-Local Listwise Preference

The LLM prompt presents the user's history titles and the panel items using anonymous labels `A`, `B`, `C`, and so on. The prompt includes candidate titles or metadata needed to compare items, but it does not reveal the target item or a global target item identifier. The LLM is asked to return a strict JSON response containing a listwise ranking with scores and confidence values for labeled panel items.

The parser maps labels back to item identifiers through the prompt-local label map. It rejects unknown labels, duplicate labels, and invalid structures; records parse success, invalid-label rate, duplicate-label rate, and partial-ranking status; and preserves the raw output for audit. If parsing fails, the system can fall back to the baseline order rather than trusting an invalid LLM response.

This design changes the LLM interface from open-ended item generation to candidate-local preference estimation. The model can express preferences only among valid panel items, which keeps the downstream ranking catalog-grounded under the sampled candidate protocol.

## 5. Calibrated Fusion

For each panel item `i in P_u`, CU-GR v2 computes four quantities:

- `s_f(i)`: a normalized fallback score derived from the fallback ranking or fallback score within the panel;
- `s_l(i)`: a normalized LLM listwise preference score from the parsed JSON response;
- `c_l(i)`: the LLM confidence associated with the candidate-local score;
- `p(i)`: a popularity penalty estimated from train-only popularity statistics.

The fused score is

```text
score(i) =
  alpha * s_f(i)
+ beta  * s_l(i)
+ gamma * c_l(i)
- lambda * p(i).
```

The weights `alpha`, `beta`, `gamma`, and `lambda` are selected using validation data only. They are not selected on the held-out test seed. MovieLens selected `alpha=0.5`, `beta=0.7`, `gamma=0.2`, `lambda=0.05`; Amazon Beauty selected `alpha=0.5`, `beta=0.3`, `gamma=0.0`, `lambda=0.1`.

The final ranking `pi_u` is formed by reranking panel items according to the fused score while preserving the fallback order for non-panel candidates. Thus, CU-GR v2 intervenes locally on the fallback ranking rather than replacing the entire candidate order. All output items remain valid members of `C_u`.

## 6. Safety Constraints

CU-GR v2 applies several safety and audit constraints:

- parser success is required for LLM-driven panel scoring;
- all panel and output items must be candidate-adherent;
- the target item and target title are not identified in the prompt;
- future interactions are excluded from user history;
- popularity penalties are computed from train-only statistics;
- harmful swap rate is monitored relative to the fallback ranking;
- fusion weights and safe-fusion thresholds are selected on validation data only;
- held-out test data is not used for tuning.

The method does not guarantee zero harmful swaps. Instead, it treats harmful swaps as a measured failure mode and constrains the selected fusion policy to keep validation harmful_swap_rate within the configured bound.

## 7. Training / Validation / Test Protocol

The full-seed protocol uses seed13 for training or fitting the fusion selection, seed21 for validation, and seed42 as the held-out test seed. The validation seed selects the fusion weights from the allowed grid and selects safe-fusion thresholds when applicable. The held-out seed42 result is then reported without changing the evaluator, candidate protocol, prompt mechanism, or fusion grid.

This split is intended to prevent test-seed tuning. Reported MovieLens and Amazon Beauty gains are therefore held-out seed42 results under the sampled candidate protocol.

## 8. Complexity / Cost

CU-GR v2 makes one LLM call per evaluated example for panel scoring. The main experiments use panel size 15, which bounds prompt size and parsing complexity independently of the full candidate set size. Non-panel candidates are not sent to the LLM and retain fallback order.

The implementation supports cache and resume, saves raw LLM outputs, records token usage, tracks latency, and exports cost summaries. Under the reported gates, the seed42 effective cost per 200 examples is 0.059083 USD for MovieLens and 0.076183 USD for Amazon Beauty. These costs are specific to the DeepSeek v4 flash API configuration used in the artifacts and should not be interpreted as production-scale inference measurements.
