# Related Positioning

> **⚠️ SUPERSEDED — do NOT assemble into the paper.** This is the stale **CU-GR v2** related-work.
> The current TRUCE headline is **RankCRC**; use **`related_positioning_rankcrc.md`** (with the verified
> `refs/conformal.bib`). Kept for history only.

- **LLM-as-ranker.** CU-GR v2 shares the conditional candidate-ranking setting with LLM-as-ranker work [LLMRank], but its motivation differs: we start from observed free-form generation and confidence failures, then restrict the LLM to candidate-local preference estimation and calibrated fusion.

- **Generative recommendation.** Generative recommendation often asks a model to produce item names, identifiers, or natural-language recommendations [GenerativeRecSurvey]. Our artifacts show that free-form title generation is brittle under strict catalog grounding in the MovieLens candidate-500 protocol. CU-GR v2 therefore uses generation only as structured preference signaling over valid candidate items, not as unconstrained final item generation.

- **Confidence calibration.** Prior work studies whether model probabilities or verbalized confidence can be calibrated [CalibrationSurvey]. CU-GR v2 does not trust verbalized confidence directly. Instead, it parses candidate-local listwise scores and combines them with fallback scores through validation-selected fusion weights. Confidence is only one optional term in the fusion formula, and Amazon Beauty selected a zero confidence weight.

- **Self-calibrated listwise scoring.** Listwise scoring prompts can expose relative preference information among candidates [SelfCalibratedListwise]. CU-GR v2 uses a listwise interface but adds candidate normalization, parser rejection, raw-output auditing, and validation-constrained fusion with a non-LLM fallback ranker.

- **Selective prediction and reject options.** CU-GR v2 can be viewed as a selective intervention on a fallback ranking rather than a standalone recommender. If the LLM response is invalid or unsafe, the system can preserve fallback behavior. This is related to reject-option and selective-prediction framing, but the decision surface is ranking-specific and measured through harmful swaps.

- **Reranking.** Unlike pure LLM reranking, CU-GR v2 only reranks a deterministic panel and preserves fallback order outside that panel. This limits the intervention surface, keeps all outputs candidate-adherent, and makes harmful swaps auditable under the sampled candidate protocol.

- **Sequential and graph baselines.** SASRec [SASRec], BERT4Rec [BERT4Rec], GRU4Rec [GRU4Rec], and LightGCN [LightGCN] are the intended reviewer-grade sequential and graph collaborative filtering comparators. The repository routes these through a RecBole adapter so that training can use the external library while final predictions are imported into the TRUCE schema and evaluated by the same evaluator as CU-GR v2. MovieLens 1M and Amazon Beauty now have TRUCE-evaluated SASRec, BERT4Rec, GRU4Rec, and LightGCN rows. The resulting framing should be complementary rather than dominance-based: CU-GR v2 is stronger than fallback and weak generative controls in the target gates, but MovieLens RecBole baselines slightly exceed CU-GR v2 by NDCG@10 under the current sampled-candidate table.
