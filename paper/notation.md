# Notation

> **⚠️ SUPERSEDED — do NOT assemble into the paper.** This is the stale **CU-GR v2** notation.
> The current TRUCE headline is **RankCRC**; use **`notation_rankcrc.md`**. Kept for history only.

| Symbol | Meaning |
| --- | --- |
| `u` | User or evaluation instance. |
| `I` | Full item catalog. |
| `i` | An item in the catalog or candidate set. |
| `H_u` | Chronological interaction history for user `u`. |
| `C_u` | Candidate set for user `u` under the sampled candidate protocol. |
| `y_u` | Held-out target item used for offline evaluation and labels, never identified in prompts. |
| `f` | Fallback ranker, implemented as BM25/fallback in the CU-GR v2 gates. |
| `g` | LLM candidate-local listwise scorer. |
| `P_u` | Candidate panel shown to the LLM, with `P_u subset C_u`. |
| `|P_u|` | Panel size; 15 in the main experiments. |
| `pi_u` | Final output ranking over candidate items. |
| `s_f(i)` | Normalized fallback score for item `i` within the panel. |
| `s_l(i)` | Normalized LLM listwise preference score for item `i` within the panel. |
| `c_l(i)` | LLM-reported confidence associated with panel item `i`. |
| `p(i)` | Popularity penalty for item `i`, computed from train-only popularity statistics. |
| `alpha` | Fusion weight for the normalized fallback score. |
| `beta` | Fusion weight for the normalized LLM listwise score. |
| `gamma` | Fusion weight for the LLM confidence term. |
| `lambda` | Fusion weight for the popularity penalty. |
| `K` | Evaluation cutoff, with `K=10` for the main reported metrics. |
| `NDCG@10` | Normalized Discounted Cumulative Gain at rank 10. |
| `Recall@10` | Recall at rank 10. |
| `MRR@10` | Mean Reciprocal Rank at rank 10. |
| `HitRate@10` | Hit rate at rank 10. |
| `harmful_swap_rate` | Fraction of examples where the CU-GR v2 intervention reduces NDCG@10 relative to fallback. |
| `target_in_panel_rate` | Fraction of examples where the held-out target appears in the candidate-local panel. |
