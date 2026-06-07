# Experimental Setting, Metrics, and Official Baselines

> **Authoritative reference.** This is the single source of truth for TRUCE-Rec's evaluation
> setting, metrics, and the eight official baselines. Any agent touching experiments, the method,
> or the paper tables must read this first. If the setting changes, update this file in the same
> commit.

## 1. Task and setting

TRUCE-Rec is evaluated as **same-candidate next-item ranking** over Amazon Reviews 2023 domains.

| Element | Value |
|---|---|
| Task | Rerank a fixed candidate set per user event (next-item ranking) |
| Candidates | **101 per event**: 1 positive + 100 popularity-sampled negatives |
| Backbone | **Qwen3-8B** for all LLM-based methods (LoRA where the method fine-tunes) |
| Domains | **8**: beauty, books, electronics, movies, sports, toys, home, tools |
| Scale | 10,000 test users per domain; **beauty = 973** (supplementary smaller-N) |
| Score schema | `source_event_id,user_id,item_id,score` |
| Score coverage | 1.0 (every candidate scored) |
| Splits | leave-last-out; test-history mode `train_plus_valid`; **test never used for selection** |

All methods — baselines and TRUCE-Rec's own method — are evaluated **identically** under this
protocol, with the same evaluator and the same candidate sets.

## 2. Metrics

Primary (ranking quality), reported per (domain, method):

- **HR@5 / HR@10 / HR@20**
- **NDCG@5 / NDCG@10 / NDCG@20**
- **MRR**

Method-side diagnostics (TRUCE-Rec only, not part of the baseline comparison): validity /
hallucination / candidate adherence, confidence calibration (ECE, Brier, risk–coverage),
coverage / diversity / novelty, popularity-stratified slices, efficiency / cost. Paired
significance testing where applicable.

## 3. The eight official baselines

The paper's main comparison is against the following eight official LLM4Rec methods, all run from
their **official code at a pinned commit**, on the **Qwen3-8B** backbone, with each method's
official/default (or reported-optimal) hyper-parameters and a declared, audited adaptation to the
shared candidate schema. Baselines are **not** exhaustively re-tuned (a standard, reviewer-accepted
controlled-comparison compromise); only TRUCE-Rec's own method may tune, and only on validation.

| key | method | family |
|-----|--------|--------|
| `elmrec`  | ELMRec  | graph-enhanced LLM4Rec |
| `irllrec` | IRLLRec | intent-aware LLM4Rec |
| `llm2rec` | LLM2Rec | sequential (SASRec-style) LLM4Rec |
| `llmemb`  | LLMEmb  | embedding-alignment LLM4Rec |
| `llmesr`  | LLM-ESR | LLM-enhanced sequential rec |
| `proex`   | ProEx   | profile / explanation LLM4Rec |
| `promax`  | ProMax  | profile-based LLM4Rec |
| `rlmrec`  | RLMRec  | graph-contrastive representation LLM4Rec |

## 4. Frozen evidence location

- **Directory:** [`data/official_baselines/`](../data/official_baselines/)
- **Master table:** `data/official_baselines/baseline_comparison_8domains.csv` — 64 rows
  (8 domains × 8 baselines), one row per pair.
- **Per-pair evidence:** `data/official_baselines/domains/<domain>/<baseline>/` — metric summary,
  ranking metrics, coverage audit, and (where available) fairness provenance + run/score audits.
- **Integrity:** `data/official_baselines/IMPORT_MANIFEST.json` lists every committed file with
  size + sha256 (330 files, 64 pairs). Re-verify with the snippet in that directory's README.
- **Frontier:** `data/official_baselines/frontier_best_baseline_per_domain.csv` — the strongest
  baseline per domain (by NDCG@10); this is the bar TRUCE-Rec must beat.

This evidence is **frozen and read-only**. It is the fixed external reference; do not recompute or
re-rank it. TRUCE-Rec's own runs are tracked separately and are the only thing measured against it.

## 5. Per-domain SOTA bar (frontier)

Best official baseline per domain by NDCG@10 (the number Ours must beat to claim SOTA):

| domain | N | best baseline | NDCG@10 | HR@10 | MRR |
|---|---:|---|---:|---:|---:|
| beauty | 973 | **ProEx** | 0.1506 | 0.2528 | 0.1429 |
| books | 10000 | LLMEmb | 0.2737 | 0.4722 | 0.2332 |
| electronics | 10000 | LLMEmb | 0.1196 | 0.2450 | 0.1067 |
| movies | 10000 | LLMEmb | 0.1690 | 0.3336 | 0.1443 |
| sports | 10000 | LLMEmb | 0.1795 | 0.3384 | 0.1539 |
| toys | 10000 | LLMEmb | 0.2049 | 0.3505 | 0.1814 |
| home | 10000 | LLMEmb | 0.0939 | 0.1856 | 0.0901 |
| tools | 10000 | LLMEmb | 0.1159 | 0.2257 | 0.1065 |

Note: **beauty's top baseline is ProEx, not LLMEmb** — on the smaller-N beauty set the
profile/explanation method leads, while LLMEmb dominates the seven large domains. The beauty bar to
beat is therefore **ProEx NDCG@10 = 0.1506 / HR@10 = 0.2528 / MRR = 0.1429**. Because beauty has the
fewest users (973), it is the fastest domain to iterate and validate on — TRUCE-Rec's method is
brought to SOTA on beauty first, then rolled out to the other seven.

## 6. Why baseline scores are reused, not re-run

The same official baseline, run on the same data under the same candidate protocol and the same
backbone, produces the same scores regardless of which method paper consumes them. TRUCE-Rec shares
this exact setting (8 domains, 8 baselines, 101-candidate protocol, Qwen3-8B), so the frozen
baseline numbers are imported as a shared public reference rather than recomputed. **No baseline is
re-run.** TRUCE-Rec's *own* method, however, is implemented, trained, and evaluated entirely
independently — it is never derived from any other project's method code.
