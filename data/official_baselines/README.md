# Official Baseline Evidence (frozen)

This directory holds the **frozen, paper-facing evidence** for the eight official LLM4Rec
baselines that TRUCE-Rec is compared against, evaluated under the project's standard
same-candidate ranking protocol across **eight benchmark domains**. It is the single source of
truth for every external-baseline number that appears in the main comparison table of the paper.

Only the lightweight, paper-relevant evidence is committed here (per-pair metric rows, coverage
audits, provenance, run/score-audit JSONs). The heavy artifacts (per-event score matrices, model
checkpoints, raw prediction dumps) live on the compute node by design, so the comparison is fully
reproducible from version control without storing multi-GB files.

This evidence is **read-only reference material**. Do not edit, recompute, or re-rank these files;
TRUCE-Rec's own method runs are tracked separately and are the only thing measured *against* this
set.

## Evaluation protocol

All baselines — and TRUCE-Rec's own method — are evaluated identically:

- **Task**: candidate ranking (rerank a fixed candidate set per user event).
- **Candidates**: same-candidate protocol, **101 candidates per user** (1 positive + 100
  popularity-sampled negatives).
- **Backbone**: Qwen3-8B for all LLM-based baselines, adapted with LoRA where the official method
  fine-tunes.
- **Users**: 10,000 test users per domain; Beauty is a supplementary smaller-N set (973 users).
- **Comparison variant**: `official_code_qwen3base_default_hparams_declared_adaptation` — each
  baseline is run from its **official code at a pinned commit**, with the baseline's official /
  default (or reported-optimal) hyper-parameters, plus a declared, audited adaptation to the shared
  candidate schema. Baselines are **not** exhaustively re-tuned; only TRUCE-Rec's own method may
  tune hyper-parameters, and only under the declared validation protocol (never on test).
- **Score coverage**: 1.0 (every candidate scored) for all completed rows.
- **Score schema**: `source_event_id,user_id,item_id,score`.

## Metrics

Reported per (domain, baseline): **HR@5/10/20**, **NDCG@5/10/20**, and **MRR**. Ranking quality is
the primary axis; per-baseline exposure/coverage diagnostics are retained in
`external_score_coverage.csv` and `ranking_metrics.csv`.

## The eight official baselines

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

Exact official repository URL, pinned commit, entrypoint, and audited adaptation are recorded per
baseline in each `fairness_provenance.json` (where available for that pair).

## The eight domains

| group | domains | users |
|-------|---------|-------|
| primary    | sports, toys, home, tools  | 10,000 each |
| additional | books, electronics, movies | 10,000 each |
| additional | beauty                     | 973 (supplementary smaller-N) |

## Files

```
baseline_comparison_8domains.csv   # master table: 64 rows (8 domains x 8 baselines), one row per pair
IMPORT_MANIFEST.json               # integrity manifest: every committed file with size + sha256
domains/<domain>/<baseline>/
    same_candidate_external_baseline_summary.csv  # full metric row (HR/NDCG/MRR + audit fields)
    ranking_metrics.csv                           # ranking metrics only
    external_score_coverage.csv                   # candidate score-coverage audit
    fairness_provenance.json                      # official repo, pinned commit, adaptation (where available)
    <baseline>_official_run_summary.json          # run metadata (where available)
    <baseline>_official_score_audit.json          # score-file validation (where available)
```

The master table is the authoritative join of the per-pair summaries. It must contain exactly **64
baseline rows** and **no internal-method rows** — TRUCE-Rec's own results never enter this file.

## Provenance completeness

All 64 (domain, baseline) pairs carry the metric summary and the coverage audit. Primary-domain
pairs additionally carry `fairness_provenance.json` and the run/score-audit JSONs; some
additional-domain pairs currently retain the metric summary and coverage only. The exact per-pair
file inventory and sha256 of every committed file are enumerated in `IMPORT_MANIFEST.json`.

To re-verify integrity:

```python
import json, hashlib, os
m = json.load(open("IMPORT_MANIFEST.json"))
for f in m["files"]:
    assert os.path.exists(f["path"]), f["path"]
    assert hashlib.sha256(open(f["path"], "rb").read()).hexdigest() == f["sha256"], f["path"]
print("OK", m["n_pairs"], "pairs,", len(m["files"]), "files")
```

## Status

This evidence makes the **baseline comparison** available and frozen. It does **not** by itself
establish where TRUCE-Rec stands relative to these baselines — that depends on TRUCE-Rec's own
method runs under the same protocol, which are tracked separately. Treat the numbers here as the
fixed reference TRUCE-Rec is measured against.
