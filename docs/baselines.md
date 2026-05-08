# Baselines

This document describes implemented and planned baselines. It contains no
experimental conclusions.

## Implemented baselines

### Random

- Input signals: candidate item IDs and seed.
- Training data used: none.
- Forbidden signals: target labels, future interactions.
- Candidate protocol: shared candidate set.
- Output schema: unified prediction schema.
- Status: smoke-capable sanity baseline.

### Popularity

- Input signals: train-split item frequencies.
- Training data used: train examples only.
- Forbidden signals: validation/test popularity, target correctness.
- Candidate protocol: shared candidate set.
- Output schema: unified prediction schema.
- Status: smoke-capable baseline; paper-ready only after real protocol runs.

### BM25

- Input signals: history item text and catalog item text.
- Training data used: catalog text; no held-out target text except when present
  in allowed history.
- Forbidden signals: target title as query, future interactions.
- Candidate protocol: shared candidate set.
- Output schema: unified prediction schema.
- Status: smoke-capable baseline.

### MF

- Input signals: train interactions/examples.
- Training data used: train split only.
- Forbidden signals: held-out interactions.
- Candidate protocol: shared candidate set.
- Output schema: unified prediction schema.
- Status: minimal MF is smoke-capable unless strengthened later.

### Sequential Markov

- Input signals: train transition counts and evaluation history.
- Training data used: train examples only.
- Forbidden signals: future sequence transitions.
- Candidate protocol: shared candidate set.
- Output schema: unified prediction schema.
- Status: smoke-capable sequential fallback/baseline.

### SASRec via RecBole adapter

- Input signals: exported canonical train interactions and evaluation histories.
- Training data used: train examples only.
- Forbidden signals: held-out targets and future interactions.
- Candidate protocol: shared candidate set.
- Output schema: imported into unified prediction schema after external scoring.
- Status: adapter/config/export/training/scoring/import are implemented. MovieLens 1M and Amazon Beauty completed end-to-end on CPU with RecBole 1.2.1 and TRUCE evaluator metrics saved under `outputs/runs/movielens_1m_r2_sasrec_recbole_seed13/` and `outputs/runs/amazon_reviews_2023_beauty_cu_gr_v2_sasrec_recbole_seed13/`.

### BERT4Rec via RecBole adapter

- Input signals: exported chronological histories with `item_id_list`.
- Training data used: train examples only.
- Forbidden signals: held-out targets and future interactions.
- Candidate protocol: shared candidate set.
- Output schema: imported into unified prediction schema after external scoring.
- Status: adapter/config/export/training/scoring/import are implemented. MovieLens 1M and Amazon Beauty completed end-to-end on CPU with TRUCE evaluator metrics saved under `outputs/runs/movielens_1m_r2_bert4rec_recbole_seed13/` and `outputs/runs/amazon_reviews_2023_beauty_cu_gr_v2_bert4rec_recbole_seed13/`.

### GRU4Rec via RecBole adapter

- Input signals: exported chronological histories with `item_id_list`.
- Training data used: train examples only.
- Forbidden signals: held-out targets and future interactions.
- Candidate protocol: shared candidate set.
- Output schema: imported into unified prediction schema after external scoring.
- Status: adapter/config/export/training/scoring/import are implemented. MovieLens 1M and Amazon Beauty completed end-to-end on CPU with TRUCE evaluator metrics saved under `outputs/runs/movielens_1m_r2_gru4rec_recbole_seed13/` and `outputs/runs/amazon_reviews_2023_beauty_cu_gr_v2_gru4rec_recbole_seed13/`.

### LightGCN via RecBole adapter

- Input signals: exported canonical train interactions.
- Training data used: train examples only.
- Forbidden signals: held-out targets, validation/test popularity leakage, and external evaluator metrics as final paper numbers.
- Candidate protocol: shared candidate set.
- Output schema: imported into unified prediction schema after external scoring.
- Status: adapter/config/export/training/scoring/import are implemented. MovieLens 1M and Amazon Beauty completed end-to-end on CPU with RecBole 1.2.1 and TRUCE evaluator metrics saved under `outputs/runs/movielens_1m_r2_lightgcn_recbole_seed13/` and `outputs/runs/amazon_reviews_2023_beauty_cu_gr_v2_lightgcn_recbole_seed13/`.

## Optional RecBole Environment

Tested local environment: Python 3.12.0, torch 2.10.0+cpu, RecBole 1.2.1, NumPy 1.26.4, SciPy 1.11.4, Ray 2.55.1 compatibility workaround, CPU only on Intel Core i5-1240P. CUDA was unavailable. RecBole 1.2.1 declares `ray<=2.6.3`, but that Ray range has no Python 3.12 Windows wheel; a cleaner paper environment should use Python 3.10 or 3.11 with the pinned baseline extra.

Install command used here:

```powershell
py -3 -m pip install recbole==1.2.1 --no-deps -i https://pypi.org/simple
py -3 -m pip install colorlog==4.7.2 colorama==0.4.4 tensorboard thop tabulate plotly texttable psutil "ray[tune]" "numpy<2" "scipy==1.11.4" -i https://pypi.org/simple
```

### LLM generative / rerank / confidence observation

- Input signals: target-excluding prompt context and visible candidates.
- Training data used: none for API/mock inference.
- Forbidden signals: target title, target ID, future interactions.
- Candidate protocol: shared candidate set where applicable.
- Output schema: unified prediction schema with raw output and metadata.
- Status: MockLLM outputs are not paper evidence; real API/HF results require
  explicit config and user confirmation.

## Non-baseline components

- `skeleton` is not a baseline. It is a Phase 1 pipeline sanity check.
- OursMethod is a method under test, not a baseline.
- Fallback-only is an ablation/control for OursMethod routing.

## Paper-Grade LLM Baseline Lane

The main compared LLM baseline protocol follows the current research decision:

```text
official source implementation
  + Qwen3-8B base model
  + LoRA adaptation
  + official default or reported-optimal baseline hyperparameters
  + shared TRUCE split/candidates/evaluator
  + example_id,user_id,item_id,score
```

Selected official baseline families:

- TALLRec;
- OpenP5;
- DEALRec;
- LC-Rec;
- LLaRA;
- LLM-ESR.

Baseline hyperparameters are not tuned on TRUCE test outcomes. Ours may tune
hyperparameters only through the declared validation protocol. API-model or
MockLLM LLM rows remain diagnostics unless they are explicitly separated from
the Qwen3-8B-LoRA controlled main table.

## Limitations

Baseline strength varies by implementation maturity. Minimal smoke-capable
baselines should not be presented as paper-grade strong baselines until the real
experiment protocol, multi-seed runs, and comparable candidate sets are
complete.

RecBole-backed baselines are optional. Install with `py -3 -m pip install -e .[baselines]` in a compatible environment before running SASRec or LightGCN.

## Strong Baseline Adapter Completion Notes

MovieLens 1M SASRec: Recall@10 0.184934, NDCG@10 0.108555, MRR@10 0.085336. MovieLens 1M BERT4Rec: Recall@10 0.199172, NDCG@10 0.107392, MRR@10 0.079387. MovieLens 1M GRU4Rec: Recall@10 0.160099, NDCG@10 0.088435, MRR@10 0.066755. MovieLens 1M LightGCN: Recall@10 0.212086, NDCG@10 0.107865, MRR@10 0.076519. Amazon Beauty SASRec: Recall@10 0.013333, NDCG@10 0.005158, MRR@10 0.002667. Amazon Beauty BERT4Rec, GRU4Rec, and LightGCN: all three top-10 ranking metrics are 0.000000.

These are TRUCE evaluator metrics computed from imported RecBole candidate scores, not RecBole evaluator metrics. SASRec, BERT4Rec, and GRU4Rec use chronological item histories exported as `item_id_list`; LightGCN uses train interactions only. Validation is used only by RecBole model selection/early stopping, and test candidates are scored only after training.
