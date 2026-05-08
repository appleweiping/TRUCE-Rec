# Qwen3-8B Controlled Baselines

This protocol defines the main fair-comparison lane for external LLM4Rec
projects. It is separate from official upstream reproduction.

## Principle

The main framework comparison should control the LLM base model and the TRUCE
data/evaluation protocol. Otherwise, a result can reflect different base
models, checkpoints, tokenizers, candidate sets, or evaluator code rather than
the project framework.

This suite is specifically for comparing TRUCE/CU-GR against reference-paper
LLM4Rec project families recommended for the paper baseline set. The goal is
not merely to reproduce their original checkpoints; it is to test whether our
framework remains stronger when each project uses the same Qwen3-8B base model,
LoRA adaptation, and the same TRUCE data/evaluator contract. Project-specific
modules such as collaborative encoders, alignment layers, and prompt/objective
logic should follow each baseline's official algorithm, but the main controlled
LLM comparison uses LoRA rather than mixing full fine-tuning, QLoRA-only, or
original-checkpoint protocols.

Important fidelity rule: after the fairness controls below are fixed, every
main-table baseline should stay as close as possible to the official project
implementation. We should replace only the parts required for comparability:
dataset/split/candidate ingestion, Qwen3-8B base-model loading, LoRA
compatibility, score export, and TRUCE import/evaluation. Prompt shape,
training objective, model-side modules, collaborative signal construction,
and scoring logic should come from the official repository whenever feasible.
Baseline hyperparameters should be the official defaults or reported optimal
settings; do not tune baselines on TRUCE test outcomes. If a
controlled run uses an internal TRUCE adapter rather than official project
code, label it as a pilot or non-official controlled adapter, not as a main
official-native baseline.

Controlled main-table candidates must use:

- base model: `/home/ajifang/models/Qwen/Qwen3-8B`;
- trainable adaptation: LoRA for every main compared LLM baseline, while
  retaining baseline-specific official project modules where feasible;
- hyperparameters: official default or reported optimal baseline settings,
  with compatibility changes recorded; Ours may tune on validation only;
- dataset: canonical TRUCE processed splits;
- candidates: fixed TRUCE candidate sets with target inclusion unchanged;
- evaluator: TRUCE evaluator only;
- import: `candidate_scores.csv -> predictions.jsonl -> metrics.json`;
- split filter: `scripts/import_external_predictions.py --split test`.

In addition to ranking metrics, each controlled baseline should be fed into the
observation analysis layer when possible. We need to check whether the
observation phenomena motivating TRUCE/CU-GR also appear in TALLRec/OpenP5/
DEALRec/LC-Rec style baselines, instead of only showing the phenomenon on a
weak base model.

Official upstream reproductions can still be useful, but they belong in a
separate reference/appendix table if they use T5, LLaMA, Vicuna, or a project
checkpoint not shared by the controlled comparison.

## Official Baseline Families

The first controlled suite used four baseline families. For final
main-table claims, the target implementation is official-native controlled:
official framework code with only the fairness substitutions listed above.
The current TRUCE-side Qwen3 LoRA adapters are useful for pipeline validation
and early diagnostics, but they are not sufficient by themselves to claim full
official-framework reproduction.

The current official baseline pool has six families:

| Baseline | Comparison role | Status |
| --- | --- | --- |
| TALLRec-Qwen3-LoRA | instruction tuning for recommendation | adapter pilot configured; official-native audit required |
| OpenP5-style-Qwen3-LoRA | P5-style generative/sequential recommendation | adapter pilot configured; official-native audit required |
| DEALRec-Qwen3-LoRA | data-efficient LLM4Rec | adapter pilot configured; official-native audit required |
| LC-Rec-Qwen3-LoRA | LLM plus collaborative signal | adapter pilot configured; official-native audit required |
| LLaRA-Qwen3-LoRA | LLM plus recommendation-signal alignment | main family; packet/config added; official-native implementation required |
| LLM-ESR-Qwen3-LoRA | long-tail sequential LLM4Rec | main family; packet/config added; official-native implementation required |

The first four cover the core external-framework families without spreading the
first experiment phase too thin. LLaRA is added as a stronger behavioral-signal
alignment baseline, and LLM-ESR is added for long-tail/sequential robustness.
CoLLM and SLMRec remain useful follow-up candidates.

All six families are treated as main official baseline families. Their analysis
emphasis can differ, for example LLM-ESR should receive long-tail/sequential
slices, but it should not be treated as a weaker or optional baseline tier.

Source check for the two added families:

- LLaRA: SIGIR 2024 project/paper page lists code at
  `https://github.com/ljy0ustc/LLaRA`.
- LLM-ESR: the Applied-Machine-Learning-Lab GitHub and the NeurIPS 2024 poster
  page identify the work as the NeurIPS 2024 implementation/poster. The
  `liuqidong07/LLM-ESR` repository is the upstream personal repository noted by
  GitHub.

The current Amazon Beauty processed set is useful for pipeline and early
comparison, but it may still be too small for final top-conference claims. When
the larger same-server dataset from the parallel data project is ready, rerun
the same controlled suite without changing the evaluator or candidate protocol.
The large protocol is documented in
`docs/week8_large_same_candidate_protocol.md`.

### TALLRec-Qwen3-LoRA

- Config:
  `configs/server/controlled_baselines/tallrec_qwen3_lora_amazon_beauty.yaml`
- Packet:
  `outputs/server_packets/tallrec_amazon_beauty`
- Training input:
  pairwise Yes/No instruction rows from the TALLRec packet.
- Scoring:
  compute Yes likelihood/logit for every test candidate, then write
  `candidate_scores.csv`.
- Paper label after completion:
  `TALLRec-Qwen3-LoRA (controlled)`.

### OpenP5-Style-Qwen3-LoRA

- Config:
  `configs/server/controlled_baselines/openp5_style_qwen3_lora_amazon_beauty.yaml`
- Packet:
  `outputs/server_packets/openp5_amazon_beauty`
- Training input:
  OpenP5-style sequential prompts using mapped item tokens.
- Scoring:
  compute causal-LM likelihood for each candidate item token, then write
  `candidate_scores.csv`.
- Paper label after completion:
  `OpenP5-style-Qwen3-LoRA (controlled)`.

This is not an official OpenP5 T5/P5 result. It is the controlled-backbone
adaptation needed for a fair framework comparison.

### DEALRec-Qwen3-LoRA

- Config:
  `configs/server/controlled_baselines/dealrec_qwen3_lora_amazon_beauty.yaml`
- Packet:
  `outputs/server_packets/dealrec_amazon_beauty`
- Training input:
  pairwise data-efficient project-style prompts from the generic packet, with
  the target plus fixed sampled negatives.
- Scoring:
  compute `Yes.` likelihood for every TRUCE candidate prompt.
- Paper label after completion:
  `DEALRec-Qwen3-LoRA (controlled)`.

### LC-Rec-Qwen3-LoRA

- Config:
  `configs/server/controlled_baselines/lc_rec_qwen3_lora_amazon_beauty.yaml`
- Packet:
  `outputs/server_packets/lc_rec_amazon_beauty`
- Training input:
  pairwise collaborative-signal prompts using user history and candidate item
  text, with the target plus fixed sampled negatives.
- Scoring:
  compute `Yes.` likelihood for every TRUCE candidate prompt.
- Paper label after completion:
  `LC-Rec-Qwen3-LoRA (controlled)`.

## Prepare Server Inputs

On the server, after pulling the latest repository:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

python scripts/prepare_project_baseline_packet.py \
  --config configs/server/project_baselines/tallrec_amazon_beauty_packet.yaml
python scripts/prepare_project_baseline_packet.py \
  --config configs/server/project_baselines/openp5_amazon_beauty_packet.yaml

python scripts/prepare_qwen_lora_controlled_baseline.py \
  --config configs/server/controlled_baselines/tallrec_qwen3_lora_amazon_beauty.yaml
python scripts/prepare_qwen_lora_controlled_baseline.py \
  --config configs/server/controlled_baselines/openp5_style_qwen3_lora_amazon_beauty.yaml
python scripts/prepare_qwen_lora_controlled_baseline.py \
  --config configs/server/controlled_baselines/dealrec_qwen3_lora_amazon_beauty.yaml
python scripts/prepare_qwen_lora_controlled_baseline.py \
  --config configs/server/controlled_baselines/lc_rec_qwen3_lora_amazon_beauty.yaml
```

Or prepare the full first suite:

```bash
python scripts/prepare_controlled_baseline_suite.py
```

Each controlled-baseline output directory contains:

- `train_sft.jsonl`
- `valid_sft.jsonl`
- `test_score_plan.jsonl`
- `controlled_baseline_manifest.json`
- `server_command_plan.md`

`prepare_controlled_baseline_suite.py` also writes a smoke run queue:

```text
outputs/server_training/controlled_baselines/qwen3_lora_main4_amazon_beauty/server_run_queue.sh
```

Newer suite preparations use:

```text
outputs/server_training/controlled_baselines/qwen3_base_adapter_main4_amazon_beauty/server_run_queue.sh
```

Run that queue first to validate the four training/scoring paths with tiny
limits, then remove `--max-*` flags for the full runs.

The server training runner must create:

- LoRA adapter checkpoint or checkpoint reference;
- `candidate_scores.csv`;
- stdout/stderr logs;
- environment and git info;
- imported TRUCE `predictions.jsonl`;
- TRUCE `metrics.json` and `metrics.csv`.

For a paper-eligible controlled baseline, also preserve enough raw artifacts to
run observation diagnostics:

- candidate score distributions;
- selected top candidates and target positions;
- prompt/source template metadata;
- latency/runtime summary;
- enough per-example metadata to slice long-tail, validity, and observation
  phenomena by bucket.

## Pilot Adapter Budget

The TRUCE-side adapter pilot currently uses this budget for smoke/full pipeline
validation. It is not the final rule for official-native baselines; official
baselines should keep their official adapter/training design when feasible.

Initial pilot budget, used only for controlled-adapter diagnostics unless it
matches an audited official setting:

- epochs: 1;
- batch size: 1;
- gradient accumulation: 16;
- learning rate: 2e-4;
- warmup ratio: 0.03;
- LoRA rank: 16;
- LoRA alpha: 32;
- dropout: 0.05;
- target modules:
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`;
- dtype: bf16;
- gradient checkpointing: true.

If memory fails in a pilot run, change the budget consistently across the pilot
suite and record the reason in each manifest. For official-native runs, record
the official hyperparameter source and why any official training detail had to
be changed for Qwen3-8B-LoRA compatibility.

## Current Execution Status

As of 2026-05-06, all four Main4 controlled baselines have completed server
smoke runs with `--max-train-examples 128`, `--max-steps 5`, and
`--max-score-rows 2`.

- TALLRec-Qwen3-LoRA: smoke completed.
- OpenP5-style-Qwen3-LoRA: smoke completed but scoring is currently too slow
  for full run without optimization.
- DEALRec-Qwen3-LoRA: smoke completed after changing generic prompts to
  pairwise `Yes.` likelihood.
- LC-Rec-Qwen3-LoRA: smoke completed after changing generic prompts to pairwise
  `Yes.` likelihood.

Next full-run order:

1. TALLRec-Qwen3-LoRA.
2. DEALRec-Qwen3-LoRA.
3. LC-Rec-Qwen3-LoRA.
4. OpenP5-style-Qwen3-LoRA only after scoring optimization.

## Evidence Boundary

Zero-shot Qwen scoring, deterministic smoke scoring, and official-project
packets are not controlled LoRA results. A controlled baseline is complete only
after real LoRA training, candidate scoring, TRUCE import, and TRUCE evaluation.
