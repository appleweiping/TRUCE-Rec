# Controlled Baseline Fidelity Audit

This document records the stricter baseline rule for paper-grade experiments:
under the shared Qwen3-8B base-model, LoRA adaptation, and TRUCE fairness
protocol, baselines should otherwise use the official project implementation as
much as possible.

## Fidelity Rule

Allowed fairness substitutions:

- use the canonical TRUCE train/valid/test split;
- use the fixed TRUCE candidate set, with target inclusion unchanged;
- use Qwen3-8B as the shared base model;
- use LoRA adaptation in the main controlled LLM baseline table;
- use official default or reported optimal baseline hyperparameters where
  feasible;
- export `candidate_scores.csv` and import through the TRUCE evaluator;
- add minimal adapters for item/user ID mapping, config loading, and score
  export.

Not allowed for final main-table baselines unless explicitly disclosed:

- replacing an official training objective with a generic internal prompt;
- replacing an official model-side collaborative module with plain text only;
- replacing a baseline's official objective/modules with a generic adapter
  unless explicitly labeled as a pilot;
- tuning baseline hyperparameters on TRUCE test outcomes;
- silently mixing LoRA, full fine-tuning, QLoRA-only, and original checkpoint
  protocols in the same main table;
- changing candidate construction, negative sampling, or split logic;
- using deterministic, mock, smoke, or zero-shot scores as experiment results;
- presenting TRUCE-side style adapters as official reproduction.

## Current Main4 Status

| Family | Current TRUCE-side state | Final paper-grade requirement |
| --- | --- | --- |
| TALLRec | Qwen3 LoRA adapter uses TALLRec-style instruction/Yes-No packet and is closest to a project-specific controlled path. | Audit against official TALLRec code and document any unavoidable Qwen3/LoRA substitutions. |
| OpenP5 | Qwen3 LoRA adapter uses OpenP5/P5-style item-token sequential prompts; full scoring is currently too slow. | Use official OpenP5/P5 training/scoring logic where possible, with Qwen3 adapter only if the official code path can support it cleanly. |
| DEALRec | Current path is a TRUCE generic pairwise adaptation with a DEALRec family label. | Replace or validate against official DEALRec code and training objective before main-table use. |
| LC-Rec | Current path is a TRUCE generic pairwise adaptation with collaborative-history/item-text prompts. | Replace or validate against official LC-Rec collaborative-signal implementation before main-table use. |
| LLaRA | Packet/config added as a new official baseline candidate. | Reuse official recommendation-signal alignment modules before main-table use. |
| LLM-ESR | Packet/config added as a long-tail/sequential candidate. | Reuse official long-tail sequential modules and prerequisite embedding artifacts before robustness-table use. |

## Evidence Labels

- `controlled_adapter_pilot`: useful for pipeline validation, runtime planning,
  observation diagnostics, and early sanity comparison.
- `official_native_controlled`: eligible for main controlled comparison after
  official implementation fidelity, Qwen3-8B-LoRA compatibility, and official
  hyperparameter provenance are audited, full training/scoring completes, and
  TRUCE metrics are produced.
- `official_original_reference`: appendix/reference only when the official
  backbone or data protocol differs from the controlled Qwen3-8B setup.

## Immediate Next Work

1. Let the current LC-Rec adapter pilot finish if it is already running; it
   helps validate the pipeline and scoring import.
2. Import/evaluate completed adapter pilots only as non-final controlled
   diagnostics.
3. Build an official-native integration checklist per baseline repository:
   cloned commit, original config, original objective, project modules reused,
   official hyperparameter source, fairness substitutions, score export shim,
   environment, and test command.
4. Upgrade the main baselines one by one, starting with the baselines whose
   official code is easiest to adapt without changing their core method.
5. For LLaRA and LLM-ESR, begin with official-repo environment and data-entry
   audits before any long server run, because both require more than a generic
   prompt adapter to preserve their method identity.
