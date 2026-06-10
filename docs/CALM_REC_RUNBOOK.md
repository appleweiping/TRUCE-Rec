# CALM-Rec Runbook — how to run experiments and write the paper

> **For any future agent.** This is the standing operating procedure for TRUCE-Rec's method
> (CALM-Rec). Everything below is already decided and implemented — you do **not** need to re-derive
> the method, re-ask the user for rules, or re-design anything. Read this + `method_calm_rec_spec.md`
> + `experimental_setting_and_baselines.md`, then **execute**. Only escalate to the user for the
> explicit gates marked **[USER GATE]**.

## 0. What is already done (do not redo)

- **Setting + baselines:** 8 domains, 101-candidate (1 pos + 100 neg), Qwen3-8B backbone. The 8
  official baselines are **frozen** in `data/official_baselines/` (64 pairs) and are **never re-run**.
  Per-domain SOTA bar in `data/official_baselines/frontier_best_baseline_per_domain.csv` (beauty =
  **ProEx NDCG@10 0.1506**; LLMEmb leads the other 7). Source of truth:
  `docs/experimental_setting_and_baselines.md`.
- **Method:** CALM-Rec is designed (20-iteration tri-agent review, ARIS 9.0/10) and the scoring core
  + 3-stage training contract + falsifiability tooling are implemented. Spec:
  `docs/method_calm_rec_spec.md`.
- **Code:** `src/llm4rec/methods/calm_rec.py` (scorer + ranker), `calm_encoders.py` (encoders +
  `build_encoders`), `calm_weak_labels.py` (attribute lexicon), `calm_trainer.py` (3-stage scaffold +
  stage-2.5 gate). Wired into the runner as method `calm_rec`. Configs:
  `configs/methods/calm_rec.yaml`, `configs/experiments/smoke_calm_rec.yaml`. Scripts:
  `scripts/build_calm_weak_labels.py`, `scripts/run_calm_rec.py`. Tests:
  `tests/unit/test_calm_rec.py`, `test_calm_encoders.py`, `tests/smoke/test_calm_rec_*.py`.

## 1. The standing discipline (applies to every task; do not ask)

1. **Local is dev, server runs experiments.** Edit/commit/push from local only; never `git push`
   from the server. Server does `git pull` → run → produce results.
2. **Package lightweight evidence back to local** (metrics/provenance/manifests, not multi-GB
   scores/checkpoints) and commit from local.
3. **Don't stop** unless a real blocker: server disk full, API/endpoint failure, an unfixable review
   rejection, or a **[USER GATE]**. Otherwise proceed through the whole pipeline autonomously.
4. **After each meaningful step:** update agentmemory (project `truce-rec`, concepts incl. `agent:cc`)
   + the relevant docs/README. Write as you go.
5. **No leakage, ever:** train-only stats with held-out targets removed; ρ/τ/λ selected on validation
   only; test split never used for selection. (Enforced in code; keep it that way.)
6. **Honesty:** a method that only wins on 973 beauty users is retracted. Report negatives. Use the
   honesty fallbacks in the spec (§9) if a mechanism doesn't earn its place.

## 2. The experiment pipeline (beauty first, then roll out)

**[USER GATE] Do not start any GPU run until the user says the server is free.** (Currently busy with
another project.) Everything in §2.1 is no-GPU and can be done now.

### 2.1 No-GPU prep (do this now; it's all local/CPU)

```bash
# 1. Build attribute weak-labels from item text for the target domain (reproducible, no model):
py -3 scripts/build_calm_weak_labels.py \
    --processed-dir data/processed/amazon_reviews_2023_beauty/<variant> \
    --out outputs/calm/beauty
# 2. Contract/smoke the full pipeline on the hashed backend (no GPU) to confirm wiring:
py -3 scripts/run_calm_rec.py --processed-dir data/processed/tiny/phase1 \
    --out outputs/calm/tiny --backend hashed --sota-ndcg10 0.0
# 3. Run the test suite:
.\.venv\bin\python.exe -m pytest tests/unit/test_calm_rec.py tests/unit/test_calm_encoders.py \
    tests/smoke/test_calm_rec_pipeline.py tests/smoke/test_calm_rec_runner.py
```

### 2.2 Wire the real Qwen3-8B encoders (no GPU to write; GPU to run)

Implement the two gated classes in `src/llm4rec/methods/calm_encoders.py`:
- `QwenItemEncoder.encode`: frozen Qwen3-8B over item text (title+brand+category+attrs+desc), mean/
  last-token pool → `h_i`; cache to an fp16 matrix per domain (offline, once).
- `QwenIntentEncoder.encode_intents`: Qwen3-8B + LoRA over the short user-side prompt
  `[system; history ≤20 as attribute tuples; K attribute-anchored soft-prompt intent slots]`; read
  `z_uk = c_k + r_uk` at the intent-slot positions and the salience features; `dropout=True` enables
  LoRA dropout for the M-pass ensemble variance. **Never put the 101 candidates in the prompt.**
- Anchors `c_k`: initialise from the attribute-prototype centroids produced from the weak-labels
  (`scripts/build_calm_weak_labels.py`), one facet per intent.
Keep the `IntentSet`/`ItemEncoder`/`IntentEncoder` contracts exactly — the scorer, trainer, and tests
already target them, so no other code changes.

### 2.3 [USER GATE] Beauty formal run on the server

Server-side, with `backend: qwen` (set `qwen_model_path` in `configs/methods/calm_rec.yaml` params):
1. **Stage A** (offline): weak-labels + anchor centroids + cached `h_i` + train-only `n_i`/`q(j)`/
   `b_i` + fit `s_prior` (use `build_train_only_stats`; the script wires this).
2. **Stage B** (GPU, LoRA): the gradient loop — listwise CE on the mixture score + L_attr/L_bal/
   L_orth/L_use/L_τ, weights in `CALMLossSpec`, τ annealed up from 1.5, **ρ held fixed mild (0.1)**.
   This is the only piece left to implement inside `QwenIntentEncoder` training; everything that
   consumes it exists.
3. **Stage C** (CPU): `calibrate_rho_on_validation(...)` — coarse grid, validation only.
4. **Stage 2.5 gate:** `stage_2p5_reliability_gate(...)` — **if AUC ≤ ~0.55, drop the trust headline**
   (ship the K-intent core with a fixed blend) per spec §9. This is an automated decision, not a
   user gate.
5. **Falsifiability ladder + verdict:**
   ```bash
   py -3 scripts/run_calm_rec.py \
       --processed-dir data/processed/amazon_reviews_2023_beauty/<variant> \
       --out outputs/calm/beauty --backend qwen \
       --qwen-model-path /home/<user>/models/Qwen/Qwen3-8B --sota-ndcg10 0.1506
   ```
   Reads `outputs/calm/beauty/calm_rec_verdict.json`. **Release requires** all of: `full_beats_sota`,
   `trust_beats_placebo`, `multi_intent_beats_K1`, `reliability_signal_real`.
6. Package `outputs/calm/beauty/` (verdict + metrics, lightweight) back to local, commit, push.

### 2.4 Decision after beauty

- **If beauty is SOTA** (beats ProEx 0.1506 + passes the falsifiability checks): roll out to the other
  7 domains (same protocol, re-fit per domain), then go to §3.
- **If NOT SOTA:** re-run the tri-agent ARIS discussion (see §4) and iterate the method on beauty
  until it is SOTA. Do not scale a losing method.

## 3. Paper writing (after the 8-domain performance table)

Once CALM-Rec has the 8-domain main table (vs the 8 frozen baselines), the paper needs three more
experiments + one figure (see `docs/followup_experiment_plan.md` for full detail — already written):
1. **Observation** — motivate uncertainty using **baseline models** (not a paid model), a small
   2-baseline / few-domain slice. Cheap.
2. **Ablation** — the pre-registered ~22-row table in `method_calm_rec_spec.md` §8 / Round-4 synthesis;
   the load-bearing rows are K=1-vs-K=4, real-ρ-vs-placebo, no-attribute-anchor, no-per-intent-pop.
   `scripts/run_calm_rec.py` already emits the core ladder; extend it for the full table.
2b. The headline-defending rows: report `p_ψ` attribute accuracy (anchor binds) + the reliability
   curve (binned ρ vs realized error, Spearman, ECE/Brier).
3. **Hyperparameter analysis** — sweep each hyperparameter over orders of magnitude (e.g. K∈{1,2,4,8},
   τ schedule, λ/β), one matplotlib line chart each; show stability.
4. **Overview figure** — draw.io/PPT or LLM-assisted; inputs → K attribute-anchored intents → mixture
   energy → trust gate vs prior → 101 scores.

Then: fill the main table, run the internal top-conference review gate
(`docs/top_conference_review_plan.md`), prep submission. **[USER GATE]** before declaring "ready to
submit" or doing the actual paper write-up — surface the evidence and let the user decide.

## 4. The tri-agent discussion protocol (when a redesign/iteration is needed)

Standing rule (also in user memory `multi-agent-discussion-rule`): when the user says "你和你分身和
GPT三个人讨论一下" — or when CALM-Rec fails a beauty gate and §2.4 sends you back — run the 3-seat
ARIS protocol:
- **Seats:** Opus lead (you, synthesis + ruling) + a second Opus 4.8 (via the Agent tool) + GPT-5.5
  xhigh (via the relay `OPENAI_BASE_URL`/`OPENAI_API_KEY`, `reasoning_effort=xhigh`; see
  `outputs/method_redesign_discussion/iterations/call_gpt.sh` + `build_req.py` for the working caller
  with WAF-retry).
- **Structure:** independent fan-out → adversarial cross-critique → converge → lead synthesizes and
  scores against the ARIS ≥8/10 design gate. Keep the iteration ledger under
  `outputs/method_redesign_discussion/iterations/` (gitignored); write the tracked decision into a
  spec doc.
- The full 20-iteration trail that produced CALM-Rec is the template:
  `outputs/method_redesign_discussion/iterations/round{1..5}_synthesis.md`.

## 5. Quick reference — files

| Purpose | Path |
|---|---|
| Method spec (live) | `docs/method_calm_rec_spec.md` |
| Setting / metrics / baselines | `docs/experimental_setting_and_baselines.md` |
| Follow-up experiments plan | `docs/followup_experiment_plan.md` |
| This runbook | `docs/CALM_REC_RUNBOOK.md` |
| Scorer + ranker | `src/llm4rec/methods/calm_rec.py` |
| Encoders (hashed + Qwen-gated) | `src/llm4rec/methods/calm_encoders.py` |
| Weak labels | `src/llm4rec/methods/calm_weak_labels.py` |
| 3-stage trainer + stage-2.5 gate | `src/llm4rec/methods/calm_trainer.py` |
| Method config | `configs/methods/calm_rec.yaml` |
| Smoke experiment | `configs/experiments/smoke_calm_rec.yaml` |
| Weak-label builder | `scripts/build_calm_weak_labels.py` |
| Run ladder + verdict | `scripts/run_calm_rec.py` |
| Frozen baselines + frontier | `data/official_baselines/` |

## 6. One-paragraph summary for a brand-new agent

TRUCE-Rec compares an own method against 8 frozen official LLM4Rec baselines on 8 Amazon domains
(101-candidate rerank, Qwen3-8B). The own method is **CALM-Rec**: a single Qwen3-8B+LoRA reads K=4
attribute-anchored intents, scores each candidate by a soft-OR mixture of per-intent
popularity-residualized energies, then **mixes toward a history-free prior by a per-user-item trust
gate** whose strength comes from the intent mixture's own responsibility entropy + ensemble variance
(the headline = *calibrated personalization-trust*; multi-intent is the vehicle). It is implemented
and tested on a CPU backend; the remaining work is the Qwen Stage-B gradient loop + running it. Your
job: do the no-GPU prep in §2.1-2.2, then on the user's go-ahead run beauty (§2.3), beat ProEx
NDCG@10 0.1506 with the falsifiability checks passing, roll out to 7 domains, then the 3 follow-up
experiments (§3) and the paper. Follow §1 discipline. Don't ask for rules — they are all here.
