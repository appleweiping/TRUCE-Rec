# RankCRC — Codex-found mismatch reconciliation

**Date:** 2026-06-15 · **Scope:** the two discrepancies Codex flagged between the CALM-Rec spec and
the frozen beauty Stage-B artifacts. Both are **reconciled by documentation here**; neither is a
silent edit, and both GPU-side follow-ups are explicitly deferred to the 8-domain rollout.

---

## Mismatch 1 — `D_uk` panel-discriminativeness deferred at γ=0

**Where.** `src/llm4rec/methods/calm_qwen.py` (module docstring, lines ~21-24) and
`outputs/calm/beauty_frozen_v2/stage_b/calm_stage_b_meta.json`:

```json
"gamma_D_uk": 0.0,
"gamma_note": "panel-discriminativeness D_uk deferred (panel-free encoder contract)"
```

**Spec text (method_calm_rec_spec.md §3).** `π_uk = softmax_k[ g_uk + λ·E_uk + γ·D_uk ]`, with
`D_uk = stop_grad(Var_{i∈C}(e_uik − s_prior_i))` the *prior-residualized panel discriminativeness*.

**Why it is γ=0 in the artifact.** `D_uk` is computed over the candidate panel `C` (the 101
candidates). But the `QwenIntentEncoderRuntime` contract is **panel-free**: `encode_intents` /
`intents_from_hiddens` see only the user history, never the candidate panel. Computing a non-zero
`D_uk` at inference would require threading the 101-candidate panel into the intent encoder, which (a)
changes the encoder contract and (b) risks reintroducing the same-candidate-protocol confound (intent
salience reading the popularity-sampled negatives). The author deliberately set `γ=0` and recorded it
in the artifact for honest reporting rather than silently dropping the term.

**Resolution.**
- **Status: intended design choice, not a bug.** The discrepancy is between the *aspirational* spec
  term and the *shipped, contract-clean* implementation. Documented, not patched.
- **Orthogonal to RankCRC.** RankCRC consumes the final per-user score vector `s_u` and the label-free
  confidence `g(u)`. `D_uk` only modulates intent salience *upstream* of `s_u`; with `γ=0` the scores
  are still well-defined and produce the validated NDCG@10 = 0.1143. The certificate does not depend on
  `D_uk`.
- **Deferred to GPU.** A panel-conditioned `D_uk` (with a revised encoder contract + leave-target-out
  guard) is a Stage-B GPU ablation for the 8-domain rollout, tracked as future work. It is **not** a
  blocker for the beauty certificate and is explicitly out of scope for this CPU milestone.

---

## Mismatch 2 — Stage-B artifact `n_intents=2` vs spec `K=4`

**Where.** `outputs/calm/beauty_frozen_v2/stage_b/calm_stage_b_meta.json` and
`outputs/calm/beauty_frozen_v2/sasrec/calm_rec_verdict.json` both record `n_intents: 2`, while
`method_calm_rec_spec.md §5` specifies **K=4 for sparse domains (beauty)**.

**Root cause (two different trainers, two different defaults).**
- `scripts/train_calm_stage_b.py` — the **Qwen3-8B + LoRA GPU** Stage-B trainer — defaults
  `--n-intents 4`, matching the spec K=4.
- `scripts/train_calm_sasrec.py` — the **CPU SASRec head over cached frozen Qwen item embeddings** —
  defaults `--n-intents 2`, and this is the trainer that produced the **frozen beauty signals**
  (`signals_{val,test}_sasrec.npz`, raw NDCG@10 = 0.1143) and `calm_rec_verdict.json`.

The frozen beauty artifacts are therefore the **CPU SASRec proxy at K=2**, not the GPU Qwen Stage-B at
K=4. Verified empirically: the responsibility entropy `H` in the signals lies in `[0, 0.693]`, and
`ln 2 ≈ 0.6931` is exactly the maximum entropy of a 2-way responsibility distribution — confirming
K=2 in the frozen signals.

**Resolution.**
- **Status: documented discrepancy between the CPU proxy and the GPU spec target; not a silent edit.**
  We do **not** rewrite the spec to K=2, and we do **not** relabel the artifact. The spec K=4 stands as
  the GPU Stage-B target; the artifact K=2 stands as the validated CPU proxy.
- **No effect on the RankCRC beauty certificate.** K controls only the dimensionality of the
  responsibility-entropy *feature family* (`H_top`, `H_min/max/mean/std`, `H_rank_of_top`) consumed by
  the label-free estimator `g`. Those features are computed and present at K=2; the certificate is over
  the final score vector `s_u` and `g(u)`, both well-defined at any K. The served-slice guarantee
  (Thm 1), the Pareto (Thm 2), and the decomposition (Prop 3) are K-agnostic.
- **Deferred to GPU.** The K=4 Qwen3-8B Stage-B re-run (and a K∈{1,2,4,8} intent ablation per the
  CALM-Rec falsifiability ladder) is part of the 8-domain rollout, which is GPU-gated (pony owns the
  4090). The beauty milestone validates RankCRC end-to-end on the K=2 CPU proxy; the K=4 confirmation
  is a rollout deliverable, not a local blocker.

---

## Net effect on the milestone

Neither mismatch blocks the RankCRC beauty certificate, because RankCRC is **model-agnostic** — it
certifies whatever score vector + label-free confidence it is handed. Both items are GPU-side
follow-ups for the 8-domain rollout and are flagged GREEN-with-caveats in the milestone REPORT:
the local certificate is valid for the shipped CPU K=2 raw scorer; the K=4 Qwen Stage-B and the
panel-conditioned `D_uk` are deferred, not silently resolved.
