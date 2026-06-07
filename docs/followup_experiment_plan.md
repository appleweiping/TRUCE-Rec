# Follow-up Experiment Plan (after the 8-domain performance table)

> Scope: what to do **after** the main performance table is complete — i.e. after TRUCE-Rec's own
> method has 8-domain results alongside the 8 frozen official baselines. These are the experiments
> that take the paper from "we have a results table" to "submittable". Sourced from the user's
> discussion with their senior advisor (2026-06-07). Do not start these until the performance phase
> is done (method validated SOTA-or-competitive on beauty, then rolled out to the other 7 domains).

## Why these, and in what order

The senior advisor's guidance: once the main performance table exists, three more experiments make
the paper essentially ready to submit (plus an overview figure for the framework). Remaining
experiments, if any, are補充 after these.

The three required experiments:

1. **Observation** — motivate *why* uncertainty is the right lens for this framework.
2. **Ablation study** — show each designed component actually contributes.
3. **Hyper-parameter analysis** — show the method is stable across hyper-parameter choices.

Plus: an **overview figure** of the whole framework.

---

## 1. Observation experiment (motivation)

**Goal.** Empirically justify the design choice — show that recommendation uncertainty is real,
structured, and exploitable, so the reader understands *why* the framework is built around
uncertainty rather than it being an arbitrary modeling choice.

**Key advisor points (verbatim intent):**
- It does **not** need an expensive frontier/paid general model. **Use the baseline models
  themselves to observe uncertainty.** No need for the most SOTA general LLM.
- It does **not** need all domains — the cited ICLR example observed on **just two baseline
  models**. Pick a small, representative slice.
- Data-volume requirement is **low** — observation is diagnostic, not a full training run.

**Reference exemplar.** An ICLR paper does its observation using **two baseline models** (not a
giant general model, not all domains). Mirror that economy: a couple of baselines on a couple of
domains is enough to make the motivating point.

**What we already have / status.** Earlier work already did a version of this — "look at baseline
models' uncertainty, not just the base model" — but possibly **not on all domains (maybe ~4) and
across the 8 baselines**, and it is somewhat stale. Revisit, tighten to the cleanest 2-baseline /
small-domain slice that tells the motivation story, and produce a clean figure/table.

**Concrete plan.**
- Pick ~2 representative official baselines (e.g. a strong one like LLMEmb + a profile one like
  ProEx) on a small set of domains (beauty is cheapest; optionally one large domain).
- Compute the uncertainty observation panel from the existing `RESEARCH_IDEA.md` schema:
  confidence vs correctness (ECE/Brier/risk-coverage), confidence vs popularity bucket, long-tail
  under/over-confidence, high-confidence-wrong rate, history-similarity / echo proxies.
- Output: a reliability-diagram / scatter / bar figure (matplotlib) + a small table.
- Label: **diagnostic / observation**, not a performance claim.
- Reuse existing observation hooks under `outputs/observations/` and the Phase-3 observation
  pipeline rather than building new infrastructure.

**Open decision to confirm with user before running:** exactly which 2 baselines × which
domain(s); whether to refresh the older ~4-domain observation or keep it as appendix.

<!-- APPEND-MARKER -->

---

## 2. Ablation study

**Goal.** Show that each component the method introduces actually earns its place. The advisor's
sharp framing: **remove a component; if performance is unchanged — or improves — that component is
badly designed.** A good ablation shows monotone degradation when you remove real contributors.

**Concrete plan.**
- Enumerate the final method's components (locked in `docs/method_redesign_decision.md`). Each
  ablation variant disables exactly one component, everything else held fixed.
- Run on the same same-candidate protocol; report the primary metrics (HR@10, NDCG@10, MRR) per
  variant, ideally on beauty first (cheapest) then 1-2 larger domains.
- Read each result honestly: a component whose removal doesn't hurt (or helps) is flagged for
  redesign or removal, not hidden.
- The existing ablation scaffolding (`docs/ablation_protocol.md`,
  `src/llm4rec/methods/ablation.py`) is the starting point but **must be re-derived against the new
  method's components**, not the retired uncertainty-gate ones.

**Acceptance.** Each retained component shows a measurable, defensible contribution; the ablation
table is consistent with the method narrative.

---

## 3. Hyper-parameter analysis

**Goal.** Show the method is **stable** — not a knife-edge that only works at one magic setting.

**Advisor's concrete recipe (verbatim intent):**
- Take each hyper-parameter the method exposes (e.g. learning rate, λ weights, thresholds).
- Sweep it across orders of magnitude while holding others fixed. Example given: if `learning_rate
  = 1e-3`, run `{1e-1, 1e-2, 1e-3, 1e-4, 1e-5}` separately.
- Plot a **performance line chart** per hyper-parameter (x = hyper-parameter value, y = metric).
- A flat-ish / smooth curve around the chosen value = stable method.

**Concrete plan.**
- List the method's tunable hyper-parameters from `docs/method_redesign_decision.md`.
- For each, define a 4-5 point sweep (≥1 order of magnitude span) with all others fixed at the
  validation-selected default.
- Run on beauty (cheapest) — and optionally one large domain — under the same protocol.
- One matplotlib line chart per hyper-parameter; overlay the validation-selected operating point.
- Selection uses validation only; **never tune on test**.

**Acceptance.** Curves show the method is robust in a neighborhood of the chosen settings; no
cherry-picked single point.

---

## 4. Overview figure (framework diagram)

**Goal.** One clear figure of the whole TRUCE-Rec framework for the paper's method section.

**Advisor's guidance on figures generally:**
- Plots (line charts, reliability diagrams, etc.): **matplotlib** — "let an LLM write the code, it's
  simple."
- The **overview / framework figure**: hand-draw in **PPT/draw.io**, or LLM-assisted generation.
- Once the paper has these (performance table + observation + ablation + hyper-parameter +
  overview), it is basically ready to submit; any further experiments are补充.

**Concrete plan.**
- Draft the framework figure after the method is locked: inputs (user history, candidate panel) →
  the method's components (as named in the decision doc) → scoring → same-candidate evaluation.
- Keep a source-editable version (draw.io `.drawio` or PPT) in `paper/figures/` plus an exported
  PDF/PNG.

---

## Tooling / cost notes

- **Plots:** matplotlib; LLM-generated plotting code is fine and expected.
- **Models for observation:** baseline models, **not** a paid frontier model. Low data requirement.
- **Scale:** observation and hyper-parameter sweeps are cheap (small slices / beauty-first); only
  the main performance table needs full 8-domain scale.

## Gating

Do not begin these until:
1. the redesigned method clears the ARIS ≥8/10 design gate, and
2. it is validated SOTA-or-competitive on beauty and rolled out to the 8-domain performance table.

All runs are server-side and obey the local↔server discipline (lightweight evidence back to local,
push from local only). The server is currently **busy with another project — wait for the user's
go-ahead** before any run.

