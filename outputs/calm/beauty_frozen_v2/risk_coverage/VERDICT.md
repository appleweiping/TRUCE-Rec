# TRUCE-Rec Risk-Coverage / Selective-Recommendation Verdict (beauty, 973 test users)

**Scorer:** validated Stage-2 SASRec head over cached LLM item embeddings (raw NDCG@10 = 0.1143,
HR@10 = 0.2148, MRR = 0.1062). CPU-only, no GPU / no Qwen re-encode; signals regenerated from
`sasrec_best.pt` + cached frozen item vectors and verified to reproduce the verdict exactly
(auc_entropy 0.785). Random abstention averaged over 200 seeds; oracle retains by realized
rank-of-positive.

## Verdict (1 paragraph)

Uncertainty-aware selective recommendation is a **genuine but modest** contribution on beauty, and
the headline AUC=0.785 **overstates** its deployable value. Retaining the most-confident users by the
deployable, label-free signal **H_top** (responsibility entropy at the model's top-ranked candidate)
raises NDCG@10 monotonically as coverage drops — 0.1143 → 0.1168 → 0.1171 → 0.1193 → 0.1219 → 0.1259
at coverage 100/90/80/70/60/50% — and beats random abstention at **every** operating coverage
(mean selective gain +0.0057 NDCG@10, min +0.0024, max +0.0114; AURC 0.0595 vs random 0.0572). This is
a small, consistent, label-free risk-coverage gain, i.e. a real selective-prediction effect. However,
the decisive caveat is that the validated AUC=0.785 signal is **H_target = entropy at the held-out
positive's slot**, which conditions on the label and is **not deployable**; as a selective signal
H_target is excellent (mean gain +0.0446, reaching NDCG@10 0.2012 at 50% coverage, nearly matching the
oracle ceiling of 0.2288), proving the *information* needed for selective recommendation exists, but the
honest label-free proxy (H_top, AUC 0.519) recovers only a fraction of it. MC-dropout variance
(Var_top, AUC 0.496) **fails** — its curve dips below random at 50% coverage (mean delta −0.0025),
consistent with the verdict's weak auc_variance. **Recommendation:** frame TRUCE's selective-recommendation
contribution on the *deployable* H_top result (real, monotone, beats random by ~+0.006 NDCG@10), and
present H_target / oracle as the achievable ceiling that motivates a better, label-free confidence
estimator — do not headline the 0.785 number as if it were a deployable selective-recommendation gain.

## Numbers at a glance

| Coverage | confident H_top | confident Var_top | confident H_target* | random (±std) | oracle |
|---------:|----------------:|------------------:|--------------------:|--------------:|-------:|
| 100% | 0.1143 | 0.1143 | 0.1143 | 0.1143 | 0.1143 |
|  90% | 0.1168 | 0.1137 | 0.1242 | 0.1144 | 0.1269 |
|  80% | 0.1171 | 0.1127 | 0.1388 | 0.1145 | 0.1429 |
|  70% | 0.1193 | 0.1144 | 0.1547 | 0.1146 | 0.1633 |
|  60% | 0.1219 | 0.1144 | 0.1767 | 0.1145 | 0.1904 |
|  50% | 0.1259 | 0.1049 | 0.2012 | 0.1145 | 0.2288 |

\* H_target peeks at the held-out positive → **not deployable** (diagnostic / ceiling only).

AUC predicting top-1 correctness (lower uncertainty ⇒ more confident): H_target **0.832**
(matches verdict's 0.785 definition), H_top **0.519**, Var_top **0.496**.
AURC (NDCG@10 over coverage∈[0.5,1.0]): confident_H_top 0.0595, random 0.0572, confident_H_target
0.0752, oracle 0.0795.

Full per-coverage metrics (NDCG@5/10/20, HR@5/10/20, MRR) for confident / random / oracle and both
uncertainty signals are in `risk_coverage_results.json`.
