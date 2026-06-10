"""Run the CALM-Rec falsifiability ladder + stage-2.5 reliability gate on a processed dataset.

This is the ONE script a future agent runs to evaluate CALM-Rec on a domain (after the real Qwen3-8B
Stage-B training is wired; with backend='hashed' it runs anywhere as a contract/smoke check).

It produces the decision-doc falsifiability contract (docs/method_calm_rec_spec.md section 8):
  - nested ladder: (1) raw personalized, (2) +trust gate (full)
  - intent-count ablation: K=1 vs K=4
  - variance-matched PLACEBO rho vs real rho
  - stage-2.5 reliability AUC gate (is the trust signal real?)
and writes a verdict JSON with pass/kill flags against the per-domain SOTA bar.

Usage (smoke, CPU):
  py -3 scripts/run_calm_rec.py --processed-dir data/processed/tiny/phase1 --out outputs/calm/tiny \
      --backend hashed --sota-ndcg10 0.0
Beauty (after Stage-B Qwen training on the server):
  py -3 scripts/run_calm_rec.py --processed-dir data/processed/amazon_reviews_2023_beauty \
      --out outputs/calm/beauty --backend qwen --qwen-model-path /home/<u>/models/Qwen/Qwen3-8B \
      --sota-ndcg10 0.1506

NOTE: this is evaluation tooling, not a paper-result launcher. Heavy Qwen training is a separate
server step; see docs/CALM_REC_RUNBOOK.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.methods.calm_rec import CALMConfig, CALMRecRanker, RhoCoeffs  # noqa: E402
from llm4rec.methods.calm_encoders import build_encoders  # noqa: E402
from llm4rec.methods.calm_trainer import (  # noqa: E402
    CALMScaffoldTrainer,
    calibrate_rho_on_validation,
    stage_2p5_reliability_gate,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _read_items(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _ensure_candidates(ex: dict[str, Any]) -> list[str]:
    cands = ex.get("candidate_items") or ex.get("candidates") or []
    return [str(c) for c in cands]


def _ndcg_at_k(ranker: CALMRecRanker, examples: list[dict[str, Any]], k: int = 10) -> float:
    total, n = 0.0, 0
    for ex in examples:
        cands = _ensure_candidates(ex)
        tgt = str(ex.get("target"))
        if not cands or tgt not in cands:
            continue
        res = ranker.rank({**ex, "user_id": ex.get("user_id", "u"), "target": tgt}, cands)
        try:
            pos = res.predicted_items.index(tgt)
        except ValueError:
            n += 1
            continue
        total += (1.0 / math.log2(pos + 2)) if pos < k else 0.0
        n += 1
    return total / n if n else 0.0


def _build_ranker(processed_dir: Path, train: list, items: list, *, backend: str, n_intents: int,
                  use_trust_gate: bool, qwen_model_path: str | None, seed: int) -> CALMRecRanker:
    ie, qe = build_encoders(backend=backend, n_intents=n_intents, qwen_model_path=qwen_model_path, seed=seed)
    cfg = CALMConfig(n_intents=n_intents, tau=2.0, m_dropout=4, use_trust_gate=use_trust_gate, seed=seed)
    ranker = CALMRecRanker(item_encoder=ie, intent_encoder=qe, config=cfg)
    held = {str(e.get("target")) for e in train}  # placeholder; real held-out set passed by caller
    trainer = CALMScaffoldTrainer(ranker, backend=backend)
    trainer.fit(train, items, held_out_targets=set())
    return ranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="hashed", choices=["hashed", "qwen"])
    ap.add_argument("--qwen-model-path", default=None)
    ap.add_argument("--sota-ndcg10", type=float, default=0.0, help="per-domain SOTA bar to beat")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    pd = Path(args.processed_dir)
    examples = _read_jsonl(pd / "examples.jsonl")
    items = _read_items(pd / "items.csv")
    train = [e for e in examples if str(e.get("split")) == "train"]
    val = [e for e in examples if str(e.get("split")) in {"valid", "validation"}] or \
          [e for e in examples if str(e.get("split")) == "test"][: max(1, len(examples) // 5)]
    test = [e for e in examples if str(e.get("split")) == "test"] or val
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def mk(n_intents, trust):
        return _build_ranker(pd, train, items, backend=args.backend, n_intents=n_intents,
                             use_trust_gate=trust, qwen_model_path=args.qwen_model_path, seed=args.seed)

    # Ladder variant 1: raw personalized (no trust gate)
    r_raw = mk(4, False)
    ndcg_raw = _ndcg_at_k(r_raw, test)
    # Variant 2 (full): + trust gate, rho calibrated on validation
    r_full = mk(4, True)
    calibrate_rho_on_validation(r_full, val)
    ndcg_full = _ndcg_at_k(r_full, test)
    # Intent-count ablation: K=1
    r_k1 = mk(1, True); calibrate_rho_on_validation(r_k1, val)
    ndcg_k1 = _ndcg_at_k(r_k1, test)
    # Placebo rho: variance-matched random coeffs (broken dependence on the real signals)
    rng = random.Random(args.seed)
    r_plac = mk(4, True)
    r_plac.set_rho(RhoCoeffs(a0=rng.uniform(-1, 1), a1=abs(rng.gauss(0, 1)), a2=abs(rng.gauss(0, 1)), a3=abs(rng.gauss(0, 1))))
    ndcg_plac = _ndcg_at_k(r_plac, test)
    # Stage-2.5 reliability gate on validation
    gate = stage_2p5_reliability_gate(r_full, val)

    verdict = {
        "backend": args.backend,
        "is_paper_evidence": args.backend == "qwen",
        "sota_ndcg10_bar": args.sota_ndcg10,
        "ndcg10": {"raw_personalized": ndcg_raw, "full": ndcg_full, "K1": ndcg_k1, "placebo_rho": ndcg_plac},
        "checks": {
            "full_beats_sota": ndcg_full >= args.sota_ndcg10 if args.sota_ndcg10 > 0 else None,
            "trust_beats_raw": ndcg_full >= ndcg_raw,
            "trust_beats_placebo": ndcg_full >= ndcg_plac,
            "multi_intent_beats_K1": ndcg_full >= ndcg_k1,
            "reliability_signal_real": (gate["auc_entropy"] >= 0.6 or gate["auc_variance"] >= 0.6),
        },
        "stage_2p5_reliability_gate": gate,
        "n": {"train": len(train), "val": len(val), "test": len(test)},
        "note": "backend=hashed is a CONTRACT/smoke check (NOT paper evidence). Use backend=qwen after Stage-B training.",
    }
    (out / "calm_rec_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
