#!/usr/bin/env python3
"""Efficient CALM-Rec formal evaluation (qwen backend) with cached signals.

run_calm_rec.py is the CPU/hashed contract runner; this script is the formal
server-side evaluator. It makes the falsifiability ladder tractable at d=4096:

  1. ONE encoder pass per user per K (no-dropout + M dropout IntentSets), for
     val and test; per-candidate signals (s_pers, H, Var_m, n_i, s_prior) are
     computed vectorized (numpy) and CACHED to disk.
  2. Stage C (rho grid), the stage-2.5 AUC gate, the ladder variants
     (raw / full / placebo) and the K=1 ablation are all evaluated from the
     cached arrays — no model re-runs per grid point (the naive grid would be
     81 combos x 973 users x 5 forwards).

The decision criteria are IDENTICAL to run_calm_rec.py / the spec section 8.
Signals math parity with calm_rec.py is covered by tests/unit/test_calm_eval_cached.py.

Usage (server):
  python scripts/eval_calm_beauty.py \
      --processed-dir data/processed/frozen_week8_beauty \
      --weak-labels outputs/calm/beauty_frozen \
      --stage-b-dir outputs/calm/beauty_frozen/stage_b \
      --qwen-model-path ~/models/Qwen/Qwen3-8B \
      --out outputs/calm/beauty_frozen/eval --sota-ndcg10 0.1506
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from llm4rec.methods.calm_rec import RhoCoeffs, reliability_auc  # noqa: E402
from llm4rec.methods.calm_trainer import build_train_only_stats  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def signals_for_examples(
    examples: list[dict[str, Any]],
    *,
    intent_encoder,
    item_lookup: dict[str, dict[str, Any]],
    item_vecs: dict[str, list[float]],
    support: dict[str, int],
    s_prior: dict[str, float],
    tau: float,
    delta: list[float],
    m_dropout: int,
    max_history: int = 20,
) -> list[dict[str, Any]]:
    """Per example: vectorized per-candidate signals from ONE set of encoder passes.

    Math mirrors calm_rec.per_intent_energy / mixture_score_and_responsibilities
    exactly (parity-tested); vectorized over the 101 candidates.
    """
    out = []
    dim = len(next(iter(item_vecs.values())))
    delta_v = np.asarray(delta, dtype=np.float64)
    for n_done, ex in enumerate(examples):
        cands = [str(c) for c in (ex.get("candidates") or ex.get("candidate_items") or [])]
        tgt = str(ex.get("target"))
        if not cands or tgt not in cands:
            continue
        hist = [str(x) for x in (ex.get("history") or [])][-max_history:]
        intents = intent_encoder.encode_intents(hist, item_lookup, dropout=False)
        ensemble = [
            intent_encoder.encode_intents(hist, item_lookup, dropout=True)
            for _ in range(m_dropout)
        ]

        H_mat = np.array(
            [item_vecs.get(c, [0.0] * dim) for c in cands], dtype=np.float64
        )  # [N, d]
        n_i = np.array([float(support.get(c, 0)) for c in cands])
        prior = np.array([float(s_prior.get(c, 0.0)) for c in cands])
        pop = np.log1p(np.clip(n_i, 0, None))

        def panel_scores(iset):
            Z = np.array(iset.z, dtype=np.float64)               # [K, d]
            pi = np.array(iset.pi, dtype=np.float64)             # [K]
            E = H_mat @ Z.T - pop[:, None] * delta_v[None, :]    # [N, K]
            logits = np.log(np.clip(pi, 1e-12, None))[None, :] + tau * E
            m = logits.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
            s_pers = lse / tau                                    # [N]
            r = np.exp(logits - lse[:, None])                     # [N, K]
            return s_pers, r

        s_pers, resp = panel_scores(intents)
        H_ent = -(resp * np.log(np.clip(resp, 1e-12, None))).sum(axis=1)
        if ensemble:
            samples = np.stack([panel_scores(e)[0] for e in ensemble])  # [M, N]
            var_m = samples.std(axis=0)
        else:
            var_m = np.zeros_like(s_pers)

        out.append(
            {
                "user_id": str(ex.get("user_id")),
                "cands": cands,
                "target_idx": cands.index(tgt),
                "s_pers": s_pers,
                "H": H_ent,
                "var_m": var_m,
                "n_i": n_i,
                "prior": prior,
            }
        )
        if (n_done + 1) % 50 == 0:
            print(f"  signals {n_done + 1}/{len(examples)}", flush=True)
    return out


def mix_scores(sig: dict[str, Any], coeffs: RhoCoeffs | None, rho_floor: float = 0.15):
    """s_ui from cached signals (None coeffs -> raw personalized, rho=0)."""
    if coeffs is None:
        return sig["s_pers"]
    pop = np.log1p(np.clip(sig["n_i"], 0, None))
    z = coeffs.a0 + coeffs.a1 * sig["H"] + coeffs.a2 * sig["var_m"] - coeffs.a3 * pop
    rho = 1.0 / (1.0 + np.exp(-z))
    rho_eff = np.clip(rho, 0.0, 1.0 - rho_floor)
    return (1.0 - rho_eff) * sig["s_pers"] + rho_eff * sig["prior"]


def ndcg10(signals: list[dict[str, Any]], coeffs: RhoCoeffs | None) -> float:
    total, n = 0.0, 0
    for sig in signals:
        s = mix_scores(sig, coeffs)
        order = np.argsort(-s, kind="stable")
        pos = int(np.where(order == sig["target_idx"])[0][0])
        total += (1.0 / math.log2(pos + 2)) if pos < 10 else 0.0
        n += 1
    return total / n if n else 0.0


def metrics_full(signals: list[dict[str, Any]], coeffs: RhoCoeffs | None) -> dict[str, float]:
    agg = {f"NDCG@{k}": 0.0 for k in (5, 10, 20)}
    agg.update({f"HR@{k}": 0.0 for k in (5, 10, 20)})
    agg["MRR"] = 0.0
    n = 0
    for sig in signals:
        s = mix_scores(sig, coeffs)
        order = np.argsort(-s, kind="stable")
        pos = int(np.where(order == sig["target_idx"])[0][0])
        for k in (5, 10, 20):
            agg[f"NDCG@{k}"] += (1.0 / math.log2(pos + 2)) if pos < k else 0.0
            agg[f"HR@{k}"] += 1.0 if pos < k else 0.0
        agg["MRR"] += 1.0 / (pos + 1)
        n += 1
    return {k: v / max(1, n) for k, v in agg.items()}


def grid_calibrate(signals_val: list[dict[str, Any]], grid=(-1.0, 0.0, 1.0)) -> RhoCoeffs:
    nonneg = [g for g in grid if g >= 0.0]
    best, best_score = RhoCoeffs(a0=-0.4), -1.0
    for a0, a1, a2, a3 in product(grid, nonneg, nonneg, nonneg):
        c = RhoCoeffs(a0=a0, a1=a1, a2=a2, a3=a3).clamp_nonneg()
        score = ndcg10(signals_val, c)
        if score > best_score:
            best_score, best = score, c
    return best


def reliability_gate(signals_val: list[dict[str, Any]]) -> dict[str, float]:
    sig_H, sig_var, correct = [], [], []
    for sig in signals_val:
        top = int(np.argmax(sig["s_pers"]))
        correct.append(1.0 if top == sig["target_idx"] else 0.0)
        sig_H.append(float(sig["H"][sig["target_idx"]]))
        sig_var.append(float(sig["var_m"][sig["target_idx"]]))
    return {
        "auc_entropy": reliability_auc(sig_H, correct),
        "auc_variance": reliability_auc(sig_var, correct),
        "n": float(len(correct)),
        "base_rate_correct": (sum(correct) / len(correct)) if correct else 0.0,
    }


def paired_bootstrap_p(signals, coeffs_a, coeffs_b, n_boot=2000, seed=13) -> float:
    """P(full <= other) via user-level paired bootstrap on per-user NDCG@10."""
    rng = np.random.default_rng(seed)
    da = []
    for sig in signals:
        sa = mix_scores(sig, coeffs_a)
        sb = mix_scores(sig, coeffs_b)
        pa = int(np.where(np.argsort(-sa, kind="stable") == sig["target_idx"])[0][0])
        pb = int(np.where(np.argsort(-sb, kind="stable") == sig["target_idx"])[0][0])
        na = (1.0 / math.log2(pa + 2)) if pa < 10 else 0.0
        nb = (1.0 / math.log2(pb + 2)) if pb < 10 else 0.0
        da.append(na - nb)
    da = np.asarray(da)
    boots = rng.choice(da, size=(n_boot, len(da)), replace=True).mean(axis=1)
    return float((boots <= 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--weak-labels", required=True)
    ap.add_argument("--stage-b-dir", required=True)
    ap.add_argument("--qwen-model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sota-ndcg10", type=float, default=0.1506)
    ap.add_argument("--m-dropout", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    from llm4rec.methods.calm_qwen import (
        QwenBackboneRuntime,
        QwenIntentEncoderRuntime,
        QwenItemEncoderRuntime,
    )

    pd_dir = Path(args.processed_dir)
    sb = Path(args.stage_b_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((sb / "calm_stage_b_meta.json").read_text(encoding="utf-8"))
    tau = float(meta.get("tau_final", 2.0))
    delta = [float(x) for x in meta.get("delta_final", [])]
    n_intents = int(meta.get("n_intents", 4))
    delta = (delta + [0.0] * n_intents)[:n_intents]

    examples = read_jsonl(pd_dir / "examples.jsonl")
    with (pd_dir / "items.csv").open(encoding="utf-8", newline="") as fh:
        items = list(csv.DictReader(fh))
    item_lookup = {str(r["item_id"]): r for r in items}
    train = [e for e in examples if str(e.get("split")) == "train"]
    val = [e for e in examples if str(e.get("split")) in {"valid", "validation"}]
    test = [e for e in examples if str(e.get("split")) == "test"]
    held = {str(e.get("target")) for e in val} | {str(e.get("target")) for e in test}
    stats = build_train_only_stats(train, items, held_out_targets=held)
    support, s_prior = stats["support"], stats["popularity"]
    exposure_q = stats["exposure_prob"]
    print(f"train={len(train)} val={len(val)} test={len(test)} items={len(items)} tau={tau}")

    backbone = QwenBackboneRuntime(args.qwen_model_path, device=args.device)
    if (sb / "lora").exists():
        backbone.attach_lora(str(sb / "lora"))
        print("LoRA attached:", sb / "lora")
    item_enc = QwenItemEncoderRuntime(backbone, cache_path=str(sb / "item_vectors_fp16.npz"))
    item_enc.encode_batch(items)  # cache hit for everything precomputed in Stage-B
    item_vecs = dict(item_enc._cache)
    anchors = torch.load(sb / "anchors.pt", map_location="cpu", weights_only=True)

    def build_encoder(k: int) -> QwenIntentEncoderRuntime:
        enc = QwenIntentEncoderRuntime(
            backbone, n_intents=k,
            anchors=anchors[:k] if k <= anchors.shape[0] else anchors,
            extras_path=str(sb / "calm_stage_b_extras.pt") if k == n_intents else None,
        )
        enc.set_stage_a(item_vecs, exposure_q)
        return enc

    cache_file = out / "signals_cache.npz"
    sig_sets: dict[str, list[dict[str, Any]]] = {}
    for key, exs, k in (("val_k4", val, n_intents), ("test_k4", test, n_intents),
                        ("val_k1", val, 1), ("test_k1", test, 1)):
        print(f"=== computing signals: {key} ({len(exs)} examples, K={k}) ===", flush=True)
        torch.manual_seed(args.seed)
        sig_sets[key] = signals_for_examples(
            exs, intent_encoder=build_encoder(k), item_lookup=item_lookup,
            item_vecs=item_vecs, support=support, s_prior=s_prior,
            tau=tau, delta=delta if k == n_intents else delta[:1],
            m_dropout=args.m_dropout,
        )
        np.savez_compressed(
            out / f"signals_{key}.npz",
            **{
                f"{i}_{f}": sig[f]
                for i, sig in enumerate(sig_sets[key])
                for f in ("s_pers", "H", "var_m", "n_i", "prior")
            },
        )

    # Stage C + gate from cached val signals
    rho_full = grid_calibrate(sig_sets["val_k4"])
    rho_k1 = grid_calibrate(sig_sets["val_k1"])
    gate = reliability_gate(sig_sets["val_k4"])
    rng = random.Random(args.seed)
    rho_placebo = RhoCoeffs(
        a0=rng.uniform(-1, 1), a1=abs(rng.gauss(0, 1)),
        a2=abs(rng.gauss(0, 1)), a3=abs(rng.gauss(0, 1)),
    )

    m_raw = metrics_full(sig_sets["test_k4"], None)
    m_full = metrics_full(sig_sets["test_k4"], rho_full)
    m_k1 = metrics_full(sig_sets["test_k1"], rho_k1)
    m_plac = metrics_full(sig_sets["test_k4"], rho_placebo)
    p_vs_placebo = paired_bootstrap_p(sig_sets["test_k4"], rho_full, rho_placebo)

    verdict = {
        "backend": "qwen",
        "is_paper_evidence": True,
        "stage_b_dir": str(sb),
        "sota_ndcg10_bar": args.sota_ndcg10,
        "tau": tau, "delta": delta,
        "rho_full": vars(rho_full), "rho_k1": vars(rho_k1), "rho_placebo": vars(rho_placebo),
        "metrics": {"raw_personalized": m_raw, "full": m_full, "K1": m_k1, "placebo_rho": m_plac},
        "ndcg10": {
            "raw_personalized": m_raw["NDCG@10"], "full": m_full["NDCG@10"],
            "K1": m_k1["NDCG@10"], "placebo_rho": m_plac["NDCG@10"],
        },
        "checks": {
            "full_beats_sota": m_full["NDCG@10"] >= args.sota_ndcg10,
            "trust_beats_raw": m_full["NDCG@10"] >= m_raw["NDCG@10"],
            "trust_beats_placebo": m_full["NDCG@10"] >= m_plac["NDCG@10"],
            "trust_vs_placebo_p": p_vs_placebo,
            "multi_intent_beats_K1": m_full["NDCG@10"] >= m_k1["NDCG@10"],
            "reliability_signal_real": (gate["auc_entropy"] >= 0.6 or gate["auc_variance"] >= 0.6),
        },
        "stage_2p5_reliability_gate": gate,
        "n": {"train": len(train), "val": len(val), "test": len(test)},
    }
    (out / "calm_rec_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps({k: verdict[k] for k in ("ndcg10", "checks", "stage_2p5_reliability_gate")}, indent=2))


if __name__ == "__main__":
    main()
