#!/usr/bin/env python3
"""CALM-Rec Stage-2: a SASRec/sequence head over CACHED LLM item embeddings (LLMEmb recipe).

Stage-1 (the corrected frozen-inner-product trainer) un-broke the scorer but the
capped residual on frozen anchors binds it below SOTA (beauty NDCG@10 ~0.099 vs the
0.1506 bar). Per diagnosis wun91fszr the convergent fix is to REPLACE the s_pers
VEHICLE with a SASRec head that builds the user representation from the user's
train-history cached item vectors, while PRESERVING the contribution: the K=2
attribute-anchored intent heads + the endogenous calibrated-trust gate rho_ui.

What this is / is NOT:
  - NOT a Qwen re-encode. Item vectors h_i are the cached fp16 npz (4096-d), frozen.
  - The sequence model is a tiny Transformer (SASRec) over those frozen vectors,
    projected to d_model; the user state is the last position. K=2 intent query
    heads (anchor-initialized) map the user state to K directions z_uk in the SAME
    projected+L2 space as the (projected) candidate vectors. Scoring is the CALM
    soft-OR mixture (normalized cosine energy + learnable logit_scale, frozen tau).
  - r_uik (responsibilities), H_ui (entropy), Var_m (MC-dropout over the seq model)
    are recomputed from this head; the rho_ui gate is re-layered post-hoc on val.

Train: minutes on one 4090, few hundred MB. Reuses calm_qwen.fit_whitening/apply_whitening
and the same listwise sampled-softmax loss / Stage-A stats as Stage-1.

Usage (server):
  python scripts/train_calm_sasrec.py \
      --processed-dir data/processed/frozen_week8_beauty \
      --weak-labels outputs/calm/beauty_frozen \
      --item-vectors outputs/calm/beauty_frozen/stage_b/item_vectors_fp16.npz \
      --out outputs/calm/beauty_frozen_v2/sasrec --sota-ndcg10 0.1506
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


def history_ids(ex: dict[str, Any], max_history: int) -> list[str]:
    hist = ex.get("history") or ex.get("history_item_ids") or []
    return [str(x) for x in hist][-max_history:]


# ---------------------------------------------------------------------------
# Model: attention-pooled multi-intent head over CACHED item vectors.
#
# CRITICAL design choice (learned the hard way): scoring stays in the ORIGINAL
# d_item (4096) whitened+L2 space, where a parameter-free history-mean cosine
# already reaches NDCG@10 ~0.106. A 4096->128 down-projection of candidates was
# lossy and overfit (test 0.035). Instead the K intent directions are built as
# attention-weighted combinations of the user's OWN history item vectors (in
# 4096-d) + a small learned residual; candidates are scored by cosine in 4096-d.
# This GUARANTEES the head can recover history-mean (uniform attention -> the
# 0.106 floor) and improve via learned per-intent attention + the soft-OR mixture.
# A small Transformer contextualizes the history to produce the attention logits.
# ---------------------------------------------------------------------------
def build_model(torch, d_item: int, d_model: int, n_intents: int, max_history: int,
                n_layers: int, n_heads: int, dropout: float, anchors_init,
                logit_scale_init: float, residual_scale: float, device: str):
    nn = torch.nn

    class IntentPoolHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(d_item, d_model)          # context encoder input
            self.pos = nn.Embedding(max_history, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            # K learned intent queries -> attention logits over history positions
            self.intent_q = nn.Parameter(torch.randn(n_intents, d_model) * 0.02)
            self.key = nn.Linear(d_model, d_model)
            self.pi_head = nn.Linear(d_model, n_intents)       # salience g_uk from mean ctx
            # small residual in item space (anchor-shaped), gated small at init
            self.resid = nn.Linear(d_model, n_intents * d_item)
            nn.init.zeros_(self.resid.weight)
            nn.init.zeros_(self.resid.bias)
            self.residual_scale = residual_scale
            # anchors in item space [K, d_item] (whitened+L2), frozen reference (FIX#3)
            self.register_buffer("anchors", anchors_init.clone() if anchors_init is not None
                                 else torch.zeros(n_intents, d_item))
            self.log_logit_scale = nn.Parameter(
                torch.tensor(math.log(max(1e-3, logit_scale_init))))

        def context(self, seq, mask):
            # seq: [B, L, d_item] whitened+L2; mask [B, L] True=valid
            B, L, _ = seq.shape
            x = self.in_proj(seq) + self.pos(
                torch.arange(L, device=seq.device).clamp(max=max_history - 1))[None]
            causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
            ctx = self.norm(self.enc(x, mask=causal, src_key_padding_mask=~mask))  # [B,L,dm]
            return ctx

        def intents(self, seq, mask):
            # returns z [B,K,d_item] (item space) and pi [B,K]
            B, L, _ = seq.shape
            ctx = self.context(seq, mask)                      # [B, L, dm]
            keys = self.key(ctx)                               # [B, L, dm]
            att = torch.einsum("kd,bld->bkl", self.intent_q, keys)  # [B, K, L]
            att = att.masked_fill(~mask[:, None, :], float("-inf"))
            a = torch.softmax(att, dim=-1)                     # [B, K, L] over history
            # intent direction = attention-pooled history vectors (item space) + residual
            z_pool = torch.einsum("bkl,bld->bkd", a, seq)      # [B, K, d_item]
            # last-valid contextual state drives salience + residual
            last = mask.sum(dim=1).clamp(min=1) - 1
            u = ctx[torch.arange(B, device=seq.device), last]  # [B, dm]
            resid = self.resid(u).reshape(B, n_intents, d_item) * self.residual_scale
            z = z_pool + resid                                 # [B, K, d_item]
            pi = torch.softmax(self.pi_head(u), dim=-1)        # [B, K]
            return z, pi

    m = IntentPoolHead().to(device)
    return m


def mixture_scores(torch, z, pi, cand, log_logit_scale, tau):
    """z [B,K,d_item], pi [B,K], cand [B,N,d_item] -> s_pers/resp via CALM soft-OR.

    Cosine energy (normalized) in the ORIGINAL item space * learnable logit_scale;
    mixture temperature tau frozen. cand is the whitened+L2 candidate vectors (lossless).
    """
    zc = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)         # [B,K,d]
    hc = cand / cand.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # [B,N,d]
    e = torch.einsum("bnd,bkd->bnk", hc, zc)                     # [B,N,K] cosine
    scale = torch.exp(log_logit_scale)
    logits = torch.log(pi.clamp(min=1e-12))[:, None, :] + scale * e   # [B,N,K]
    s_pers = torch.logsumexp(logits, dim=2) / tau                # [B,N]
    resp = torch.softmax(logits, dim=2)                          # [B,N,K]
    return s_pers, resp


# ---------------------------------------------------------------------------
# Eval helpers (cached-array, mirrors eval_calm_beauty ladder)
# ---------------------------------------------------------------------------
def mix_scores(sig, coeffs: RhoCoeffs | None, rho_floor=0.15):
    if coeffs is None:
        return sig["s_pers"]
    pop = np.log1p(np.clip(sig["n_i"], 0, None))
    z = coeffs.a0 + coeffs.a1 * sig["H"] + coeffs.a2 * sig["var_m"] - coeffs.a3 * pop
    rho = 1.0 / (1.0 + np.exp(-z))
    rho_eff = np.clip(rho, 0.0, 1.0 - rho_floor)
    return (1.0 - rho_eff) * sig["s_pers"] + rho_eff * sig["prior"]


def metrics_full(signals, coeffs):
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


def ndcg10(signals, coeffs):
    return metrics_full(signals, coeffs)["NDCG@10"]


def grid_calibrate(signals_val, grid=(-1.0, 0.0, 1.0)):
    nonneg = [g for g in grid if g >= 0.0]
    best, best_score = RhoCoeffs(a0=-0.4), -1.0
    for a0, a1, a2, a3 in product(grid, nonneg, nonneg, nonneg):
        c = RhoCoeffs(a0=a0, a1=a1, a2=a2, a3=a3).clamp_nonneg()
        score = ndcg10(signals_val, c)
        if score > best_score:
            best_score, best = score, c
    return best


def reliability_gate(signals_val):
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


def paired_bootstrap_p(signals, coeffs_a, coeffs_b, n_boot=2000, seed=13):
    rng = np.random.default_rng(seed)
    da = []
    for sig in signals:
        sa, sb = mix_scores(sig, coeffs_a), mix_scores(sig, coeffs_b)
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
    ap.add_argument("--item-vectors", required=True, help="cached fp16 npz (NO Qwen re-encode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sota-ndcg10", type=float, default=0.1506)
    ap.add_argument("--n-intents", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max-history", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-neg", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--logit-scale-init", type=float, default=10.0)
    ap.add_argument("--freeze-tau", type=float, default=1.0)
    ap.add_argument("--residual-scale", type=float, default=0.3,
                    help="gate on the item-space residual added to attention-pooled intents")
    ap.add_argument("--w-orth", type=float, default=0.1)
    ap.add_argument("--whiten-k", type=int, default=1)
    ap.add_argument("--m-dropout", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    from llm4rec.methods.calm_qwen import (
        QwenItemEncoderRuntime,
        anchors_from_weak_labels,
        apply_whitening,
        fit_whitening,
    )

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd_dir = Path(args.processed_dir)
    dev = args.device

    # ---------------- data ----------------
    examples = read_jsonl(pd_dir / "examples.jsonl")
    with (pd_dir / "items.csv").open(encoding="utf-8", newline="") as fh:
        items = list(csv.DictReader(fh))
    train = [e for e in examples if str(e.get("split")) == "train"]
    val = [e for e in examples if str(e.get("split")) in {"valid", "validation"}]
    test = [e for e in examples if str(e.get("split")) == "test"]
    held = {str(e.get("target")) for e in val} | {str(e.get("target")) for e in test}
    stats = build_train_only_stats(train, items, held_out_targets=held)
    support, s_prior = stats["support"], stats["popularity"]
    print(f"train={len(train)} val={len(val)} test={len(test)} items={len(items)}", flush=True)

    # ---------------- cached item vectors (FROZEN) + whitening (FIX#3) ----------------
    item_enc = QwenItemEncoderRuntime(backbone=None, cache_path=args.item_vectors)  # type: ignore[arg-type]
    item_vecs_raw = dict(item_enc._cache)
    d_item = len(next(iter(item_vecs_raw.values())))
    whiten_mu, whiten_comp = fit_whitening(item_vecs_raw, k=max(0, int(args.whiten_k)))
    ids = sorted(item_vecs_raw)
    X = torch.tensor([item_vecs_raw[i] for i in ids], dtype=torch.float32)
    if whiten_comp.shape[0] > 0:
        X = apply_whitening(X, whiten_mu, whiten_comp)
    X = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-8)          # L2 (input space)
    id_to_row = {i: j for j, i in enumerate(ids)}
    Xg = X.to(dev)                                               # [I, d_item] whitened+L2
    print(f"item vectors: {tuple(X.shape)} whiten_k={whiten_comp.shape[0]}", flush=True)

    # anchors (whitened+L2, then projected later via the model's in_proj at init)
    weak = read_jsonl(Path(args.weak_labels) / "calm_weak_labels.jsonl")
    facet_top: dict[str, list[str]] = {}
    for r in weak:
        f = r.get("dominant_facet")
        if f:
            facet_top.setdefault(f, []).append(str(r["item_id"]))
    item_vecs_w = {i: X[id_to_row[i]].tolist() for i in ids}
    anchors = anchors_from_weak_labels(facet_top, item_vecs_w, args.n_intents).to(dev)  # [K,d_item]
    anchors = anchors / anchors.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # ---------------- model ----------------
    model = build_model(
        torch, d_item=d_item, d_model=args.d_model, n_intents=args.n_intents,
        max_history=args.max_history, n_layers=args.n_layers, n_heads=args.n_heads,
        dropout=args.dropout, anchors_init=anchors,
        logit_scale_init=args.logit_scale_init, residual_scale=args.residual_scale, device=dev,
    )
    tau = float(args.freeze_tau)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train panels with usable history + target in catalog
    panels = [e for e in train
              if str(e.get("target")) in id_to_row and history_ids(e, args.max_history)]
    print(f"usable train panels: {len(panels)}", flush=True)
    pop_weights = np.array([float(support.get(i, 0)) + 1.0 for i in ids])
    pop_weights = pop_weights / pop_weights.sum()
    all_rows = np.arange(len(ids))

    def make_seq_batch(batch):
        L = args.max_history
        seq = torch.zeros(len(batch), L, d_item, device=dev)
        mask = torch.zeros(len(batch), L, dtype=torch.bool, device=dev)
        for b, ex in enumerate(batch):
            hist = [h for h in history_ids(ex, L) if h in id_to_row]
            hist = hist[-L:]
            for j, h in enumerate(hist):
                seq[b, j] = Xg[id_to_row[h]]
                mask[b, j] = True
            if not hist:
                mask[b, 0] = True
        return seq, mask

    n_steps_per_epoch = math.ceil(len(panels) / args.batch_size)
    best_val = -1.0
    log_rows = []
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(panels)
        ep_loss = 0.0
        ep_n = 0
        for bstart in range(0, len(panels), args.batch_size):
            batch = panels[bstart: bstart + args.batch_size]
            seq, mask = make_seq_batch(batch)
            # candidates: target (row 0) + n_neg popularity negatives per panel
            B = len(batch)
            cand_idx = np.zeros((B, 1 + args.n_neg), dtype=np.int64)
            for b, ex in enumerate(batch):
                tgt = str(ex.get("target"))
                banned = set(history_ids(ex, args.max_history)) | {tgt}
                negs = []
                while len(negs) < args.n_neg:
                    pick = np.random.choice(all_rows, size=args.n_neg, p=pop_weights)
                    for r in pick:
                        i = ids[r]
                        if i not in banned and r not in negs:
                            negs.append(int(r))
                        if len(negs) >= args.n_neg:
                            break
                cand_idx[b, 0] = id_to_row[tgt]
                cand_idx[b, 1:] = negs[: args.n_neg]
            cand = Xg[torch.tensor(cand_idx, device=dev)]        # [B, 1+neg, d_item] whitened+L2
            z, pi = model.intents(seq, mask)                     # z in item space [B,K,d_item]
            s_pers, resp = mixture_scores(torch, z, pi, cand, model.log_logit_scale, tau)
            l_rank = torch.nn.functional.cross_entropy(
                s_pers, torch.zeros(B, dtype=torch.long, device=dev))   # target at col 0
            zc = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            G = torch.einsum("bkd,bjd->bkj", zc, zc)
            offdiag = G - torch.diag_embed(torch.diagonal(G, dim1=1, dim2=2))
            l_orth = (offdiag ** 2).mean()
            loss = l_rank + args.w_orth * l_orth
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(l_rank.detach())
            ep_n += 1
        # quick val NDCG@10 (raw s_pers) to track + early model selection
        val_ndcg = eval_ndcg_quick(torch, model, val, id_to_row, ids, Xg, support,
                                   args, dev, tau)
        row = {"epoch": epoch, "L_rank": round(ep_loss / max(1, ep_n), 4),
               "val_ndcg10_raw": round(val_ndcg, 4),
               "logit_scale": round(float(torch.exp(model.log_logit_scale).detach()), 3)}
        log_rows.append(row)
        print(json.dumps(row), flush=True)
        if val_ndcg > best_val:
            best_val = val_ndcg
            torch.save(model.state_dict(), out / "sasrec_best.pt")

    # ---------------- final eval from best checkpoint ----------------
    model.load_state_dict(torch.load(out / "sasrec_best.pt", map_location=dev, weights_only=True))
    print(f"loaded best (val raw NDCG@10={best_val:.4f}); computing full signals", flush=True)

    def signals(examples_set):
        return compute_signals(torch, model, examples_set, id_to_row, ids, Xg,
                               support, s_prior, args, dev, tau, m_dropout=args.m_dropout)

    sig_val = signals(val)
    sig_test = signals(test)

    rho_full = grid_calibrate(sig_val)
    gate = reliability_gate(sig_val)
    rng2 = random.Random(args.seed)
    rho_placebo = RhoCoeffs(a0=rng2.uniform(-1, 1), a1=abs(rng2.gauss(0, 1)),
                            a2=abs(rng2.gauss(0, 1)), a3=abs(rng2.gauss(0, 1)))
    m_raw = metrics_full(sig_test, None)
    m_full = metrics_full(sig_test, rho_full)
    m_plac = metrics_full(sig_test, rho_placebo)
    p_vs_placebo = paired_bootstrap_p(sig_test, rho_full, rho_placebo)

    verdict = {
        "backend": "sasrec_head_over_cached_llm_embeddings",
        "is_paper_evidence": True,
        "sota_ndcg10_bar": args.sota_ndcg10,
        "tau": tau, "n_intents": args.n_intents,
        "rho_full": vars(rho_full), "rho_placebo": vars(rho_placebo),
        "metrics": {"raw_personalized": m_raw, "full": m_full, "placebo_rho": m_plac},
        "ndcg10": {"raw_personalized": m_raw["NDCG@10"], "full": m_full["NDCG@10"],
                   "placebo_rho": m_plac["NDCG@10"]},
        "checks": {
            "full_beats_sota": m_full["NDCG@10"] >= args.sota_ndcg10,
            "trust_beats_raw": m_full["NDCG@10"] >= m_raw["NDCG@10"],
            "trust_beats_placebo": m_full["NDCG@10"] >= m_plac["NDCG@10"],
            "trust_vs_placebo_p": p_vs_placebo,
            "reliability_signal_real": (gate["auc_entropy"] >= 0.6 or gate["auc_variance"] >= 0.6),
        },
        "stage_2p5_reliability_gate": gate,
        "best_val_ndcg10_raw": best_val,
        "args": {k: v for k, v in vars(args).items()},
        "n": {"train": len(train), "val": len(val), "test": len(test)},
    }
    (out / "calm_rec_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    (out / "training_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in log_rows) + "\n", encoding="utf-8")
    print(json.dumps({k: verdict[k] for k in
                      ("ndcg10", "checks", "stage_2p5_reliability_gate")}, indent=2))


def _seq_from(examples_set, id_to_row, Xg, d_item, max_history, dev, torch):
    L = max_history
    seq = torch.zeros(len(examples_set), L, d_item, device=dev)
    mask = torch.zeros(len(examples_set), L, dtype=torch.bool, device=dev)
    for b, ex in enumerate(examples_set):
        hist = [h for h in history_ids(ex, L) if h in id_to_row][-L:]
        for j, h in enumerate(hist):
            seq[b, j] = Xg[id_to_row[h]]
            mask[b, j] = True
        if not hist:
            mask[b, 0] = True
    return seq, mask


def eval_ndcg_quick(torch, model, examples_set, id_to_row, ids, Xg, support, args, dev, tau):
    """Fast val NDCG@10 on raw s_pers (no dropout, no gate) for model selection."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for bstart in range(0, len(examples_set), 256):
            batch = examples_set[bstart: bstart + 256]
            valid = [ex for ex in batch
                     if str(ex.get("target")) in id_to_row
                     and str(ex.get("target")) in
                     [str(c) for c in (ex.get("candidates") or ex.get("candidate_items") or [])]]
            if not valid:
                continue
            seq, mask = _seq_from(valid, id_to_row, Xg, Xg.shape[1], args.max_history, dev, torch)
            z, pi = model.intents(seq, mask)
            for b, ex in enumerate(valid):
                cands = [str(c) for c in (ex.get("candidates") or ex.get("candidate_items") or [])]
                rows = torch.tensor([id_to_row[c] for c in cands], device=dev)
                cand = Xg[rows][None]                              # [1, N, d_item]
                s_pers, _ = mixture_scores(
                    torch, z[b: b + 1], pi[b: b + 1], cand, model.log_logit_scale, tau)
                order = torch.argsort(-s_pers[0])
                pos = int((order == cands.index(str(ex.get("target")))).nonzero()[0])
                total += (1.0 / math.log2(pos + 2)) if pos < 10 else 0.0
                n += 1
    return total / n if n else 0.0


def compute_signals(torch, model, examples_set, id_to_row, ids, Xg, support, s_prior,
                    args, dev, tau, m_dropout):
    """Per example: s_pers, H (entropy of r_uik), Var_m (MC-dropout std), n_i, prior."""
    out = []
    for bstart in range(0, len(examples_set), 256):
        batch = examples_set[bstart: bstart + 256]
        valid = [ex for ex in batch
                 if str(ex.get("target")) in id_to_row
                 and str(ex.get("target")) in
                 [str(c) for c in (ex.get("candidates") or ex.get("candidate_items") or [])]]
        if not valid:
            continue
        seq, mask = _seq_from(valid, id_to_row, Xg, Xg.shape[1], args.max_history, dev, torch)
        # no-dropout pass
        model.eval()
        with torch.no_grad():
            z0, pi0 = model.intents(seq, mask)
        # MC-dropout passes (model.train enables dropout in the encoder)
        ens_z, ens_pi = [], []
        if m_dropout > 0:
            model.train()
            with torch.no_grad():
                for _ in range(m_dropout):
                    zm, pim = model.intents(seq, mask)
                    ens_z.append(zm)
                    ens_pi.append(pim)
        for b, ex in enumerate(valid):
            cands = [str(c) for c in (ex.get("candidates") or ex.get("candidate_items") or [])]
            rows = torch.tensor([id_to_row[c] for c in cands], device=dev)
            cand = Xg[rows][None]                                  # [1, N, d_item]
            with torch.no_grad():
                s_pers, resp = mixture_scores(
                    torch, z0[b: b + 1], pi0[b: b + 1], cand, model.log_logit_scale, tau)
                s_pers = s_pers[0].cpu().numpy()
                resp = resp[0].cpu().numpy()                       # [N, K]
                H = -(resp * np.log(np.clip(resp, 1e-12, None))).sum(axis=1)
                if ens_z:
                    samples = []
                    for zm, pim in zip(ens_z, ens_pi):
                        sp, _ = mixture_scores(
                            torch, zm[b: b + 1], pim[b: b + 1], cand,
                            model.log_logit_scale, tau)
                        samples.append(sp[0].cpu().numpy())
                    var_m = np.stack(samples).std(axis=0)
                else:
                    var_m = np.zeros_like(s_pers)
            n_i = np.array([float(support.get(c, 0)) for c in cands])
            prior = np.array([float(s_prior.get(c, 0.0)) for c in cands])
            out.append({
                "user_id": str(ex.get("user_id")), "cands": cands,
                "target_idx": cands.index(str(ex.get("target"))),
                "s_pers": s_pers, "H": H, "var_m": var_m, "n_i": n_i, "prior": prior,
            })
    return out


if __name__ == "__main__":
    main()
