#!/usr/bin/env python3
"""CALM-Rec Stage-B: the Qwen3-8B+LoRA gradient loop (server/GPU task).

Implements the locked training contract (docs/method_calm_rec_spec.md section 4,
CALMLossSpec in calm_trainer.py):

  L = L_rank + 0.3 L_attr + 0.1 L_bal + 0.05 L_orth + 0.05 L_use + 0.01 L_tau
  - L_rank: listwise sampled-softmax CE over [target + n_neg popularity-sampled
    negatives] on the FIXED-rho (0.1) mixed score (the only label-bearing term).
  - L_attr: facet classifier psi on z_uk vs intent k's anchor facet (the anchor
    binds; psi accuracy is the paper's p_psi evidence).
  - L_bal / L_use: load balance + usage floor on mean responsibilities.
  - L_orth: off-diagonal of normalized (Z)(Z)^T (z is already W-projected).
  - L_tau: tether tau to a schedule annealing UP 1.5 -> 4.0 (bounded [1, 8]).

Trainables: LoRA(q,k,v,o) + K soft slot embeddings + W_r + g head + lambda +
delta_k + theta_tau (+ psi). Anchors c_k are FROZEN weak-label centroids.
Item vectors h_i are FROZEN (cached fp16, computed here if missing).
rho is FIXED at 0.1 (Stage C calibrates the real gate post-hoc on validation).

Leakage: stats via build_train_only_stats with held_out_targets = val+test
targets; negatives sampled from train-support distribution; val carved from
train when no valid split exists; test is NEVER read here.

Usage (server):
  python scripts/train_calm_stage_b.py \
      --processed-dir data/processed/amazon_reviews_2023_beauty/cu_gr_v2 \
      --weak-labels outputs/calm/beauty \
      --out outputs/calm/beauty/stage_b \
      --qwen-model-path /home/ajifang/models/Qwen/Qwen3-8B
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

from llm4rec.methods.calm_trainer import CALMLossSpec, build_train_only_stats  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_items(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def history_ids(ex: dict[str, Any], max_history: int = 20) -> list[str]:
    hist = ex.get("history") or ex.get("history_item_ids") or []
    return [str(x) for x in hist][-max_history:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--weak-labels", required=True, help="dir with calm_weak_labels.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qwen-model-path", required=True)
    ap.add_argument("--n-intents", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n-neg", type=int, default=50)
    ap.add_argument("--batch-panels", type=int, default=4, help="panels per optimizer step")
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-heads", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-train", type=int, default=0, help="cap train panels (0 = all)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import numpy as np
    import torch

    from llm4rec.methods.calm_qwen import (
        QwenBackboneRuntime,
        QwenIntentEncoderRuntime,
        QwenItemEncoderRuntime,
        anchors_from_weak_labels,
        save_extras,
        torch_calm_scores,
    )

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd = Path(args.processed_dir)

    # ---------------- data + splits (test never read) ----------------
    examples = read_jsonl(pd / "examples.jsonl")
    items = read_items(pd / "items.csv")
    item_lookup = {str(r["item_id"]): r for r in items}
    train_all = [e for e in examples if str(e.get("split")) == "train"]
    val = [e for e in examples if str(e.get("split")) in {"valid", "validation"}]
    if not val:  # carve validation from train (never test)
        rng.shuffle(train_all)
        n_val = max(1, int(len(train_all) * args.val_frac))
        val, train_all = train_all[:n_val], train_all[n_val:]
    test_targets = {
        str(e.get("target")) for e in examples if str(e.get("split")) == "test"
    }
    held = {str(e.get("target")) for e in val} | test_targets
    if args.max_train:
        train_all = train_all[: args.max_train]
    print(f"train={len(train_all)} val={len(val)} items={len(items)} held_out={len(held)}")

    # ---------------- Stage A: frozen statistics + vectors ----------------
    stats = build_train_only_stats(train_all, items, held_out_targets=held)
    support = stats["support"]
    exposure_q = stats["exposure_prob"]
    s_prior = stats["popularity"]  # log1p(support), the testable default prior

    backbone = QwenBackboneRuntime(args.qwen_model_path, device=args.device)
    cache_path = str(out / "item_vectors_fp16.npz")
    item_enc = QwenItemEncoderRuntime(backbone, cache_path=cache_path)
    print("precomputing item vectors (frozen, cached)...", flush=True)
    item_enc.encode_batch(items)
    item_enc.save_cache()
    item_vecs = dict(item_enc._cache)
    dim = len(next(iter(item_vecs.values())))

    weak = read_jsonl(Path(args.weak_labels) / "calm_weak_labels.jsonl")
    facet_top: dict[str, list[str]] = {}
    for r in weak:
        f = r.get("dominant_facet")
        if f:
            facet_top.setdefault(f, []).append(str(r["item_id"]))
    facets = sorted(facet_top)
    anchors = anchors_from_weak_labels(facet_top, item_vecs, args.n_intents)
    torch.save(anchors, out / "anchors.pt")
    print(f"anchors from facets {facets} -> {tuple(anchors.shape)}")

    # ---------------- model: LoRA + heads ----------------
    from peft import LoraConfig, get_peft_model

    model = backbone.model  # loads now
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    backbone._model = model  # the intent runtime forwards through the PEFT model

    intent_enc = QwenIntentEncoderRuntime(backbone, n_intents=args.n_intents, anchors=anchors)
    intent_enc.set_stage_a(item_vecs, exposure_q)
    heads = intent_enc.build_heads(seed=args.seed)
    dev = args.device
    for k in heads:
        heads[k] = heads[k].to(dev).float().requires_grad_(True)
    n_facets = max(1, len(facets))
    psi = torch.zeros(n_facets, dim, device=dev, requires_grad=True)        # attr head
    psi_b = torch.zeros(n_facets, device=dev, requires_grad=True)
    delta = torch.zeros(args.n_intents, device=dev, requires_grad=True)     # per-intent pop residual
    theta_tau = torch.tensor(-2.565, device=dev, requires_grad=True)        # tau=1.5 at init

    spec = CALMLossSpec()
    lora_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(heads.values()) + [psi, psi_b, delta, theta_tau]
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr_lora},
            {"params": head_params, "lr": args.lr_heads},
        ],
        weight_decay=0.01,
    )

    # tensors for fast candidate assembly
    all_ids = sorted(item_vecs)
    id_to_row = {i: j for j, i in enumerate(all_ids)}
    H = torch.tensor([item_vecs[i] for i in all_ids], device=dev)            # [I, d]
    n_sup = torch.tensor([float(support.get(i, 0)) for i in all_ids], device=dev)
    prior_t = torch.tensor([float(s_prior.get(i, 0.0)) for i in all_ids], device=dev)
    pop_weights = [float(support.get(i, 0)) + 1.0 for i in all_ids]          # pop-matched sampling
    facet_of_intent = [k % n_facets for k in range(args.n_intents)]

    def tau_value():
        return 1.0 + 7.0 * torch.sigmoid(theta_tau)

    panels = [e for e in train_all if str(e.get("target")) in id_to_row and history_ids(e)]
    print(f"usable train panels: {len(panels)}")
    total_steps = max(1, args.epochs * math.ceil(len(panels) / args.batch_panels))
    step = 0
    log_rows = []

    model.train()
    for epoch in range(args.epochs):
        rng.shuffle(panels)
        for bstart in range(0, len(panels), args.batch_panels):
            batch = panels[bstart : bstart + args.batch_panels]
            opt.zero_grad()
            tau_target = 1.5 + (4.0 - 1.5) * min(1.0, step / max(1, total_steps - 1))
            losses = {k: 0.0 for k in ("rank", "attr", "bal", "orth", "use", "tau")}
            n_ok = 0
            resp_means = []
            for ex in batch:
                # tau must be recomputed per example: each example calls backward()
                # immediately, which frees the graph through sigmoid(theta_tau) —
                # reusing one tau across the batch raises "backward a second time".
                tau = tau_value()
                tgt = str(ex.get("target"))
                hist = history_ids(ex)
                # popularity-matched negatives, excluding history + target
                banned = set(hist) | {tgt}
                negs: list[str] = []
                while len(negs) < args.n_neg:
                    pick = rng.choices(all_ids, weights=pop_weights, k=args.n_neg)
                    negs.extend(i for i in pick if i not in banned and i not in negs)
                negs = negs[: args.n_neg]
                cand_rows = [id_to_row[tgt]] + [id_to_row[i] for i in negs]
                idx = torch.tensor(cand_rows, device=dev)

                u = intent_enc.slot_hiddens(hist, item_lookup, dropout=True)
                z, pi = intent_enc.intents_from_hiddens(u, hist)
                scores, _s_pers, resp = torch_calm_scores(
                    z, pi, H[idx], n_sup[idx], prior_t[idx], delta,
                    tau=tau, rho=spec.rho_fixed_during_stage_b,
                )
                l_rank = -torch.log_softmax(scores, dim=0)[0]                # target at row 0
                logits_psi = z @ psi.T + psi_b                               # [K, F]
                tgt_f = torch.tensor(facet_of_intent, device=dev)
                l_attr = torch.nn.functional.cross_entropy(logits_psi, tgt_f)
                zn = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                G = zn @ zn.T
                offdiag = G - torch.diag(torch.diag(G))
                l_orth = (offdiag ** 2).mean()
                r_mean = resp.mean(dim=0)                                    # [K]
                resp_means.append(r_mean.detach())
                uniform = torch.full_like(r_mean, 1.0 / args.n_intents)
                l_bal = (r_mean * (r_mean.clamp(min=1e-8) / uniform).log()).sum()
                l_use = torch.relu(0.05 - r_mean).pow(2).sum()
                l_tau = (tau - tau_target) ** 2

                loss = (
                    spec.w_rank * l_rank + spec.w_attr * l_attr + spec.w_bal * l_bal
                    + spec.w_orth * l_orth + spec.w_use * l_use + spec.w_tau * l_tau
                ) / len(batch)
                loss.backward()
                for key, v in (("rank", l_rank), ("attr", l_attr), ("bal", l_bal),
                               ("orth", l_orth), ("use", l_use), ("tau", l_tau)):
                    losses[key] += float(v.detach()) / len(batch)
                n_ok += 1
            if not n_ok:
                continue
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            opt.step()
            step += 1
            if step % 20 == 0 or step == total_steps:
                rm = torch.stack(resp_means).mean(dim=0).tolist() if resp_means else []
                row = {
                    "step": step, "epoch": epoch, "tau": float(tau.detach()),
                    "tau_target": tau_target,
                    **{f"L_{k}": round(v, 5) for k, v in losses.items()},
                    "mean_resp": [round(x, 4) for x in rm],
                    "delta": [round(float(x), 4) for x in delta.detach().tolist()],
                }
                log_rows.append(row)
                print(json.dumps(row), flush=True)

    # ---------------- save artifacts ----------------
    model.save_pretrained(str(out / "lora"))
    heads_all = dict(heads)
    heads_all.update({"psi": psi, "psi_b": psi_b, "delta": delta, "theta_tau": theta_tau})
    meta = {
        "n_intents": args.n_intents,
        "facets": facets,
        "facet_of_intent": facet_of_intent,
        "tau_final": float(tau_value().detach()),
        "delta_final": [float(x) for x in delta.detach().tolist()],
        "gamma_D_uk": 0.0,
        "gamma_note": "panel-discriminativeness D_uk deferred (panel-free encoder contract)",
        "loss_spec": spec.as_dict(),
        "args": {k: v for k, v in vars(args).items()},
        "n_train_panels": len(panels),
        "n_val": len(val),
        "steps": step,
        "item_vector_cache": cache_path,
    }
    save_extras(out / "calm_stage_b_extras.pt", heads_all, meta)
    (out / "training_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in log_rows) + "\n", encoding="utf-8"
    )
    # validation targets list for Stage-C (rho calibration uses val only)
    (out / "val_user_ids.json").write_text(
        json.dumps(sorted({str(e.get("user_id")) for e in val})), encoding="utf-8"
    )
    print(f"Stage-B done: {step} steps; artifacts in {out}")


if __name__ == "__main__":
    main()
