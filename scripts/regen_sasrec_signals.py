#!/usr/bin/env python3
"""Regenerate the VALIDATED SASRec scorer's per-user signals (CPU only, no GPU/Qwen).

Reuses the exact parity-tested functions from scripts/train_calm_sasrec.py so the
signals match the run that produced sasrec/calm_rec_verdict.json (raw NDCG@10=0.1143,
auc_entropy=0.785). Saves per-user signals + target_idx to npz, and verifies the
verdict metrics reproduce before any downstream analysis trusts them.
"""
from __future__ import annotations
import csv, json, math, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # repo root (portable: server or local)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
torch.set_num_threads(8)

import train_calm_sasrec as T
from llm4rec.methods.calm_rec import RhoCoeffs, reliability_auc
from llm4rec.methods.calm_trainer import build_train_only_stats
from llm4rec.methods.calm_qwen import (
    QwenItemEncoderRuntime, anchors_from_weak_labels, apply_whitening, fit_whitening,
)

DEV = "cpu"
VERDICT = json.loads((ROOT / "outputs/calm/beauty_frozen_v2/sasrec/calm_rec_verdict.json").read_text())
A = VERDICT["args"]
SEED = int(A["seed"])
OUT = ROOT / "outputs/calm/beauty_frozen_v2/sasrec"

class Args: pass
args = Args()
for k, v in A.items(): setattr(args, k, v)
args.device = DEV

random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
rng = random.Random(SEED)

pd_dir = ROOT / A["processed_dir"]
examples = T.read_jsonl(pd_dir / "examples.jsonl")
with (pd_dir / "items.csv").open(encoding="utf-8", newline="") as fh:
    items = list(csv.DictReader(fh))
train = [e for e in examples if str(e.get("split")) == "train"]
val   = [e for e in examples if str(e.get("split")) in {"valid", "validation"}]
test  = [e for e in examples if str(e.get("split")) == "test"]
held = {str(e.get("target")) for e in val} | {str(e.get("target")) for e in test}
stats = build_train_only_stats(train, items, held_out_targets=held)
support, s_prior = stats["support"], stats["popularity"]
print(f"train={len(train)} val={len(val)} test={len(test)} items={len(items)}", flush=True)

# cached frozen item vectors + whitening (FIX#3) -- exact replica of train_calm_sasrec.main
item_enc = QwenItemEncoderRuntime(backbone=None, cache_path=str(ROOT / A["item_vectors"]))
item_vecs_raw = dict(item_enc._cache)
d_item = len(next(iter(item_vecs_raw.values())))
whiten_mu, whiten_comp = fit_whitening(item_vecs_raw, k=max(0, int(A["whiten_k"])))
ids = sorted(item_vecs_raw)
X = torch.tensor([item_vecs_raw[i] for i in ids], dtype=torch.float32)
if whiten_comp.shape[0] > 0:
    X = apply_whitening(X, whiten_mu, whiten_comp)
X = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-8)
id_to_row = {i: j for j, i in enumerate(ids)}
Xg = X.to(DEV)
print(f"item vectors {tuple(X.shape)} whiten_k={whiten_comp.shape[0]}", flush=True)

weak = T.read_jsonl(ROOT / A["weak_labels"] / Path("calm_weak_labels.jsonl"))
facet_top = {}
for r in weak:
    f = r.get("dominant_facet")
    if f: facet_top.setdefault(f, []).append(str(r["item_id"]))
item_vecs_w = {i: X[id_to_row[i]].tolist() for i in ids}
anchors = anchors_from_weak_labels(facet_top, item_vecs_w, A["n_intents"]).to(DEV)
anchors = anchors / anchors.norm(dim=-1, keepdim=True).clamp(min=1e-8)

model = T.build_model(
    torch, d_item=d_item, d_model=A["d_model"], n_intents=A["n_intents"],
    max_history=A["max_history"], n_layers=A["n_layers"], n_heads=A["n_heads"],
    dropout=A["dropout"], anchors_init=anchors,
    logit_scale_init=A["logit_scale_init"], residual_scale=A["residual_scale"], device=DEV,
)
model.load_state_dict(torch.load(OUT / "sasrec_best.pt", map_location=DEV, weights_only=True))
tau = float(A["freeze_tau"])

# IMPORTANT: reproduce the SAME RNG sequence the training run used at signal time.
# In train_calm_sasrec.main the signals are computed AFTER training; MC-dropout draws
# depend on global torch RNG state. We reset to seed before compute_signals to make the
# regeneration deterministic + reproducible (and verify it matches the verdict AUC).
torch.manual_seed(SEED)
def signals(es):
    return T.compute_signals(torch, model, es, id_to_row, ids, Xg, support, s_prior,
                             args, DEV, tau, m_dropout=A["m_dropout"])
sig_val = signals(val)
sig_test = signals(test)

# verify verdict parity
m_raw = T.metrics_full(sig_test, None)
gate = T.reliability_gate(sig_val)
print("REPRO raw NDCG@10 =", round(m_raw["NDCG@10"], 4), "(verdict 0.1143)", flush=True)
print("REPRO HR@10 =", round(m_raw["HR@10"], 4), "MRR =", round(m_raw["MRR"], 4), flush=True)
print("REPRO gate auc_entropy =", round(gate["auc_entropy"], 4), "(verdict 0.785)",
      "auc_variance =", round(gate["auc_variance"], 4), "(verdict 0.296)", flush=True)
print("REPRO base_rate_correct =", round(gate["base_rate_correct"], 4), flush=True)

def dump(sig_list, name):
    payload = {}
    for i, s in enumerate(sig_list):
        payload[f"{i}_s_pers"] = s["s_pers"].astype(np.float32)
        payload[f"{i}_H"] = s["H"].astype(np.float32)
        payload[f"{i}_var_m"] = s["var_m"].astype(np.float32)
        payload[f"{i}_n_i"] = s["n_i"].astype(np.float32)
        payload[f"{i}_prior"] = s["prior"].astype(np.float32)
        payload[f"{i}_target_idx"] = np.int64(s["target_idx"])
    payload["n_users"] = np.int64(len(sig_list))
    np.savez_compressed(OUT / f"signals_{name}_sasrec.npz", **payload)
    print(f"saved signals_{name}_sasrec.npz ({len(sig_list)} users)", flush=True)
dump(sig_val, "val")
dump(sig_test, "test")
print("DONE", flush=True)
