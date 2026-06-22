"""Real Qwen3-8B implementation for CALM-Rec encoders + the differentiable scorer.

This module is GPU/server-oriented and imports torch/transformers lazily, so the
rest of the package stays importable on CPU-only dev boxes. It provides:

  - ``QwenBackboneRuntime``: one shared lazy loader for tokenizer + model (+ LoRA).
  - ``QwenItemEncoderRuntime``: frozen Qwen3-8B over item text -> h_i (mean-pooled
    last hidden state), with an fp16 on-disk cache per domain (offline, once).
  - ``QwenIntentEncoderRuntime``: Qwen3-8B(+LoRA) over the short user-side prompt
    with K attribute-anchored soft intent slots; returns IntentSet (z already in
    item space: z_uk = c_k + clip(W_r u_k)) and salience pi from g_uk + lambda*E_uk.
  - ``torch_calm_scores``: differentiable replica of the pure-python scoring core
    (per_intent_energy -> soft-OR mixture -> fixed-rho prior mixing) used by the
    Stage-B gradient loop. Parity with calm_rec.py is enforced by unit tests.

Design notes (docs/method_calm_rec_spec.md):
  - The 101 candidates NEVER enter the prompt; candidate scoring is vector-space.
  - Anchors c_k come from attribute weak-label centroids of ITEM vectors, so z and
    h live in the same 4096-d space; only the residual r_uk = clip(W_r u_k) is
    personalized (overfit control + the firewall vs ComiRec).
  - pi_uk = softmax(g_uk + lambda*E_uk). The panel-discriminativeness term D_uk
    (gamma) is deliberately deferred: the IntentEncoder contract is panel-free
    (encode_intents sees only history), so inference-time D_uk would require a
    contract change. gamma=0 is recorded in artifacts for honest reporting.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from llm4rec.methods.calm_rec import IntentSet

_HELP = (
    "CALM-Rec Qwen runtime requires torch + transformers (+ peft for LoRA) and the "
    "Qwen3-8B weights on a GPU host. See docs/CALM_REC_RUNBOOK.md."
)


def _require_torch():
    try:
        import torch  # noqa: F401

        return torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(_HELP) from e


def item_text(item_row: dict[str, Any]) -> str:
    """Canonical item text: title | brand | category | attrs | description."""
    parts = []
    for f in ("title", "brand", "category", "attributes", "description"):
        v = str(item_row.get(f, "") or "").strip()
        if v:
            parts.append(v)
    return " | ".join(parts) or str(item_row.get("item_id", ""))


def history_tuple_text(item_row: dict[str, Any]) -> str:
    """Compact attribute tuple for one history item inside the user prompt."""
    title = str(item_row.get("title", "") or item_row.get("item_id", ""))[:80]
    brand = str(item_row.get("brand", "") or "")[:30]
    cat = str(item_row.get("category", "") or "").split(">")[-1].strip()[:40]
    bits = [title]
    if brand:
        bits.append(f"brand={brand}")
    if cat:
        bits.append(f"cat={cat}")
    return "(" + "; ".join(bits) + ")"


class QwenBackboneRuntime:
    """Lazy shared tokenizer+model loader (frozen by default; LoRA optional)."""

    def __init__(self, model_path: str, *, device: str = "cuda", dtype: str = "bfloat16") -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._tok = None
        self._model = None

    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        return self._tok

    @property
    def model(self):
        if self._model is None:
            torch = _require_torch()
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=getattr(torch, self.dtype),
                device_map=self.device,
                trust_remote_code=True,
            ).eval()
        return self._model

    def attach_lora(self, lora_path: str) -> None:
        from peft import PeftModel

        self._model = PeftModel.from_pretrained(self.model, lora_path).eval()

    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)


@dataclass
class QwenItemEncoderRuntime:
    """Frozen Qwen3-8B item encoder with an fp16 disk cache (ItemEncoder protocol)."""

    backbone: QwenBackboneRuntime
    pooling: str = "mean"           # {"mean", "last"}
    max_tokens: int = 256
    cache_path: str | None = None   # <domain>.npz with keys: item_ids, vectors(fp16)
    batch_size: int = 16
    _cache: dict[str, list[float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.cache_path and Path(self.cache_path).exists():
            self._load_cache()

    def _load_cache(self) -> None:
        import numpy as np

        data = np.load(self.cache_path, allow_pickle=False)
        ids = [str(x) for x in data["item_ids"]]
        vecs = data["vectors"].astype("float32")
        self._cache = {i: vecs[j].tolist() for j, i in enumerate(ids)}

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        import numpy as np

        ids = sorted(self._cache)
        vecs = np.array([self._cache[i] for i in ids], dtype="float16")
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cache_path, item_ids=np.array(ids), vectors=vecs)

    def _pool(self, hidden, mask):
        torch = _require_torch()
        if self.pooling == "last":
            lengths = mask.sum(dim=1).clamp(min=1) - 1
            return hidden[torch.arange(hidden.shape[0]), lengths]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def encode_batch(self, rows: Sequence[dict[str, Any]]) -> None:
        """Encode many items at once into the cache (offline precompute path)."""
        torch = _require_torch()
        todo = [r for r in rows if str(r.get("item_id")) not in self._cache]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start : start + self.batch_size]
            texts = [item_text(r) for r in chunk]
            enc = self.backbone.tok(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_tokens,
            ).to(self.backbone.device)
            with torch.no_grad():
                out = self.backbone.model(
                    **enc, output_hidden_states=True, use_cache=False
                )
            pooled = self._pool(out.hidden_states[-1], enc["attention_mask"])
            for j, r in enumerate(chunk):
                self._cache[str(r["item_id"])] = pooled[j].float().cpu().tolist()

    def encode(self, item_id: str, item_row: dict[str, Any]) -> list[float]:
        key = str(item_id)
        if key not in self._cache:
            self.encode_batch([{**item_row, "item_id": key}])
        return self._cache[key]


SYSTEM_PROMPT = (
    "You are a recommendation profiler. Read the user's purchase history and "
    "summarize their latent shopping intents."
)


@dataclass
class QwenIntentEncoderRuntime:
    """Qwen3-8B(+LoRA) intent encoder with K attribute-anchored soft slots.

    encode_intents reads the K slot hidden states u_k from one short user-side
    forward, forms z_uk = c_k + clip(W_r u_k, eps) in item space, and pi from
    softmax(g(u_k) + lambda * E_uk). dropout=True runs the forward in train mode
    (LoRA/backbone dropout active) for the M-pass ensemble variance.
    """

    backbone: QwenBackboneRuntime
    n_intents: int
    anchors: Any = None             # torch [K, d_item] (weak-label centroids)
    max_history: int = 20
    residual_eps: float = 40.0      # ||r_uk|| <= eps  (FIX#2: 2.0 -> 40 ~ 0.35*||anchor||;
                                    #   2.0 froze z to a <=1.76% perturbation of the anchor)
    lambda_sal: float = 1.0
    tau_for_E: float = 1.0
    extras_path: str | None = None  # trained heads (W_r, g, lambda, slots)
    _heads: Any = field(default=None, repr=False)
    _item_vecs: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _exposure_q: dict[str, float] = field(default_factory=dict, repr=False)

    def set_stage_a(self, item_vecs: dict[str, list[float]], exposure_q: dict[str, float]) -> None:
        """Inject cached item vectors + train-only exposure probs (for E_uk)."""
        self._item_vecs = item_vecs
        self._exposure_q = exposure_q

    # --- trainable pieces -------------------------------------------------
    def build_heads(self, *, seed: int = 0):
        """Create (or load) slot embeddings, W_r, g head, lambda. Idempotent."""
        torch = _require_torch()
        if self._heads is not None:
            return self._heads
        d = self.backbone.hidden_size()
        g = torch.Generator().manual_seed(seed)
        emb = self.backbone.model.get_input_embeddings().weight
        scale = float(emb.detach().float().std())
        heads = {
            "slots": torch.randn(self.n_intents, d, generator=g) * scale,
            "w_r": torch.zeros(d, d),          # zero-init: z starts AT the anchor
            "g_head": torch.zeros(d),
            "g_bias": torch.zeros(self.n_intents),
            "lambda_sal": torch.tensor(float(self.lambda_sal)),
        }
        if self.extras_path and Path(self.extras_path).exists():
            loaded = torch.load(self.extras_path, map_location="cpu", weights_only=True)
            heads.update({k: v for k, v in loaded.items() if k in heads})
        self._heads = heads
        return heads

    def _prompt_ids(self, history_item_ids, item_lookup):
        hist = [str(h) for h in history_item_ids][-self.max_history :]
        lines = [SYSTEM_PROMPT, "Purchase history (oldest to newest):"]
        for h in hist:
            lines.append(history_tuple_text(item_lookup.get(h, {"item_id": h})))
        lines.append("Intent slots:")
        text = "\n".join(lines)
        return self.backbone.tok(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).input_ids.to(self.backbone.device)

    def slot_hiddens(self, history_item_ids, item_lookup, *, dropout: bool = False):
        """One user-side forward -> [K, d] hidden states at the slot positions."""
        torch = _require_torch()
        heads = self.build_heads()
        model = self.backbone.model
        ids = self._prompt_ids(history_item_ids, item_lookup)
        tok_emb = model.get_input_embeddings()(ids)  # [1, T, d]
        slots = heads["slots"].to(tok_emb.device, tok_emb.dtype).unsqueeze(0)  # [1, K, d]
        inputs = torch.cat([tok_emb, slots], dim=1)
        was_training = model.training
        model.train(dropout)
        try:
            ctx = torch.enable_grad() if torch.is_grad_enabled() else torch.no_grad()
            with ctx:
                out = model(
                    inputs_embeds=inputs, output_hidden_states=True, use_cache=False
                )
        finally:
            model.train(was_training)
        return out.hidden_states[-1][0, -self.n_intents :, :]  # [K, d]

    def intents_from_hiddens(self, u, history_item_ids) -> "tuple[Any, Any]":
        """(z [K,d], pi [K]) from slot hiddens. Differentiable when grads are on."""
        torch = _require_torch()
        heads = self.build_heads()
        dev, dt = u.device, torch.float32
        u = u.to(dt)
        anchors = self.anchors.to(dev, dt)
        w_r = heads["w_r"].to(dev, dt)
        r = u @ w_r.T                                            # [K, d]
        norms = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        r = r * (self.residual_eps * torch.tanh(norms / self.residual_eps) / norms)
        z = anchors + r                                          # [K, d]
        g = u @ heads["g_head"].to(dev, dt) + heads["g_bias"].to(dev, dt)  # [K]
        e_hist = self._history_evidence(z, history_item_ids)     # [K]
        logits = g + heads["lambda_sal"].to(dev, dt) * e_hist
        pi = torch.softmax(logits, dim=-1)
        return z, pi

    def _history_evidence(self, z, history_item_ids):
        """E_uk = mean_j exp(clip(z_k . h_j)) / q(j) over history items (spec section 3)."""
        torch = _require_torch()
        hs, qs = [], []
        for h in [str(x) for x in history_item_ids][-self.max_history :]:
            v = self._item_vecs.get(h)
            if v is None:
                continue
            hs.append(v)
            qs.append(self._exposure_q.get(h, 1e-6))
        if not hs:
            return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        H = torch.tensor(hs, device=z.device, dtype=z.dtype)      # [J, d]
        q = torch.tensor(qs, device=z.device, dtype=z.dtype)      # [J]
        d = max(z.shape[-1], 1)
        e = (z @ H.T) / math.sqrt(d)                              # scale-stable
        e = e.clamp(-10, 10)
        return (torch.exp(self.tau_for_E * e) / q.unsqueeze(0)).mean(dim=-1).log1p()

    def encode_intents(self, history_item_ids, item_lookup, *, dropout: bool = False) -> IntentSet:
        torch = _require_torch()
        with torch.no_grad():
            u = self.slot_hiddens(history_item_ids, item_lookup, dropout=dropout)
            z, pi = self.intents_from_hiddens(u, history_item_ids)
        return IntentSet(z=[row.tolist() for row in z.cpu()], pi=pi.cpu().tolist())


def torch_calm_scores(
    z,                  # [K, d] intent directions (already item-space)
    pi,                 # [K]
    h,                  # [N, d] candidate item vectors
    n_support,          # [N] train-only item support counts
    s_prior,            # [N] history-free prior scores
    delta,              # [K] per-intent popularity residual coefficients
    *,
    tau,                # mixture (logsumexp) temperature; scalar tensor or float
    rho: float = 0.1,   # FIXED mild rho during Stage-B (spec section 4)
    rho_floor: float = 0.15,
    normalize: bool = False,   # FIX#1: L2-normalize h AND z -> e_k(i) is a cosine in [-1,1]
    logit_scale=None,          # FIX#1: learnable O(1-10) scale on the cosine (replaces tau*raw)
):
    """Differentiable replica of the calm_rec.py scoring core (parity-tested).

    e_k(i) = <z_k_hat, h_i_hat> - delta_k log(1+n_i)        [cosine when normalize=True]
    logits_k(i) = log pi_k + logit_scale * e_k(i)
    s_pers(i) = (1/tau) logsumexp_k(logits_k(i))
    s(i) = (1-rho') s_pers(i) + rho' s_prior(i)
    Returns (scores [N], s_pers [N], responsibilities [N, K]).

    FIX#1 (the dominant bug): on raw mean-pooled Qwen vectors ||h||~117 the inner
    product e~1.25e4 and the listwise CE saturated 250-900x its log(51) floor. With
    normalize=True the energy is a bounded cosine and logit_scale (init ~10, learnable)
    sets the softmax peakiness; tau is the *separate* mixture temperature (FIX#4: frozen 1.0).
    Legacy behaviour (normalize=False, logit_scale=None) -> tau*raw, kept for parity tests.
    """
    torch = _require_torch()
    if normalize:
        h = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    pop = torch.log1p(n_support.clamp(min=0))                    # [N]
    e = h @ z.T - pop.unsqueeze(1) * delta.unsqueeze(0)          # [N, K]
    tau_t = tau if torch.is_tensor(tau) else torch.tensor(float(tau), dtype=e.dtype, device=e.device)
    if logit_scale is None:
        scale = tau_t                                            # legacy: tau doubles as scale
    else:
        scale = logit_scale if torch.is_tensor(logit_scale) else torch.tensor(
            float(logit_scale), dtype=e.dtype, device=e.device)
    logits = torch.log(pi.clamp(min=1e-12)).unsqueeze(0) + scale * e
    s_pers = torch.logsumexp(logits, dim=1) / tau_t              # [N]
    resp = torch.softmax(logits, dim=1)                          # [N, K]
    rho_eff = min(max(float(rho), 0.0), 1.0 - float(rho_floor))
    scores = (1.0 - rho_eff) * s_pers + rho_eff * s_prior
    return scores, s_pers, resp


def fit_whitening(item_vecs: dict[str, list[float]], *, k: int = 1):
    """FIX#3: all-but-the-top. Returns (mu [d], components [k, d]) from the catalog.

    The raw mean-pooled Qwen vectors are strongly anisotropic (PC1 ~26% variance,
    pairwise cosine ~0.916); that dominant direction is a common-mode offset that
    swamps the intent geometry. We mean-center and remove the top-k PCs from h, z
    AND the anchors *consistently* (the same mu/components are persisted and reused
    at eval). With k=0 this is the identity (legacy).
    """
    torch = _require_torch()
    ids = sorted(item_vecs)
    X = torch.tensor([item_vecs[i] for i in ids], dtype=torch.float32)   # [N, d]
    mu = X.mean(dim=0)                                                   # [d]
    if k <= 0:
        return mu, torch.zeros(0, X.shape[1])
    Xc = X - mu
    # economy SVD; right-singular vectors are the principal directions
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return mu, Vh[:k].contiguous()                                      # [k, d]


def apply_whitening(x, mu, components):
    """Remove the persisted top-k directions: x' = (x - mu) - sum_j <x-mu,v_j> v_j."""
    torch = _require_torch()
    if components is None or components.shape[0] == 0:
        return x
    mu = mu.to(x.device, x.dtype)
    comp = components.to(x.device, x.dtype)                             # [k, d]
    xc = x - mu
    proj = (xc @ comp.T) @ comp                                        # [..., d]
    return xc - proj


def anchors_from_weak_labels(
    facet_top_items: dict[str, list[str]],
    item_vecs: dict[str, list[float]],
    n_intents: int,
):
    """c_k = mean item vector of facet k's top-assigned items (one facet per intent)."""
    torch = _require_torch()
    facets = sorted(facet_top_items)
    dim = len(next(iter(item_vecs.values())))
    anchors = []
    for k in range(n_intents):
        facet = facets[k % max(1, len(facets))] if facets else None
        vecs = [item_vecs[i] for i in facet_top_items.get(facet, []) if i in item_vecs]
        if vecs:
            anchors.append(torch.tensor(vecs).mean(dim=0))
        else:
            anchors.append(torch.zeros(dim))
    return torch.stack(anchors)  # [K, d]


def save_extras(path: str | Path, heads: dict, meta: dict[str, Any]) -> None:
    torch = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in heads.items()}, path)
    (path.parent / "calm_stage_b_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


# =====================================================================================
# Stage-B GPU gradient loop (the only remaining piece of CALM-Rec).
# =====================================================================================
# Spec: docs/method_calm_rec_spec.md §4 + §8. Implements
#   L = L_rank + 0.3*L_attr + 0.1*L_bal + 0.05*L_orth + 0.05*L_use + 0.01*L_τ
# where L_rank is the only label-bearing term (listwise sampled-softmax CE over the
# 101 candidates), tau is annealed UP from ~1.5 (bounded [1,8]), and rho is held
# FIXED mild (~0.1) so the panel learns to rank without leaning on the prior.

def _stage_b_losses(
    resp,             # [N, K] candidate->intent responsibilities (softmax)
    r,                # [K, d] residual r_uk = W_r u_k (after clip)
    z,                # [K, d] intent directions in item space (anchored)
    tau_t,            # scalar learnable tau (or fixed tensor)
    *,
    tau_target: float,
    use_floor_frac: float = 0.5,
):
    """Compute the 5 regularization losses + return them as a dict.

    Canonical per-spec implementations (the spec gives the controlled object but not
    the exact formula; these are the natural choices):
      L_attr  = mean( ||r_uk||^2 )                                 (anchors stay representative)
      L_bal   = KL( mean_n(resp_nk) || Uniform[1/K] )              (load balance)
      L_orth  = || off_diag( Z_hat @ Z_hat.T ) ||_F^2              (orthogonalize WZ)
      L_use   = sum_k max(0, use_floor - mean_n(resp_nk))^2        (usage floor)
      L_tau   = (log tau_t - log tau_target)^2                     (track schedule)
    where Z_hat = z / ||z|| row-normalised so the off-diagonal is a cosine matrix
    (matches the FIX#1 normalised energy form used in torch_calm_scores).
    """
    torch = _require_torch()
    K = int(resp.shape[1])

    # L_attr: keep residuals small so z stays near c (anchor representativeness).
    L_attr = (r * r).sum(dim=-1).mean()

    # L_bal: KL of mean-responsibility from uniform (prevents intent collapse).
    p = resp.mean(dim=0).clamp(min=1e-12)                          # [K]
    uniform = torch.full_like(p, 1.0 / K)
    L_bal = (p * (p.log() - uniform.log())).sum()

    # L_orth: off-diagonal of normalised Z @ Z.T (orthogonalise WZ in item space).
    z_hat = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)       # [K, d]
    gram = z_hat @ z_hat.T                                          # [K, K]
    eye = torch.eye(K, device=gram.device, dtype=gram.dtype)
    L_orth = ((gram - eye) ** 2).sum() / max(K * (K - 1), 1)

    # L_use: per-intent usage floor (penalises intents that drop below use_floor/K).
    threshold = use_floor_frac / K
    L_use = (torch.clamp(threshold - p, min=0.0) ** 2).sum()

    # L_tau: log-space penalty toward the annealed target (symmetric, scale-free).
    tau_target_t = torch.tensor(float(tau_target), device=tau_t.device, dtype=tau_t.dtype)
    L_tau = (tau_t.clamp(min=1e-3).log() - tau_target_t.clamp(min=1e-3).log()) ** 2

    return {"L_attr": L_attr, "L_bal": L_bal, "L_orth": L_orth, "L_use": L_use, "L_tau": L_tau}


def _example_panel(example: dict[str, Any]) -> "tuple[list[str], int]":
    """Resolve the 101-candidate panel + positive index from a processed example.

    Supports the two common processed-event schemas:
      - {"candidate_items": [...], "target": "<pos_id>"}
      - {"candidates": [...], "label": <pos_index>}  (legacy)
    Raises if neither shape is present.
    """
    if "candidate_items" in example:
        cands = [str(c) for c in example["candidate_items"]]
        pos = str(example.get("target") or "")
        if pos in cands:
            return cands, cands.index(pos)
    if "candidates" in example:
        cands = [str(c) for c in example["candidates"]]
        lab = example.get("label")
        if isinstance(lab, int) and 0 <= lab < len(cands):
            return cands, int(lab)
    raise KeyError("example must provide candidate_items+target or candidates+label")


def _anneal_tau(epoch_frac: float, *, start: float, end: float, lo: float, hi: float) -> float:
    """Linear schedule from `start` to `end` over `epoch_frac` in [0,1], clamped to [lo,hi]."""
    val = start + (end - start) * float(epoch_frac)
    return max(lo, min(hi, val))


def train_stage_b_gpu(
    encoder: QwenIntentEncoderRuntime,
    train_examples: Sequence[dict[str, Any]],
    item_lookup: dict[str, dict[str, Any]],
    *,
    n_support: dict[str, float],
    s_prior: dict[str, float],
    loss_spec=None,                       # CALMLossSpec | None
    num_epochs: int = 3,
    lr_heads: float = 1e-3,
    lr_lora: float = 1e-4,
    grad_accum_steps: int = 8,
    seed: int = 0,
    log_every: int = 100,
    save_extras_path: str | None = None,
    max_examples_per_epoch: int | None = None,
) -> dict[str, Any]:
    """End-to-end LoRA + intent-heads gradient loop for Stage-B (the GPU task).

    Trainable: encoder._heads (slots, w_r, g_head, g_bias, lambda_sal), per-intent
    delta (popularity residual coefficients), logit_scale (FIX#1), tau (annealed),
    and any LoRA params on the backbone (frozen base otherwise).

    Frozen: Qwen3-8B base weights, anchors, item vectors h, train-only stats.

    rho is held FIXED at loss_spec.rho_fixed_during_stage_b (~0.1) per spec §4.

    Returns a dict of metrics; persists trained heads to `save_extras_path` if given.
    """
    torch = _require_torch()
    if loss_spec is None:                                          # local import to avoid cycle
        from llm4rec.methods.calm_trainer import CALMLossSpec      # noqa: PLC0415
        loss_spec = CALMLossSpec()

    # --- materialise trainable heads on device with grad --------------------
    encoder.build_heads(seed=seed)
    dev = encoder.backbone.device
    dt = torch.float32
    heads = {}
    for name, t in encoder._heads.items():
        heads[name] = t.to(dev, dt).clone().detach().requires_grad_(True)
    encoder._heads = heads                                          # so forward calls see grads

    K = int(encoder.n_intents)
    delta = torch.zeros(K, device=dev, dtype=dt, requires_grad=True)
    logit_scale = torch.tensor(10.0, device=dev, dtype=dt, requires_grad=True)
    tau = torch.tensor(float(loss_spec.tau_start), device=dev, dtype=dt, requires_grad=True)

    # --- LoRA params (if attached) are the only trainable backbone params ---
    lora_params: list = []
    for n, p in encoder.backbone.model.named_parameters():
        if "lora" in n.lower():
            p.requires_grad_(True)
            lora_params.append(p)
        else:
            p.requires_grad_(False)

    head_params = [heads["slots"], heads["w_r"], heads["g_head"], heads["g_bias"],
                   heads["lambda_sal"], delta, logit_scale, tau]
    param_groups = [{"params": head_params, "lr": lr_heads}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora})
    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)

    # --- precompute candidate-side stats as device tensors keyed by id ------
    rho = float(loss_spec.rho_fixed_during_stage_b)
    item_ids = list(encoder._item_vecs.keys())
    id_to_idx = {i: k for k, i in enumerate(item_ids)}
    H = torch.tensor([encoder._item_vecs[i] for i in item_ids], device=dev, dtype=dt)  # [|I|, d_item]
    n_t = torch.tensor([float(n_support.get(i, 0.0)) for i in item_ids], device=dev, dtype=dt)
    p_t = torch.tensor([float(s_prior.get(i, 0.0)) for i in item_ids], device=dev, dtype=dt)

    metrics = {"per_epoch_loss": [], "per_epoch_loss_breakdown": [], "tau_schedule": [],
               "n_trainable_heads": sum(p.numel() for p in head_params),
               "n_trainable_lora": sum(p.numel() for p in lora_params)}

    losses_w = {"rank": loss_spec.w_rank, "attr": loss_spec.w_attr, "bal": loss_spec.w_bal,
                "orth": loss_spec.w_orth, "use": loss_spec.w_use, "tau": loss_spec.w_tau}

    rng = torch.Generator().manual_seed(int(seed))
    n_train = len(train_examples)
    cap = int(max_examples_per_epoch) if max_examples_per_epoch else n_train

    for epoch in range(num_epochs):
        order = torch.randperm(n_train, generator=rng)[:cap].tolist()
        # tau target = linear schedule over (epoch + step_frac)/num_epochs
        epoch_tot = {k: 0.0 for k in ("L_total", "L_rank", "L_attr", "L_bal", "L_orth", "L_use", "L_tau")}
        opt.zero_grad(set_to_none=True)
        seen = 0
        for step_i, idx in enumerate(order):
            frac = (epoch + step_i / max(cap, 1)) / max(num_epochs, 1)
            tau_target = _anneal_tau(frac, start=loss_spec.tau_start, end=loss_spec.tau_max,
                                     lo=loss_spec.tau_min, hi=loss_spec.tau_max)

            ex = train_examples[idx]
            try:
                panel, pos_idx = _example_panel(ex)
            except KeyError:
                continue
            hist = ex.get("history") or ex.get("history_item_ids") or []

            # Forward through Qwen+LoRA -> intent slot hiddens -> (z, pi)
            u = encoder.slot_hiddens(hist, item_lookup, dropout=False)  # [K, d_h]
            z, pi = encoder.intents_from_hiddens(u, hist)               # [K, d_item], [K]
            # Recompute residual r for L_attr (intents_from_hiddens did not return it)
            anchors = encoder.anchors.to(z.device, z.dtype)
            r = z - anchors                                              # [K, d_item]

            # Candidate vectors / stats indexed by id
            cand_idx = torch.tensor([id_to_idx[c] for c in panel if c in id_to_idx],
                                    device=dev, dtype=torch.long)
            if cand_idx.numel() == 0:
                continue
            h_c = H.index_select(0, cand_idx)                            # [N, d]
            n_c = n_t.index_select(0, cand_idx)                          # [N]
            p_c = p_t.index_select(0, cand_idx)                          # [N]

            scores, s_pers, resp = torch_calm_scores(
                z=z, pi=pi, h=h_c, n_support=n_c, s_prior=p_c, delta=delta,
                tau=tau.clamp(loss_spec.tau_min, loss_spec.tau_max),
                rho=rho, normalize=True, logit_scale=logit_scale,
            )
            # Adjust pos_idx if any candidate was missing from the cache
            kept_ids = [c for c in panel if c in id_to_idx]
            pos_new = kept_ids.index(panel[pos_idx]) if panel[pos_idx] in kept_ids else None
            if pos_new is None:
                continue
            target = torch.tensor([pos_new], device=dev)

            L_rank = torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)
            extras = _stage_b_losses(resp, r, z, tau, tau_target=tau_target)
            L_total = (losses_w["rank"] * L_rank
                       + losses_w["attr"] * extras["L_attr"]
                       + losses_w["bal"]  * extras["L_bal"]
                       + losses_w["orth"] * extras["L_orth"]
                       + losses_w["use"]  * extras["L_use"]
                       + losses_w["tau"]  * extras["L_tau"])

            (L_total / grad_accum_steps).backward()
            seen += 1
            if seen % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in head_params + lora_params if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                # keep tau hard-clamped after the step
                with torch.no_grad():
                    tau.clamp_(loss_spec.tau_min, loss_spec.tau_max)

            epoch_tot["L_total"] += float(L_total.detach())
            epoch_tot["L_rank"]  += float(L_rank.detach())
            for k in ("L_attr", "L_bal", "L_orth", "L_use", "L_tau"):
                epoch_tot[k] += float(extras[k].detach())

            if (step_i + 1) % max(log_every, 1) == 0:
                # leave logging to the caller (script wraps in tqdm)
                pass

        # Flush any remaining grads
        if seen % grad_accum_steps:
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                tau.clamp_(loss_spec.tau_min, loss_spec.tau_max)
        n_examples = max(seen, 1)
        metrics["per_epoch_loss"].append(epoch_tot["L_total"] / n_examples)
        metrics["per_epoch_loss_breakdown"].append({k: v / n_examples for k, v in epoch_tot.items()})
        metrics["tau_schedule"].append(float(tau.detach()))

    # persist trained heads (+ delta, logit_scale, tau) for the eval runner
    if save_extras_path:
        meta = {
            "num_epochs": num_epochs, "lr_heads": lr_heads, "lr_lora": lr_lora,
            "grad_accum_steps": grad_accum_steps, "rho_fixed": rho,
            "tau_final": float(tau.detach()), "logit_scale_final": float(logit_scale.detach()),
            "loss_spec": loss_spec.as_dict(),
            "n_trainable_heads": metrics["n_trainable_heads"],
            "n_trainable_lora": metrics["n_trainable_lora"],
            "per_epoch_loss": metrics["per_epoch_loss"],
            "tau_schedule": metrics["tau_schedule"],
        }
        head_out = {**heads, "delta": delta, "logit_scale": logit_scale, "tau": tau}
        save_extras(save_extras_path, head_out, meta)

    return metrics
