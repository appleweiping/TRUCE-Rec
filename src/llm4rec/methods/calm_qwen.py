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
