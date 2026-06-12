"""CALM-Rec encoders: cached item encoder + (LoRA) intent encoder.

Two implementations of the ``ItemEncoder`` / ``IntentEncoder`` protocols from
``calm_rec.py``:

1. ``HashedItemEncoder`` / ``LexiconIntentEncoder`` — dependency-free, deterministic, CPU-only.
   These let the FULL CALM-Rec pipeline (fit -> train scaffold -> rank -> evaluate) run locally
   with no GPU and no HF download, so any agent can smoke the end-to-end path. They are NOT paper
   evidence: the item vector is a hashed character-n-gram bag, and intents are attribute-lexicon
   projections. Good enough to exercise the math and the contracts; not a trained model.

2. ``QwenItemEncoder`` / ``QwenIntentEncoder`` — the real server-side path (Qwen3-8B, + LoRA and
   K attribute-anchored soft-prompt intent slots for the intent encoder). These are gated behind
   an explicit ``backend="qwen"`` + model-path argument and raise a clear error if the weights /
   transformers are not available, so they never silently run on the wrong machine. Wiring them is
   a server task; the interface they satisfy is already exercised by the hashed encoders' tests.

See docs/method_calm_rec_spec.md (sections 1, 4, 5, 10) and docs/CALM_REC_RUNBOOK.md.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from llm4rec.methods.calm_rec import IntentSet
from llm4rec.methods.calm_weak_labels import AttributeLexicon, default_beauty_lexicon


def _tokens(text: str) -> list[str]:
    return [t for t in str(text).lower().replace("|", " ").replace("/", " ").split() if t]


def _item_text(item_row: dict[str, Any]) -> str:
    return " ".join(
        str(item_row.get(f, "") or "")
        for f in ("title", "category", "brand", "description", "ingredients", "genres")
    ).strip()


def _hash_dim(token: str, dim: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % dim


# --------------------------------------------------------------------------------------
# CPU / dependency-free encoders (deterministic; NOT paper evidence).
# --------------------------------------------------------------------------------------
class HashedItemEncoder:
    """Deterministic hashed bag-of-tokens item vector (L2-normalised). CPU-only, no model.

    Optionally appends the lexicon facet soft-label as extra dimensions so the intent encoder's
    attribute anchoring has a real signal to align to.
    """

    def __init__(self, *, dim: int = 64, lexicon: AttributeLexicon | None = None) -> None:
        self.dim = int(dim)
        self.lexicon = lexicon
        self.facet_dim = lexicon.n_facets if lexicon else 0

    @property
    def out_dim(self) -> int:
        return self.dim + self.facet_dim

    def encode(self, item_id: str, item_row: dict[str, Any]) -> list[float]:
        vec = [0.0] * self.dim
        toks = _tokens(_item_text(item_row)) or [str(item_id)]
        for tok in toks:
            vec[_hash_dim(tok, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        if self.lexicon:
            soft = self.lexicon.soft_label(item_row)
            vec = vec + [soft[f] for f in self.lexicon.facet_names]
        return vec


class LexiconIntentEncoder:
    """K attribute-anchored intents built from the user's history via the attribute lexicon.

    Anchor c_k = the unit vector selecting facet k's dimension(s) in the hashed-item space; the
    personalized residual is the user's facet emphasis aggregated over history. dropout=True adds a
    small deterministic, seeded perturbation so the M-pass ensemble variance is non-zero (drives the
    trust gate) without any stochastic model. The projected intent z_uk lives in item-vector space,
    matching the ``IntentSet`` contract (z already W-projected).
    """

    def __init__(
        self,
        item_encoder: HashedItemEncoder,
        *,
        n_intents: int,
        lexicon: AttributeLexicon,
        jitter: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.item_encoder = item_encoder
        self.n_intents = int(n_intents)
        self.lexicon = lexicon
        self.jitter = float(jitter)
        self.dim = item_encoder.out_dim
        self.facet_base = item_encoder.dim  # facet dims start here
        self._draw = int(seed)

    def encode_intents(self, history_item_ids, item_lookup, *, dropout: bool = False) -> IntentSet:
        # Aggregate history item vectors and facet emphasis.
        agg = [0.0] * self.dim
        facet_mass = [0.0] * self.lexicon.n_facets
        n_hist = 0
        for h in history_item_ids:
            row = item_lookup.get(str(h), {})
            hv = self.item_encoder.encode(str(h), row)
            for j, v in enumerate(hv):
                agg[j] += v
            soft = self.lexicon.soft_label(row)
            for fi, f in enumerate(self.lexicon.facet_names):
                facet_mass[fi] += soft[f]
            n_hist += 1
        z: list[list[float]] = []
        weights: list[float] = []
        for k in range(self.n_intents):
            facet_idx = k % max(1, self.lexicon.n_facets)
            zk = [0.0] * self.dim
            # personalized residual: history emphasis on the (hashed) content dims
            for j in range(self.facet_base):
                zk[j] = agg[j]
            # anchor: select facet k's dimension
            if self.lexicon.n_facets:
                zk[self.facet_base + facet_idx] = 1.0
            if dropout:
                self._draw += 1
                scale = 1.0 + self.jitter * (((self._draw * 7) % 5) - 2)
                zk = [v * scale for v in zk]
            mass = (facet_mass[facet_idx] if self.lexicon.n_facets else 1.0) + 1e-6
            z.append(zk)
            weights.append(mass)
        total = sum(weights) or float(self.n_intents)
        pi = [w / total for w in weights]
        return IntentSet(z=z, pi=pi)


# --------------------------------------------------------------------------------------
# Real Qwen3-8B path (server-side). Gated: raises a clear error unless explicitly enabled
# with available weights + transformers. Wiring/loading is a server task.
# --------------------------------------------------------------------------------------
_QWEN_HELP = (
    "CALM-Rec Qwen3-8B encoders require the model weights + transformers/peft on a GPU host. "
    "This is the server path; see docs/CALM_REC_RUNBOOK.md. On the local/dev machine use the "
    "hashed/lexicon encoders (backend='hashed') for smoke runs."
)


@dataclass
class QwenItemEncoder:
    """Frozen Qwen3-8B item encoder (offline, fp16 disk cache). Server/GPU path.

    Thin shell over calm_qwen.QwenItemEncoderRuntime so this module stays
    importable without torch; the runtime (and torch) load on first use.
    """

    model_path: str
    pooling: str = "mean"
    cache_path: str | None = None
    device: str = "cuda"
    _rt: Any = field(default=None, repr=False)

    def _runtime(self):
        if self._rt is None:
            try:
                from llm4rec.methods.calm_qwen import (
                    QwenBackboneRuntime,
                    QwenItemEncoderRuntime,
                )
            except ImportError as e:  # pragma: no cover
                raise NotImplementedError(_QWEN_HELP) from e
            backbone = QwenBackboneRuntime(self.model_path, device=self.device)
            self._rt = QwenItemEncoderRuntime(
                backbone, pooling=self.pooling, cache_path=self.cache_path
            )
        return self._rt

    def encode(self, item_id: str, item_row: dict[str, Any]) -> list[float]:  # pragma: no cover
        try:
            return self._runtime().encode(item_id, item_row)
        except (ImportError, OSError, EnvironmentError) as e:
            # gated: never silently runs on a box without torch/weights
            raise NotImplementedError(_QWEN_HELP) from e


@dataclass
class QwenIntentEncoder:
    """Qwen3-8B + LoRA intent encoder with K attribute-anchored soft-prompt slots.

    Thin shell over calm_qwen.QwenIntentEncoderRuntime (torch loads lazily).
    ``anchors`` / Stage-A stats are injected by the trainer / run script before
    the first encode_intents call.
    """

    model_path: str
    n_intents: int
    lora_path: str | None = None
    extras_path: str | None = None
    device: str = "cuda"
    _rt: Any = field(default=None, repr=False)

    def runtime(self):
        if self._rt is None:
            try:
                from llm4rec.methods.calm_qwen import (
                    QwenBackboneRuntime,
                    QwenIntentEncoderRuntime,
                )
            except ImportError as e:  # pragma: no cover
                raise NotImplementedError(_QWEN_HELP) from e
            backbone = QwenBackboneRuntime(self.model_path, device=self.device)
            if self.lora_path:
                backbone.attach_lora(self.lora_path)
            self._rt = QwenIntentEncoderRuntime(
                backbone, n_intents=self.n_intents, extras_path=self.extras_path
            )
        return self._rt

    def encode_intents(self, history_item_ids, item_lookup, *, dropout: bool = False) -> IntentSet:  # pragma: no cover
        try:
            rt = self.runtime()
            if rt.anchors is None:
                raise RuntimeError(
                    "QwenIntentEncoder needs anchors before encoding: call "
                    "runtime().anchors = anchors_from_weak_labels(...) (see "
                    "scripts/train_calm_stage_b.py / run_calm_rec.py wiring)."
                )
            return rt.encode_intents(history_item_ids, item_lookup, dropout=dropout)
        except (ImportError, OSError, EnvironmentError) as e:
            # gated: never silently runs on a box without torch/weights
            raise NotImplementedError(_QWEN_HELP) from e


def build_encoders(
    *,
    backend: str = "hashed",
    n_intents: int = 4,
    dim: int = 64,
    lexicon: AttributeLexicon | None = None,
    qwen_model_path: str | None = None,
    qwen_lora_path: str | None = None,
    qwen_cache_path: str | None = None,
    qwen_extras_path: str | None = None,
    seed: int = 0,
) -> tuple[Any, Any]:
    """Construct (item_encoder, intent_encoder) for the configured backend.

    backend='hashed' (default): CPU, dependency-free, runs anywhere — for smoke / contract tests.
    backend='qwen': real Qwen3-8B (+LoRA); requires a GPU host with weights (raises otherwise).
    qwen_cache_path: fp16 item-vector cache (.npz); qwen_extras_path: trained Stage-B heads (.pt).
    """
    lex = lexicon or default_beauty_lexicon()
    if backend == "hashed":
        ie = HashedItemEncoder(dim=dim, lexicon=lex)
        qe = LexiconIntentEncoder(ie, n_intents=n_intents, lexicon=lex, seed=seed)
        return ie, qe
    if backend == "qwen":
        if not qwen_model_path:
            raise ValueError("backend='qwen' requires qwen_model_path. " + _QWEN_HELP)
        ie = QwenItemEncoder(model_path=qwen_model_path, cache_path=qwen_cache_path)
        qe = QwenIntentEncoder(
            model_path=qwen_model_path,
            n_intents=n_intents,
            lora_path=qwen_lora_path,
            extras_path=qwen_extras_path,
        )
        return ie, qe
    raise ValueError(f"unknown CALM encoder backend: {backend!r} (use 'hashed' or 'qwen')")
