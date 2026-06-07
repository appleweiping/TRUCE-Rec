"""CALM-Rec — Calibrated trust over Attribute-anchored Latent Multi-intent (TRUCE-Rec Ours v2).

Evolution of SCALR via a 20-iteration tri-agent design review
(see docs/method_calm_rec_spec.md and outputs/method_redesign_discussion/iterations/).

Headline contribution: CALM-Rec learns, PER USER-ITEM, how much to trust the LLM's personalized
multi-intent judgment versus a history-free item prior, using the geometry of its own intent
mixture (responsibility entropy + ensemble variance) as an ENDOGENOUS reliability signal. Multi-intent
is the vehicle; calibrated personalization-trust is the contribution.

Scoring (per candidate i, user u with K attribute-anchored intents):
    e_uik    = <W z_uk, h_i> - delta_k * log(1 + n_i)          # per-intent energy, popularity-residualized
    s_pers   = (1/tau) * logsumexp_k( log pi_uk + tau * e_uik ) # soft-OR mixture over intents
    r_uik    = softmax_k( log pi_uk + tau * e_uik )             # intent responsibilities
    H(r)     = -sum_k r_uik log r_uik                           # responsibility entropy (reliability signal)
    rho_ui   = sigmoid( a0 + a1*H + a2*Var_m - a3*log(1+n_i) )  # trust gate (a1,a2>=0)
    s_ui     = (1 - rho_ui) * s_pers + rho_ui * s_prior_i       # calibrated trust mixing
    score    = s_ui   (rank the 101 candidates by score, descending)

Design invariants (locked by the review; see the spec doc):
- Candidate item vectors h_i are PRECOMPUTED by a cached item encoder. The 101 candidates are NEVER
  placed in the LLM prompt (position noise breaks same-candidate consistency). Scoring is a vector-space
  matmul, so it is fully testable here without a GPU.
- Uncertainty changes the PREDICTION (mixes toward the prior), it never gates/accepts/abstains and
  never collapses to a fallback ranker (the R3 failure of the retired method).
- Attribute anchors c_k are the firewall vs generic multi-interest (ComiRec): only the residual r_uk
  is personalized. Per-intent popularity residual + prior mixing keep popularity from dominating.
- rho's coefficients are fit POST-HOC on validation; a stage-2.5 AUC gate checks the reliability
  signal is real before trusting it. Test split is never used for selection.

The LLM (intent encoder + item encoder) is abstracted behind ``IntentEncoder`` / ``ItemEncoder`` so
this module has NO GPU dependency and is unit/smoke testable with deterministic mocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from llm4rec.rankers.base import CheckpointNotImplementedMixin, RankingResult


# --------------------------------------------------------------------------------------
# Linear-algebra helpers (pure python; small d, K, 101 candidates -> no numpy needed).
# --------------------------------------------------------------------------------------
def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = max(values)
    return float(m + math.log(sum(math.exp(v - m) for v in values)))


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def _entropy(probs: Sequence[float]) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 0.0))


# --------------------------------------------------------------------------------------
# Encoder abstractions. Real impl = Qwen3-8B (+ LoRA for the intent encoder), server-side.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IntentSet:
    """K intent vectors z_uk (already W-projected to item space) + mixture weights pi_uk."""

    z: list[list[float]]      # shape [K][d_item], = W z_uk (projected intent directions)
    pi: list[float]           # shape [K], mixture weights, sum to 1


class ItemEncoder(Protocol):
    """Caches a fixed vector h_i per item id (frozen Qwen3-8B over item text), offline."""

    def encode(self, item_id: str, item_row: dict[str, Any]) -> list[float]:
        ...


class IntentEncoder(Protocol):
    """Reads K projected intent vectors + weights from the user-side forward (LoRA, dropout-able)."""

    def encode_intents(
        self,
        history_item_ids: Sequence[str],
        item_lookup: dict[str, dict[str, Any]],
        *,
        dropout: bool = False,
    ) -> IntentSet:
        ...


# --------------------------------------------------------------------------------------
# Pure scoring core (the testable heart of CALM-Rec). No model, no GPU, no I/O.
# --------------------------------------------------------------------------------------
def per_intent_energy(
    intents: IntentSet,
    item_vec: Sequence[float],
    *,
    item_support: int,
    delta: Sequence[float],
) -> list[float]:
    """e_uik = <W z_uk, h_i> - delta_k * log(1 + n_i), per intent k."""
    pop = math.log(1.0 + max(0, int(item_support)))
    return [_dot(intents.z[k], item_vec) - float(delta[k]) * pop for k in range(len(intents.z))]


def mixture_score_and_responsibilities(
    intents: IntentSet,
    energies: Sequence[float],
    *,
    tau: float,
) -> tuple[float, list[float]]:
    """Soft-OR mixture-energy score s_pers and intent responsibilities r_k.

    s_pers = (1/tau) logsumexp_k( log pi_k + tau e_k );  r_k = softmax_k( log pi_k + tau e_k ).
    """
    logits = [math.log(max(p, 1e-12)) + float(tau) * float(e) for p, e in zip(intents.pi, energies)]
    s_pers = _logsumexp(logits) / float(tau)
    responsibilities = _softmax(logits)
    return s_pers, responsibilities


def trust_gate(
    *,
    resp_entropy: float,
    ensemble_var: float,
    item_support: int,
    coeffs: "RhoCoeffs",
) -> float:
    """rho_ui = sigmoid(a0 + a1*H(r) + a2*Var_m - a3*log(1+n_i)); a1,a2,a3 >= 0 enforced upstream."""
    pop = math.log(1.0 + max(0, int(item_support)))
    z = coeffs.a0 + coeffs.a1 * resp_entropy + coeffs.a2 * ensemble_var - coeffs.a3 * pop
    return 1.0 / (1.0 + math.exp(-z))


def calibrated_mix(s_pers: float, s_prior: float, rho: float, *, rho_floor: float = 0.15) -> float:
    """s_ui = (1 - rho') s_pers + rho' s_prior, with rho capped so personalization never vanishes."""
    rho_eff = min(max(float(rho), 0.0), 1.0 - float(rho_floor))
    return (1.0 - rho_eff) * float(s_pers) + rho_eff * float(s_prior)


@dataclass
class RhoCoeffs:
    """Trust-gate coefficients, fit POST-HOC on validation (a1,a2,a3 constrained >= 0)."""

    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0

    def clamp_nonneg(self) -> "RhoCoeffs":
        self.a1 = max(0.0, self.a1)
        self.a2 = max(0.0, self.a2)
        self.a3 = max(0.0, self.a3)
        return self


@dataclass
class CALMConfig:
    """CALM-Rec hyper-parameters. tau is learned/annealed in training; here it is the frozen value."""

    n_intents: int = 4          # K attribute-anchored intents (4 for sparse domains e.g. beauty)
    tau: float = 2.0            # mixture temperature, bounded [1, 8]
    m_dropout: int = 4          # MC-dropout passes for ensemble variance (8 for audit)
    max_history: int = 20
    use_trust_gate: bool = True # if False -> pure s_pers (ablation row rho=0)
    rho_floor: float = 0.15
    seed: int = 0

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "CALMConfig":
        p = dict(params or {})
        return cls(
            n_intents=int(p.get("n_intents", p.get("K", 4))),
            tau=float(min(8.0, max(1.0, p.get("tau", 2.0)))),
            m_dropout=int(p.get("m_dropout", 4)),
            max_history=int(p.get("max_history", 20)),
            use_trust_gate=bool(p.get("use_trust_gate", True)),
            rho_floor=float(p.get("rho_floor", 0.15)),
            seed=int(p.get("seed", 0)),
        )


def reliability_auc(signal: Sequence[float], correct: Sequence[float]) -> float:
    """Stage-2.5 gate: AUC of a reliability ``signal`` predicting whether s_pers was correct.

    Used to decide whether the trust gate is real (AUC > ~0.6) or noise (~0.5 -> drop the trust
    headline). Higher signal should mean LESS reliable, so we score (-signal) against correctness.
    Pure rank-based AUC (Mann-Whitney); no sklearn dependency.
    """
    pos = [-float(s) for s, c in zip(signal, correct) if c > 0.5]
    neg = [-float(s) for s, c in zip(signal, correct) if c <= 0.5]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _history_ids(example: dict[str, Any], max_history: int) -> list[str]:
    hist = example.get("history") or example.get("history_item_ids") or []
    return [str(x) for x in hist][-int(max_history):]


def _pstdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    return float((sum((v - mu) ** 2 for v in values) / n) ** 0.5)


class CALMRecRanker(CheckpointNotImplementedMixin):
    """Calibrated trust over Attribute-anchored Latent Multi-intent (TRUCE-Rec Ours).

    The Qwen3-8B item encoder and (LoRA) intent encoder are injected; pass deterministic mocks for
    smoke tests. Item vectors are cached at ``fit`` time; scoring is vector-space only.
    """

    method_name = "calm_rec"

    def __init__(
        self,
        *,
        item_encoder: ItemEncoder,
        intent_encoder: IntentEncoder,
        config: CALMConfig | None = None,
        method_config: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.item_encoder = item_encoder
        self.intent_encoder = intent_encoder
        mc = dict(method_config or {})
        self.method_name = str(mc.get("name") or self.method_name)
        self.config = config or CALMConfig.from_params(dict(mc.get("params") or {}))
        if config is None and not (mc.get("params") or {}).get("seed"):
            self.config.seed = int(mc.get("seed") or seed)
        self.item_lookup: dict[str, dict[str, Any]] = {}
        self.item_vec: dict[str, list[float]] = {}
        self.item_support: dict[str, int] = {}
        self.s_prior: dict[str, float] = {}
        self.delta: list[float] = [0.0] * self.config.n_intents
        self.rho: RhoCoeffs = RhoCoeffs(a0=-0.4)  # mild default until validation-calibrated

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.item_lookup = {str(r["item_id"]): r for r in item_catalog}
        # Train-only item support, with ALL targets excluded (leakage control).
        support: dict[str, int] = {str(r["item_id"]): 0 for r in item_catalog}
        for ex in train_examples:
            for h in _history_ids(ex, self.config.max_history):
                support[h] = support.get(h, 0) + 1
        self.item_support = support
        # Cache item vectors once (the cached item encoder).
        self.item_vec = {
            str(r["item_id"]): list(self.item_encoder.encode(str(r["item_id"]), r))
            for r in item_catalog
        }
        # History-free item prior s_prior_i = log(1 + support) (train-only, no user history).
        # The real prior is b_i + beta^T x_i; this train-only popularity proxy is the testable default.
        self.s_prior = {iid: math.log(1.0 + n) for iid, n in support.items()}

    def set_rho(self, coeffs: RhoCoeffs) -> None:
        """Install validation-calibrated trust-gate coefficients (a1,a2,a3 clamped >= 0)."""
        self.rho = coeffs.clamp_nonneg()

    def _score_candidate(
        self,
        intents: IntentSet,
        candidate: str,
        *,
        ensemble_var: float,
    ) -> tuple[float, dict[str, float]]:
        h_i = self.item_vec.get(candidate)
        if h_i is None:
            h_i = list(self.item_encoder.encode(candidate, self.item_lookup.get(candidate, {})))
        n_i = self.item_support.get(candidate, 0)
        energies = per_intent_energy(intents, h_i, item_support=n_i, delta=self.delta)
        s_pers, resp = mixture_score_and_responsibilities(intents, energies, tau=self.config.tau)
        s_prior = self.s_prior.get(candidate, 0.0)
        if not self.config.use_trust_gate:
            return s_pers, {"rho": 0.0, "H": _entropy(resp), "s_pers": s_pers, "s_prior": s_prior}
        H = _entropy(resp)
        rho = trust_gate(resp_entropy=H, ensemble_var=ensemble_var, item_support=n_i, coeffs=self.rho)
        score = calibrated_mix(s_pers, s_prior, rho, rho_floor=self.config.rho_floor)
        return score, {"rho": rho, "H": H, "s_pers": s_pers, "s_prior": s_prior}

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        candidates = [str(c) for c in candidate_items]
        history = _history_ids(example, self.config.max_history)

        # Intent posterior (one user-side forward), plus M dropout passes for ensemble variance.
        intents = self.intent_encoder.encode_intents(history, self.item_lookup, dropout=False)
        ensemble: list[IntentSet] = []
        if self.config.use_trust_gate and self.config.m_dropout > 1:
            ensemble = [
                self.intent_encoder.encode_intents(history, self.item_lookup, dropout=True)
                for _ in range(self.config.m_dropout)
            ]

        scores: dict[str, float] = {}
        audit: dict[str, dict[str, float]] = {}
        for c in candidates:
            var_m = 0.0
            if ensemble:
                h_i = self.item_vec.get(c) or list(self.item_encoder.encode(c, self.item_lookup.get(c, {})))
                n_i = self.item_support.get(c, 0)
                samples = []
                for ens in ensemble:
                    e = per_intent_energy(ens, h_i, item_support=n_i, delta=self.delta)
                    sp, _ = mixture_score_and_responsibilities(ens, e, tau=self.config.tau)
                    samples.append(sp)
                var_m = _pstdev(samples)
            scores[c], audit[c] = self._score_candidate(intents, c, ensemble_var=var_m)

        ordered = sorted(candidates, key=lambda c: (-scores.get(c, 0.0), c))
        return RankingResult(
            user_id=str(example["user_id"]),
            target_item=str(example["target"]),
            candidate_items=candidates,
            predicted_items=ordered,
            scores=[float(scores.get(c, 0.0)) for c in ordered],
            method=self.method_name,
            domain=str(example.get("domain") or "tiny"),
            raw_output=None,
            metadata={
                "example_id": example.get("example_id"),
                "split": example.get("split"),
                "calm_rec": {
                    "n_intents": self.config.n_intents,
                    "tau": self.config.tau,
                    "uses_trust_gate": bool(self.config.use_trust_gate),
                    "uses_gate_decision": False,   # mixes, never accepts/abstains
                    "uses_generation": False,
                    "mean_rho": (sum(a["rho"] for a in audit.values()) / len(audit)) if audit else 0.0,
                },
            },
        )


# --------------------------------------------------------------------------------------
# Deterministic mock encoders for smoke / unit tests. NOT a model; NOT paper evidence.
# --------------------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return [t for t in str(text).lower().replace("|", " ").split() if t]


class MockItemEncoder:
    """Bag-of-attribute one-hot item vectors over a fixed vocabulary of K attribute groups.

    Each item maps to a small vector whose dimensions correspond to attribute keywords, so the
    mock has genuine multi-attribute structure for the intent mixture to exploit (no GPU).
    """

    def __init__(self, vocab: Sequence[str]) -> None:
        self.vocab = list(vocab)
        self.index = {w: i for i, w in enumerate(self.vocab)}

    def encode(self, item_id: str, item_row: dict[str, Any]) -> list[float]:
        text = " ".join(str(item_row.get(f, "")) for f in ("title", "category", "brand", "description"))
        vec = [0.0] * len(self.vocab)
        for tok in _tokens(text):
            if tok in self.index:
                vec[self.index[tok]] = 1.0
        return vec


class MockIntentEncoder:
    """Builds K intent directions from the user's history bag-of-attributes.

    Intent k is anchored to the k-th slice of the attribute vocabulary (attribute-anchored),
    activated by history overlap. dropout=True adds small deterministic-but-seeded jitter so the
    M-pass ensemble variance is non-zero (drives the trust gate) without any model.
    """

    def __init__(self, item_encoder: MockItemEncoder, *, n_intents: int, jitter: float = 0.05) -> None:
        self.item_encoder = item_encoder
        self.n_intents = int(n_intents)
        self.jitter = float(jitter)
        self.dim = len(item_encoder.vocab)
        # partition vocab dims into K contiguous anchor groups
        self.groups = [list(range(i, self.dim, self.n_intents)) for i in range(self.n_intents)]
        self._draw = 0

    def encode_intents(self, history_item_ids, item_lookup, *, dropout: bool = False) -> IntentSet:
        agg = [0.0] * self.dim
        for h in history_item_ids:
            row = item_lookup.get(str(h), {})
            for j, v in enumerate(self.item_encoder.encode(str(h), row)):
                agg[j] += v
        z: list[list[float]] = []
        weights: list[float] = []
        for k in range(self.n_intents):
            zk = [0.0] * self.dim
            mass = 0.0
            for j in self.groups[k]:
                val = agg[j]
                if dropout:
                    self._draw += 1
                    val *= 1.0 + self.jitter * (((self._draw * 7) % 5) - 2)
                zk[j] = val
                mass += val
            z.append(zk)
            weights.append(mass)
        total = sum(weights)
        pi = [w / total for w in weights] if total > 0 else [1.0 / self.n_intents] * self.n_intents
        return IntentSet(z=z, pi=pi)
