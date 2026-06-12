"""Parity: eval_calm_beauty's cached-signal scorer vs the pure-python ranker.

The formal evaluator computes per-candidate signals once (vectorized) and then
derives every ladder variant from the cache. Any drift vs CALMRecRanker's
per-candidate math would silently change reported numbers. Skips without numpy.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from llm4rec.methods.calm_rec import CALMConfig, CALMRecRanker, IntentSet, RhoCoeffs

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
try:
    from eval_calm_beauty import metrics_full, mix_scores, signals_for_examples
finally:
    sys.path.remove(str(_SCRIPTS))

D = 6
K = 3
M = 4


class DictItemEncoder:
    def __init__(self, vecs):
        self.vecs = vecs

    def encode(self, item_id, item_row):
        return list(self.vecs[str(item_id)])


class SeqIntentEncoder:
    """Deterministic stand-in: fixed no-dropout IntentSet, cycling dropout sets."""

    def __init__(self, base: IntentSet, ensemble: list[IntentSet]):
        self.base = base
        self.ensemble = ensemble
        self._i = 0

    def encode_intents(self, history_item_ids, item_lookup, *, dropout: bool = False):
        if not dropout:
            return self.base
        s = self.ensemble[self._i % len(self.ensemble)]
        self._i += 1
        return s


def _intent_set(rng) -> IntentSet:
    z = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(K)]
    raw = [rng.random() + 0.05 for _ in range(K)]
    t = sum(raw)
    return IntentSet(z=z, pi=[p / t for p in raw])


def _setup(seed=5):
    rng = random.Random(seed)
    items = [f"i{j}" for j in range(12)]
    vecs = {i: [rng.gauss(0, 1) for _ in range(D)] for i in items}
    catalog = [{"item_id": i, "title": i} for i in items]
    train = [
        {"user_id": f"u{t}", "history": [items[(t + j) % 12] for j in range(3)],
         "target": items[(t + 3) % 12], "split": "train"}
        for t in range(8)
    ]
    tests = [
        {"user_id": f"u{t}", "history": [items[(t + j) % 12] for j in range(3)],
         "target": items[(t + 5) % 12], "split": "test",
         "candidates": items[:9], "candidate_items": items[:9]}
        for t in range(4)
    ]
    base = _intent_set(rng)
    ensemble = [_intent_set(rng) for _ in range(M)]
    delta = [rng.random() * 0.2 for _ in range(K)]
    return vecs, catalog, train, tests, base, ensemble, delta


@pytest.mark.parametrize("coeffs", [None, RhoCoeffs(a0=-0.4, a1=1.0, a2=0.5, a3=0.3)])
def test_cached_signals_match_ranker_scores(coeffs):
    vecs, catalog, train, tests, base, ensemble, delta = _setup()
    cfg = CALMConfig(n_intents=K, tau=2.0, m_dropout=M,
                     use_trust_gate=coeffs is not None, seed=0)
    ranker = CALMRecRanker(
        item_encoder=DictItemEncoder(vecs),
        intent_encoder=SeqIntentEncoder(base, list(ensemble)),
        config=cfg,
    )
    ranker.fit(train, catalog)
    ranker.delta = list(delta)
    if coeffs is not None:
        ranker.set_rho(RhoCoeffs(**vars(coeffs)))

    # path B: cached signals with a FRESH encoder (same deterministic sequence)
    signals = signals_for_examples(
        tests,
        intent_encoder=SeqIntentEncoder(base, list(ensemble)),
        item_lookup={r["item_id"]: r for r in catalog},
        item_vecs=vecs,
        support=ranker.item_support,
        s_prior=ranker.s_prior,
        tau=cfg.tau,
        delta=delta,
        m_dropout=M,
    )
    assert len(signals) == len(tests)

    for ex, sig in zip(tests, signals):
        res = ranker.rank(ex, ex["candidate_items"])
        ranker_scores = dict(zip(res.predicted_items, res.scores))
        cached = mix_scores(sig, coeffs)
        for j, cand in enumerate(sig["cands"]):
            assert math.isclose(
                float(cached[j]), ranker_scores[cand], rel_tol=1e-9, abs_tol=1e-9
            ), (cand, float(cached[j]), ranker_scores[cand])


def test_metrics_full_ndcg_matches_manual():
    vecs, catalog, train, tests, base, ensemble, delta = _setup()
    signals = signals_for_examples(
        tests,
        intent_encoder=SeqIntentEncoder(base, list(ensemble)),
        item_lookup={r["item_id"]: r for r in catalog},
        item_vecs=vecs,
        support={i["item_id"]: 1 for i in catalog},
        s_prior={i["item_id"]: 0.0 for i in catalog},
        tau=2.0,
        delta=delta,
        m_dropout=M,
    )
    m = metrics_full(signals, None)
    # manual NDCG@10 from the same cached arrays
    total = 0.0
    for sig in signals:
        order = np.argsort(-sig["s_pers"], kind="stable")
        pos = int(np.where(order == sig["target_idx"])[0][0])
        total += (1.0 / math.log2(pos + 2)) if pos < 10 else 0.0
    assert math.isclose(m["NDCG@10"], total / len(signals), rel_tol=1e-12)
    assert 0.0 <= m["MRR"] <= 1.0 and 0.0 <= m["HR@20"] <= 1.0
