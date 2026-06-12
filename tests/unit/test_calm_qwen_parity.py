"""Parity tests: the differentiable Stage-B scorer must match the pure-python core.

torch_calm_scores (calm_qwen.py) is what the GPU gradient loop optimizes;
calm_rec.py's pure-python functions are what inference/eval reports. Any drift
between them silently invalidates training. Skips without torch/numpy.
"""

from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from llm4rec.methods.calm_qwen import anchors_from_weak_labels, torch_calm_scores
from llm4rec.methods.calm_rec import (
    IntentSet,
    calibrated_mix,
    mixture_score_and_responsibilities,
    per_intent_energy,
)


def _rand_case(seed: int, K: int = 4, N: int = 17, d: int = 12):
    rng = random.Random(seed)
    z = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(K)]
    raw_pi = [rng.random() + 0.05 for _ in range(K)]
    s = sum(raw_pi)
    pi = [p / s for p in raw_pi]
    h = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(N)]
    n_sup = [rng.randrange(0, 50) for _ in range(N)]
    prior = [rng.gauss(0, 1) for _ in range(N)]
    delta = [rng.random() * 0.3 for _ in range(K)]
    return z, pi, h, n_sup, prior, delta


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("tau", [1.0, 2.0, 5.5])
def test_torch_scores_match_pure_python_core(seed, tau):
    z, pi, h, n_sup, prior, delta, = _rand_case(seed)
    rho = 0.1

    scores_t, s_pers_t, resp_t = torch_calm_scores(
        torch.tensor(z, dtype=torch.float64),
        torch.tensor(pi, dtype=torch.float64),
        torch.tensor(h, dtype=torch.float64),
        torch.tensor([float(x) for x in n_sup], dtype=torch.float64),
        torch.tensor(prior, dtype=torch.float64),
        torch.tensor(delta, dtype=torch.float64),
        tau=tau,
        rho=rho,
    )

    intents = IntentSet(z=z, pi=pi)
    for i in range(len(h)):
        e = per_intent_energy(intents, h[i], item_support=n_sup[i], delta=delta)
        s_pers, resp = mixture_score_and_responsibilities(intents, e, tau=tau)
        score = calibrated_mix(s_pers, prior[i], rho)
        assert math.isclose(float(s_pers_t[i]), s_pers, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(float(scores_t[i]), score, rel_tol=1e-9, abs_tol=1e-9)
        for k in range(len(pi)):
            assert math.isclose(float(resp_t[i, k]), resp[k], rel_tol=1e-8, abs_tol=1e-9)


def test_torch_scores_rho_floor_caps_prior_weight():
    z, pi, h, n_sup, prior, delta = _rand_case(3)
    big_rho, floor = 0.99, 0.15
    scores_t, s_pers_t, _ = torch_calm_scores(
        torch.tensor(z), torch.tensor(pi), torch.tensor(h),
        torch.tensor([float(x) for x in n_sup]), torch.tensor(prior),
        torch.tensor(delta), tau=2.0, rho=big_rho, rho_floor=floor,
    )
    intents = IntentSet(z=z, pi=pi)
    e = per_intent_energy(intents, h[0], item_support=n_sup[0], delta=delta)
    s_pers, _ = mixture_score_and_responsibilities(intents, e, tau=2.0)
    expected = calibrated_mix(s_pers, prior[0], big_rho, rho_floor=floor)
    assert math.isclose(float(scores_t[0]), expected, rel_tol=1e-5, abs_tol=1e-5)


def test_torch_scores_are_differentiable_through_z_pi_delta_tau():
    z, pi, h, n_sup, prior, delta = _rand_case(9)
    zt = torch.tensor(z, requires_grad=True)
    pit = torch.tensor(pi, requires_grad=True)
    dt = torch.tensor(delta, requires_grad=True)
    taut = torch.tensor(2.0, requires_grad=True)
    scores, _, _ = torch_calm_scores(
        zt, pit, torch.tensor(h), torch.tensor([float(x) for x in n_sup]),
        torch.tensor(prior), dt, tau=taut, rho=0.1,
    )
    loss = -torch.log_softmax(scores, dim=0)[0]
    loss.backward()
    for t in (zt, pit, dt, taut):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()


def test_anchors_from_weak_labels_centroids_and_fallback():
    vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    facet_top = {"f1": ["a", "c"], "f2": ["b"]}
    anchors = anchors_from_weak_labels(facet_top, vecs, n_intents=4)
    assert anchors.shape == (4, 2)
    assert torch.allclose(anchors[0], torch.tensor([1.0, 0.5]))   # mean of a, c
    assert torch.allclose(anchors[1], torch.tensor([0.0, 1.0]))   # b
    assert torch.allclose(anchors[2], anchors[0])                 # wraps around facets
    # facet with no known vectors -> zero anchor
    anchors2 = anchors_from_weak_labels({"f": ["missing"]}, vecs, n_intents=1)
    assert torch.allclose(anchors2[0], torch.zeros(2))


def test_item_encoder_cache_roundtrip(tmp_path):
    """Cache load path needs only numpy (no model): write npz, read via runtime."""
    from llm4rec.methods.calm_qwen import QwenItemEncoderRuntime

    path = tmp_path / "vecs.npz"
    ids = np.array(["i1", "i2"])
    vecs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float16")
    np.savez_compressed(path, item_ids=ids, vectors=vecs)
    enc = QwenItemEncoderRuntime(backbone=None, cache_path=str(path))  # type: ignore[arg-type]
    assert enc.encode("i1", {}) == pytest.approx([1.0, 2.0])
    assert enc.encode("i2", {}) == pytest.approx([3.0, 4.0])
