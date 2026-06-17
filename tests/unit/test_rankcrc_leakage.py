"""Negative-control leakage test for RankCRC (formalization fix #7).

The no-leakage audit (`assert_no_leakage`) enforces a column-allowlist: g's feature
matrix must NOT contain any H_target-derived / label-derived column. This test proves
the audit HAS TEETH by intentionally injecting a forbidden column and asserting the
audit FAILS — a column-allowlist that never rejects anything is no safeguard.

It also confirms the B1/B2 frozen-lambda split's disjointness assert rejects overlap.

Skips cleanly if numpy/sklearn are unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
try:
    from rankcrc_validate import (
        FEATURE_NAMES,
        FORBIDDEN_FEATURE_NAMES,
        assert_disjoint,
        assert_no_leakage,
        build_features_labels,
        crc_certify_at_coverage,
    )
finally:
    sys.path.remove(str(_SCRIPTS))


def _toy_user(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    s = rng.normal(size=101)
    return {
        "s_pers": s,
        "H": rng.uniform(0.0, 1.0, size=101),
        "var_m": rng.uniform(0.0, 0.5, size=101),
        "n_i": rng.integers(1, 500, size=101).astype(float),
        "prior": rng.uniform(-5.0, 0.0, size=101),
        "target_idx": int(rng.integers(0, 101)),
    }


# ---------------------------------------------------------------------------
# Negative control: injecting an H_target-derived feature MUST trip the audit.
# ---------------------------------------------------------------------------
def test_clean_feature_list_passes_audit() -> None:
    """The shipped label-free feature list must pass (no false positive)."""
    assert_no_leakage(list(FEATURE_NAMES))  # must not raise


@pytest.mark.parametrize("forbidden", FORBIDDEN_FEATURE_NAMES)
def test_injected_leak_feature_fails_audit(forbidden: str) -> None:
    """Inject each forbidden (H_target / label-derived) column; the audit MUST raise."""
    poisoned = list(FEATURE_NAMES) + [forbidden]
    with pytest.raises(ValueError, match="leakage audit FAILED|H_target FORBIDDEN"):
        assert_no_leakage(poisoned)


def test_build_features_labels_blocks_injected_leak() -> None:
    """The end-to-end builder must refuse a feature_names list containing a leak column
    BEFORE any model is fit — proving the audit gates the g-training path."""
    users = [_toy_user(i) for i in range(8)]
    poisoned = list(FEATURE_NAMES) + ["H_target"]
    with pytest.raises(ValueError, match="leakage audit FAILED|H_target FORBIDDEN"):
        build_features_labels(users, k=10, feature_names=poisoned)


def test_build_features_labels_clean_path_succeeds() -> None:
    """Sanity: the clean (default) feature path builds without raising."""
    users = [_toy_user(i) for i in range(8)]
    X, ranks, corr, risks, ndcg = build_features_labels(users, k=10)
    assert X.shape == (8, len(FEATURE_NAMES))
    assert set(np.unique(corr)).issubset({0.0, 1.0})
    assert np.all((risks >= 0.0) & (risks <= 1.0))


# ---------------------------------------------------------------------------
# B1/B2 disjointness: lambda-selection indices must be disjoint from loss-averaging.
# ---------------------------------------------------------------------------
def test_disjoint_folds_pass() -> None:
    a = np.array([0, 1, 2])
    b1 = np.array([3, 4])
    b2 = np.array([5, 6])
    c = np.array([7, 8, 9])
    assert_disjoint(a, b1, b2, c)  # must not raise


def test_overlapping_folds_fail() -> None:
    """B1 and B2 sharing an index (lambda chosen on a user whose loss is averaged) MUST raise."""
    b1 = np.array([3, 4, 5])
    b2 = np.array([5, 6])  # index 5 also in B1 -> lambda/loss leakage
    with pytest.raises(AssertionError, match="disjointness FAILED"):
        assert_disjoint(b1, b2)


def test_duplicate_within_fold_fails() -> None:
    with pytest.raises(AssertionError, match="disjointness FAILED"):
        assert_disjoint(np.array([1, 1, 2]))


# ---------------------------------------------------------------------------
# crc_certify_at_coverage: lambda comes from B1, losses averaged on B2 (structural).
# ---------------------------------------------------------------------------
def test_crc_certificate_uses_disjoint_arms() -> None:
    """lambda must be the (1-c) quantile of g_b1, and the served mean must come from
    risks_b2 (NOT g_b1). We verify by making B2 risks all zero: the corrected alpha
    must collapse to the pure CRC slack 1/(m+1), independent of B1."""
    rng = np.random.default_rng(0)
    g_b1 = rng.uniform(0, 1, size=200)
    g_b2 = rng.uniform(0, 1, size=200)
    risks_b2 = np.zeros(200)
    cc = crc_certify_at_coverage(g_b1, g_b2, risks_b2, c_frac=0.5, B=1.0, jitter_seed=0)
    assert cc["m_b2"] > 0
    # raw served mean over B2 is 0 -> certified alpha == B/(m+1)
    assert abs(cc["raw_b2"] - 0.0) < 1e-12
    assert abs(cc["certified_alpha"] - 1.0 / (cc["m_b2"] + 1.0)) < 1e-9


def test_crc_jitter_triggers_on_atom() -> None:
    """If g_b1 has a heavy atom at its operating quantile, jitter must be applied.

    Construct g so the (1-c) quantile lands exactly ON the atom: a mass of 100 points
    at 0.5 below a continuous tail in [0.6,1.0]. At c_frac=0.6 the (1-0.6)=0.4 quantile
    of the 200-point array sits at 0.5 (on the atom)."""
    g_b1 = np.concatenate([np.full(100, 0.5), np.linspace(0.6, 1.0, 100)])
    g_b2 = np.concatenate([np.full(100, 0.5), np.linspace(0.6, 1.0, 100)])
    risks_b2 = np.zeros(200)
    cc = crc_certify_at_coverage(g_b1, g_b2, risks_b2, c_frac=0.6, B=1.0, jitter_seed=0)
    assert cc["jitter_applied"] is True
    assert cc["jitter_eps"] > 0.0


def test_crc_no_jitter_on_continuous_g() -> None:
    """A continuous (tie-free) g must NOT trigger jitter (no false positive)."""
    rng = np.random.default_rng(1)
    g_b1 = rng.uniform(0, 1, size=200)
    g_b2 = rng.uniform(0, 1, size=200)
    risks_b2 = rng.uniform(0, 1, size=200)
    cc = crc_certify_at_coverage(g_b1, g_b2, risks_b2, c_frac=0.5, B=1.0, jitter_seed=0)
    assert cc["jitter_applied"] is False
