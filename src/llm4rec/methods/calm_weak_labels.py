"""CALM-Rec attribute weak-labels (reproducible, no paid model, no leakage).

Produces soft multi-label attribute distributions for items from ITEM TEXT ONLY (title / brand /
category / description / ingredients) — never interactions, never reviews. Used to (a) supervise the
attribute bottleneck loss ``L_attr`` and (b) seed the K attribute-anchored intent centroids ``c_k``.

Beauty uses a versioned lexicon of seed phrases per facet
{skin concern, finish/effect, shade/color, brand-ingredient}. Non-beauty domains use the same
machinery with a domain lexicon, or fall back to unsupervised clustering (handled by the trainer).

Determinism: the lexicon is a plain dict committed in this file (and dumped to YAML by
``scripts/build_calm_weak_labels.py``). Soft labels are lexicon hit-counts normalised to a
distribution; ties / no-hits -> a small uniform mass so every item has a defined (possibly diffuse)
label. No randomness, no network, no model. See docs/method_calm_rec_spec.md section 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Beauty attribute lexicon. Facet -> {sub-label -> seed phrases}. Extend/version as needed; any
# change must be committed and noted in the runbook so labels reproduce exactly.
BEAUTY_LEXICON: dict[str, dict[str, list[str]]] = {
    "skin_concern": {
        "oily": ["oily", "oil control", "shine", "mattifying"],
        "dry": ["dry", "hydrating", "moisturizing", "moisture"],
        "sensitive": ["sensitive", "fragrance free", "fragrance-free", "gentle", "hypoallergenic"],
        "acne": ["acne", "blemish", "breakout", "salicylic", "blackhead"],
        "aging": ["anti aging", "anti-aging", "wrinkle", "firming", "retinol", "collagen"],
        "brightening": ["brightening", "dark spot", "vitamin c", "niacinamide", "radiance"],
    },
    "finish_effect": {
        "matte": ["matte", "matt"],
        "dewy": ["dewy", "glow", "luminous", "radiant"],
        "volumizing": ["volumizing", "volume", "plumping"],
        "lengthening": ["lengthening", "length"],
        "waterproof": ["waterproof", "long lasting", "long-lasting", "smudge proof"],
        "smoothing": ["smoothing", "smooth", "blurring"],
    },
    "shade_color": {
        "red": ["red", "crimson", "ruby", "scarlet"],
        "nude": ["nude", "beige", "natural"],
        "pink": ["pink", "rose", "blush", "berry"],
        "dark": ["black", "espresso", "dark brown", "deep"],
        "neutral": ["neutral", "taupe", "ivory", "sand"],
    },
    "brand_ingredient": {
        "retinol": ["retinol", "retinoid"],
        "hyaluronic": ["hyaluronic", "hyaluronic acid"],
        "niacinamide": ["niacinamide"],
        "vitamin_c": ["vitamin c", "ascorbic"],
        "spf": ["spf", "sunscreen", "uv"],
        "salicylic": ["salicylic", "bha"],
    },
}

FACETS: list[str] = list(BEAUTY_LEXICON.keys())


def _norm(text: str) -> str:
    return " " + " ".join(str(text).lower().replace("|", " ").replace("/", " ").split()) + " "


@dataclass
class AttributeLexicon:
    """A facet -> sub-label -> seed-phrases lexicon, with deterministic soft-labeling."""

    facets: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    @property
    def facet_names(self) -> list[str]:
        return list(self.facets.keys())

    @property
    def n_facets(self) -> int:
        return len(self.facets)

    def item_text(self, item_row: dict[str, Any]) -> str:
        return " ".join(
            str(item_row.get(f, "") or "")
            for f in ("title", "category", "brand", "description", "ingredients", "genres")
        )

    def facet_hits(self, item_row: dict[str, Any]) -> dict[str, float]:
        """Raw per-facet hit mass: count of (sub-label seed phrase) occurrences in the item text."""
        text = _norm(self.item_text(item_row))
        hits: dict[str, float] = {}
        for facet, sublabels in self.facets.items():
            mass = 0.0
            for _sub, phrases in sublabels.items():
                for ph in phrases:
                    needle = " " + ph.strip().lower() + " "
                    if needle in text:
                        mass += 1.0
            hits[facet] = mass
        return hits

    def soft_label(self, item_row: dict[str, Any], *, smoothing: float = 0.05) -> dict[str, float]:
        """Per-facet soft distribution (sums to 1) with uniform smoothing so it is always defined."""
        hits = self.facet_hits(item_row)
        n = max(1, len(hits))
        smoothed = {f: hits.get(f, 0.0) + smoothing for f in self.facets}
        total = sum(smoothed.values()) or float(n)
        return {f: v / total for f, v in smoothed.items()}

    def dominant_facet(self, item_row: dict[str, Any]) -> str | None:
        hits = self.facet_hits(item_row)
        if not hits or max(hits.values()) <= 0.0:
            return None
        return max(hits, key=lambda f: hits[f])


def default_beauty_lexicon() -> AttributeLexicon:
    return AttributeLexicon(facets={k: dict(v) for k, v in BEAUTY_LEXICON.items()})
