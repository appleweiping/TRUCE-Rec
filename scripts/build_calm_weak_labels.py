"""Build CALM-Rec attribute weak-labels for a processed dataset (item text only, no leakage).

Reads <processed_dir>/items.csv, applies the attribute lexicon, and writes:
  <out>/calm_weak_labels.jsonl   # one row per item: item_id + per-facet soft distribution
  <out>/calm_lexicon.yaml        # the exact lexicon used (for reproduction)

No interactions, no reviews, no paid model, no network. Deterministic. See
docs/method_calm_rec_spec.md section 6 and docs/CALM_REC_RUNBOOK.md.

Usage:
  py -3 scripts/build_calm_weak_labels.py --processed-dir data/processed/<domain> --out outputs/calm/<domain>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.methods.calm_weak_labels import default_beauty_lexicon  # noqa: E402


def _read_items(processed_dir: Path) -> list[dict[str, str]]:
    path = processed_dir / "items.csv"
    if not path.exists():
        raise SystemExit(f"items.csv not found in {processed_dir}")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoothing", type=float, default=0.05)
    args = parser.parse_args()

    lexicon = default_beauty_lexicon()
    items = _read_items(Path(args.processed_dir))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in items:
        soft = lexicon.soft_label(item, smoothing=args.smoothing)
        dominant = lexicon.dominant_facet(item)
        rows.append({"item_id": str(item.get("item_id")), "soft_label": soft, "dominant_facet": dominant})

    with (out_dir / "calm_weak_labels.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # dump the lexicon for reproducibility (plain nested structure)
    with (out_dir / "calm_lexicon.yaml").open("w", encoding="utf-8") as fh:
        for facet, sublabels in lexicon.facets.items():
            fh.write(f"{facet}:\n")
            for sub, phrases in sublabels.items():
                fh.write(f"  {sub}: [{', '.join(repr(p) for p in phrases)}]\n")

    labeled = sum(1 for r in rows if r["dominant_facet"])
    print(json.dumps({
        "items": len(rows),
        "with_dominant_facet": labeled,
        "facets": lexicon.facet_names,
        "out_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
