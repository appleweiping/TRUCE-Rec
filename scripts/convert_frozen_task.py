#!/usr/bin/env python3
"""Convert the frozen same-candidate task files into TRUCE-Rec's runner format.

The authoritative evaluation panels live in the shared frozen protocol exports
(`.../outputs/baselines/external_tasks/<domain>_..._{test,valid}_same_candidate/
ranking_test.jsonl`) — the SAME candidate sets the 8 frozen official baselines
were scored on. (Verified 2026-06-12: the uncertainty project's panels share
users/positives but NOT candidate sets — do not evaluate on those.)

Emits `data/processed/<out-variant>/`:
  examples.jsonl  - split=test/valid rows carry the frozen 101-candidate panels
                    (history = item-id list, chronological); split=train rows are
                    leave-one-out next-item transitions inside the train prefix
                    (history excludes the valid/test targets).
  items.csv       - union of panel candidates + history items; title from the
                    task files, enriched (category/description) from an optional
                    metadata CSV keyed by item_id.

Leakage notes: train rows never contain valid/test targets as targets, and the
runner's Stage-A stats additionally exclude ALL held-out targets via
build_train_only_stats(held_out_targets=...).

Usage (server, beauty):
  python scripts/convert_frozen_task.py \
    --test  ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/beauty_supplementary_smallerN_100neg_test_same_candidate/ranking_test.jsonl \
    --valid ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/beauty_supplementary_smallerN_100neg_valid_same_candidate/ranking_test.jsonl \
    --items-meta ~/projects/uncertainty-llm4rec/data/processed/amazon_beauty/items.csv \
    --domain beauty --out data/processed/frozen_week8_beauty
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


def listish(x):
    if isinstance(x, list):
        return x
    if not x:
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def read_panels(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        cands = [str(c) for c in listish(d.get("candidate_item_ids"))]
        pos = d.get("positive_item_index")
        if not cands or pos is None or not (0 <= int(pos) < len(cands)):
            continue
        rows.append(
            {
                "user_id": str(d.get("user_id")),
                "source_event_id": str(d.get("source_event_id", "")),
                "history_ids": [str(h) for h in listish(d.get("history_item_ids"))],
                "history_titles": [str(t) for t in listish(d.get("history"))],
                "cand_ids": cands,
                "cand_titles": [str(t) for t in listish(d.get("candidate_titles"))],
                "cand_texts": [str(t) for t in listish(d.get("candidate_texts"))],
                "target": str(cands[int(pos)]),
                "target_title": str(d.get("positive_item_title", "")),
                "timestamp": d.get("timestamp"),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--valid", default=None)
    ap.add_argument("--items-meta", default=None,
                    help="optional CSV; supports both item_id,title,categories,description and the "
                         "frozen-export item_id,candidate_title,candidate_text,popularity_group")
    ap.add_argument("--train-interactions", default=None,
                    help="optional CSV user_id,item_id,timestamp[,sequence_index] — the canonical "
                         "uncapped train stream from the frozen export (preferred over deriving "
                         "transitions from the capped test histories)")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    test = read_panels(Path(args.test))
    valid = read_panels(Path(args.valid)) if args.valid else []
    print(f"frozen panels: test={len(test)} valid={len(valid)}")

    meta: dict[str, dict[str, str]] = {}
    if args.items_meta and Path(args.items_meta).exists():
        with open(args.items_meta, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                meta[str(r.get("item_id"))] = r
        print(f"items-meta rows: {len(meta)}")

    train_seqs: dict[str, list[str]] = {}
    if args.train_interactions and Path(args.train_interactions).exists():
        by_user: dict[str, list[tuple[float, int, str]]] = {}
        with open(args.train_interactions, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                ts = float(r.get("timestamp") or -1)
                si = int(r.get("sequence_index") or 0)
                by_user.setdefault(str(r["user_id"]), []).append((ts, si, str(r["item_id"])))
        for u, evs in by_user.items():
            train_seqs[u] = [i for _, _, i in sorted(evs)]
        print(f"train interactions: {sum(len(v) for v in train_seqs.values())} over {len(train_seqs)} users")

    # ---- items.csv: union of everything the panels mention ----
    titles: dict[str, str] = {}
    for row in test + valid:
        for cid, t in zip(row["cand_ids"], row["cand_titles"]):
            titles.setdefault(cid, t)
        for hid, t in zip(row["history_ids"], row["history_titles"]):
            titles.setdefault(hid, t)
        if row["target_title"]:
            titles.setdefault(row["target"], row["target_title"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["item_id", "title", "description", "category", "brand", "domain", "raw_text"]
        )
        w.writeheader()
        all_ids = set(titles) | set(train_seqs and {i for s in train_seqs.values() for i in s} or set())
        for iid in sorted(all_ids):
            m = meta.get(iid, {})
            text = m.get("candidate_text", "")
            desc = m.get("description", "")
            if not desc and "Description:" in text:
                desc = text.split("Description:", 1)[1].strip()
            w.writerow(
                {
                    "item_id": iid,
                    "title": m.get("title") or m.get("candidate_title") or titles.get(iid, iid),
                    "description": desc,
                    "category": m.get("categories", m.get("category", "")),
                    "brand": m.get("brand", ""),
                    "domain": args.domain,
                    "raw_text": text,
                }
            )

    # ---- examples.jsonl ----
    valid_targets = {(r["user_id"], r["target"]) for r in valid}
    held_targets = {r["target"] for r in valid} | {r["target"] for r in test}
    n_train = 0
    with (out / "examples.jsonl").open("w", encoding="utf-8") as fh:
        for split, rows in (("test", test), ("valid", valid)):
            for r in rows:
                fh.write(
                    json.dumps(
                        {
                            "example_id": r["source_event_id"] or f"{r['user_id']}::{split}",
                            "user_id": r["user_id"],
                            "domain": args.domain,
                            "split": split,
                            "history": r["history_ids"],
                            "history_titles": r["history_titles"],
                            "candidates": r["cand_ids"],
                            "target": r["target"],
                        }
                    )
                    + "\n"
                )
        # train transitions: prefer the canonical uncapped train stream; fall back
        # to deriving from the (length-capped) test histories. Either way, drop
        # the user's valid target from the sequence (supervised by the valid
        # panel instead) and never supervise on any held-out target.
        if train_seqs:
            sources = [(u, seq) for u, seq in sorted(train_seqs.items())]
        else:
            sources = [(r["user_id"], r["history_ids"]) for r in test]
        for user_id, raw_seq in sources:
            seq = [h for h in raw_seq if (user_id, h) not in valid_targets]
            for j in range(1, len(seq)):
                if seq[j] in held_targets:
                    continue  # never supervise on a held-out target
                fh.write(
                    json.dumps(
                        {
                            "example_id": f"{user_id}::train{j}",
                            "user_id": user_id,
                            "domain": args.domain,
                            "split": "train",
                            "history": seq[:j],
                            "candidates": [],
                            "target": seq[j],
                        }
                    )
                    + "\n"
                )
                n_train += 1

    manifest = {
        "builder": "convert_frozen_task.py",
        "domain": args.domain,
        "test_source": str(Path(args.test).resolve()),
        "valid_source": str(Path(args.valid).resolve()) if args.valid else None,
        "items_meta": str(Path(args.items_meta).resolve()) if args.items_meta else None,
        "n_test": len(test),
        "n_valid": len(valid),
        "n_train_transitions": n_train,
        "n_items": len(titles),
        "note": "frozen same-candidate panels; baselines scored on these exact sets",
    }
    (out / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
