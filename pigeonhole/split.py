"""Hold out a slice of teacher-labelled rows, per template, before training.

    python split.py data/claude/labelled.jsonl --heldout 0.1

Writes train.jsonl and heldout.jsonl beside the input. The held-out rows are
labelled by the same teacher, so scoring on them measures agreement with the
teacher, not correctness; the Pigeonhole tests stay the real score.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def split(rows: list[dict], heldout: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_template = defaultdict(list)
    for r in rows:
        by_template[r["template"]].append(r)
    rng = random.Random(seed)
    train, held = [], []
    for name in sorted(by_template):
        group = by_template[name][:]
        rng.shuffle(group)
        k = max(1, round(len(group) * heldout))
        held += group[:k]
        train += group[k:]
    return train, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labelled")
    ap.add_argument("--heldout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = Path(args.labelled)
    rows = [json.loads(line) for line in src.open()]
    train, held = split(rows, args.heldout, args.seed)
    for name, part in (("train.jsonl", train), ("heldout.jsonl", held)):
        with (src.parent / name).open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in part)
    print(f"{len(train)} train, {len(held)} held out -> {src.parent}")


if __name__ == "__main__":
    main()
