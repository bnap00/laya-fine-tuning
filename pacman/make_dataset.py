"""Play games, and write every state the teacher was asked about as a training row.

    python make_dataset.py --games 400 --out data/train.jsonl
    python make_dataset.py --games 60 --first-seed 900000 --out data/test.jsonl

Rows are in the JSONL shape `laya-train` reads: the state, the question, and the
teacher's probabilities under `gold`. Training on the teacher's full preference,
not just its top move, is what Laya's trainer is built for.

With `--student` the student checkpoint drives the games and the teacher only
labels the states it lands in (DAgger). That's how you teach a model to recover
from its own mistakes, which a teacher's clean games never show it.
"""
import argparse
import json
import random
from pathlib import Path

from agents import LayaPlayer, Teacher, question, state_text, teacher_probs
from game import Game, legal_moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--explore", type=float, default=0.2,
                    help="chance of a random move, so the data includes bad spots to get out of")
    ap.add_argument("--student", help="checkpoint that drives the games (DAgger); the teacher only labels")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.first_seed)
    teacher = Teacher()
    driver = LayaPlayer(args.student) if args.student else teacher
    seen, rows, outcomes = set(), [], []
    for seed in range(args.first_seed, args.first_seed + args.games):
        game = Game(seed=seed)
        while not game.done:
            state, q = state_text(game), question(game)
            if len(q["move"]["criteria"]) > 1 and state not in seen:
                seen.add(state)
                rows.append({"state": state, "questions": q,
                             "gold": {"move": {"probabilities": teacher_probs(game)}}})
            if rng.random() < args.explore:
                game.step(rng.choice(legal_moves(game.pac)))
            else:
                game.step(driver.act(game))
        outcomes.append(game.won)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} states from {args.games} games ({sum(outcomes)} won) -> {out}")


if __name__ == "__main__":
    main()
