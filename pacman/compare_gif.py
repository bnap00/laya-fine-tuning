"""Record several players on the same seed, side by side, as one GIF.

    python compare_gif.py --seed 500003 --players english ../checkpoints/pacman-v1 --out results/side-by-side.gif
"""
import argparse
from pathlib import Path

from play import make_player
from agents import play
from render import save_side_by_side

ap = argparse.ArgumentParser()
ap.add_argument("--players", nargs="+", required=True)
ap.add_argument("--labels", nargs="+")
ap.add_argument("--seed", type=int, default=500000)
ap.add_argument("--out", required=True)
args = ap.parse_args()

runs = []
for i, spec in enumerate(args.players):
    game, frames = play(make_player(spec), args.seed, record=True)
    label = args.labels[i] if args.labels else Path(spec).name
    print(f"{label}: score {game.score}, {'won' if game.won else 'lost'} after {game.steps} steps")
    runs.append((label, frames))
save_side_by_side(runs, args.out)
print("->", args.out)
