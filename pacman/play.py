"""Play a batch of seeded games with each player and print a scoreboard.

    python play.py --games 50 --players random teacher english ./checkpoints/pacman
    python play.py --games 1 --seed 7 --players ./checkpoints/pacman --gif out/pacman.gif

The seeds start at 500000, far away from the ones the dataset was built from,
so no game here was ever seen in training.
"""
import argparse
import json
import statistics as st
from pathlib import Path

from agents import LayaPlayer, RandomPlayer, Teacher, play, teacher_probs
from game import PELLETS


def make_player(spec: str):
    if spec == "random":
        return RandomPlayer(seed=1)
    if spec == "teacher":
        return Teacher()
    return LayaPlayer(spec, name=Path(spec).name if "/" in spec else spec)


class Agreement:
    """Wraps a player and counts how often its move is also the teacher's favourite."""

    def __init__(self, player):
        self.player, self.name, self.hits, self.n = player, player.name, 0, 0

    def act(self, game):
        move = self.player.act(game)
        p = teacher_probs(game)
        self.hits += move == max(p, key=p.get)
        self.n += 1
        return move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", nargs="+", default=["random", "teacher"])
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=500000)
    ap.add_argument("--gif", help="record the first game of each Laya player to this path")
    ap.add_argument("--json", help="write the scoreboard here too")
    args = ap.parse_args()

    board = []
    for spec in args.players:
        player = Agreement(make_player(spec))
        games = []
        for i in range(args.games):
            game, frames = play(player, args.seed + i, record=bool(args.gif) and i == 0)
            games.append(game)
            if frames:
                from render import save_gif
                path = Path(args.gif)
                path = path.with_name(f"{path.stem}-{player.name}{path.suffix}") if len(args.players) > 1 else path
                save_gif(frames, path, title=player.name)
        lat = getattr(player.player, "latencies", [])
        row = {
            "player": player.name,
            "games": len(games),
            "won": sum(g.won for g in games),
            "pellets": round(st.mean(len(PELLETS) - len(g.pellets) for g in games), 1),
            "steps": round(st.mean(g.steps for g in games), 1),
            "score": round(st.mean(g.score for g in games), 1),
            "agrees_with_teacher": round(player.hits / max(1, player.n), 3),
            "ms_per_move": round(1000 * st.median(lat), 1) if lat else None,
        }
        board.append(row)
        print(json.dumps(row), flush=True)

    print("\n| player | won | pellets eaten (of %d) | score | agrees with teacher | ms/move |" % len(PELLETS))
    print("| --- | --- | --- | --- | --- | --- |")
    for r in board:
        print(f"| {r['player']} | {r['won']} / {r['games']} | {r['pellets']} | {r['score']} | "
              f"{r['agrees_with_teacher']:.0%} | {r['ms_per_move'] or '-'} |")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(board, indent=2))


if __name__ == "__main__":
    main()
