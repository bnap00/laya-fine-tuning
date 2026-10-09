"""Who plays: a random player, a hand-written expert, and Laya.

The expert is the teacher. It is twenty lines of distance arithmetic, which is
exactly why it is useful: it costs nothing to ask it about a million states, and
it gives a preference over every move, not just the one it picked.
"""
from __future__ import annotations

import math
import random
import time

from game import DIST, MOVES, Game, legal_moves, nearest, step_pos

# The decision Laya is asked every tick. The options change with the maze
# position, so the question is rebuilt per state.
INSTRUCTIONS = (
    "You are Pac-Man. Pick the move that keeps eating pellets without getting caught. "
    "A ghost 1 or 2 steps away from where you land can catch you, so never move next to one; "
    "otherwise head for the closest pellet."
)


def move_facts(game: Game, move: str) -> dict:
    n = step_pos(game.pac, move)
    return {
        "eats": n in game.pellets,
        "pellet": nearest(n, game.pellets) if game.pellets else None,
        "ghost": min(DIST[n][g] for g in game.ghosts),
        "exits": len(legal_moves(n)) - 1,  # ways out of the landing cell besides coming back
    }


def describe_move(f: dict) -> str:
    parts = ["eats a pellet" if f["eats"] else f"closest pellet {f['pellet']} steps away"
             if f["pellet"] is not None else "no pellets left"]
    parts.append(f"closest ghost {f['ghost']} steps away")
    if f["exits"] == 0:
        parts.append("dead end")
    return ", ".join(parts)


def question(game: Game) -> dict:
    return {
        "move": {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": {m: describe_move(move_facts(game, m)) for m in legal_moves(game.pac)},
        }
    }


def state_text(game: Game) -> str:
    ghosts = sorted(DIST[game.pac][g] for g in game.ghosts)
    return (
        f"Pellets left: {len(game.pellets)}. Ghosts are {ghosts[0]} and {ghosts[1]} steps away. "
        f"Last move: {game.last_move or 'none'}.\n"
        f"Maze (P is you, G is a ghost, . is a pellet, # is a wall):\n{game.render()}"
    )


# ---------------------------------------------------------------- teacher

def move_value(game: Game, move: str) -> float:
    f = move_facts(game, move)
    g = f["ghost"]
    if g <= 1:
        return -100.0 + g  # walking into a ghost, or next to one that moves next
    v = 0.0
    v += 6.0 if f["eats"] else -float(f["pellet"] or 0)
    if g <= 4:
        v -= 12.0 / g  # stay away from ghosts, harder the closer they are
        if f["exits"] == 0:
            v -= 10.0  # don't get cornered with a ghost nearby
    if game.last_move and MOVES[move] == tuple(-x for x in MOVES[game.last_move]) and g > 4:
        v -= 1.5  # don't dither back and forth when nothing is chasing
    return v


def teacher_probs(game: Game, temperature: float = 1.5) -> dict[str, float]:
    """A soft preference over legal moves: what Laya is trained to copy."""
    vals = {m: move_value(game, m) for m in legal_moves(game.pac)}
    top = max(vals.values())
    exps = {m: math.exp((v - top) / temperature) for m, v in vals.items()}
    z = sum(exps.values())
    return {m: e / z for m, e in exps.items()}


class Teacher:
    name = "teacher"

    def act(self, game: Game) -> str:
        p = teacher_probs(game)
        return max(p, key=p.get)


class RandomPlayer:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, game: Game) -> str:
        return self.rng.choice(legal_moves(game.pac))


class LayaPlayer:
    """Asks a Laya checkpoint for the move, once per tick."""

    def __init__(self, checkpoint: str, name: str | None = None, device: str | None = None):
        import laya

        self.agent = laya.load(checkpoint, **({"device": device} if device else {}))
        self.name = name or checkpoint
        self.latencies: list[float] = []

    def probs(self, game: Game) -> dict[str, float]:
        t = time.perf_counter()
        out = self.agent.predict(state_text(game), question(game))
        self.latencies.append(time.perf_counter() - t)
        ans = out["answers"]["move"]
        return ans.get("probabilities") or {ans["choice"]: 1.0}

    def act(self, game: Game) -> str:
        p = self.probs(game)
        return max(p, key=p.get)


def play(player, seed: int, record: bool = False):
    game = Game(seed=seed)
    frames = [game.copy()] if record else None
    while not game.done:
        game.step(player.act(game))
        if record:
            frames.append(game.copy())
    return game, frames
