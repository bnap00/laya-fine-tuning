"""A tiny Pac-Man: one maze, pellets, two ghosts, no power pellets.

Small on purpose. The point is not the game, it's having a decision that is
asked hundreds of times per episode, has a cheap expert to copy, and gives a
score at the end that nobody can argue with.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

MAZE = """\
###############
#......#......#
#.##.#.#.#.##.#
#.............#
#.##.#####.##.#
#....#...#....#
####.#.#.#.####
#......G......#
#.##.##.##.##.#
#..#.......#..#
##.#.#####.#.##
#......P......#
#.####.#.####.#
#.............#
###############"""

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
PELLET_POINTS = 10
WIN_BONUS = 500
DEATH_PENALTY = -500
MAX_STEPS = 400

Pos = tuple[int, int]


def parse(maze: str = MAZE):
    walls, pellets, ghosts, pac = set(), set(), [], None
    rows = maze.splitlines()
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "#":
                walls.add((r, c))
            elif ch == ".":
                pellets.add((r, c))
            elif ch == "G":
                ghosts.append((r, c))
            elif ch == "P":
                pac = (r, c)
    return walls, pellets, ghosts, pac, len(rows), len(rows[0])


WALLS, PELLETS, GHOST_HOME, PAC_START, H, W = parse()


def step_pos(p: Pos, move: str) -> Pos:
    dr, dc = MOVES[move]
    return (p[0] + dr, p[1] + dc)


def legal_moves(p: Pos) -> list[str]:
    return [m for m in MOVES if step_pos(p, m) not in WALLS]


def bfs(start: Pos) -> dict[Pos, int]:
    """Maze distance from `start` to every reachable cell."""
    dist = {start: 0}
    q = deque([start])
    while q:
        p = q.popleft()
        for m in MOVES:
            n = step_pos(p, m)
            if n not in WALLS and n not in dist:
                dist[n] = dist[p] + 1
                q.append(n)
    return dist


# Distances never change (the maze is static), so compute them once.
DIST = {p: bfs(p) for p in [(r, c) for r in range(H) for c in range(W) if (r, c) not in WALLS]}


@dataclass
class Game:
    seed: int = 0
    pac: Pos = PAC_START
    ghosts: list[Pos] = field(default_factory=lambda: [GHOST_HOME, GHOST_HOME])
    pellets: set[Pos] = field(default_factory=lambda: set(PELLETS))
    score: int = 0
    steps: int = 0
    done: bool = False
    won: bool = False
    last_move: str | None = None
    # Each ghost has a "personality": 0 chases Pac-Man, 1 wanders and only sometimes chases.
    ghost_chase: tuple[float, float] = (0.6, 0.3)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        # Ghosts start spread out so the opening is not decided by the first move.
        self.ghosts = [(1, 1), (1, W - 2)]
        self.prev_ghosts = list(self.ghosts)

    def copy(self) -> "Game":
        g = Game.__new__(Game)
        g.__dict__ = {k: (set(v) if isinstance(v, set) else list(v) if isinstance(v, list) else v)
                      for k, v in self.__dict__.items() if k != "rng"}
        g.rng = random.Random()
        g.rng.setstate(self.rng.getstate())  # copying must not consume randomness from the real game
        return g

    def _move_ghost(self, i: int, g: Pos) -> Pos:
        options = legal_moves(g)
        prev = self.prev_ghosts[i]
        # Ghosts don't reverse unless they're in a dead end, like the arcade ones.
        forward = [m for m in options if step_pos(g, m) != prev] or options
        if self.rng.random() < self.ghost_chase[i]:
            return min((step_pos(g, m) for m in forward), key=lambda n: DIST[n].get(self.pac, 99))
        return step_pos(g, self.rng.choice(forward))

    def step(self, move: str) -> None:
        if self.done:
            return
        if move not in legal_moves(self.pac):
            raise ValueError(f"illegal move {move!r} at {self.pac}")
        old_pac = self.pac
        self.pac = step_pos(self.pac, move)
        self.last_move = move
        self.steps += 1
        if self.pac in self.pellets:
            self.pellets.discard(self.pac)
            self.score += PELLET_POINTS
        # Caught by moving into a ghost.
        if self.pac in self.ghosts:
            return self._die()
        new_ghosts = [self._move_ghost(i, g) for i, g in enumerate(self.ghosts)]
        # Caught by a ghost moving onto Pac-Man, or by swapping places with one.
        for g_old, g_new in zip(self.ghosts, new_ghosts):
            if g_new == self.pac or (g_new == old_pac and g_old == self.pac):
                self.prev_ghosts, self.ghosts = self.ghosts, new_ghosts
                return self._die()
        self.prev_ghosts, self.ghosts = self.ghosts, new_ghosts
        if not self.pellets:
            self.done, self.won = True, True
            self.score += WIN_BONUS
        elif self.steps >= MAX_STEPS:
            self.done = True

    def _die(self) -> None:
        self.done = True
        self.score += DEATH_PENALTY

    def render(self) -> str:
        grid = [["#" if (r, c) in WALLS else "." if (r, c) in self.pellets else " "
                 for c in range(W)] for r in range(H)]
        for g in self.ghosts:
            grid[g[0]][g[1]] = "G"
        grid[self.pac[0]][self.pac[1]] = "P"
        return "\n".join("".join(row) for row in grid)


def nearest(src: Pos, targets) -> int | None:
    d = DIST[src]
    return min((d[t] for t in targets if t in d), default=None)
