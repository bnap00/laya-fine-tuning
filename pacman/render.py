"""Turn a list of game frames into an animated GIF for the blog."""
from pathlib import Path

from PIL import Image, ImageDraw

from game import H, W, WALLS

CELL = 24
BG, WALL, PELLET, PAC, TEXT = (12, 12, 20), (40, 70, 200), (250, 220, 180), (255, 214, 0), (230, 230, 230)
GHOSTS = [(255, 70, 70), (255, 150, 220)]


def frame(game, title: str) -> Image.Image:
    img = Image.new("RGB", (W * CELL, H * CELL + 28), BG)
    d = ImageDraw.Draw(img)
    for r in range(H):
        for c in range(W):
            x, y = c * CELL, r * CELL
            if (r, c) in WALLS:
                d.rounded_rectangle([x + 2, y + 2, x + CELL - 2, y + CELL - 2], 5, fill=WALL)
            elif (r, c) in game.pellets:
                d.ellipse([x + 10, y + 10, x + 14, y + 14], fill=PELLET)
    for i, (r, c) in enumerate(game.ghosts):
        x, y = c * CELL, r * CELL
        d.rounded_rectangle([x + 3, y + 3, x + CELL - 3, y + CELL - 2], 8, fill=GHOSTS[i % 2])
        d.ellipse([x + 7, y + 8, x + 11, y + 12], fill=(255, 255, 255))
        d.ellipse([x + 13, y + 8, x + 17, y + 12], fill=(255, 255, 255))
    r, c = game.pac
    x, y = c * CELL, r * CELL
    mouth = {"right": 0, "down": 90, "left": 180, "up": 270}.get(game.last_move or "right", 0)
    d.pieslice([x + 3, y + 3, x + CELL - 3, y + CELL - 3], mouth + 35, mouth + 325, fill=PAC)
    status = "WON" if game.won else "CAUGHT" if game.done and game.steps < 400 else ""
    d.text((6, H * CELL + 8), f"{title}   score {game.score}   step {game.steps}   {status}", fill=TEXT)
    return img


def save_gif(frames, path, title: str = "", ms: int = 90) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [frame(g, title) for g in frames]
    imgs += [imgs[-1]] * 15  # hold the last frame so you can see how it ended
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=True)


def save_side_by_side(runs, path, ms: int = 90) -> None:
    """runs: [(title, frames), ...] played on the same seed, shown next to each other."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(len(f) for _, f in runs) + 15
    strips = []
    for i in range(n):
        tiles = [frame(frames[min(i, len(frames) - 1)], title) for title, frames in runs]
        strip = Image.new("RGB", (sum(t.width for t in tiles) + 12 * (len(tiles) - 1), tiles[0].height), BG)
        x = 0
        for t in tiles:
            strip.paste(t, (x, 0))
            x += t.width + 12
        strips.append(strip)
    strips[0].save(path, save_all=True, append_images=strips[1:], duration=ms, loop=0, optimize=True)
