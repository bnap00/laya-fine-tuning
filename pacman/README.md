# Laya plays Pac-Man

![untrained, fine-tuned and DAgger-trained Laya playing the same game](results/side-by-side.gif)

A tiny Pac-Man (one 15x15 maze, 111 pellets, two ghosts) where every move is a Laya
`choice` question. The options are the legal moves, each described the way you'd glance at
it: *"eats a pellet, closest ghost 4 steps away"*. The state is a status line plus the maze
in ASCII.

```bash
python make_dataset.py --games 300 --out data/train.jsonl          # teacher-labelled states
python make_dataset.py --games 40 --first-seed 900000 --out data/test.jsonl
python ../finetune.py data/train.jsonl ../checkpoints/pacman-v1 --epochs 2 --eval-data data/test.jsonl

python play.py --games 200 --players random teacher english ../checkpoints/pacman-v1

# DAgger: v1 drives, the teacher labels where it ends up, retrain from v1
python make_dataset.py --games 150 --first-seed 200000 --explore 0.05 \
       --student ../checkpoints/pacman-v1 --out data/dagger.jsonl
# (mix in some of train.jsonl, then)
python ../finetune.py data/v2.jsonl ../checkpoints/pacman-v2 --base ../checkpoints/pacman-v1 --epochs 1

python compare_gif.py --seed 500022 --players english ../checkpoints/pacman-v1 ../checkpoints/pacman-v2 \
       --labels untrained fine-tuned "+ DAgger" --out results/side-by-side.gif
```

| file | |
| --- | --- |
| `game.py` | the game: maze, ghosts (one chases 60% of the time, one 30%), scoring |
| `agents.py` | the question Laya is asked, the teacher, a random player, the Laya player |
| `make_dataset.py` | plays games and writes each state with the teacher's probabilities as `gold` |
| `play.py` | the scoreboard; seeds start at 500000 so no game was seen in training |
| `render.py`, `compare_gif.py` | GIFs |

## The teacher is free

The teacher scores each legal move with a few lines of distance arithmetic: dying is -100,
eating is +6, every step to the nearest pellet is -1, being near a ghost costs `12 / distance`,
a dead end near a ghost costs 10. A softmax over those scores is the training target, so Laya
learns *how much* the teacher prefers each move, not just which one.

It wins 39% of games. That's the ceiling we're copying, and v2 reaches it.

## Why DAgger

v1 only ever saw states the teacher gets itself into. When v1 makes a mistake it lands
somewhere the teacher never would, and it has never been shown what to do there. DAgger
fixes that by letting v1 drive and asking the teacher what it *should* have done at each
step. One round: 63 → 87 wins out of 200 on the RTX 3050, 63 → 77 when the whole pipeline
was rerun on an M4 Pro. Either way, level with the teacher's 78.

Numbers are in [`../RESULTS.md`](../RESULTS.md).
