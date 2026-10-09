# Fine-tuning Laya on a 6 GB GPU

Two worked examples of fine-tuning [Laya](https://github.com/NandhaKishorM/laya), the
open-source (Apache 2.0) decision model, for a specific job, on a desktop RTX 3050
with 6 GB of memory, no cloud GPU.

| example | the decision | the teacher | what we score on |
| --- | --- | --- | --- |
| [`pacman/`](pacman/) | which way Pac-Man moves, every tick | a hand-written expert, free | games won and pellets eaten |
| [`pigeonhole/`](pigeonhole/) | the five [Pigeonhole](https://github.com/bnap00/pigeonhole) template classifiers | Jev on OpenRouter, ~$0.07 | the templates' own 50 test cases |

![untrained, fine-tuned and DAgger-trained Laya playing the same game](pacman/results/side-by-side.gif)

| | untrained Laya | fine-tuned Laya | the reference |
| --- | --- | --- | --- |
| Pac-Man, games won of 200 | 0 | **87** (after one DAgger round) | teacher: 78 |
| Pigeonhole, tests passed of 50 | 31 | **44** (ties Clef) | Jev: 47 |

Full numbers, costs and timings are in [`RESULTS.md`](RESULTS.md); the write-up is on
[bnap.dev](https://bnap.dev).

## The recipe

Both examples are the same four steps:

1. **Inputs.** Get states the model will be asked about. Play games; or have a cheap chat
   model write realistic inputs from the classifier's description (never from its tests).
2. **Teacher labels.** Ask something better than Laya for a *probability* over every option,
   not just its answer. Laya's trainer is built to copy a distribution.
3. **Train.** [`finetune.py`](finetune.py) wraps `laya.train.finetune`: it trains the decision
   head and the top N encoder layers, fits calibration temperatures on a held-out slice,
   and saves a checkpoint `laya.load` opens.
4. **Score it on something it has never seen.** Unseen game seeds; the template tests.

## Why "top N layers"

A full fine-tune of the 421M English checkpoint needs about 7 GB for weights, gradients and
AdamW state alone. Freezing the embeddings and the bottom 22 of 28 encoder layers leaves 100M
trainable parameters and peaks at about 3.8 GB. `laya.train` skips parameters that don't
require gradients, so this is a 20-line wrapper, not a fork.

```bash
python finetune.py data.jsonl checkpoints/mine --top-layers 6 --epochs 2
```

| flag | default | |
| --- | --- | --- |
| `--base` | `english` | `english`, `multilingual`, `typed-decisions`, a Hub id or a directory |
| `--top-layers` | `6` | encoder layers to train; `-1` trains everything (needs a bigger card) |
| `--epochs` | `3` | |
| `--micro-batch`, `--grad-accum` | `8`, `4` | effective batch 32 |
| `--loss` | `rlcd` | Laya's proper-scoring-rule objective, or `soft-ce` |
| `--eval-data` | | a held-out JSONL to score before and after |

## Setup

```bash
uv venv --python 3.12 .venv
# NVIDIA driver older than CUDA 13 (check `nvidia-smi`)? Pick a matching torch build first:
uv pip install --python .venv/bin/python torch --torch-backend=cu126
uv pip install --python .venv/bin/python -r requirements.txt
```

The Pigeonhole example needs an `OPENROUTER_API_KEY` in `.env` (for the input generator
and Jev). The Pac-Man example needs nothing but the GPU.

## License

Apache 2.0. The Pigeonhole templates in `pigeonhole/templates/` are copied from
[bnap00/pigeonhole](https://github.com/bnap00/pigeonhole), also Apache 2.0.
