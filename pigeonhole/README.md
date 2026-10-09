# Distilling Jev into Laya on the Pigeonhole templates

[Pigeonhole](https://github.com/bnap00/pigeonhole) turns a plain-English description into a
classifier and ships each one with test cases. Out of the box, Laya passed 31 of the 50 tests
in the five templates; Jev, the hosted decision model, passed 47. Fine-tuned, Laya passes 44,
level with Cloudflare's Clef and at a fifth of Jev's latency. This example closes most of that gap by teaching Laya to copy Jev's probabilities on inputs nobody wrote by hand.

```bash
set -a; . ../.env; set +a                       # OPENROUTER_API_KEY

python generate.py --per-template 400            # ~2,000 synthetic inputs, ~$0.10
python label.py data/inputs.jsonl data/train.jsonl   # Jev's distributions, ~$0.07
python ../finetune.py data/train.jsonl ../checkpoints/pigeonhole --epochs 3   # ~28 min on an RTX 3050

python harness.py jev-latest openai/gpt-6-luna-decisions cloudflare/clef english ../checkpoints/pigeonhole -v
```

## Files

| file | what it does |
| --- | --- |
| `harness.py` | the parts of Pigeonhole a fair score needs: its wire format (Jev gets structured criteria, Laya gets them flattened to one line), `when` gates, the templates' rules, and the test matcher |
| `generate.py` | a cheap chat model writes inputs from each template's description and questions, in ten styles; never from its tests, and near-duplicates of a test are dropped |
| `label.py` | asks Jev every question about every input and keeps the full distribution as `gold`; gated questions that Jev's answers switch off are left out |
| `templates/` | the five templates, copied from Pigeonhole at `19366f8` |

## Things that matter

- **Train on the text Laya will see.** Pigeonhole flattens each criterion into one line before
  sending it to Laya (`criterion_text`), so the training rows use the same flattening. A
  checkpoint trained on a different rendering of the same question is learning a different question.
- **Keep the tests out of the generator's prompt.** They are the only honest score you have.
- **One checkpoint, five classifiers.** All templates train together; the question text tells
  the model which job it is doing.
