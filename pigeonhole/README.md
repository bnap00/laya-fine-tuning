# Distilling Jev into Laya on the Pigeonhole templates

[Pigeonhole](https://github.com/bnap00/pigeonhole) turns a plain-English description into a
classifier and ships each one with test cases. Out of the box, Laya passed 31 of the 50 tests
in the five templates; Jev, the hosted decision model, passed 47. Fine-tuned, Laya passes 44,
level with Cloudflare's Clef and at a fifth of Jev's latency. This example closes most of that gap by teaching Laya to copy Jev's probabilities on inputs nobody wrote by hand.

```bash
set -a; . ../.env; set +a                       # OPENROUTER_API_KEY

python generate.py --per-template 400            # ~2,000 synthetic inputs, ~$0.10
python label.py data/inputs.jsonl data/train.jsonl   # Jev's distributions, ~$0.07
python ../finetune.py data/train.jsonl ../checkpoints/pigeonhole --epochs 3   # ~28 min on an RTX 3050, ~26 on an M4 Pro

python harness.py jev-latest openai/gpt-6-luna-decisions cloudflare/clef english ../checkpoints/pigeonhole -v
```

## Files

| file | what it does |
| --- | --- |
| `harness.py` | the parts of Pigeonhole a fair score needs: its wire format (Jev gets structured criteria, Laya gets them flattened to one line), `when` gates, the templates' rules, and the test matcher |
| `generate.py` | a cheap chat model writes inputs from each template's description and questions, in ten styles; never from its tests, and near-duplicates of a test are dropped |
| `label.py` | asks Jev every question about every input and keeps the full distribution as `gold`; gated questions that Jev's answers switch off are left out |
| `templates/` | the five templates, copied from Pigeonhole at `19366f8` |
| `teach.py` | the no-API-key route: Claude Code or Codex writes inputs and labels them in one pass, on your subscription |
| `agent_rows.py` | checks an agent's reply against the template and builds rows in the same shape `label.py` writes |
| `split.py` | holds out a slice of teacher-labelled rows per template, for `--eval-data` |

## Things that matter

- **Train on the text Laya will see.** Pigeonhole flattens each criterion into one line before
  sending it to Laya (`criterion_text`), so the training rows use the same flattening. A
  checkpoint trained on a different rendering of the same question is learning a different question.
- **Keep the tests out of the generator's prompt.** They are the only honest score you have.
- **One checkpoint, five classifiers.** All templates train together; the question text tells
  the model which job it is doing.

## Use your Claude Code or Codex subscription as the teacher

No OpenRouter key, no per-call bill: the coding agent you already pay for writes the inputs
*and* labels them, in one pass.

> Generating training data with Claude Code or Codex is a great way to learn and to get a
> first model working. For a real deployment, you will be much more accurate training on real
> data: a golden dataset of actual inputs with labels a person has checked.

```bash
python teach.py --agent claude --model claude-opus-5-5   # or: --agent codex --model gpt-6.1-sol
python split.py data/claude/labelled.jsonl                # 90% train, 10% held out per template
python ../finetune.py data/claude/train.jsonl ../checkpoints/pigeonhole-claude --epochs 3 \
       --eval-data data/claude/heldout.jsonl
python harness.py ../checkpoints/pigeonhole-claude -v
```

Each call asks for 25 inputs of one template in one style, with the agent's probability over
every option of every question. The agent sees the template's description and questions,
never its tests, and `teach.py` drops anything too close to a test anyway. It calls the CLIs
headless with a one-line system prompt and no tools, so each call carries ~1.6k input tokens
for Claude Code instead of ~30k. Usage per call lands in `usage.jsonl`, totals in
`summary.json`.

The held-out slice is labelled by the same agent, so it measures agreement with the teacher,
not correctness. The template tests are still the score.

Or skip the commands and ask your agent, from the repo root:

```text
Read pigeonhole/README.md. Teach Laya the five templates using yourself as the teacher:
run teach.py with your own CLI, split the data, fine-tune, and score it on the tests.
```

Numbers for both agents are in [`../RESULTS.md`](../RESULTS.md#agent-teachers).
