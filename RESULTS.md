# Results

The main results ran on one desktop: Intel Core i5-8500, 32 GB RAM, NVIDIA RTX 3050 (6 GB),
`laya` 0.4.1, torch 2.14.1+cu126. Every run was repeated on a Mac mini with an M4 Pro; those
numbers are in [Apple M4 Pro](#apple-m4-pro) at the end. All fine-tunes start from
`convaiinnovations/laya` (English, 421M) and train the top 6 of 28 encoder layers plus the
decision head.

## Pac-Man

200 games per player on seeds 500000-500199. No training game used those seeds.

| player | games won | pellets eaten (of 111) | mean score | picks the teacher's move | ms per move |
| --- | --- | --- | --- | --- | --- |
| random | 0 / 200 | 10.2 | -398 | 42% | - |
| Laya, untrained | 0 / 200 | 11.9 | -381 | 38% | 42.8 |
| Laya, fine-tuned (v1) | 63 / 200 | 81.0 | 625 | 91% | 42.9 |
| Laya, + one DAgger round (v2) | **87 / 200** | **85.1** | **786** | 92% | 42.8 |
| teacher (hand-written expert) | 78 / 200 | 82.3 | 713 | 100% | - |

v2 vs the teacher (87 vs 78 wins) is within noise for 200 games; read it as "level with the
teacher", not "better".

| run | data | training | time |
| --- | --- | --- | --- |
| v1 | 12,595 states from 300 teacher games (20% random moves) | 2 epochs, 764 updates | 41 min |
| v2 | 11,925 states from 150 games **driven by v1**, labelled by the teacher, + 6,000 v1 states | 1 epoch from v1 | 28 min |

v1 on 1,694 held-out states (`pacman/data/test.jsonl`, 195 of them shared opening positions):
accuracy against the teacher's top move 0.62 → 0.90, ECE 0.134 → 0.041.

## Pigeonhole templates

The 50 test cases of Pigeonhole's five templates (as of
[`19366f8`](https://github.com/bnap00/pigeonhole/commit/19366f8), with the quoted option text),
scored with Pigeonhole's own matcher. Hosted models through OpenRouter's Decisions endpoint;
Laya on the RTX 3050. Median latency per call (every question of a pipeline in one call).

| | email routing | issue labelling | lead qualification | moderation | support triage | **total** | median latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jev | 9 | 10 | 9 | 9 | 10 | **47 / 50** | ~500 ms |
| GPT-6 Luna | 8 | 10 | 8 | 10 | 10 | **46 / 50** | ~330 ms* |
| Clef | 8 | 10 | 9 | 7 | 10 | **44 / 50** | ~600 ms |
| Clef-flash | 8 | 10 | 8 | 7 | 10 | **43 / 50** | ~680 ms |
| Laya, untrained | 6 | 5 | 4 | 10 | 6 | **31 / 50** | ~100 ms |
| **Laya, fine-tuned** | 8 | 9 | 7 | 10 | 10 | **44 / 50** | ~100 ms |

\* GPT-6 Luna refuses `is_good_first_issue` on title-only issues; dropping the refused question
and asking again (as Pigeonhole does) put its issue-labelling median at ~17 s from here.

| step | what | cost |
| --- | --- | --- |
| inputs | 1,971 synthetic inputs, Claude Haiku 5.5 | ~$0.10 |
| labels | Jev on every question of every input | ~$0.07 |
| training | 6,358 question-items, 3 epochs, 597 updates | 28 min |

Total OpenRouter spend for the whole project, including a full relabel after the template fix
and every test run of every hosted model: $0.27.

Still wrong after fine-tuning: a contractor pitch goes to `sales` instead of `recruiting`; "Buy
cheap followers" is a `bug`; lead qualification answers a keyboard question with `none` where
the test expects the gate to skip it, misses an Airbyte complaint, and calls "TIL about window
functions" a data-pipeline lead. One email-routing test expects `urgent` to be empty for an
outage, which no model passes.

An earlier run on the templates *before* the quoting fix (support triage and issue labelling
had option descriptions cut at the first comma) scored 41 / 50 fine-tuned.

## Apple M4 Pro

The same three training runs and the same tests again, on a Mac mini: Apple M4 Pro (12-core
CPU, 16-core GPU), 24 GB unified memory, macOS 26.5, `laya` 0.4.1, torch 2.14.1 on the MPS
backend. Same data, same commands, same flags. Laya's trainer uses fp16 autocast and gradient
checkpointing only on CUDA, so these runs are full fp32.

### Training

| run | updates | RTX 3050 | M4 Pro | M4 Pro peak memory |
| --- | --- | --- | --- | --- |
| Pigeonhole, 6,358 items × 3 epochs | 597 | 28 min | **25.6 min** | 13.0 GB |
| Pac-Man v1, 12,195 items × 2 epochs | 764 | 41 min | **37.0 min** | 8.8 GB |
| Pac-Man v2 (DAgger), 17,632 items × 1 epoch | 548 / 551 | 28 min | **24.0 min** | 8.8 GB |

Peak memory is the process's peak footprint from `/usr/bin/time -l`, GPU allocations included.
The RTX 3050 peaked at about 3.8 GB on the same runs.

### Pigeonhole templates

| | email routing | issue labelling | lead qualification | moderation | support triage | **total** | median latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Laya, untrained | 6 | 5 | 4 | 10 | 6 | **31 / 50** | ~105-150 ms |
| Laya, fine-tuned on the M4 Pro | 7 | 9 | 6 | 10 | 10 | **42 / 50** | ~105-150 ms |
| *Laya, fine-tuned on the RTX 3050* | *8* | *9* | *7* | *10* | *10* | *44 / 50* | *~100 ms* |

Untrained scores are identical on both machines. The Mac fine-tune lands two cases below the
3050 one, from identical data and settings; the difference is the run itself (fp32 vs fp16
autocast, different kernels), the kind of one-or-two-case swing the main results already treat
as a tie. The cases moved both ways: the Mac checkpoint gets the Airbyte complaint right, and
gets a declined-card email (`junk` instead of `invoices`), a Postgres-to-Snowflake complaint
and one lead-readiness score wrong. Scores: [`pigeonhole/results/laya-m4pro.json`](pigeonhole/results/laya-m4pro.json).

### Pac-Man

200 games per player on seeds 500000-500199, as above. Random and teacher don't touch the GPU
and reproduce exactly, so the game is deterministic across machines.

| player | games won | pellets eaten (of 111) | mean score | picks the teacher's move | ms per move |
| --- | --- | --- | --- | --- | --- |
| random | 0 / 200 | 10.2 | -398 | 42% | - |
| Laya, untrained | 0 / 200 | 12.3 | -377 | 40% | 53.6 |
| Laya, fine-tuned (v1) | 63 / 200 | 78.8 | 603 | 88% | 53.7 |
| Laya, + one DAgger round (v2) | **77 / 200** | **83.8** | **723** | 93% | 51.7 |
| teacher (hand-written expert) | 78 / 200 | 82.3 | 713 | 100% | - |

v1 matches the 3050's v1 exactly on wins (63). v2 lands at 77, against 87 on the 3050: one
DAgger round still adds 14 wins and puts the student level with its teacher, and the 3050's 87
was the lucky end of that range. The DAgger set differs between machines because each v1
drives its own games (12,032 states here, 11,925 on the 3050); both were mixed with 6,000 v1
states.

v1 on the 1,694 held-out states: accuracy against the teacher's top move 0.62 → 0.90, ECE
0.134 → 0.039 (3050: 0.90 and 0.041).

Single moves are about a quarter slower than on the RTX 3050 (52-54 ms vs ~43 ms): one row
doesn't fill the GPU. Scoreboards: [`pacman/results/m4pro/`](pacman/results/m4pro/).

## Agent teachers

The Pigeonhole fine-tune again, with no API key: a coding agent on a personal subscription
writes the inputs and labels them in one pass ([`pigeonhole/teach.py`](pigeonhole/teach.py)).
Same five templates, 400 inputs per template, 25 per call, the same 50 tests. Trained and scored
on the Mac mini (M4 Pro), so compare against the Jev-taught run on the same machine.

> Generating training data with an agent is a great way to learn and to get a first model
> working. For a real deployment, train on a golden dataset of real inputs with checked labels.

### What the teachers did

| | Claude Code, Opus 5.5 (`claude-opus-5-5`) | Codex, Sol 6.1 (`gpt-6.1-sol`) |
| --- | --- | --- |
| calls of 25 inputs | 80 | 80 (+16 retried after a usage limit, below) |
| rows kept | 1,997 | 1,998 |
| dropped | 2 missing an answer, 1 too close to a test | 2 too close to a test |
| wall time (4 calls at a time) | 19.7 min | 25.6 min (two runs) |
| input tokens | 124,085 (13,234 cached) | 1,579,913 (1,311,872 cached) |
| output tokens | 472,084 | 336,803 (incl. reasoning) |
| same work on the API | $10.47 (reported by Claude Code) | ~$4.04 (estimate at $2 / $0.10 cached / $10 per M tokens) |

Claude Code runs headless with a one-line `--system-prompt` and no tools: ~1.6k input tokens a
call instead of ~30k with its default prompt. Codex can't drop its built-in prompt, so each call
carries ~20k input tokens, mostly cached.

**The usage limit.** On ChatGPT Plus ($20 a month), Codex hit its five-hour limit after 64 of
the 80 calls; every `support-triage` call failed. After one rate-limit reset (both windows back
to 0%), the 16 `support-triage` calls used **28% of the five-hour window and 4% of the weekly
one**. Scaled up, one 2,000-input run needs about 1.4 five-hour windows and ~20% of a Plus week.
Claude Code, on a $100 Claude plan (five times the price), finished in one go.

### Teacher confidence

Highest probability per label, over every label each teacher wrote:

| teacher | mean | labels ≥ 0.99 | labels < 0.7 |
| --- | --- | --- | --- |
| Jev | 0.86 | 26% | 18% |
| Opus 5.5 | 0.84 | 13% | 21% |
| Sol 6.1 | 0.89 | 32% | 14% |

Writing its own inputs did not make Opus 5.5 overconfident: it hedged more than Jev. Sol 6.1 was
the most decisive. [`pigeonhole/results/teacher-confidence.json`](pigeonhole/results/teacher-confidence.json)

### Training

| | Opus 5.5-taught | Sol 6.1-taught | Jev-taught (same Mac) |
| --- | --- | --- | --- |
| train / held out | 1,797 / 200 inputs | 1,798 / 200 inputs | 6,358 items, no split |
| question-items, updates | 5,713, 537 | 5,746, 540 | 6,358, 597 |
| time, peak memory | 22.6 min, 12.8 GB | 21.3 min, 10.9 GB | 25.6 min, 13.0 GB |
| agreement with its teacher, held out | 0.61 → 0.70 | 0.61 → 0.72 | |
| ECE, held out | 0.116 → 0.057 | 0.108 → 0.081 | |

The held-out rows are labelled by the same agent, so they measure agreement with the teacher,
not correctness.

### The 50 tests

| | email routing | issue labelling | lead qualification | moderation | support triage | **total** |
| --- | --- | --- | --- | --- | --- | --- |
| Laya, untrained | 6 | 5 | 4 | 10 | 6 | **31 / 50** |
| Laya, Jev-taught (M4 Pro) | 7 | 9 | 6 | 10 | 10 | **42 / 50** |
| **Laya, Opus 5.5-taught** | 5 | 9 | 6 | 10 | 10 | **40 / 50** |
| **Laya, Sol 6.1-taught** | 5 | 9 | 5 | 10 | 9 | **38 / 50** |

Both agent-taught models land two and four cases below the Jev-taught one, and seven and nine
above untrained. The gap is mostly email routing. Both agent-taught models get the same five
email cases wrong; the Jev-taught model misses three of those too (a declined-card notice to
`junk`, a contractor pitch to `sales`, and the outage flagged urgent, which no model passes).
The two only the agent-taught models miss: a pricing question goes to `junk` and an API-key
question to `other`. Both agents wrote more `junk` than Jev labelled (20% and 17% of their email
inputs, against 13%); their few declined-card emails were labelled correctly, so the cause isn't
established.

Scores: [`pigeonhole/results/claude-taught.json`](pigeonhole/results/claude-taught.json),
[`pigeonhole/results/codex-taught.json`](pigeonhole/results/codex-taught.json). Data, usage
per call and summaries: [`pigeonhole/data/claude/`](pigeonhole/data/claude/),
[`pigeonhole/data/codex/`](pigeonhole/data/codex/).
