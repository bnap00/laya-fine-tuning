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
