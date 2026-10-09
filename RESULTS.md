# Results

Everything below ran on one desktop: Intel Core i5-8500, 32 GB RAM, NVIDIA RTX 3050 (6 GB),
`laya` 0.4.1, torch 2.14.1+cu126. All fine-tunes start from `convaiinnovations/laya` (English,
421M) and train the top 6 of 28 encoder layers plus the decision head.

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
