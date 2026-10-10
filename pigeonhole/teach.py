"""Use your Claude Code or Codex subscription as the teacher, in one pass.

    python teach.py --agent claude --model claude-opus-5-5
    python teach.py --agent codex --model gpt-6.1-sol

Each call asks the agent to write a batch of realistic inputs for one template
and, for each, its probability over every option of every question. The agent
sees the template's description and questions, never its tests; anything too
close to a test is dropped anyway. Rows are written in the shape `label.py`
writes for Jev, so `split.py` and `finetune.py` take it from there.
"""
import argparse
import json
import random
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent_rows import parse_reply, to_row, too_close
from generate import STYLES
from harness import TEMPLATES, load, to_wire

SYSTEM = "You write realistic test data and label it carefully. Reply with JSON only."


def build_prompt(spec: dict, style: str, focus: str, n: int) -> str:
    questions = to_wire(spec, for_laya=True)
    field = spec["input"]["required"][0]
    shape = {}
    for qid, q in questions.items():
        if q["type"] == "choice":
            shape[qid] = {k: "probability" for k in q["criteria"]}
        elif q["type"] == "score":
            shape[qid] = {str(i): f"probability of '{label}'" for i, label in enumerate(q["criteria"])}
        else:
            shape[qid] = "probability the statement is true"
    return (
        f"A classifier is being built for this job:\n\n{spec['description'].strip()}\n\n"
        f"It answers these questions about each input:\n{json.dumps(questions, indent=1)}\n\n"
        f"1. Write {n} realistic, varied inputs it will receive, each the plain text of its `{field}` field. "
        f"Style for this batch: {style}. {focus} Vary length, names, products, numbers and wording.\n"
        "2. For each input, answer every question as an expert reviewer would, with honest probabilities: "
        "spread probability when an input is genuinely ambiguous, be confident only when it is clear. "
        "Answer every question for every input, even ones that seem not to apply.\n\n"
        'Return one JSON object and nothing else: {"rows": [{"input": "...", "answers": ANSWERS}]}\n'
        f"where ANSWERS has this shape (numbers from 0 to 1; each question's options sum to 1):\n"
        f"{json.dumps(shape, indent=1)}"
    )


def run_claude(prompt: str, model: str) -> tuple[str, dict]:
    t = time.time()
    with tempfile.TemporaryDirectory() as cwd:  # no project CLAUDE.md, hooks or settings
        out = subprocess.run(
            ["claude", "-p", "--model", model, "--output-format", "json", "--system-prompt", SYSTEM,
             "--tools", "", "--strict-mcp-config", "--setting-sources", ""],
            input=prompt, capture_output=True, text=True, cwd=cwd, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(f"claude exited {out.returncode}: {out.stderr[-300:]}")
    d = json.loads(out.stdout)
    u = d.get("usage") or {}
    return d.get("result") or "", {
        "seconds": round(time.time() - t, 1),
        "input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
        "cached_input_tokens": u.get("cache_read_input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "cost_usd": d.get("total_cost_usd"),
    }


def parse_codex_events(stdout: str) -> tuple[str, dict, str | None]:
    """The reply text, token usage and error message from `codex exec --json` output.

    Codex reports failures such as a plan's usage limit as events on stdout, not stderr.
    """
    text, usage, error = "", {}, None
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "item.completed" and ev["item"].get("type") == "agent_message":
            text = ev["item"]["text"]
        elif ev.get("type") == "turn.completed":
            usage = ev.get("usage") or {}
        elif ev.get("type") == "error":
            error = ev.get("message")
        elif ev.get("type") == "turn.failed":
            error = (ev.get("error") or {}).get("message") or error
    return text, usage, error


def run_codex(prompt: str, model: str) -> tuple[str, dict]:
    t = time.time()
    with tempfile.TemporaryDirectory() as cwd:
        out = subprocess.run(
            ["codex", "exec", "--json", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-m", model, "-"],
            input=prompt, capture_output=True, text=True, cwd=cwd, timeout=900)
    text, usage, error = parse_codex_events(out.stdout)
    if out.returncode != 0 and not text:
        raise RuntimeError(f"codex exited {out.returncode}: {error or out.stderr[-300:]}")
    return text, {
        "seconds": round(time.time() - t, 1),
        "input_tokens": usage.get("input_tokens", 0),
        "cached_input_tokens": usage.get("cached_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0) + usage.get("reasoning_output_tokens", 0),
        "cost_usd": None,
    }


def call_with_retry(run, prompt: str, model: str, tries: int = 2):
    usage = []
    for _ in range(tries):
        try:
            text, u = run(prompt, model)
        except Exception as e:  # a failed call costs one batch, not the run
            usage.append({"error": str(e)[:300]})
            continue
        usage.append(u)
        try:
            return parse_reply(text), usage
        except ValueError as e:
            u["error"] = str(e)
    return None, usage


def run_jobs(jobs, run, model: str, out: Path, workers: int) -> Counter:
    """Run every (spec, prompt) job and write each batch's rows and usage as it finishes.

    Writing as we go means an interrupted run keeps every batch it already paid for.
    """
    counts, seen = Counter(), set()
    with (out / "labelled.jsonl").open("w") as rows_f, (out / "usage.jsonl").open("w") as usage_f, \
            ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(call_with_retry, run, prompt, model): spec for spec, prompt in jobs}
        for future in as_completed(futures):
            spec = futures[future]
            rows, usage = future.result()
            for u in usage:
                usage_f.write(json.dumps({"template": spec["id"], **u}) + "\n")
            if rows is None:
                counts["failed_calls"] += 1
            else:
                field = spec["input"]["required"][0]
                tests = [t["input"][field] for t in spec["tests"]]
                for r in rows:
                    text = str(r.get("input", "")).strip() if isinstance(r, dict) else ""
                    if not text or (spec["id"], text.lower()) in seen:
                        counts["duplicate_or_empty"] += 1
                        continue
                    seen.add((spec["id"], text.lower()))
                    if too_close(text, tests):
                        counts["too_close_to_a_test"] += 1
                        continue
                    row, why = to_row(spec, text, r.get("answers") or {})
                    counts[why] += 1
                    if row:
                        rows_f.write(json.dumps(row) + "\n")
            rows_f.flush()
            usage_f.flush()
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["claude", "codex"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-template", type=int, default=400)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--templates", nargs="*", help="template ids; default all five")
    ap.add_argument("--out")
    args = ap.parse_args()

    run = run_claude if args.agent == "claude" else run_codex
    out = Path(args.out or f"data/{args.agent}")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    specs = [s for s in map(load, TEMPLATES) if not args.templates or s["id"] in args.templates]

    jobs = []
    for spec in specs:
        first = next(n for n in spec["nodes"].values() if n["type"] == "choice")
        focuses = [f"Most of them should be about: {k}." for k in first["criteria"]] + ["Mix everything."]
        for i in range(args.per_template // args.batch):
            jobs.append((spec, build_prompt(spec, rng.choice(STYLES), focuses[i % len(focuses)], args.batch)))

    t0 = time.time()
    counts = run_jobs(jobs, run, args.model, out, args.workers)

    usage = [json.loads(line) for line in (out / "usage.jsonl").open()]
    ok = [u for u in usage if "seconds" in u]
    summary = {
        "agent": args.agent, "model": args.model, "calls": len(usage), "failed_calls": counts.pop("failed_calls", 0),
        "rows": dict(counts), "wall_minutes": round((time.time() - t0) / 60, 1),
        "input_tokens": sum(u.get("input_tokens", 0) for u in ok),
        "cached_input_tokens": sum(u.get("cached_input_tokens", 0) for u in ok),
        "output_tokens": sum(u.get("output_tokens", 0) for u in ok),
        "cost_usd": round(sum(u.get("cost_usd") or 0 for u in ok), 2) if args.agent == "claude" else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
