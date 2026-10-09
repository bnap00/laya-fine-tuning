"""Write synthetic inputs for each Pigeonhole template with a cheap chat model.

    python generate.py --per-template 400

The generator sees the template's description and its questions, never its
tests: the tests are what we score on afterwards, so they must stay unseen. A
near-duplicate filter drops anything that still looks too much like a test.
"""
import argparse
import difflib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from harness import TEMPLATES, load

MODEL = "anthropic/claude-haiku-5.5"

STYLES = [
    "short and blunt, one sentence",
    "long and rambling, with irrelevant detail before the point",
    "polite and formal",
    "angry, with caps or exclamation marks",
    "full of typos and no punctuation",
    "ambiguous: it could reasonably belong to two of the options",
    "an edge case that sits right on one of the boundaries the spec draws with 'not_for'",
    "written by a non-native English speaker",
    "mentions several topics but has one main point",
    "sarcastic or understated",
]


def ask(http, prompt: str) -> list[str]:
    r = http.post("https://openrouter.ai/api/v1/chat/completions", json={
        "model": MODEL,
        "temperature": 1.0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "inputs", "strict": True, "schema": {
            "type": "object", "additionalProperties": False, "required": ["inputs"],
            "properties": {"inputs": {"type": "array", "items": {"type": "string"}}}}}},
    })
    r.raise_for_status()
    body = r.json()
    ask.cost += (body.get("usage") or {}).get("cost") or 0
    return json.loads(body["choices"][0]["message"]["content"])["inputs"]


ask.cost = 0.0


def prompt_for(spec: dict, style: str, focus: str, n: int) -> str:
    questions = {k: {kk: v[kk] for kk in ("type", "instructions", "criteria", "scale") if kk in v}
                 for k, v in spec["nodes"].items() if v["type"] != "rule"}
    return (
        f"A classifier is being built for this job:\n\n{spec['description'].strip()}\n\n"
        f"It answers these questions about each input:\n{json.dumps(questions, indent=1)}\n\n"
        f"Write {n} realistic, varied inputs it will receive, each as the plain text of its "
        f"`{spec['input']['required'][0]}` field. "
        f"Style for this batch: {style}. {focus} "
        "Vary length, names, products, numbers and wording; do not reuse the examples above verbatim. "
        "Return JSON: {\"inputs\": [...]}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=400)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", default="data/inputs.jsonl")
    args = ap.parse_args()

    http = httpx.Client(timeout=120, headers={"authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    rng = random.Random(0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w") as f:
        for path in TEMPLATES:
            spec = load(path)
            field = spec["input"]["required"][0]  # `title` for issues, `body` everywhere else
            tests = [t["input"][field] for t in spec["tests"]]
            # Spread batches over the options of the template's first choice question, so
            # the data is not ninety percent of whatever the generator finds easiest.
            first = next(n for n in spec["nodes"].values() if n["type"] == "choice")
            focuses = [f"Most of them should be about: {k}." for k in first["criteria"]] + ["Mix everything."]
            jobs = [prompt_for(spec, rng.choice(STYLES), focuses[i % len(focuses)], args.batch)
                    for i in range(args.per_template // args.batch)]
            with ThreadPoolExecutor(8) as pool:
                batches = list(pool.map(lambda p: ask(http, p), jobs))
            seen, kept, leaked = set(), 0, 0
            for text in (t.strip() for b in batches for t in b):
                key = text.lower()
                if not text or key in seen:
                    continue
                seen.add(key)
                if any(difflib.SequenceMatcher(None, key, t.lower()).ratio() > 0.75 for t in tests):
                    leaked += 1
                    continue
                f.write(json.dumps({"template": spec["id"], "input": {field: text}}) + "\n")
                kept += 1
            total += kept
            print(f"{spec['id']}: {kept} inputs ({leaked} dropped as too close to a test)", flush=True)
    print(f"{total} inputs, ${ask.cost:.4f}")


if __name__ == "__main__":
    main()
