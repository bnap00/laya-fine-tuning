"""Just enough of Pigeonhole to score a model on its templates, in Python.

Pigeonhole only lets its own Laya provider talk to the stock checkpoints, so
this file reproduces the parts that matter for a fair comparison:

- the questions, sent exactly as Pigeonhole sends them (Jev gets the structured
  criteria; Laya gets each criterion flattened to one line, see `criterion_text`),
- the state, which is the raw input object,
- `when` gates, the five templates' rules and `low_confidence`,
- the test matcher (`matches`), copied from `src/spec/parse.ts`.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).parent
TEMPLATES = sorted((HERE / "templates").glob("*.yaml"))
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"  # what Pigeonhole calls


def load(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def templates_by_id() -> dict:
    return {s["id"]: s for s in map(load, TEMPLATES)}


def asked_nodes(spec: dict) -> dict:
    return {k: v for k, v in spec["nodes"].items() if v["type"] != "rule"}


# ------------------------------------------------------------ wire format

def criterion_text(c):
    """Pigeonhole's `criterionText` (src/provider/laya.ts): what. e.g. a; b. not for x"""
    if not isinstance(c, dict):
        return "; ".join(c) if isinstance(c, list) else c
    parts = [str(c["what"]) if c.get("what") else "",
             "e.g. " + "; ".join(c["examples"]) if c.get("examples") else "",
             "not for " + str(c["not_for"]) if c.get("not_for") else ""]
    return ". ".join(p for p in parts if p)


def to_wire(spec: dict, for_laya: bool) -> dict:
    out = {}
    for qid, n in asked_nodes(spec).items():
        if n["type"] == "score":
            s = n["scale"]
            out[qid] = {"type": "score", "instructions": n["instructions"],
                        "criteria": [s.get("labels", {}).get(str(v), str(v)) for v in range(s["min"], s["max"] + 1)]}
        elif n["type"] == "noul":
            out[qid] = {"type": "noul", "instructions": n["instructions"]}
        else:
            crit = n["criteria"]
            out[qid] = {"type": "choice", "instructions": n["instructions"],
                        "criteria": {k: criterion_text(v) for k, v in crit.items()} if for_laya else crit}
    return out


# ------------------------------------------------------------ models

class Jev:
    def __init__(self, model: str = "jev-latest"):
        self.name = model
        self.http = httpx.Client(timeout=60, headers={
            "authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "X-Title": "laya-fine-tuning"})
        self.cost = 0.0

    def decide(self, spec, state) -> dict:
        questions = to_wire(spec, for_laya=False)
        while True:
            body = {"model": self.name if "/" in self.name else f"~typesafe/{self.name}",
                    "state": state, "questions": questions}
            for attempt in range(4):
                r = self.http.post(DECISIONS_URL, json=body)
                if r.status_code < 500 and r.status_code != 429:
                    break
                time.sleep(2 ** attempt)
            # GPT-6 Luna can refuse one question and OpenRouter then fails the whole call.
            # Like Pigeonhole, drop the refused question and ask the rest again.
            refused = re.search(r'refused to answer question \\?"([^"\\]+)', r.text)
            if r.status_code >= 400 and refused and refused[1] in questions and len(questions) > 1:
                del questions[refused[1]]
                continue
            r.raise_for_status()
            raw = r.json()
            self.cost += (raw.get("usage") or {}).get("cost") or 0
            return raw["answers"]


class Laya:
    def __init__(self, checkpoint: str, device: str | None = None):
        import laya

        self.name = checkpoint
        self.agent = laya.load(checkpoint, **({"device": device} if device else {}))

    def decide(self, spec, state) -> dict:
        return self.agent.predict(state, to_wire(spec, for_laya=True))["answers"]


# ------------------------------------------------------------ executor

def node_results(spec, answers) -> dict:
    """Raw answers -> Pigeonhole's node results (choice / p / score / confidence)."""
    res = {}
    for qid, n in asked_nodes(spec).items():
        a = answers.get(qid) or {}
        if n["type"] == "noul":
            r = {"p": a.get("noul"), "confidence": abs(a.get("noul", 0.5) - 0.5) * 2}
        elif n["type"] == "score":
            probs = a.get("probabilities") or {}
            idx = int(max(probs, key=probs.get)) if probs else round(a.get("score", 0))
            r = {"score": n["scale"]["min"] + idx, "confidence": a.get("confidence")}
        else:
            r = {"choice": a.get("choice"), "confidence": a.get("confidence"), "probabilities": a.get("probabilities")}
        if "min_confidence" in n:
            r["low_confidence"] = (r.get("confidence") or 0) < n["min_confidence"]
        res[qid] = r
    return res


# The five templates' `when` gates and rules, transcribed by hand. A missing or
# skipped node reads as None, which is how a gate that's false ends up as `null`.
def _g(r, node, field):
    return (r.get(node) or {}).get(field)


WHEN = {
    ("support-triage", "return_reason"): lambda r: _g(r, "department", "choice") == "returns",
    ("lead-qualification", "fit"): lambda r: (_g(r, "is_buying_signal", "p") or 0) > 0.5,
}

RULES = {
    "support-triage": {"priority": lambda r: "high" if (_g(r, "angry", "p") or 0) > 0.7 or (_g(r, "urgency", "score") or 0) >= 2
                       else ("normal" if (_g(r, "urgency", "score") or 0) >= 1 else "low")},
    "email-routing": {"queue": lambda r: _g(r, "destination", "choice") + "-urgent"
                      if (_g(r, "is_urgent", "p") or 0) > 0.6 else _g(r, "destination", "choice")},
    "issue-labeling": {"labels": lambda r: f'{_g(r, "kind", "choice")}/{_g(r, "area", "choice")}'},
    "lead-qualification": {"score": lambda r: round((_g(r, "is_buying_signal", "p") or 0) * 50 + (_g(r, "readiness", "score") or 0) * 25)
                           if (_g(r, "is_buying_signal", "p") or 0) > 0.6 and _g(r, "fit", "choice") != "none" else 0},
    "moderation": {"action": lambda r: "publish" if _g(r, "category", "choice") == "ok" else
                   ("escalate" if _g(r, "category", "choice") == "self_harm" or (_g(r, "is_targeting_minor", "p") or 0) > 0.5
                    else ("block" if (_g(r, "severity", "score") or 0) >= 2 else "review"))},
}


def execute(spec, answers) -> dict:
    r = node_results(spec, answers)
    for (sid, node), gate in WHEN.items():
        if sid == spec["id"] and not gate(r):
            r[node] = None
    for node, rule in RULES.get(spec["id"], {}).items():
        r[node] = {"value": rule(r)}
    out = {}
    for key, path in spec["output"].items():
        node, field = path.split(".")
        out[key] = (r.get(node) or {}).get(field)
    return out


def matches(expected, actual) -> bool:
    """`matchesExpectation` from Pigeonhole's src/spec/parse.ts."""
    if actual is None:
        return expected is None or expected == ""
    want = str(expected).strip()
    if isinstance(actual, bool):
        return str(actual).lower() == want.lower()
    if isinstance(actual, (int, float)) and want in ("true", "false"):
        return (actual >= 0.5) == (want == "true")
    if isinstance(actual, (int, float)):
        try:
            return abs(actual - float(want)) < 1e-9
        except ValueError:
            return False
    return str(actual) == want


def run_tests(model, specs=None, verbose=False) -> dict:
    board = {}
    for path in specs or TEMPLATES:
        spec = load(path)
        passed, lat = 0, []
        for t in spec["tests"]:
            t0 = time.perf_counter()
            out = execute(spec, model.decide(spec, t["input"]))
            lat.append(time.perf_counter() - t0)
            ok = all(matches(v, out.get(k)) for k, v in t["expect"].items())
            passed += ok
            if verbose and not ok:
                print(f"  ✗ {spec['id']}: {t['input']} expected {t['expect']} got "
                      f"{ {k: out.get(k) for k in t['expect']} }")
        board[spec["id"]] = {"passed": passed, "total": len(spec["tests"]),
                             "median_ms": round(1000 * sorted(lat)[len(lat) // 2])}
    return board


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score models on the Pigeonhole template tests")
    ap.add_argument("models", nargs="+", help="jev-latest, english, or a checkpoint directory")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--device", help="cpu, cuda; Laya picks one by default")
    args = ap.parse_args()
    results = {}
    for m in args.models:
        model = Jev(m) if m.startswith("jev") or "/" in m and not m.startswith((".", "/")) else Laya(m, args.device)
        results[m] = run_tests(model, verbose=args.verbose)
        total = sum(b["passed"] for b in results[m].values()), sum(b["total"] for b in results[m].values())
        print(f"{m}: {total[0]}/{total[1]}  " + "  ".join(f"{k} {b['passed']}/{b['total']}" for k, b in results[m].items()), flush=True)
        if isinstance(model, Jev):
            print(f"  cost ${model.cost:.4f}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
