"""Turn a coding agent's one-pass reply into Laya training rows.

The agent writes inputs and, for each, its probabilities over every option of
every question. This module checks that reply against the template and builds
rows in exactly the shape `label.py` writes for Jev: the questions in the
one-line form Pigeonhole sends Laya, `gold` in the trainer's shapes, and `when`
gates applied from the teacher's own answers.
"""
from __future__ import annotations

import difflib
import json
import re

from harness import WHEN, node_results, to_wire


def parse_reply(text: str) -> list[dict]:
    """The `rows` list from a reply that may wrap its JSON in prose or a code fence."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    candidates = [fenced[1]] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
            return obj["rows"]
    raise ValueError("no JSON object with a `rows` list in the reply")


def normalise(probs: dict, options: list[str]) -> dict | None:
    """Probabilities over exactly `options`, summing to 1, or None if they can't be trusted.

    Missing options count as 0. Percentages (a sum near 100) are rescaled. Unknown
    option names, negatives, and sums outside [0.9, 1.1] after that are rejected.
    """
    if not isinstance(probs, dict) or set(map(str, probs)) - set(options):
        return None
    try:
        p = {o: float(probs.get(o, 0)) for o in options}
    except (TypeError, ValueError):
        return None
    if any(v < 0 for v in p.values()):
        return None
    total = sum(p.values())
    if 90 <= total <= 110:
        p, total = {o: v / 100 for o, v in p.items()}, total / 100
    if not 0.9 <= total <= 1.1:
        return None
    return {o: v / total for o, v in p.items()}


def _options(question: dict) -> list[str]:
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["false", "true"]


def to_row(spec: dict, text: str, answers: dict) -> tuple[dict | None, str]:
    questions = to_wire(spec, for_laya=True)
    gold, jev_style = {}, {}
    for qid, q in questions.items():
        if qid not in answers:
            return None, "missing_question"
        a = answers[qid]
        if q["type"] == "noul":
            try:
                p_true = float(a["true"] if isinstance(a, dict) else a)
            except (TypeError, ValueError, KeyError):
                return None, "bad_probabilities"
            if not 0 <= p_true <= 1:  # no percent guess here: 1.7 is an error, not 1.7%
                return None, "bad_probabilities"
            gold[qid] = {"probabilities": {"false": 1 - p_true, "true": p_true}}
            jev_style[qid] = {"noul": p_true}
            continue
        opts = _options(q)
        if not isinstance(a, dict) or set(map(str, a)) - set(opts):
            return None, "bad_options"
        p = normalise({str(k): v for k, v in a.items()}, opts)
        if p is None:
            return None, "bad_probabilities"
        gold[qid] = {"probabilities": p}
        top = max(p, key=p.get)
        jev_style[qid] = {"probabilities": p, "choice": top, "confidence": p[top]}
    results = node_results(spec, jev_style)
    for (sid, node), gate in WHEN.items():
        if sid == spec["id"] and not gate(results):
            questions.pop(node, None)
            gold.pop(node, None)
    field = spec["input"]["required"][0]
    return {"template": spec["id"], "state": {field: text}, "questions": questions, "gold": gold}, "ok"


def too_close(text: str, tests: list[str]) -> bool:
    key = text.lower()
    return any(difflib.SequenceMatcher(None, key, t.lower()).ratio() > 0.75 for t in tests)
