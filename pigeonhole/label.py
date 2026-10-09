"""Ask Jev every question about every synthetic input, and keep its probabilities.

    python label.py data/inputs.jsonl data/train.jsonl

Each row is written in the shape `laya-train` reads, with the questions in the
exact one-line form Pigeonhole sends to Laya, so the fine-tuned checkpoint is
trained on the same text it will be asked at inference. The targets are Jev's
full distributions (`gold`), not just its top answer.

A question gated by `when` that Jev's own answers switch off (no return reason
for a billing ticket) is left out of that row: there is no right answer to copy.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness import WHEN, Jev, load, node_results, templates_by_id, to_wire


def gold_for(question: dict, answer: dict) -> dict:
    if question["type"] == "noul":
        p = float(answer["noul"])
        return {"probabilities": {"false": 1 - p, "true": p}}
    return {"probabilities": answer["probabilities"]}


def main(src: str, dst: str):
    rows = [json.loads(line) for line in open(src)]
    specs = templates_by_id()
    jev = Jev()

    def label(row):
        spec = specs[row["template"]]
        try:
            answers = jev.decide(spec, row["input"])
        except Exception as e:  # one bad request shouldn't cost the whole run
            print("skipped:", e, file=sys.stderr)
            return None
        results = node_results(spec, answers)
        questions = to_wire(spec, for_laya=True)
        for (sid, node), gate in WHEN.items():
            if sid == spec["id"] and not gate(results):
                questions.pop(node, None)
        return {"template": spec["id"], "state": row["input"], "questions": questions,
                "gold": {q: gold_for(questions[q], answers[q]) for q in questions if q in answers}}

    with ThreadPoolExecutor(8) as pool:
        labelled = [r for r in pool.map(label, rows) if r]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for r in labelled:
            f.write(json.dumps(r) + "\n")
    print(f"{len(labelled)} of {len(rows)} inputs labelled by Jev, ${jev.cost:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
