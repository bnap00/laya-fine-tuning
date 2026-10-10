import json

from harness import templates_by_id
from teach import build_prompt, call_with_retry

SPEC = templates_by_id()["support-triage"]


def test_prompt_has_questions_but_never_tests():
    p = build_prompt(SPEC, "polite and formal", "Most of them should be about: returns.", 25)
    assert "Which team should handle this ticket?" in p
    assert "25" in p and "polite and formal" in p and '"rows"' in p
    for t in SPEC["tests"]:
        assert t["input"]["body"] not in p


def test_retry_then_success():
    calls = []

    def flaky(prompt, model):
        calls.append(1)
        if len(calls) == 1:
            return "not json", {"seconds": 1.0}
        return json.dumps({"rows": [{"input": "x", "answers": {}}]}), {"seconds": 2.0}

    rows, usage = call_with_retry(flaky, "p", "m")
    assert rows == [{"input": "x", "answers": {}}]
    assert len(usage) == 2


def test_retry_gives_up_after_errors():
    def broken(prompt, model):
        raise RuntimeError("cli exited 1")

    rows, usage = call_with_retry(broken, "p", "m")
    assert rows is None
    assert len(usage) == 2 and all(u.get("error") for u in usage)
