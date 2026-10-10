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


def test_codex_events_surface_usage_limit_error():
    from teach import parse_codex_events

    stdout = "\n".join(json.dumps(e) for e in [
        {"type": "thread.started"},
        {"type": "turn.started"},
        {"type": "error", "message": "You've hit your usage limit. Try again at 2:59 PM."},
        {"type": "turn.failed", "error": {"message": "You've hit your usage limit. Try again at 2:59 PM."}},
    ])
    text, usage, error = parse_codex_events(stdout)
    assert text == "" and usage == {}
    assert "usage limit" in error


def test_codex_events_reply_and_usage():
    from teach import parse_codex_events

    stdout = "\n".join(json.dumps(e) for e in [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "{\"rows\": []}"}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
    ])
    assert parse_codex_events(stdout) == ('{"rows": []}', {"input_tokens": 10, "output_tokens": 2}, None)


def test_completed_batches_survive_an_interrupt(tmp_path):
    import pytest
    from teach import run_jobs

    answers = {"department": {"returns": 0.1, "shipping": 0.8, "billing": 0.1, "other": 0.0},
               "return_reason": {"other": 1.0}, "angry": 0.1, "urgency": {"0": 0.7, "1": 0.2, "2": 0.1}}
    calls = []

    def run(prompt, model):
        calls.append(prompt)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return json.dumps({"rows": [{"input": f"parcel {len(calls)} is late", "answers": answers}]}), {"seconds": 1.0}

    jobs = [(SPEC, f"p{i}") for i in range(3)]
    with pytest.raises(KeyboardInterrupt):
        run_jobs(jobs, run, "m", tmp_path, workers=1)
    rows = (tmp_path / "labelled.jsonl").read_text().splitlines()
    assert [json.loads(r)["state"]["body"] for r in rows] == ["parcel 1 is late", "parcel 2 is late"]
    assert len((tmp_path / "usage.jsonl").read_text().splitlines()) == 2
