import pytest

from agent_rows import normalise, parse_reply, to_row, too_close
from harness import templates_by_id

SPECS = templates_by_id()
TRIAGE = SPECS["support-triage"]


def triage_answers(dept="returns"):
    return {
        "department": {"returns": 0.9, "shipping": 0.05, "billing": 0.05, "other": 0.0} if dept == "returns"
        else {"returns": 0.0, "shipping": 0.9, "billing": 0.1, "other": 0.0},
        "return_reason": {"wrong_size": 0.8, "wrong_item": 0.1, "damaged": 0.1, "changed_mind": 0.0, "other": 0.0},
        "angry": 0.2,
        "urgency": {"0": 0.6, "1": 0.3, "2": 0.1},
    }


def test_parse_reply_plain_json():
    assert parse_reply('{"rows": [{"input": "a", "answers": {}}]}') == [{"input": "a", "answers": {}}]


def test_parse_reply_fenced_with_prose():
    text = 'Here you go:\n```json\n{"rows": [{"input": "a", "answers": {}}]}\n```\nDone.'
    assert parse_reply(text)[0]["input"] == "a"


def test_parse_reply_garbage_raises():
    with pytest.raises(ValueError):
        parse_reply("sorry, I can't do that")


def test_normalise_fills_missing_and_rescales():
    assert normalise({"a": 0.5, "b": 0.47}, ["a", "b", "c"]) == pytest.approx({"a": 0.5 / 0.97, "b": 0.47 / 0.97, "c": 0.0})


def test_normalise_percentages():
    assert normalise({"a": 90, "b": 10}, ["a", "b"]) == pytest.approx({"a": 0.9, "b": 0.1})


def test_normalise_rejects_unknown_option_negative_and_far_sums():
    assert normalise({"a": 0.5, "zzz": 0.5}, ["a", "b"]) is None
    assert normalise({"a": -0.1, "b": 1.1}, ["a", "b"]) is None
    assert normalise({"a": 0.3, "b": 0.3}, ["a", "b"]) is None


def test_to_row_shapes_gold_like_label_py():
    row, why = to_row(TRIAGE, "These are too small, can I get a 10?", triage_answers())
    assert why == "ok"
    assert row["template"] == "support-triage"
    assert row["state"] == {"body": "These are too small, can I get a 10?"}
    assert set(row["questions"]) == {"department", "return_reason", "angry", "urgency"}
    assert row["gold"]["angry"] == {"probabilities": {"false": pytest.approx(0.8), "true": 0.2}}
    assert row["gold"]["urgency"]["probabilities"] == pytest.approx({"0": 0.6, "1": 0.3, "2": 0.1})
    assert row["questions"]["department"]["criteria"]["returns"].startswith("Exchanges, refunds")


def test_to_row_drops_gated_question():
    row, why = to_row(TRIAGE, "Where is my parcel?", triage_answers(dept="shipping"))
    assert why == "ok"
    assert "return_reason" not in row["questions"] and "return_reason" not in row["gold"]


def test_to_row_reports_reasons():
    a = triage_answers(); del a["angry"]
    assert to_row(TRIAGE, "x", a) == (None, "missing_question")
    a = triage_answers(); a["department"] = {"refunds": 1.0}
    assert to_row(TRIAGE, "x", a) == (None, "bad_options")
    a = triage_answers(); a["angry"] = 1.7
    assert to_row(TRIAGE, "x", a) == (None, "bad_probabilities")


def test_too_close():
    tests = ["Shoes came in the wrong size, can I swap for a 10?"]
    assert too_close("shoes came in the wrong size, can i swap for a 10??", tests)
    assert not too_close("My parcel has been stuck in Leeds for a week.", tests)
