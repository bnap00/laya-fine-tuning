from split import split


def rows(n_a, n_b):
    return [{"template": "a", "i": i} for i in range(n_a)] + [{"template": "b", "i": i} for i in range(n_b)]


def test_split_is_stratified_and_complete():
    train, held = split(rows(100, 50), heldout=0.1, seed=0)
    assert sum(r["template"] == "a" for r in held) == 10
    assert sum(r["template"] == "b" for r in held) == 5
    assert len(train) + len(held) == 150
    assert not {id(r) for r in train} & {id(r) for r in held}


def test_split_is_deterministic_and_seeded():
    data = rows(40, 40)
    assert split(data, 0.1, 0) == split(data, 0.1, 0)
    assert split(data, 0.1, 0)[1] != split(data, 0.1, 1)[1]


def test_split_keeps_one_heldout_for_tiny_templates():
    _, held = split(rows(3, 0), heldout=0.1, seed=0)
    assert len(held) == 1
