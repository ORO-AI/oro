"""Regression tests for `subnet.test_runner._score_output`."""

import json

import subnet.test_runner as tr


def _write(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _problem():
    return [
        {
            "problem_id": "p1",
            "query": "find a thing",
            "category": "Product",
            "reward": {"product_id": "x", "title": ["x"]},
        }
    ]


def test_score_output_skips_non_dict_first_step(tmp_path, capsys):
    """Dialogue whose first entry is not a dict must skip cleanly,
    not raise AttributeError on `output[0].get(...)` — stale sandbox
    images can write this shape."""
    output_file = tmp_path / "output.jsonl"
    _write(
        output_file,
        [{"problem_id": "p1", "status": "SUCCESS", "dialogue": ["raw text"]}],
    )

    score = tr._score_output(output_file, _problem(), skip_reasoning=True)

    assert score == -1.0
    assert "No problems scored" in capsys.readouterr().err
