"""Local test-runner score_output robustness.

Regression for the AttributeError that surfaced after miners pulled the
new test_runner without rebuilding the sandbox image. The runner now
skips any output line whose first dialogue entry is not a dict, so a
stale image cache produces a clean "no problems scored" warning instead
of a traceback.
"""

import json

import pytest

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
    output_file = tmp_path / "output.jsonl"
    _write(
        output_file,
        [
            # Envelope with a dialogue whose first entry is not a dict —
            # what an older sandbox image could write.
            {
                "problem_id": "p1",
                "status": "SUCCESS",
                "dialogue": ["raw text instead of dict"],
            }
        ],
    )

    score = tr._score_output(output_file, _problem(), skip_reasoning=True)

    assert score == -1.0
    err = capsys.readouterr().err
    assert "No problems scored" in err


def test_score_output_skips_empty_dialogue(tmp_path, capsys):
    output_file = tmp_path / "output.jsonl"
    _write(output_file, [{"problem_id": "p1", "status": "SUCCESS", "dialogue": []}])

    score = tr._score_output(output_file, _problem(), skip_reasoning=True)

    assert score == -1.0
    assert "No problems scored" in capsys.readouterr().err


def test_score_output_skips_failed_envelope(tmp_path, capsys):
    output_file = tmp_path / "output.jsonl"
    _write(
        output_file,
        [{"problem_id": "p1", "status": "TIMED_OUT", "dialogue": []}],
    )

    score = tr._score_output(output_file, _problem(), skip_reasoning=True)

    assert score == -1.0
    assert "No problems scored" in capsys.readouterr().err
