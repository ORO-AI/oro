"""Test fixture: agent that uses `from __future__ import annotations`
combined with `@dataclass(frozen=True)`.

This pattern requires the loader to register the module in sys.modules
before exec_module so Python's dataclass machinery can resolve string
annotations via sys.modules[cls.__module__].
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Tunables:
    answer: str = "hello-frozen"
    threshold: float = 9.0


_T = _Tunables()


def agent_main(problem):
    return {"answer": _T.answer, "query": problem.get("query", "")}
