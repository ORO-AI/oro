"""Compatibility message for the retired direct local test command."""

import sys


def main() -> None:
    print(
        "This legacy test command cannot use the inference proxy. "
        "Run `docker compose run test --agent-file my_agent.py` instead.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
