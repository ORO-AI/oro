# oro

Public validator and miner code for the ORO Bittensor subnet. Start with `README.md`,
`docs/validator-guide.md` and `docs/miner-guide.md`.

## Rules

- **This repository is public.** Commits, branch names, PR titles and bodies, and
  code comments must not contain internal ticket IDs, internal hostnames or URLs,
  credentials, or unreleased evaluation content (agent code, task suites). Operator
  procedures are maintained privately; don't add them here.
- **Tests import `validator.X`, never `subnet.validator.X`.** Pytest puts `subnet`
  on the path (`pyproject.toml`). Importing the same module under both names
  registers its Prometheus metrics twice and fails with a duplicate-timeseries error.
- **The validator talks to the backend through `oro_sdk`.** Use the SDK functions,
  and `BackendClient._call_api` (`subnet/validator/backend_client.py`) for calls the
  SDK doesn't wrap. Don't hand-roll HTTP.
- **Validator image dependencies live in `docker/validator/`**
  (`pyproject.toml` + `uv.lock`, installed with
  `uv sync --frozen --group ${RUNTIME_PROFILE}`). Adding a runtime dependency to the
  root `requirements.txt` does not put it in the image.
- **Read from `origin/main`, not a local checkout.** Local clones go stale:
  `git fetch && git show origin/main:<path>`.
