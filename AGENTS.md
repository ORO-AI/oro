# oro

Public validator and miner code for the ORO Bittensor subnet. Start with `README.md`,
`docs/validator-guide.md` and `docs/miner-guide.md`.

## Rules

- **This repository is public.** Commits, branch names, PR titles and bodies, and
  code comments must not contain internal ticket IDs, internal hostnames or URLs,
  credentials, or unreleased evaluation content (agent code, task suites). Operator
  procedures are maintained privately; don't add them here.
- **Don't import a metrics-bearing module under both `validator.` and
  `subnet.validator.` in one process.** Pytest puts `subnet` on the path
  (`pyproject.toml`), so both names resolve, and loading a module twice registers
  its Prometheus metrics twice (duplicate-timeseries error). In tests, import a
  module under the same path the code under test uses, so class identities match
  (e.g. `subnet.validator.env_pack_loader` for `subnet/local_generated_validator.py`).
- **The validator talks to the backend only through `oro_sdk`.** Synchronous calls
  go through `BackendClient._call_api` (`subnet/validator/backend_client.py`), which
  wraps an SDK `sync_detailed` function. Async environment calls (pack fetch,
  episode emit) go through `call_environment_api`
  (`subnet/validator/env_backend.py`) with the SDK's `asyncio_detailed` function,
  which keeps their bounded retries and doesn't block the event loop. For an
  endpoint the SDK doesn't cover, add the operation to the SDK first. Don't
  hand-roll HTTP.
- **Validator image dependencies live in `docker/validator/`**
  (`pyproject.toml` + `uv.lock`), installed with
  `uv sync --frozen --no-install-project --no-default-groups --group ${RUNTIME_PROFILE}`.
  Adding a runtime dependency to the root `requirements.txt` does not put it in the image.
- **Don't trust a stale `main` checkout.** For how the default branch behaves now,
  `git fetch` and read `origin/main` (`git show origin/main:<path>`), because a
  local `main` may be behind. For the code you're changing, read your branch's
  working tree. There, `origin/main` is the base, not the current state.
