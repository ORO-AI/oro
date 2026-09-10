# Trajectory Room

A dependency-free viewer for ORO generated-environment episodes and ATIF v1.x
trajectories. It is vendored from the ORO environment generator and bundled
into every local test run.

## How miners use it

`docker compose run test --agent-file my_agent.py` writes
`logs/environment-runs/<run-id>/trajectories.html` next to `summary.json`.
Open that file in any browser. It contains the viewer and every episode from
the run, so it can be copied off a VPS as a single file.

The viewer shows the shopper request, assistant messages, tool calls,
observations, simulator and harness events, state changes, verdict checks,
reward, and provenance, with the raw JSON one click away. Use **Open JSON** or
drag `.json` files onto the page to compare episodes from other runs.

ORO artifacts preserve assistant messages emitted through the `message` tool.
They do not contain hidden model reasoning, so the viewer cannot show reasoning
that was not recorded in the source artifact.

## How the report is built

`subnet/local_report.py` concatenates the ES modules in this directory into one
classic script, inlines `styles.css`, and embeds the run's episodes as a JSON
`<script>` block. Browsers refuse `type="module"` scripts from `file://`, which
is why the modules are bundled rather than linked. Keep new code in this
directory free of top-level name collisions across files and of dynamic
`import()`.

## Supported formats

- `oro.environment_episode.v1`
- Released ATIF versions from `ATIF-v1.0` through `ATIF-v1.7`

## Tests

```bash
node --test trajectory-viewer/tests/*.test.mjs
```

The Python suite also runs these when `node` is on `PATH`, and checks that the
generated bundle parses.
