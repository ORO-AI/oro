import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeOroEpisode } from "../adapters/oro.js";
import { normalizeAtifTrajectory } from "../adapters/atif.js";
import { normalizeTrajectory } from "../adapters/index.js";

const sampleDirectory = new URL("./fixtures/oro-episodes/", import.meta.url);

async function loadSamples() {
  const filenames = (await readdir(sampleDirectory)).filter((name) => name.endsWith(".json"));
  return Promise.all(
    filenames.map(async (filename) => ({
      filename,
      value: JSON.parse(await readFile(new URL(filename, sampleDirectory), "utf8")),
    })),
  );
}

test("normalizes the bundled ORO episode fixtures", async () => {
  const samples = await loadSamples();
  const trajectories = samples.map(({ filename, value }) => normalizeOroEpisode(value, filename));

  assert.equal(trajectories.length, 2);
  assert.equal(trajectories.filter((trajectory) => trajectory.correct).length, 0);
  assert.deepEqual(
    new Set(trajectories.map((trajectory) => trajectory.task.family)),
    new Set(["recovery", "ranking"]),
  );
  assert.ok(trajectories.every((trajectory) => trajectory.query.length > 20));
  assert.ok(trajectories.every((trajectory) => trajectory.verdict.explanation));
});

test("pairs each ORO turn's calls and observations without duplicating ledger actions", async () => {
  const samples = await loadSamples();
  const ranking = samples.find(({ value }) => value.episode.family === "ranking");
  const trajectory = normalizeOroEpisode(ranking.value, ranking.filename);

  assert.equal(trajectory.timeline.length, ranking.value.episode.call_trace.length);
  assert.deepEqual(
    trajectory.timeline.map((turn) => turn.turn),
    Array.from({ length: 30 }, (_, index) => index + 1),
  );
  assert.equal(trajectory.timeline[0].toolCalls[0].name, "search");
  assert.equal(trajectory.timeline[0].toolCalls[0].observation.value.returned, 10);
  assert.ok(trajectory.timeline[20].events.some((event) => event.kind === "harness_event"));
  assert.ok(trajectory.timeline[29].messages.some((message) => message.role === "assistant"));
  assert.ok(trajectory.timeline[29].messages.some((message) => message.role === "user"));
  assert.equal(trajectory.terminalReason, "step_limit");
});

test("normalizes valid ATIF v1.7 messages, same-step results, and metrics", async () => {
  const value = JSON.parse(
    await readFile(new URL("./fixtures/atif-basic.json", import.meta.url), "utf8"),
  );
  const trajectory = normalizeAtifTrajectory(value, "atif-basic.json");

  assert.equal(trajectory.id, "atif-trajectory-1");
  assert.equal(trajectory.format, "ATIF v1.7");
  assert.equal(trajectory.query, "Find a quiet mechanical keyboard under $100.");
  assert.equal(trajectory.timeline.length, 2);
  assert.equal(trajectory.timeline[1].reasoning, "Start broad, then inspect the best candidates.");
  assert.equal(
    trajectory.timeline[1].messages[0].content,
    "I will compare switches and price.\n[Image: screenshots/results.png]",
  );
  assert.equal(trajectory.timeline[1].toolCalls[0].name, "search");
  assert.equal(trajectory.timeline[1].toolCalls[0].observation.value.results[0].price, 89);
  assert.equal(trajectory.timeline[1].latencyMs, 125);
  assert.equal(trajectory.timeline[1].metrics.prompt_tokens, 30);
  assert.equal(trajectory.reward, 0.75);
  assert.equal(trajectory.correct, true);
  assert.equal(trajectory.terminalReason, "completed");
  assert.deepEqual(trajectory.task, { id: "keyboard-search", family: "retrieval" });
});

test("uses trajectory_id when session ids are shared", () => {
  const base = {
    schema_version: "ATIF-v1.7",
    session_id: "shared-session",
    agent: { name: "Agent", version: "1" },
    steps: [{ step_id: 1, source: "user", message: "hello" }],
  };
  assert.equal(normalizeAtifTrajectory({ ...base, trajectory_id: "trajectory-a" }).id, "trajectory-a");
  assert.equal(normalizeAtifTrajectory({ ...base, trajectory_id: "trajectory-b" }).id, "trajectory-b");
});

test("renders ATIF observation content arrays without object coercion", () => {
  const trajectory = normalizeAtifTrajectory({
    schema_version: "ATIF-v1.7",
    trajectory_id: "multimodal-observation",
    agent: { name: "Agent", version: "1" },
    steps: [
      {
        step_id: 1,
        source: "agent",
        message: "Inspecting the result.",
        tool_calls: [{ tool_call_id: "call-1", function_name: "view", arguments: {} }],
        observation: {
          results: [
            {
              source_call_id: "call-1",
              content: [
                { type: "text", text: "Product screenshot" },
                { type: "image", source: { media_type: "image/png", path: "result.png" } },
              ],
            },
          ],
        },
      },
    ],
  });

  assert.equal(
    trajectory.timeline[0].toolCalls[0].observation.value,
    "Product screenshot\n[Image: result.png]",
  );
});

test("dispatches supported formats and names malformed inputs", async () => {
  const [{ filename, value }] = await loadSamples();
  assert.equal(normalizeTrajectory(value, filename).format, "ORO episode v1");
  assert.throws(
    () => normalizeTrajectory({ schema_version: "ATIF-v2.0" }, "future.json"),
    /future\.json: unsupported schema_version ATIF-v2\.0/,
  );
  assert.throws(
    () => normalizeTrajectory({}, "empty.json"),
    /empty\.json: missing schema_version/,
  );
  assert.throws(
    () => normalizeTrajectory({ schema_version: "ATIF-v1.8" }, "future-atif.json"),
    /future-atif\.json: unsupported schema_version ATIF-v1\.8/,
  );
});

test("splits the shopper goal from the task rules the runtime appends", async () => {
  const [{ value }] = await loadSamples();
  const episode = structuredClone(value);
  episode.episode.bootstrap.policy_view.query =
    'find an in-stock listing under 693.00 PHP, and buy one\n\nTask rules:\nYou are a shopping agent. Search the catalog.\nCategory: Groceries';

  const trajectory = normalizeOroEpisode(episode, "task-rules.json");

  assert.equal(trajectory.query, "find an in-stock listing under 693.00 PHP, and buy one");
  assert.match(trajectory.taskRules, /^You are a shopping agent\./);
  assert.match(trajectory.taskRules, /Category: Groceries$/);
  assert.ok(!trajectory.query.includes("Task rules"));
});

test("keeps the whole query when the runtime appends no task rules", async () => {
  const [{ value }] = await loadSamples();
  const episode = structuredClone(value);
  episode.episode.bootstrap.policy_view.query = "Want a gaming monitor around 12599 PHP.";

  const trajectory = normalizeOroEpisode(episode, "plain.json");

  assert.equal(trajectory.query, "Want a gaming monitor around 12599 PHP.");
  assert.equal(trajectory.taskRules, "");
});
