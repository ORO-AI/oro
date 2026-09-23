import assert from "node:assert/strict";
import test from "node:test";

import {
  escapeHtml,
  renderEpisodeList,
  renderOutcome,
  renderTimeline,
  summarizeObservation,
} from "../render.js";

const trajectory = {
  id: "run:task",
  sourceName: "sample.json",
  format: "ORO episode v1",
  task: { id: "TF5-ranking-200000", family: "ranking" },
  query: "Find <quiet> keyboards",
  correct: false,
  reward: 0,
  terminalReason: "step_limit",
  timeline: [
    {
      turn: 1,
      messages: [{ role: "assistant", content: "I found <three> options." }],
      reasoning: null,
      toolCalls: [
        {
          id: "call-1",
          name: "search",
          arguments: { query: "quiet & tactile" },
          observation: { value: { returned: 3, results: [{ title: "Board A" }] }, error: null },
        },
      ],
      events: [{ actor: "harness", kind: "stockout", payload: { sku: "one" } }],
      latencyMs: 12.3,
      metrics: {},
    },
  ],
  verdict: {
    explanation: "no order placed",
    checks: { within_budget: true, final_in_stock: false },
    family_metrics: { ranking_aligned_success: false },
  },
  provenance: { runtime_version: "0.2.0", verifier_version: "0.3.0" },
  raw: {},
};

test("escapes untrusted text and JSON", () => {
  assert.equal(escapeHtml('<script>alert("x")</script>'), "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;");
  const html = renderTimeline(trajectory);
  assert.ok(html.includes("I found &lt;three&gt; options."));
  assert.ok(!html.includes("<three>"));
  assert.ok(html.includes("quiet &amp; tactile"));
});

test("escapes untrusted turn labels", () => {
  const unsafe = {
    ...trajectory,
    timeline: [{ ...trajectory.timeline[0], turn: '<img src=x onerror="alert(1)">' }],
  };
  const html = renderTimeline(unsafe);
  assert.ok(html.includes("TURN &lt;img src=x onerror=&quot;alert(1)&quot;&gt;"));
  assert.ok(!html.includes('<img src=x onerror="alert(1)">'));
});

test("renders episode state and keeps selected item identifiable", () => {
  const passing = { ...trajectory, id: "pass", correct: true, task: { id: "TF2", family: "retrieval" } };
  const html = renderEpisodeList([trajectory, passing], trajectory.id, { query: "", status: "all" });
  assert.ok(html.includes("FAILED"));
  assert.ok(html.includes("PASSED"));
  assert.ok(html.includes('aria-current="true"'));
});

test("renders chronological turns with calls, observations, and events", () => {
  const html = renderTimeline(trajectory);
  assert.match(html, /TURN 01/);
  assert.match(html, /search/);
  assert.match(html, /Tool arguments/);
  assert.match(html, /3 results/);
  assert.match(html, /stockout/);
});

test("summarizes common observation shapes", () => {
  assert.equal(summarizeObservation({ returned: 3, results: [1, 2, 3] }), "3 results");
  assert.equal(summarizeObservation({ sent: true }), "Message sent");
  assert.equal(summarizeObservation({ done: true, order_id: "order-1" }), "Order order-1 completed");
});

test("renders failed checks and provenance evidence", () => {
  const html = renderOutcome(trajectory);
  assert.match(html, /final in stock/);
  assert.match(html, /FAIL/);
  assert.match(html, /runtime version/);
  assert.match(html, /0\.2\.0/);
  assert.match(html, /no order placed/);
  assert.match(html, /ranking aligned success/);
});

test("summarizes non-object observations as themselves", () => {
  assert.equal(summarizeObservation("timeout waiting for tool"), "timeout waiting for tool");
  assert.equal(summarizeObservation(["a", "b"]), "2 items");
  assert.equal(summarizeObservation(42), "42");
});
