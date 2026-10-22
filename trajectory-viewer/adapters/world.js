// A world-pack episode (`oro.world_episode.v1`, runtime contract 8): the runtime's ledger records each
// numbered step's action and observation, the shopper messages and replies, and the finish.
function requireWorldEpisode(input, sourceName) {
  if (!input || input.schema_version !== "oro.world_episode.v1" || !input.episode) {
    throw new Error(`${sourceName}: expected schema_version oro.world_episode.v1 with an episode object`);
  }
  return input.episode;
}

function worldTimeline(records) {
  const turns = [];
  let current = null;
  for (const record of records) {
    if (record.kind === "action") {
      current = {
        turn: record.step,
        messages: [],
        toolCalls: [{ id: String(record.step), name: record.tool ?? "unknown", arguments: record.args ?? {}, observation: null }],
        events: [],
        latencyMs: null,
        error: null,
        stateBefore: null,
        stateAfter: null,
        raw: record,
      };
      turns.push(current);
    } else if (record.kind === "observation" && current && record.step === current.turn) {
      const value = record.observation ?? null;
      current.toolCalls[0].observation = { value, error: value?.error ?? null, done: false };
    } else if ((record.kind === "message" || record.kind === "reply") && current) {
      const content = typeof record.text === "string" ? record.text : "";
      if (content.trim()) current.messages.push({ role: record.kind === "message" ? "assistant" : "user", content, metadata: record });
    } else if (current && !["open", "start", "finish"].includes(record.kind)) {
      current.events.push({ actor: "environment", kind: record.kind, payload: record, sequence: null, stateHash: null });
    }
  }
  return turns;
}

export function normalizeWorldEpisode(input, sourceName = "trajectory.json") {
  const episode = requireWorldEpisode(input, sourceName);
  const records = Array.isArray(episode.ledger?.episode) ? episode.ledger.episode : [];
  const verdict = episode.verdict ?? null;
  const finish = records.findLast?.((r) => r.kind === "finish") ?? records.filter((r) => r.kind === "finish").pop();
  return {
    id: episode.evaluation_run_id ? `${episode.evaluation_run_id}:${episode.task_id}` : sourceName,
    sourceName,
    format: "ORO world episode v1",
    task: { id: episode.task_id ?? "Unknown task", family: episode.world_id ?? "world" },
    query: episode.request_text ?? "",
    taskRules: "",
    correct: verdict ? verdict.score === 1 : null,
    reward: verdict?.score ?? null,
    terminalReason: episode.terminal_reason ?? finish?.reason ?? episode.outcome ?? "unknown",
    timeline: worldTimeline(records),
    verdict: verdict ?? {},
    failure: episode.error ? { message: episode.error, outcome: episode.outcome } : null,
    provenance: {
      agent_version_id: episode.agent_version_id ?? null,
      evaluation_run_id: episode.evaluation_run_id ?? null,
      session_id: episode.session_id ?? null,
      pack_sha256: episode.pack_sha256 ?? null,
    },
    raw: input,
  };
}
