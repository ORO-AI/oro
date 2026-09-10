function requireEpisode(input, sourceName) {
  if (!input || input.schema_version !== "oro.environment_episode.v1" || !input.episode) {
    throw new Error(`${sourceName}: expected schema_version oro.environment_episode.v1 with an episode object`);
  }
  return input.episode;
}

function messagesForTurn(ledger, turn) {
  return ledger
    .filter((entry) => entry.turn === turn && ["model_message", "user_message"].includes(entry.kind))
    .map((entry) => ({
      role: entry.kind === "model_message" ? "assistant" : "user",
      content: entry.payload?.content ?? "",
      metadata: entry.payload,
    }))
    .filter((message) => message.content.trim().length > 0);
}

function eventsForTurn(ledger, turn) {
  return ledger
    .filter(
      (entry) =>
        entry.turn === turn &&
        !["model_action", "model_message", "observation", "user_message"].includes(entry.kind),
    )
    .map((entry) => ({
      actor: entry.actor,
      kind: entry.kind,
      payload: entry.payload,
      sequence: entry.seq,
      stateHash: entry.state_hash,
    }));
}

function pairCalls(trace) {
  const observations = new Map(
    (trace.response?.calls ?? []).map((call) => [
      call.call_id,
      call.observation
        ? {
            value: call.observation.observation ?? null,
            error: call.observation.error ?? null,
            done: call.observation.done ?? false,
          }
        : null,
    ]),
  );
  return (trace.request?.calls ?? []).map((call) => ({
    id: call.call_id,
    name: call.action?.name ?? "unknown",
    arguments: call.action?.args ?? {},
    observation: observations.get(call.call_id) ?? null,
  }));
}

export function normalizeOroEpisode(input, sourceName = "trajectory.json") {
  const episode = requireEpisode(input, sourceName);
  const ledger = Array.isArray(episode.ledger) ? episode.ledger : [];
  const callTrace = Array.isArray(episode.call_trace) ? episode.call_trace : [];

  return {
    id: episode.evaluation_run_id ? `${episode.evaluation_run_id}:${episode.task_id}` : sourceName,
    sourceName,
    format: "ORO episode v1",
    task: {
      id: episode.task_id ?? "Unknown task",
      family: episode.family ?? "unknown",
    },
    query: episode.bootstrap?.policy_view?.query ?? "",
    correct: typeof episode.verdict?.correct === "boolean" ? episode.verdict.correct : null,
    reward: episode.verdict?.paid_reward ?? null,
    terminalReason: episode.terminal_reason ?? episode.outcome ?? "unknown",
    timeline: callTrace.map((trace, index) => {
      const turn = trace.request?.turn ?? trace.response?.turn ?? index + 1;
      const events = eventsForTurn(ledger, turn);
      if (trace.state_hash_before && trace.state_hash_after && trace.state_hash_before !== trace.state_hash_after) {
        events.push({
          actor: "environment",
          kind: "state_change",
          payload: { before: trace.state_hash_before, after: trace.state_hash_after },
        });
      }
      return {
        turn,
        messages: messagesForTurn(ledger, turn),
        toolCalls: pairCalls(trace),
        events,
        latencyMs: trace.latency_ms ?? trace.timing_ms?.total ?? null,
        error: trace.error ?? null,
        stateBefore: trace.state_hash_before ?? null,
        stateAfter: trace.state_hash_after ?? null,
        raw: trace,
      };
    }),
    verdict: episode.verdict ?? {},
    provenance: {
      ...(episode.provenance ?? {}),
      execution_contract_id: input.execution_contract_id ?? null,
      agent_version_id: episode.agent_version_id ?? null,
      evaluation_run_id: episode.evaluation_run_id ?? null,
      session_id: episode.session_id ?? episode.bootstrap?.session_id ?? null,
    },
    raw: input,
  };
}
