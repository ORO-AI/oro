const RELEASED_ATIF_VERSION = /^ATIF-v1\.[0-7]$/;

function requireAtif(input, sourceName) {
  if (!input || typeof input.schema_version !== "string" || !RELEASED_ATIF_VERSION.test(input.schema_version)) {
    throw new Error(`${sourceName}: expected a released ATIF version from ATIF-v1.0 through ATIF-v1.7`);
  }
  if (!Array.isArray(input.steps)) {
    throw new Error(`${sourceName}: ATIF trajectory is missing a steps array`);
  }
}

function contentForDisplay(content, sourceName, field) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) {
    throw new Error(`${sourceName}: ${field} must be a string or ATIF content array`);
  }
  return content
    .map((part, index) => {
      if (part?.type === "text" && typeof part.text === "string") return part.text;
      if (part?.type === "image" && typeof part.source?.path === "string") {
        return `[Image: ${part.source.path}]`;
      }
      throw new Error(`${sourceName}: ${field}[${index}] is not a supported ATIF content part`);
    })
    .join("\n");
}

function observationValue(result, sourceName, stepIndex, resultIndex) {
  const content = result?.content;
  if (typeof content === "string") {
    try {
      return JSON.parse(content);
    } catch {
      return content;
    }
  }
  if (Array.isArray(content)) {
    return contentForDisplay(content, sourceName, `steps[${stepIndex}].observation.results[${resultIndex}].content`);
  }
  return null;
}

function normalizeToolCalls(step, sourceName, stepIndex) {
  const results = step.observation?.results ?? [];
  return (step.tool_calls ?? []).map((call) => {
    const matchingResults = results
      .map((result, resultIndex) => ({ result, resultIndex }))
      .filter(({ result }) => result.source_call_id === call.tool_call_id);
    const values = matchingResults.map(({ result, resultIndex }) =>
      observationValue(result, sourceName, stepIndex, resultIndex),
    );
    const extras = matchingResults.map(({ result }) => result.extra).filter(Boolean);
    return {
      id: call.tool_call_id,
      name: call.function_name,
      arguments: call.arguments,
      observation: matchingResults.length
        ? {
            value: values.length === 1 ? values[0] : values,
            error: extras.find((extra) => extra?.error)?.error ?? null,
            done: extras.some((extra) => extra?.done === true),
            raw: matchingResults.map(({ result }) => result),
          }
        : null,
    };
  });
}

function unmatchedObservationEvents(step, toolCalls, sourceName, stepIndex) {
  const matchedIds = new Set(toolCalls.map((call) => call.id));
  return (step.observation?.results ?? [])
    .map((result, resultIndex) => ({ result, resultIndex }))
    .filter(({ result }) => !result.source_call_id || !matchedIds.has(result.source_call_id))
    .map(({ result, resultIndex }) => ({
      actor: "environment",
      kind: "observation",
      payload: {
        source_call_id: result.source_call_id ?? null,
        content: observationValue(result, sourceName, stepIndex, resultIndex),
        extra: result.extra ?? null,
      },
    }));
}

export function normalizeAtifTrajectory(input, sourceName = "trajectory.json") {
  requireAtif(input, sourceName);
  const timeline = input.steps.map((step, index) => {
    if (!["system", "user", "agent"].includes(step.source)) {
      throw new Error(`${sourceName}: steps[${index}].source must be system, user, or agent`);
    }
    const toolCalls = normalizeToolCalls(step, sourceName, index);
    const role = step.source === "agent" ? "assistant" : step.source;
    return {
      turn: step.step_id ?? index + 1,
      messages: step.message == null
        ? []
        : [{ role, content: contentForDisplay(step.message, sourceName, `steps[${index}].message`), metadata: {} }],
      toolCalls,
      events: unmatchedObservationEvents(step, toolCalls, sourceName, index),
      reasoning: step.reasoning_content ?? null,
      metrics: step.metrics ?? {},
      latencyMs: step.metrics?.extra?.latency_ms ?? null,
      error: step.extra?.error ?? null,
      raw: step,
    };
  });

  const firstUserMessage = timeline
    .flatMap((turn) => turn.messages)
    .find((message) => message.role === "user")?.content;
  const oroMetadata = input.extra?.oro ?? {};
  const oroOutcome = input.final_metrics?.extra?.oro ?? {};

  return {
    id: input.trajectory_id ?? sourceName,
    sourceName,
    format: input.schema_version.replace("ATIF-", "ATIF "),
    task: {
      id: oroMetadata.task_id ?? input.trajectory_id ?? sourceName,
      family: oroMetadata.family ?? "ATIF",
    },
    query: firstUserMessage ?? "",
    correct: typeof oroOutcome.success === "boolean" ? oroOutcome.success : null,
    reward: oroOutcome.reward ?? null,
    terminalReason: oroOutcome.termination_reason ?? "completed",
    timeline,
    verdict: { ...oroOutcome, atif_metrics: input.final_metrics ?? {} },
    provenance: {
      trajectory_id: input.trajectory_id ?? null,
      session_id: input.session_id ?? null,
      agent_name: input.agent?.name ?? null,
      agent_version: input.agent?.version ?? null,
      schema_version: input.schema_version,
    },
    raw: input,
  };
}
