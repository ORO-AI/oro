export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function label(value) {
  return String(value ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function json(value) {
  return escapeHtml(JSON.stringify(value, null, 2));
}

function statusFor(trajectory) {
  if (trajectory.correct === true) return { key: "pass", text: "PASSED" };
  if (trajectory.correct === false) return { key: "fail", text: "FAILED" };
  return { key: "unknown", text: "UNSCORED" };
}

export function summarizeObservation(value) {
  if (value == null) return "No observation";
  if (typeof value !== "object") return String(value);
  if (Array.isArray(value)) return `${value.length} items`;
  if (value.error) return `Error: ${value.error}`;
  if (value.done && value.order_id) return `Order ${value.order_id} completed`;
  if (value.sent === true) return "Message sent";
  if (Array.isArray(value.results)) return `${value.returned ?? value.results.length} results`;
  if (Array.isArray(value.items)) return `${value.items.length} items`;
  if (value.action) return label(value.action);
  if (value.status) return label(value.status);
  const keys = Object.keys(value);
  return keys.length ? keys.slice(0, 3).map(label).join(" · ") : "Empty response";
}

export function renderEpisodeList(trajectories, selectedId, filters = {}) {
  const query = (filters.query ?? "").trim().toLowerCase();
  const status = filters.status ?? "all";
  const visible = trajectories.filter((trajectory) => {
    const itemStatus = statusFor(trajectory).key;
    const haystack = [trajectory.task.id, trajectory.task.family, trajectory.query, trajectory.sourceName]
      .join(" ")
      .toLowerCase();
    return (!query || haystack.includes(query)) && (status === "all" || status === itemStatus);
  });

  if (!visible.length) {
    return '<p class="empty-state">No episodes match this view.</p>';
  }

  return visible
    .map((trajectory) => {
      const state = statusFor(trajectory);
      const selected = trajectory.id === selectedId;
      return `<button class="episode-card${selected ? " is-selected" : ""}" data-episode-id="${escapeHtml(trajectory.id)}"${selected ? ' aria-current="true"' : ""}>
        <span class="episode-card__top"><span class="family-label">${escapeHtml(trajectory.task.family)}</span><span class="status-chip status-chip--${state.key}">${state.text}</span></span>
        <strong>${escapeHtml(trajectory.task.id)}</strong>
        <span class="episode-card__meta">${trajectory.timeline.length} turns · ${escapeHtml(trajectory.terminalReason)}</span>
      </button>`;
    })
    .join("");
}

function renderMessage(message) {
  return `<article class="message message--${escapeHtml(message.role)}">
    <span class="eyebrow">${escapeHtml(message.role)}</span>
    <p>${escapeHtml(message.content)}</p>
  </article>`;
}

function renderToolCall(call) {
  const observation = call.observation;
  return `<article class="tool-pair">
    <section class="tool-action">
      <span class="eyebrow">Action</span>
      <h4>${escapeHtml(call.name)}</h4>
      <details><summary>Tool arguments</summary><pre>${json(call.arguments)}</pre></details>
    </section>
    <section class="tool-observation${observation?.error ? " has-error" : ""}">
      <span class="eyebrow">Observation</span>
      <h4>${escapeHtml(summarizeObservation(observation?.value ?? observation?.error ?? null))}</h4>
      <details><summary>Full observation</summary><pre>${json(observation)}</pre></details>
    </section>
  </article>`;
}

function renderEvent(event) {
  return `<details class="event-card event-card--${escapeHtml(event.kind)}">
    <summary><span>${escapeHtml(event.kind)}</span><small>${escapeHtml(event.actor)}</small></summary>
    <pre>${json(event.payload)}</pre>
  </details>`;
}

export function renderTimeline(trajectory) {
  return trajectory.timeline
    .map(
      (turn) => `<section class="turn" data-turn="${escapeHtml(turn.turn)}">
        <div class="turn-marker"><span>TURN ${escapeHtml(String(turn.turn).padStart(2, "0"))}</span></div>
        <div class="turn-body">
          ${turn.messages.map(renderMessage).join("")}
          ${turn.reasoning ? `<details class="reasoning"><summary>Reasoning</summary><p>${escapeHtml(turn.reasoning)}</p></details>` : ""}
          ${turn.toolCalls.map(renderToolCall).join("")}
          ${turn.events.map(renderEvent).join("")}
          <div class="turn-foot">${turn.latencyMs != null ? `${Number(turn.latencyMs).toFixed(1)} ms` : ""}${turn.error ? ` · ${escapeHtml(turn.error)}` : ""}</div>
        </div>
      </section>`,
    )
    .join("");
}

function renderChecks(checks) {
  const entries = Object.entries(checks ?? {});
  if (!entries.length) return '<p class="muted">No explicit verifier checks recorded.</p>';
  return `<div class="evidence-list">${entries
    .map(([key, value]) => {
      const state = value === true ? "pass" : value === false ? "fail" : "unknown";
      const text = value === true ? "PASS" : value === false ? "FAIL" : "N/A";
      return `<div class="evidence-row"><span>${escapeHtml(label(key).toLowerCase())}</span><strong class="evidence-${state}">${text}</strong></div>`;
    })
    .join("")}</div>`;
}

function renderProvenance(provenance) {
  return `<div class="evidence-list">${Object.entries(provenance ?? {})
    .filter(([, value]) => value != null)
    .map(
      ([key, value]) =>
        `<div class="evidence-row evidence-row--stack"><span>${escapeHtml(label(key).toLowerCase())}</span><code title="${escapeHtml(value)}">${escapeHtml(value)}</code></div>`,
    )
    .join("")}</div>`;
}

export function renderOutcome(trajectory) {
  const state = statusFor(trajectory);
  const familyMetrics = trajectory.verdict?.family_metrics ?? {};
  const booleanFamilyMetrics = Object.fromEntries(
    Object.entries(familyMetrics).filter(
      ([key, value]) =>
        typeof value === "boolean" &&
        /(success|gate|aligned|uptake|commitment|rerank)$/.test(key),
    ),
  );
  return `<section class="outcome-hero outcome-hero--${state.key}">
      <span class="eyebrow">Evaluation outcome</span>
      <strong>${state.text}</strong>
      <p>${escapeHtml(trajectory.verdict?.explanation ?? trajectory.terminalReason)}</p>
      <div class="outcome-numbers"><span>Reward <b>${escapeHtml(trajectory.reward ?? "n/a")}</b></span><span>Terminal <b>${escapeHtml(trajectory.terminalReason)}</b></span></div>
    </section>
    <section class="evidence-section"><h3>Verifier checks</h3>${renderChecks(trajectory.verdict?.checks)}</section>
    <section class="evidence-section"><h3>Family evidence</h3>${renderChecks(booleanFamilyMetrics)}<details class="evidence-details"><summary>View all family metrics</summary><pre>${json(familyMetrics)}</pre></details></section>
    <section class="evidence-section"><h3>Provenance</h3>${renderProvenance(trajectory.provenance)}</section>`;
}
