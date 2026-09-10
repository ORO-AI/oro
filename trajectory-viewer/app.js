import { addOrReplaceTrajectory, parseTrajectoryText, validateFilename } from "./loading.js";
import { normalizeTrajectory } from "./adapters/index.js";
import { escapeHtml, renderEpisodeList, renderOutcome, renderTimeline } from "./render.js";

const state = { trajectories: [], selectedId: null, filters: { query: "", status: "all" } };
const elements = {
  count: document.querySelector("#episode-count"),
  dropOverlay: document.querySelector("#drop-overlay"),
  fileInput: document.querySelector("#file-input"),
  header: document.querySelector("#trajectory-header"),
  list: document.querySelector("#episode-list"),
  outcome: document.querySelector("#outcome"),
  raw: document.querySelector("#raw-json"),
  search: document.querySelector("#episode-search"),
  status: document.querySelector("#load-status"),
  timeline: document.querySelector("#timeline"),
};

function replaceTrajectory(trajectory) {
  state.trajectories = addOrReplaceTrajectory(state.trajectories, trajectory);
  if (!state.selectedId) state.selectedId = trajectory.id;
}

function selectedTrajectory() {
  return state.trajectories.find((trajectory) => trajectory.id === state.selectedId) ?? null;
}

function renderHeader(trajectory) {
  const rules = trajectory.taskRules
    ? `<details class="task-rules"><summary>Task rules given to the agent</summary><p>${escapeHtml(trajectory.taskRules)}</p></details>`
    : "";
  return `<p class="kicker">${escapeHtml(trajectory.format)} / ${escapeHtml(trajectory.task.family)}</p>
    <h2>${escapeHtml(trajectory.task.id)}</h2>
    <blockquote>${escapeHtml(trajectory.query)}</blockquote>
    ${rules}
    <div class="trajectory-meta"><span>${trajectory.timeline.length} turns</span><span>${escapeHtml(trajectory.sourceName)}</span><span>${escapeHtml(trajectory.terminalReason)}</span></div>`;
}

function render() {
  elements.count.textContent = String(state.trajectories.length);
  elements.list.innerHTML = renderEpisodeList(state.trajectories, state.selectedId, state.filters);
  for (const button of elements.list.querySelectorAll("[data-episode-id]")) {
    button.addEventListener("click", () => {
      state.selectedId = button.dataset.episodeId;
      render();
      document.querySelector(".trajectory-panel")?.scrollTo({ top: 0 });
    });
  }

  const trajectory = selectedTrajectory();
  if (!trajectory) {
    elements.header.innerHTML = "";
    elements.timeline.innerHTML = document.querySelector("#empty-template").innerHTML;
    elements.outcome.innerHTML = "";
    elements.raw.textContent = "";
    return;
  }
  elements.header.innerHTML = renderHeader(trajectory);
  elements.timeline.innerHTML = renderTimeline(trajectory);
  elements.outcome.innerHTML = renderOutcome(trajectory);
  elements.raw.textContent = JSON.stringify(trajectory.raw, null, 2);
}

async function ingestText(text, sourceName) {
  replaceTrajectory(parseTrajectoryText(text, sourceName));
}

async function ingestFiles(files) {
  const errors = [];
  for (const file of files) {
    try {
      validateFilename(file.name);
      await ingestText(await file.text(), file.name);
    } catch (error) {
      errors.push(error.message);
    }
  }
  elements.status.textContent = loadStatus(files.length - errors.length, "local file", errors);
  render();
}

function loadStatus(loaded, noun, errors, suffix = "") {
  const plural = loaded === 1 ? noun : `${noun}s`;
  return errors.length
    ? `${loaded} loaded${suffix}. ${errors.join(" ")}`
    : `${loaded} ${plural}${suffix}`;
}

function loadEmbeddedEpisodes() {
  const node = document.querySelector("#oro-episodes");
  if (!node) return false;
  const payload = JSON.parse(node.textContent);
  const errors = [];
  for (const entry of payload.episodes ?? []) {
    try {
      replaceTrajectory(normalizeTrajectory(entry.value, entry.name));
    } catch (error) {
      errors.push(error.message);
    }
  }
  const source = payload.run_id ? ` from ${payload.run_id}` : "";
  elements.status.textContent = loadStatus(state.trajectories.length, "episode", errors, source);
  render();
  return true;
}

elements.fileInput.addEventListener("change", () => ingestFiles([...elements.fileInput.files]));
elements.search.addEventListener("input", () => {
  state.filters.query = elements.search.value;
  render();
});
for (const filter of document.querySelectorAll(".filter")) {
  filter.addEventListener("click", () => {
    state.filters.status = filter.dataset.status;
    for (const button of document.querySelectorAll(".filter")) button.classList.toggle("is-active", button === filter);
    render();
  });
}

let dragDepth = 0;
document.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth += 1;
  elements.dropOverlay.classList.add("is-visible");
});
document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dragDepth -= 1;
  if (dragDepth <= 0) elements.dropOverlay.classList.remove("is-visible");
});
document.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  elements.dropOverlay.classList.remove("is-visible");
  ingestFiles([...event.dataTransfer.files]);
});

if (!loadEmbeddedEpisodes()) {
  elements.status.textContent = "Open a trajectory JSON to begin";
  render();
}
