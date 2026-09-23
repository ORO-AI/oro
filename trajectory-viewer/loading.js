import { normalizeTrajectory } from "./adapters/index.js";

export function validateFilename(filename) {
  if (!filename.toLowerCase().endsWith(".json")) {
    throw new Error(`${filename}: expected a .json file`);
  }
}

export function parseTrajectoryText(text, sourceName) {
  let value;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`${sourceName}: invalid JSON (${error.message})`);
  }
  return normalizeTrajectory(value, sourceName);
}

export function addOrReplaceTrajectory(trajectories, trajectory) {
  const next = [...trajectories];
  const existing = next.findIndex((item) => item.id === trajectory.id);
  if (existing >= 0) next.splice(existing, 1, trajectory);
  else next.push(trajectory);
  return next;
}
