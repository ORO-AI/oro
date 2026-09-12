import { normalizeAtifTrajectory } from "./atif.js";
import { normalizeOroEpisode } from "./oro.js";

export function normalizeTrajectory(input, sourceName = "trajectory.json") {
  if (!input || typeof input !== "object" || !input.schema_version) {
    throw new Error(`${sourceName}: missing schema_version`);
  }
  if (input.schema_version === "oro.environment_episode.v1") {
    return normalizeOroEpisode(input, sourceName);
  }
  if (/^ATIF-v1\.[0-7]$/.test(input.schema_version)) {
    return normalizeAtifTrajectory(input, sourceName);
  }
  throw new Error(`${sourceName}: unsupported schema_version ${input.schema_version}`);
}
