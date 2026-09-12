import assert from "node:assert/strict";
import test from "node:test";

import { addOrReplaceTrajectory, parseTrajectoryText, validateFilename } from "../loading.js";

test("validates local JSON filenames", () => {
  assert.doesNotThrow(() => validateFilename("episode.JSON"));
  assert.throws(() => validateFilename("notes.txt"), /notes\.txt: expected a \.json file/);
});

test("reports filename-specific JSON and schema errors", () => {
  assert.throws(() => parseTrajectoryText("{", "broken.json"), /broken\.json: invalid JSON/);
  assert.throws(() => parseTrajectoryText("{}", "empty.json"), /empty\.json: missing schema_version/);
});

test("replaces a duplicate trajectory without changing its position", () => {
  const first = { id: "one", reward: 0 };
  const second = { id: "two", reward: 0 };
  const replacement = { id: "one", reward: 1 };
  const result = addOrReplaceTrajectory([first, second], replacement);

  assert.deepEqual(result, [replacement, second]);
});
