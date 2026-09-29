import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const shared = new Map();
globalThis.ngx = {
  shared: { oro_models: {
    get: (key) => shared.get(key),
    set: (key, value) => shared.set(key, value),
  } },
};
const script = readFileSync(new URL("../docker/proxy/validate_model.js", import.meta.url), "utf8");
const proxy = (await import(`data:text/javascript,${encodeURIComponent(script)}`)).default;

const chutesId = "Qwen/Qwen3.8-27B-TEE";
const openrouterId = "qwen/qwen3.8-27b";

function request(model, provider, catalogs, modelHeaders = {}) {
  const calls = [];
  const r = {
    method: "POST",
    uri: "/inference/chat/completions",
    requestText: JSON.stringify({ model }),
    headersIn: { Authorization: provider === "openrouter" ? "Bearer sk-or-test" : "Bearer cak_test" },
    headersOut: {},
    variables: { args: "" },
    error: () => {},
    return(status, body) { this.status = status; this.body = body; },
    subrequest(uri, options, callback) {
      calls.push({ uri, options });
      if (uri === "/_inference_grant") callback({ status: 204, headersOut: { "X-ORO-Run-ID": "test" } });
      else if (uri === "/_backend_models") callback({ status: 200, responseText: JSON.stringify(catalogs[options.args.split("=")[1]]), headersOut: modelHeaders });
      else callback({ status: 200, responseText: "{}", headersOut: {} });
    },
  };
  proxy.validate(r);
  return { r, calls };
}

test("OpenRouter translates a cached Chutes alias and preserves native IDs", () => {
  shared.clear();
  const catalogs = {
    openrouter: { models: [openrouterId], aliases: { [chutesId]: openrouterId } },
  };
  const first = request(openrouterId, "openrouter", catalogs);
  const second = request(chutesId, "openrouter", catalogs);
  assert.equal(first.r.status, 200);
  assert.equal(second.r.status, 200);
  assert.equal(JSON.parse(shared.get("state:openrouter")).aliases[chutesId], openrouterId);
  assert.equal(second.calls.filter((call) => call.uri === "/_backend_models").length, 0);
  assert.equal(JSON.parse(second.calls.at(-1).options.body).model, openrouterId);
});

test("Mistral substitution comes from Backend aliases", () => {
  shared.clear();
  const nemo = "unsloth/Mistral-Nemo-Instruct-2407-TEE";
  const small = "mistralai/mistral-small-2603";
  const catalogs = { openrouter: { models: [small], aliases: { [nemo]: small } } };
  const { r, calls } = request(nemo, "openrouter", catalogs);
  assert.equal(r.status, 200);
  assert.equal(JSON.parse(calls.at(-1).options.body).model, small);
});

test("OpenRouter rejects missing or unallowlisted aliases", () => {
  shared.clear();
  const catalogs = { openrouter: { models: [openrouterId], aliases: { [chutesId]: "other/model" } } };
  assert.equal(request(chutesId, "openrouter", catalogs).r.status, 403);
  assert.equal(request("unknown/model", "openrouter", catalogs).r.status, 403);
});

test("a no-store fallback serves native IDs without replacing cached aliases", () => {
  shared.clear();
  const stale = { allowlist: [openrouterId], aliases: { [chutesId]: openrouterId }, expiresAt: Date.now() - 1 };
  shared.set("state:openrouter", JSON.stringify(stale));
  const catalogs = { openrouter: { models: [openrouterId], aliases: {} } };
  const headers = { "Cache-Control": "No-Store" };

  assert.equal(request(chutesId, "openrouter", catalogs, headers).r.status, 200);
  assert.deepEqual(JSON.parse(shared.get("state:openrouter")), stale);

  shared.clear();
  headers["Cache-Control"] = "no-store";
  assert.equal(request(openrouterId, "openrouter", catalogs, headers).r.status, 200);
  assert.equal(shared.size, 0);
});

test("Chutes retains the shopper simulator's Mistral mapping", () => {
  shared.clear();
  const nemo = "unsloth/Mistral-Nemo-Instruct-2407-TEE";
  const catalogs = { chutes: { models: [chutesId, nemo] } };
  assert.equal(request(openrouterId, "chutes", catalogs).r.status, 403);
  assert.equal(request(chutesId, "chutes", catalogs).r.status, 200);
  const { r, calls } = request("mistralai/mistral-small-2603", "chutes", catalogs);
  assert.equal(r.status, 200);
  assert.equal(JSON.parse(calls.at(-1).options.body).model, nemo);
});
