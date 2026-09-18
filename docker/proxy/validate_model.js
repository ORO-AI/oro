// Inference proxy: validates outgoing requests against the per-provider
// allowlist before forwarding to either Chutes or OpenRouter, dispatched
// by the bearer token's prefix.
//
// The allowlists are fetched from the ORO Backend (`GET
// /v1/public/inference/models?provider=<name>`) via the internal
// `/_backend_models` location and cached per-provider in an nginx
// shared-dict zone (`oro_models`, declared in nginx.conf.template) so all
// worker processes share the same cache. njs module-level vars are
// per-worker, so the previous per-worker cache made every worker cold-start
// independently — 8 workers × 1 fetch each can already exhaust the
// Backend's 100/min global IP rate limit, leaving most workers permanently
// uncached and answering every inference call with 503.
//
// After the cache expires we attempt a refresh; if Backend returns a non-200
// (rate-limited, unreachable, malformed), we keep serving the previous
// allowlist for STALE_GRACE_MS instead of failing closed.
//
// Provider dispatch:
//   - Bearer token starts with "sk-or-" → OpenRouter (allowlist enforced)
//   - Any other token shape (e.g. cak_*) → Chutes (allowlist enforced)
//
// Cross-provider model-name rewriting: agents can use any model identifier
// listed in model_pairs.json regardless of which provider funds the run. If
// the request `model` matches the inactive provider's side of a pair we
// rewrite the body's `model` field to the active provider's side before
// allowlist validation, so the same agent code works on either provider.
// Models with no pair entry pass through unchanged and hit the existing
// allowlist check as before.
//
// Per-request outcome tagging: `$upstream_status` on the parent /inference/
// access log line is always `-` because the location uses `js_content` rather
// than `proxy_pass`. To let CloudWatch metric filters distinguish
// upstream-relayed errors (Chutes/OpenRouter returned 5xx) from proxy-internal
// failures (allowlist unavailable, model not allowed), we stash a short label
// on the request object before every `r.return(...)` and expose it via
// `js_set $proxy_outcome validate_model.outcome` so it lands in the access
// log. See ORO-1159.

import fs from "fs";

// Tag this request with what happened so the access log can record it.
// Called before every r.return(...) in validate(). Read back via outcome(r)
// which is bound to $proxy_outcome by js_set in nginx.conf.template.
function _tag(r, label) {
  r._oroOutcome = label;
}

function outcome(r) {
  return r._oroOutcome || "unknown";
}

// Build the upstream-* label from a subrequest reply status. Keeps the label
// space small enough for CloudWatch term-match patterns: filters key off the
// `upstream-2xx-` / `upstream-4xx-` / `upstream-5xx-` prefix.
function _upstreamLabel(status) {
  var bucket;
  if (status >= 500 && status < 600) bucket = "5xx";
  else if (status >= 400 && status < 500) bucket = "4xx";
  else if (status >= 200 && status < 300) bucket = "ok";
  else bucket = "other";
  return "upstream-" + bucket + "-" + status;
}

function detectProvider(r) {
  var auth = r.headersIn["Authorization"] || "";
  if (auth.indexOf("Bearer sk-or-") === 0) {
    return "openrouter";
  }
  return "chutes";
}

// Lookup tables for both directions, populated once at module init. Each
// nginx worker loads the file independently; reload (`nginx -s reload`)
// picks up edits.
var MODEL_PAIRS_PATH = "/etc/nginx/model_pairs.json";
var _pairsByChutes = {};
var _pairsByOpenrouter = {};
try {
  var _pairsDoc = JSON.parse(fs.readFileSync(MODEL_PAIRS_PATH, "utf8"));
  for (var i = 0; i < _pairsDoc.pairs.length; i++) {
    var p = _pairsDoc.pairs[i];
    _pairsByChutes[p.chutes] = p.openrouter;
    _pairsByOpenrouter[p.openrouter] = p.chutes;
  }
} catch (e) {
  // Don't crash the worker — same-provider names still work via the
  // existing allowlist check. Surface the error so it's noticeable.
  ngx.log(ngx.ERR, "model_pairs load failed: " + e.message);
}

// Returns the request model rewritten for `activeProvider`, or null if no
// rewrite is needed (already on the active side, or unknown — let allowlist
// validation handle it).
function rewriteModelFor(activeProvider, requested) {
  if (activeProvider === "chutes") {
    if (_pairsByChutes[requested] !== undefined) return null;
    if (_pairsByOpenrouter[requested] !== undefined) return _pairsByOpenrouter[requested];
  } else if (activeProvider === "openrouter") {
    if (_pairsByOpenrouter[requested] !== undefined) return null;
    if (_pairsByChutes[requested] !== undefined) return _pairsByChutes[requested];
  }
  return null;
}

var CACHE_TTL_MS = 15 * 60 * 1000;
// Window beyond CACHE_TTL_MS where we still serve the cached list if a
// refresh fails. After this we give up and fail closed.
var STALE_GRACE_MS = 60 * 60 * 1000;

var ZONE = "oro_models";

function _stateKey(provider) {
  return "state:" + provider;
}

function _readState(provider) {
  var raw = ngx.shared[ZONE].get(_stateKey(provider));
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

function _writeState(provider, allowlist, expiresAt) {
  ngx.shared[ZONE].set(
    _stateKey(provider),
    JSON.stringify({ allowlist: allowlist, expiresAt: expiresAt })
  );
}

function getAllowlist(r, provider, callback) {
  var state = _readState(provider);
  if (state && state.allowlist && Date.now() < state.expiresAt) {
    callback(state.allowlist);
    return;
  }

  r.subrequest(
    "/_backend_models",
    { method: "GET", args: "provider=" + provider },
    function (reply) {
      if (reply.status === 200) {
        try {
          var data = JSON.parse(reply.responseText);
          if (data && Array.isArray(data.models) && data.models.length > 0) {
            _writeState(provider, data.models, Date.now() + CACHE_TTL_MS);
            callback(data.models);
            return;
          }
          r.error(
            "Backend models response missing or empty 'models' array for provider=" + provider
          );
        } catch (e) {
          r.error("Backend models JSON parse failed: " + e.message);
        }
      } else {
        r.error("Backend models fetch returned status " + reply.status + " for provider=" + provider);
      }

      state = _readState(provider);
      if (state && state.allowlist && Date.now() < state.expiresAt + STALE_GRACE_MS) {
        var graceLeft = state.expiresAt + STALE_GRACE_MS - Date.now();
        if (graceLeft < STALE_GRACE_MS) {
          r.error(
            "Serving stale " + provider + " allowlist after fetch failure (" +
              (graceLeft / 1000).toFixed(0) +
              "s grace remaining)"
          );
        }
        callback(state.allowlist);
        return;
      }

      callback(null);
    }
  );
}

function validate(r) {
  var provider = detectProvider(r);
  var upstreamLocation = provider === "openrouter" ? "/_openrouter_proxy/" : "/_chutes_proxy/";

  if (r.method !== "POST") {
    var passUri = upstreamLocation + r.uri.replace(/^\/inference\//, "");
    r.subrequest(passUri, { method: r.method, args: r.variables.args || "" }, function (reply) {
      _tag(r, _upstreamLabel(reply.status));
      for (var h in reply.headersOut) {
        r.headersOut[h] = reply.headersOut[h];
      }
      r.return(reply.status, reply.responseText);
    });
    return;
  }

  var body = r.requestText;

  if (!body) {
    _tag(r, "internal-bad-request");
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Missing or unreadable request body" }));
    return;
  }

  var parsed;
  try {
    parsed = JSON.parse(body);
  } catch (e) {
    _tag(r, "internal-bad-request");
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Invalid JSON in request body" }));
    return;
  }

  if (!parsed.model) {
    _tag(r, "internal-bad-request");
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Missing 'model' field in request body" }));
    return;
  }

  if (parsed.stream === true) {
    _tag(r, "internal-bad-request");
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Streaming is not supported through the proxy" }));
    return;
  }

  // Enforce the allowlist across the whole request, not just the scalar `model`.
  // OpenRouter also honours `models[]` (candidate list), `provider` preferences,
  // `route`, `transforms`, and `preset`; strip them so every request runs on the
  // single validated model rather than one steered by these fields. Log any that
  // were present — the proxy does not otherwise record them.
  var stripped = [];
  ["models", "provider", "route", "transforms", "preset"].forEach(function (f) {
    if (parsed[f] !== undefined) {
      delete parsed[f];
      stripped.push(f);
    }
  });
  if (stripped.length > 0) {
    r.error("stripped inference routing fields: " + stripped.join(","));
  }

  // OpenRouter only returns `usage.cost` (USD) on the response when the
  // request body sets `usage.include=true`. Force it on so per-call cost
  // lands in every response — both InferenceStats (per-episode budget
  // tracking) and miner agent code can then read `resp.usage.cost`
  // deterministically. Chutes ignores unknown top-level fields but skip
  // there to keep the outbound body untouched.
  var usageInjected = false;
  if (provider === "openrouter") {
    parsed.usage = { include: true };
    usageInjected = true;
  }

  // Proxy-authored fallback for the shopper user-simulator. The validator's
  // user-simulator runs on Mistral Small, which has been rate-limiting (429) on
  // OpenRouter and hard-failing episodes. Re-add an OpenRouter `models[]`
  // candidate list so OpenRouter itself fails over to the Qwen instruct model on
  // a primary 429/5xx. The strip above (#260) removes a *client-supplied*
  // models[] to stop a caller steering onto an unvalidated model; this list is
  // proxy-hardcoded to two known models and is not client-controllable, so that
  // threat model does not apply. The scalar `parsed.model` stays Mistral, so the
  // allowlist check below still validates the primary. The fallback entry is
  // intentionally NOT allowlist-validated (proxy-authored exception). OpenRouter
  // only. TODO(ORO): replay-parity + record served model before relying on this
  // for scoring — cross-validator fallback timing adds score variance.
  var fallbackInjected = false;
  if (provider === "openrouter" && parsed.model === "mistralai/mistral-small-2603") {
    parsed.models = [
      "mistralai/mistral-small-2603",
      "qwen/qwen3-30b-a3b-instruct-2507",
    ];
    fallbackInjected = true;
  }

  var rewritten = rewriteModelFor(provider, parsed.model);
  if (rewritten !== null) {
    parsed.model = rewritten;
  }
  var forwardBody =
    rewritten !== null || stripped.length > 0 || usageInjected || fallbackInjected
      ? JSON.stringify(parsed)
      : body;

  getAllowlist(r, provider, function (allowed) {
    if (!allowed) {
      _tag(r, "internal-allowlist-unavailable");
      r.headersOut["Content-Type"] = "application/json";
      r.return(503, JSON.stringify({ error: "Inference allowlist unavailable" }));
      return;
    }

    if (allowed.indexOf(parsed.model) === -1) {
      _tag(r, "internal-model-not-allowed");
      r.error("Model not allowed for " + provider + ": " + parsed.model);
      r.headersOut["Content-Type"] = "application/json";
      r.return(
        403,
        JSON.stringify({
          error: "Model '" + parsed.model + "' is not allowed for provider " + provider,
          allowed_models: allowed,
        })
      );
      return;
    }

    var uri = upstreamLocation + r.uri.replace(/^\/inference\//, "");
    r.subrequest(
      uri,
      { method: "POST", body: forwardBody, args: r.variables.args || "" },
      function (reply) {
        _tag(r, _upstreamLabel(reply.status));
        // Belt-and-suspenders forensic trail for ORO-2191. The Python side
        // (SimulatorCompletion / ProxyClient.last_error) also captures the
        // body, but a message posted here survives even for callers that
        // don't read last_error yet — grep the proxy container logs on any
        // env_error incident for the actual upstream reason.
        if (reply.status >= 400) {
          var body = reply.responseText || "";
          r.error(
            "upstream " +
              reply.status +
              " on " +
              r.uri +
              " body: " +
              body.substring(0, 800)
          );
        }
        for (var h in reply.headersOut) {
          r.headersOut[h] = reply.headersOut[h];
        }
        r.return(reply.status, reply.responseText);
      }
    );
  });
}

export default { validate: validate, outcome: outcome };
