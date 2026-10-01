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
// A no-store OpenRouter response has the curated models but only fallback
// aliases; use the last good aliases for that request without caching it.
//
// Provider dispatch:
//   - Bearer token starts with "sk-or-" → OpenRouter (allowlist enforced)
//   - Any other token shape (e.g. cak_*) → Chutes (allowlist enforced)
//
// On OpenRouter runs, Backend may supply aliases from Chutes IDs to
// OpenRouter IDs. The translated ID must still pass the active allowlist.
//
// Per-request outcome tagging: `$upstream_status` on the parent /inference/
// access log line is always `-` because the location uses `js_content` rather
// than `proxy_pass`. To let CloudWatch metric filters distinguish
// upstream-relayed errors (Chutes/OpenRouter returned 5xx) from proxy-internal
// failures (allowlist unavailable, model not allowed), we stash a short label
// on the request object before every `r.return(...)` and expose it via
// `js_set $proxy_outcome validate_model.outcome` so it lands in the access
// log. See ORO-1159.

// Tag this request with what happened so the access log can record it.
// Called before every r.return(...) in validate(). Read back via outcome(r)
// which is bound to $proxy_outcome by js_set in nginx.conf.template.
function _tag(r, label) {
  r._oroOutcome = label;
}

function outcome(r) {
  return r._oroOutcome || "unknown";
}

function fields(r) {
  return r._oroFields || "-";
}

function details(r) {
  return r._oroDetails || "-";
}

function runId(r) {
  return r._oroRunId || "-";
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

function _writeState(provider, allowlist, aliases, expiresAt) {
  ngx.shared[ZONE].set(
    _stateKey(provider),
    JSON.stringify({ allowlist: allowlist, aliases: aliases, expiresAt: expiresAt })
  );
}

function getAllowlist(r, provider, callback) {
  var state = _readState(provider);
  if (state && state.allowlist && Date.now() < state.expiresAt) {
    callback(state.allowlist, state.aliases);
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
            var cacheControl = reply.headersOut["Cache-Control"] || reply.headersOut["cache-control"] || "";
            if (provider === "openrouter" && cacheControl.toLowerCase().indexOf("no-store") !== -1) {
              callback(data.models, state && state.aliases ? state.aliases : data.aliases);
              return;
            }
            _writeState(provider, data.models, data.aliases, Date.now() + CACHE_TTL_MS);
            callback(data.models, data.aliases);
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
        callback(state.allowlist, state.aliases);
        return;
      }

      callback(null);
    }
  );
}

// Every route an inference request may take, as "METHOD decoded-path". Only an exact match
// passes, so no escape, query, fragment, parameter, dot segment, trailing slash or case
// variant (`/inference/alpha/decisions%3F`, `%252F`, `/Chat/Completions/`) reaches upstream,
// and the upstream path is always one of these literals.
var ROUTES = [
  "GET /inference/models",
  "POST /inference/chat/completions",
  "POST /inference/alpha/decisions",
];
// Typed decisions are the validator's own (the simulator's disclosure reader); an agent
// could rehearse the reader with them. They go to their own upstream location, never to
// the generic one agent routes use.
var DECISIONS = "/inference/alpha/decisions";
// The one model the decisions route serves: the runtime's pinned disclosure reader. It is
// checked here instead of against the agent allowlist, so it never becomes an agent model.
var DECISIONS_MODEL = "typesafe/jev-1.13";

function _forbid(r) {
  _tag(r, "internal-forbidden");
  r.headersOut["Content-Type"] = "application/json";
  r.return(403, JSON.stringify({ error: "forbidden" }));
}

function _validatedRequest(r) {
  var provider = detectProvider(r);
  var decisions = r.uri === DECISIONS;
  if (decisions && (r._oroCaller !== "validator" || provider !== "openrouter")) {
    _forbid(r);
    return;
  }
  var upstreamLocation = decisions
    ? "/_openrouter_decisions"
    : (provider === "openrouter" ? "/_openrouter_proxy/" : "/_chutes_proxy/") +
      r.uri.replace(/^\/inference\//, "");

  if (r.method === "GET") {
    // The model listing: the one read passed through.
    r.subrequest(upstreamLocation, { method: "GET" }, function (reply) {
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

  // Capture caller fields before validation, stripping, or proxy additions.
  // Bound both log size and work on a request with many top-level keys.
  var names = [];
  for (var name in parsed || {}) {
    if (names.length === 32) {
      names.push("+more");
      break;
    }
    names.push(name.slice(0, 64).replace(/[^A-Za-z0-9_.-]/g, "_"));
  }
  r._oroFields = names.join(",");

  // Record only tool types and template option names, never tool schemas or values.
  var functionTools = 0;
  var serverTools = 0;
  var otherTools = 0;
  if (Array.isArray(parsed.tools)) {
    parsed.tools.slice(0, 32).forEach(function (tool) {
      if (tool && tool.type === "function") functionTools++;
      else if (tool && typeof tool.type === "string" && tool.type.indexOf("openrouter:") === 0) serverTools++;
      else otherTools++;
    });
  } else if (parsed.tools !== undefined) otherTools++;
  var templateKeys = [];
  if (parsed.chat_template_kwargs && typeof parsed.chat_template_kwargs === "object") {
    for (var key in parsed.chat_template_kwargs) {
      if (templateKeys.length === 16) {
        templateKeys.push("+more");
        break;
      }
      templateKeys.push(key.slice(0, 64).replace(/[^A-Za-z0-9_.-]/g, "_"));
    }
  }
  r._oroDetails = "tool_fn=" + functionTools + " tool_server=" + serverTools +
    " tool_other=" + otherTools + " tool_more=" + (Array.isArray(parsed.tools) && parsed.tools.length > 32 ? 1 : 0) +
    " template_keys=" + (templateKeys.join(",") || "-");

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
  // The decisions endpoint is forwarded exactly as the runtime sends it.
  var usageInjected = false;
  if (provider === "openrouter" && !decisions) {
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

  getAllowlist(r, provider, function (allowed, aliases) {
    if (!allowed) {
      _tag(r, "internal-allowlist-unavailable");
      r.headersOut["Content-Type"] = "application/json";
      r.return(503, JSON.stringify({ error: "Inference allowlist unavailable" }));
      return;
    }

    var requestedModel = parsed.model;
    if (provider === "chutes" && parsed.model === "mistralai/mistral-small-2603") {
      // The sealed shopper simulator requests this OpenRouter ID on Chutes-funded runs.
      parsed.model = "unsloth/Mistral-Nemo-Instruct-2407-TEE";
    } else if (provider === "openrouter" && allowed.indexOf(parsed.model) === -1 &&
        aliases && Object.prototype.hasOwnProperty.call(aliases, parsed.model)) {
      parsed.model = aliases[parsed.model];
    }

    if (decisions ? parsed.model !== DECISIONS_MODEL : allowed.indexOf(parsed.model) === -1) {
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

    var forwardBody = stripped.length > 0 || usageInjected || fallbackInjected || parsed.model !== requestedModel
      ? JSON.stringify(parsed) : body;
    r.subrequest(
      upstreamLocation,
      { method: "POST", body: forwardBody },
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

function validate(r) {
  if (ROUTES.indexOf(r.method + " " + r.uri) === -1) {
    _forbid(r);
    return;
  }
  r.subrequest("/_inference_grant", { method: "GET" }, function (reply) {
    r._oroRunId = reply.headersOut["X-ORO-Run-ID"];
    if (reply.status !== 204) {
      var unauthorized = reply.status === 401;
      _tag(r, unauthorized ? "internal-unauthorized" : "internal-grant-unavailable");
      r.return(unauthorized ? 401 : 503, JSON.stringify({
        error: unauthorized ? "Inference key does not match active run" : "Inference grant unavailable"
      }));
      return;
    }
    r._oroCaller = reply.headersOut["X-ORO-Caller"];
    if (!r._oroRunId) {
      _tag(r, "internal-grant-unavailable");
      r.return(503, JSON.stringify({ error: "Inference grant unavailable" }));
      return;
    }
    _validatedRequest(r);
  });
}

export default { validate: validate, outcome: outcome, fields: fields, details: details, runId: runId };
