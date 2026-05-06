// Inference proxy: validates outgoing requests against the ORO inference
// allowlist before forwarding to Chutes, or bypasses the allowlist and
// forwards directly to OpenRouter when the bearer token starts with "sk-or-".
//
// The allowlist is fetched from the ORO Backend (`GET /v1/public/inference/models`)
// via the internal `/_backend_models` location and cached in an nginx
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
//   - Bearer token starts with "sk-or-" → OpenRouter (no allowlist check)
//   - Any other token shape (e.g. cak_*) → Chutes (allowlist enforced)

function detectProvider(r) {
  var auth = r.headersIn["Authorization"] || "";
  // Bearer tokens issued by OpenRouter start with "sk-or-".
  if (auth.indexOf("Bearer sk-or-") === 0) {
    return "openrouter";
  }
  // Default to Chutes for any other token shape (including the existing
  // cak_* prefix). Keeps behavior unchanged for existing eval flows.
  return "chutes";
}

var CACHE_TTL_MS = 15 * 60 * 1000;
// Window beyond CACHE_TTL_MS where we still serve the cached list if a
// refresh fails. After this we give up and fail closed.
var STALE_GRACE_MS = 60 * 60 * 1000;

var ZONE = "oro_models";
var STATE_KEY = "state";

function _readState() {
  var raw = ngx.shared[ZONE].get(STATE_KEY);
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

function _writeState(allowlist, expiresAt) {
  ngx.shared[ZONE].set(
    STATE_KEY,
    JSON.stringify({ allowlist: allowlist, expiresAt: expiresAt })
  );
}

function getAllowlist(r, callback) {
  var state = _readState();
  if (state && state.allowlist && Date.now() < state.expiresAt) {
    callback(state.allowlist);
    return;
  }

  r.subrequest("/_backend_models", { method: "GET" }, function (reply) {
    if (reply.status === 200) {
      try {
        var data = JSON.parse(reply.responseText);
        if (data && Array.isArray(data.models) && data.models.length > 0) {
          _writeState(data.models, Date.now() + CACHE_TTL_MS);
          callback(data.models);
          return;
        }
        r.error("Backend models response missing or empty 'models' array");
      } catch (e) {
        r.error("Backend models JSON parse failed: " + e.message);
      }
    } else {
      r.error("Backend models fetch returned status " + reply.status);
    }

    // Re-read in case another worker just succeeded while this one was
    // waiting on a failing subrequest.
    state = _readState();
    if (state && state.allowlist && Date.now() < state.expiresAt + STALE_GRACE_MS) {
      var graceLeft = state.expiresAt + STALE_GRACE_MS - Date.now();
      if (graceLeft < STALE_GRACE_MS) {
        r.error(
          "Serving stale allowlist after fetch failure (" +
            (graceLeft / 1000).toFixed(0) +
            "s grace remaining)"
        );
      }
      callback(state.allowlist);
      return;
    }

    callback(null);
  });
}

function validate(r) {
  var provider = detectProvider(r);
  var upstreamLocation = provider === "openrouter" ? "/_openrouter_proxy/" : "/_chutes_proxy/";

  if (r.method !== "POST") {
    var passUri = upstreamLocation + r.uri.replace(/^\/inference\//, "");
    r.subrequest(passUri, { method: r.method, args: r.variables.args || "" }, function (reply) {
      for (var h in reply.headersOut) {
        r.headersOut[h] = reply.headersOut[h];
      }
      r.return(reply.status, reply.responseText);
    });
    return;
  }

  var body = r.requestText;

  if (!body) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Missing or unreadable request body" }));
    return;
  }

  var parsed;
  try {
    parsed = JSON.parse(body);
  } catch (e) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Invalid JSON in request body" }));
    return;
  }

  if (!parsed.model) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Missing 'model' field in request body" }));
    return;
  }

  if (parsed.stream === true) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Streaming is not supported through the proxy" }));
    return;
  }

  // OpenRouter has 300+ models with no curated allowlist; per-key USD cap
  // bounds damage. Skip model validation for that path.
  if (provider === "openrouter") {
    var orUri = upstreamLocation + r.uri.replace(/^\/inference\//, "");
    r.subrequest(
      orUri,
      { method: "POST", body: body, args: r.variables.args || "" },
      function (reply) {
        for (var h in reply.headersOut) {
          r.headersOut[h] = reply.headersOut[h];
        }
        r.return(reply.status, reply.responseText);
      }
    );
    return;
  }

  getAllowlist(r, function (allowed) {
    if (!allowed) {
      r.headersOut["Content-Type"] = "application/json";
      r.return(503, JSON.stringify({ error: "Inference allowlist unavailable" }));
      return;
    }

    if (allowed.indexOf(parsed.model) === -1) {
      r.error("Model not allowed: " + parsed.model);
      r.headersOut["Content-Type"] = "application/json";
      r.return(
        403,
        JSON.stringify({
          error: "Model '" + parsed.model + "' is not allowed",
          allowed_models: allowed,
        })
      );
      return;
    }

    var uri = upstreamLocation + r.uri.replace(/^\/inference\//, "");
    r.subrequest(
      uri,
      { method: "POST", body: body, args: r.variables.args || "" },
      function (reply) {
        for (var h in reply.headersOut) {
          r.headersOut[h] = reply.headersOut[h];
        }
        r.return(reply.status, reply.responseText);
      }
    );
  });
}

export default { validate };
