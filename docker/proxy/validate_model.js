// Inference proxy: validates outgoing requests against the ORO inference
// allowlist before forwarding to Chutes.
//
// The allowlist is fetched from the ORO Backend (`GET /v1/public/inference/models`)
// via the internal `/_backend_models` location. Each nginx worker caches the
// fetched list for CACHE_TTL_MS. If the Backend is unreachable on cache miss,
// we fail closed with 503 — a stale fallback would only paper over a deeper
// outage (the validator is broken anyway when the Backend is down).

var CACHE_TTL_MS = 15 * 60 * 1000;

var cachedList = null;
var cacheExpiresAt = 0;

function getAllowlist(r, callback) {
  if (cachedList && Date.now() < cacheExpiresAt) {
    callback(cachedList);
    return;
  }

  r.subrequest("/_backend_models", { method: "GET" }, function (reply) {
    if (reply.status === 200) {
      try {
        var data = JSON.parse(reply.responseText);
        if (data && Array.isArray(data.models) && data.models.length > 0) {
          cachedList = data.models;
          cacheExpiresAt = Date.now() + CACHE_TTL_MS;
          callback(cachedList);
          return;
        }
      } catch (e) {
        r.error("Backend models JSON parse failed: " + e.message);
      }
    } else {
      r.error("Backend models fetch returned status " + reply.status);
    }
    callback(null);
  });
}

function validate(r) {
  if (r.method !== "POST") {
    var passUri = "/_chutes_proxy/" + r.uri.replace(/^\/inference\//, "");
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

    var uri = "/_chutes_proxy/" + r.uri.replace(/^\/inference\//, "");
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
