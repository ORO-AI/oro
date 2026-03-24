var fs = require("fs");

var allowedModels = JSON.parse(
  fs.readFileSync("/etc/nginx/allowed_models.json")
);
var allowedSet = {};
for (var i = 0; i < allowedModels.length; i++) {
  allowedSet[allowedModels[i]] = true;
}

function validate(r) {
  // Non-POST requests (e.g. GET /inference/models) pass through
  if (r.method !== "POST") {
    var uri = "/_chutes_proxy/" + r.uri.replace(/^\/inference\//, "");
    r.subrequest(uri, { method: r.method, args: r.variables.args || "" }, function (reply) {
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

  // Streaming is not supported — subrequest pattern buffers the full response
  if (parsed.stream === true) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Streaming is not supported through the proxy" }));
    return;
  }

  if (!allowedSet[parsed.model]) {
    r.error("Model not allowed: " + parsed.model);
    r.headersOut["Content-Type"] = "application/json";
    r.return(
      403,
      JSON.stringify({
        error: "Model '" + parsed.model + "' is not allowed",
        allowed_models: allowedModels,
      })
    );
    return;
  }

  // Model is allowed — forward to Chutes via internal location
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
}

export default { validate };
