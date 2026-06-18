// ORO-1372 Task 10: proxy nonce mint + verify for catalogue tool calls.
//
// The sandbox proxy is the single trust boundary between the agent and the
// outside world. This module makes it the canonical authority for "did the
// LLM emit this tool call?" by minting HMAC-signed nonces per parsed LLM
// tool_call response, and verifying them on `/search/*` dispatches.
//
// Two entry points:
//   - enrichResponseText(r, evalRunId, bodyText): called by validate_model
//     in its `/inference/chat/completions` subrequest success callback.
//     Parses the LLM response, mints nonces for every parsed tool_call (both
//     native `choices[i].message.tool_calls` and XML `<tool_call>...` blocks
//     in `choices[i].message.content`), injects `oro_metadata` into the body,
//     and returns the modified body string. Returns the original bytes on any
//     error (non-JSON, unexpected shape) — fail-open so inference still works
//     even if enrichment breaks.
//
//   - verifyDispatch(r): js_content handler for `/search/*`. Reads
//     X-Tool-Nonce + X-Tool-Call-Id, recomputes HMAC, checks expiry, args
//     hash, single-use via shared dict, sets X-Nonce-Status response header,
//     and proxies the request forward to the search-server. In Phase 0-2
//     (default, ORO_PROXY_NONCE_STRICT unset) always forwards regardless of
//     status; in Phase 3 (ORO_PROXY_NONCE_STRICT=true) returns 403 on any
//     non-`valid` status.
//
// Crypto choice: HMAC-SHA256 via njs Web Crypto (njs >= 0.9.7). The HMAC key
// is read once per worker on first mint/verify from $ORO_PROXY_HMAC_KEY and
// cached.
//
// Replay defence: js_shared_dict_zone `nonce_used` (declared in
// nginx.conf.template) holds used nonces for 120s — the longest plausible
// race window beyond the 60s mint TTL. `add()` returns false on duplicate
// key, giving us atomic single-use across all worker processes without
// Redis. The dict zone is sized for steady-state catalogue dispatch rate
// (one nonce per dispatch, ~few hundred concurrent runs) with comfortable
// headroom.
//
// Payload encoding: a fixed-order template literal — DO NOT switch to
// JSON.stringify(object). njs's stringify key ordering is implementation-
// defined and would silently break HMAC verification.

const NONCE_TTL_MS = 60000;
const NONCE_SHARED_ZONE = "nonce_used";
const TOOL_PATH_PREFIX = "/search/";

const CATALOGUE_TOOLS = {
    "find_product": true,
    "view_product_information": true,
    "check_product_match": true,
    "find_products_in_same_shop": true,
    "calculate_voucher": true,
};

let _cachedKey = null;

async function _getKey() {
    if (_cachedKey) return _cachedKey;
    const keyStr = process.env.ORO_PROXY_HMAC_KEY || "dev-key-not-for-prod";
    const keyBytes = (new TextEncoder()).encode(keyStr);
    _cachedKey = await crypto.subtle.importKey(
        "raw", keyBytes, { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]
    );
    return _cachedKey;
}

function _b64urlEncode(buf) {
    // Buffer.toString("base64") accepts ArrayBuffer / Uint8Array / Buffer.
    const std = Buffer.from(buf).toString("base64");
    return std.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function _b64urlDecode(str) {
    let s = str.replace(/-/g, "+").replace(/_/g, "/");
    while (s.length % 4) s += "=";
    return Buffer.from(s, "base64");
}

async function _sha256B64Url(input) {
    const bytes = typeof input === "string"
        ? (new TextEncoder()).encode(input)
        : input;
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return _b64urlEncode(new Uint8Array(digest));
}

async function _hmacB64Url(payloadStr) {
    const k = await _getKey();
    const sig = await crypto.subtle.sign(
        "HMAC", k, (new TextEncoder()).encode(payloadStr)
    );
    return _b64urlEncode(new Uint8Array(sig));
}

// Fixed-order template literal. DO NOT use JSON.stringify(object) — njs
// orders object keys implementation-defined-ly and HMAC verification would
// silently fail.
function _buildPayload(evalRunId, callId, toolName, argsHash, expiresAt) {
    return '{"e":"' + evalRunId +
        '","c":"' + callId +
        '","t":"' + toolName +
        '","a":"' + argsHash +
        '","x":' + expiresAt + '}';
}

async function _mintNonce(evalRunId, callId, toolName, argsRaw) {
    const argsHash = await _sha256B64Url(argsRaw);
    const expiresAt = Date.now() + NONCE_TTL_MS;
    const payload = _buildPayload(evalRunId, callId, toolName, argsHash, expiresAt);
    const sig = await _hmacB64Url(payload);
    const payloadB64 = _b64urlEncode((new TextEncoder()).encode(payload));
    return payloadB64 + "." + sig;
}

// Find <tool_call>...</tool_call> blocks in legacy XML-style content. The
// inner payload is either a single JSON object {name, arguments} or an array
// of such. Returns ordered list of {raw, name} where raw is the
// JSON-substring (verbatim, untouched bytes — used for arg-hash).
function _extractXmlToolCalls(content) {
    if (typeof content !== "string") return [];
    const out = [];
    // Non-greedy match between <tool_call> ... </tool_call>; allow newlines.
    // We capture either an array `[...]` or object `{...}` payload.
    const re = /<tool_call>\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*<\/tool_call>/g;
    let m;
    while ((m = re.exec(content)) !== null) {
        const raw = m[1];
        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch (e) {
            continue;
        }
        const items = Array.isArray(parsed) ? parsed : [parsed];
        for (let i = 0; i < items.length; i++) {
            const it = items[i];
            if (it && typeof it.name === "string") {
                out.push({ raw: raw, name: it.name });
            }
        }
    }
    return out;
}

// Walk the chat-completions JSON, mint a nonce per catalogue tool call, and
// return {nonces: {callId -> nonceStr}, parsed: [{call_id,tool_name,args_hash}]}.
// Fail-open: any error returns empty maps so the surrounding response still
// flows. The agent will just see no nonces and dispatches will record
// `nonce_status=missing`.
async function _buildNonceMap(evalRunId, parsedBody) {
    const tool_nonces = {};
    const parsed_tool_calls = [];

    if (!parsedBody || !Array.isArray(parsedBody.choices)) {
        return { tool_nonces: tool_nonces, parsed_tool_calls: parsed_tool_calls };
    }

    for (let ci = 0; ci < parsedBody.choices.length; ci++) {
        const choice = parsedBody.choices[ci];
        if (!choice || !choice.message) continue;
        const msg = choice.message;

        // Native tool_calls array (OpenAI format).
        if (Array.isArray(msg.tool_calls)) {
            for (let ti = 0; ti < msg.tool_calls.length; ti++) {
                const tc = msg.tool_calls[ti];
                if (!tc || !tc.function || typeof tc.function.name !== "string") continue;
                const toolName = tc.function.name;
                if (!CATALOGUE_TOOLS[toolName]) continue;
                const callId = (typeof tc.id === "string" && tc.id.length > 0)
                    ? tc.id
                    : "native_" + ci + "_" + ti;
                // The arguments come from the model as a string already.
                const argsRaw = typeof tc.function.arguments === "string"
                    ? tc.function.arguments
                    : JSON.stringify(tc.function.arguments || {});
                const argsHash = await _sha256B64Url(argsRaw);
                const nonce = await _mintNonceFromHash(
                    evalRunId, callId, toolName, argsHash
                );
                tool_nonces[callId] = nonce;
                parsed_tool_calls.push({
                    call_id: callId,
                    tool_name: toolName,
                    args_hash: argsHash,
                });
            }
        }

        // Legacy XML tool_call blocks embedded in content.
        if (typeof msg.content === "string") {
            const blocks = _extractXmlToolCalls(msg.content);
            for (let bi = 0; bi < blocks.length; bi++) {
                const blk = blocks[bi];
                if (!CATALOGUE_TOOLS[blk.name]) continue;
                const callId = "xml_" + ci + "_" + bi;
                const argsHash = await _sha256B64Url(blk.raw);
                const nonce = await _mintNonceFromHash(
                    evalRunId, callId, blk.name, argsHash
                );
                tool_nonces[callId] = nonce;
                parsed_tool_calls.push({
                    call_id: callId,
                    tool_name: blk.name,
                    args_hash: argsHash,
                });
            }
        }
    }

    return { tool_nonces: tool_nonces, parsed_tool_calls: parsed_tool_calls };
}

async function _mintNonceFromHash(evalRunId, callId, toolName, argsHash) {
    const expiresAt = Date.now() + NONCE_TTL_MS;
    const payload = _buildPayload(evalRunId, callId, toolName, argsHash, expiresAt);
    const sig = await _hmacB64Url(payload);
    const payloadB64 = _b64urlEncode((new TextEncoder()).encode(payload));
    return payloadB64 + "." + sig;
}

// Public: enrich an `/inference/chat/completions` response body string with
// minted nonces. Returns the new body string. On any error, returns the
// original body unchanged (fail-open: better to lose nonce enforcement than
// to break inference).
async function enrichResponseText(r, bodyText) {
    const evalRunId = process.env.ORO_EVAL_RUN_ID || "";
    if (!evalRunId) {
        // No run id configured — nothing to mint against. Pass through.
        return bodyText;
    }
    if (typeof bodyText !== "string" || bodyText.length === 0) {
        return bodyText;
    }
    let parsed;
    try {
        parsed = JSON.parse(bodyText);
    } catch (e) {
        return bodyText;
    }
    if (!parsed || typeof parsed !== "object") {
        return bodyText;
    }

    let result;
    try {
        result = await _buildNonceMap(evalRunId, parsed);
    } catch (e) {
        r.error("tool_nonce.enrichResponseText mint failed: " + (e && e.message));
        return bodyText;
    }

    if (Object.keys(result.tool_nonces).length === 0) {
        // No catalogue tool calls in this response — leave body unchanged so
        // the byte-level shape stays identical to upstream's response.
        return bodyText;
    }

    const meta = parsed.oro_metadata && typeof parsed.oro_metadata === "object"
        ? parsed.oro_metadata
        : {};
    meta.tool_nonces = result.tool_nonces;
    meta.parsed_tool_calls = result.parsed_tool_calls;
    parsed.oro_metadata = meta;

    try {
        return JSON.stringify(parsed);
    } catch (e) {
        r.error("tool_nonce.enrichResponseText serialize failed: " + (e && e.message));
        return bodyText;
    }
}

// ---- Dispatch verification ----------------------------------------------

function _classifyStatus(status) {
    return status;
}

// Compare the nonce payload's claims against the live request. Returns one
// of: "valid", "missing", "mismatch", "expired", "replayed", "malformed".
async function _verifyNonce(r, nonceStr, body) {
    if (!nonceStr) return "missing";
    const dot = nonceStr.lastIndexOf(".");
    if (dot <= 0 || dot >= nonceStr.length - 1) return "malformed";

    const payloadB64 = nonceStr.substring(0, dot);
    const sigB64 = nonceStr.substring(dot + 1);

    let payloadBytes;
    try {
        payloadBytes = _b64urlDecode(payloadB64);
    } catch (e) {
        return "malformed";
    }

    const payloadStr = payloadBytes.toString("utf8");
    let payload;
    try {
        payload = JSON.parse(payloadStr);
    } catch (e) {
        return "malformed";
    }

    // Recompute HMAC over the exact bytes we b64-decoded — not over a
    // canonicalised re-serialisation. The mint side emitted these bytes
    // verbatim.
    let expectedSig;
    try {
        expectedSig = await _hmacB64Url(payloadStr);
    } catch (e) {
        r.error("tool_nonce verify hmac failed: " + (e && e.message));
        return "mismatch";
    }
    if (expectedSig !== sigB64) {
        return "mismatch";
    }

    const evalRunId = process.env.ORO_EVAL_RUN_ID || "";
    if (payload.e !== evalRunId) return "mismatch";

    // Path check: /search/<tool> → "tool".
    let toolFromPath = r.uri;
    if (toolFromPath.indexOf(TOOL_PATH_PREFIX) === 0) {
        toolFromPath = toolFromPath.substring(TOOL_PATH_PREFIX.length);
    }
    // Strip any trailing slash / query.
    const q = toolFromPath.indexOf("?");
    if (q >= 0) toolFromPath = toolFromPath.substring(0, q);
    if (toolFromPath.endsWith("/")) toolFromPath = toolFromPath.slice(0, -1);

    if (payload.t !== toolFromPath) return "mismatch";

    if (typeof payload.x !== "number" || payload.x <= Date.now()) {
        return "expired";
    }

    // Hash the request body and compare. If body is missing/empty treat as
    // empty string for hash purposes; mint side did the same when the LLM
    // emitted arguments=""
    const bodyForHash = (typeof body === "string") ? body : "";
    let bodyHash;
    try {
        bodyHash = await _sha256B64Url(bodyForHash);
    } catch (e) {
        return "mismatch";
    }
    if (payload.a !== bodyHash) return "mismatch";

    // Atomic single-use. dict.add returns false if key already present.
    const dict = ngx.shared[NONCE_SHARED_ZONE];
    if (dict) {
        try {
            const added = dict.add(nonceStr, "1");
            if (added === false) return "replayed";
        } catch (e) {
            // dict.add throws if the key already exists in some njs builds
            // (older add() semantics). Treat throw as replay.
            return "replayed";
        }
    }
    return "valid";
}

// js_content handler for /search/*.
async function verifyDispatch(r) {
    // get_product_raw is blocked at the location level; we only see other
    // catalogue endpoints. Read body first so r.requestBuffer / r.requestText
    // is populated.
    const body = r.requestText || "";
    const nonceHdr = r.headersIn["X-Tool-Nonce"] || r.headersIn["x-tool-nonce"] || "";

    let status;
    try {
        status = await _verifyNonce(r, nonceHdr, body);
    } catch (e) {
        r.error("tool_nonce.verifyDispatch error: " + (e && e.message));
        status = "mismatch";
    }

    // Strict mode: reject any non-valid status. Default (Phase 0-2): allow
    // through and just stamp the status header so the trajectory recorder
    // (Task 9) can log it.
    const strict = (process.env.ORO_PROXY_NONCE_STRICT || "").toLowerCase() === "true";
    if (strict && status !== "valid") {
        r.headersOut["X-Nonce-Status"] = status;
        r.headersOut["Content-Type"] = "application/json";
        r.return(403, JSON.stringify({ error: "tool_nonce " + status }));
        return;
    }

    // Forward to upstream via internal subrequest. Mirror validate_model's
    // pattern: build internal URI, forward method/body/args, copy
    // headers+body back to the client.
    const internalUri = "/_search_upstream" + r.uri.replace(/^\/search/, "");
    const args = r.variables.args || "";
    const opts = {
        method: r.method,
        args: args,
    };
    if (r.method === "POST" || r.method === "PUT" || r.method === "PATCH") {
        opts.body = body;
    }

    r.subrequest(internalUri, opts, function (reply) {
        for (const h in reply.headersOut) {
            r.headersOut[h] = reply.headersOut[h];
        }
        r.headersOut["X-Nonce-Status"] = status;
        r.return(reply.status, reply.responseText);
    });
}

export default {
    enrichResponseText: enrichResponseText,
    verifyDispatch: verifyDispatch,
};
