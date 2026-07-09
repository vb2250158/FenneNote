import fs from "node:fs";
import http from "node:http";
import path from "node:path";

const host = process.env.XIAOAI_FENNENOTE_BRIDGE_HOST || process.env.XIAOAI_BRIDGE_HOST || "127.0.0.1";
const port = Number(process.env.XIAOAI_FENNENOTE_BRIDGE_PORT || process.env.XIAOAI_BRIDGE_PORT || "8799");
const fenneNoteXiaoAiUrl = process.env.FENNENOTE_XIAOAI_URL || "http://127.0.0.1:8793/api/fennenote/xiaoai";
const fenneNoteXiaoAiConfigUrl =
  process.env.FENNENOTE_XIAOAI_CONFIG_URL || "http://127.0.0.1:8793/api/fennenote/xiaoai/config";
const fenneNoteXiaoAiLevelUrl =
  process.env.FENNENOTE_XIAOAI_LEVEL_URL || "http://127.0.0.1:8793/api/fennenote/xiaoai/level";
const interceptPatternText = process.env.XIAOAI_INTERCEPT_REGEX || "";
const interceptPattern = interceptPatternText ? new RegExp(interceptPatternText, "i") : null;
const defaultInterceptSpeakText = process.env.XIAOAI_INTERCEPT_SPEAK_TEXT || "收到，已经转给芬妮笔记。";
const logPath = process.env.XIAOAI_FENNENOTE_LOG_PATH || path.join(process.cwd(), "xiaoai-fennenote.log.jsonl");

const counters = {
  transcripts: 0,
  decisions: 0,
  intercepted: 0,
  ignored: 0,
  forwardErrors: 0
};

const lastEvents = [];
const lastSpeakRequests = [];

function jsonResponse(response, statusCode, payload) {
  response.writeHead(statusCode, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

function remember(list, item) {
  list.push(item);
  while (list.length > 10) list.shift();
}

function appendLog(kind, payload) {
  const entry = {
    time: new Date().toISOString(),
    kind,
    ...payload
  };
  fs.appendFile(logPath, `${JSON.stringify(entry)}\n`, () => {});
}

function trace(label, payload = {}) {
  console.log(`[芬妮笔记桥 ${new Date().toISOString()}] ${label} ${JSON.stringify(payload)}`);
  appendLog("trace", { label, ...payload });
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    request.on("data", (chunk) => {
      chunks.push(chunk);
      if (Buffer.concat(chunks).byteLength > 1024 * 1024) {
        reject(new Error("Payload too large"));
        request.destroy();
      }
    });
    request.on("end", () => {
      const body = Buffer.concat(chunks).toString("utf8");
      resolve(body ? JSON.parse(body) : {});
    });
    request.on("error", reject);
  });
}

function extractText(payload) {
  return String(payload.text ?? payload.message ?? payload.content ?? payload.query ?? "").trim();
}

function normalizeForFenneNote(payload) {
  const text = extractText(payload);
  if (!text) {
    throw new Error("Missing text/message/content");
  }

  return {
    text,
    source: "xiaoai",
    adapterType: "xiaoai",
    sourceDeviceId: payload.sourceDeviceId ?? payload.deviceId,
    sourceDeviceName: payload.sourceDeviceName ?? payload.deviceName,
    deviceId: payload.deviceId ?? payload.sourceDeviceId,
    deviceName: payload.deviceName ?? payload.sourceDeviceName,
    sourceArea: payload.sourceArea ?? payload.area,
    area: payload.area ?? payload.sourceArea,
    sessionId: payload.sessionId,
    messageId: payload.messageId ?? payload.id ?? `xiaoai-${Date.now()}`,
    time: payload.time ?? Math.floor(Date.now() / 1000),
    confidence: payload.confidence,
    beginOffset: payload.beginOffset ?? payload.begin_offset,
    endOffset: payload.endOffset ?? payload.end_offset,
    originText: payload.originText ?? payload.origin_text,
    rawXiaoAI: payload.rawXiaoAI,
    raw: payload
  };
}

async function forwardToFenneNote(payload) {
  const fennePayload = normalizeForFenneNote(payload);
  trace("开始转发到芬妮笔记", {
    地址: fenneNoteXiaoAiUrl,
    文本: fennePayload.text,
    sessionId: fennePayload.sessionId,
    messageId: fennePayload.messageId,
    置信度: fennePayload.confidence,
    开始偏移: fennePayload.beginOffset,
    结束偏移: fennePayload.endOffset,
  });
  const response = await fetch(fenneNoteXiaoAiUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(fennePayload)
  });

  const responseText = await response.text();
  let responseBody = responseText;
  try {
    responseBody = responseText ? JSON.parse(responseText) : null;
  } catch {
    // Keep the plain text body.
  }

  if (!response.ok) {
    throw new Error(`FenneNote returned ${response.status}: ${responseText}`);
  }
  trace("芬妮笔记返回", {
    HTTP状态: response.status,
    文本: fennePayload.text,
    sessionId: fennePayload.sessionId,
    messageId: fennePayload.messageId,
    芬妮笔记状态: responseBody?.status,
    芬妮笔记ID: responseBody?.id,
  });

  counters.transcripts += 1;
  const event = {
    text: fennePayload.text,
    deviceName: fennePayload.deviceName,
    sessionId: fennePayload.sessionId,
    messageId: fennePayload.messageId,
    fenneNote: responseBody
  };
  remember(lastEvents, { time: new Date().toISOString(), ...event });
  appendLog("forwarded", event);
  return { sent: fennePayload, response: responseBody };
}

async function fetchFenneNoteXiaoAiConfig() {
  const response = await fetch(fenneNoteXiaoAiConfigUrl);
  const bodyText = await response.text();
  let body = {};
  try {
    body = bodyText ? JSON.parse(bodyText) : {};
  } catch {
    throw new Error(`FenneNote config returned non-JSON: ${bodyText}`);
  }
  if (!response.ok) {
    throw new Error(`FenneNote config returned ${response.status}: ${bodyText}`);
  }
  return body;
}

async function forwardLevelToFenneNote(payload) {
  const response = await fetch(fenneNoteXiaoAiLevelUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(`FenneNote level returned ${response.status}: ${await response.text()}`);
  }
}

async function buildDecision(payload) {
  const text = extractText(payload);
  if (!text) {
    throw new Error("Missing text/message/content");
  }
  trace("收到 decision 请求", {
    文本: text,
    sessionId: payload.sessionId,
    messageId: payload.messageId,
    置信度: payload.confidence,
    开始偏移: payload.beginOffset,
    结束偏移: payload.endOffset,
  });

  const forwarded = await forwardToFenneNote(payload);
  counters.decisions += 1;
  const matched = interceptPattern ? interceptPattern.test(text) : false;

  if (matched) {
    counters.intercepted += 1;
  } else {
    counters.ignored += 1;
  }
  return {
    ok: true,
    action: matched ? "intercept" : "ignore",
    reason: matched
      ? "Matched local XiaoAI intercept rule. Transcript was forwarded to FenneNote; native XiaoAI playback is suppressed."
      : "Transcript was forwarded to FenneNote. Native XiaoAI may continue.",
    matchedRule: matched ? interceptPatternText : "",
    forwarded
  };
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${request.headers.host || `${host}:${port}`}`);

  try {
    if (request.method === "GET" && url.pathname === "/health") {
      jsonResponse(response, 200, {
        ok: true,
        service: "xiaoai-fennenote-adapter",
        fenneNoteXiaoAiUrl,
        interceptPattern: interceptPatternText,
        endpoints: {
          transcript: "/v1/xiaoai/transcript",
          decision: "/v1/xiaoai/decision",
          config: "/v1/xiaoai/config",
          level: "/v1/xiaoai/level",
          speak: "/v1/xiaoai/speak"
        },
        counters,
        lastEvents,
        lastSpeakRequests
      });
      return;
    }

    if (request.method === "GET" && url.pathname === "/v1/xiaoai/config") {
      try {
        const config = await fetchFenneNoteXiaoAiConfig();
        jsonResponse(response, 200, config);
      } catch (error) {
        jsonResponse(response, 200, {
          ok: false,
          recordThreshold: 0.01,
          wakeCooldownMs: 0,
          error: error instanceof Error ? error.message : String(error)
        });
      }
      return;
    }

    if (request.method === "POST" && url.pathname === "/v1/xiaoai/level") {
      const payload = await readJson(request);
      try {
        await forwardLevelToFenneNote(payload);
      } catch (error) {
        appendLog("level_forward_error", { error: error instanceof Error ? error.message : String(error), payload });
      }
      jsonResponse(response, 202, { ok: true });
      return;
    }

    if (request.method === "POST" && (url.pathname === "/v1/xiaoai/transcript" || url.pathname === "/api/fennenote/xiaoai")) {
      const payload = await readJson(request);
      let forwarded;
      try {
        forwarded = await forwardToFenneNote(payload);
      } catch (error) {
        counters.forwardErrors += 1;
        appendLog("forward_error", { error: error instanceof Error ? error.message : String(error), payload });
        throw error;
      }
      jsonResponse(response, 200, { ok: true, forwarded });
      return;
    }

    if (request.method === "POST" && url.pathname === "/v1/xiaoai/decision") {
      const payload = await readJson(request);
      let decision;
      try {
        decision = await buildDecision(payload);
      } catch (error) {
        counters.forwardErrors += 1;
        appendLog("decision_error", { error: error instanceof Error ? error.message : String(error), payload });
        throw error;
      }
      jsonResponse(response, 200, decision);
      return;
    }

    if (request.method === "POST" && url.pathname === "/v1/xiaoai/speak") {
      const payload = await readJson(request);
      const item = {
        receivedAt: new Date().toISOString(),
        deviceId: payload.deviceId,
        deviceName: payload.deviceName,
        text: payload.text,
        interrupt: payload.interrupt !== false,
        requestId: payload.requestId
      };
      remember(lastSpeakRequests, item);
      appendLog("speak_placeholder", item);
      jsonResponse(response, 202, {
        ok: true,
        queued: item,
        note: "This bridge records speak requests only. Wire speaker playback after the XiaoAI-side runtime is selected."
      });
      return;
    }

    jsonResponse(response, 404, { ok: false, error: "not_found" });
  } catch (error) {
    jsonResponse(response, 500, {
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

server.listen(port, host, () => {
  console.log(`XiaoAI FenneNote adapter listening on http://${host}:${port}`);
  console.log(`Forwarding XiaoAI transcripts to ${fenneNoteXiaoAiUrl}`);
  console.log(`Decision intercept regex: ${interceptPatternText || "(disabled; forward all, ignore native interruption)"}`);
});
