const baseUrl = process.env.XIAOAI_FENNENOTE_BRIDGE_URL || "http://127.0.0.1:8799";

const payload = {
  deviceId: "test_xiaoai",
  deviceName: "测试小爱",
  area: "desk",
  sessionId: `xiaoai-smoke-${Date.now()}`,
  messageId: `xiaoai-smoke-${Date.now()}`,
  text: process.argv.slice(2).join(" ") || "小爱桥接测试，写入芬妮笔记。"
};

const response = await fetch(`${baseUrl}/v1/xiaoai/decision`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(payload)
});

const bodyText = await response.text();
console.log(`POST ${baseUrl}/v1/xiaoai/decision -> ${response.status}`);
console.log(bodyText);

if (!response.ok) {
  process.exitCode = 1;
}
