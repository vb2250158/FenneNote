# RabiRoute 接入

FenneNote 可以作为 RabiRoute 的语音输入端，也可以接收 RabiRoute 或 Agent 的反向气泡消息。

## FenneNote -> RabiRoute

FenneNote 在每段有效转写完成后，可以发送一个 `voice_transcript` 事件。

GUI 设置：

1. 打开“路由”页。
2. 勾选“转写完成后推送到 RabiRoute”。
3. 推送 URL 使用默认值：

```text
http://127.0.0.1:8791/webhook
```

发送 payload 示例：

```json
{
  "type": "voice_transcript",
  "source": "fennenote",
  "text": "这是一段语音笔记",
  "startedAt": "2026-06-03T12:00:00",
  "endedAt": "2026-06-03T12:00:04",
  "durationSeconds": 4.0,
  "peak": 0.032,
  "time": 1780500000,
  "messageId": "fennenote-1780500000000"
}
```

## RabiRoute 侧

RabiRoute 需要启用 `Webhook / FenneNote` 消息适配器。

推荐用单独 gateway：

- `messageAdapters`: `["webhook"]`
- `gatewayPort`: `8791`
- `webhookPath`: `/webhook`
- 路由类型：`voice_transcript`

RabiRoute 收到后会写入：

```text
voice-transcripts.jsonl
```

并按 `voice_transcript` 路由规则决定是否转发给 Codex Desktop 或其他处理端。

## RabiRoute -> FenneNote 气泡

FenneNote 的“应用”页默认开启左下角气泡。

默认本地回调地址：

```text
http://127.0.0.1:8792/reply
```

POST 示例：

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8792/reply" `
  -ContentType "application/json; charset=utf-8" `
  -Body (@{
    title = "RabiRoute"
    text = "已收到语音笔记。"
  } | ConvertTo-Json -Compress)
```

支持字段：

- `title`
- `sender`
- `text`
- `message`
- `content`

FenneNote 会优先显示 `title`，正文优先使用 `text`。

如果设置了气泡访问令牌，请带请求头：

```text
X-FenneNote-Token: <token>
```

气泡默认显示 3 秒，可在 GUI 中调整。

