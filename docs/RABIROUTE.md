# RabiRoute 接入

FenneNote 可以作为 RabiRoute 的语音输入端，也可以接收 RabiRoute 或 Agent 的反向气泡消息。

## 推荐工作站拓扑

完整角色语音工作站建议按职责拆开：

```text
FenneNote
  麦克风监听、切段、转写
  输出 voice_transcript webhook

RabiRoute
  接收 webhook
  记录 voice-transcripts.jsonl
  按路由规则转发给 Codex/Agent

Codex/Agent
  把转写文本当作用户在电脑旁的输入事件
  执行任务、生成回复文本

OumuQ / TTS worker
  根据角色配置生成语音回声
  只对可信的 FenneNote 语音输入恢复语音输出
```

QQ、群聊和机器人平台侧消息建议默认只走文字，不自动发语音。这样可以避免把远程聊天误判为电脑旁语音输入，也能减少群聊里不必要的语音打扰。

## FenneNote -> RabiRoute

FenneNote 在每段有效转写完成后，可以发送一个 `voice_transcript` 事件。

GUI 设置：

1. 打开“路由”页。
2. 勾选“转写完成后推送到 RabiRoute”。
3. 推送 URL 使用默认值：

```text
http://127.0.0.1:8791/webhook
```

如果部署在公司环境或远程 gateway，请把真实 URL 和 token 只写入本机 `config.json` 或 GUI 保存的运行时配置。公开仓库里只保留模板和占位说明。

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

## Codex/Agent 与角色 TTS 边界

建议 RabiRoute 转发给 Agent 时保留这些语义：

- `type = voice_transcript`：表示这条消息来自 FenneNote 语音输入。
- `source = fennenote`：表示电脑旁麦克风输入端。
- `text`：只放转写后的文本，不附带原始音频。
- `messageId`：用于去重和追踪一次语音切段。

Agent 侧可以把 `voice_transcript` 当作“用户在电脑旁开口说话”，处理完成后再调用 OumuQ 或 TTS worker 做角色语音回声。普通 QQ 文本、群聊、私聊、回复事件不应默认触发语音回声，除非上游明确标记为可信语音输入。

## 识别语言建议

当前办公场景建议优先使用简体中文普通话。日语识别在部分模型和混合语境下可能不稳定，尤其是中文、英文技术词和日语混说时。

推荐初始提示词：

```text
以下是简体中文普通话办公场景转写，可能包含 Unity、Editor、GPU、CPU、AI、Bug、微信、项目、功能、消息、发送、测试等术语。请保持中文为简体，英文术语保留英文。
```

如果需要日语场景，建议单独建立本地配置并反复测试，不要把中文办公提示词和日语提示词混在同一套公开模板里。

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

## 开源脱敏要求

提交或推送前必须确认：

- 不提交真实 RabiRoute webhook URL、OumuQ URL、TTS worker URL、token、cookie、API key。
- 不提交 `config.json`、公司环境配置、个人路径、私有角色配置。
- 不提交 `transcripts/`、`cache/audio/`、录音文件、会议文本、运行日志。
- `config.example.json` 只能包含本地回环地址、空 token、空 API key 和可公开的默认提示词。

部署者应在公司环境本地填写真实参数，并把这些文件留在 `.gitignore` 覆盖范围内。

