# XiaoAI FenneNote Adapter

This is the PC-side bridge for sending recognized XiaoAI text into FenneNote.

It does not discover or control the speaker by itself. The XiaoAI side still needs an Open-XiaoAI / MiGPT / cloud-function style runtime that can POST recognized text to this bridge.

## Path

```text
XiaoAI speaker / cloud function / Open-XiaoAI runtime
  -> POST http://127.0.0.1:8799/v1/xiaoai/decision
  -> this bridge
  -> POST http://127.0.0.1:8793/api/fennenote/xiaoai
  -> FenneNote transcript preview / transcript file
  -> optional FenneNote -> RabiRoute forwarding
```

The bridge forwards all transcripts to FenneNote. By default it returns `action: "ignore"` for `/decision`, so native XiaoAI can continue. Set `XIAOAI_INTERCEPT_REGEX` only if the XiaoAI runtime should stop native handling for matching commands.

## Run

```powershell
cd .\plugin-adapters\xiaoai-fennenote
$env:FENNENOTE_XIAOAI_URL = "http://127.0.0.1:8793/api/fennenote/xiaoai"
npm.cmd start
```

Optional:

```powershell
$env:XIAOAI_FENNENOTE_BRIDGE_PORT = "8799"
$env:XIAOAI_INTERCEPT_REGEX = "^(问\s*Rabi|让\s*Rabi|Rabi|兔兔)"
```

## Smoke

Start FenneNote first, then:

```powershell
cd .\plugin-adapters\xiaoai-fennenote
npm.cmd run smoke
```

Expected: FenneNote appends the text to `dist\FenneNote\transcripts\YYYY-MM-DD.txt` when running from the packaged app.

## API

### POST `/v1/xiaoai/transcript`

```json
{
  "deviceId": "bedroom_xiaoai",
  "deviceName": "卧室小爱",
  "area": "bedroom",
  "sessionId": "xiaoai-session-001",
  "messageId": "xiaoai-001",
  "text": "问 Rabi 今天电脑任务跑完了吗"
}
```

### POST `/v1/xiaoai/decision`

Same payload as `/transcript`. This endpoint is useful for Open-XiaoAI / MiGPT style integrations because it forwards the transcript and returns:

```json
{
  "ok": true,
  "action": "ignore",
  "reason": "Transcript was forwarded to FenneNote. Native XiaoAI may continue."
}
```

If `XIAOAI_INTERCEPT_REGEX` matches, it returns `action: "intercept"` plus `speakText`.

### POST `/v1/xiaoai/speak`

Currently only records the request. Speaker playback should be wired after the concrete XiaoAI-side runtime is chosen.
