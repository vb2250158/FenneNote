# 开源发布检查清单

这份清单用于把 FenneNote 发布到公开 GitHub 仓库前做脱敏确认。

## 必须保留在本地的内容

不要提交这些内容：

- 真实 RabiRoute webhook URL、OumuQ URL、TTS worker URL。
- token、cookie、API key、Authorization header、公司环境密钥。
- `config.json`、`.env`、公司环境配置、个人绝对路径。
- `transcripts/`、音频缓存、私有录音、会议记录、运行日志。
- `cache/`、`dist/`、`build/`、`.venv-gpu/`、模型缓存和打包产物。

公开仓库只保留：

- `config.example.json` 脱敏模板。
- 本地回环地址示例，例如 `http://127.0.0.1:8791/webhook`。
- 空 token、空 API key 和可公开的默认提示词。
- RabiRoute/OumuQ/Codex 工作流说明。

## 提交前命令

```powershell
git status --short
git diff --staged --stat
git diff --staged
```

如果要做敏感词快速扫描：

```powershell
rg -n --hidden --glob '!/.git/**' --glob '!cache/**' --glob '!dist/**' --glob '!build/**' --glob '!transcripts/**' "token|cookie|api[_-]?key|authorization|Bearer|webhook|secret|password|private|公司|内网"
```

出现命中不代表一定有泄漏，但必须逐条确认。真实值只允许存在于未跟踪的本地运行时文件里。

## 公司环境部署原则

公司部署时从公开仓库拉取代码后，在部署机器本地生成或填写 `config.json`：

1. 启动 GUI 生成 `config.json`。
2. 在“路由”页填入真实 RabiRoute webhook URL 和 token。
3. 如使用 API 转写，在本地填入 API Key。
4. 如接入 OumuQ 或 TTS worker，在 RabiRoute/OumuQ 各自项目的本地配置中填写真实参数。
5. 不把公司环境配置复制回公开仓库。

FenneNote 的职责是把电脑旁用户语音变成 `voice_transcript` 文本事件。QQ/群聊消息默认只走文字链路，只有可信 FenneNote 语音输入才建议触发角色 TTS 回声。
