# FenneNote 云端 ASR 历史归档

FenneNote 已于 2026-07-17 结束维护。DashScope、MiMo 和云端说话人分离不再属于可运行配置；`config_version: 6` 只接受本地 faster-whisper，旧配置选择 `model_source: api` 时会明确失败，不会静默调用或产生费用。

为了避免在最终归档版本中复制一套容易被误启用的付费实现，完整历史快照保存在 Git 对象中：

```powershell
git show 15846bd:transcribe_mic.py
git show 15846bd:qt_gui.py
git show 15846bd:docs/CONFIGURATION.md
git show 15846bd:config.example.json
```

最终源码仍保留一段永不可达的转写迁移快照，便于阅读旧配置与本地数据迁移；入口前置守卫始终拒绝 `model_source: api`。GUI 不再提供 API 模型来源，也不会保存 `api_*` 或 `speaker_diarization_*` 字段。

需要继续使用语音识别时，请迁移到 RabiRoute 的 RabiPC「语音消息端」和本地 RabiSpeech 服务。新的公开代码、参数契约、模型列表和性能报告由 RabiRoute 仓库维护。

此目录不保存 token、个人配置、转写文本、录音或云端请求结果。
