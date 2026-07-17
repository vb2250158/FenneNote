# 配置说明（仅本地 ASR）

FenneNote 当前只使用本机 `faster-whisper`，不再提供 DashScope、MiMo 或 OpenAI-compatible ASR。历史 API 实现和迁移前配置由仓库内的 [云端 ASR 历史归档索引](../archive/cloud-api-20260717/README.md) 定位到冻结的 Git 对象。

推荐配置：

```json
{
  "config_version": 6,
  "model_source": "local",
  "model": "small",
  "device": "cuda",
  "compute_type": "int8_float16",
  "language_mode": "zh",
  "simplify_chinese": true,
  "save_audio_segments": false
}
```

GUI 的模型页只管理 `cache/models/` 中的本地 Whisper 模型。默认推荐 `small`；需要更高质量时可尝试 `medium`，但会增加显存占用和延迟。

说话人识别继续使用本地声纹库和本地轻量 embedding。云端说话人分离已停用，因此“说话人字幕”不会触发远程 ASR。

真实配置保存在 `config.json`，不会提交 GitHub。录音、转写、缓存和声纹资料均为本机私有数据。
