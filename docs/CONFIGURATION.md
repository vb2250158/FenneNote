# 配置说明

FenneNote 的配置文件是程序所在目录下的 `config.json`。如果文件不存在，程序会从 `config.example.json` 创建。

`config.example.json` 是公开脱敏模板，真实部署参数只应写入本机 `config.json`。不要把公司环境 webhook、token、cookie、API key、个人路径、录音或日志提交到 GitHub。

## 推荐默认值

默认配置偏向“8GB 显存 + 同时运行 Unity”的场景：

```json
{
  "model": "small",
  "device": "cuda",
  "compute_type": "int8_float16",
  "language_mode": "zh",
  "simplify_chinese": true,
  "input_gain": 1.0,
  "output_dir": "transcripts",
  "cache_dir": "cache",
  "cache_retention_minutes": 0.0,
  "save_audio_segments": false,
  "audio_retention_minutes": 10.0
}
```

## GUI 分页

输入页：

- `麦克风`：选择输入设备。
- `音频路由预设`：记录当前路由意图。QQ 混音模式下，FenneNote 仍应选择物理麦克风，QQ 才选择 VoiceMeeter Output/B1 或其他混音虚拟麦克风。
- `混合麦克风和电脑声音用于转写`：开启后，FenneNote 会把麦克风和另一个“电脑播放声输入”混合成一段音频，再送去转写。
- `电脑声音输入`：选择能代表电脑播放声的输入设备，例如 `立体声混音`、虚拟声卡 Input、Steam/网易/AudioRelay 的回放输入。当前版本不创建 Windows 系统级虚拟麦克风。
- `电脑声音增益`：电脑播放声混入转写前的音量倍数，默认 `1.0`。
- `模型`：默认 `small`。质量不够可试 `medium`，但显存和延迟会增加。
- `运行设备`：固定 `GPU / CUDA`。
- `计算精度`：默认 `int8_float16`。
- `语言`：可选自动识别、简体中文、英文、日文、韩文。
- `输出简体中文`：开启后繁体会转简体，英文术语保留。

模型页：

- 顶部性能表：按系列展示参数量、显存参考、相对速度和 FenneNote 建议。
- 当前模型资料：展示发布者、仓库、下载/说明页、上游模型和中文用途说明。
- `选择`：把该模型写入当前配置，并同步到输入页下拉框。
- `下载`：下载未安装模型到 `cache/models/`。
- `检查`：已安装模型再次检查本地缓存。
- `删除`：删除该模型的本地 Hugging Face 缓存目录。

触发页：

- `自动适应环境噪声`：开启后会根据环境底噪抬高开始录音的音量线。
- `麦克风音量放大`：默认 `1.0`，范围 `1.0` 到 `5.0`。保存录音片段时，保存的是经过这个放大后的音频。
- `开始录音的音量线`：超过后开始截取音频。
- `值得转写的音量线`：本段峰值超过后才送去转写。
- `句首多留几秒`：把触发前一小段音频也带上，减少句首丢字。
- `最短有效录音`：短于这个时间的片段会被过滤。
- `安静多久后切句`：已经达到值得转写音量线后，安静超过这个时间就切段。
- `安静多久后丢弃杂音`：如果这段声音一直没达到值得转写的音量线，安静超过这个时间后，当作杂音丢掉。
- `单段最长录音`：防止一段音频无限增长。

应用页：

- `本地路径`：集中管理本机文件位置。相对路径按程序所在目录解析，跨平台部署时可改成任意本机绝对路径。
- `转写记录目录`：按日期长期保存转写 TXT 的目录，例如 `transcripts/`。这不是缓存，不受缓存保留时间清理。
- `模型/运行缓存目录`：本地缓存根目录。模型、Hugging Face 下载缓存、运行临时文件和录音片段默认都放在这里的子目录中。
- `录音片段目录`：默认是 `cache/audio/`，由缓存根目录派生。只在开启“保存录音片段”后写入 WAV。
- `本地模型缓存`：默认是 `cache/models/`，模型页的下载、检查、删除作用在这里。
- `运行临时缓存`：默认是 `cache/temp/`，只受“运行临时缓存保留分钟”清理。
- `TTS guard 文件`：默认是 `cache/tts_guard.json`，供 TTS 侧写入防回流状态。
- `配置文件`：打开本机 `config.json`。
- `运行日志`：打开当前 GUI 会话的日志抽屉。
- `启动后自动开始`：打开程序后自动开始转写。
- `保存录音片段`：默认关闭，避免误存隐私。开启后，每段送去转写的麦克风音频会保存到录音片段目录，可作为以后训练或克隆自己声线的私有素材。
- `录音保留分钟`：只影响录音片段目录里的 WAV。最低 `1` 分钟，默认 `10` 分钟。这个下限是为了给声纹建档、异步说话人分离、声纹比对和人工命名留下安全窗口，避免录音刚生成就被清理。
- `运行临时缓存保留分钟`：只清理运行临时缓存目录，不会删除模型缓存、转写记录，也不会删除录音片段。
- `启用左下角气泡`：接收反向路由消息后弹桌面气泡，默认开启。
- `监听端口`：气泡回调端口，默认 `8792`。
- `持续秒数`：气泡显示时间，默认 `3.0` 秒。
- `访问令牌`：可选。填写后，反向回调必须带 `X-FenneNote-Token` 请求头。

OumuQ 扩展页：

- `OumuQ URL`：OumuQ 路由层 `/api/speak` 或兼容 worker `/speak` 地址。默认示例是 `http://127.0.0.1:8780/api/speak`。
- `角色注册表`：OumuQ 的 `voice-references/reference-index.json`。FenneNote 用它显示角色名、音色和引擎信息。
- `语音语言`：默认 `auto`，按测试文本自动推断；也可以固定为 `Chinese` / `Japanese` / `English`，避免角色默认语言把中文测试文本当成日语。
- `角色/音色 ID`：传给 OumuQ 的 `character_id`，例如本机角色注册表里的 `tamamo_no_mae`。可以从注册表选择，也可以手动输入。
- `角色名`：从 OumuQ 注册表读取出的显示名、引擎和语音语言。
- `请求 OumuQ/worker 播放`：控制请求体里的 `play` 字段。
- `guard 秒数`：提交播放前写入 `TTS guard 文件` 的保护窗口。实际转写防回流仍需要在应用页启用 `TTS guard`。
- `测试文本`：手动提交一段短文本到 OumuQ，用于验证播放桥和 guard 写入。

这一页只放 OumuQ 相关扩展。FenneNote 不依赖 OumuQ 才能转写，OumuQ 也不依赖 FenneNote 才能播放；两者联动时，FenneNote 负责写防回流 guard 和提交播放请求，OumuQ/worker 负责生成与播放。

声纹识别页：

- `声纹库`：FenneNote 本地 speaker registry。当前阶段会显示用户本人、手动导入样本和从 OumuQ 角色自动生成的 TTS speaker 映射。
- `名称` / `保存名称`：选中一个声纹档案后，可以给自动生成的 `unknown_*` 或 `tts_<character_id>` 档案写入人类可读名称。
- `添加本人声纹`：录制一段本机麦克风样本，保存到 speaker registry，并计算本地轻量 embedding。
- `导入样本`：给选中的 speaker profile 导入 WAV/MP3/FLAC/M4A/OGG 样本，并重算 embedding。
- `重算声纹`：根据选中 profile 的所有样本重新计算 embedding。
- `测试识别`：选择一段音频，和已建模 profile 做整段相似度匹配，显示 Top 匹配结果。
- `启用声纹识别`：开启后才允许后续转写流程对录音片段做匹配。当前 GUI 已支持本地轻量频谱 embedding；正式拦截前仍建议用真人/TTS 样本校准阈值。
- `启用说话人字幕`：开启后使用后续说话人分离字幕流程。当前版本保存这个开关；后端接入阿里非实时 ASR + `diarization_enabled` 后，会按 `speaker_id + begin_time/end_time + text` 生成带说话人的字幕/分段结果。
- `识别机制`：整段匹配只能判断整段像谁，一段录音里多种音色需要先做 speaker diarization 再逐段匹配。

这一页只放声纹识别和说话人相关能力。OumuQ 只作为可能的 TTS 样本来源，不属于声纹识别本身。

RabiRoute 扩展页：

- `转写完成后推送到 RabiRoute`：开启后，每段有效转写都会推送到 RabiRoute。
- `推送 URL`：默认 `http://127.0.0.1:8791/webhook`。
- `来源 ID`：默认 `fennenote`。
- `访问令牌`：可选。填写后会用 `Authorization: Bearer <token>` 发送。

公司环境部署时，`推送 URL` 和 `访问令牌` 在部署机器本地填写。公开仓库不要写入真实内网地址、公网域名或任何密钥。

## 语言与提示词

FenneNote 默认面向简体中文普通话办公场景。推荐提示词保留常见英文技术术语：

```text
以下是简体中文普通话办公场景转写，可能包含 Unity、Editor、GPU、CPU、AI、Bug、微信、项目、功能、消息、发送、测试等术语。请保持中文为简体，英文术语保留英文。
```

日语识别可能不稳定，尤其是中文、英文技术词和日语混合输入。需要日语时建议建立单独本地配置做测试，不要把未验证的日语提示词作为公开默认值。

## QQ 与语音边界

FenneNote 代表电脑旁麦克风输入，可以把 `voice_transcript` 交给 RabiRoute、Codex/Agent，再由 OumuQ 或 TTS worker 生成角色语音回声。

QQ、微信群聊或机器人平台消息默认应只按文字处理，不自动发语音。只有明确来自 FenneNote 的 `voice_transcript`，才建议恢复 Codex 侧语音输出。

## QQ 混音虚拟麦克风

QQ 语音通话里如果需要朋友同时听到用户真实麦克风和 YeYu/Codex TTS，请把 FenneNote 和 QQ 的输入设备分开：

```text
FenneNote 输入 = 用户物理麦克风
QQ 麦克风 = VoiceMeeter Output/B1 或其他混音虚拟麦克风
```

推荐 `audio_route_preset`：

```json
{
  "audio_route_preset": "qq_mixed_output_mode",
  "mixed_input_enabled": false,
  "system_audio_device": null,
  "system_audio_gain": 1.0
}
```

`qq_mixed_output_mode` 这个预设只记录路由意图并在 GUI 里显示说明，不会安装驱动，也不会自动修改 Windows、QQ、VoiceMeeter 或 VB-CABLE 设置。完整拓扑和验证步骤见 [QQ 混音虚拟麦克风路由](AUDIO_ROUTING.md)。

如果只是要“FenneNote 自己转写时同时听见麦克风和电脑播放声”，开启 `mixed_input_enabled`，再选择一个电脑播放声输入设备即可；它是 FenneNote 内部混音转写，不会把混合音频暴露成 QQ/Discord 可选的系统麦克风。

可选的 TTS 防回流辅助配置：

```json
{
  "tts_guard_enabled": false,
  "tts_guard_file": "cache/tts_guard.json",
  "tts_guard_resume_margin_seconds": 0.8,
  "tts_guard_recent_text_window_seconds": 20.0,
  "tts_guard_similarity_threshold": 0.86,
  "oumuq_url": "http://127.0.0.1:8780/api/speak",
  "oumuq_registry_path": "../OumuQ/voice-references/reference-index.json",
  "oumuq_character_id": "",
  "oumuq_language": "auto",
  "oumuq_play": true,
  "oumuq_guard_seconds": 8.0,
  "speaker_registry_file": "cache/speakers/speaker_registry.json",
  "speaker_recognition_enabled": false,
  "speaker_subtitle_enabled": false,
  "speaker_auto_enroll_enabled": true,
  "speaker_match_threshold": 0.92,
  "speaker_unknown_prefix": "unknown"
}
```

`speaker_recognition_enabled` 是 GUI “声纹识别”页里的“启用声纹识别”总开关。默认关闭；勾选后，点击“开始”的平时录音会在切段后保存转写片段、计算本地轻量 embedding、匹配声纹库，并把 `speaker_id` / `speaker_name` / `speaker_confidence` 写入转写结果和 RabiRoute 事件。

`speaker_auto_enroll_enabled` 表示未匹配到已知声纹时自动创建 `unknown_*` 待命名档案。关闭后只把该段标记为 unknown，不写入新 speaker profile。

`speaker_match_threshold` 是声纹相似度阈值。数值越高越不容易认错人，但更容易把同一个人判成新声纹；默认 `0.92`，建议先用本人样本和 TTS 样本校准后再调高或调低。

当前声纹识别是“整段音频像谁”的匹配：如果一段录音里混入两个人，需要后续的说话人分离字幕流程先按时间段切开，再逐段匹配声纹。

默认关闭。开启后，TTS 侧可以写入 `cache/tts_guard.json`，用 `ignore_until` 临时跳过输入，或用 `text` / `tts_text` / `recent_texts` 提供最近 TTS 文本供 FenneNote 丢弃疑似回声。设备隔离仍是主防线。

## 阈值逻辑

```text
麦克风持续监听
-> 超过录音线：开始截取音频，并带上触发前保留音频
-> 录音过程中如果峰值超过转写线：标记为值得转写
-> 低于转写线达到等待秒数：切段并送 GPU Whisper
-> 如果一直没达到转写线：按“安静多久后丢弃杂音”丢掉
```

办公室环境建议先用默认值。如果讲话经常不触发，先把 `开始录音的音量线` 调低到 `0.010`，再观察波形和录音线。

## 配置版本

`config.json` 包含 `config_version`。版本一致时，GUI 保存的参数会继续生效；当代码里的配置版本升级时，会重置为新的默认参数，只保留少量个人偏好。

当前跨版本保留：

- `mic_device`
- `auto_start`

## 模型缓存

Whisper 模型不随工具打包，也不会提交到 GitHub。

模型下载位置：

```text
cache/models/
```

GUI 的“模型”页只下载或删除模型文件，不开始录音，也不加载 GPU 推理。没有提前安装时，点击“开始”也会自动下载并加载当前模型。

当前可直接管理的模型名称：

```text
tiny.en, tiny, base.en, base, small.en, small,
medium.en, medium, large-v1, large-v2, large-v3, large
```

完整发布者、下载连接和性能建议见 [模型说明](MODELS.md)。

从配置版本 5 开始，`vad_filter` 默认关闭。FenneNote 使用自己的录音线和转写线做切段，不再要求打包内置 Silero VAD 模型。
