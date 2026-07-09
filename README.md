# FenneNote

<p align="center">
  <img src="assets/fennenote-readme-banner.png" alt="FenneNote README banner" />
</p>

<h3 align="center">芬妮笔记：本地 GPU 实时语音转写工具</h3>

<p align="center">
  常驻监听麦克风，按语音活动自动切段，用本地 GPU Whisper 转写，并按日期写入文本。
</p>

## 这是什么

FenneNote 是一个面向 Windows 桌面的实时语音笔记工具。它适合办公室里做持续语音备忘、会议碎片记录、开发时的口述想法记录，也可以作为 RabiRoute 的语音输入端。

项目形象采用耳廓狐娘方向：大耳朵代表灵敏听觉，耳机麦克风代表实时监听，沙金色和青绿色波形代表温暖但偏工具化的桌面助手气质。

## 功能

- Windows 中文 GUI，支持开始、暂停、停止和实时预览。
- 固定 GPU / CUDA 模式，默认 `small + int8_float16`，兼顾 8GB 显存和 Unity 同时运行。
- Whisper 模型不随工具打包，可在 GUI 的“模型”页下载、删除和检查本地缓存。
- 麦克风音量监测、滚动柱状波形、录音线和转写线。
- 录音触发阈值与转写判定阈值分离，支持动态底噪。
- 触发前保留音频，减少句首丢字。
- 低于转写线一段时间后自动切段并转写。
- 中文输出简体，英文术语保留。
- 按天写入 `transcripts/YYYY-MM-DD.txt`。
- 模型、临时缓存、配置和输出默认都放在安装目录下。
- 可选推送转写事件到 RabiRoute 的 `voice_transcript` 路由。
- 可接收 RabiRoute/Agent 反向消息，并在屏幕左下角弹出气泡。

## 快速上手

要求：

- Windows 10/11
- Python 3.10
- NVIDIA GPU
- CUDA 11.8 运行库，或机器上已有可被脚本加入 `PATH` 的 cuDNN/CUDA DLL

启动 GUI：

```powershell
cd C:\Path\To\FenneNote
.\run_gui.ps1
```

第一次运行会自动创建 `.venv-gpu`，安装 GPU 依赖，并从 `config.example.json` 创建本机 `config.json`。

打开后：

1. 在“输入”页选择麦克风。
2. 保持默认 `small`、`GPU / CUDA`、`int8_float16`。
3. 打开“模型”页，选择并下载需要的 Whisper 模型。
4. 回到“输入”页，确认下拉框选择了要使用的模型。
5. 在主界面确认麦克风波形有变化。
6. 点击“开始”。
7. 说话后在“转写预览”查看文本，完整记录会写入 `transcripts/`。

更多细节见 [快速上手](docs/QUICK_START.md)。

## 常用入口

GUI：

```powershell
.\run_gui.ps1
```

终端版：

```powershell
.\run.ps1
```

打包 Windows exe：

```powershell
.\build_windows.ps1
.\dist\FenneNote\FenneNote.exe
```

列出麦克风设备：

```powershell
.\.venv-gpu\Scripts\python.exe .\transcribe_mic.py --list-devices
```

## 文档

- [快速上手](docs/QUICK_START.md)
- [配置说明](docs/CONFIGURATION.md)
- [QQ 混音虚拟麦克风路由](docs/AUDIO_ROUTING.md)
- [模型说明](docs/MODELS.md)
- [RabiRoute 接入](docs/RABIROUTE.md)
- [录音与转写架构](docs/RECORDING_TRANSCRIPTION_ARCHITECTURE.md)
- [开源发布检查清单](docs/OPEN_SOURCE_CHECKLIST.md)
- [排错指南](docs/TROUBLESHOOTING.md)

## 语音工作站链路

FenneNote 只负责“电脑旁用户语音 -> 转写文本事件”。推荐把完整 voice interaction workstation 拆成下面几段：

```text
FenneNote 麦克风转写
-> RabiRoute /webhook 接收 voice_transcript
-> Codex/Agent 按文本事件处理任务
-> OumuQ/worker 按角色配置生成 TTS 回声
```

边界建议：

- FenneNote 的语音输入代表用户在电脑旁说话，适合触发 Codex 侧角色语音回声。
- QQ、群聊、机器人侧消息默认按文字处理，不应自动回发语音；只有确认来自 FenneNote 的 `voice_transcript` 才进入语音回声链路。
- 日语识别可能不稳定。办公场景建议优先使用简体中文普通话，并在提示词里保留常见技术术语：Unity、Editor、GPU、CPU、AI、Bug、微信、项目、功能、消息、发送、测试。
- RabiRoute、OumuQ、角色 TTS worker 的真实 URL、token、cookie、角色私有配置和公司环境参数都应在部署机器本地填写，不要写进公开仓库。

完整接入说明见 [RabiRoute 接入](docs/RABIROUTE.md)。

## QQ 混音虚拟麦克风

如果 QQ 语音通话里需要朋友同时听到用户真实麦克风和 YeYu/Codex TTS，请使用 VoiceMeeter 或 VB-CABLE + VoiceMeeter 做混音虚拟麦克风。关键规则是：

```text
FenneNote 输入 = 用户物理麦克风
QQ 麦克风 = VoiceMeeter Output/B1 或其他混音虚拟麦克风
```

不要把 FenneNote 的麦克风选择成混音虚拟设备，否则 TTS 可能被再次转写并回流到 Codex。详细拓扑、TTS guard 占位配置和验证步骤见 [QQ 混音虚拟麦克风路由](docs/AUDIO_ROUTING.md)。

## 公开发布安全

公开仓库只保留脱敏模板和部署说明。提交到 GitHub 前请确认没有包含：

- 真实 webhook URL、公网域名、内网地址、token、cookie、API key。
- `config.json`、`.env`、公司环境配置、个人绝对路径。
- `transcripts/`、音频缓存、私有录音、会议记录、运行日志。
- 打包产物、模型缓存、临时文件。

本机运行时请从 `config.example.json` 生成 `config.json`，再在 GUI 或本地配置文件中填写真实参数。`config.json` 已被 `.gitignore` 排除。

## 目录和数据

运行期文件默认保存在程序所在目录：

```text
config.json         本机配置
transcripts/        按天保存的转写文本
cache/models/       Whisper 模型缓存
cache/huggingface/  Hugging Face 下载缓存
cache/temp/         临时文件，只受“缓存保留分钟”清理
cache/audio/        私有录音片段，用于以后训练/克隆自己的声线
```

源码运行和打包版运行会分别读取各自目录下的 `config.json`。如果同时使用 `.\run_gui.ps1` 和 `dist\FenneNote\FenneNote.exe`，它们的配置互不共享。

“保存录音片段”默认关闭，避免误存隐私。开启后，送去转写的片段会保存为 WAV 到 `cache/audio/`；这些音频已经过“麦克风音量放大”处理。`录音保留分钟` 最低 1 分钟，默认 10 分钟，用来给声纹建档、异步识别和人工命名留下安全窗口。`cache/audio/` 属于私有训练素材，不提交 GitHub。

当前 GUI 可管理的模型见 [模型说明](docs/MODELS.md)。默认推荐 `small`，8GB 显存同时运行 Unity 时不建议常驻使用 `large-v3`。

## 开发验证

```powershell
py -3.10 -m py_compile qt_gui.py transcribe_mic.py
```

构建 exe：

```powershell
.\build_windows.ps1
```

`dist/`、`build/`、`.venv-gpu/`、`config.json`、`transcripts/`、`cache/`、`reference-cache/` 不提交到 GitHub。

## 角色资产

- README 主视觉：[assets/fennenote-readme-banner.png](assets/fennenote-readme-banner.png)
- 角色设定图：[assets/fennec-mascot-sheet.png](assets/fennec-mascot-sheet.png)
- 窗口图标：[assets/fennenote.ico](assets/fennenote.ico)
- 状态差分：`assets/fennenote-state-*.png`
