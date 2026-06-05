# 排错指南

## 点开始后没有文字

先看 GUI 状态栏：

- `正在准备模型`：首次运行可能正在下载模型，等待完成。
- `已捕获待转写片段`：麦克风触发正常，模型还在处理或过滤。
- `本段已转写但没有保留有效文字`：识别结果被过滤，多见于环境噪声或幻听短句。

再检查：

1. 麦克风是否选对。
2. 波形是否在说话时明显升高。
3. `录音触发阈值` 是否过高。
4. `转写判定阈值` 是否过高。
5. 说话是否太短。

办公室环境可以先点“灵敏”预设，观察是否能稳定触发。

## 如何管理模型

打开 GUI 的“模型”页，可以下载、检查或删除本地模型缓存。“输入”页的模型下拉框只负责选择转写时要使用的模型。

模型会下载到：

```text
cache/models/
```

这个过程不开始录音，也不加载 GPU 推理。下载完成后再点击“开始”。

模型发布者、下载页、上游模型和性能建议见 [模型说明](MODELS.md)。

## CUDA 运行库缺失

如果看到类似：

```text
Library cublas64_12.dll is not found or cannot be loaded
```

说明 GPU 运行库不完整或没有进入 `PATH`。

处理方式：

1. 使用 `.\run_gui.ps1` 启动，而不是直接双击 Python 文件。
2. 确认安装了 CUDA 11.8 运行库。
3. 如果安装了 NVIDIA Canvas，确认路径存在：

```text
C:\Program Files\NVIDIA Corporation\NVIDIA Canvas
```

脚本会把它加入 `PATH`，用于加载部分 cuDNN DLL。

## 源码版和 exe 配置不一致

这是正常现象。

源码版读取：

```text
C:\Path\To\FenneNote\config.json
```

打包版读取：

```text
C:\Path\To\FenneNote\dist\FenneNote\config.json
```

如果两个入口混用，需要分别保存配置，或手动复制 `config.json`。

## 识别出没说过的英文短句

Whisper 在安静环境、短音频或噪声片段里可能出现幻听，例如 `Thank you`。

可尝试：

- 语言选择 `简体中文`，减少自动语言漂移。
- 保持 `输出简体中文` 开启。
- 提高 `转写判定阈值`。
- 增加 `低于转写线等待秒数`，避免片段太碎。
- 使用更高模型，例如 `medium`，但会增加显存和延迟。

## 模型下载慢或重复下载

模型默认缓存到：

```text
cache/models/
cache/huggingface/
```

不要删除这两个目录，除非你想重新下载模型。

`缓存保留分钟` 只清理 `cache/temp/` 和 `cache/audio/`，不会删除模型缓存。

## RabiRoute 收不到转写

检查 FenneNote：

- “路由”页是否勾选“转写完成后推送到 RabiRoute”。
- 推送 URL 是否是 RabiRoute 的 webhook 地址，例如 `http://127.0.0.1:8791/webhook`。

检查 RabiRoute：

- Gateway 是否启用 `Webhook / FenneNote`。
- Gateway 端口是否和 FenneNote URL 一致。
- 规则是否勾选 `voice_transcript`。
- 数据目录下是否出现 `voice-transcripts.jsonl`。
