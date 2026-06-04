# 快速上手

这份文档从空环境开始，把 FenneNote 跑起来。

## 1. 准备环境

推荐环境：

- Windows 10/11
- Python 3.10
- NVIDIA GPU
- CUDA 11.8 运行库

项目脚本会尝试把下面路径加入 `PATH`：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA Corporation\NVIDIA Canvas
```

第二个路径用于复用 NVIDIA Canvas 自带的 cuDNN DLL。如果你的机器没有这些路径，仍可运行脚本，但 CUDA 依赖缺失时会在 GUI 状态栏里提示。

## 2. 启动 GUI

```powershell
cd C:\Data\CottonProject\FenneNote
.\run_gui.ps1
```

脚本会做这些事：

1. 创建 `.venv-gpu`。
2. 安装 `requirements-gpu-cu11.txt`。
3. 安装 `faster-whisper==0.10.1`。
4. 启动中文 GUI。

首次启动会从 `config.example.json` 复制生成 `config.json`。

## 3. 开始转写

1. 打开“输入”页。
2. 选择当前使用的麦克风。
3. 模型保持 `small`。
4. 计算精度保持 `int8_float16`。
5. 语言按需求选择，默认是简体中文。
6. 打开“模型”页，点击当前模型旁边的“下载”。
7. 需要切换模型时，在“模型”页点“选择”，或回到“输入”页使用模型下拉框。
8. 回到主界面确认波形在动。
9. 点击“开始”。

模型不随工具打包。“模型”页会把模型下载到 `cache/models/`，也可以删除不再使用的本地模型缓存。如果没有提前安装，点击“开始”时也会自动检查并下载当前模型。

## 4. 查看结果

GUI 右侧会显示转写预览。

完整文本按日期保存：

```text
transcripts/YYYY-MM-DD.txt
```

实时查看：

```powershell
Get-Content .\transcripts\2026-06-03.txt -Wait -Encoding UTF8
```

把日期换成当天即可。

## 5. 打包 exe

```powershell
.\build_windows.ps1
```

打包完成后启动：

```powershell
.\dist\FenneNote\FenneNote.exe
```

打包版会使用 `dist\FenneNote\config.json`。如果你之前用源码版调过配置，需要在打包版里重新保存一次，或手动复制配置。

打包版不包含 Whisper 模型。用户第一次使用时仍需要通过“模型”页或“开始”下载模型到安装目录下的 `cache/models/`。
