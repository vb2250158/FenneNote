# FenneNote

<p align="center">
  <img src="assets/fennenote-readme-banner.png" alt="FenneNote README banner" />
</p>

<h3 align="center">芬妮笔记：耳廓狐娘实时语音转写工具</h3>

<p align="center">
  本地麦克风常驻监听，GPU 实时转写，按天写入文本。适合办公室里做持续语音备忘和临时记录。
</p>

## 形象设定

**FenneNote / 芬妮笔记** 的 mascot 采用耳廓狐娘方向：大耳朵代表灵敏听觉，耳机麦克风代表实时监听，沙金色与青绿色波形代表温暖但偏工具化的桌面助手气质。

- 名称：`FenneNote` / `芬妮笔记`
- 主视觉：暖白、沙金、青绿色音频波形
- 图标方向：耳朵 + 声波，适合小尺寸显示
- 完整设定图：[assets/fennec-mascot-sheet.png](assets/fennec-mascot-sheet.png)

## 功能

- 中文 GUI，支持开始、暂停、停止和实时预览
- 固定 GPU/CUDA 模式，不提供 CPU 模式，避免长时间转写拖慢整机
- 麦克风音量监测、滚动柱状音量、录音线和转写线
- 录音触发阈值与转写判定阈值分离
- 触发前预留音频，避免丢掉句首
- 低于转写线一段时间后自动切段并转写
- 中文转简体，英文术语保留
- 按日期写入 `transcripts/YYYY-MM-DD.txt`
- 配置可保存，支持配置版本迁移和启动后自动开始

## 启动

```powershell
cd C:\Data\CottonProject\GameDevelopmentSkills\voice-transcriber
.\run_gui.ps1
```

首次运行会创建 `.venv-gpu`，安装 GPU 兼容依赖，并读取 `config.json`。当前本机方案使用 Python 3.10、CUDA 11.8 和 NVIDIA Canvas 自带 cuDNN 路径。

终端版：

```powershell
.\run.ps1
```

## 推荐配置

默认配置偏向 Unity 同时运行时的性能：

```json
{
  "model": "small",
  "device": "cuda",
  "compute_type": "int8_float16",
  "language_mode": "zh",
  "simplify_chinese": true
}
```

如果识别质量不够，可以在 GUI 里把模型切到 `medium`；如果 Unity 明显卡顿，先回到 `small`。

## 阈值逻辑

这个工具把“录音”和“转写”拆成两层判断：

```text
麦克风持续监听
→ 超过录音线：开始截取音频，并带上触发前预留秒数
→ 录音过程中如果超过转写线：标记为值得转写
→ 低于转写线达到等待秒数：切段并送 GPU Whisper
→ 如果一直没达到转写线：按噪声丢弃等待秒数丢掉
```

常用调参：

- `麦克风增益`：声音太小先调这个
- `录音触发阈值`：决定什么时候开始截音频
- `转写判定阈值`：决定这一段是否值得送去识别
- `低于转写线等待秒数`：越短出字越快，越长越不容易把一句话切碎

## 配置版本

`config.json` 包含 `config_version`。版本一致时，GUI 保存的参数会继续生效；当代码里的配置版本升级时，会重置为新的默认参数，只保留个人偏好：

- `mic_device`
- `auto_start`

GUI 的“启动后自动开始”勾选会写入 `auto_start`。勾选后，下次打开 FenneNote 会自动开始监听和转写。

## 输出

转写文本按天保存：

```text
voice-transcriber/transcripts/2026-06-03.txt
```

实时查看：

```powershell
Get-Content .\transcripts\2026-06-03.txt -Wait -Encoding UTF8
```

## 麦克风选择

列出音频设备：

```powershell
.\.venv-gpu\Scripts\python.exe .\transcribe_mic.py --list-devices
```

然后在 GUI 中选择目标麦克风，或修改 `config.json` 里的 `mic_device`。
