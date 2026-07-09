# 录音与转写架构

本文记录 FenneNote 当前录音、切句、转写、外部文字接入和 RabiRoute 推送链路。目标是让录音事实、转写事实和界面表现各有归口，避免后续把麦克风输入、小爱文本、ASR provider、声纹识别和 UI 反馈混成一团。

## 模块分工

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| `qt_gui.py` | 配置采集、界面状态、外部 HTTP 入口、转写预览、启动/停止后台进程 | 直接做麦克风切句、直接调用本地 Whisper、保存真实模型缓存 |
| `transcribe_mic.py` | 麦克风采样、动态阈值、切句、ASR 调度、转写后处理、写入每日文本、RabiRoute 推送 | 管理 GUI 控件、小爱桥接运行时、外部文字 HTTP 服务 |
| `rabiroute_sdk.py` | 读取 RabiRoute manager 的实例、路由和 Agent 选项 | 保存 FenneNote 转写配置、发送 voice transcript webhook |
| `plugin-adapters/xiaoai-fennenote` | 把小爱/Open-XiaoAI 文本转发给 FenneNote，按规则返回 `ignore` 或 `intercept` | 直接控制 FenneNote GUI、自己做 ASR、保存 FenneNote 转写记录 |

配置的唯一真源是 `config.json`。GUI 通过 `collect_config()` 生成运行配置；后台转写进程只读取启动时写入的配置快照。外部文字模式没有原始音频，因此不会做声纹识别、说话人分离或音频级 TTS 防回流。

## 总体架构

```mermaid
flowchart LR
  subgraph GUI["界面层 qt_gui.py"]
    UI["配置页和状态页"]
    Api["本机 HTTP 入口\n/api/fennenote/*"]
    Preview["转写预览和日志"]
  end

  subgraph Worker["录音转写进程 transcribe_mic.py"]
    Recorder["录音切句器\ncollect_phrases"]
    Provider["识别供应商\n本地 Whisper / DashScope / MiMo"]
    Post["转写后处理\n简繁转换 / 专名修正 / 回声保护"]
    Persist["每日文本\ntranscripts/YYYY-MM-DD.txt"]
  end

  subgraph External["外部文字输入"]
    XiaoAI["小爱音箱 / Open-XiaoAI / 云函数"]
    Bridge["小爱转发桥\nxiaoai-fennenote:8799"]
  end

  subgraph Route["可选下游"]
    RabiRoute["RabiRoute Webhook\n语音转写事件"]
    Manager["RabiRoute 管理端\n路由和 Agent 选项"]
  end

  UI -->|"保存运行配置"| Worker
  Worker -->|"FN_STATUS / FN_TRANSCRIPT"| GUI
  Recorder --> Provider --> Post --> Persist
  Post -->|"语音转写事件"| RabiRoute
  XiaoAI --> Bridge --> Api --> Preview
  Api --> Persist
  Api -->|"外部文字转写事件"| RabiRoute
  UI -->|"刷新/切换绑定"| Manager
```

## 麦克风录音流程

```mermaid
flowchart TD
  A["打开转写"] --> B["GUI 写入 config.json\n启动 transcribe_mic.py"]
  B --> C["音频流读取麦克风片段"]
  C --> D["应用输入增益\n可选混合电脑声音"]
  D --> E{"当前是否还没开始录音"}
  E -->|"还没开始"| F["维护前置缓存\n更新动态底噪"]
  F --> G{"当前音量是否达到录音线"}
  G -->|"未达到"| C
  G -->|"已达到"| H["从前置缓存建立语音片段\n记录开始时间"]
  E -->|"已经开始"| I["追加音频片段\n更新峰值、有效语音和静音时长"]
  H --> I
  I --> J{"是否达到切句条件"}
  J -->|"继续等待"| C
  J -->|"可以切句"| K{"片段峰值是否达到转写线"}
  K -->|"未达到"| L["丢弃低强度片段\n回到监听"]
  K -->|"已达到"| M["送入待转写队列"]
  M --> N["回声保护\n音频级检查"]
  N --> O["识别供应商转写"]
  O --> P["简繁转换和专名修正"]
  P --> Q["文本过滤和回声文本过滤"]
  Q --> R["写入每日文本\n推送 RabiRoute\n发送 GUI 预览"]
  L --> C
  R --> C
```

切句由三组状态共同决定：

- `record_threshold`：进入录音窗口的最低门槛，动态阈值只在未开始录音时抬高它。
- `transcribe_threshold`：片段能否进入 ASR 的门槛，低于它的背景声不会送去转写。
- `transcribe_pause_seconds` / `silence_seconds` / `max_phrase_seconds`：分别处理正常停顿、长时间低声噪音和超长句。

## 外部文字流程

```mermaid
sequenceDiagram
  participant XA as 小爱或上游识别服务
  participant B as 小爱转发桥 8799
  participant G as FenneNote 本机入口 8793
  participant UI as FenneNote 界面层
  participant RR as RabiRoute

  XA->>B: 提交识别文本、设备名和会话号
  B->>G: 转发到小爱文字入口
  G->>UI: 发出外部文字事件
  UI->>UI: 执行专名修正
  UI->>UI: 写入每日文本并刷新预览
  UI-->>RR: 可选推送语音转写事件
  B-->>XA: 返回继续原生响应或拦截
```

`decision` 默认只负责转发并返回 `ignore`，让小爱原生响应继续。只有配置 `XIAOAI_INTERCEPT_REGEX` 且文本命中时，桥接器才返回 `intercept`。

## 维护原则

- 录音采样和切句只放在 `collect_phrases()`，GUI 只显示阈值和状态。
- ASR provider 的差异收敛在 `transcribe_phrase_with_*()`，共同输出纯文本。
- 转写后处理统一走 `apply_transcript_corrections()`，麦克风和外部文字都复用同一规则。
- `append_line()` 是每日文本写入入口，RabiRoute 推送只通过 `post_rabiroute_event()`。
- 外部文字不伪装成音频识别结果；它可以写入和推送，但没有音频、峰值、声纹和 speaker turns。
- 普通麦克风转写停止时不管理小爱桥接进程；只有外部文字模式停止才尝试停止相关桥接运行时。

## 验证清单

发布前至少执行：

```powershell
py -3.10 -m py_compile qt_gui.py transcribe_mic.py rabiroute_sdk.py
node --check plugin-adapters/xiaoai-fennenote/index.mjs
```

有本机依赖时可继续做：

```powershell
.\.venv-gpu\Scripts\python.exe .\transcribe_mic.py --list-devices
npm --prefix .\plugin-adapters\xiaoai-fennenote run smoke
```

真实语音验证重点看三件事：轻声噪音不入队、正常说完后按 `transcribe_pause_seconds` 切句、TTS guard 生效时不会把本机回声写入转写。
