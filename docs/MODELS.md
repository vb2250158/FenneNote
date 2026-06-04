# 模型说明

FenneNote 使用的是 SYSTRAN 发布的 faster-whisper 转换模型。它们来自 OpenAI Whisper，转换为 CTranslate2 格式后由 `faster-whisper` 加载，本工具会下载到安装目录下的 `cache/models/`。

## 性能速览

参数量、显存参考和相对速度来自 OpenAI Whisper 的模型规格表。实际占用会受 `compute_type`、驱动、CUDA/cuDNN、当前桌面负载和是否同时运行 Unity 影响。

| 系列 | 参数量 | 显存参考 | 相对速度 | FenneNote 建议 |
| --- | ---: | ---: | ---: | --- |
| tiny | 39M | 约 1 GB | 约 10x | 极轻量测试，中文质量较弱 |
| base | 74M | 约 1 GB | 约 7x | 快速草稿，短句可用 |
| small | 244M | 约 2 GB | 约 4x | 默认推荐，适合 8GB 显存常驻 |
| medium | 769M | 约 5 GB | 约 2x | 质量更好，占用和延迟更高 |
| large | 1550M | 约 10 GB | 1x | 准确率优先，8GB + Unity 不建议常驻 |

## 可管理模型

下载连接指向 Hugging Face 模型页。这个页面同时是模型说明页；GUI 会通过 Hugging Face Hub 下载整个模型仓库，不是下载单个文件。

| 模型 | 语言 | 发布者 | 下载/说明页 | 上游模型 | FenneNote 建议 |
| --- | --- | --- | --- | --- | --- |
| `tiny.en` | 仅英文 | SYSTRAN | [Systran/faster-whisper-tiny.en](https://huggingface.co/Systran/faster-whisper-tiny.en) | [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en) | 英文实时草稿和功能测试，不适合中文 |
| `tiny` | 多语言，含中文 | SYSTRAN | [Systran/faster-whisper-tiny](https://huggingface.co/Systran/faster-whisper-tiny) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 最快的中文可用模型，适合确认流程 |
| `base.en` | 仅英文 | SYSTRAN | [Systran/faster-whisper-base.en](https://huggingface.co/Systran/faster-whisper-base.en) | [openai/whisper-base.en](https://huggingface.co/openai/whisper-base.en) | 英文短句更稳，中文场景不要选 |
| `base` | 多语言，含中文 | SYSTRAN | [Systran/faster-whisper-base](https://huggingface.co/Systran/faster-whisper-base) | [openai/whisper-base](https://huggingface.co/openai/whisper-base) | 低占用中文草稿，质量明显弱于 small |
| `small.en` | 仅英文 | SYSTRAN | [Systran/faster-whisper-small.en](https://huggingface.co/Systran/faster-whisper-small.en) | [openai/whisper-small.en](https://huggingface.co/openai/whisper-small.en) | 英文场景轻量推荐，中文场景不要选 |
| `small` | 多语言，含中文 | SYSTRAN | [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 默认推荐，速度、质量和 8GB 显存占用比较均衡 |
| `medium.en` | 仅英文 | SYSTRAN | [Systran/faster-whisper-medium.en](https://huggingface.co/Systran/faster-whisper-medium.en) | [openai/whisper-medium.en](https://huggingface.co/openai/whisper-medium.en) | 英文质量优先，显存和延迟高于 small |
| `medium` | 多语言，含中文 | SYSTRAN | [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium) | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) | 中文质量优先时可选，同开 Unity 时观察显存 |
| `large-v1` | 多语言，含中文 | SYSTRAN | [Systran/faster-whisper-large-v1](https://huggingface.co/Systran/faster-whisper-large-v1) | [openai/whisper-large](https://huggingface.co/openai/whisper-large) | 第一版 large，历史兼容用途为主 |
| `large-v2` | 多语言，含中文 | SYSTRAN | [Systran/faster-whisper-large-v2](https://huggingface.co/Systran/faster-whisper-large-v2) | [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) | 准确率优先，但不适合 8GB 常驻 |
| `large-v3` | 多语言，含中文 | SYSTRAN | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 质量优先；8GB 显存同时开 Unity 时慎用 |
| `large` | 多语言，含中文 | SYSTRAN | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | `large-v3` 的别名，效果和 large-v3 相同 |

## 资料来源

- [SYSTRAN faster-whisper 项目主页](https://github.com/SYSTRAN/faster-whisper)
- [SYSTRAN Hugging Face 模型页](https://huggingface.co/Systran)
- [OpenAI Whisper 模型规格表](https://github.com/openai/whisper#available-models-and-languages)
