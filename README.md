# 0wd-v3 源码仓库

这是从 `repos/main-new-audio-draft-beat` 复制建立的独立 Git 仓库，用于后续音频 encoder 重构。复制来源提交为 `3de8e75706aa4b8a1f486b4a42125faca3214592`。保留的旧模型、运行配置和必要包资源保持来源内容；复制时排除了测试及运行证据。

新建的 `maidata` 包目前仅有可运行入口和空模型位置，参数集中在 `src/maidata/entry.py` 顶部的空 `CONFIG` 字典。架构仍在讨论，没有迁入网络或训练系统。用法见 [最小入口](docs/maidata-entry.md)。

**新 encoder 目前是设计，尚未进入代码。** 设计正本位于 [new-audio-encoder.md](../0wd-v3-refactor-docs/new-audio-encoder.md)。目录名 `0wd-v3` 不改变旧模型的架构或版本兼容值；旧模型版本仍只在 `src/maimai_baseline/runtime-versions.toml` 维护，新空入口不读取该表。

## 当前代码与计划的区别

| 内容 | 当前源快照 | 新 encoder 设计 |
|---|---|---|
| 音频输入 | 外部冻结教师生产的原生缓存 | 可训练的 Mel/beat-patch 输入 |
| 局部音频窗口 | 22 拍：前 4＋目标 16＋后 2 | 24 拍；两侧分配待定 |
| 全曲路径 | 独立 SongAudioEncoder 类仍保留；native 训练步消费缓存 | 全曲 G₀＋24 个动态 S，共享四块 encoder |
| Decoder 音频范围 | native P0 中心窗口拼接的全曲 bank | 当前局部 L/FPN；全曲信息通过 S 进入 |
| 一拍草稿 | 已有整曲 P0 输入的草稿代码 | 是否改读 G₀/S 等尚未决定 |

本次没有改窗口常量、解冻 encoder、替换缓存合同或启动训练。旧 checkpoint 与未来新架构不自动兼容。

## 导航

- [当前源码状态](docs/current-runtime.md)
- [Maidata 最小入口](docs/maidata-entry.md)
- [复制范围、独立 Git 与外部依赖](docs/import-and-layout.md)
- [文档索引](docs/README.md)
- [一拍草稿实现参考](docs/audio-draft.md)
- [训练与恢复](docs/luna-training.md)
- [推理与物化](docs/inference.md)
- [数据与工具](docs/data-and-tools.md)
- [行内字段与 Physical](docs/physical.md)
- [版本维护](docs/runtime-versioning.md)
- [历史规格](docs/history/index.md)

## 目录与开发入口

当前目录是 Git 根；其下 `src/` 才是 Python 包根。新骨架为 `maidata`；保留包为 `maimai_baseline`、`maimai_dataset`、`maimai_chart_ir`。新入口只依赖标准库，旧模型依赖仍在 `pyproject.toml` 和 `uv.lock` 中声明。

```powershell
Set-Location D:/01YD/01code/codex/maimai-ai-charting/repos-0wd-v3/src
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m maidata --help
python -m maidata --show-config
```

本轮只验证新入口的帮助、配置展示和未实现提示，没有执行旧模型 CLI、安装环境或训练。测试目录、fixture、权重、数据、缓存和运行产物没有复制。旧配置和远端 launcher 是来源参考，不是本工作区的新实验默认命令；使用前先读当前源码状态与外部依赖说明。
