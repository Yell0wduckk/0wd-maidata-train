# 0wd-v3 源码仓库

这是从 `repos/main-new-audio-draft-beat` 复制建立的独立 Git 仓库，用于后续音频 encoder 重构。复制来源提交为 `3de8e75706aa4b8a1f486b4a42125faca3214592`。保留的旧模型、运行配置和必要包资源保持来源内容；复制时排除了测试及运行证据。

新 `maidata` 包已实现独立、可训练的音频 Encoder。18 项架构配置集中在 `src/maidata/entry.py` 顶部的普通 `CONFIG` 字典；构造器无第二套默认值，尺寸随配置推导。用法见 [Encoder 入口与接口](docs/maidata-entry.md)。

**当前交付包括 Encoder、独立音频输入生产和字段/Physical数据规则层。** frontend、geometry、attention、blocks、pyramid、encoder 和 types 已实现；Encoder公开结果仅有 P0/P1/P2，支持按曲准备 G0 和按窗编码。`maidata.preparation` 生产直接拍网格 Mel、全曲 aux32、坐标和 mask，并构造 SongInputs/WindowInputs，见[输入缓存说明](docs/encoder-audio-cache.md)。Decoder网络、草稿、训练循环、loss 和 checkpoint 尚未接入。设计正本位于 [new-audio-encoder.md](../0wd-v3-refactor-docs/new-audio-encoder.md)。目录名 `0wd-v3` 不改变旧模型的架构或版本兼容值；旧模型版本仍只在 `src/maimai_baseline/runtime-versions.toml` 维护，新 Encoder 不读取该表。

独立的 `maidata.chart`、`maidata.time` 与 `maidata.physical` 已实现新字段的数据与规则层：IR编解码、Route-71连续链/分支、时间标签、Physical候选查询及动作回退。它们不实现可学习字段模块或Decoder网络，说明见[新字段与Physical](docs/chart-fields-physical.md)。

仅覆盖 Encoder 的模块、接口与验证要求见 [实施文档](../0wd-v3-refactor-docs/encoder-execution-plan.md)。

## 保留的旧模型与新 Encoder

| 内容 | 保留的旧模型 | 已实现的新 Encoder |
|---|---|---|
| 音频输入 | 外部冻结教师生产的原生缓存 | 消费准备好的 Mel/aux32，前端与 beat-patch tokenizer 可训练 |
| 局部音频窗口 | 22 拍：前 4＋目标 16＋后 2 | 默认前 4＋目标 16＋后 4＝24 拍，可配置 |
| 全曲路径 | 独立 SongAudioEncoder 类仍保留；native 训练步消费缓存 | 全曲 G₀＋每拍一个动态 S；默认24槽与1+1+2共享块 |
| Decoder 音频范围 | native P0 中心窗口拼接的全曲 bank | 输出当前局部 P0/P1/P2；全曲信息通过 S 进入，Decoder 尚未实现 |
| 一拍草稿 | 已有整曲 P0 输入的草稿代码 | 尚未接入；是否改读 G₀/S 等另行确定 |

本次没有改旧模型、替换缓存合同或启动训练。旧 checkpoint 与隐藏缓存不自动兼容新架构；内容未变的已验收原始音频/Mel 不因新源码失效。

## 导航

- [当前源码状态](docs/current-runtime.md)
- [Maidata Encoder 入口与张量接口](docs/maidata-entry.md)
- [全官谱音频输入缓存](docs/encoder-audio-cache.md)
- [新字段、时间与 Physical](docs/chart-fields-physical.md)
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

当前目录是 Git 根；其下 `src/` 才是 Python 包根。新 Encoder 位于 `maidata`；保留包为 `maimai_baseline`、`maimai_dataset`、`maimai_chart_ir`。帮助和配置展示只依赖标准库，实际构建 Encoder 时延迟导入 Torch；新包不依赖旧模型包或参考仓库。依赖仍在 `pyproject.toml` 和 `uv.lock` 中声明。

```powershell
Set-Location D:/01YD/01code/codex/maimai-ai-charting/repos-0wd-v3/src
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m maidata --help
python -m maidata --show-config
python -m maidata
```

默认命令成功构建 `AudioEncoder` 并说明 Decoder 与训练尚未实现。小型 CPU 合成测试及命令见 [入口文档](docs/maidata-entry.md#cpu-合成验证)；不代表 CUDA、真实资产、训练或整模型推理已验证。旧测试、fixture、权重、数据、缓存和运行产物没有复制；本次新增测试只覆盖新 Encoder。旧配置和远端 launcher 是来源参考，不是新实验默认命令。
