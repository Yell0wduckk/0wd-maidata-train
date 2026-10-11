# 0wd-v3 源码仓库

当前主 Decoder 变更以 [独立 CONT 序列合同](docs/decoder-cont-token-plan.md) 为准：删除 FieldReader，CONT 写入主 KV，窗口升级 v5。
这是从 `repos/main-new-audio-draft-beat` 复制建立的独立 Git 仓库，用于后续音频 encoder 重构。复制来源提交为 `3de8e75706aa4b8a1f486b4a42125faca3214592`。保留的旧模型、运行配置和必要包资源保持来源内容；复制时排除了测试及运行证据。

新 `maidata` 包已实现独立、可训练的音频 Encoder、Draft 与 Decoder。架构配置集中在 `src/maidata/entry.py` 顶部的普通 `CONFIG` 字典；构造器无第二套默认值，尺寸随配置推导。用法见 [Encoder 入口与接口](docs/maidata-entry.md)、[Draft 接入](docs/draft.md)和[Decoder 接入](docs/decoder.md)。

**当前交付包括 Encoder、Draft D/R/P、Decoder、独立音频输入生产、字段/Physical数据规则层和窗口采样计划生成。** Encoder公开结果仅有 P0/P1/P2，支持按曲准备 G0 和按窗编码。`maidata.preparation` 生产直接拍网格 Mel、全曲 aux32、坐标和 mask，并构造 SongInputs/WindowInputs，见[输入缓存说明](docs/encoder-audio-cache.md)。`maidata.draft` 独立实现全曲P0组合、逐拍D、重建R、Planner P、标签适配及辅助loss。`maidata.sampling` 保留完整基础窗及空尾窗，对复杂节奏窗口总采样2倍，按完整音频组输出引用计划，见[采样说明](docs/sampling.md)。`maidata.decoder`实现独立网络、内容小头、时间分布及主/Physical/几何loss；阶段二训练与checkpoint由独立的`maidata.decoder_training`接入，整曲推理与MiaCode物化由独立的[`maidata.inference`](docs/maidata-inference.md)提供。设计正本位于 [new-audio-encoder.md](../0wd-v3-refactor-docs/new-audio-encoder.md)和[new-draft.md](../0wd-v3-refactor-docs/new-draft.md)。目录名 `0wd-v3` 不改变旧模型的架构或版本兼容值；旧模型版本仍只在 `src/maimai_baseline/runtime-versions.toml` 维护，新模块不读取该表。

独立的 `maidata.chart`、`maidata.time` 与 `maidata.physical` 已实现新字段的数据与规则层：IR编解码、Route-71连续链/分支、时间标签、Physical候选查询及动作回退。它们不实现可学习字段模块或Decoder网络，说明见[新字段与Physical](docs/chart-fields-physical.md)。

当前拆分式D直接输出最终LayerNorm后的256维Z，R直接索引所属拍并通过共享残差MLP解析，MOMENT采用拍内96类，duration退出Draft监督；P保留目标全部16拍。阶段一checkpoint升级v3，详见[实施记录](docs/drafter-split-implementation.md)。主M删除额外FieldReader，普通字段读取主干，三类时间保留385类并将CONT展开为独立主序列输入。训练默认GT plan，推理用P预测；新窗口v5与模型序列缓存的增量/从零生产见[迁移命令](docs/time384-cache-migration.md)。源原谱和Physical仍为v2。

仅覆盖 Encoder 的模块、接口与验证要求见 [实施文档](../0wd-v3-refactor-docs/encoder-execution-plan.md)。

## 保留的旧模型与新 Encoder

| 内容 | 保留的旧模型 | 已实现的新 Encoder |
|---|---|---|
| 音频输入 | 外部冻结教师生产的原生缓存 | 消费准备好的 Mel/aux32，前端与 beat-patch tokenizer 可训练 |
| 局部音频窗口 | 22 拍：前 4＋目标 16＋后 2 | 默认前 4＋目标 16＋后 4＝24 拍，可配置 |
| 全曲路径 | 独立 SongAudioEncoder 类仍保留；native 训练步消费缓存 | 全曲 G₀＋每拍一个动态 S；默认24槽与1+1+2共享块 |
| Decoder 音频范围 | native P0 中心窗口拼接的全曲 bank | Encoder输出局部P0/P1/P2，独立Decoder读取公开P0中心拼接的全曲bank |
| 一拍草稿 | 已有整曲 P0 输入的草稿代码 | 公开P0中心拼接为全曲bank，独立D输出逐拍Z，R/P分别读出；独立阶段一与阶段二训练入口已实现 |

本次没有改旧模型、替换缓存合同或启动训练。旧 checkpoint 与隐藏缓存不自动兼容新架构；内容未变的已验收原始音频/Mel 不因新源码失效。

## 导航

- [当前源码状态](docs/current-runtime.md)
- [Maidata Encoder 入口与张量接口](docs/maidata-entry.md)
- [Draft D/R/P、窗口、标签与损失接口](docs/draft.md)
- [冻结主 Decoder ckpt45 的 P0 与 Drafter 50/100 轮训练](docs/drafter-frozen-p0-training.md)
- [Decoder、Router/MMoE、小头、窗口与损失接口](docs/decoder.md)
- [主 Decoder 阶段二训练与恢复](docs/decoder-training.md)
- [冻结 Encoder/Drafter、读取 P0/Z 的 Decoder 适配入口](docs/decoder-frozen-adaptation.md)
- [384 tick 时间监督缓存迁移与从零生产](docs/time384-cache-migration.md)
- [当前模型整曲推理与可播放 maidata 导出](docs/maidata-inference.md)
- [全官谱音频输入缓存](docs/encoder-audio-cache.md)
- [新字段、时间与 Physical](docs/chart-fields-physical.md)
- [全官谱原谱全曲缓存](docs/chart-cache.md)
- [基础窗口与复杂节奏采样](docs/sampling.md)
- [复制范围、独立 Git 与外部依赖](docs/import-and-layout.md)
- [文档索引](docs/README.md)
- [一拍草稿实现参考](docs/audio-draft.md)
- [训练与恢复](docs/luna-training.md)
- [旧模型推理参考](docs/inference.md)
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

默认命令成功构建 `AudioEncoder`；`--component draft/planner/reconstruction` 分别构建独立组件，R使用直接Z解析查询，无固定行容量。Decoder可通过`--component decoder`独立构建；阶段一训练使用`python -m maidata.training`，阶段二使用`python -m maidata.decoder_training`；整曲推理使用`python -m maidata.inference`，物化使用`python -m maidata.inference.materialize`。CPU验证命令见[Encoder入口](docs/maidata-entry.md#cpu-合成验证)及[Draft接口](docs/draft.md)；不代表CUDA、训练或谱面质量已验证。旧测试、fixture、权重、数据、缓存和运行产物没有复制；新测试按各模块范围维护。旧配置和远端launcher是来源参考，不是新实验默认命令。
